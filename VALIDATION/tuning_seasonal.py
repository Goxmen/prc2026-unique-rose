"""Jan/Jul validation, and a hyperparameter search for v43's ordinary regressor on it.

Validation (the user's design, 2026-10-01): two fits.
  1. hold out January 2025, train on March-November 2025
  2. hold out July 2025, train on January-March and September-December 2025
The score is the average of the two fits' RMSE. History inputs built from departure
labels freeze at each fit's own training end (30 November / 31 December), as the
ranking months freeze at 31 December. Inputs, Rome classifier, corrections and rules
are built by the same code as the submission scripts; every fit is scored on the final
prediction (rules and corrections applied).

Search selection: the same two-fit average restricted to departures that have a flight
ID, an input-only filter that leaves out most corrupted records; the all-row score is
reported beside it. The winner is then
re-checked at two more seeds and confirmed on the original 16 folds at three
seeds with Weighted CV, against v43's saved predictions.

Before the search is trusted, the validation itself is checked: the last five uploads
(v39-v43) are refitted on both splits at three seeds and their scores compared with the
official leaderboard, beside the existing validations (`submissions`, `submission_report`).

Stages (each resumable; run in order):
  prepare   build both splits, fit the classifier and the control regressor
  submissions        refit v39-v43 on both splits, seeds 42/142/242
  submission_report  levels and changes against the leaderboard, beside the ledger's CV
  validate  prepare + submissions + submission_report
  search    stage A: random settings at learning rate 0.05, scored every 200 rounds
            stage B: the stage-A winner at learning rate 0.02
            seeds:   winner and control at seeds 142 and 242
  confirm   16 folds x 3 seeds with the winner, v43 classifiers and corrections
  report    write VALIDATION/reports/tuning_seasonal/result.json

Run: .venv/bin/python VALIDATION/tuning_seasonal.py <stage>
No full-year fits, submissions or ledger edits.
"""
import argparse
import gc
import glob
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
from common import TRAIN_GLOB, RANKING
import modeling
import occupancy_run
import opdi_disruption_submit as v42_submit
import opdi_run
import opdi_submit
import overnight_models
import overnight_run
from records import RUNS, data_identity, sha256, write_json
import rome_hybrid
import seasonal
import stand_coords_run as v43
import stand_features as sf
import taxi_occupancy

cv, signals, features, weather, recipe = v43.v42, v43.signals, v43.features, v43.weather, v43.recipe
DEFAULT = RUNS / 'v3_20261001_jan_jul'
REPORT = ROOT / 'VALIDATION/reports/jan_jul_validation'
V43 = v43.DEFAULT / 'main'
KIND, THREADS, SEED = 'clock', 8, 42
# Held-out month -> training months.
FOLDS = {'2025-01': [f'2025-{m:02d}' for m in range(3, 12)],
         '2025-07': ['2025-01', '2025-02', '2025-03', '2025-09', '2025-10', '2025-11', '2025-12']}
SPLITS = list(FOLDS)


def fold_freeze(month):
    """Departure-label history ends with the fit's own training data."""
    return pd.Period(max(FOLDS[month]), 'M').end_time.strftime('%Y-%m-%d')


ORDINARY, CLASSIFIER = list(v43.CANDIDATE), list(v43.CLASSIFIER)
CONTROL_ROUNDS = 800
STEP = 200
STAGE_A = dict(trials=20, learning_rate=0.05, rounds=2000, seed=20261001)
STAGE_B = dict(learning_rate=0.02, factor=2.5, max_rounds=5000)
CHECK_SEEDS = [142, 242]
# LightGBM defaults for the categorical/regularisation keys, i.e. the current recipe.
CONTROL = dict(num_leaves=255, min_data_in_leaf=200, lambda_l2=0.0, feature_fraction=0.9,
               bagging_fraction=0.8, path_smooth=0.0, cat_smooth=10.0, cat_l2=10.0,
               min_data_per_group=100, max_cat_threshold=32)
# The last five uploads (SUBMISSIONS.md rows 42-46): regressor/classifier extra inputs,
# whether rule-overwritten rows leave the fit, and the official score of each upload.
RECIPES = {
    'v39': dict(row=42, lb=277.3629, ordinary=list(opdi_run.ORDINARIES['opdi']),
                classifier=list(opdi_run.CLASSIFIERS['opdi']), exclude_rule_rows=False),
    'v40': dict(row=43, lb=275.4146, ordinary=list(opdi_run.ORDINARIES['opdi']),
                classifier=list(opdi_run.CLASSIFIERS['opdi']), exclude_rule_rows=True),
    'v41': dict(row=44, lb=275.9592, ordinary=list(occupancy_run.ORDINARIES['occ']),
                classifier=list(occupancy_run.CLASSIFIERS['occ']), exclude_rule_rows=True),
    'v42': dict(row=45, lb=274.9076, ordinary=list(cv.EXTRA), classifier=list(cv.CLASSIFIER),
                exclude_rule_rows=True),
    'v43': dict(row=46, lb=274.7660, ordinary=ORDINARY, classifier=CLASSIFIER, exclude_rule_rows=True),
}
# Each upload changed one thing from its parent: (parent, child).
LINEAGE = [('v39', 'v40'), ('v40', 'v41'), ('v40', 'v42'), ('v42', 'v43')]
SUBMISSION_SEEDS = [42, 142, 242]
SPACE = dict(num_leaves=[127, 255, 511, 1023],
             min_data_in_leaf=[50, 100, 200, 400, 800, 1600],
             lambda_l2=[0.0, 1.0, 10.0, 100.0],
             feature_fraction=[0.5, 0.6, 0.7, 0.8, 0.9],
             bagging_fraction=[0.6, 0.7, 0.8, 0.9],
             path_smooth=[0.0, 10.0, 100.0],
             cat_smooth=[1.0, 10.0, 50.0, 100.0],
             cat_l2=[1.0, 10.0, 50.0],
             min_data_per_group=[50, 100, 200, 500],
             max_cat_threshold=[16, 32, 64])


def read(path):
    return json.loads(Path(path).read_text())


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


# ---------------------------------------------------------------- search space

def sample_settings(trials=STAGE_A['trials'], seed=STAGE_A['seed']):
    """Deterministic random settings; trial 0 is the current recipe."""
    rng = np.random.default_rng(seed)
    out = [dict(CONTROL)]
    while len(out) < trials + 1:
        pick = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        pick = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in pick.items()}
        if pick not in out:
            out.append(pick)
    return out


def lgb_params(settings, seed, learning_rate):
    """v43's regressor parameters with the searched keys overridden."""
    params = overnight_models.model_params('control', seed, THREADS)
    if settings != CONTROL or learning_rate != params['learning_rate']:
        params.update(settings, learning_rate=learning_rate)
    return params


def checkpoints(rounds, step=STEP):
    return list(range(step, rounds + 1, step))


# ---------------------------------------------------------------- scoring

def scores(y, pred, has_id, month):
    """Per-fit RMSE, all rows and flight-ID rows, and their average over the fits present."""
    se = (np.asarray(y, float) - np.asarray(pred, float)) ** 2
    out = {}
    present = [m for m in SPLITS if (month == m).any()]
    for m in present:
        k = month == m
        out[f'all_{m}'] = float(np.sqrt(se[k].mean()))
        out[f'id_{m}'] = float(np.sqrt(se[k & has_id].mean()))
    out['all'] = float(np.mean([out[f'all_{m}'] for m in present]))
    out['id'] = float(np.mean([out[f'id_{m}'] for m in present]))
    return out


def paired_bootstrap(day, month, se_a, se_b, reps=2000, seed=0):
    """Change in the average per-fit RMSE (b - a), resampling whole days within each fit."""
    day, month = np.asarray(day), np.asarray(month)
    se_a, se_b = np.asarray(se_a, float), np.asarray(se_b, float)
    rng = np.random.default_rng(seed)
    months = np.unique(month)
    change, draws = 0.0, np.zeros(reps)
    for m in months:
        k = month == m
        codes, _ = pd.factorize(day[k])
        cnt = np.bincount(codes)
        sa, sb = np.bincount(codes, se_a[k]), np.bincount(codes, se_b[k])
        draw = rng.integers(0, len(cnt), size=(reps, len(cnt)))
        n = cnt[draw].sum(1)
        draws += (np.sqrt(sb[draw].sum(1) / n) - np.sqrt(sa[draw].sum(1) / n)) / len(months)
        change += (np.sqrt(se_b[k].mean()) - np.sqrt(se_a[k].mean())) / len(months)
    change = float(change)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(change=change, ci95=[float(lo), float(hi)],
                decision='improvement' if hi < 0 else 'regression' if lo > 0 else 'inconclusive')


# ---------------------------------------------------------------- prepare

def split_dir(directory, month):
    return directory / 'splits' / month


def recipe_columns(names):
    """Every regressor and classifier input the named recipes read, in first-seen order."""
    out = []
    for name in names:
        r = RECIPES[name]
        for c in (rome_hybrid.regressor_features(KIND, r['ordinary']) + rome_hybrid.regressor_features(KIND)
                  + list(r['classifier'])):
            if c not in out:
                out.append(c)
    return out


def minimal_raw(airport, index):
    """All that the clock-kind fitters read from raw rows: the departure airport."""
    return pd.DataFrame({'ADEP_mvt': np.asarray(airport)}, index=index)


def prepare_split(dep, arr, month, config, reports, grid, tables, out):
    """One fit's split, built as the submission scripts build the full year.

    Two training matrices, because the scripts differ in when the 127-style rule rows
    leave: v39-v41 add inputs to every row and drop afterwards (opdi_submit.py,
    occupancy_submit.py); v42-v43 drop first (opdi_disruption_run.training_inputs).
    The `airline` categorical takes its categories from the rows present, so the order
    is kept per recipe rather than assumed harmless.
    """
    train = FOLDS[month]
    td, ta, validation = seasonal.split_frames(dep, arr, dict(train=train, val=[month]))
    vd, va = validation[month]
    log(f'{month}: training rows {len(td):,}, held-out rows {len(vd):,}')
    X45, y, cats, raw = overnight_run.prepare_training(td, ta, config)
    # New windows: corrections are fitted here from training labels, before rule exclusion.
    parameters = signals.fit_parameters(X45, y, raw)
    fitted, _ = v42_submit.fit_corrections(td, ta, config, parameters['late_mean'])
    drop = recipe.rule_rows(raw, config['prediction_rule'])
    out.mkdir(parents=True, exist_ok=True)
    # v39-v41 order: inputs on every row; the rule mask travels with the matrix.
    X = recipe.add_inputs(X45.copy(), raw, reports, td, ta)
    X = taxi_occupancy.add_occupancy(X, raw, modeling.observable(td))
    cols = recipe_columns(['v39', 'v40', 'v41'])
    pd.to_pickle(dict(X=X.loc[:, cols], y=np.asarray(y, float), airport=raw.ADEP_mvt.to_numpy(), rule=drop),
                 out / 'before_drop.pkl')
    before = X.loc[~drop, cols]
    del X
    gc.collect()
    # v42-v43 order: drop, then inputs (opdi_disruption_run.training_inputs + stand inputs).
    X, y_kept, raw_kept = X45.loc[~drop], np.asarray(y, float)[~drop], raw.loc[~drop]
    X = recipe.add_inputs(X, raw_kept, reports, td, ta)
    X = features.add_features(X, raw_kept, ta, grid)
    X = sf.add_features(X, raw_kept, tables)
    numeric = [c for c in before.columns if c in X.columns and pd.api.types.is_numeric_dtype(X[c].dtype)]
    differing = [c for c in numeric if not np.array_equal(before[c].to_numpy(float), X[c].to_numpy(float),
                                                          equal_nan=True)]
    del before, X45
    gc.collect()
    log(f'{month}: training matrix {X.shape}; numeric inputs differing by drop order: {differing or "none"}')
    classifier = signals.fit_classifier(X, y_kept, raw_kept, KIND, SEED, threads=THREADS, extra=CLASSIFIER)
    v43.check_classifier(classifier)
    started = time.monotonic()
    control = signals.fit_ordinary(X, y_kept, raw_kept, KIND, SEED, rounds=CONTROL_ROUNDS, threads=THREADS,
                                   extra=ORDINARY)
    control_seconds = time.monotonic() - started
    if control.feature_name() != rome_hybrid.regressor_features(KIND, ORDINARY):
        raise ValueError('Regressor feature order changed')
    s = rome_hybrid.substituted(X, y_kept, raw_kept)
    pd.to_pickle(dict(X=X.loc[:, recipe_columns(['v42', 'v43'])], y=y_kept, airport=raw_kept.ADEP_mvt.to_numpy()),
                 out / 'after_drop.pkl')
    del X, raw_kept
    gc.collect()
    log(f'{month}: building held-out inputs')
    V = modeling.validation_matrix(td, ta, vd, va, fold_freeze(month), cats, config)
    arrivals = pd.concat([ta, va], ignore_index=True)
    events = pd.concat([td, vd], ignore_index=True)
    V = recipe.add_inputs(V, vd, reports, events, arrivals)
    V = features.add_features(V, vd, arrivals, grid)
    V = sf.add_features(V, vd, tables)
    V = taxi_occupancy.add_occupancy(V, vd, modeling.observable(events))
    if not vd.index.equals(V.index):
        raise ValueError('Held-out raw rows and inputs are misaligned')
    vd, V = vd.reset_index(drop=True), V.reset_index(drop=True)
    p = signals.substitution_probability(classifier, V, vd, KIND, THREADS, CLASSIFIER)
    base = signals.ordinary_prediction(control, V, KIND, THREADS, ORDINARY)
    final = final_prediction(base, p, V, vd, parameters, fitted)
    pd.to_pickle(dict(V=V, raw=vd, p=p), out / 'heldout.pkl')
    classifier.save_model(str(out / 'classifier.txt'))
    control.save_model(str(out / 'control_regressor.txt'))
    meta = dict(month=month, train_months=train, freeze=fold_freeze(month), training_rows=len(td), excluded_rule_rows=int(drop.sum()),
                regressor_rows=int((~s).sum()), substituted_rows=int(s.sum()), heldout_rows=len(vd),
                drop_order_differences=differing, parameters=parameters, corrections=fitted,
                control_seconds=control_seconds,
                control_rmse=scores(vd.taxi_sec, final, vd.FLIGHT_ID_mvt.notna().to_numpy(),
                                    np.full(len(vd), month)),
                stand_coverage=sf.coverage(V), regressor_features=rome_hybrid.regressor_features(KIND, ORDINARY))
    pd.DataFrame(dict(mvt_id=vd.mvt_id.to_numpy(), ordinary=base, pred=final)).to_parquet(out / 'control.parquet')
    write_json(out / 'meta.json', meta)
    log(f'{month}: prepared; v43 control RMSE all {meta["control_rmse"]["all"]:.3f}, '
        f'ID {meta["control_rmse"]["id"]:.3f}, regressor fit {control_seconds:.0f}s')


def final_prediction(base, p, V, raw, parameters, fitted):
    out = signals.combine(base, p, V, raw, late_mean=parameters['late_mean'], dayshift_weight=None)
    frame, _ = opdi_submit.apply_corrections(out['final'], raw, V, fitted, out['p'])
    return frame.pred.to_numpy(float)


def prepare(directory):
    record = read(V43 / 'result.json')
    config = dict(record['config'], threads=THREADS, prediction_threads=THREADS, fit_workers=1)
    identity = data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR])
    if identity != record['data_identity']:
        raise ValueError('Movement/weather data differs from v43')
    if data_identity([sf.COORDS, sf.ENDS]) != record['stand_identity']:
        raise ValueError('Stand tables differ from v43')
    todo = [m for m in SPLITS if not (split_dir(directory, m) / 'meta.json').exists()]
    if not todo:
        log('prepare: both splits already built')
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', dict(v43_run=str(V43.relative_to(ROOT)), config=config, folds=FOLDS,
               freezes={m: fold_freeze(m) for m in SPLITS},
               data_identity=identity, stage_a=STAGE_A, stage_b=STAGE_B, control=CONTROL, space=SPACE,
               check_seeds=CHECK_SEEDS, step=STEP, source_sha256=sha256(Path(__file__))))
    tables = sf.load()
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr = overnight_run.load_inputs(1.)
    for month in todo:
        prepare_split(dep, arr, month, config, reports, grid, tables, split_dir(directory, month))
        gc.collect()


# ---------------------------------------------------------------- submissions

def submissions(directory):
    """Refit the last five uploads on both splits at three seeds with production fitters."""
    for month in SPLITS:
        d = split_dir(directory, month)
        todo = [n for n in RECIPES if not (d / f'submission_{n}.parquet').exists()]
        if not todo:
            log(f'submissions: {month} complete')
            continue
        meta, held = read(d / 'meta.json'), pd.read_pickle(d / 'heldout.pkl')
        V, vd = held['V'], held['raw']
        control = pd.read_parquet(d / 'control.parquet')
        for source, names in [('before_drop', ['v39', 'v40', 'v41']), ('after_drop', ['v42', 'v43'])]:
            names = [n for n in names if n in todo]
            if not names:
                continue
            data = pd.read_pickle(d / f'{source}.pkl')
            for name in names:
                r = RECIPES[name]
                keep = (~data['rule'] if r['exclude_rule_rows'] else np.ones(len(data['y']), bool)) \
                    if source == 'before_drop' else np.ones(len(data['y']), bool)
                X, y = data['X'].loc[keep], data['y'][keep]
                raw = minimal_raw(data['airport'][keep], X.index)
                frame = pd.DataFrame(dict(mvt_id=vd.mvt_id.to_numpy(), month=month, y=vd.taxi_sec.to_numpy(float),
                                          has_id=vd.FLIGHT_ID_mvt.notna().to_numpy(),
                                          day=pd.to_datetime(vd.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d')))
                for seed in SUBMISSION_SEEDS:
                    started = time.monotonic()
                    clf = signals.fit_classifier(X, y, raw, KIND, seed, threads=THREADS, extra=r['classifier'])
                    reg = signals.fit_ordinary(X, y, raw, KIND, seed, rounds=CONTROL_ROUNDS, threads=THREADS,
                                               extra=r['ordinary'])
                    p = signals.substitution_probability(clf, V, vd, KIND, THREADS, r['classifier'])
                    base = signals.ordinary_prediction(reg, V, KIND, THREADS, r['ordinary'])
                    frame[f'pred_{seed}'] = final_prediction(base, p, V, vd, meta['parameters'], meta['corrections'])
                    if name == 'v43' and seed == SEED and not (
                            np.array_equal(base, control.ordinary.to_numpy())
                            and np.array_equal(frame[f'pred_{seed}'].to_numpy(), control.pred.to_numpy())):
                        raise ValueError(f'{month}: stored matrices do not reproduce the prepared v43 control')
                    rmse = float(np.sqrt(np.mean((frame.y - frame[f'pred_{seed}']) ** 2)))
                    log(f'submissions: {month} {name} seed {seed}: RMSE {rmse:.3f} '
                        f'({len(y):,} training rows, {time.monotonic() - started:.0f}s)')
                    del clf, reg
                overnight_run.save_frame(d / f'submission_{name}.parquet', frame)
                del X, y, raw
                gc.collect()
            del data
            gc.collect()


def ledger_rows():
    """Rows 42-46 of SUBMISSIONS.md: the other validations and the official score."""
    keys = dict(next_month='CV next-month', longer_gap='CV longer-gap', combined='CV combined',
                weighted='Weighted CV', lb='Official LB')
    out, header = {}, None
    for line in (ROOT / 'SUBMISSIONS.md').read_text().splitlines():
        cells = [c.strip().strip('*') for c in line.strip().strip('|').split('|')]
        if header is None and cells[0] == 'Order':
            header = cells
        for name, r in RECIPES.items():
            if header and cells[0] == str(r['row']) and cells[-1] == name:
                values = {k: float(cells[header.index(column)]) for k, column in keys.items()}
                if values['lb'] != r['lb']:
                    raise ValueError(f'{name}: ledger score differs from RECIPES')
                out[name] = values
    if set(out) != set(RECIPES):
        raise ValueError('Could not read all five ledger rows')
    return out


def submission_report(directory):
    frames = {n: pd.concat([pd.read_parquet(split_dir(directory, m) / f'submission_{n}.parquet') for m in SPLITS],
                           ignore_index=True) for n in RECIPES}
    ref = frames['v43']
    for f in frames.values():
        if not f.mvt_id.equals(ref.mvt_id):
            raise ValueError('Recipes scored different flights')
    y, has_id, month, day = ref.y.to_numpy(), ref.has_id.to_numpy(), ref.month.to_numpy(), ref.day.to_numpy()
    seeds = [f'pred_{s}' for s in SUBMISSION_SEEDS]
    se = {n: np.stack([(y - f[c].to_numpy()) ** 2 for c in seeds]) for n, f in frames.items()}
    levels = {}
    for n, e in se.items():
        def fit(mask):
            # One fit's RMSE: seeds average in squared error, as the 16-fold CV does.
            return dict(three_seed=float(np.sqrt(e[:, mask].mean())), seed42=float(np.sqrt(e[0, mask].mean())),
                        by_seed=[float(np.sqrt(v)) for v in e[:, mask].mean(1)])

        def average(parts):
            return dict(three_seed=float(np.mean([p['three_seed'] for p in parts])),
                        seed42=float(np.mean([p['seed42'] for p in parts])),
                        by_seed=np.mean([p['by_seed'] for p in parts], axis=0).tolist())
        fits = {m: fit(month == m) for m in SPLITS}
        id_fits = {f'id_{m}': fit(has_id & (month == m)) for m in SPLITS}
        levels[n] = dict(all=average(list(fits.values())), id=average(list(id_fits.values())), **fits, **id_fits)
    ledger = ledger_rows()
    views = {'jan_jul': {n: levels[n]['all']['three_seed'] for n in RECIPES},
             'jan_jul_seed42': {n: levels[n]['all']['seed42'] for n in RECIPES},
             'jan_jul_id': {n: levels[n]['id']['three_seed'] for n in RECIPES},
             **{k: {n: ledger[n][k] for n in RECIPES} for k in ('next_month', 'longer_gap', 'combined', 'weighted')}}
    lb = {n: ledger[n]['lb'] for n in RECIPES}
    tracking = {}
    for view, values in views.items():
        gaps = [lb[n] - values[n] for n in RECIPES]
        changes = []
        for a, b in LINEAGE:
            d_lb, d_v = lb[b] - lb[a], values[b] - values[a]
            changes.append(dict(pair=f'{a}->{b}', lb=d_lb, validation=d_v, error=d_v - d_lb,
                                same_direction=bool(np.sign(d_lb) == np.sign(d_v))))
        tracking[view] = dict(mean_gap=float(np.mean(gaps)), mean_abs_gap=float(np.mean(np.abs(gaps))),
                              gap_spread=float(np.std(gaps)), mean_abs_change_error=float(np.mean([abs(c['error']) for c in changes])),
                              directions=f"{sum(c['same_direction'] for c in changes)}/{len(changes)}", changes=changes)
    intervals = {}
    for a, b in LINEAGE:
        mean_a, mean_b = se[a].mean(0), se[b].mean(0)
        intervals[f'{a}->{b}'] = dict(all=paired_bootstrap(day, month, mean_a, mean_b),
                                      id=paired_bootstrap(day[has_id], month[has_id], mean_a[has_id], mean_b[has_id]))
    result = dict(levels=levels, tracking=tracking, intervals=intervals, lb=lb, ledger=ledger,
                  rows=dict(total=len(y), **{m: int((month == m).sum()) for m in SPLITS}),
                  prepared={m: {k: v for k, v in read(split_dir(directory, m) / 'meta.json').items()
                                if k not in ('corrections', 'regressor_features')} for m in SPLITS})
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / 'submissions.json', result)
    for view, t in tracking.items():
        log(f'{view:20s} gap {t["mean_gap"]:+8.3f} (|gap| {t["mean_abs_gap"]:7.3f}, spread {t["gap_spread"]:.3f}) '
            f'change error {t["mean_abs_change_error"]:.3f} directions {t["directions"]}')


# ---------------------------------------------------------------- search

class Splits:
    """Both prepared splits held in memory for repeated fits."""

    def __init__(self, directory):
        self.items = {}
        for month in SPLITS:
            d = split_dir(directory, month)
            meta = read(d / 'meta.json')
            train, held = pd.read_pickle(d / 'after_drop.pkl'), pd.read_pickle(d / 'heldout.pkl')
            raw = held['raw']
            names = meta['regressor_features']
            # signals.fit_ordinary's rows: schedule-recorded Rome rows never reach the regressor.
            s = rome_hybrid.substituted(train['X'], train['y'], minimal_raw(train['airport'], train['X'].index))
            X, y = train['X'].loc[~s, names].copy(), train['y'][~s]
            del train
            gc.collect()
            self.items[month] = dict(meta=meta, X=X, y=y, V=held['V'], raw=raw,
                                     Vn=held['V'].loc[:, names], p=held['p'],
                                     control=pd.read_parquet(d / 'control.parquet'))
        parts = [self.items[m]['raw'] for m in SPLITS]
        self.y = np.concatenate([r.taxi_sec.to_numpy(float) for r in parts])
        self.has_id = np.concatenate([r.FLIGHT_ID_mvt.notna().to_numpy() for r in parts])
        self.month = np.concatenate([np.full(len(r), m) for m, r in zip(SPLITS, parts)])
        self.day = np.concatenate([pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
                                   for r in parts])
        self.mvt_id = np.concatenate([r.mvt_id.to_numpy() for r in parts])


def train(X, y, params, rounds):
    """signals._train without the progress bar: same dataset construction, same result."""
    dataset = lgb.Dataset(X, y, categorical_feature=signals._categorical(X.columns))
    return lgb.train(params, dataset, num_boost_round=rounds)


def run_trial(splits, settings, seed, learning_rate, rounds, verify_control=False):
    """Fit both splits; final predictions at every checkpoint."""
    params = lgb_params(settings, seed, learning_rate)
    marks = checkpoints(rounds)
    preds = {k: [] for k in marks}
    seconds = {}
    for month in SPLITS:
        item = splits.items[month]
        started = time.monotonic()
        booster = train(item['X'], item['y'], params, rounds)
        seconds[month] = time.monotonic() - started
        meta = item['meta']
        for k in marks:
            base = modeling.predict(booster, item['Vn'], num_iteration=k, threads=THREADS)
            preds[k].append(final_prediction(base, item['p'], item['V'], item['raw'],
                                             meta['parameters'], meta['corrections']))
        if verify_control:
            base = modeling.predict(booster, item['Vn'], num_iteration=CONTROL_ROUNDS, threads=THREADS)
            if not np.array_equal(base, item['control'].ordinary.to_numpy()):
                worst = np.max(np.abs(base - item['control'].ordinary.to_numpy()))
                raise ValueError(f'{month}: search fit does not reproduce the production regressor ({worst})')
            if not np.array_equal(preds[CONTROL_ROUNDS][-1], item['control'].pred.to_numpy()):
                raise ValueError(f'{month}: reloaded corrections do not reproduce the control prediction')
        del booster
        gc.collect()
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    return preds, seconds


def trial_record(name, stage, settings, seed, learning_rate, rounds, preds, splits, seconds):
    by_round = {str(k): scores(splits.y, p, splits.has_id, splits.month) for k, p in preds.items()}
    best = min(by_round, key=lambda k: by_round[k]['id'])
    return dict(name=name, stage=stage, settings=settings, seed=seed, learning_rate=learning_rate,
                rounds=rounds, fit_seconds=seconds, by_round=by_round, best_round=int(best),
                best=by_round[best])


def save_preds(directory, name, preds, splits):
    frame = pd.DataFrame({'mvt_id': splits.mvt_id})
    for k, p in preds.items():
        frame[f'r{k}'] = p.astype(np.float32)
    overnight_run.save_frame(directory / 'search' / f'{name}.parquet', frame)


def load_trials(directory):
    path = directory / 'search' / 'trials.jsonl'
    if not path.exists():
        return {}
    return {r['name']: r for r in map(json.loads, path.read_text().splitlines())}


def append_trial(directory, record):
    path = directory / 'search' / 'trials.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(record) + '\n')


def do_trial(directory, splits, done, name, stage, settings, seed, learning_rate, rounds, verify=False):
    if name in done:
        return done[name]
    log(f'{name}: lr {learning_rate}, {rounds} rounds, seed {seed}, {settings}')
    preds, seconds = run_trial(splits, settings, seed, learning_rate, rounds, verify_control=verify)
    record = trial_record(name, stage, settings, seed, learning_rate, rounds, preds, splits, seconds)
    save_preds(directory, name, preds, splits)
    append_trial(directory, record)
    done[name] = record
    b = record['best']
    log(f'{name}: best round {record["best_round"]}: ID {b["id"]:.3f}, all {b["all"]:.3f} '
        f'(fits {sum(seconds.values()):.0f}s)')
    return record


def stage_a_winner(done):
    stage_a = [r for r in done.values() if r['stage'] == 'A']
    return min(stage_a, key=lambda r: r['best']['id'])


def search(directory):
    splits = Splits(directory)
    done = load_trials(directory)
    settings = sample_settings()
    for i, s in enumerate(settings):
        do_trial(directory, splits, done, f'A{i:02d}', 'A', s, SEED, STAGE_A['learning_rate'],
                 STAGE_A['rounds'], verify=(i == 0))
    winner = stage_a_winner(done)
    rounds_b = min(STAGE_B['max_rounds'], int(np.ceil(winner['best_round'] * STAGE_B['factor'] / STEP) * STEP))
    do_trial(directory, splits, done, 'B00', 'B', winner['settings'], SEED, STAGE_B['learning_rate'], rounds_b)
    chosen = choose(done)
    for seed in CHECK_SEEDS:
        for label, s, lr, r in [('control', CONTROL, STAGE_A['learning_rate'], CONTROL_ROUNDS),
                                ('chosen', chosen['settings'], chosen['learning_rate'], chosen['rounds'])]:
            do_trial(directory, splits, done, f'S_{label}_{seed}', 'seed', s, seed, lr, r)
    write_json(directory / 'search' / 'chosen.json', chosen)
    log(f'search complete; chosen {chosen}')


def choose(done):
    """Best (trial, round) by flight-ID RMSE across stages A and B, at seed 42."""
    best = min((r for r in done.values() if r['stage'] in ('A', 'B')), key=lambda r: r['best']['id'])
    return dict(trial=best['name'], settings=best['settings'], learning_rate=best['learning_rate'],
                rounds=best['best_round'], score=best['best'])


# ---------------------------------------------------------------- confirm

def confirm(directory):
    chosen = read(directory / 'search' / 'chosen.json')
    record, v43_progress = read(V43 / 'result.json'), read(V43 / 'progress.json')
    config = dict(record['config'], threads=THREADS, prediction_threads=THREADS, fit_workers=1)
    out = directory / 'confirm'
    (out / 'models').mkdir(parents=True, exist_ok=True)
    progress_path = out / 'progress.json'
    progress = read(progress_path) if progress_path.exists() else dict(models={}, folds={}, predictions={})
    if progress.get('chosen', chosen) != chosen:
        raise ValueError('Chosen setting changed since confirmation started')
    progress['chosen'] = chosen
    tables = sf.load()
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr = overnight_run.load_inputs(1.)
    jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
    for job in jobs:
        if all(ev['fold'] in progress['folds'] for ev in job['evals']):
            log(f'confirm: reusing {job["id"]}')
            continue
        log(f'confirm: preparing {job["id"]}')
        td, ta, validation = overnight_run.split_job(dep, arr, job)
        params = v43_progress['parameters'][job['id']]
        X, y, cats, raw, fitted_params, _ = cv.training_inputs(td, ta, config, reports, grid, params)
        X = sf.add_features(X, raw, tables)
        for seed in config['seeds']:
            key = f'{job["id"]}__tuned__{seed}'
            path = out / 'models' / f'{key}.txt'
            if key in progress['models'] and path.exists():
                continue
            started = time.monotonic()
            s = rome_hybrid.substituted(X, y, raw)
            names = rome_hybrid.regressor_features(KIND, ORDINARY)
            booster = train(X.loc[~s, names], np.asarray(y, float)[~s],
                            lgb_params(chosen['settings'], seed, chosen['learning_rate']), chosen['rounds'])
            booster.save_model(str(path))
            progress['models'][key] = dict(path=str(path.relative_to(out)), seconds=time.monotonic() - started,
                                           sha256=sha256(path))
            write_json(progress_path, progress)
            log(f'confirm: fitted {key} in {progress["models"][key]["seconds"]:.0f}s')
        del X, y, raw
        gc.collect()
        for ev in job['evals']:
            name = ev['fold']
            if name in progress['folds']:
                continue
            vr, V = recipe.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
            donors = pd.concat([ta, validation[ev['month']][1]], ignore_index=True)
            V = features.add_features(V, vr, donors, grid)
            V = sf.add_features(V, vr, tables)
            paths = {}
            for seed in config['seeds']:
                classifier = v43.load_model(V43, v43_progress, f'{job["id"]}__shared__{seed}')
                p = signals.substitution_probability(classifier, V, vr, recipe.KIND, THREADS, CLASSIFIER)
                corrections = record['correction_parameters'][job['id']]
                frames = {}
                for arm, booster in [('control', v43.load_model(V43, v43_progress, f'{job["id"]}__candidate__{seed}')),
                                     ('tuned', lgb.Booster(model_file=str(out / 'models' / f'{job["id"]}__tuned__{seed}.txt')))]:
                    ordinary = signals.ordinary_prediction(booster, V, recipe.KIND, THREADS, ORDINARY)
                    frames[arm] = (ordinary, cv.prior.frame_for(vr, V, job, ev, seed, ordinary, p, config,
                                                                params, corrections))
                saved = pd.read_parquet(V43 / v43_progress['folds'][name]['predictions'][str(seed)],
                                        columns=['mvt_id', 'pred_stand'])
                control = frames['control'][1]
                if not (np.array_equal(saved.mvt_id.to_numpy(), control.mvt_id.to_numpy())
                        and np.array_equal(saved.pred_stand.to_numpy(), control.pred_combined.to_numpy())):
                    raise ValueError(f'{name}/{seed}: rebuilt v43 does not reproduce its saved predictions')
                frame = control[v43.KEEP].copy()
                frame['pred_control'] = control.pred_combined.to_numpy()
                frame['pred_tuned'] = frames['tuned'][1].pred_combined.to_numpy()
                frame['ordinary_control'], frame['ordinary_tuned'] = frames['control'][0], frames['tuned'][0]
                frame['p'] = p
                frame['rule_triggered'] = control.rule_triggered.to_numpy()
                file = f'oof_{name}_seed{seed}.parquet'
                overnight_run.save_frame(out / file, frame)
                paths[str(seed)] = file
            progress['folds'][name] = dict(v43_progress['folds'][name], predictions=paths)
            write_json(progress_path, progress)
            log(f'confirm: scored {name} ({ev["month"]}) on {len(vr):,} flights; v43 reproduced')
            del vr, V, donors
            gc.collect()
        del td, ta, validation
        gc.collect()
    if len(progress['folds']) != 16:
        raise ValueError('Incomplete confirmation roster')
    progress['status'] = 'complete'
    write_json(progress_path, progress)
    log('confirm complete')


# ---------------------------------------------------------------- report

def report(directory):
    import reporting
    import score_disruption_experiment as scoring
    import score_stand_coords as v43_scoring
    done = load_trials(directory)
    chosen = read(directory / 'search' / 'chosen.json')
    splits = Splits(directory)
    trials = []
    for r in sorted(done.values(), key=lambda r: r['name']):
        trials.append(dict(name=r['name'], stage=r['stage'], seed=r['seed'], learning_rate=r['learning_rate'],
                           settings=r['settings'], best_round=r['best_round'], best=r['best'],
                           at_800=r['by_round'].get(str(CONTROL_ROUNDS)), fit_seconds=r['fit_seconds']))

    def pred(name, rounds):
        return pd.read_parquet(directory / 'search' / f'{name}.parquet', columns=[f'r{rounds}'])[f'r{rounds}'].to_numpy(float)

    def change(a, b, mask):
        se_a, se_b = (splits.y - a) ** 2, (splits.y - b) ** 2
        return paired_bootstrap(splits.day[mask], splits.month[mask], se_a[mask], se_b[mask])

    every = np.ones(len(splits.y), bool)
    seasonal_changes = {}
    for seed, control_name, chosen_name in [(SEED, 'A00', chosen['trial'])] + \
            [(s, f'S_control_{s}', f'S_chosen_{s}') for s in CHECK_SEEDS]:
        a, b = pred(control_name, CONTROL_ROUNDS), pred(chosen_name, chosen['rounds'])
        seasonal_changes[str(seed)] = dict(id=change(a, b, splits.has_id), all=change(a, b, every),
                                           **{m: change(a, b, every & (splits.month == m)) for m in SPLITS})
    result = dict(chosen=chosen, trials=trials, seasonal_changes=seasonal_changes,
                  prepared={m: read(split_dir(directory, m) / 'meta.json') for m in SPLITS})
    confirm_progress = directory / 'confirm' / 'progress.json'
    if confirm_progress.exists() and read(confirm_progress).get('status') == 'complete':
        progress = read(confirm_progress)
        record = read(V43 / 'result.json')
        record = dict(record, folds=progress['folds'])
        w, weight_hashes = v43_scoring.weights()
        losses = {arm: reporting.load_losses(dict(record, prediction_column=col), directory / 'confirm')
                  for arm, col in [('control', 'pred_control'), ('candidate', 'pred_tuned')]}
        groups = {arm: reporting.score_groups(value) for arm, value in losses.items()}
        ratios = {}
        for name in ('R_A', 'R_B_long'):
            reporting.same_flights(groups['control'][name].meta, groups['candidate'][name].meta)
            ratios[name] = w[name].loc[groups['control'][name].meta.index].to_numpy(float)
        cv_scores = {arm: scoring.levels(value, ratios) for arm, value in groups.items()}
        result['confirm'] = dict(scores=cv_scores, changes=scoring.compare(groups, ratios, cv_scores),
                                 airports=v43_scoring.airport_changes(losses), weights_sha256=weight_hashes,
                                 models=progress['models'])
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / 'result.json', result)
    log(f'report written to {REPORT / "result.json"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'submissions', 'submission_report', 'validate',
                                          'search', 'confirm', 'report'])
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    stages = (['prepare', 'submissions', 'submission_report'] if args.stage == 'validate' else [args.stage])
    for stage in stages:
        started = time.monotonic()
        globals()[stage](directory)
        log(f'{stage} finished in {(time.monotonic() - started) / 60:.1f} min')


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
