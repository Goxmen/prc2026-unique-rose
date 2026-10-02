"""16-fold CV (next-month, longer-gap, combined, Weighted CV) of the ADSB.lol inputs on v44.

v44 = v43's recipe (stand_coords_run.py: v42 inputs, stand inputs, Rome classifier, corrections
and rules) with the ordinary regressor refitted at the Jan/Jul search's chosen settings
(search/chosen.json: learning rate 0.02, 5,000 rounds). v44 itself was never scored on these
folds (its `confirm` stage was cancelled), so both arms are fitted here:

    control    v44's ordinary regressor
    adsblol    the same plus `adsblol.ADSB_FEATURES` (ordinary regressor only)

The Rome classifier is refitted here per window and seed with v43's inputs and shared by both
arms; corrections and parameters are v43's recorded ones (checked against a refit from training
labels). Folds, seeds 42/142/242 and the January retrospective follow stand_coords_run.py.

Stages:  fit     (resumable; writes VALIDATION/runs/v3_20261002_adsblol_cv/)
         score   Weighted CV and the original diagnostics from saved predictions
Run: .venv/bin/python VALIDATION/adsblol_cv.py fit [--workers 2]; then ... score
No full-year fits, submissions or ledger edits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import glob
import json
import multiprocessing
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import overnight_run
import rome_hybrid
import seasonal
import stand_coords_run as v43
import stand_features as sf
import tuning_seasonal as ts
from common import TRAIN_GLOB, RANKING
from records import RUNS, data_identity, sha256, write_json

v42, features, prior, recipe, signals, weather, opdi = (v43.v42, v43.features, v43.prior, v43.recipe, v43.signals,
                                                       v43.weather, v43.opdi)
OUT = RUNS / 'v3_20261002_adsblol_cv'
V43 = v43.DEFAULT / 'main'
ORDINARY, CLASSIFIER = list(v43.CANDIDATE), list(v43.CLASSIFIER)
ARMS = {'control': ORDINARY, 'adsblol': ORDINARY + adsblol.ADSB_FEATURES}
COLUMNS = {'control': 'pred_control', 'adsblol': 'pred_adsblol'}
THREADS = 8


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def checked_inputs(record):
    for key, files in (('data_identity', [*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]),
                       ('stand_identity', [sf.COORDS, sf.ENDS]), ('opdi_identity', [opdi.FEATURES_PATH])):
        if data_identity(files) != record[key]:
            raise ValueError(f'{key} differs from v43/v44')


def fit_regressor(matrix, job_id, arm, seed):
    """Runs in a spawned worker (OpenMP is not fork-safe once the parent has used LightGBM)."""
    data = pd.read_pickle(matrix)
    names = rome_hybrid.regressor_features(recipe.KIND, ARMS[arm])
    chosen = data['chosen']
    started = time.monotonic()
    booster = ts.train(data['X'].loc[:, names], data['y'],
                       ts.lgb_params(chosen['settings'], seed, chosen['learning_rate']), chosen['rounds'])
    path = OUT / 'models' / f'{job_id}__{arm}__{seed}.txt'
    booster.save_model(str(path))
    return f'{job_id}__{arm}__{seed}', dict(path=str(path.relative_to(OUT)), sha256=sha256(path),
                                           seconds=time.monotonic() - started)


def fit(workers):
    record = ts.read(V43 / 'result.json')
    checked_inputs(record)
    chosen = ts.read(RUNS / 'v3_20261001_jan_jul' / 'search' / 'chosen.json')
    config = dict(record['config'], threads=THREADS, prediction_threads=THREADS, fit_workers=1)
    (OUT / 'models').mkdir(parents=True, exist_ok=True)
    progress_path = OUT / 'progress.json'
    progress = ts.read(progress_path) if progress_path.exists() else dict(models={}, predictions={}, folds={},
                                                                         january_holdout={})
    write_json(OUT / 'config.json', dict(v43_run=str(V43.relative_to(ROOT)), chosen=chosen, config=config,
               arms={a: c for a, c in COLUMNS.items()}, adsblol_features=adsblol.ADSB_FEATURES,
               features_sha256=sha256(adsblol.FEATURES_PATH), source_sha256=dict(
                   runner=sha256(Path(__file__)), adsblol=sha256(ROOT / 'VALIDATION/adsblol.py'))))
    table = adsblol.load_features()
    tables, reports = sf.load(), weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr = overnight_run.load_inputs(1.)
    for job in [j for j in overnight_run.schedule() if j['scheme'] == 'v3']:
        winter = job['id'] == 'v3_2025-11'
        if all(ev['fold'] in progress['folds'] for ev in job['evals']) and (not winter or progress['january_holdout']):
            log(f'reusing {job["id"]}')
            continue
        log(f'preparing {job["id"]}')
        td, ta, validation = overnight_run.split_job(dep, arr, job)
        params = record['parameters'][job['id']]
        X, y, cats, raw, _, _ = v42.training_inputs(td, ta, config, reports, grid, params)
        X = adsblol.add_adsblol(sf.add_features(X, raw, tables), raw, table)
        corrections = record['correction_parameters'][job['id']]
        for seed in config['seeds']:
            key = f'{job["id"]}__shared__{seed}'
            if key not in progress['models']:
                started = time.monotonic()
                model = signals.fit_classifier(X, y, raw, recipe.KIND, seed, threads=THREADS, extra=CLASSIFIER)
                v43.check_classifier(model)
                path = OUT / 'models' / f'{key}.txt'
                model.save_model(str(path))
                progress['models'][key] = dict(path=str(path.relative_to(OUT)), sha256=sha256(path),
                                               seconds=time.monotonic() - started)
                write_json(progress_path, progress)
        todo = [(job['id'], a, s) for s in config['seeds'] for a in ARMS
                if f'{job["id"]}__{a}__{s}' not in progress['models']]
        if todo:
            # The ordinary regressor's rows: schedule-recorded Rome rows never reach it (signals.fit_ordinary).
            s = rome_hybrid.substituted(X, y, raw)
            cols = rome_hybrid.regressor_features(recipe.KIND, ARMS['adsblol'])
            matrix = OUT / 'models' / f'{job["id"]}__matrix.pkl'
            pd.to_pickle(dict(X=X.loc[~s, cols], y=np.asarray(y, float)[~s], chosen=chosen), matrix)
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                for key, info in pool.map(fit_regressor, [matrix] * len(todo), *zip(*todo)):
                    progress['models'][key] = info
                    write_json(progress_path, progress)
                    log(f'fitted {key} in {info["seconds"] / 60:.1f} min')
            matrix.unlink()
        del X, y, raw
        gc.collect()
        evals = list(job['evals'])
        if winter:
            validation.update(seasonal.split_frames(dep, arr, dict(train=job['train'], val=['2025-01']))[2])
            evals.append(dict(fold='JAN_RETRO', month='2025-01', role='retrospective', gap=None))
        for ev in evals:
            name = ev['fold']
            if name in progress['folds'] or (name == 'JAN_RETRO' and progress['january_holdout']):
                continue
            vr, V = recipe.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
            donors = pd.concat([ta, validation[ev['month']][1]], ignore_index=True)
            V = features.add_features(V, vr, donors, grid)
            V = adsblol.add_adsblol(sf.add_features(V, vr, tables), vr, table)
            paths = {}
            for seed in config['seeds']:
                load = lambda part: lgb.Booster(model_file=str(OUT / progress['models'][f'{job["id"]}__{part}__{seed}']['path']))
                p = signals.substitution_probability(load('shared'), V, vr, recipe.KIND, THREADS, CLASSIFIER)
                frames, ordinary = {}, {}
                for arm, extra in ARMS.items():
                    ordinary[arm] = signals.ordinary_prediction(load(arm), V, recipe.KIND, THREADS, extra)
                    frames[arm] = prior.frame_for(vr, V, job, ev, seed, ordinary[arm], p, config, params, corrections)
                out = frames['control'][v43.KEEP].copy()
                for arm, col in COLUMNS.items():
                    out[col] = frames[arm].pred_combined.to_numpy()
                    out[f'ordinary_{arm}'] = ordinary[arm]
                out['p_shared'] = p
                out['rule_triggered'] = frames['control'].rule_triggered.to_numpy()
                for col in ('ab_matched', 'ab_seg_sec', 'ab_move_sec'):
                    out[col] = V[col].to_numpy()
                if not np.isfinite(out[['y', *COLUMNS.values(), 'p_shared']].to_numpy()).all():
                    raise ValueError('Nonfinite predictions')
                path = OUT / f'oof_{name}_seed{seed}.parquet'
                overnight_run.save_frame(path, out)
                paths[str(seed)] = path.name
                progress['predictions'][f'{name}__{seed}'] = dict(path=path.name, rows=len(out), sha256=sha256(path))
            if name == 'JAN_RETRO':
                progress['january_holdout'] = dict(name=name, val=ev['month'], model_id=job['id'], predictions=paths)
            else:
                progress['folds'][name] = dict(record['folds'][name], predictions=paths)
            write_json(progress_path, progress)
            log(f'scored {name} ({ev["month"]}) on {len(vr):,} flights')
            del vr, V, donors
            gc.collect()
        del td, ta, validation
        gc.collect()
    if len(progress['folds']) != 16 or not progress['january_holdout']:
        raise ValueError('Incomplete roster')
    progress['status'] = 'complete'
    write_json(progress_path, progress)
    log('fit complete')


def weights():
    """The adopted Weighted CV weights. Ioana's transferred copies differ from the adopted bytes
    (encoding only, per VALIDATION/reports/weighted_cv_recent_submissions.json); the discriminator
    files are not on this machine, so accept them only if row counts and effective sample size
    reproduce the adopted record to 1e-12."""
    adopted = ts.read(ROOT / 'VALIDATION/reports/weighted_cv_leaderboard.json')['weights']
    out, hashes = {}, {}
    for name in ('R_A', 'R_B_long'):
        spec = adopted[name]
        ratio = pd.read_parquet(ROOT / spec['path']).ratio
        w = ratio.to_numpy(float)
        ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
        if len(w) != spec['rows'] or abs(ess - spec['ess_fraction']) > 1e-12:
            raise ValueError(f'{name} weights do not reproduce the adopted weights')
        hashes[name] = dict(local=sha256(ROOT / spec['path']), adopted=spec['sha256'], ess_fraction=float(ess))
        out[name] = ratio
    return out, hashes


def score():
    import reporting
    import score_disruption_experiment as scoring
    import score_stand_coords as v43_scoring
    progress = ts.read(OUT / 'progress.json')
    if progress.get('status') != 'complete':
        raise ValueError('Run the fit stage to completion first')
    config = ts.read(OUT / 'config.json')['config']
    record = dict(config=config, folds=progress['folds'])
    w, weight_hashes = weights()
    names = {'control': 'control', 'adsblol': 'candidate'}   # the scoring helpers call the new arm 'candidate'
    losses = {names[a]: reporting.load_losses(dict(record, prediction_column=c), OUT) for a, c in COLUMNS.items()}
    groups = {arm: reporting.score_groups(value) for arm, value in losses.items()}
    ratios = {}
    for name in ('R_A', 'R_B_long'):
        reporting.same_flights(groups['control'][name].meta, groups['candidate'][name].meta)
        ids = groups['control'][name].meta.index
        if len(ids) != len(w[name]) or len(ids.difference(w[name].index)):
            raise ValueError('Weight/scoring population differs')
        ratios[name] = w[name].loc[ids].to_numpy(float)
    scores = {arm: scoring.levels(value, ratios) for arm, value in groups.items()}
    changes = scoring.compare(groups, ratios, scores)
    winter_record = dict(record, folds={'JAN_RETRO': progress['january_holdout']})
    winter = {names[a]: reporting.load_losses(dict(winter_record, prediction_column=c), OUT)['JAN_RETRO']
              for a, c in COLUMNS.items()}
    winter_scores = {arm: reporting.rmse(loss) for arm, loss in winter.items()}
    calendar, draws = reporting.bootstrap_calendar(winter, 2000, 0, block_days=1)
    winter_change = reporting.interval(winter_scores['candidate'] - winter_scores['control'],
                                       reporting.boot_scores(winter['candidate'], calendar, draws)
                                       - reporting.boot_scores(winter['control'], calendar, draws))
    result = dict(run=str(OUT.relative_to(ROOT)), arms=names, weights_sha256=weight_hashes, scores=scores,
                  changes=changes, winter_scores=winter_scores, winter_change=winter_change,
                  airports=v43_scoring.airport_changes(losses), ageing=v43_scoring.ageing(losses),
                  source_sha256=sha256(Path(__file__)))
    write_json(OUT / 'scores.json', result)
    print(json.dumps(dict(scores={arm: {m: {n: round(v['score'], 4) for n, v in s[m].items()} for m in s}
                                  for arm, s in scores.items()},
                          weighted_combined_change=changes['1']['weighted']['R_combined'],
                          three_day=changes['3']['weighted']['R_combined']['ci95'], january=winter_change), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['fit', 'score'])
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    fit(args.workers) if args.stage == 'fit' else score()


if __name__ == '__main__':
    main()
