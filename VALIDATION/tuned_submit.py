"""Refit and submit v43 with the Jan/Jul-tuned ordinary regressor (v44 by default).

prepare fits seed 42 on all of 2025 and writes the ranking file; upload performs one
conditional PUT; result reads the official score. This is v43's submission
(`stand_coords_submit.py`) step for step. The only recipe change is the ordinary
regressor's LightGBM settings, learning rate and rounds: the setting the Jan/Jul search
chose (`tuning_seasonal.py search`, trial B00, checked at seeds 42/142/242).

Guards: before the tuned fit, v43's ordinary regressor and Rome classifier are refitted
from the rebuilt full-year inputs and must equal v43's saved model files byte for byte,
and the rebuilt ranking inputs must equal v43's saved ranking inputs.
"""
import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
from common import TEMPLATE
import modeling
import overnight_models
import overnight_run
from records import runtime_identity, sha256, source_snapshot, write_json
import rome_hybrid
from rome_hybrid_run import save_booster
from rome_hybrid_submit import FREEZE, KIND, MONTHS, SEED, file_name, now, submission
import stand_coords_submit as v43s
import stand_day_submit as transfer
from submit import fill_template
import tuning_seasonal as ts

cv, features, signals, weather, run, sf = v43s.cv, v43s.features, v43s.signals, v43s.weather, v43s.run, v43s.sf
PARENT = run.DEFAULT / 'submission_v43'
SEARCH = ts.DEFAULT / 'search'
NAME = 'v43-jan-jul-tuned'
MODEL_KEYS = v43s.MODEL_KEYS
CHECK_KEY = 'refit2025__ordinary_v43__42'
# The search's chosen setting (search/chosen.json), fixed here so a changed file is caught.
CHOSEN = dict(trial='B00', learning_rate=0.02, rounds=5000,
              settings=dict(num_leaves=127, min_data_in_leaf=200, lambda_l2=100.0, feature_fraction=0.7,
                            bagging_fraction=0.8, path_smooth=0.0, cat_smooth=100.0, cat_l2=10.0,
                            min_data_per_group=100, max_cat_threshold=16))
SEED_CHECKS = {42: ('A00', 'B00'), 142: ('S_control_142', 'S_chosen_142'), 242: ('S_control_242', 'S_chosen_242')}


def search_evidence():
    """The chosen setting and its Jan/Jul change against v43 (800 rounds) at every seed."""
    chosen = cv.read(SEARCH / 'chosen.json')
    if {k: chosen[k] for k in ('trial', 'learning_rate', 'rounds', 'settings')} != CHOSEN:
        raise ValueError('search/chosen.json is not the setting this script submits')
    trials = ts.load_trials(ts.DEFAULT)
    if len([r for r in trials.values() if r['stage'] == 'A']) != ts.STAGE_A['trials'] + 1:
        raise ValueError('Search stage A is incomplete')
    seeds = {}
    for seed, (control, tuned) in SEED_CHECKS.items():
        a, b = trials[control], trials[tuned]
        if a['seed'] != seed or b['seed'] != seed or b['settings'] != CHOSEN['settings'] \
                or b['learning_rate'] != CHOSEN['learning_rate'] or a['settings'] != ts.CONTROL:
            raise ValueError(f'Seed {seed}: search records do not match')
        a, b = a['by_round'][str(ts.CONTROL_ROUNDS)], b['by_round'][str(CHOSEN['rounds'])]
        seeds[str(seed)] = dict(control=a, tuned=b, change={k: b[k] - a[k] for k in b})
        if seeds[str(seed)]['change']['all'] >= 0 or seeds[str(seed)]['change']['id'] >= 0:
            raise ValueError(f'Seed {seed}: the tuned setting does not beat v43 on Jan/Jul CV')
    return chosen, seeds


def tuned_params():
    """v43's regressor parameters with the chosen keys and learning rate, as in the search."""
    params = ts.lgb_params(CHOSEN['settings'], SEED, CHOSEN['learning_rate'])
    base = overnight_models.model_params('control', SEED, ts.THREADS)
    if {k: v for k, v in params.items() if k not in CHOSEN['settings'] and k != 'learning_rate'} != \
            {k: v for k, v in base.items() if k not in CHOSEN['settings'] and k != 'learning_rate'}:
        raise ValueError('Tuned parameters change more than the searched keys')
    return params


def fit_ordinary(directory, progress, key, X, y, raw, params, rounds):
    """signals.fit_ordinary with explicit parameters and rounds."""
    if key in progress['models']:
        return
    started = time.monotonic()
    s = rome_hybrid.substituted(X, y, raw)
    names = rome_hybrid.regressor_features(cv.recipe.KIND, run.CANDIDATE)
    model = signals._train(params, X.loc[~s, names], np.asarray(y, float)[~s], rounds,
                           f'ordinary+{len(run.CANDIDATE)}/{SEED}')
    if model.feature_name() != list(names):
        raise ValueError(f'Model feature order changed: {key}')
    path = directory / 'models' / f'{key}.txt'
    save_booster(model, path)
    progress['models'][key] = dict(path=str(path.relative_to(directory)), part='ordinary', seed=SEED,
        features=model.feature_name(), params=params, rounds=rounds,
        artifacts=overnight_run.artifact_hashes(directory, path), seconds=time.monotonic() - started)
    write_json(directory / 'progress.json', progress)
    print(f'Fitted {key} in {time.monotonic() - started:.1f}s', flush=True)


def same_as_parent(directory, progress, key, parent_progress, parent_key):
    mine = sha256(directory / progress['models'][key]['path'])
    theirs = sha256(PARENT / parent_progress['models'][parent_key]['path'])
    if mine != theirs:
        raise ValueError(f'{key} does not reproduce v43 ({parent_key})')


def same_inputs(frame, month):
    """Rebuilt ranking inputs and rule rows equal v43's saved ranking frame."""
    parent = pd.read_parquet(PARENT / f'ranking_{month}.parquet')
    columns = ['mvt_id', 'rule_triggered', 'p', *features.FEATURES, *sf.FEATURES]
    for column in columns:
        a, b = frame[column].to_numpy(), parent[column].to_numpy()
        same = np.array_equal(a, b) if column in ('mvt_id', 'rule_triggered') else \
            np.allclose(a.astype(float), b.astype(float), rtol=0, atol=0, equal_nan=True)
        if not same:
            raise ValueError(f'{month}: {column} differs from v43')
    return parent


def prepare(version, resume):
    started = time.monotonic()
    record, scores = v43s.validated()
    chosen, seeds = search_evidence()
    parent_manifest = cv.read(PARENT / 'submission_v43.json')
    parent_progress = cv.read(PARENT / 'progress.json')
    if parent_manifest['state'] != 'scored' or parent_manifest['version'] != 43:
        raise ValueError('v43 is not the scored parent')
    config = record['config'].copy()
    config.update(threads=run.THREADS, prediction_threads=run.THREADS, fit_workers=1)
    params = tuned_params()
    directory = ts.DEFAULT / f'submission_v{version}'
    manifest_path = directory / f'submission_v{version}.json'
    if manifest_path.exists():
        raise ValueError('Submission is already prepared; use upload/result')
    if submission(version).exists():
        raise ValueError('Submission file already exists; inspect it before continuing')
    sources = {**record['signature'], **{name: sha256(ROOT / name) for name in (
        'VALIDATION/tuned_submit.py', 'VALIDATION/tuning_seasonal.py', 'VALIDATION/stand_coords_submit.py',
        'VALIDATION/opdi_disruption_submit.py', 'VALIDATION/opdi_submit.py', 'VALIDATION/stand_day_submit.py',
        'VALIDATION/rome_hybrid_submit.py', 'VALIDATION/overnight_submit.py', 'VALIDATION/submit.py')}}
    identity = dict(version=version, signature=sources, config=config, params=params, rounds=CHOSEN['rounds'],
        data_identity=record['data_identity'], opdi_identity=record['opdi_identity'],
        stand_identity=record['stand_identity'], template_sha256=sha256(TEMPLATE),
        anchors={str(p.relative_to(ROOT)): sha256(p) for p in (
            run.DEFAULT / 'main' / 'result.json', v43s.SCORES, v43s.PARENT_MANIFEST,
            PARENT / 'submission_v43.json', SEARCH / 'chosen.json', SEARCH / 'trials.jsonl')})
    if resume:
        saved = cv.read(directory / 'refit.json')
        if saved['identity'] != identity:
            raise ValueError('Cannot resume changed code, recipe, data or search evidence')
        progress = cv.read(directory / 'progress.json')
        overnight_run.check_artifacts(directory, progress)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'models').mkdir()
        write_json(directory / 'refit.json', dict(identity=identity, runtime=runtime_identity(),
            started_at=now(), source=source_snapshot(directory)))
        progress = dict(models={}, predictions={})
        write_json(directory / 'progress.json', progress)
    previous = cv.read(v43s.PARENT_MANIFEST)
    expected = previous['model']['parameters']
    tables = sf.load()
    print(f'v{version}: verified v43 CV, search evidence, source and data hashes; loading local inputs', flush=True)
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS or len(rank) != transfer.EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Ranking population changed')
    keys = {*MODEL_KEYS.values(), CHECK_KEY}
    if set(progress['models']) != keys:
        print('Preparing full-year features; fitting the v43 checks and the tuned regressor', flush=True)
        fitted, correction_audit = v43s.v42.fit_corrections(dep, arr, config, expected['late_mean'])
        v43s.v42.check_parameters(fitted, previous['model']['corrections'])
        X, y, cats, raw, parameters, membership = cv.training_inputs(dep, arr, config, reports, grid, expected)
        if len(membership) != 2084659 or membership.excluded_by_rule.sum() != 127 or len(raw) != 2084532:
            raise ValueError('Full-year training population differs from v40/v42/v43')
        X = sf.add_features(X, raw, tables)
        overnight_run.save_frame(directory / 'training_membership.parquet', membership)
        overnight_run.save_frame(directory / 'correction_audit.parquet', correction_audit)
        write_json(directory / 'categories.json', overnight_run.category_values(cats))
        if sha256(directory / 'categories.json') != sha256(PARENT / 'categories.json'):
            raise ValueError('Category values differ from v43')
        progress['training'] = dict(parameters=parameters, corrections=fitted,
            rows_before_exclusion=len(membership), excluded=127, rows=len(raw),
            ordinary_rows=int(membership.used_by_ordinary.sum()),
            classifier_rows=int(membership.used_by_classifier.sum()), stand_coverage=sf.coverage(X),
            artifacts=overnight_run.artifact_hashes(directory, directory / 'training_membership.parquet',
                directory / 'correction_audit.parquet', directory / 'categories.json'))
        write_json(directory / 'progress.json', progress)
        cv.fitted_model(directory, progress, MODEL_KEYS['classifier'], X, y, raw, 'classifier',
                        cv.CLASSIFIER, SEED, config)
        same_as_parent(directory, progress, MODEL_KEYS['classifier'], parent_progress, MODEL_KEYS['classifier'])
        fit_ordinary(directory, progress, CHECK_KEY, X, y, raw,
                     overnight_models.model_params('control', SEED, ts.THREADS), ts.CONTROL_ROUNDS)
        same_as_parent(directory, progress, CHECK_KEY, parent_progress, MODEL_KEYS['ordinary'])
        print('v43 classifier and ordinary regressor reproduced byte for byte', flush=True)
        fit_ordinary(directory, progress, MODEL_KEYS['ordinary'], X, y, raw, params, CHOSEN['rounds'])
        del X, y, raw, membership, correction_audit
        gc.collect()
    cats = overnight_run.read_categories(directory / 'categories.json')
    parameters, fitted = (progress['training'][name] for name in ('parameters', 'corrections'))
    boosters = {part: run.load_model(directory, progress, key) for part, key in MODEL_KEYS.items()}
    run.check_classifier(boosters['classifier'])
    pieces, all_masks, against = [], {}, {}
    for month in MONTHS:
        path = directory / f'ranking_{month}.parquet'
        if month in progress['predictions']:
            cv.prior.checked_file(directory, progress['predictions'][month])
            frame = pd.read_parquet(path)
            masks = progress['predictions'][month]['masks']
        else:
            print(f'v{version}: predicting ranking month {month}', flush=True)
            raw, ra = modeling.month_rows(rank, month), modeling.month_rows(rank_arr, month)
            V = modeling.validation_matrix(dep, arr, raw, ra, FREEZE, cats, config)
            arrivals = pd.concat([arr, ra], ignore_index=True)
            V = cv.recipe.add_inputs(V, raw, reports, pd.concat([dep, raw], ignore_index=True), arrivals)
            V = features.add_features(V, raw, arrivals, grid)
            V = sf.add_features(V, raw, tables)
            frame, masks = v43s.predict(boosters['ordinary'], boosters['classifier'], V, raw, parameters, fitted)
            frame['month'] = month
            overnight_run.save_frame(path, frame)
            progress['predictions'][month] = dict(path=path.name, rows=len(frame), masks=masks,
                stand_coverage=sf.coverage(V), artifacts=overnight_run.artifact_hashes(directory, path))
            write_json(directory / 'progress.json', progress)
            del V, arrivals, raw, ra
            gc.collect()
        parent = same_inputs(frame, month)
        diff = frame.pred.to_numpy() - parent.pred.to_numpy()
        against[month] = dict(rows=len(frame), mean_change=float(diff.mean()),
                              mean_abs_change=float(np.abs(diff).mean()), rmse_change=float(np.sqrt((diff ** 2).mean())),
                              corr=float(np.corrcoef(frame.pred, parent.pred)[0, 1]))
        pieces.append(frame)
        all_masks[month] = masks
    frame = pd.concat(pieces, ignore_index=True)
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('Invalid prediction movement IDs')
    if len(frame) != len(rank) or set(frame.mvt_id) != set(rank.mvt_id):
        raise ValueError('Incomplete ranking predictions')
    frame = frame.set_index('mvt_id').reindex(rank.mvt_id).reset_index()
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    transfer.validate_submission(table, template)
    path = submission(version)
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, temp)
    temp.replace(path)
    manifest = dict(version=version, file=file_name(version), recipe=NAME, parent='v43', state='prepared',
        prepared_at=now(), sha256=sha256(path), rows=table.num_rows, seed=SEED, rounds=CHOSEN['rounds'],
        learning_rate=CHOSEN['learning_rate'], tuned_settings=CHOSEN['settings'], params=params,
        classifier_rounds=300, freeze=FREEZE, ranking_months=MONTHS,
        search=dict(directory=str(SEARCH.relative_to(ROOT)), chosen=chosen, jan_jul_by_seed=seeds),
        v43_validation_run=str((run.DEFAULT / 'main').relative_to(ROOT)),
        ordinary_extra=run.CANDIDATE, classifier_extra=cv.CLASSIFIER, disruption=features.SETTINGS,
        stand=sf.SETTINGS, late=True, dayshift=False, exclude_rule_rows=True,
        model=progress['models'], training=progress['training'], predictions=progress['predictions'],
        correction_masks=all_masks, rule_triggered_rows=int(frame.rule_triggered.sum()),
        against_v43=against, prediction_summary=frame.pred.describe().to_dict(), identity=identity,
        external_data=parent_manifest['external_data'], minutes=(time.monotonic() - started) / 60)
    write_json(manifest_path, manifest)
    print(f'Prepared {file_name(version)}: {len(frame):,} rows, SHA256 {manifest["sha256"]}', flush=True)
    for month, change in against.items():
        print(f'  {month} against v43: mean {change["mean_change"]:+.2f} s, mean |change| '
              f'{change["mean_abs_change"]:.2f} s, corr {change["corr"]:.4f}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    parser.add_argument('--version', type=int, default=44)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        with threadpool_limits(limits=1, user_api='blas'):
            prepare(args.version, args.resume)
    else:
        transfer.VERSION, transfer.NAME = args.version, NAME
        transfer.RUN = ts.DEFAULT / f'submission_v{args.version}'
        transfer.MANIFEST = transfer.RUN / f'submission_v{args.version}.json'
        (transfer.upload if args.action == 'upload' else transfer.result)()


if __name__ == '__main__':
    main()
