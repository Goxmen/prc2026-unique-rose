"""Matched v39/v40 controls and the fixed 12-input disruption package.

Run: .venv/bin/python VALIDATION/opdi_disruption_run.py --variant v39 [--resume]
Repeat for v40. No full-year fits, submissions or ledger edits.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import glob
import json
from pathlib import Path
import sys
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import disruption_features as features
import disruption_run as prior
import opdi
import opdi_run as recipe
import overnight_run
import rome_hybrid
import rome_signals as signals
import seasonal
import weather
from common import TRAIN_GLOB, RANKING
from records import RUNS, data_identity, runtime_identity, sha256, source_snapshot, write_json
from rome_hybrid_run import save_booster

DEFAULT = RUNS / 'v3_20260924_opdi_disruption'
DESIGN = ROOT / 'VALIDATION/docs/OPDI_DISRUPTION_EXPERIMENT.md'
HISTORICAL = {
    'v39': ('v3_20260919T211914721668Z', 'v3_20260919T223006736822Z'),
    'v40': ('v3_20260920T152734631417Z', 'v3_20260920T170600876190Z'),
}
ORDINARY = list(recipe.ORDINARIES['opdi'])
CLASSIFIER = list(recipe.CLASSIFIERS['opdi'])
EXTRA = ORDINARY + features.FEATURES
KEEP = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan', 'arrival_history_available',
        'arrival_history_days_available', 'raw_flight_id_missing', 'fold', 'month', 'seed']


def read(path):
    return json.loads(Path(path).read_text())


def historical(variant):
    base, combined = (read(RUNS / name / 'result.json') for name in HISTORICAL[variant])
    reference = read(RUNS / HISTORICAL['v40'][0] / 'result.json')
    # Current helper code is the original v40 implementation. v39 predates only
    # the training-exclusion CLI addition; its unchanged recipe is reconstructed
    # explicitly with exclusion disabled. All modeling-helper hashes must match.
    changed = [name for name, digest in reference['signature'].items() if sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f'Historical v40 helpers changed: {changed}')
    differences = {name: digest for name, digest in base['signature'].items() if sha256(ROOT / name) != digest}
    allowed = ({'VALIDATION/opdi_run.py': '503326f2c1227f8d7ffe57b41e9334f1f0d08676115b7d637565422664060c6d'}
               if variant == 'v39' else {})
    if differences != allowed:
        raise ValueError(f'Unexpected historical source difference: {differences}')
    for record in (base, combined):
        config = record['config']
        if (record['status'] != 'complete' or len(record['folds']) != 16
                or config['seeds'] != [42, 142, 242] or config['rounds'] != 800
                or config['sample'] != 1 or config['recovery'] != 'off'
                or config['prediction_rule'] != 'rome_4h' or not config['late'] or config['dayshift']
                or bool(config.get('exclude_rule_rows', False)) != (variant == 'v40')
                or config['ordinaries']['opdi'] != ORDINARY or config['classifiers']['opdi'] != CLASSIFIER):
            raise ValueError(f'Unexpected historical {variant} recipe')
    corrections = read(prior.V38 / 'result.json')['correction_parameters']
    if combined['correction_parameters'] != corrections:
        raise ValueError('Historical combined corrections differ from v38')
    for window, values in corrections.items():
        if values['late_mean'] != base['parameters'][window]['late_mean'] or values['day']['handler_stratified']:
            raise ValueError(f'Unexpected correction parameters: {window}')
    return base, combined, differences


def training_inputs(td, ta, config, reports, grid, expected_parameters):
    X, y, cats, raw = overnight_run.prepare_training(td, ta, config)
    params = signals.fit_parameters(X, y, raw)
    if params != expected_parameters:
        raise ValueError('Training correction parameters differ from historical recipe')
    drop = recipe.rule_rows(raw, config['prediction_rule']) if config['exclude_rule_rows'] else np.zeros(len(raw), bool)
    audit = pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), excluded_by_rule=drop))
    # Context donors and cats stay as prepared from all training movements.
    X, y, raw = X.loc[~drop], y[~drop], raw.loc[~drop]
    X = recipe.add_inputs(X, raw, reports, td, ta)
    X = features.add_features(X, raw, ta, grid)
    substituted = rome_hybrid.substituted(X, y, raw)
    audit['used_by_ordinary'] = audit.mvt_id.isin(raw.loc[~substituted, 'mvt_id'])
    audit['used_by_classifier'] = audit.mvt_id.isin(raw.loc[raw.ADEP_mvt.eq('LIRF'), 'mvt_id'])
    return X, y, cats, raw, params, audit


def fitted_model(directory, progress, key, X, y, raw, part, extra, seed, config):
    if key in progress['models']:
        return
    started = time.monotonic()
    if part == 'classifier':
        model = signals.fit_classifier(X, y, raw, recipe.KIND, seed, threads=config['threads'], extra=extra)
        expected = signals.classifier_matrix(X.iloc[:1], raw.iloc[:1], recipe.KIND, extra).columns.tolist()
    else:
        model = signals.fit_ordinary(X, y, raw, recipe.KIND, seed, rounds=config['rounds'],
                                    threads=config['threads'], extra=extra)
        expected = rome_hybrid.regressor_features(recipe.KIND, extra)
    if model.feature_name() != list(expected):
        raise ValueError(f'Model feature order changed: {key}')
    if part == 'classifier' and set(features.FEATURES) & set(model.feature_name()):
        raise ValueError('New inputs leaked into the shared classifier')
    path = directory / 'models' / f'{key}.txt'
    save_booster(model, path)
    progress['models'][key] = dict(path=str(path.relative_to(directory)), part=part, seed=seed,
        features=model.feature_name(), artifacts=overnight_run.artifact_hashes(directory, path),
        seconds=time.monotonic() - started)
    write_json(directory / 'progress.json', progress)
    print(f'Fitted {key} in {time.monotonic() - started:.1f}s', flush=True)


def load_model(directory, progress, key):
    info = progress['models'][key]
    prior.checked_file(directory, info)
    return lgb.Booster(model_file=str(directory / info['path']), params={'num_threads': 20})


def check_shared_preparation(raw, V, job, ev, parent, parent_progress, corrections, config):
    """Before expensive fits, prove unchanged upstream inputs reproduce saved v38."""
    maxima = {}
    for seed in config['seeds']:
        for part in ('ordinary', 'classifier'):
            prior.checked_file(prior.PARENT, parent_progress['models'][prior.control.model_key(job, part, seed)])
        old = prior.control.load_models(prior.PARENT, parent_progress, job, seed, 20)
        p = signals.substitution_probability(old['classifier'], V, raw, recipe.KIND, 20, prior.control.CLASSIFIER)
        base = signals.ordinary_prediction(old['ordinary'], V, recipe.KIND, 20, prior.control.ORDINARY)
        frame = prior.frame_for(raw, V, job, ev, seed, base, p, config, parent['parameters'][job['id']], corrections)
        v38 = read(prior.V38 / 'result.json')
        progress = read(prior.V38 / 'progress.json')
        prior.checked_file(prior.V38, progress['predictions'][f"{ev['fold']}__{seed}"])
        saved = pd.read_parquet(prior.V38 / v38['folds'][ev['fold']]['predictions'][str(seed)])
        maxima[str(seed)] = prior.verify_prediction(frame, saved)
    return maxima


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=sorted(HISTORICAL))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    variant = args.variant
    base, combined, source_differences = historical(variant)
    cache = read(opdi.OPDI_DIR / 'build_manifest.json')
    if cache['status'] != 'complete' or sha256(opdi.FEATURES_PATH) != cache['features']['sha256']:
        raise ValueError('Missing or changed rebuilt OPDI cache')
    config = copy.deepcopy(base['config'])
    config.update(label=f'{variant}-snow-and-completed-arrivals', threads=6, prediction_threads=20,
                  fit_workers=1, exclude_rule_rows=(variant == 'v40'), disruption=features.SETTINGS,
                  candidate_ordinary_extra=EXTRA, candidates={'control': 'pred_control', 'candidate': 'pred_disruption'})
    sources = sorted(set(recipe.SOURCES + prior.control.SOURCES + list(read(prior.V38 / 'result.json')['signature']) + [
        'VALIDATION/opdi_disruption_run.py', 'VALIDATION/disruption_run.py',
        'VALIDATION/disruption_features.py', 'VALIDATION/docs/OPDI_DISRUPTION_EXPERIMENT.md',
        'VALIDATION/tools/score_opdi_disruption.py', 'VALIDATION/tests/test_opdi_disruption.py']))
    signature = {name: sha256(ROOT / name) for name in sources}
    identity = data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR])
    if identity != base['data_identity']:
        raise ValueError('Movement/weather data differs from historical model')
    opdi_identity = data_identity([opdi.FEATURES_PATH])
    anchors = {str(path.relative_to(ROOT)): sha256(path) for path in [
        *(RUNS / name / 'result.json' for name in HISTORICAL[variant]),
        prior.PARENT / 'result.json', prior.PARENT / 'progress.json', prior.V38 / 'result.json',
        opdi.OPDI_DIR / 'build_manifest.json', ROOT / 'SUBMISSIONS.md',
        ROOT / 'VALIDATION/reports/weighted_cv_leaderboard.json']}
    directory = (args.output_dir or DEFAULT / variant).resolve()
    if args.resume:
        record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
        for key, value in [('signature', signature), ('config', config), ('data_identity', identity),
                           ('opdi_identity', opdi_identity), ('anchors', anchors)]:
            if record[key] != value:
                raise ValueError(f'Cannot resume: changed {key}')
        now = runtime_identity()
        if any(record['runtime_identity'][k] != now[k] for k in ('python', 'versions')):
            raise ValueError('Cannot resume: changed numerical runtime')
        overnight_run.check_artifacts(directory, progress)
        for item in progress['training_audits'].values():
            prior.checked_file(directory, item)
        if record['status'] == 'complete':
            print(f'{variant} already complete', flush=True)
            return
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'models').mkdir()
        record = dict(run_id=directory.name, variant=variant, status='running',
            at=datetime.now(timezone.utc).isoformat(), protocol=base['protocol'], score_version=base['score_version'],
            signature=signature, config=config, data_identity=identity, opdi_identity=opdi_identity,
            anchors=anchors, runtime_identity=runtime_identity(), source=source_snapshot(directory),
            historical_runs=HISTORICAL[variant], historical_source_differences=source_differences,
            historical_opdi_identity=base['opdi_identity'], ranking_weights=base['ranking_weights'],
            full_benchmark=True, prediction_column='pred_disruption', design_sha256=sha256(DESIGN),
            correction_parameters=combined['correction_parameters'], folds={}, january_holdout={}, fits=0, uploads=0)
        progress = dict(models={}, predictions={}, folds={}, january_holdout={}, parameters={},
                        coverage={}, training_audits={}, preparation_checks={})
        write_json(directory / 'result.json', record)
        write_json(directory / 'progress.json', progress)
    started = time.monotonic()
    try:
        reports = weather.load_reports()
        grid = features.snow_grid(pd.read_parquet(weather.METAR))
        dep, arr = overnight_run.load_inputs(1.)
        parent, parent_progress = read(prior.PARENT / 'result.json'), read(prior.PARENT / 'progress.json')
        jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
        for number, job in enumerate(jobs):
            winter = job['id'] == 'v3_2025-11'
            if all(ev['fold'] in progress['folds'] for ev in job['evals']) and (not winter or progress['january_holdout']):
                print(f'Reusing {variant}/{job["id"]}', flush=True)
                continue
            print(f'Preparing {variant}/{job["id"]}', flush=True)
            td, ta, validation = overnight_run.split_job(dep, arr, job)
            X, y, cats, raw, params, audit = training_inputs(td, ta, config, reports, grid, base['parameters'][job['id']])
            progress['parameters'][job['id']] = params
            progress['coverage'][job['id']] = prior.coverage(X, raw, ta)
            progress['coverage'][job['id']]['opdi'] = recipe.coverage(X, raw)
            path = directory / 'models' / f'{job["id"]}.train_ids.parquet'
            if job['id'] in progress['training_audits']:
                pd.testing.assert_frame_equal(audit, pd.read_parquet(path))
            else:
                overnight_run.save_frame(path, audit)
                progress['training_audits'][job['id']] = dict(path=str(path.relative_to(directory)),
                    rows=len(audit), excluded=int(audit.excluded_by_rule.sum()),
                    ordinary_rows=int(audit.used_by_ordinary.sum()), classifier_rows=int(audit.used_by_classifier.sum()),
                    artifacts=overnight_run.artifact_hashes(directory, path))
            print(f'Training rows {len(raw):,}; rule exclusions {int(audit.excluded_by_rule.sum())}', flush=True)
            del audit
            if number == 0 and not progress['preparation_checks']:
                ev = job['evals'][0]
                vr, V = recipe.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                progress['preparation_checks'] = check_shared_preparation(vr, V, job, ev, parent, parent_progress,
                    record['correction_parameters'][job['id']], config)
                print('Upstream preparation reproduces all three v38 control seeds exactly', flush=True)
                del vr, V
                gc.collect()
            write_json(directory / 'progress.json', progress)
            for seed in config['seeds']:
                for name, part, extra in [('control', 'ordinary', ORDINARY), ('candidate', 'ordinary', EXTRA),
                                           ('shared', 'classifier', CLASSIFIER)]:
                    key = f'{job["id"]}__{name}__{seed}'
                    fitted_model(directory, progress, key, X, y, raw, part, extra, seed, config)
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
                print(f'Preparing evaluation {variant}/{name}', flush=True)
                raw, V = recipe.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                donors = pd.concat([ta, validation[ev['month']][1]], ignore_index=True)
                V = features.add_features(V, raw, donors, grid)
                progress['coverage'][name] = prior.coverage(V, raw, donors)
                paths = {}
                for seed in config['seeds']:
                    key = f'{name}__{seed}'
                    if key in progress['predictions']:
                        paths[str(seed)] = progress['predictions'][key]['path']
                        continue
                    classifier = load_model(directory, progress, f'{job["id"]}__shared__{seed}')
                    p = signals.substitution_probability(classifier, V, raw, recipe.KIND, 20, CLASSIFIER)
                    frames, predictions = {}, {}
                    for arm, extra in [('control', ORDINARY), ('candidate', EXTRA)]:
                        booster = load_model(directory, progress, f'{job["id"]}__{arm}__{seed}')
                        ordinary = signals.ordinary_prediction(booster, V, recipe.KIND, 20, extra)
                        predictions[arm] = ordinary
                        frames[arm] = prior.frame_for(raw, V, job, ev, seed, ordinary, p, config, params,
                                                     record['correction_parameters'][job['id']])
                        del booster
                    out = frames['control'][KEEP].copy()
                    out['pred_control'] = frames['control'].pred_combined.to_numpy()
                    out['pred_disruption'] = frames['candidate'].pred_combined.to_numpy()
                    out['ordinary_control'], out['ordinary_disruption'] = predictions['control'], predictions['candidate']
                    out['p_shared'] = p
                    out['rule_triggered'] = frames['control'].rule_triggered.to_numpy()
                    for col in features.FEATURES + opdi.OPDI_FEATURES:
                        out[col] = V[col].to_numpy()
                    if not np.isfinite(out[['y', 'pred_control', 'pred_disruption', 'p_shared']].to_numpy()).all():
                        raise ValueError('Nonfinite predictions')
                    path = directory / f'oof_{name}_seed{seed}.parquet'
                    overnight_run.save_frame(path, out)
                    paths[str(seed)] = path.name
                    progress['predictions'][key] = dict(path=path.name, rows=len(out),
                        artifacts=overnight_run.artifact_hashes(directory, path))
                    write_json(directory / 'progress.json', progress)
                    del classifier, frames, predictions, out
                if name == 'JAN_RETRO':
                    progress['january_holdout'] = dict(name=name, val=ev['month'], model_id=job['id'], predictions=paths)
                else:
                    progress['folds'][name] = dict(base['folds'][name], predictions=paths)
                write_json(directory / 'progress.json', progress)
                print(f'Saved {variant}/{name}: {len(raw):,} rows x three seeds', flush=True)
                del raw, V, donors
                gc.collect()
            del td, ta, validation, cats
            gc.collect()
        if len(progress['models']) != 54 or len(progress['folds']) != 16 or len(progress['predictions']) != 51:
            raise ValueError('Incomplete experiment roster')
        record.update(status='complete', **{k: progress[k] for k in
            ('models', 'folds', 'january_holdout', 'parameters', 'coverage', 'training_audits', 'preparation_checks')},
            fits=len(progress['models']))
        record.pop('error', None)
        print(f'Complete {variant}: 54 fits, 16 folds and January', flush=True)
    except BaseException:
        record.update(status='interrupted', error=traceback.format_exc())
        raise
    finally:
        record['minutes'] = record.get('minutes', 0) + (time.monotonic() - started) / 60
        write_json(directory / 'result.json', record)


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
