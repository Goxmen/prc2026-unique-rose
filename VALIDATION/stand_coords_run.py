"""Stand position inputs on v42: matched control, candidate, and a hidden-stand test.

Design: VALIDATION/docs/STAND_COORDINATES_EXPERIMENT.md (frozen before the fits).

Run:  .venv/bin/python VALIDATION/stand_coords_run.py --mode main   [--resume]
      .venv/bin/python VALIDATION/stand_coords_run.py --mode hidden [--resume]
`hidden` reuses the main run's seed-42 classifiers, so it runs second.
No full-year fits, submissions or ledger edits.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import glob
import hashlib
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
import opdi_disruption_run as v42
import stand_features as sf
from common import TRAIN_GLOB, RANKING
from records import RUNS, data_identity, runtime_identity, sha256, source_snapshot, write_json

features, prior, recipe, signals = v42.features, v42.prior, v42.recipe, v42.signals
overnight_run, seasonal, weather, opdi = v42.overnight_run, v42.seasonal, v42.weather, v42.opdi

DEFAULT = RUNS / 'v3_20260925_stand_coords'
DESIGN = ROOT / 'VALIDATION/docs/STAND_COORDINATES_EXPERIMENT.md'
CONTROL = list(v42.EXTRA)
CANDIDATE = CONTROL + sf.FEATURES
CLASSIFIER = list(v42.CLASSIFIER)
THREADS = 8                       # Apple M3, 8 cores
HIDDEN = dict(share=0.10, seed=42, salt='stand-coords-hidden',
              blanked=['stand', 'stand_taxi_90d'])
KEEP = v42.KEEP
EXPECTED = {'main': dict(models=54, folds=16, predictions=51),
            'hidden': dict(models=12, folds=16, predictions=16)}


def read(path):
    return json.loads(Path(path).read_text())


def stand_keys(raw):
    return (raw['ADEP_mvt'].astype('string') + ':' + raw['STAND_mvt'].astype('string'))


def is_hidden(keys, window, share=HIDDEN['share'], salt=HIDDEN['salt']):
    """Deterministic ~share of stand names per window; missing stands are never hidden."""
    def draw(key):
        digest = hashlib.sha256(f'{salt}|{window}|{key}'.encode()).hexdigest()
        return int(digest[:16], 16) / 16 ** 16 < share
    keys = pd.Series(keys, dtype='string')
    chosen = {k: draw(k) for k in keys.dropna().unique()}
    return keys.map(chosen).fillna(False).to_numpy(bool)


def blank_hidden(V, mask):
    """How a first-seen stand reaches the model: unseen name, no stand history."""
    V = V.copy()
    for col in HIDDEN['blanked']:
        V.loc[mask, col] = np.nan
    return V


def load_model(directory, progress, key):
    info = progress['models'][key]
    prior.checked_file(directory, info)
    return lgb.Booster(model_file=str(directory / info['path']), params={'num_threads': THREADS})


def check_classifier(model):
    if set(sf.FEATURES + features.FEATURES) & set(model.feature_name()):
        raise ValueError('New inputs leaked into the shared classifier')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=sorted(EXPECTED))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    mode, hidden = args.mode, args.mode == 'hidden'
    base, combined, differences = v42.historical('v40')
    config = copy.deepcopy(base['config'])
    config.update(label=f'v42-stand-coordinates-{mode}', threads=THREADS, prediction_threads=THREADS,
                  fit_workers=1, exclude_rule_rows=True, disruption=features.SETTINGS, stand=sf.SETTINGS,
                  control_ordinary_extra=CONTROL, candidate_ordinary_extra=CANDIDATE,
                  candidates={'control': 'pred_control', 'candidate': 'pred_stand'})
    if hidden:
        config.update(seeds=[HIDDEN['seed']], hidden=HIDDEN)
    v40 = read(RUNS / 'v3_20260924_opdi_disruption/v40/result.json')
    own = ['VALIDATION/stand_coords_run.py', 'VALIDATION/stand_features.py', 'VALIDATION/stand_geometry.py',
           'VALIDATION/docs/STAND_COORDINATES_EXPERIMENT.md', 'VALIDATION/tests/test_stand_features.py']
    signature = {name: sha256(ROOT / name) for name in sorted(set(v40['signature']) | set(own))
                 if (ROOT / name).exists()}
    # v42's CV code must be byte-identical to what gunes ran, or the control is not v42
    changed = sorted(n for n, d in v40['signature'].items() if signature.get(n) != d)
    if changed:
        raise ValueError(f'v42 experiment sources changed since its run: {changed}')
    identity = data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR])
    if identity != base['data_identity']:
        raise ValueError('Movement/weather data differs from historical model')
    stand_identity = data_identity([sf.COORDS, sf.ENDS])
    opdi_identity = data_identity([opdi.FEATURES_PATH])
    directory = (args.output_dir or DEFAULT / mode).resolve()
    main_dir = (args.output_dir.parent / 'main' if args.output_dir else DEFAULT / 'main').resolve()
    if hidden:
        main_record, main_progress = read(main_dir / 'result.json'), read(main_dir / 'progress.json')
        if main_record['status'] != 'complete' or main_record['signature'] != signature:
            raise ValueError('Hidden-stand test needs the complete main run from the same sources')
    if args.resume:
        record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
        for key, value in [('signature', signature), ('config', config), ('data_identity', identity),
                           ('stand_identity', stand_identity), ('opdi_identity', opdi_identity)]:
            if record[key] != value:
                raise ValueError(f'Cannot resume: changed {key}')
        overnight_run.check_artifacts(directory, progress)
        if record['status'] == 'complete':
            print(f'{mode} already complete', flush=True)
            return
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'models').mkdir()
        record = dict(run_id=directory.name, mode=mode, status='running', at=datetime.now(timezone.utc).isoformat(),
            protocol=base['protocol'], score_version=base['score_version'], signature=signature, config=config,
            data_identity=identity, stand_identity=stand_identity, opdi_identity=opdi_identity,
            v42_cv_opdi_identity=v40['opdi_identity'], runtime_identity=runtime_identity(),
            source=source_snapshot(directory), historical_runs=v42.HISTORICAL['v40'],
            historical_source_differences=differences, ranking_weights=base['ranking_weights'],
            prediction_column='pred_stand', design_sha256=sha256(DESIGN),
            correction_parameters=combined['correction_parameters'], folds={}, january_holdout={}, fits=0, uploads=0)
        progress = dict(models={}, predictions={}, folds={}, january_holdout={}, parameters={},
                        coverage={}, hidden_stands={})
        write_json(directory / 'result.json', record)
        write_json(directory / 'progress.json', progress)
    started = time.monotonic()
    tables = sf.load()
    try:
        reports = weather.load_reports()
        grid = features.snow_grid(pd.read_parquet(weather.METAR))
        dep, arr = overnight_run.load_inputs(1.)
        jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
        for job in jobs:
            winter = job['id'] == 'v3_2025-11' and not hidden
            if all(ev['fold'] in progress['folds'] for ev in job['evals']) and (not winter or progress['january_holdout']):
                print(f'Reusing {mode}/{job["id"]}', flush=True)
                continue
            print(f'Preparing {mode}/{job["id"]}', flush=True)
            td, ta, validation = overnight_run.split_job(dep, arr, job)
            X, y, cats, raw, params, _ = v42.training_inputs(td, ta, config, reports, grid,
                                                             base['parameters'][job['id']])
            X = sf.add_features(X, raw, tables)
            progress['parameters'][job['id']] = params
            progress['coverage'][job['id']] = sf.coverage(X)
            if hidden:
                mask = is_hidden(stand_keys(raw), job['id'])
                progress['hidden_stands'][job['id']] = dict(training_rows_removed=int(mask.sum()),
                    stands=int(stand_keys(raw)[mask].nunique()), training_rows=len(raw))
                X, y, raw = X.loc[~mask], np.asarray(y)[~mask], raw.loc[~mask]
                print(f'Hidden stands: removed {int(mask.sum()):,} training rows', flush=True)
            write_json(directory / 'progress.json', progress)
            arms = [('control', 'ordinary', CONTROL), ('candidate', 'ordinary', CANDIDATE)]
            if not hidden:
                arms.append(('shared', 'classifier', CLASSIFIER))
            for seed in config['seeds']:
                for name, part, extra in arms:
                    v42.fitted_model(directory, progress, f'{job["id"]}__{name}__{seed}',
                                     X, y, raw, part, extra, seed, config)
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
                print(f'Preparing evaluation {mode}/{name}', flush=True)
                vr, V = recipe.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                donors = pd.concat([ta, validation[ev['month']][1]], ignore_index=True)
                V = features.add_features(V, vr, donors, grid)
                V = sf.add_features(V, vr, tables)
                mask = is_hidden(stand_keys(vr), job['id']) if hidden else np.zeros(len(vr), bool)
                if hidden:
                    V = blank_hidden(V, mask)
                progress['coverage'][name] = sf.coverage(V)
                paths = {}
                for seed in config['seeds']:
                    key = f'{name}__{seed}'
                    if key in progress['predictions']:
                        paths[str(seed)] = progress['predictions'][key]['path']
                        continue
                    source, source_progress = (main_dir, main_progress) if hidden else (directory, progress)
                    classifier = load_model(source, source_progress, f'{job["id"]}__shared__{seed}')
                    check_classifier(classifier)
                    p = signals.substitution_probability(classifier, V, vr, recipe.KIND, THREADS, CLASSIFIER)
                    frames, ordinary = {}, {}
                    for arm, extra in [('control', CONTROL), ('candidate', CANDIDATE)]:
                        booster = load_model(directory, progress, f'{job["id"]}__{arm}__{seed}')
                        ordinary[arm] = signals.ordinary_prediction(booster, V, recipe.KIND, THREADS, extra)
                        frames[arm] = prior.frame_for(vr, V, job, ev, seed, ordinary[arm], p, config, params,
                                                      record['correction_parameters'][job['id']])
                        del booster
                    out = frames['control'][KEEP].copy()
                    out['pred_control'] = frames['control'].pred_combined.to_numpy()
                    out['pred_stand'] = frames['candidate'].pred_combined.to_numpy()
                    out['ordinary_control'], out['ordinary_stand'] = ordinary['control'], ordinary['candidate']
                    out['p_shared'] = p
                    out['rule_triggered'] = frames['control'].rule_triggered.to_numpy()
                    out['stand_key'] = stand_keys(vr).to_numpy()
                    out['hidden_stand'] = mask
                    for col in sf.FEATURES:
                        out[col] = V[col].to_numpy()
                    if not np.isfinite(out[['y', 'pred_control', 'pred_stand', 'p_shared']].to_numpy()).all():
                        raise ValueError('Nonfinite predictions')
                    path = directory / f'oof_{name}_seed{seed}.parquet'
                    overnight_run.save_frame(path, out)
                    paths[str(seed)] = path.name
                    progress['predictions'][key] = dict(path=path.name, rows=len(out),
                        hidden_rows=int(mask.sum()), artifacts=overnight_run.artifact_hashes(directory, path))
                    write_json(directory / 'progress.json', progress)
                    del classifier, frames, ordinary, out
                if name == 'JAN_RETRO':
                    progress['january_holdout'] = dict(name=name, val=ev['month'], model_id=job['id'], predictions=paths)
                else:
                    progress['folds'][name] = dict(base['folds'][name], predictions=paths)
                write_json(directory / 'progress.json', progress)
                print(f'Saved {mode}/{name}: {len(vr):,} rows ({int(mask.sum()):,} hidden)', flush=True)
                del vr, V, donors
                gc.collect()
            del td, ta, validation, cats
            gc.collect()
        want = EXPECTED[mode]
        if (len(progress['models']), len(progress['folds']), len(progress['predictions'])) != \
                (want['models'], want['folds'], want['predictions']):
            raise ValueError('Incomplete experiment roster')
        record.update(status='complete', **{k: progress[k] for k in
            ('models', 'folds', 'january_holdout', 'parameters', 'coverage', 'hidden_stands')},
            fits=len(progress['models']))
        record.pop('error', None)
        print(f'Complete {mode}: {len(progress["models"])} fits', flush=True)
    except BaseException:
        record.update(status='interrupted', error=traceback.format_exc())
        raise
    finally:
        record['minutes'] = record.get('minutes', 0) + (time.monotonic() - started) / 60
        write_json(directory / 'result.json', record)


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
