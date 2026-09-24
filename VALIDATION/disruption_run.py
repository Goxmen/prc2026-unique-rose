"""One frozen feature experiment versus saved v38; no refits or submissions.

Run .venv/bin/python VALIDATION/disruption_run.py [--resume]
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
import artifact_control_run as control
import artifact_experiments_run as derived
import disruption_features as features
import overnight_run
import plan_window
import record_fixes
import rome_signals as signals
import seasonal
import weather
from common import TRAIN_GLOB, RANKING
from records import RUNS, data_identity, runtime_identity, sha256, source_snapshot, write_json
from rome_hybrid_run import save_booster

PARENT = RUNS / 'v3_20260919_artifact_control'
V38 = RUNS / 'v3_20260919_artifact_experiments'
DEFAULT = RUNS / 'v3_20260921_disruption'
DESIGN = ROOT / 'VALIDATION/docs/DISRUPTION_EXPERIMENT.md'
EXTRA = list(control.ORDINARY) + features.FEATURES


def read(path):
    return json.loads(path.read_text())


def checked_file(directory, item):
    for name, digest in item['artifacts'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Changed saved artifact: {directory / name}')


def frame_for(raw, V, job, ev, seed, base, probability, config, params, corrections):
    out = signals.combine(base, probability, V, raw, late_mean=params['late_mean'], dayshift_weight=None)
    frame = overnight_run.prediction_metadata(raw, V, job, ev).assign(seed=seed)
    frame['pred_noid_cap'] = record_fixes.noid_cap(plan_window.apply(out['final'], raw), raw)
    for col in ('ordinary', 'p', 'rule_triggered', 'late_rows'):
        frame[col] = out[col]
    frame['gap_sec'] = V[control.rome_hybrid.GAP].to_numpy(float)
    audit = {col: raw[col].to_numpy() for col in control.RAW_AUDIT}
    audit.update({col: V[col].to_numpy() for col in control.FEATURE_AUDIT})
    frame = pd.concat([frame, pd.DataFrame(audit, index=frame.index)], axis=1)
    return derived.derive(frame, corrections, record_fixes.twin_rows(frame))


def verify_prediction(actual, saved):
    if actual.mvt_id.duplicated().any() or saved.mvt_id.duplicated().any():
        raise ValueError('Duplicate prediction IDs')
    a, b = (frame.set_index('mvt_id').sort_index() for frame in (actual, saved))
    if not a.index.equals(b.index):
        raise ValueError('Control IDs changed')
    for col in ('y', 'ordinary', 'p', 'pred_combined'):
        np.testing.assert_allclose(a[col], b[col], atol=1e-8, rtol=0, err_msg=f'v38 replay mismatch: {col}')
    return float(np.max(np.abs(a.pred_combined - b.pred_combined)))


def coverage(X, raw, arrivals):
    taxi = arrivals.taxi_sec.to_numpy(float)
    return dict(departures=len(raw), arrival_donors=len(arrivals),
                invalid_arrival_labels=int((~np.isfinite(taxi) | (taxi < 0)).sum()),
                features={col: dict(known=int(X[col].notna().sum()),
                    minimum=float(X[col].min()), maximum=float(X[col].max())) for col in features.FEATURES})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    control.historical_recipe()  # Preserve existing historical source guards.
    parent, parent_progress, v38 = read(PARENT / 'result.json'), read(PARENT / 'progress.json'), read(V38 / 'result.json')
    for record in (parent, v38):
        if record['status'] != 'complete':
            raise ValueError('Control is incomplete')
        changed = [name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest]
        if changed:
            raise ValueError(f'Frozen control source changed: {changed}')
    config = copy.deepcopy(parent['config'])
    config.update(label='snow-duration-and-completed-arrival-congestion', disruption=features.SETTINGS,
                  ordinary_extra=EXTRA, threads=6, prediction_threads=20, fit_workers=1)
    sources = sorted(set(control.SOURCES + list(v38['signature']) + [
        'VALIDATION/disruption_features.py', 'VALIDATION/disruption_run.py',
        'VALIDATION/docs/DISRUPTION_EXPERIMENT.md', 'VALIDATION/tests/test_disruption_features.py']))
    signature = {name: sha256(ROOT / name) for name in sources}
    identity = data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR])
    if identity != parent['data_identity']:
        raise ValueError('Data differs from the saved control')
    anchors = {str(path.relative_to(ROOT)): sha256(path) for path in
               (PARENT / 'result.json', PARENT / 'progress.json', V38 / 'result.json', V38 / 'progress.json')}
    directory = args.output_dir.resolve()
    if args.resume:
        record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
        for name, value in [('signature', signature), ('data_identity', identity), ('config', config), ('anchors', anchors)]:
            if record[name] != value:
                raise ValueError(f'Cannot resume: changed {name}')
        now = runtime_identity()
        if any(record['runtime_identity'][k] != now[k] for k in ('python', 'versions')):
            raise ValueError('Cannot resume with changed runtime')
        overnight_run.check_artifacts(directory, progress)
        if record['status'] == 'complete':
            print('Run already complete', flush=True)
            return
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'models').mkdir()
        record = dict(run_id=directory.name, status='running', at=datetime.now(timezone.utc).isoformat(),
            protocol=parent['protocol'], score_version=parent['score_version'], config=config,
            signature=signature, data_identity=identity, anchors=anchors, runtime_identity=runtime_identity(),
            ranking_weights=parent['ranking_weights'], prediction_column='pred_disruption',
            baseline_id=V38.name, design_sha256=sha256(DESIGN), full_benchmark=True,
            source=source_snapshot(directory), folds={}, january_holdout={}, fits=0, uploads=0)
        progress = dict(models={}, predictions={}, folds={}, january_holdout={}, coverage={}, replay_checks={})
        write_json(directory / 'result.json', record)
        write_json(directory / 'progress.json', progress)
    started = time.monotonic()
    try:
        reports = weather.load_reports()
        grid = features.snow_grid(pd.read_parquet(weather.METAR))
        dep, arr = overnight_run.load_inputs(1.)
        jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
        for job in jobs:
            winter = job['id'] == 'v3_2025-11'
            if all(ev['fold'] in progress['folds'] for ev in job['evals']) and (not winter or progress['january_holdout']):
                print(f"Reusing {job['id']}", flush=True)
                continue
            print(f"Preparing {job['id']}", flush=True)
            td, ta, validation = overnight_run.split_job(dep, arr, job)
            X, y, cats, raw = overnight_run.prepare_training(td, ta, config)
            X = control.row30.add_inputs(X, raw, reports, td, ta)
            X = features.add_features(X, raw, ta, grid)
            progress['coverage'][job['id']] = coverage(X, raw, ta)
            for seed in config['seeds']:
                key = f"{job['id']}__{seed}"
                if key in progress['models']:
                    continue
                start_fit = time.monotonic()
                booster = signals.fit_ordinary(X, y, raw, control.row30.KIND, seed,
                    rounds=800, threads=6, extra=EXTRA)
                path = directory / 'models' / f'{key}.txt'
                save_booster(booster, path)
                progress['models'][key] = dict(path=str(path.relative_to(directory)),
                    artifacts=overnight_run.artifact_hashes(directory, path), seconds=time.monotonic() - start_fit)
                write_json(directory / 'progress.json', progress)
                print(f"Fitted {key} in {time.monotonic() - start_fit:.1f}s", flush=True)
                del booster
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
                print(f"Preparing evaluation {name}", flush=True)
                raw, V = control.row30.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                donors = pd.concat([ta, validation[ev['month']][1]], ignore_index=True)
                V = features.add_features(V, raw, donors, grid)
                progress['coverage'][name] = coverage(V, raw, donors)
                paths = {}
                for seed in config['seeds']:
                    key = f'{name}__{seed}'
                    if key in progress['predictions']:
                        paths[str(seed)] = progress['predictions'][key]['path']
                        continue
                    for part in ('ordinary', 'classifier'):
                        checked_file(PARENT, parent_progress['models'][control.model_key(job, part, seed)])
                    old = control.load_models(PARENT, parent_progress, job, seed, 20)
                    p = signals.substitution_probability(old['classifier'], V, raw, control.row30.KIND,
                                                         20, control.CLASSIFIER)
                    base = signals.ordinary_prediction(old['ordinary'], V, control.row30.KIND, 20, control.ORDINARY)
                    params, corrections = parent['parameters'][job['id']], v38['correction_parameters'][job['id']]
                    replay = frame_for(raw, V, job, ev, seed, base, p, config, params, corrections)
                    if name != 'JAN_RETRO':
                        saved_path = V38 / v38['folds'][name]['predictions'][str(seed)]
                        checked_file(V38, read(V38 / 'progress.json')['predictions'][key])
                        progress['replay_checks'][key] = verify_prediction(replay, pd.read_parquet(saved_path))
                    booster = lgb.Booster(model_file=str(directory / progress['models'][f"{job['id']}__{seed}"]['path']),
                                          params={'num_threads': 20})
                    candidate_base = signals.ordinary_prediction(booster, V, control.row30.KIND, 20, EXTRA)
                    candidate = frame_for(raw, V, job, ev, seed, candidate_base, p, config, params, corrections)
                    keep = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan', 'arrival_history_available',
                            'arrival_history_days_available', 'raw_flight_id_missing', 'fold', 'month', 'seed']
                    out = replay[keep].copy()
                    out['pred_control'] = replay.pred_combined.to_numpy()
                    out['pred_disruption'] = candidate.pred_combined.to_numpy()
                    for col in features.FEATURES:
                        out[col] = V[col].to_numpy()
                    if not np.isfinite(out[['y', 'pred_control', 'pred_disruption']].to_numpy()).all():
                        raise ValueError('Nonfinite predictions')
                    path = directory / f'oof_{name}_seed{seed}.parquet'
                    overnight_run.save_frame(path, out)
                    paths[str(seed)] = path.name
                    progress['predictions'][key] = dict(path=path.name, rows=len(out),
                        artifacts=overnight_run.artifact_hashes(directory, path))
                    write_json(directory / 'progress.json', progress)
                    del old, booster, replay, candidate, out
                if name == 'JAN_RETRO':
                    progress['january_holdout'] = dict(name=name, val=ev['month'], model_id=job['id'], predictions=paths)
                else:
                    progress['folds'][name] = dict(v38['folds'][name], predictions=paths)
                write_json(directory / 'progress.json', progress)
                print(f"Saved {name}: {len(raw):,} rows x 3 seeds", flush=True)
                del raw, V, donors
                gc.collect()
            del td, ta, validation, cats
            gc.collect()
        record.update(status='complete', folds=progress['folds'], january_holdout=progress['january_holdout'],
                      fits=len(progress['models']), models=progress['models'], coverage=progress['coverage'],
                      replay_checks=progress['replay_checks'], minutes=time.monotonic() / 60 - started / 60)
        record.pop('error', None)
        write_json(directory / 'result.json', record)
        print(f"Complete: {record['fits']} fits, {len(record['folds'])} folds + January; {record['minutes']:.1f} minutes", flush=True)
    except BaseException:
        record.update(status='interrupted', error=traceback.format_exc())
        write_json(directory / 'result.json', record)
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
