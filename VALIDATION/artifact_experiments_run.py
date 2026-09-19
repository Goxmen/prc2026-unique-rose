"""Validate frozen recording-artifact corrections on the replay's saved predictions.

No new boosters, full-year models or submissions are created. Every candidate
uses all 16 folds and three seeds. Correction parameters use each window's sparse
training audit only. Twins are found among the current validation month's rows.

Run after the parent has completed:
  .venv/bin/python VALIDATION/artifact_experiments_run.py \
      --parent v3_20260919_artifact_control --output-dir VALIDATION/runs/v3_<name>
Resume with the same options plus --resume. Add --sensitivity combined,clock12
to compute three-day bootstrap comparisons; completed comparisons are reused.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import recording_artifacts as corrections
import record_fixes
from protocol import SCORE_VERSION, VERSION, select_folds
from records import RUNS, append_log, read_run, runtime_identity, sha256, source_snapshot, write_json
import reporting

CONTROL = 'pred_noid_cap'
SOURCES = [
    'VALIDATION/artifact_experiments_run.py', 'VALIDATION/recording_artifacts.py',
    'VALIDATION/record_fixes.py', 'VALIDATION/plan_window.py', 'VALIDATION/protocol.py',
    'VALIDATION/records.py', 'VALIDATION/reporting.py',
    'VALIDATION/docs/RECORDING_ARTIFACT_EXPERIMENTS.md',
]
META = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan', 'arrival_history_available',
        'arrival_history_days_available', 'raw_flight_id_missing', 'fold', 'month', 'seed', 'rule_triggered']
KEEP = list(dict.fromkeys([
    *META, CONTROL, 'ordinary', 'p', 'gap_sec', 'late_rows',
    *corrections.RAW, *corrections.SIGNALS, 'FLIGHT_mvt', 'STAND_mvt',
    'ADES_mvt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt',
]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--parent', required=True, help='completed artifact-control run ID or result.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--label', default='recording-artifact-corrections')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sensitivity', default='', help='comma-separated candidate names for 3-day intervals, or all')
    args = parser.parse_args(argv)
    if args.resume and args.output_dir is None:
        parser.error('--resume needs --output-dir')
    args.sensitivity = list(corrections.CANDIDATES) if args.sensitivity == 'all' else list(
        dict.fromkeys(name for name in args.sensitivity.split(',') if name))
    if any(name not in corrections.CANDIDATES for name in args.sensitivity):
        parser.error('--sensitivity contains an unknown candidate')
    return args


def checked_path(directory, name):
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f'Artifact path escapes the run directory: {name}')
    return path


def parent_inputs(source, directory):
    """Validate protocol completeness and hash only the parent artifacts actually read."""
    if (source.get('status') != 'complete' or source.get('protocol') != VERSION
            or source.get('score_version') != SCORE_VERSION or source.get('prediction_column') != CONTROL
            or source['config']['sample'] != 1 or source['config']['seeds'] != [42, 142, 242]
            or source['config']['rounds'] != 800 or source['config']['train_months'] != 6):
        raise ValueError('Parent must be a complete, full three-seed row-34 control replay')
    roster, skipped = select_folds('main', months=6)
    if skipped or set(source['folds']) != {fold.name for fold, _ in roster}:
        raise ValueError('Parent must contain every original validation fold')
    for fold, start in roster:
        saved = source['folds'][fold.name]
        expected = dict(train_start=str(start.date()), train_end=fold.train_end, val=fold.val,
                        freeze=fold.freeze, departure_history=fold.departure_history)
        if any(saved[key] != value for key, value in expected.items()):
            raise ValueError(f'Parent fold {fold.name} differs from the original protocol')
    progress = json.loads((directory / 'progress.json').read_text())
    fingerprints = {name: sha256(directory / name) for name in ('result.json', 'progress.json')}
    for name, fold in source['folds'].items():
        if set(fold['predictions']) != {'42', '142', '242'}:
            raise ValueError(f'Parent fold {name} has incomplete seeds')
        for seed, path in fold['predictions'].items():
            entry = progress['predictions'][f'{name}__{seed}']
            if entry['path'] != path:
                raise ValueError(f'Parent prediction path differs in {name}/{seed}')
            fingerprints[path] = entry['artifacts'][path]
        model = fold['model_id']
        audit = source['training_audits'][model]
        if audit != progress['training_audits'][model]:
            raise ValueError(f'Parent training-audit metadata differs for {model}')
        fingerprints[audit['path']] = audit['artifacts'][audit['path']]
    verify_artifacts(directory, fingerprints)
    return fingerprints


def verify_artifacts(directory, fingerprints):
    for name, digest in fingerprints.items():
        path = checked_path(directory, name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Missing or changed artifact: {path}')


def verify_resume(record, progress, directory, signature, parent_fingerprints, config):
    if record['signature'] != signature or record['config'] != config:
        raise ValueError('Cannot resume: correction source or configuration changed')
    if record['parent_input_fingerprints'] != parent_fingerprints:
        raise ValueError('Cannot resume: parent artifacts changed')
    runtime = runtime_identity()
    if any(record['runtime_identity'][key] != runtime[key] for key in ('python', 'versions')):
        raise ValueError('Cannot resume: Python or numerical-library versions changed')
    for kind in ('predictions', 'comparisons', 'summaries'):
        for entry in progress[kind].values():
            verify_artifacts(directory, entry['artifacts'])


def fit_parameters(source, parent, model_id):
    audit = source['training_audits'][model_id]
    training = pd.read_parquet(checked_path(parent, audit['path']))
    if len(training) != audit['rows']:
        raise ValueError(f'{model_id}: sparse training row count changed')
    info = next(fold for fold in source['folds'].values() if fold['model_id'] == model_id)
    times = pd.to_datetime(training.MVT_TIME_UTC_mvt, utc=True)
    start = pd.Timestamp(info['train_start'], tz='UTC')
    end = pd.Timestamp(info['train_end'], tz='UTC') + pd.Timedelta(days=1)
    if times.isna().any() or not ((times >= start) & (times < end)).all():
        raise ValueError(f'{model_id}: correction training row lies outside its window')
    return corrections.fit(training, late_mean=source['parameters'][model_id]['late_mean'])


def derive(frame, parameters, twin_mask):
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('OOF movements must be present and unique within a fold/seed')
    if not np.isfinite(frame[['y', CONTROL]].to_numpy(float)).all():
        raise ValueError('OOF labels and control predictions must be finite')
    predictions, masks = corrections.predict(frame[CONTROL].to_numpy(float), frame, parameters,
                                               probability=frame.p.to_numpy(float), twin_mask=twin_mask)
    out = frame.loc[:, KEEP].copy()
    for column, values in predictions.items():
        out[column] = values
    for name, mask in masks.items():
        out['mask_' + name] = mask
    return out


def change_stats(frame):
    y = frame.y.to_numpy(float)
    baseline = frame[CONTROL].to_numpy(float)
    old_sq = (y - baseline) ** 2
    results, any_changed = {}, np.zeros(len(frame), bool)
    for name, column in corrections.CANDIDATES.items():
        prediction = frame[column].to_numpy(float)
        changed = prediction != baseline
        new_sq = (y - prediction) ** 2
        results[name] = dict(rows_changed=int(changed.sum()),
            closer=int((changed & (new_sq < old_sq)).sum()), farther=int((changed & (new_sq > old_sq)).sum()),
            equal_error=int((changed & (new_sq == old_sq)).sum()),
            raised=int((prediction > baseline).sum()), lowered=int((prediction < baseline).sum()),
            repeated_row_sse_change=float((new_sq - old_sq).sum()))
        any_changed |= changed
    audit = frame.loc[any_changed, [*META, CONTROL, *corrections.CANDIDATES.values()]].copy()
    return results, audit


def save_frame(path, frame):
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def derive_predictions(source, parent, directory, record, progress):
    for fold, info in source['folds'].items():
        if all(f'{fold}__{seed}' in progress['predictions'] for seed in source['config']['seeds']):
            continue
        model = info['model_id']
        if model not in progress['parameters']:
            progress['parameters'][model] = fit_parameters(source, parent, model)
            write_json(directory / 'progress.json', progress)
        ids, twin_mask = None, None
        for seed in source['config']['seeds']:
            key = f'{fold}__{seed}'
            if key in progress['predictions']:
                continue
            frame = pd.read_parquet(parent / info['predictions'][str(seed)], columns=KEEP)
            if not frame.seed.eq(seed).all() or not frame.fold.eq(fold).all():
                raise ValueError(f'Parent frame identifies the wrong fold/seed: {key}')
            if not pd.to_datetime(frame.mvt_time, utc=True).dt.strftime('%Y-%m').eq(info['val']).all():
                raise ValueError(f'Parent frame exposes another validation month: {key}')
            if ids is None:
                ids = frame.mvt_id.to_numpy()
                twin_mask = record_fixes.twin_rows(frame)
            elif not np.array_equal(ids, frame.mvt_id.to_numpy()):
                raise ValueError(f'Parent movement order changes between seeds in {fold}')
            out = derive(frame, progress['parameters'][model], twin_mask)
            stats, changed = change_stats(out)
            path = directory / info['predictions'][str(seed)]
            changed_path = directory / f'changed_{fold}_seed{seed}.parquet'
            save_frame(path, out)
            save_frame(changed_path, changed)
            progress['predictions'][key] = dict(path=path.name, changes_path=changed_path.name,
                rows=len(out), statistics=stats, artifacts={path.name: sha256(path), changed_path.name: sha256(changed_path)})
            write_json(directory / 'progress.json', progress)
        print(f'Derived {fold}: all three seeds, {len(corrections.CANDIDATES)} frozen corrections', flush=True)
    record['correction_parameters'] = progress['parameters']
    record['folds'] = copy.deepcopy(source['folds'])


def summarize_changes(directory, progress):
    frames = [pd.read_parquet(directory / entry['changes_path']) for entry in progress['predictions'].values()]
    changed = pd.concat(frames, ignore_index=True)
    summary = {}
    for name, column in corrections.CANDIDATES.items():
        rows = changed[changed[column] != changed[CONTROL]]
        totals = {key: sum(entry['statistics'][name][key] for entry in progress['predictions'].values())
                  for key in next(iter(progress['predictions'].values()))['statistics'][name]}
        by_seed = {}
        for seed in (42, 142, 242):
            part = rows[rows.seed == seed]
            entries = [entry for key, entry in progress['predictions'].items() if key.endswith('__' + str(seed))]
            by_seed[str(seed)] = dict(
                **{key: sum(entry['statistics'][name][key] for entry in entries) for key in totals},
                unique_movements=int(part.mvt_id.nunique()),
                unique_utc_dates=int(pd.to_datetime(part.mvt_time, utc=True).dt.normalize().nunique()))
        summary[name] = dict(**totals, unique_movements=int(rows.mvt_id.nunique()),
            unique_utc_dates=int(pd.to_datetime(rows.mvt_time, utc=True).dt.normalize().nunique()), by_seed=by_seed,
            by_fold_seed={key: entry['statistics'][name] for key, entry in progress['predictions'].items()})
    return dict(note='SSE sums here count repeated fold/seed appearances; use paired report scores for protocol-weighted changes.',
                candidates=summary)


def comparison(record, directory, progress, name, block_days):
    key = f'{name}__block{block_days}'
    if key in progress['comparisons']:
        path = directory / progress['comparisons'][key]['path']
        return json.loads(path.read_text())
    candidate = copy.deepcopy(record)
    candidate['prediction_column'] = corrections.CANDIDATES[name]
    candidate['config'].update(block_days=block_days, bootstrap_reps=2000, bootstrap_seed=0)
    baseline = copy.deepcopy(candidate)
    baseline['prediction_column'] = CONTROL
    result = reporting.report(candidate, directory, (baseline, directory))
    payload = dict(run_id=record['run_id'], candidate=name,
        prediction_column=candidate['prediction_column'], baseline_prediction_column=CONTROL,
        bootstrap=dict(block_days=block_days, bootstrap_reps=2000, bootstrap_seed=0), report=result)
    path = directory / f'comparison_{name}_block{block_days}.json'
    write_json(path, payload)
    progress['comparisons'][key] = dict(path=path.name, artifacts={path.name: sha256(path)})
    write_json(directory / 'progress.json', progress)
    scores = result['scores']
    change = result['comparison']['scores']['R_combined']
    print(f"{name} / {block_days}-day: {scores['R_A']:.4f} / {scores['R_B_long']:.4f} / "
          f"{scores['R_combined']:.4f}; combined change {change['difference']:+.4f}", flush=True)
    return payload


def scorecard(record, directory, progress, sensitivity):
    rows = {}
    for name in corrections.CANDIDATES:
        result = comparison(record, directory, progress, name, 1)['report']
        rows[name] = dict(column=corrections.CANDIDATES[name],
            scores={key: result['scores'][key] for key in ('R_A', 'R_B_long', 'R_combined')},
            changes={key: result['comparison']['scores'][key] for key in ('R_A', 'R_B_long', 'R_combined')})
    for name in sensitivity:
        comparison(record, directory, progress, name, 3)
    return rows


def main(argv=None):
    args = parse_args(argv)
    source, parent = read_run(args.parent)
    fingerprints = parent_inputs(source, parent)
    signature = {name: sha256(ROOT / name) for name in SOURCES}
    config = copy.deepcopy(source['config'])
    config.update(label=args.label, candidates=dict(corrections.CANDIDATES, control=CONTROL),
                  recording_artifact_settings=corrections.SETTINGS,
                  twin_context='current validation month only; no labels',
                  bootstrap_reps=2000, bootstrap_seed=0, block_days=1)
    config = json.loads(json.dumps(config))
    directory = (args.output_dir or RUNS / datetime.now(timezone.utc).strftime('v3_%Y%m%dT%H%M%S%fZ')).resolve()
    if args.resume:
        record = json.loads((directory / 'result.json').read_text())
        progress = json.loads((directory / 'progress.json').read_text())
        verify_resume(record, progress, directory, signature, fingerprints, config)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        record = copy.deepcopy(source)
        for key in ('report', 'replay_checks', 'training_audits', 'parameters', 'input_coverage', 'reconstruction', 'minutes'):
            record.pop(key, None)
        record.update(run_id=directory.name, label=args.label, status='running',
            at=datetime.now(timezone.utc).isoformat(timespec='seconds'), config=config,
            signature=signature, source_signature=source['signature'], source=source_snapshot(directory),
            parent_input_fingerprints=fingerprints, runtime_identity=runtime_identity(),
            derived_from=dict(run=source['run_id'], column=CONTROL), models={}, official_rmse=None,
            baseline_id=source['run_id'], fits=dict(new_boosters=0, full_year=0, correction_windows=6),
            uploads=0, training_audits_parent=source['training_audits'], folds={},
            prediction_column=CONTROL)
        progress = dict(parameters={}, predictions={}, comparisons={}, summaries={})
        write_json(directory / 'progress.json', progress)
    started = time.monotonic()
    record['status'] = 'running'
    record.pop('error', None)
    write_json(directory / 'result.json', record)
    try:
        derive_predictions(source, parent, directory, record, progress)
        if 'changes' not in progress['summaries']:
            path = directory / 'change_summary.json'
            write_json(path, summarize_changes(directory, progress))
            progress['summaries']['changes'] = dict(path=path.name, artifacts={path.name: sha256(path)})
            write_json(directory / 'progress.json', progress)
        record['changes'] = json.loads((directory / progress['summaries']['changes']['path']).read_text())
        if 'report' not in record:
            record['report'] = reporting.report(record, directory)
            write_json(directory / 'result.json', record)
        record['candidate_scores'] = scorecard(record, directory, progress, args.sensitivity)
        record['comparisons'] = progress['comparisons']
        record['status'] = 'complete'
    except BaseException as exc:
        record['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['minutes'] = round(record.get('minutes', 0) + (time.monotonic() - started) / 60, 3)
        write_json(directory / 'result.json', record)
        if record['status'] == 'complete':
            append_log(record, directory.parent, replace_existing=True)
    print(f'Wrote {directory / "result.json"}; no boosters/refits/uploads', flush=True)


if __name__ == '__main__':
    main()
