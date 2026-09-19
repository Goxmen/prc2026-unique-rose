"""Reconstruct only the selected row-30/34 control when its local artifacts are absent.

Six original training windows, seeds 42/142/242, 800 ordinary-regressor rounds,
300 Rome-classifier rounds, and all sixteen evaluation folds. This uses row 30's
unchanged helpers and features, then applies row 33's plan window and row 34's
no-ID cap. It does not repeat the historical ablations or fit a full-year model.

Run: .venv/bin/python VALIDATION/artifact_control_run.py --output-dir <new run>
Resume with the same options plus --resume. --dry-run prints the fixed workload;
--smoke uses 1% of movements, five ordinary rounds and one seed, for execution only.
Historical aggregates are descriptive replay checks; original artifacts are absent
and changing machine or external-weather bytes can change numerical results.
"""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import glob
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import RANKING, TRAIN_GLOB, connect
import harness
import overnight_models
import overnight_run
import plan_window
from protocol import SCORE_VERSION, VERSION
import record_fixes
from records import RUNS, append_log, data_identity, runtime_identity, sha256, source_snapshot, write_json
import reporting
import rome_hybrid
from rome_hybrid_run import save_booster, v3_folds
import rome_signals as signals
import stand_day
import stand_day_run as row30
import weather

ROW30_RUN = RUNS / 'v3_20260918T151436598224Z'
ROW34_RUN = RUNS / 'v3_20260919T104157403241Z'
ORDINARY = row30.ORDINARIES['stand']
CLASSIFIER = row30.CLASSIFIERS['stand']
SOURCES = sorted(set(row30.SOURCES + [
    'VALIDATION/artifact_control_run.py', 'VALIDATION/plan_window.py',
    'VALIDATION/record_fixes.py', 'VALIDATION/protocol.py',
    'VALIDATION/records.py', 'VALIDATION/reporting.py', 'VALIDATION/harness.py', 'VALIDATION/seasonal.py',
]))
COLUMNS = {'row30': 'pred_stand', 'row33': 'pred_window', 'control': 'pred_noid_cap'}
RAW_AUDIT = [
    'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'FLIGHT_ID_mvt', 'STAND_mvt',
    'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
    'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'AOBT_3_flt', 'raw_aobt3_missing',
    'raw_plan_block_missing',
]
FEATURE_AUDIT = list(dict.fromkeys([
    *signals.SIGNALS, *stand_day.ALL_FEATURES, *overnight_models.CLOCK_FEATURES,
    *[f'mvt_minus_{name}_sec' for name in ('sched', 'lobt', 'iobt', 'eobt1', 'aobt3')],
    'q_rwy_prev_takeoff_gap_sec', 'q_rwy_next_takeoff_gap_sec',
    'q_queue_ahead_since_aobt3', 'wx_visibility_m', 'wx_wind_kt', 'wx_deicing',
]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label', default='recording-artifact-control-replay')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--prediction-threads', type=int, default=20)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.threads < 1 or args.prediction_threads < 1:
        parser.error('Thread counts must be positive')
    if args.resume and not args.output_dir:
        parser.error('--resume needs --output-dir')
    return args


def historical_recipe():
    """Keep original source guards; never rewrite or resume the historical run."""
    record = json.loads((ROW30_RUN / 'result.json').read_text())
    config = record['config']
    if (record['status'] != 'complete' or config['rounds'] != 800 or config['sample'] != 1
            or config['seeds'] != [42, 142, 242] or not config['late'] or config['dayshift']
            or config['ordinaries']['stand'] != list(ORDINARY)
            or config['classifiers']['stand'] != list(CLASSIFIER)):
        raise ValueError('Historical record is not the selected complete row-30 recipe')
    changed = [name for name, digest in record['signature'].items()
               if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f'Row-30 source changed: {changed}; use its frozen implementation')
    return record


def configuration(args, historical):
    config = copy.deepcopy(historical['config'])
    for name in ('control_source', 'reuse_control'):
        config.pop(name, None)
    config.update(label=args.label, threads=args.threads, prediction_threads=args.prediction_threads,
                  fit_workers=1, seeds=[42] if args.smoke else [42, 142, 242],
                  rounds=5 if args.smoke else 800, sample=0.01 if args.smoke else 1.0,
                  smoke=args.smoke, candidates=COLUMNS,
                  candidate_specs={'row30': dict(ordinary='stand', classifier='stand')},
                  ordinaries={'stand': list(ORDINARY)}, classifiers={'stand': list(CLASSIFIER)},
                  replay_of=ROW30_RUN.name, selected_control='row34',
                  plan_window_margin_sec=plan_window.MARGIN_SEC,
                  training_audit_selector='LIRF without original flight ID OR flight ID present and max(takeoff-LOBT,takeoff-AOBT3)>=3600s',
                  noid_cap={name: getattr(record_fixes, name) for name in (
                      'LATE_GAP_SEC', 'LATE_CAP_SEC', 'GAP_CAP_MIN_SEC', 'CAP_EXCLUDED', 'LATE_CAP_EXCLUDED')},
                  raw_audit=RAW_AUDIT, feature_audit=FEATURE_AUDIT)
    # Normalize tuples for exact comparison after JSON serialization on resume.
    return json.loads(json.dumps(config))


def source_signature():
    return {name: sha256(ROOT / name) for name in SOURCES}


def identity_changes(current, historical):
    old, new = ({item['file']: item for item in values} for values in (historical, current))
    return [name for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name)]


def check_resume(record, progress, directory, config, signature, identity):
    if record['signature'] != signature or record['config'] != config:
        raise ValueError('Cannot resume: source or configuration changed')
    if record['data_identity'] != identity:
        raise ValueError('Cannot resume: input data changed')
    if record.get('runtime_identity'):
        runtime = runtime_identity()
        if any(record['runtime_identity'][key] != runtime[key] for key in ('python', 'versions')):
            raise ValueError('Cannot resume: Python or numerical-library versions changed')
    overnight_run.check_artifacts(directory, progress)
    overnight_run.check_artifacts(directory, {'predictions': progress.get('training_audits', {})})


def model_key(job, part, seed):
    return f"{job['id']}__{part}_stand__{seed}"


def fit_window(directory, progress, job, X, y, raw, config):
    for seed in config['seeds']:
        for part in ('ordinary', 'classifier'):
            key = model_key(job, part, seed)
            if key in progress['models']:
                continue
            started = time.monotonic()
            if part == 'ordinary':
                booster = signals.fit_ordinary(X, y, raw, row30.KIND, seed,
                                               rounds=config['rounds'], threads=config['threads'], extra=ORDINARY)
            else:
                booster = signals.fit_classifier(X, y, raw, row30.KIND, seed,
                                                 threads=config['threads'], extra=CLASSIFIER)
            path = directory / 'models' / f'{key}.txt'
            save_booster(booster, path)
            progress['models'][key] = dict(path=str(path.relative_to(directory)),
                paths={part: path.name}, artifacts=overnight_run.artifact_hashes(directory, path),
                seconds=round(time.monotonic() - started, 3))
            write_json(directory / 'progress.json', progress)
            print(f"Fitted {key} in {progress['models'][key]['seconds']} s", flush=True)
            del booster


def load_models(directory, progress, job, seed, threads):
    return {part: lgb.Booster(model_file=str(directory / progress['models'][model_key(job, part, seed)]['path']),
                              params={'num_threads': threads}) for part in ('ordinary', 'classifier')}


def training_audit(raw, X, y):
    """Sparse correction-fitting rows; all labels belong to this training window."""
    if not raw.index.equals(X.index) or len(y) != len(raw):
        raise ValueError('Training audit raw/features/labels are misaligned')
    no_id = raw.FLIGHT_ID_mvt.isna().to_numpy()
    late_clock = ((X.mvt_minus_lobt_sec.to_numpy(float) >= 3600)
                  | (X.mvt_minus_aobt3_sec.to_numpy(float) >= 3600))
    selected = ((raw.ADEP_mvt.to_numpy() == signals.AIRPORT) & no_id) | (~no_id & late_clock)
    columns = ['mvt_id', *RAW_AUDIT]
    frame = raw.loc[selected, columns].reset_index(drop=True).copy()
    frame['y'] = np.asarray(y, float)[selected]
    frame['raw_flight_id_missing'] = no_id[selected]
    frame['gap_sec'] = X.loc[selected, rome_hybrid.GAP].to_numpy(float)
    audit = {column: X.loc[selected, column].to_numpy() for column in FEATURE_AUDIT}
    return pd.concat([frame, pd.DataFrame(audit, index=frame.index)], axis=1)


def save_training_audit(directory, progress, job, raw, X, y):
    if job['id'] in progress['training_audits']:
        return
    path = directory / f"training_audit_{job['id']}.parquet"
    frame = training_audit(raw, X, y)
    overnight_run.save_frame(path, frame)
    progress['training_audits'][job['id']] = dict(path=path.name, rows=len(frame),
        train_months=job['train'], artifacts=overnight_run.artifact_hashes(directory, path))
    write_json(directory / 'progress.json', progress)
    print(f"Saved {job['id']} correction-training audit: {len(frame):,} rows", flush=True)


def prediction_frame(raw, V, job, ev, seed, ordinary, classifier, parameters, config):
    out = signals.predict(ordinary, classifier, V, raw, row30.KIND,
                          threads=config['prediction_threads'], extra=ORDINARY,
                          classifier_extra=CLASSIFIER, late_mean=parameters['late_mean'], dayshift_weight=None)
    frame = overnight_run.prediction_metadata(raw, V, job, ev).assign(seed=seed)
    frame['pred_stand'] = out['final']
    frame['pred_window'] = plan_window.apply(out['final'], raw)
    frame['pred_noid_cap'] = record_fixes.noid_cap(frame.pred_window.to_numpy(), raw)
    for column in ('ordinary', 'p', 'blended', 'rule_triggered', 'late_rows'):
        frame[column] = out[column]
    frame['gap_sec'] = V[rome_hybrid.GAP].to_numpy(float)
    audit = {column: raw[column].to_numpy() for column in RAW_AUDIT}
    audit.update({column: V[column].to_numpy() for column in FEATURE_AUDIT})
    frame = pd.concat([frame, pd.DataFrame(audit, index=frame.index)], axis=1)
    if not np.isfinite(frame[list(COLUMNS.values())].to_numpy(float)).all():
        raise ValueError('Nonfinite replay prediction')
    rule = frame.rule_triggered.to_numpy(bool)
    if not np.array_equal(frame.pred_stand.to_numpy()[rule], frame.pred_noid_cap.to_numpy()[rule]):
        raise ValueError('Postprocessing changed a Rome-rule row')
    return frame


def score_window(directory, progress, job, td, ta, validation, cats, config, folds, reports):
    for ev in job['evals']:
        name = ev['fold']
        if name in progress['folds']:
            continue
        raw, V = row30.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
        files = {}
        for seed in config['seeds']:
            key = f'{name}__{seed}'
            if key in progress['predictions']:
                files[str(seed)] = progress['predictions'][key]['path']
                continue
            boosters = load_models(directory, progress, job, seed, config['prediction_threads'])
            frame = prediction_frame(raw, V, job, ev, seed, boosters['ordinary'], boosters['classifier'],
                                     progress['parameters'][job['id']], config)
            path = directory / f'oof_{name}_seed{seed}.parquet'
            overnight_run.save_frame(path, frame)
            progress['predictions'][key] = dict(path=path.name, rows=len(frame),
                                                 artifacts=overnight_run.artifact_hashes(directory, path))
            files[str(seed)] = path.name
            write_json(directory / 'progress.json', progress)
            del boosters, frame
        fold, start = folds[name]
        progress['folds'][name] = {**asdict(fold), 'train_start': start,
                                  'model_id': job['id'], 'predictions': files}
        write_json(directory / 'progress.json', progress)
        print(f"Scored {name} ({ev['month']}): {len(raw):,} departures, all seeds", flush=True)
        del raw, V
        gc.collect()


def aggregate_scores(record, directory, column):
    candidate = dict(record, prediction_column=column)
    losses = reporting.load_losses(candidate, directory)
    scores = {name: reporting.rmse(loss) for name, loss in reporting.score_groups(losses).items()}
    weight = record['ranking_weights']['jan_fraction']
    scores['R_combined'] = float(np.sqrt(weight * scores['R_A'] ** 2 + (1 - weight) * scores['R_B_long'] ** 2))
    return {name: scores[name] for name in ('R_A', 'R_B_long', 'R_combined')}


def replay_checks(record, directory):
    references = {
        'row30': (ROW30_RUN / f'comparison_pred_stand_vs_pred_{ROW30_RUN.name}_block1.json', 'pred_stand'),
        'row34': (ROW34_RUN / f'comparison_pred_noid_cap_vs_pred_window_{ROW34_RUN.name}_block1.json', 'pred_noid_cap'),
    }
    checks = {}
    for name, (path, column) in references.items():
        original = json.loads(path.read_text())['report']['scores']
        current = aggregate_scores(record, directory, column)
        checks[name] = dict(reference=str(path.relative_to(ROOT)), column=column,
                            original={key: original[key] for key in current}, replay=current,
                            difference={key: current[key] - original[key] for key in current})
    return dict(interpretation='Descriptive aggregate comparisons, not identical-prediction verification.',
                rows=checks)


def main(argv=None):
    args = parse_args(argv)
    historical = historical_recipe()
    config = configuration(args, historical)
    jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
    print(f"Control replay: {len(jobs)} windows, {len(config['seeds'])} seeds, "
          f"{2 * len(jobs) * len(config['seeds'])} boosters; 16 folds, no full-year fit", flush=True)
    if args.dry_run:
        print(json.dumps(dict(config=config, weather_exists=weather.METAR.exists()), indent=2))
        return
    if not weather.METAR.is_file():
        raise FileNotFoundError(f'Missing {weather.METAR}; fetch METAR before the replay')
    signature = source_signature()
    identity = data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR])
    changed_data = identity_changes(identity, historical['data_identity'])
    changed_movements = [name for name in changed_data if name != weather.METAR.name]
    if changed_movements:
        raise ValueError(f'Historical movement data changed: {changed_movements}')
    directory = (args.output_dir or RUNS / datetime.now(timezone.utc).strftime('v3_%Y%m%dT%H%M%S%fZ')).resolve()
    if args.resume:
        record = json.loads((directory / 'result.json').read_text())
        progress = json.loads((directory / 'progress.json').read_text())
        check_resume(record, progress, directory, config, signature, identity)
        if record['status'] in ('complete', 'execution_check_only'):
            print(f'Already complete: {directory}', flush=True)
            return
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'models').mkdir()
        con = connect()
        try:
            ranking_weights = harness.ranking_weights(con)
        finally:
            con.close()
        record = dict(run_id=directory.name, protocol=VERSION, score_version=SCORE_VERSION,
                      label=args.label, at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                      status='running', full_benchmark=not args.smoke, config=config,
                      prediction_column='pred_noid_cap', baseline_id=None, official_rmse=None,
                      signature=signature, data_identity=identity, runtime_identity=runtime_identity(),
                      ranking_weights=ranking_weights, source=source_snapshot(directory),
                      reconstruction=dict(reason='Historical row-30/34 boosters and OOF predictions absent on this host.',
                          row30_run=ROW30_RUN.name, row34_run=ROW34_RUN.name,
                          historical_source_signature=historical['signature'],
                          changed_data=changed_data))
        progress = dict(models={}, predictions={}, folds={}, parameters={}, input_coverage={}, training_audits={})
        write_json(directory / 'result.json', record)
        write_json(directory / 'progress.json', progress)
    folds = v3_folds()
    started = time.monotonic()
    record['status'] = 'running'
    record.pop('error', None)
    write_json(directory / 'result.json', record)
    try:
        reports = weather.load_reports()
        dep, arr = overnight_run.load_inputs(config['sample'])
        for job in jobs:
            if all(ev['fold'] in progress['folds'] for ev in job['evals']):
                print(f"Reusing completed {job['id']}", flush=True)
                continue
            td, ta, validation = overnight_run.split_job(dep, arr, job)
            print(f"Preparing {job['id']}: {len(td):,} training departures", flush=True)
            X, y, cats, raw = overnight_run.prepare_training(td, ta, config)
            parameters = signals.fit_parameters(X, y, raw)
            old = progress['parameters'].get(job['id'])
            if old is not None and old != parameters:
                raise ValueError(f"{job['id']}: training correction parameters changed on resume")
            progress['parameters'][job['id']] = parameters
            X = row30.add_inputs(X, raw, reports, td, ta)
            progress['input_coverage'][job['id']] = row30.coverage(X, raw)
            save_training_audit(directory, progress, job, raw, X, y)
            write_json(directory / 'progress.json', progress)
            fit_window(directory, progress, job, X, y, raw, config)
            del X, y, raw
            gc.collect()
            score_window(directory, progress, job, td, ta, validation, cats, config, folds, reports)
            del td, ta, validation, cats
            gc.collect()
        del dep, arr, reports
        record.update(folds=progress['folds'], models=progress['models'], parameters=progress['parameters'],
                      input_coverage=progress['input_coverage'], training_audits=progress['training_audits'])
        record['report'] = reporting.report(record, directory)
        if not args.smoke:
            record['replay_checks'] = replay_checks(record, directory)
            print(json.dumps(record['replay_checks'], indent=2), flush=True)
        record['status'] = 'execution_check_only' if args.smoke else 'complete'
        reporting.print_report(record['report'], slices=False)
    except BaseException as exc:
        record['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['minutes'] = round(record.get('minutes', 0) + (time.monotonic() - started) / 60, 3)
        write_json(directory / 'result.json', record)
        if record['status'] in ('complete', 'execution_check_only'):
            append_log(record, directory.parent, replace_existing=True)
    print(f'Wrote {directory / "result.json"}', flush=True)


if __name__ == '__main__':
    main()
