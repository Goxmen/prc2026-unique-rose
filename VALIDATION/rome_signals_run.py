"""Protocol-v3 A/B validation of the Rome recording signals on row 28.

The control is row 28, Rome-hybrid-4h-clock-weather-queue. Its boosters are not
refitted: they are loaded from the completed runway-queue run and must reproduce
that run's saved `pred_queue_all` exactly on every fold and seed. Each candidate
changes one thing relative to the control, so every paired comparison against
`pred` measures exactly one change (see `rome_signals.py` for the definitions):

    pred               row 28 control, reloaded
    pred_late          late-row estimate for LIRF no-ID rows at 1-4 h     (no refit)
    pred_cls_airline   Rome classifier + airline                          (classifier refit)
    pred_cls_handler   Rome classifier + arrival handler share and count  (classifier refit)
    pred_cls_format    Rome classifier + flight-number format             (classifier refit)
    pred_dayshift      day-shift blend on next-day rule rows from 14 h    (no refit)
    pred_reg_airline   ordinary regressor + airline, at every airport     (regressor refit)

Two combinations show whether the single changes add up:

    pred_cls_all       Rome classifier + all three classifier inputs
    pred_all           reg_airline + cls_all + late + dayshift

The late mean and day-shift weight are fitted on each window's training labels.
Model audit columns (ordinary predictions, probabilities, rule and correction
flags, the signals themselves) are saved beside the predictions. The result
record follows harness.py, so `compare.py --prediction-column` works unchanged.
No full-year refits or uploads happen here.

Run:    .venv/bin/python VALIDATION/rome_signals_run.py --label rome-signals
Smoke:  add --smoke (1% sample, five rounds, seed 42, control refitted; execution check only)
Resume: add --resume --output-dir <run directory>
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import RANKING, TRAIN_GLOB, connect
import harness
import modeling
import overnight_models as models
import overnight_run
from protocol import SCORE_VERSION, VERSION
from records import RUNS, append_log, data_identity, sha256, source_snapshot, write_json
from reporting import print_report, report
import rome_hybrid
from rome_hybrid_run import load, save_booster, v3_folds
import rome_signals as signals
import runway_queue
import runway_queue_run as queue_run
import weather

KIND = 'clock'
CONTROL_RUN = RUNS / 'v3_20260917T172934966809Z'
CONTROL_MODEL = 'queue_all'
CONTROL_SAVED_COLUMN = 'pred_queue_all'
REGRESSOR_INPUTS = tuple(queue_run.CANDIDATES[CONTROL_MODEL])
ORDINARIES = {'control': REGRESSOR_INPUTS, 'airline': REGRESSOR_INPUTS + (signals.AIRLINE,)}
CLASSIFIERS = {'control': (), 'airline': (signals.AIRLINE,), 'handler': tuple(signals.HANDLER),
               'format': (signals.FORMAT,), 'all': tuple(signals.SIGNALS)}
CANDIDATES = {
    'control': dict(ordinary='control', classifier='control', late=False, dayshift=False),
    'late': dict(ordinary='control', classifier='control', late=True, dayshift=False),
    'cls_airline': dict(ordinary='control', classifier='airline', late=False, dayshift=False),
    'cls_handler': dict(ordinary='control', classifier='handler', late=False, dayshift=False),
    'cls_format': dict(ordinary='control', classifier='format', late=False, dayshift=False),
    'dayshift': dict(ordinary='control', classifier='control', late=False, dayshift=True),
    'reg_airline': dict(ordinary='airline', classifier='control', late=False, dayshift=False),
    'cls_all': dict(ordinary='control', classifier='all', late=False, dayshift=False),
    'all': dict(ordinary='airline', classifier='all', late=True, dayshift=True),
}
COLUMNS = {name: 'pred' if name == 'control' else f'pred_{name}' for name in CANDIDATES}
SOURCES = ['VALIDATION/rome_signals.py', 'VALIDATION/rome_signals_run.py', *queue_run.SOURCES]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='rome-signals')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--output-dir', type=Path, help='run directory (default: new VALIDATION/runs/v3_<time>)')
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error('--threads must be positive')
    if args.resume and not args.output_dir:
        ap.error('--resume needs --output-dir')
    return args


def control_source():
    """Completed runway-queue run whose row-28 boosters serve as the control."""
    record = json.loads((CONTROL_RUN / 'result.json').read_text())
    progress = json.loads((CONTROL_RUN / 'progress.json').read_text())
    if record['status'] != 'complete' or record['config']['candidates'].get(CONTROL_MODEL) != CONTROL_SAVED_COLUMN:
        raise ValueError(f'{CONTROL_RUN.name} is not a completed run with {CONTROL_MODEL}')
    changed = sorted(name for name, digest in record['signature'].items()
                     if (ROOT / name).is_file() and sha256(ROOT / name) != digest)
    return record, progress, changed


def fitted(part, config):
    names = ORDINARIES if part == 'ordinary' else CLASSIFIERS
    return [name for name in names if name != 'control' or not config['reuse_control']]


def booster(directory, progress, control_progress, job, part, name, seed, config):
    if name == 'control' and config['reuse_control']:
        return load(CONTROL_RUN, control_progress['models'][f"{job['id']}__{CONTROL_MODEL}__{seed}"], part)
    return load(directory, progress['models'][f"{job['id']}__{part}_{name}__{seed}"], part)


def add_inputs(X, raw, reports, dep_events, arr_events):
    X = queue_run.add_inputs(X, raw, reports, dep_events, arr_events)
    return signals.add_signals(X, raw, arr_events)


def validation_inputs(job, ev, td, ta, validation, cats, config, reports):
    vd, va = validation[ev['month']]
    V = modeling.validation_matrix(td, ta, vd, va, job['freeze'], cats, config)
    # Current context for a validation month: its training window plus that month only (protocol rule 5).
    V = add_inputs(V, vd, reports, pd.concat([td, vd], ignore_index=True), pd.concat([ta, va], ignore_index=True))
    return vd, V


def check_control(frame, ev, seed, control_progress):
    saved = pd.read_parquet(CONTROL_RUN / control_progress['folds'][ev['fold']]['predictions'][str(seed)],
                            columns=['mvt_id', CONTROL_SAVED_COLUMN])
    if not np.array_equal(saved.mvt_id.to_numpy(), frame.mvt_id.to_numpy()):
        raise ValueError(f"{ev['fold']}/{seed}: validation rows differ from the control run")
    if not np.array_equal(saved[CONTROL_SAVED_COLUMN].to_numpy(), frame['pred'].to_numpy()):
        worst = np.nanmax(np.abs(saved[CONTROL_SAVED_COLUMN].to_numpy() - frame['pred'].to_numpy()))
        raise ValueError(f"{ev['fold']}/{seed}: reloaded control does not reproduce row 28 (max diff {worst})")


def control_prediction(directory, progress, control_progress, job, vd, V, seed, config):
    ordinary = booster(directory, progress, control_progress, job, 'ordinary', 'control', seed, config)
    classifier = booster(directory, progress, control_progress, job, 'classifier', 'control', seed, config)
    return signals.predict(ordinary, classifier, V, vd, KIND, threads=config['threads'],
                           extra=ORDINARIES['control'])['final']


def predict_frame(directory, progress, control_progress, job, ev, vd, V, seed, config):
    frame = overnight_run.prediction_metadata(vd, V, job, ev).assign(seed=seed)
    threads, parameters = config['threads'], progress['parameters'][job['id']]
    base = {name: signals.ordinary_prediction(
                booster(directory, progress, control_progress, job, 'ordinary', name, seed, config),
                V, KIND, threads, extra)
            for name, extra in ORDINARIES.items()}
    p = {name: signals.substitution_probability(
             booster(directory, progress, control_progress, job, 'classifier', name, seed, config),
             V, vd, KIND, threads, extra)
         for name, extra in CLASSIFIERS.items()}
    for name, spec in CANDIDATES.items():
        out = signals.combine(base[spec['ordinary']], p[spec['classifier']], V, vd,
                              late_mean=parameters['late_mean'] if spec['late'] else None,
                              dayshift_weight=parameters['dayshift_weight'] if spec['dayshift'] else None)
        frame[COLUMNS[name]] = out['final']
    if config['reuse_control']:
        check_control(frame, ev, seed, control_progress)
    for name in ORDINARIES:
        frame[f'ordinary_{name}'] = base[name]
    for name in CLASSIFIERS:
        frame[f'p_{name}'] = p[name]
    # `all` switches both corrections on, so its flags mark every row a correction touched.
    for key in ['rule_triggered', 'late_rows', 'dayshift_rows']:
        frame[key] = out[key]
    frame['gap_sec'] = V[rome_hybrid.GAP].to_numpy(float)
    frame['raw_aobt3_missing'] = vd['AOBT_3_flt'].isna().to_numpy()
    frame[signals.AIRLINE] = V[signals.AIRLINE].astype(object).to_numpy()
    for column in [*signals.HANDLER, signals.FORMAT]:
        frame[column] = V[column].to_numpy()
    return frame


def fit_window(directory, progress, job, X, y, raw_kept, config):
    progress['parameters'][job['id']] = signals.fit_parameters(X, y, raw_kept)
    write_json(directory / 'progress.json', progress)
    print(f"Fitted corrections for {job['id']}: {progress['parameters'][job['id']]}", flush=True)
    for seed in config['seeds']:
        for part in ['ordinary', 'classifier']:
            for name in fitted(part, config):
                key = f"{job['id']}__{part}_{name}__{seed}"
                if key in progress['models']:
                    continue
                started = time.monotonic()
                if part == 'ordinary':
                    model = signals.fit_ordinary(X, y, raw_kept, KIND, seed, rounds=config['rounds'],
                                                 threads=config['threads'], extra=ORDINARIES[name])
                else:
                    model = signals.fit_classifier(X, y, raw_kept, KIND, seed, threads=config['threads'],
                                                   extra=CLASSIFIERS[name])
                path = directory / 'models' / f'{key}.txt'
                save_booster(model, path)
                progress['models'][key] = dict(paths={part: path.name}, seconds=round(time.monotonic() - started, 1))
                write_json(directory / 'progress.json', progress)
                print(f'Fitted {key} in {progress["models"][key]["seconds"]} s', flush=True)


def score_window(directory, progress, control_progress, job, td, ta, validation, cats, config, folds, reports):
    for ev in job['evals']:
        if ev['fold'] in progress['folds']:
            continue
        vd, V = validation_inputs(job, ev, td, ta, validation, cats, config, reports)
        files = {}
        for seed in config['seeds']:
            frame = predict_frame(directory, progress, control_progress, job, ev, vd, V, seed, config)
            file = f"oof_{ev['fold']}_seed{seed}.parquet"
            frame.to_parquet(directory / file, index=False)
            files[str(seed)] = file
        fold, start = folds[ev['fold']]
        progress['folds'][ev['fold']] = {**asdict(fold), 'train_start': start, 'model_id': job['id'],
                                         'predictions': files}
        write_json(directory / 'progress.json', progress)
        print(f"Scored {ev['fold']} ({ev['month']}) on {len(vd):,} flights"
              f"{'; control reproduces row 28' if config['reuse_control'] else ''}.", flush=True)
        del V, vd
        gc.collect()


def coverage(X, raw):
    lirf_no_id = (raw['ADEP_mvt'].to_numpy() == signals.AIRPORT) & raw['FLIGHT_ID_mvt'].isna().to_numpy()
    return dict(airline=round(float(X[signals.AIRLINE].notna().mean()), 5),
                handler=round(float(X[signals.HANDLER[0]].notna().mean()), 5),
                lirf_no_id_rows=int(lirf_no_id.sum()),
                lirf_no_id_airline=round(float(X.loc[lirf_no_id, signals.AIRLINE].notna().mean()), 5)
                if lirf_no_id.any() else None)


def main(argv=None):
    args = parse_args(argv)
    seeds = [42] if args.smoke else [42, 142, 242]
    config = dict(models.BASE_CONFIG, seeds=seeds, rounds=5 if args.smoke else 800,
                  sample=0.01 if args.smoke else 1.0, threads=args.threads,
                  prediction_threads=args.threads, train_months=6, bootstrap_reps=2000,
                  bootstrap_seed=0, block_days=1, label=args.label, smoke=args.smoke,
                  rome_hybrid=rome_hybrid.SETTINGS, weather=weather.SETTINGS, queue=runway_queue.SETTINGS,
                  rome_signals=signals.SETTINGS, candidates=COLUMNS, candidate_specs=CANDIDATES,
                  ordinaries={k: list(v) for k, v in ORDINARIES.items()},
                  classifiers={k: list(v) for k, v in CLASSIFIERS.items()},
                  reuse_control=not args.smoke,
                  control_source=dict(run=CONTROL_RUN.name, model=CONTROL_MODEL, column=CONTROL_SAVED_COLUMN))
    config.pop('recipes', None)
    signature = {name: sha256(ROOT / name) for name in SOURCES}
    control_record, control_progress, changed = control_source()
    if config['reuse_control']:
        if control_record['config']['rounds'] != config['rounds'] or control_record['config']['seeds'] != seeds:
            raise ValueError('Control run rounds/seeds differ from this run')
        if changed:
            print(f'Note: files changed since the control run: {changed}. '
                  'Exact control reproduction is checked on every fold and seed.', flush=True)
    if args.resume:
        directory = args.output_dir.resolve()
        record = json.loads((directory / 'result.json').read_text())
        if record['signature'] != signature or record['config'] != json.loads(json.dumps(config)):
            raise ValueError('Cannot resume: source or configuration changed')
        progress = json.loads((directory / 'progress.json').read_text())
    else:
        run_id = datetime.now(timezone.utc).strftime('v3_%Y%m%dT%H%M%S%fZ')
        directory = (args.output_dir or RUNS / run_id).resolve()
        directory.mkdir(parents=True)
        (directory / 'models').mkdir()
        record = dict(run_id=directory.name, protocol=VERSION, score_version=SCORE_VERSION,
                      label=('smoke:' if args.smoke else '') + args.label,
                      at=datetime.now(timezone.utc).isoformat(timespec='seconds'), config=config,
                      signature=signature, baseline_id=None, official_rmse=None, status='running',
                      full_benchmark=not args.smoke)
        record['source'] = source_snapshot(directory)
        record['data_identity'] = data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR])
        if config['reuse_control'] and record['data_identity'] != control_record['data_identity']:
            raise ValueError('Data files differ from the control run')
        con = connect()
        record['ranking_weights'] = harness.ranking_weights(con)
        con.close()
        progress = dict(models={}, folds={}, parameters={}, signal_coverage={})
        write_json(directory / 'result.json', record)
        write_json(directory / 'progress.json', progress)
    folds = v3_folds()
    jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
    started = time.monotonic()
    try:
        reports = weather.load_reports()
        dep, arr = overnight_run.load_inputs(config['sample'])
        for number, job in enumerate(jobs):
            if all(ev['fold'] in progress['folds'] for ev in job['evals']):
                print(f"Reusing completed {job['id']}", flush=True)
                continue
            td, ta, validation = overnight_run.split_job(dep, arr, job)
            print(f"Preparing {job['id']}: {len(td):,} training departures", flush=True)
            X, y, cats, raw_kept = overnight_run.prepare_training(td, ta, config)
            if config['reuse_control'] and number == 0 and not progress['folds']:
                # Fail before any fitting if the reloaded control cannot reproduce row 28.
                ev = job['evals'][0]
                vd, V = validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                frame = overnight_run.prediction_metadata(vd, V, job, ev)
                frame['pred'] = control_prediction(directory, progress, control_progress, job, vd, V, seeds[0], config)
                check_control(frame, ev, seeds[0], control_progress)
                print(f"Control check passed on {ev['fold']}/{seeds[0]}: row 28 reproduced exactly.", flush=True)
                del vd, V, frame
                gc.collect()
            # Training rows see only their own window's movements; every supplied departure counts (rule 6).
            X = add_inputs(X, raw_kept, reports, td, ta)
            progress['signal_coverage'][job['id']] = coverage(X, raw_kept)
            write_json(directory / 'progress.json', progress)
            fit_window(directory, progress, job, X, y, raw_kept, config)
            del X, y, raw_kept
            gc.collect()
            score_window(directory, progress, control_progress, job, td, ta, validation, cats, config, folds, reports)
            del td, ta, validation, cats
            gc.collect()
        del dep, arr
        record['folds'] = progress['folds']
        record['models'] = progress['models']
        record['parameters'] = progress['parameters']
        record['signal_coverage'] = progress['signal_coverage']
        record['report'] = report(record, directory)
        print_report(record['report'], slices=False)
        record['status'] = 'execution_check_only' if args.smoke else 'complete'
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
