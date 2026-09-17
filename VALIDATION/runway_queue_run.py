"""Protocol-v3 validation of runway queue inputs on the weather Rome hybrid (row 27).

The control is row 27, Rome-hybrid-4h-clock-weather. Its boosters are not refitted:
they are loaded from the completed weather run and must reproduce that run's saved
`pred_weather` exactly on every fold and seed, which also proves this run builds
the same validation inputs. Two candidates add the queue inputs of
`runway_queue.py` to the ordinary regressor (the Rome classifier is unchanged):

    pred              row 27 control, reloaded
    pred_queue_past   + PAST_FEATURES   (movements before takeoff only)
    pred_queue_all    + ALL_FEATURES    (also movements after takeoff)

Queue inputs are saved beside the predictions for slice analysis. The result
record follows harness.py, so `compare.py --prediction-column` works unchanged.
No full-year refits or uploads happen here.

Run:    .venv/bin/python VALIDATION/runway_queue_run.py --label runway-queue-weather
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
import runway_queue
import weather

KIND = 'clock'
CONTROL_RUN = RUNS / 'v3_20260917T155106873031Z'
CONTROL_MODEL = 'rome_hybrid_4h_clock_weather'
CONTROL_SAVED_COLUMN = 'pred_weather'
WEATHER = tuple(weather.WEATHER_FEATURES)
CANDIDATES = {
    'control': WEATHER,
    'queue_past': WEATHER + tuple(runway_queue.PAST_FEATURES),
    'queue_all': WEATHER + tuple(runway_queue.ALL_FEATURES),
}
COLUMNS = {'control': 'pred', 'queue_past': 'pred_queue_past', 'queue_all': 'pred_queue_all'}
SOURCES = ['VALIDATION/runway_queue.py', 'VALIDATION/runway_queue_run.py', 'VALIDATION/weather.py',
           'VALIDATION/weather_run.py', 'VALIDATION/rome_hybrid.py', 'VALIDATION/rome_hybrid_run.py',
           'VALIDATION/overnight_models.py', 'VALIDATION/overnight_run.py', 'VALIDATION/modeling.py',
           'VALIDATION/features_time.py', 'VALIDATION/movement_data.py', 'pipeline/common.py',
           'pipeline/clock_features.py', 'pipeline/registered_rules.py', 'pipeline/prediction_rules.py']


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='runway-queue-weather')
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
    """Completed weather run whose row-27 boosters serve as the control."""
    record = json.loads((CONTROL_RUN / 'result.json').read_text())
    progress = json.loads((CONTROL_RUN / 'progress.json').read_text())
    if record['status'] != 'complete' or record['config']['candidates'].get(CONTROL_MODEL) != CONTROL_SAVED_COLUMN:
        raise ValueError(f'{CONTROL_RUN.name} is not a completed run with {CONTROL_MODEL}')
    changed = sorted(name for name, digest in record['signature'].items()
                     if (ROOT / name).is_file() and sha256(ROOT / name) != digest)
    return record, progress, changed


def fitted_names(config):
    return [name for name in CANDIDATES if name != 'control' or not config['reuse_control']]


def boosters(directory, progress, control_progress, job, name, seed, config):
    if name == 'control' and config['reuse_control']:
        entry = control_progress['models'][f"{job['id']}__{CONTROL_MODEL}__{seed}"]
        return load(CONTROL_RUN, entry, 'ordinary'), load(CONTROL_RUN, entry, 'classifier')
    entry = progress['models'][f"{job['id']}__{name}__{seed}"]
    return load(directory, entry, 'ordinary'), load(directory, entry, 'classifier')


def add_inputs(X, raw, reports, dep_events, arr_events):
    X = weather.add_weather(X, raw, reports)
    return runway_queue.add_queue(X, raw, modeling.observable(dep_events), modeling.observable(arr_events))


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
        raise ValueError(f"{ev['fold']}/{seed}: reloaded control does not reproduce row 27 (max diff {worst})")


def predict_frame(directory, progress, control_progress, job, ev, vd, V, seed, config):
    frame = overnight_run.prediction_metadata(vd, V, job, ev).assign(seed=seed)
    for name, extra in CANDIDATES.items():
        ordinary, classifier = boosters(directory, progress, control_progress, job, name, seed, config)
        out = rome_hybrid.predict(ordinary, classifier, V, vd, KIND, threads=config['threads'], extra=extra)
        frame[COLUMNS[name]] = out['final']
        frame[f'ordinary_{COLUMNS[name]}'] = out['ordinary']
        frame[f'p_{COLUMNS[name]}'] = out['p']
        frame[f'rule_{COLUMNS[name]}'] = out['rule_triggered']
    if config['reuse_control']:
        check_control(frame, ev, seed, control_progress)
    frame['gap_sec'] = V[rome_hybrid.GAP].to_numpy(float)
    frame['raw_aobt3_missing'] = vd['AOBT_3_flt'].isna().to_numpy()
    for column in runway_queue.ALL_FEATURES:
        frame[column] = V[column].to_numpy()
    return frame


def fit_window(directory, progress, job, X, y, raw_kept, config):
    for seed in config['seeds']:
        for name in fitted_names(config):
            key = f"{job['id']}__{name}__{seed}"
            if key in progress['models']:
                continue
            started = time.monotonic()
            ordinary, classifier, details = rome_hybrid.fit_components(
                X, y, raw_kept, KIND, seed, rounds=config['rounds'], threads=config['threads'],
                extra=CANDIDATES[name])
            paths = {}
            for part, booster in [('ordinary', ordinary), ('classifier', classifier)]:
                path = directory / 'models' / f'{key}.{part}.txt'
                save_booster(booster, path)
                paths[part] = path.name
            progress['models'][key] = dict(paths=paths, seconds=round(time.monotonic() - started, 1), **details)
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
              f"{'; control reproduces row 27' if config['reuse_control'] else ''}.", flush=True)
        del V, vd
        gc.collect()


def coverage(X):
    return {name: round(float(X[name].notna().mean()), 5) for name in runway_queue.ALL_FEATURES}


def main(argv=None):
    args = parse_args(argv)
    seeds = [42] if args.smoke else [42, 142, 242]
    config = dict(models.BASE_CONFIG, seeds=seeds, rounds=5 if args.smoke else 800,
                  sample=0.01 if args.smoke else 1.0, threads=args.threads,
                  prediction_threads=args.threads, train_months=6, bootstrap_reps=2000,
                  bootstrap_seed=0, block_days=1, label=args.label, smoke=args.smoke,
                  rome_hybrid=rome_hybrid.SETTINGS, weather=weather.SETTINGS, queue=runway_queue.SETTINGS,
                  candidates=COLUMNS, reuse_control=not args.smoke,
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
        progress = dict(models={}, folds={}, queue_coverage={})
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
                # Fail before any fitting if the reloaded control cannot reproduce row 27.
                ev = job['evals'][0]
                vd, V = validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                frame = overnight_run.prediction_metadata(vd, V, job, ev)
                ordinary, classifier = boosters(directory, progress, control_progress, job, 'control', seeds[0], config)
                frame['pred'] = rome_hybrid.predict(ordinary, classifier, V, vd, KIND, threads=config['threads'],
                                                    extra=CANDIDATES['control'])['final']
                check_control(frame, ev, seeds[0], control_progress)
                print(f"Control check passed on {ev['fold']}/{seeds[0]}: row 27 reproduced exactly.", flush=True)
                del vd, V, frame
                gc.collect()
            # Training rows see only their own window's movements; every supplied departure counts (rule 6).
            X = add_inputs(X, raw_kept, reports, td, ta)
            progress['queue_coverage'][job['id']] = coverage(X)
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
        record['queue_coverage'] = progress['queue_coverage']
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
