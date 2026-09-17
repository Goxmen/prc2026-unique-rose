"""Protocol-v3 validation of METAR weather inputs on the clock-input Rome hybrid.

One run fits, per six-month window and seed, the clock-input Rome hybrid twice:
unchanged, and with the weather inputs of `weather.py` added to its ordinary
regressor (the Rome classifier is identical in both). Every prediction file
holds both candidates, so the comparison is paired on folds, seeds and flights:

    pred           clock-input Rome hybrid (control)
    pred_weather   the same hybrid with weather inputs

Selected weather inputs are saved beside the predictions for slice analysis.
The result record follows harness.py, so `compare.py --prediction-column`
works unchanged. No full-year refits or uploads happen here.

Needs:  .venv/bin/python VALIDATION/tools/fetch_metar.py
Run:    .venv/bin/python VALIDATION/weather_run.py --label weather-hybrid-clock
Smoke:  add --smoke (1% sample, five rounds, seed 42; execution check only)
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
import weather

KIND = 'clock'
HYBRIDS = {'rome_hybrid_4h_clock': (), 'rome_hybrid_4h_clock_weather': tuple(weather.WEATHER_FEATURES)}
COLUMNS = {'rome_hybrid_4h_clock': 'pred', 'rome_hybrid_4h_clock_weather': 'pred_weather'}
SAVED_WEATHER = ['wx_visibility_m', 'wx_temp_c', 'wx_gust_kt', 'wx_thunder', 'wx_snow',
                 'wx_freezing', 'wx_fog', 'wx_deicing', 'wx_2h_thunder', 'wx_age_min']
SOURCES = ['VALIDATION/weather.py', 'VALIDATION/weather_run.py', 'VALIDATION/rome_hybrid.py',
           'VALIDATION/rome_hybrid_run.py', 'VALIDATION/overnight_models.py', 'VALIDATION/overnight_run.py',
           'VALIDATION/modeling.py', 'VALIDATION/features_time.py', 'VALIDATION/movement_data.py',
           'pipeline/common.py', 'pipeline/clock_features.py', 'pipeline/registered_rules.py',
           'pipeline/prediction_rules.py']


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='weather-hybrid-clock')
    ap.add_argument('--seeds', type=int, choices=[1, 3], default=3)
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


def fit_window(directory, progress, job, X, y, raw_kept, config):
    for seed in config['seeds']:
        for name, extra in HYBRIDS.items():
            key = f"{job['id']}__{name}__{seed}"
            if key in progress['models']:
                continue
            started = time.monotonic()
            ordinary, classifier, details = rome_hybrid.fit_components(
                X, y, raw_kept, KIND, seed, rounds=config['rounds'], threads=config['threads'], extra=extra)
            paths = {}
            for part, booster in [('ordinary', ordinary), ('classifier', classifier)]:
                path = directory / 'models' / f'{key}.{part}.txt'
                save_booster(booster, path)
                paths[part] = path.name
            progress['models'][key] = dict(paths=paths, seconds=round(time.monotonic() - started, 1), **details)
            write_json(directory / 'progress.json', progress)
            print(f'Fitted {key} in {progress["models"][key]["seconds"]} s', flush=True)


def score_window(directory, progress, job, td, ta, validation, cats, config, folds, reports):
    for ev in job['evals']:
        if ev['fold'] in progress['folds']:
            continue
        vd, va = validation[ev['month']]
        V = modeling.validation_matrix(td, ta, vd, va, job['freeze'], cats, config)
        V = weather.add_weather(V, vd, reports)
        meta = overnight_run.prediction_metadata(vd, V, job, ev)
        files = {}
        for seed in config['seeds']:
            frame = meta.assign(seed=seed)
            for name, extra in HYBRIDS.items():
                entry = progress['models'][f"{job['id']}__{name}__{seed}"]
                out = rome_hybrid.predict(load(directory, entry, 'ordinary'), load(directory, entry, 'classifier'),
                                          V, vd, KIND, threads=config['threads'], extra=extra)
                frame[COLUMNS[name]] = out['final']
                frame[f'ordinary_{COLUMNS[name]}'] = out['ordinary']
                frame['p_substituted'] = out['p']
                frame['rule_triggered'] = out['rule_triggered']
            frame['gap_sec'] = V[rome_hybrid.GAP].to_numpy(float)
            for column in SAVED_WEATHER:
                frame[column] = V[column].to_numpy()
            file = f"oof_{ev['fold']}_seed{seed}.parquet"
            frame.to_parquet(directory / file, index=False)
            files[str(seed)] = file
        fold, start = folds[ev['fold']]
        progress['folds'][ev['fold']] = {**asdict(fold), 'train_start': start, 'model_id': job['id'],
                                         'predictions': files}
        write_json(directory / 'progress.json', progress)
        print(f"Scored {ev['fold']} ({ev['month']}) on {len(meta):,} flights.", flush=True)
        del V, meta
        gc.collect()


def coverage(X):
    return {name: round(float(X[name].notna().mean()), 5) for name in ('wx_temp_c', 'wx_visibility_m', 'wx_ceiling_ft')}


def main(argv=None):
    args = parse_args(argv)
    seeds = [42] if args.smoke or args.seeds == 1 else [42, 142, 242]
    config = dict(models.BASE_CONFIG, seeds=seeds, rounds=5 if args.smoke else 800,
                  sample=0.01 if args.smoke else 1.0, threads=args.threads,
                  prediction_threads=args.threads, train_months=6, bootstrap_reps=2000,
                  bootstrap_seed=0, block_days=1, label=args.label, smoke=args.smoke,
                  rome_hybrid=rome_hybrid.SETTINGS, weather=weather.SETTINGS, candidates=COLUMNS)
    config.pop('recipes', None)
    signature = {name: sha256(ROOT / name) for name in SOURCES}
    if not weather.METAR.is_file():
        raise FileNotFoundError(f'Missing {weather.METAR}; run VALIDATION/tools/fetch_metar.py first')
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
        con = connect()
        record['ranking_weights'] = harness.ranking_weights(con)
        con.close()
        progress = dict(models={}, folds={}, weather_coverage={})
        write_json(directory / 'result.json', record)
        write_json(directory / 'progress.json', progress)
    folds = v3_folds()
    jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
    started = time.monotonic()
    try:
        reports = weather.load_reports()
        print(f'Loaded {len(reports):,} weather reports.', flush=True)
        dep, arr = overnight_run.load_inputs(config['sample'])
        for job in jobs:
            if all(ev['fold'] in progress['folds'] for ev in job['evals']):
                print(f"Reusing completed {job['id']}", flush=True)
                continue
            td, ta, validation = overnight_run.split_job(dep, arr, job)
            print(f"Preparing {job['id']}: {len(td):,} training departures", flush=True)
            X, y, cats, raw_kept = overnight_run.prepare_training(td, ta, config)
            X = weather.add_weather(X, raw_kept, reports)
            progress['weather_coverage'][job['id']] = coverage(X)
            print(f"Weather coverage {progress['weather_coverage'][job['id']]}", flush=True)
            fit_window(directory, progress, job, X, y, raw_kept, config)
            del X, y, raw_kept
            gc.collect()
            score_window(directory, progress, job, td, ta, validation, cats, config, folds, reports)
            del td, ta, validation, cats
            gc.collect()
        del dep, arr
        record['folds'] = progress['folds']
        record['models'] = progress['models']
        record['weather_coverage'] = progress['weather_coverage']
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
