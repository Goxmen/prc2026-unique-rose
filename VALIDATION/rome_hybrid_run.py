"""Protocol-v3 validation of the Rome hybrid against paired overnight recipes.

One run fits, per six-month window and seed, the overnight `control` (Q0-4h),
`clock_features` and `rule_routing` boosters plus the Rome hybrid components
with the original and clock inputs. Every prediction file holds all five
candidates, so comparisons are paired on identical folds, seeds and flights:

    pred                        control (Q0-4h)
    pred_clock_features         v19 recipe
    pred_rule_routing           v21 recipe
    pred_rome_hybrid_4h         hybrid, original inputs
    pred_rome_hybrid_4h_clock   hybrid, original + clock inputs

The result record follows harness.py, so `compare.py --prediction-column`
works unchanged. No full-year refits or uploads happen here.

Run:    .venv/bin/python VALIDATION/rome_hybrid_run.py --label rome-hybrid-4h
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

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import RANKING, TRAIN_GLOB, connect
import harness
import modeling
import overnight_models as models
import overnight_run
from protocol import SCORE_VERSION, VERSION, select_folds
from records import RUNS, append_log, data_identity, sha256, source_snapshot, write_json
from registered_rules import apply_rule
from reporting import print_report, report
import rome_hybrid

BOOSTERS = ['control', 'clock_features', 'rule_routing']
HYBRIDS = {'rome_hybrid_4h': 'original', 'rome_hybrid_4h_clock': 'clock'}
COLUMNS = {'control': 'pred', 'clock_features': 'pred_clock_features',
           'rule_routing': 'pred_rule_routing', 'rome_hybrid_4h': 'pred_rome_hybrid_4h',
           'rome_hybrid_4h_clock': 'pred_rome_hybrid_4h_clock'}
SOURCES = ['VALIDATION/rome_hybrid.py', 'VALIDATION/rome_hybrid_run.py',
           'VALIDATION/overnight_models.py', 'VALIDATION/overnight_run.py',
           'VALIDATION/modeling.py', 'VALIDATION/features_time.py', 'VALIDATION/movement_data.py',
           'pipeline/common.py', 'pipeline/clock_features.py', 'pipeline/registered_rules.py',
           'pipeline/prediction_rules.py']


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='rome-hybrid-4h')
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


def v3_folds():
    folds, skipped = select_folds('main', months=6)
    if skipped:
        raise ValueError('Unexpected skipped v3 fold')
    return {fold.name: (fold, str(start.date())) for fold, start in folds}


def save_booster(booster, path):
    temporary = path.with_suffix('.tmp')
    booster.save_model(str(temporary))
    temporary.replace(path)


def fit_window(directory, progress, job, X, y, raw_kept, config):
    for seed in config['seeds']:
        for recipe in BOOSTERS:
            key = f"{job['id']}__{recipe}__{seed}"
            if key in progress['models']:
                continue
            started = time.monotonic()
            booster, _ = models.fit_model(X, y, raw_kept, recipe, seed,
                                          rounds=config['rounds'], threads=config['threads'])
            path = directory / 'models' / f'{key}.txt'
            save_booster(booster, path)
            progress['models'][key] = dict(paths={'booster': path.name}, seconds=round(time.monotonic() - started, 1))
            write_json(directory / 'progress.json', progress)
            print(f'Fitted {key} in {progress["models"][key]["seconds"]} s', flush=True)
        for name, kind in HYBRIDS.items():
            key = f"{job['id']}__{name}__{seed}"
            if key in progress['models']:
                continue
            started = time.monotonic()
            ordinary, classifier, details = rome_hybrid.fit_components(
                X, y, raw_kept, kind, seed, rounds=config['rounds'], threads=config['threads'])
            paths = {}
            for part, booster in [('ordinary', ordinary), ('classifier', classifier)]:
                path = directory / 'models' / f'{key}.{part}.txt'
                save_booster(booster, path)
                paths[part] = path.name
            progress['models'][key] = dict(paths=paths, seconds=round(time.monotonic() - started, 1), **details)
            write_json(directory / 'progress.json', progress)
            print(f'Fitted {key} in {progress["models"][key]["seconds"]} s', flush=True)


def load(directory, entry, part):
    return lgb.Booster(model_file=str(directory / 'models' / entry['paths'][part]))


def score_window(directory, progress, job, td, ta, validation, cats, config, folds):
    for ev in job['evals']:
        if ev['fold'] in progress['folds']:
            continue
        vd, va = validation[ev['month']]
        V = modeling.validation_matrix(td, ta, vd, va, job['freeze'], cats, config)
        meta = overnight_run.prediction_metadata(vd, V, job, ev)
        files = {}
        for seed in config['seeds']:
            frame = meta.assign(seed=seed)
            for recipe in BOOSTERS:
                entry = progress['models'][f"{job['id']}__{recipe}__{seed}"]
                raw_pred = models.predict_model(load(directory, entry, 'booster'), V, recipe,
                                                threads=config['threads'])
                frame[COLUMNS[recipe]] = apply_rule(raw_pred, vd, rome_hybrid.RULE)[0]
            for name, kind in HYBRIDS.items():
                entry = progress['models'][f"{job['id']}__{name}__{seed}"]
                out = rome_hybrid.predict(load(directory, entry, 'ordinary'), load(directory, entry, 'classifier'),
                                          V, vd, kind, threads=config['threads'])
                frame[COLUMNS[name]] = out['final']
                frame[f'p_{name}'] = out['p']
                frame[f'ordinary_{name}'] = out['ordinary']
                frame['rule_triggered'] = out['rule_triggered']
            frame['gap_sec'] = V[rome_hybrid.GAP].to_numpy(float)
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


def main(argv=None):
    args = parse_args(argv)
    seeds = [42] if args.smoke or args.seeds == 1 else [42, 142, 242]
    config = dict(models.BASE_CONFIG, seeds=seeds, rounds=5 if args.smoke else 800,
                  sample=0.01 if args.smoke else 1.0, threads=args.threads,
                  prediction_threads=args.threads, train_months=6, bootstrap_reps=2000,
                  bootstrap_seed=0, block_days=1, label=args.label, smoke=args.smoke,
                  rome_hybrid=rome_hybrid.SETTINGS, candidates=COLUMNS)
    config.pop('recipes', None)
    signature = {name: sha256(ROOT / name) for name in SOURCES}
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
        record['data_identity'] = data_identity([*glob.glob(TRAIN_GLOB), RANKING])
        con = connect()
        record['ranking_weights'] = harness.ranking_weights(con)
        con.close()
        progress = dict(models={}, folds={})
        write_json(directory / 'result.json', record)
        write_json(directory / 'progress.json', progress)
    folds = v3_folds()
    jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
    started = time.monotonic()
    try:
        dep, arr = overnight_run.load_inputs(config['sample'])
        for job in jobs:
            if all(ev['fold'] in progress['folds'] for ev in job['evals']):
                print(f"Reusing completed {job['id']}", flush=True)
                continue
            td, ta, validation = overnight_run.split_job(dep, arr, job)
            print(f"Preparing {job['id']}: {len(td):,} training departures", flush=True)
            X, y, cats, raw_kept = overnight_run.prepare_training(td, ta, config)
            fit_window(directory, progress, job, X, y, raw_kept, config)
            del X, y, raw_kept
            gc.collect()
            score_window(directory, progress, job, td, ta, validation, cats, config, folds)
            del td, ta, validation, cats
            gc.collect()
        del dep, arr
        record['folds'] = progress['folds']
        record['models'] = progress['models']
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
