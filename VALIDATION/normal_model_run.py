"""Protocol-v3 validation of row 38: a separate regressor for normal flights.

The ordinary regressor behind v37 learns from every training departure except Rome's
schedule-recorded rows, so day-long and other broken labels (up to 200,000 s) stay in
its squared-error training. A residual check on v37's saved predictions found about
1 s of RMSE left in inputs the regressor already has, on normal flights only.

Normal flights are departures outside LIRF with a flight ID. For them v37 is exactly
plan_window.apply(v30's `stand` ordinary regressor) (checked on saved runs). The
candidate keeps v37 everywhere else and, on normal flights, replaces that regressor
with one trained on normal training flights only:

    pred_dayshift   v37, copied from run v3_20260919T112057144323Z (control)
    pred_normal     v37, except normal flights: plan_window.apply(normal regressor)

Same inputs (v30's `stand` ordinary set), parameters, rounds, windows and seeds as
v30. Before scoring each fold and seed, v30's saved `stand` booster is reloaded and
must reproduce v30's saved `ordinary_stand` on the rebuilt validation inputs, so the
inputs are identical. Score with

    .venv/bin/python VALIDATION/compare.py --run <this run> \
        --prediction-column pred_normal --baseline-prediction-column pred_dayshift

Run:    .venv/bin/python VALIDATION/normal_model_run.py
Smoke:  add --smoke (first window, seed 42, five rounds; execution check only)
"""

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import overnight_models as models
import overnight_run
import plan_window
from records import RUNS, append_log, sha256, source_snapshot, write_json
from reporting import print_report, report
import rome_hybrid
from rome_hybrid_run import load, save_booster
import rome_signals as signals
import stand_day_run as run30

V30_RUN = RUNS / 'v3_20260918T151436598224Z'
V37_RUN = RUNS / 'v3_20260919T112057144323Z'
CONTROL = 'pred_dayshift'
COLUMN = 'pred_normal'
EXTRA = run30.ORDINARIES['stand']
KIND = run30.KIND
SOURCES = ['VALIDATION/normal_model_run.py', 'VALIDATION/plan_window.py', *run30.SOURCES]


def normal_rows(raw):
    """Departures outside Rome with a flight ID."""
    return (raw['ADEP_mvt'].to_numpy(object) != rome_hybrid.AIRPORT) & raw['FLIGHT_ID_mvt'].notna().to_numpy()


def fit_normal(X, y, raw, seed, rounds, threads):
    keep = normal_rows(raw)
    names = rome_hybrid.regressor_features(KIND, EXTRA)
    return signals._train(models.model_params('control', seed, threads), X.loc[keep, names],
                          np.asarray(y, float)[keep], rounds, f'normal/{seed}')


def check_inputs(v30_progress, job, ev, V, seed, threads):
    """v30's own booster on the rebuilt inputs must give v30's saved ordinary_stand."""
    key = f"{job['id']}__ordinary_stand__{seed}"
    booster = load(V30_RUN, v30_progress['models'][key], 'ordinary')
    rebuilt = signals.ordinary_prediction(booster, V, KIND, threads, EXTRA)
    saved = pd.read_parquet(V30_RUN / v30_progress['folds'][ev['fold']]['predictions'][str(seed)],
                            columns=['ordinary_stand']).ordinary_stand.to_numpy()
    if not np.array_equal(rebuilt, saved):
        raise ValueError(f"{ev['fold']}/{seed}: rebuilt inputs do not reproduce v30 "
                         f"(max diff {np.nanmax(np.abs(rebuilt - saved))})")


def candidate(frame, base, vd):
    """v37 with normal flights replaced by the plan-window-clipped normal regressor."""
    if (not np.array_equal(frame.mvt_id.to_numpy(), vd['mvt_id'].to_numpy())
            or not np.array_equal(frame.y.to_numpy(float), vd['taxi_sec'].to_numpy(float))):
        raise ValueError('Validation rows differ from v37')
    rows = normal_rows(vd)
    out = frame[CONTROL].to_numpy(float).copy()
    out[rows] = plan_window.apply(base, vd)[rows]
    return out, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(argv)
    started = time.monotonic()
    source = json.loads((V37_RUN / 'result.json').read_text())
    v30_record = json.loads((V30_RUN / 'result.json').read_text())
    v30_progress = json.loads((V30_RUN / 'progress.json').read_text())
    if source['status'] != 'complete' or source['prediction_column'] != CONTROL:
        raise ValueError(f'{V37_RUN.name} is not the completed v37 run')
    if v30_record['config']['ordinaries']['stand'] != list(EXTRA):
        raise ValueError('v30 stand inputs differ from stand_day_run.ORDINARIES["stand"]')
    seeds = [42] if args.smoke else [int(s) for s in source['config']['seeds']]
    rounds = 5 if args.smoke else int(v30_record['config']['rounds'])
    config = dict(v30_record['config'], threads=args.threads, prediction_threads=args.threads, sample=1.0)
    at = datetime.now(timezone.utc)
    run_id = 'v3_' + at.strftime('%Y%m%dT%H%M%S%fZ')
    out_dir = RUNS / run_id
    (out_dir / 'models').mkdir(parents=True)
    jobs = [job for job in overnight_run.schedule() if job['scheme'] == 'v3']
    if args.smoke:
        jobs = jobs[:1]
    total = len(jobs) * len(seeds) + sum(len(job['evals']) for job in jobs)
    step = 0
    record = copy.deepcopy(source)
    folds, boosters, counts = {}, {}, {}
    reports = run30.weather.load_reports()
    dep, arr = overnight_run.load_inputs(1.0)
    for job in jobs:
        td, ta, validation = overnight_run.split_job(dep, arr, job)
        X, y, cats, raw_kept = overnight_run.prepare_training(td, ta, config)
        X = run30.add_inputs(X, raw_kept, reports, td, ta)
        counts[job['id']] = dict(training_rows=len(y), normal_training_rows=int(normal_rows(raw_kept).sum()))
        fitted = {}
        for seed in seeds:
            t0 = time.monotonic()
            model = fit_normal(X, y, raw_kept, seed, rounds, args.threads)
            key = f"{job['id']}__normal__{seed}"
            path = out_dir / 'models' / f'{key}.txt'
            save_booster(model, path)
            boosters[key] = dict(paths={'ordinary': path.name}, seconds=round(time.monotonic() - t0, 1))
            fitted[seed] = model
            step += 1
            print(f'STEP {step}/{total} fitted {key} in {boosters[key]["seconds"]} s', flush=True)
        del X, y, raw_kept
        gc.collect()
        for ev in job['evals']:
            vd, V = run30.validation_inputs(job, ev, td, ta, validation, cats, config, reports)
            info = copy.deepcopy(source['folds'][ev['fold']])
            for seed in seeds:
                check_inputs(v30_progress, job, ev, V, seed, args.threads)
                name = info['predictions'][str(seed)]
                frame = pd.read_parquet(V37_RUN / name)
                base = signals.ordinary_prediction(fitted[seed], V, KIND, args.threads, EXTRA)
                frame[COLUMN], rows = candidate(frame, base, vd)
                frame['normal_row'] = rows
                frame['normal_regressor'] = base
                frame.to_parquet(out_dir / name, index=False)
            info['predictions'] = {str(s): info['predictions'][str(s)] for s in seeds}
            folds[ev['fold']] = info
            step += 1
            print(f"STEP {step}/{total} scored {ev['fold']} ({ev['month']}); v30 inputs reproduced", flush=True)
            del vd, V
            gc.collect()
        del td, ta, validation, cats
        gc.collect()
    record.update(run_id=run_id, label=('smoke:' if args.smoke else '') + 'normal-model',
                  at=at.isoformat(timespec='seconds'), prediction_column=COLUMN,
                  derived_from=dict(run=V37_RUN.name, column=CONTROL, inputs_from=V30_RUN.name),
                  signature={name: sha256(ROOT / name) for name in SOURCES},
                  source_signature=source['signature'], source=source_snapshot(out_dir),
                  folds=folds, models=boosters, official_rmse=None, baseline_id=None,
                  normal_model=dict(rows='ADEP != LIRF and FLIGHT_ID_mvt present', extra_features=list(EXTRA),
                                    rounds=rounds, windows=counts),
                  full_benchmark=not args.smoke)
    record['config'] = dict(record['config'], label='normal-model', seeds=seeds,
                            candidates={'normal': COLUMN, 'control': CONTROL})
    if not args.smoke:
        record['report'] = report(record, out_dir)
        print_report(record['report'], slices=False)
    record['status'] = 'execution_check_only' if args.smoke else 'complete'
    record['minutes'] = round((time.monotonic() - started) / 60, 3)
    write_json(out_dir / 'result.json', record)
    if not args.smoke:
        append_log(record)
    print(f'DONE wrote {out_dir.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
