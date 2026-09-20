"""Protocol-v3 A/B validation of the OPDI ground-track inputs on row 30.

The control is row 30, Stand-clock. Its boosters are not refitted: they are loaded
from the completed stand-clock run and must reproduce row 30 (`pred_stand`) exactly
on every fold and seed. Every candidate keeps row 30's late-row estimate and leaves
the day-shift off. The inputs are defined in `opdi.py` and read from the prebuilt
table `data/external/opdi/opdi_features.parquet`:

    pred              row 30 control, reloaded
    pred_opdi         ordinary regressor and Rome classifier + OPDI_FEATURES
    pred_reg_opdi     ordinary regressor + OPDI_FEATURES, classifier unchanged (no extra fit)

Each column is also saved after row 34's post-processing (the flight-plan window of
row 33, then the no-ID cap of row 34), so `pred34` is row 34 and `pred34_opdi` is
row 34 with the OPDI inputs:

    pred34, pred34_opdi, pred34_reg_opdi

Compare with `compare.py --run <dir> --prediction-column pred34_opdi
--baseline-prediction-column pred34`. No full-year refits or uploads happen here.

Run:    .venv/bin/python VALIDATION/opdi_run.py --label opdi
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
import opdi
import overnight_models as models
import overnight_run
import plan_window
from protocol import SCORE_VERSION, VERSION
import record_fixes
from records import RUNS, append_log, data_identity, sha256, source_snapshot, write_json
from registered_rules import apply_rule as _apply_rule
from reporting import print_report, report
import rome_hybrid
from rome_hybrid_run import load, save_booster, v3_folds
import rome_signals as signals
import runway_queue
import stand_day as sd
import stand_day_run as run30
import weather

KIND = 'clock'
CONTROL_RUN = RUNS / 'v3_20260918T151436598224Z'
CONTROL_PARTS = {'ordinary': 'stand', 'classifier': 'stand'}
CONTROL_COLUMN = 'pred_stand'
REGRESSOR = run30.ORDINARIES[CONTROL_PARTS['ordinary']]
CLASSIFIER = run30.CLASSIFIERS[CONTROL_PARTS['classifier']]
OPDI = tuple(opdi.OPDI_FEATURES)
ADDED = {'control': (), 'opdi': OPDI}
ORDINARIES = {name: REGRESSOR + extra for name, extra in ADDED.items()}
CLASSIFIERS = {name: CLASSIFIER + extra for name, extra in ADDED.items()}
CANDIDATES = {
    'control': dict(ordinary='control', classifier='control'),
    'opdi': dict(ordinary='opdi', classifier='opdi'),
    'reg_opdi': dict(ordinary='opdi', classifier='control'),
}
COLUMNS = {name: 'pred' if name == 'control' else f'pred_{name}' for name in CANDIDATES}
POST_COLUMNS = {name: 'pred34' if name == 'control' else f'pred34_{name}' for name in CANDIDATES}
SOURCES = ['VALIDATION/opdi.py', 'VALIDATION/opdi_run.py', 'VALIDATION/plan_window.py',
           'VALIDATION/record_fixes.py', *run30.SOURCES]
_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = opdi.load_features()
    return _TABLE


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='opdi')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seeds', default='42,142,242',
                    help='comma-separated seeds; one seed is a quick read, not a protocol score')
    ap.add_argument('--max-taxi-sec', type=int, default=models.BASE_CONFIG['max_taxi_sec'],
                    help='drop training departures whose label exceeds this; evaluation keeps every row')
    ap.add_argument('--exclude-rule-rows', action='store_true',
                    help='drop training departures the prediction rule overwrites; evaluation keeps every row')
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
    """Completed stand-clock run whose row-30 boosters serve as the control."""
    record = json.loads((CONTROL_RUN / 'result.json').read_text())
    progress = json.loads((CONTROL_RUN / 'progress.json').read_text())
    spec = record['config']['candidate_specs'].get('stand')
    if (record['status'] != 'complete' or spec != dict(ordinary='stand', classifier='stand')
            or record['config']['ordinaries']['stand'] != list(REGRESSOR)
            or record['config']['classifiers']['stand'] != list(CLASSIFIER)):
        raise ValueError(f'{CONTROL_RUN.name} is not a completed stand-clock run with row 30')
    changed = sorted(name for name, digest in record['signature'].items()
                     if (ROOT / name).is_file() and sha256(ROOT / name) != digest)
    return record, progress, changed


def fitted(part, config):
    names = ORDINARIES if part == 'ordinary' else CLASSIFIERS
    return [name for name in names if name != 'control' or not config['reuse_control']]


def booster(directory, progress, control_progress, job, part, name, seed, config):
    if name == 'control' and config['reuse_control']:
        key = f"{job['id']}__{part}_{CONTROL_PARTS[part]}__{seed}"
        return load(CONTROL_RUN, control_progress['models'][key], part)
    return load(directory, progress['models'][f"{job['id']}__{part}_{name}__{seed}"], part)


def rule_rows(raw, rule):
    """Rows whose prediction the rule replaces, from observable inputs only.

    The zero vector is a placeholder: only the audit's trigger mask is used, so
    no label and no model prediction enters this decision.
    """
    _, audit = _apply_rule(np.zeros(len(raw), dtype=float), raw, rule)
    return audit.rule_triggered.to_numpy(bool)


def add_inputs(X, raw, reports, dep_events, arr_events):
    X = run30.add_inputs(X, raw, reports, dep_events, arr_events)
    return opdi.add_opdi(X, raw, table())


def validation_inputs(job, ev, td, ta, validation, cats, config, reports):
    vd, va = validation[ev['month']]
    V = modeling.validation_matrix(td, ta, vd, va, job['freeze'], cats, config)
    # Current context for a validation month: its training window plus that month only (protocol rule 5).
    V = add_inputs(V, vd, reports, pd.concat([td, vd], ignore_index=True), pd.concat([ta, va], ignore_index=True))
    return vd, V


def check_control(frame, ev, seed, control_progress):
    saved = pd.read_parquet(CONTROL_RUN / control_progress['folds'][ev['fold']]['predictions'][str(seed)],
                            columns=['mvt_id', CONTROL_COLUMN])
    if not np.array_equal(saved.mvt_id.to_numpy(), frame.mvt_id.to_numpy()):
        raise ValueError(f"{ev['fold']}/{seed}: validation rows differ from the control run")
    if not np.array_equal(saved[CONTROL_COLUMN].to_numpy(), frame['pred'].to_numpy()):
        worst = np.nanmax(np.abs(saved[CONTROL_COLUMN].to_numpy() - frame['pred'].to_numpy()))
        raise ValueError(f"{ev['fold']}/{seed}: reloaded control does not reproduce row 30 (max diff {worst})")


def post_process(pred, raw):
    """Row 34's post-processing: flight-plan window (row 33), then the no-ID cap."""
    return record_fixes.noid_cap(plan_window.apply(pred, raw), raw)


def predictions(directory, progress, control_progress, job, vd, V, seed, config, names):
    threads, late_mean = config['threads'], progress['parameters'][job['id']]['late_mean']
    base = {name: signals.ordinary_prediction(
                booster(directory, progress, control_progress, job, 'ordinary', name, seed, config),
                V, KIND, threads, ORDINARIES[name])
            for name in {CANDIDATES[c]['ordinary'] for c in names}}
    p = {name: signals.substitution_probability(
             booster(directory, progress, control_progress, job, 'classifier', name, seed, config),
             V, vd, KIND, threads, CLASSIFIERS[name])
         for name in {CANDIDATES[c]['classifier'] for c in names}}
    outs = {c: signals.combine(base[CANDIDATES[c]['ordinary']], p[CANDIDATES[c]['classifier']], V, vd,
                               late_mean=late_mean, dayshift_weight=None)
            for c in names}
    return base, p, outs


def predict_frame(directory, progress, control_progress, job, ev, vd, V, seed, config):
    frame = overnight_run.prediction_metadata(vd, V, job, ev).assign(seed=seed)
    base, p, outs = predictions(directory, progress, control_progress, job, vd, V, seed, config, list(CANDIDATES))
    for name, out in outs.items():
        frame[COLUMNS[name]] = out['final']
        frame[POST_COLUMNS[name]] = post_process(out['final'], vd)
    if config['reuse_control']:
        check_control(frame, ev, seed, control_progress)
    for name in ORDINARIES:
        frame[f'ordinary_{name}'] = base[name]
    for name in CLASSIFIERS:
        frame[f'p_{name}'] = p[name]
    for key in ['rule_triggered', 'late_rows']:
        frame[key] = outs['control'][key]
    frame['gap_sec'] = V[rome_hybrid.GAP].to_numpy(float)
    for column in OPDI:
        frame[column] = V[column].to_numpy()
    return frame


def fit_parameters(directory, progress, control_progress, job, X, y, raw_kept, config):
    parameters = signals.fit_parameters(X, y, raw_kept)
    if config['reuse_control']:
        saved = control_progress['parameters'][job['id']]['late_mean']
        if parameters['late_mean'] != saved:
            raise ValueError(f"{job['id']}: late mean {parameters['late_mean']} differs from the control run's {saved}")
    progress['parameters'][job['id']] = parameters
    write_json(directory / 'progress.json', progress)
    print(f"Fitted corrections for {job['id']}: late mean {parameters['late_mean']}", flush=True)


def fit_window(directory, progress, job, X, y, raw_kept, config):
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
              f"{'; control reproduces row 30' if config['reuse_control'] else ''}.", flush=True)
        del V, vd
        gc.collect()


def coverage(X, raw):
    airports = raw['ADEP_mvt'].to_numpy(object)
    out = dict(rows=len(X), **{c: round(float(X[c].notna().mean()), 5) for c in OPDI})
    out['op_ground_by_airport'] = {a: round(float(X.loc[airports == a, 'op_ground_sec'].notna().mean()), 4)
                                   for a in sorted(set(airports))}
    return out


def main(argv=None):
    args = parse_args(argv)
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(',')]
    config = dict(models.BASE_CONFIG, seeds=seeds, rounds=5 if args.smoke else 800,
                  sample=0.01 if args.smoke else 1.0, threads=args.threads,
                  prediction_threads=args.threads, max_taxi_sec=args.max_taxi_sec,
                  exclude_rule_rows=args.exclude_rule_rows,
                  train_months=6, bootstrap_reps=2000,
                  bootstrap_seed=0, block_days=1, label=args.label, smoke=args.smoke,
                  rome_hybrid=rome_hybrid.SETTINGS, weather=weather.SETTINGS, queue=runway_queue.SETTINGS,
                  rome_signals=signals.SETTINGS, stand_day=sd.SETTINGS, opdi=opdi.SETTINGS,
                  plan_window_margin_sec=plan_window.MARGIN_SEC,
                  candidates=COLUMNS, post_candidates=POST_COLUMNS,
                  candidate_specs=CANDIDATES, late=True, dayshift=False,
                  ordinaries={k: list(v) for k, v in ORDINARIES.items()},
                  classifiers={k: list(v) for k, v in CLASSIFIERS.items()},
                  reuse_control=not args.smoke,
                  control_source=dict(run=CONTROL_RUN.name, parts=CONTROL_PARTS, column=CONTROL_COLUMN))
    config.pop('recipes', None)
    signature = {name: sha256(ROOT / name) for name in SOURCES}
    control_record, control_progress, changed = control_source()
    if config['reuse_control']:
        if control_record['config']['rounds'] != config['rounds'] or not set(seeds) <= set(control_record['config']['seeds']):
            raise ValueError('Control run rounds differ from this run, or it lacks one of its seeds')
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
        record['opdi_identity'] = data_identity([opdi.FEATURES_PATH])
        if config['reuse_control'] and record['data_identity'] != control_record['data_identity']:
            raise ValueError('Data files differ from the control run')
        con = connect()
        record['ranking_weights'] = harness.ranking_weights(con)
        con.close()
        progress = dict(models={}, folds={}, parameters={}, input_coverage={})
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
            fit_parameters(directory, progress, control_progress, job, X, y, raw_kept, config)
            if config['reuse_control'] and number == 0 and not progress['folds']:
                # Fail before any fitting if the reloaded control cannot reproduce row 30.
                ev = job['evals'][0]
                vd, V = validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                frame = overnight_run.prediction_metadata(vd, V, job, ev)
                frame['pred'] = predictions(directory, progress, control_progress, job, vd, V, seeds[0], config,
                                            ['control'])[2]['control']['final']
                check_control(frame, ev, seeds[0], control_progress)
                print(f"Control check passed on {ev['fold']}/{seeds[0]}: row 30 reproduced exactly.", flush=True)
                del vd, V, frame
                gc.collect()
            if config['exclude_rule_rows']:
                # A row the rule overwrites never reaches the model at scoring time,
                # so fitting its label only distorts the rest. Context, categories,
                # history and the correction parameters above still see every row.
                drop = rule_rows(raw_kept, config['prediction_rule'])
                X, y, raw_kept = X.loc[~drop], y[~drop], raw_kept.loc[~drop]
                print(f"Excluded {int(drop.sum()):,} rule-covered training rows from {job['id']}", flush=True)
            # Training rows see only their own window's movements; every supplied departure counts (rule 6).
            X = add_inputs(X, raw_kept, reports, td, ta)
            progress['input_coverage'][job['id']] = coverage(X, raw_kept)
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
        record['input_coverage'] = progress['input_coverage']
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
