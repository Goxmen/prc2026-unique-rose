"""Protocol-v3 A/B validation of the stand clock and dirty-day inputs on row 29.

The control is row 29, Rome-signals. Its boosters are not refitted: they are loaded
from the completed Rome-signals run and must reproduce row 29 exactly on every fold
and seed. Row 29 is that run's `pred_all` with the day-shift switched off, i.e. the
rule's gap wherever `dayshift_rows` is set. Every candidate keeps row 29's late-row
estimate and leaves the day-shift off. The inputs are defined in `stand_day.py`:

    pred            row 29 control, reloaded
    pred_stand      ordinary regressor and Rome classifier + STAND_FEATURES
    pred_day        ordinary regressor and Rome classifier + DAY_FEATURES
    pred_both       ordinary regressor and Rome classifier + both
    pred_reg_both   ordinary regressor + both, classifier unchanged   (no extra fit)
    pred_cls_both   Rome classifier + both, regressor unchanged       (no extra fit)

`stand` and `day` alone against `both` show whether the two need each other;
`reg_both` and `cls_both` show which component carries a gain. The late mean is
fitted on each window's training labels and must equal the control run's.
Model audit columns and the new inputs are saved beside the predictions. The
result record follows harness.py, so `compare.py --prediction-column` works
unchanged. No full-year refits or uploads happen here.

Run:    .venv/bin/python VALIDATION/stand_day_run.py --label stand-day
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
import rome_signals_run as run29
import runway_queue
import stand_day as sd
import weather

KIND = 'clock'
CONTROL_RUN = RUNS / 'v3_20260918T120331999764Z'
CONTROL_PARTS = {'ordinary': 'airline', 'classifier': 'all'}
REGRESSOR = run29.ORDINARIES[CONTROL_PARTS['ordinary']]
CLASSIFIER = run29.CLASSIFIERS[CONTROL_PARTS['classifier']]
STAND, DAY = tuple(sd.STAND_FEATURES), tuple(sd.DAY_FEATURES)
ADDED = {'control': (), 'stand': STAND, 'day': DAY, 'both': STAND + DAY}
ORDINARIES = {name: REGRESSOR + extra for name, extra in ADDED.items()}
CLASSIFIERS = {name: CLASSIFIER + extra for name, extra in ADDED.items()}
CANDIDATES = {
    'control': dict(ordinary='control', classifier='control'),
    'stand': dict(ordinary='stand', classifier='stand'),
    'day': dict(ordinary='day', classifier='day'),
    'both': dict(ordinary='both', classifier='both'),
    'reg_both': dict(ordinary='both', classifier='control'),
    'cls_both': dict(ordinary='control', classifier='both'),
}
COLUMNS = {name: 'pred' if name == 'control' else f'pred_{name}' for name in CANDIDATES}
SOURCES = ['VALIDATION/stand_day.py', 'VALIDATION/stand_day_run.py', *run29.SOURCES]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='stand-day')
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
    """Completed Rome-signals run whose row-29 boosters serve as the control."""
    record = json.loads((CONTROL_RUN / 'result.json').read_text())
    progress = json.loads((CONTROL_RUN / 'progress.json').read_text())
    spec = record['config']['candidate_specs'].get('all')
    if (record['status'] != 'complete' or spec != dict(ordinary='airline', classifier='all', late=True, dayshift=True)
            or record['config']['ordinaries']['airline'] != list(REGRESSOR)
            or record['config']['classifiers']['all'] != list(CLASSIFIER)):
        raise ValueError(f'{CONTROL_RUN.name} is not a completed Rome-signals run with row 29')
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


def add_inputs(X, raw, reports, dep_events, arr_events):
    X = run29.add_inputs(X, raw, reports, dep_events, arr_events)
    return sd.add_stand_day(X, raw, dep_events, arr_events)


def validation_inputs(job, ev, td, ta, validation, cats, config, reports):
    vd, va = validation[ev['month']]
    V = modeling.validation_matrix(td, ta, vd, va, job['freeze'], cats, config)
    # Current context for a validation month: its training window plus that month only (protocol rule 5).
    V = add_inputs(V, vd, reports, pd.concat([td, vd], ignore_index=True), pd.concat([ta, va], ignore_index=True))
    return vd, V


def row29(control_progress, ev, seed):
    saved = pd.read_parquet(CONTROL_RUN / control_progress['folds'][ev['fold']]['predictions'][str(seed)],
                            columns=['mvt_id', 'pred_all', 'dayshift_rows', 'gap_sec'])
    return saved.mvt_id.to_numpy(), np.where(saved.dayshift_rows, saved.gap_sec, saved.pred_all)


def check_control(frame, ev, seed, control_progress):
    ids, expected = row29(control_progress, ev, seed)
    if not np.array_equal(ids, frame.mvt_id.to_numpy()):
        raise ValueError(f"{ev['fold']}/{seed}: validation rows differ from the control run")
    if not np.array_equal(expected, frame['pred'].to_numpy()):
        worst = np.nanmax(np.abs(expected - frame['pred'].to_numpy()))
        raise ValueError(f"{ev['fold']}/{seed}: reloaded control does not reproduce row 29 (max diff {worst})")


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
    if config['reuse_control']:
        check_control(frame, ev, seed, control_progress)
    for name in ORDINARIES:
        frame[f'ordinary_{name}'] = base[name]
    for name in CLASSIFIERS:
        frame[f'p_{name}'] = p[name]
    for key in ['rule_triggered', 'late_rows']:
        frame[key] = outs['control'][key]
    frame['gap_sec'] = V[rome_hybrid.GAP].to_numpy(float)
    frame['raw_aobt3_missing'] = vd['AOBT_3_flt'].isna().to_numpy()
    frame[signals.AIRLINE] = V[signals.AIRLINE].astype(object).to_numpy()
    for column in sd.ALL_FEATURES:
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
              f"{'; control reproduces row 29' if config['reuse_control'] else ''}.", flush=True)
        del V, vd
        gc.collect()


def coverage(X, raw):
    no_id = raw['FLIGHT_ID_mvt'].isna().to_numpy()
    share = lambda column, rows: round(float(X.loc[rows, column].notna().mean()), 5) if rows.any() else None
    return dict(rows=len(X), no_id_rows=int(no_id.sum()),
                **{f'{column}': share(column, np.ones(len(X), bool)) for column in sd.ALL_FEATURES},
                no_id_stand_clock=share('st_last_arr_sec', no_id))


def main(argv=None):
    args = parse_args(argv)
    seeds = [42] if args.smoke else [42, 142, 242]
    config = dict(models.BASE_CONFIG, seeds=seeds, rounds=5 if args.smoke else 800,
                  sample=0.01 if args.smoke else 1.0, threads=args.threads,
                  prediction_threads=args.threads, train_months=6, bootstrap_reps=2000,
                  bootstrap_seed=0, block_days=1, label=args.label, smoke=args.smoke,
                  rome_hybrid=rome_hybrid.SETTINGS, weather=weather.SETTINGS, queue=runway_queue.SETTINGS,
                  rome_signals=signals.SETTINGS, stand_day=sd.SETTINGS, candidates=COLUMNS,
                  candidate_specs=CANDIDATES, late=True, dayshift=False,
                  ordinaries={k: list(v) for k, v in ORDINARIES.items()},
                  classifiers={k: list(v) for k, v in CLASSIFIERS.items()},
                  reuse_control=not args.smoke,
                  control_source=dict(run=CONTROL_RUN.name, parts=CONTROL_PARTS,
                                      column="where(dayshift_rows, gap_sec, pred_all)"))
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
                # Fail before any fitting if the reloaded control cannot reproduce row 29.
                ev = job['evals'][0]
                vd, V = validation_inputs(job, ev, td, ta, validation, cats, config, reports)
                frame = overnight_run.prediction_metadata(vd, V, job, ev)
                frame['pred'] = predictions(directory, progress, control_progress, job, vd, V, seeds[0], config,
                                            ['control'])[2]['control']['final']
                check_control(frame, ev, seeds[0], control_progress)
                print(f"Control check passed on {ev['fold']}/{seeds[0]}: row 29 reproduced exactly.", flush=True)
                del vd, V, frame
                gc.collect()
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
