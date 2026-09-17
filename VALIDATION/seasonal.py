"""Retrospective January/July holdouts for the five already-scored recipes.

Run both strategies with --strategy both. --dry-run checks the recipe and prints
the schedule; --smoke performs a small execution check, never a scored experiment.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
from common import TRAIN_GLOB, RANKING, connect
from movement_data import load_phase
from records import data_identity, runtime_identity, sha256, write_json
from registered_rules import apply_rule

MANIFEST = ROOT / 'VALIDATION/reports/leaderboard_selected_20260916.json'
R3_RECORD = ROOT / 'VALIDATION/runs/v3_20260916T113256202842Z/result.json'
EQUIVALENCE = ROOT / 'VALIDATION/reports/stage1_off_equivalence.json'
MONTHS = tuple(f'2025-{i:02}' for i in range(1, 13))
HOLDOUTS = ('2025-01', '2025-07')
FREEZE = '2025-12-31'
PROTOCOL = 'seasonal-retrospective-v1'
RECIPES = {
    'B0-800': ('baseline', 'off'), 'Q0': ('baseline', 'rome_6h'),
    'Q1': ('recovery', 'rome_6h'), 'Q0-4h': ('baseline', 'rome_4h'),
    'Q1-4h': ('recovery', 'rome_4h'),
}
DEPENDENCIES = (
    'pipeline/common.py', 'pipeline/recovery.py', 'pipeline/recovery_relaxed.py',
    'pipeline/recovery_history.py', 'VALIDATION/features_time.py',
    'VALIDATION/movement_data.py', 'VALIDATION/protocol.py',
)


def schedule(strategy):
    jobs = []
    if strategy in ('10m', 'both'):
        jobs.append(dict(id='10m', strategy='10m',
                         train=[m for m in MONTHS if m not in HOLDOUTS], val=list(HOLDOUTS)))
    if strategy in ('11m', 'both'):
        for month in HOLDOUTS:
            jobs.append(dict(id='11m_' + month, strategy='11m',
                             train=[m for m in MONTHS if m != month], val=[month]))
    if not jobs:
        raise ValueError('Unknown seasonal strategy')
    return jobs


def month_subset(frame, months):
    labels = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    return frame.loc[labels.isin(months)].copy()


def split_frames(dep, arr, job):
    if set(job['train']) & set(job['val']):
        raise ValueError('Training and held-out months overlap')
    td, ta = month_subset(dep, job['train']), month_subset(arr, job['train'])
    validation = {month: (month_subset(dep, [month]), month_subset(arr, [month]))
                  for month in job['val']}
    if td.empty or any(vd.empty for vd, _ in validation.values()):
        raise ValueError('Empty training or held-out departure population')
    for vd, _ in validation.values():
        if set(td.mvt_id) & set(vd.mvt_id):
            raise ValueError('A held-out departure entered training')
    return td, ta, validation


def references():
    batch = json.loads(MANIFEST.read_text())
    items = {item['codename']: item for item in batch['submissions']}
    if set(items) != set(RECIPES):
        raise ValueError('Expected exactly the five recorded submissions')
    for name, (parent, rule) in RECIPES.items():
        item = items[name]
        if (item['model'], item['prediction_rule']) != (parent, rule):
            raise ValueError(f'Changed submitted recipe: {name}')
        official = item['official_result']
        if (official['status'] != 'Succeeded' or official['used_pairs'] != 344841
                or official['score'] != item['score'] or not np.isfinite(item['score'])):
            raise ValueError(f'No matching successful official score: {name}')
        for filename, digest in item['rule_source'].items():
            if sha256(ROOT / filename) != digest:
                raise ValueError(f'Changed submitted rule: {filename}')
    return batch, items


def model_source():
    """Recover the hash-verified Stage 1 wrapper, not Stage 3/4 model changes.

    The baseline uses its audited recovery-off path. The audit compared all six
    original training matrices and all sixteen validation matrices with B0-800.
    This is an explicitly recorded equivalent implementation, not a claim of
    byte-identical B0 source or a bypass of the submission source guards.
    """
    reference = json.loads(R3_RECORD.read_text())
    expected = reference['source']['files']
    for name in DEPENDENCIES:
        if sha256(ROOT / name) != expected[name]:
            raise ValueError(f'Stage 1 dependency changed: {name}')
    name = 'VALIDATION/modeling.py'
    wanted = expected[name]
    archived = R3_RECORD.parent / 'source' / name
    candidates = [archived, ROOT / name]
    blob = next((p.read_bytes() for p in candidates if p.exists() and sha256(p) == wanted), None)
    if blob is None:
        commits = subprocess.check_output(
            ['git', 'log', '--all', '--format=%H', '--', name], cwd=ROOT, text=True).splitlines()
        for commit in commits:
            candidate = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
            if hashlib.sha256(candidate).hexdigest() == wanted:
                blob = candidate
                break
    if blob is None:
        raise ValueError('Stage 1 wrapper unavailable: restore its archived source or fetch its branch')
    audit = json.loads(EQUIVALENCE.read_text())
    baseline = json.loads((ROOT / 'VALIDATION/runs' / audit['control'] / 'result.json').read_text())
    if (audit['current_modeling_sha256'] != wanted
            or audit['frozen_modeling_sha256'] != baseline['source']['files'][name]):
        raise ValueError('Recovery-off equivalence evidence does not match the two implementations')
    return blob, {name: wanted, **{p: expected[p] for p in DEPENDENCIES}}, reference


def load_backend(blob, directory):
    path = directory / 'source/VALIDATION/modeling.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location('seasonal_stage1_modeling', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_frame(raw, matrix, job, month):
    times = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
    days = times.dt.normalize()
    available = np.zeros(len(raw), dtype=np.int8)
    # Noncontiguous training: an excluded month is not available just because
    # it lies between the first and last training dates.
    for lag in range(1, 8):
        previous_month = (days - pd.Timedelta(days=lag)).dt.strftime('%Y-%m')
        available += previous_month.isin(job['train'] + [month]).to_numpy(np.int8)
    return pd.DataFrame({
        'mvt_id': raw.mvt_id.to_numpy(), 'mvt_time': times.to_numpy(),
        'airport': raw.ADEP_mvt.to_numpy(), 'y': raw.taxi_sec.to_numpy(float),
        'has_flight_plan': raw.AOBT_3_flt.notna().to_numpy(),
        'arrival_history_available': matrix.arr_taxiin_7d.notna().to_numpy(),
        'arrival_history_days_available': available,
        'raw_flight_id_missing': raw.FLIGHT_ID_mvt.isna().to_numpy(),
    })


def check_artifacts(directory, progress):
    for kind in ('models', 'predictions'):
        for item in progress[kind].values():
            for name, digest in item['artifacts'].items():
                path = directory / name
                if not path.is_file() or sha256(path) != digest:
                    raise ValueError(f'Missing/changed completed artifact: {path}')


def execute(dep, arr, jobs, config, backend, directory, progress):
    for job in jobs:
        for parent in ('baseline', 'recovery'):
            prefix = f"{job['id']}_{parent}"
            keys = [f'{prefix}_{month}_{seed}' for month in job['val'] for seed in config['seeds']]
            if all(key in progress['predictions'] for key in keys):
                print(f'Reusing completed {prefix}', flush=True)
                continue
            td, ta, validation = split_frames(dep, arr, job)
            cfg = dict(config, recovery='off' if parent == 'baseline' else 'history')
            print(f"Preparing {prefix}: {len(job['train'])} training months, {len(td):,} departures", flush=True)
            prepared = []
            X, y, cats = backend.training_matrix(td, ta, cfg, prepared_out=prepared)
            if list(X) != config['features']:
                raise ValueError('Feature names/order differ from the submitted 27-feature recipe')
            for seed in config['seeds']:
                key = f'{prefix}_{seed}'
                if key in progress['models']:
                    continue
                started, history = time.monotonic(), {}
                booster = backend.fit_booster(X, y, seed, cfg['rounds'], cfg, key, training_history=history)
                model = directory / 'models' / f'{key}.txt'
                model.parent.mkdir(parents=True, exist_ok=True)
                temporary = model.with_suffix('.tmp')
                booster.save_model(str(temporary))
                temporary.replace(model)
                curve = model.with_suffix('.loss.json')
                write_json(curve, history)
                progress['models'][key] = {
                    'path': str(model.relative_to(directory)), 'rows': len(y), 'seed': seed,
                    'job': job, 'parent': parent, 'minutes': (time.monotonic() - started) / 60,
                    'artifacts': {str(p.relative_to(directory)): sha256(p) for p in (model, curve)},
                }
                write_json(directory / 'progress.json', progress)
                del booster
            del X, y
            gc.collect()
            for month, (vd, va) in validation.items():
                if all(f'{prefix}_{month}_{seed}' in progress['predictions'] for seed in config['seeds']):
                    continue
                options = {'prepared_training': prepared[0]} if prepared else {}
                V = backend.validation_matrix(td, ta, vd, va, FREEZE, cats, cfg, **options)
                meta = prediction_frame(vd, V, job, month)
                for seed in config['seeds']:
                    key = f'{prefix}_{month}_{seed}'
                    if key in progress['predictions']:
                        continue
                    booster = lgb.Booster(model_file=str(directory / progress['models'][f'{prefix}_{seed}']['path']))
                    pred = backend.predict(booster, V, threads=cfg['prediction_threads'])
                    frame = meta.copy()
                    frame['seed'], frame['month'], frame['parent_pred'] = seed, month, pred
                    for name, (source_parent, rule) in RECIPES.items():
                        if source_parent == parent:
                            frame[name], audit = apply_rule(pred, vd, rule)
                            frame[name + '_triggered'] = audit.rule_triggered.to_numpy()
                    path = directory / f'{key}.parquet'
                    temporary = path.with_suffix('.parquet.tmp')
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                    progress['predictions'][key] = {
                        'path': path.name, 'job': job['id'], 'strategy': job['strategy'],
                        'parent': parent, 'month': month, 'seed': seed, 'rows': len(frame),
                        'artifacts': {path.name: sha256(path)},
                    }
                    write_json(directory / 'progress.json', progress)
                    del booster, frame
                del V, meta
            del td, ta, validation, prepared
            gc.collect()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=['both', '10m', '11m'], default='both')
    parser.add_argument('--seeds', type=int, choices=[1, 3], default=3,
                        help='1 uses submission seed 42; 3 adds 142/242 (default)')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--prediction-threads', type=int, default=8)
    parser.add_argument('--bootstrap-reps', type=int, default=2000)
    parser.add_argument('--negligible-sec', type=float, default=1.0,
                        help='predeclared tolerance for direction comparisons, in seconds')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'VALIDATION/runs/seasonal_jan_jul')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='1%% rows, five rounds, seed 42; execution only')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if min(args.threads, args.prediction_threads, args.bootstrap_reps) < 1 or not np.isfinite(args.negligible_sec) or args.negligible_sec < 0:
        parser.error('Threads/bootstrap repetitions must be positive; tolerance must be finite and nonnegative')
    return args


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process is using this seasonal run directory') from exc
        yield


def main(argv=None):
    args = parse_args(argv)
    batch, items = references()
    blob, sources, reference = model_source()
    runtime = runtime_identity()
    if runtime['versions'] != reference['runtime']['versions']:
        raise ValueError('Library versions differ from the submitted recipes')
    jobs = schedule(args.strategy)
    seeds = [42] if args.smoke else [42 + 100 * i for i in range(args.seeds)]
    for job in jobs:
        print(f"{job['id']}: train {', '.join(job['train'])}; evaluate {', '.join(job['val'])}")
    print(f'{len(jobs)*2*len(seeds)} fits; baseline and R3; five rule recipes; seeds {seeds}.')
    print(f'Retrospective history: training-only labels, frozen at {FREEZE}; 27 features.')
    print(f"Hardware: {runtime['cpu']}; {args.threads} fit / {args.prediction_threads} prediction threads.")
    if args.dry_run:
        print('Recipe, source hashes and library versions verified. No data loaded or models fitted.')
        return
    directory = args.output_dir.resolve()
    with run_lock(directory):
        config = dict(reference['config'], seeds=seeds, rounds=5 if args.smoke else 800,
                      sample=0.01 if args.smoke else 1.0, threads=args.threads,
                      prediction_threads=args.prediction_threads, fit_workers=1,
                      bootstrap_reps=50 if args.smoke else args.bootstrap_reps,
                      negligible_sec=args.negligible_sec, features=batch['models']['recovery']['features'])
        # Keep only recipe/execution fields that apply to this new protocol.
        for name in ('folds', 'train_months', 'label', 'smoke', 'recovery', 'checkpoints'):
            config.pop(name, None)
        config['smoke'] = args.smoke
        paths = sorted((ROOT / 'data').glob('training_*.parquet')) + [RANKING]
        identity = data_identity(paths)
        expected_data = items['B0-800']['data_identity']
        if identity != sorted(expected_data, key=lambda x: x['file']):
            raise ValueError('Dataset hashes differ from the scored submissions')
        runner_files = ['VALIDATION/seasonal.py', 'VALIDATION/seasonal_report.py',
                        'VALIDATION/reporting.py', 'VALIDATION/records.py',
                        'pipeline/prediction_rules.py', 'pipeline/registered_rules.py']
        signature = dict(protocol=PROTOCOL, jobs=jobs, config=config, data_identity=identity,
                         model_source=sources, runner_source={n: sha256(ROOT / n) for n in runner_files},
                         manifest_sha256=sha256(MANIFEST), equivalence_sha256=sha256(EQUIVALENCE),
                         versions=runtime['versions'], freeze=FREEZE,
                         ranking_weights=reference['ranking_weights'])
        result_path = directory / 'result.json'
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if not args.resume:
                raise ValueError('Existing run: add --resume or choose another --output-dir')
            if record['signature'] != signature:
                raise ValueError('Cannot resume: configuration, source, data or references changed')
            progress = json.loads((directory / 'progress.json').read_text())
            check_artifacts(directory, progress)
        else:
            if args.resume:
                raise ValueError('No run to resume at this --output-dir')
            record = dict(signature=signature, status='running', runtime=runtime,
                          started_at=datetime.now(timezone.utc).isoformat(),
                          scores_checked_at=batch['scores_checked_at'], references=items)
            progress = dict(models={}, predictions={})
            write_json(result_path, record)
            write_json(directory / 'progress.json', progress)
        for name in list(sources) + runner_files:
            if name == 'VALIDATION/modeling.py':
                continue
            destination = directory / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        backend = load_backend(blob, directory)
        if backend.PARAMS != reference['config']['params']:
            raise ValueError('Model parameters differ from the frozen recipe')
        required = sum(len(j['val']) for j in jobs) * 2 * len(seeds)
        if len(progress['predictions']) != required:
            con = connect()
            try:
                con.execute('SET threads=2')
                dep = load_phase(con, TRAIN_GLOB, 'DEP', config['sample'])
                arr = load_phase(con, TRAIN_GLOB, 'ARR', config['sample'])
            finally:
                con.close()
            if dep.mvt_id.isna().any() or dep.mvt_id.duplicated().any() or not np.isfinite(dep.taxi_sec).all():
                raise ValueError('Departure IDs/targets must be unique, present and finite')
            if dep.MVT_TIME_UTC_mvt.isna().any() or arr.MVT_TIME_UTC_mvt.isna().any():
                raise ValueError('Missing movement timestamps')
            print(f'Loaded {len(dep):,} departures and {len(arr):,} arrivals.', flush=True)
            execute(dep, arr, jobs, config, backend, directory, progress)
            del dep, arr
        from seasonal_report import report
        record['report'] = report(directory, record, progress)
        record['status'] = 'execution_check_only' if args.smoke else 'complete'
        write_json(result_path, record)
        print(f"Saved {directory / 'comparison.md'}", flush=True)


if __name__ == '__main__':
    main()
