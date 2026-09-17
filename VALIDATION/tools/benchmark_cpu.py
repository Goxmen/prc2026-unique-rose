"""Measure local LightGBM throughput and prediction equivalence before long runs."""

import argparse
import gc
import json
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load
from modeling import training_matrix, period_rows, fit_booster
from records import write_json, runtime_identity
from common import connect


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--threads', default='4,6,8,12,20')
    ap.add_argument('--rounds', type=int, default=300)
    args = ap.parse_args()
    counts = [int(x) for x in args.threads.split(',')]
    cfg = dict(max_taxi_sec=200000, lag_train_days=0, fpl_flag=False,
               early_stopping_rounds=0, recovery='off', threads=counts[0])
    con = connect()
    dep, arr = load(con, 1.0)
    con.close()
    X, y, _ = training_matrix(period_rows(dep, '2025-01-01', '2025-06-30'),
                               period_rows(arr, '2025-01-01', '2025-06-30'), cfg)
    del dep, arr
    gc.collect()
    result = json.loads(args.output.read_text()) if args.output.exists() else {'cpu': next(line.split(':', 1)[1].strip() for line in
                         Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name')),
              'affinity': sorted(os.sched_getaffinity(0)),
              'python': platform.python_version(), 'lightgbm': lightgbm.__version__,
              'rows': len(y), 'rounds': args.rounds, 'seed': 42,
              'workers': 1, 'measurements': [],
              'memory_note': 'Peak RSS is process lifetime high-water mark, including loading.'}
    runtime = runtime_identity()
    if (result['cpu'] != runtime['cpu'] or result['affinity'] != runtime['affinity']
            or result['lightgbm'] != lightgbm.__version__ or result['rows'] != len(y)
            or result['rounds'] != args.rounds):
        raise ValueError('Existing benchmark belongs to different hardware/data/settings; choose a new output file')
    probe = X.iloc[np.linspace(0, len(X)-1, 10000, dtype=int)]
    reference = {}
    def timeout(*_):
        raise TimeoutError('Thread setting exceeded benchmark time budget')
    signal.signal(signal.SIGALRM, timeout)
    def measure(threads, rounds, repeat):
        cfg['threads'] = threads
        signal.setitimer(signal.ITIMER_REAL, 45)
        try:
            fit_booster(X, y, 42, 20, cfg, f'warmup/{threads}')
            signal.setitimer(signal.ITIMER_REAL, 180)
            before = time.monotonic()
            model = fit_booster(X, y, 42, rounds, cfg, f'cpu/{threads}/{repeat}')
        except TimeoutError as exc:
            result.setdefault('excluded', []).append(dict(threads=threads, reason=str(exc)))
            write_json(args.output, result)
            return
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - before
        pred = model.predict(probe, num_threads=threads)
        reference.setdefault(rounds, pred)
        delta = float(np.max(np.abs(pred-reference[rounds])))
        row = dict(threads=threads, rounds=rounds, repeat=repeat,
                   seconds=elapsed, max_abs_prediction_difference=delta,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        result['measurements'].append(row)
        write_json(args.output, result)
        print(row, flush=True)
        if not np.allclose(pred, reference[rounds], rtol=1e-10, atol=1e-7):
            raise ValueError('Thread settings changed predictions materially')
    for repeat, order in enumerate([counts, counts[::-1]], 1):
        for threads in order:
            if any(r['threads'] == threads for r in result.get('excluded', [])):
                continue
            if args.rounds in reference and any(r['threads'] == threads and r['repeat'] == repeat
                                               for r in result['measurements']):
                continue
            measure(threads, args.rounds, repeat)
    eligible = [n for n in counts if not any(r['threads'] == n for r in result.get('excluded', []))]
    best = sorted(eligible, key=lambda n: np.median([r['seconds'] for r in
                  result['measurements'] if r['threads'] == n]))[:2]
    for threads in best:
        measure(threads, 800, 'sustained')
    result['selected_threads'] = min(result['measurements'][-2:], key=lambda r:r['seconds'])['threads']
    write_json(args.output, result)
    print('SELECTED', result['selected_threads'], flush=True)


if __name__ == '__main__':
    main()
