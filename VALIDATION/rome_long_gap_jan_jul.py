"""Fixed R20 specialist for Rome departures more than one hour past schedule.

Training membership is the sole model change: retain the original 68 inputs,
binary labels, R20 settings and 1700 rounds, but fit only the input-defined
long-gap population. Apply its probabilities only within that same gate.
Score full and fixed-half replacement through the unchanged v58 final stack.
No threshold search, new features, upload, or automatic seed expansion.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd

import rome_own_arrival_jan_jul as source

ROOT = source.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261010_rome_long_gap_jan_jul'
GAP = 'mvt_minus_sched_sec'
THRESHOLD = 3600.
ARMS = ['long_gap', 'half_long_gap_half_R20']
DESIGN = dict(parent='v58', part='Rome recording classifier', input_columns=68,
    new_features=[], membership='original R20 rows with takeoff-minus-schedule >3600 seconds',
    inference_gate='Rome finite takeoff-minus-schedule >3600 seconds',
    params=source.PARAMS, rounds=1700, seed=42, threshold_sec=THRESHOLD,
    target='unchanged original R20 binary schedule-copy target',
    arms=ARMS, outside_gate='original R20 probability exactly unchanged',
    network_share=.71, selection='no parameter, threshold or stopping search',
    exploratory_reused_development_months=True)


def identity():
    return dict(design=DESIGN, runner_sha256=source.sha(Path(__file__)),
                helper_sha256=source.sha(Path(source.__file__)))


def packet(month):
    d, meta = source.load_packet(month)
    names = [c for c in d['A'] if c not in source.FEATURES]
    assert len(names) == 68 and GAP in names
    A, B = d['A'][names], d['B'][names]
    train_gap, held_gap = A[GAP].to_numpy(float), B[GAP].to_numpy(float)
    train_gate = np.isfinite(train_gap) & (train_gap > THRESHOLD)
    held_gate = np.isfinite(held_gap) & (held_gap > THRESHOLD)
    expected = {'2025-01': (14940, 1307), '2025-07': (6978, 697)}[month]
    assert (int(train_gate.sum()), int(d['target'][train_gate].sum())) == expected
    assert not np.intersect1d(d['train_ids'][train_gate], d['ids']).size
    return d, meta, A, B, train_gate, held_gate


def fit():
    import lightgbm as lgb
    current = identity()
    OUT.mkdir(parents=True, exist_ok=True)
    design_path = OUT / 'design.json'
    if design_path.exists() and source.read(design_path) != current:
        raise ValueError('Frozen specialist source/design changed')
    source.write(design_path, current)
    for month in source.MONTHS:
        path = OUT / 'predictions' / f'{month}_s42.parquet'
        record_path = path.with_suffix('.json')
        model_path = OUT / 'models' / f'{month}_s42.txt'
        d, meta, A, B, train_gate, held_gate = packet(month)
        if record_path.exists():
            old = source.read(record_path)
            assert old['identity'] == current and old['input_manifest'] == meta
            assert source.sha(path) == old['prediction_sha256']
            assert source.sha(model_path) == old['model_sha256']
            source.log(f'{month}: existing guarded fit; skip')
            continue
        if path.exists() or model_path.exists():
            raise ValueError('Unrecorded artifacts; inspect interrupted fit')
        started = time.monotonic()
        source.log(f'{month}: original68 R20 specialist, {int(train_gate.sum()):,} rows')
        dataset = lgb.Dataset(A.loc[train_gate], d['target'][train_gate],
                              categorical_feature=d['categoricals'])
        model = lgb.train(source.PARAMS, dataset, num_boost_round=DESIGN['rounds'])
        full_gate = np.zeros(len(d['ids']), dtype=bool)
        full_gate[np.flatnonzero(d['use'])] = held_gate
        probability = np.full(len(d['ids']), np.nan)
        probability[full_gate] = model.predict(B.loc[held_gate], num_threads=8)
        assert np.isfinite(probability[full_gate]).all()
        assert ((probability[full_gate] >= 0) & (probability[full_gate] <= 1)).all()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        pd.DataFrame(dict(mvt_id=d['ids'], gate=full_gate, p=probability)).to_parquet(temporary, index=False)
        temporary.replace(path)
        source.write(record_path, dict(identity=current, input_manifest=meta, columns=list(A),
            categoricals=d['categoricals'], training_rows=int(train_gate.sum()),
            positives=int(d['target'][train_gate].sum()), held_gate_rows=int(full_gate.sum()),
            model_path=str(model_path.relative_to(ROOT)), model_sha256=source.sha(model_path),
            prediction_sha256=source.sha(path), hostname=socket.gethostname(),
            lightgbm_version=lgb.__version__, minutes=(time.monotonic()-started)/60))
        source.log(f'{month}: saved fixed specialist; {int(full_gate.sum()):,} eligible held rows')
        del d, A, B, dataset, model
        gc.collect()


def score():
    sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline')]
    import open_meteo_jan_jul as baseline
    import hpo_jan_jul as hpo
    import tuning_seasonal as ts
    sc, original, nn, trees = baseline.baseline()
    errors = {name: {} for name in ['control', *ARMS]}
    rows = []
    for month in source.MONTHS:
        path = OUT/'predictions'/f'{month}_s42.parquet'
        record = source.read(path.with_suffix('.json'))
        assert record['identity'] == identity() and source.sha(path) == record['prediction_sha256']
        frame = pd.read_parquet(path)
        np.testing.assert_array_equal(frame.mvt_id, sc.d[month]['ids'])
        raw, V = sc.d[month]['held']['raw'], sc.d[month]['held']['V']
        gap = V[GAP].to_numpy(float)
        expected = raw.ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(gap) & (gap > THRESHOLD)
        np.testing.assert_array_equal(frame.gate, expected)
        full = original[month].copy()
        full[expected] = frame.p.to_numpy()[expected]
        probabilities = dict(control=original[month], long_gap=full,
                             half_long_gap_half_R20=.5*full+.5*original[month])
        for arm, p in probabilities.items():
            np.testing.assert_array_equal(p[~expected], original[month][~expected])
            errors[arm][month] = np.mean([sc.se(month, seed, nn=nn[month], p=p,
                **{hpo.MEMBER_OF[part]: trees[part][seed][month] for part in trees})
                for seed in hpo.SEEDS], axis=0)
            if arm != 'control':
                np.testing.assert_array_equal(errors[arm][month][~expected], errors['control'][month][~expected])
        rows.append(pd.DataFrame(dict(mvt_id=frame.mvt_id, month=month, day=sc.d[month]['day'],
            gate=expected, **{arm: errors[arm][month] for arm in errors})))
    frame = pd.concat(rows, ignore_index=True)
    scores = {arm: sc.cv(value) for arm, value in errors.items()}
    assert abs(scores['control']['cv']-295.7414415741871) < 1e-7
    comparisons = {arm: ts.paired_bootstrap(frame.day, frame.month, frame.control, frame[arm]) for arm in ARMS}
    result = dict(identity=identity(), scores=scores, comparisons=comparisons,
        gate_rows=int(frame.gate.sum()), outside_gate_exact=True,
        machine_caveat='Specialist host is recorded; original R20 control fit on desktop.')
    frame.to_parquet(OUT/'scored_errors.parquet', index=False)
    source.write(OUT/'result.json', result)
    source.log(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    globals()[parser.parse_args().stage]()
