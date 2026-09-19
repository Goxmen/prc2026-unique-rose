"""Validate rows 34-36 on row 33's saved predictions: no-ID cap, twin records, seed average.

Nothing is refitted. Each candidate changes the `pred_window` column of run
v3_20260918T181924599658Z (row 33, Plan-window) on every fold and seed, one change
per candidate, and the result is written as a derived v3 run with four columns:

    pred_window      row 33, unchanged (control)
    pred_noid_cap    row 34: record_fixes.noid_cap
    pred_twins       row 35: record_fixes.twin_records
    pred_seed_avg    row 36: each flight's mean of the three seeds' pred_window in that fold

Twins are found among all 2025 departures, not only the rows of one fold. Score each
candidate against the control with

    .venv/bin/python VALIDATION/compare.py --run <this run> \
        --prediction-column pred_noid_cap --baseline-prediction-column pred_window

Run: .venv/bin/python VALIDATION/record_fixes_run.py
"""

import copy
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import TRAIN_GLOB
import record_fixes
from records import RUNS, append_log, sha256, source_snapshot, write_json
from reporting import report

SOURCE_RUN = RUNS / 'v3_20260918T181924599658Z'
SOURCE_COLUMN = 'pred_window'
CANDIDATES = {'noid_cap': 'pred_noid_cap', 'twins': 'pred_twins', 'seed_avg': 'pred_seed_avg'}
KEEP = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan', 'arrival_history_available',
        'arrival_history_days_available', 'raw_flight_id_missing', 'fold', 'month', 'seed', 'rule_triggered']
SOURCES = ['VALIDATION/record_fixes.py', 'VALIDATION/record_fixes_run.py']


def departures():
    """Columns the two record fixes read, for every 2025 departure, one row per movement ID; plus the twin mask."""
    frame = duckdb.sql(
        "SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, FLIGHT_mvt, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt, "
        "LOBT_flt FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'", params=[sorted(glob.glob(TRAIN_GLOB))]).df()
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('2025 departures must have unique movement IDs')
    frame['twin'] = record_fixes.twin_rows(frame)
    return frame.set_index('mvt_id')


def derive(frames, dep):
    """The three seeds' files of one fold, each with the control and the three candidate columns."""
    mean = pd.concat(frames.values()).groupby('mvt_id')[SOURCE_COLUMN].mean()
    out = {}
    for seed, frame in frames.items():
        raw = dep.reindex(frame.mvt_id)
        if raw.MVT_TIME_UTC_mvt.isna().any():
            raise ValueError('Saved predictions include movements missing from the 2025 departures')
        if not np.array_equal(pd.to_datetime(frame.mvt_time, utc=True).to_numpy(),
                              pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).to_numpy()):
            raise ValueError('Saved takeoff times differ from the raw departures')
        pred = frame[SOURCE_COLUMN].to_numpy(float)
        part = frame.loc[:, KEEP + [SOURCE_COLUMN]].copy()
        part[CANDIDATES['noid_cap']] = record_fixes.noid_cap(pred, raw)
        part[CANDIDATES['twins']] = record_fixes.twin_records(pred, raw, mask=raw.twin.to_numpy(bool))
        part[CANDIDATES['seed_avg']] = mean.reindex(frame.mvt_id).to_numpy()
        out[seed] = part
    return out


def changes(frame, rule):
    y, control = frame.y.to_numpy(), frame[SOURCE_COLUMN].to_numpy()
    result = {}
    for name, column in CANDIDATES.items():
        pred = frame[column].to_numpy()
        moved = pred != control
        if name != 'seed_avg' and (moved & rule).any():
            raise ValueError(f'{name}: a Rome rule row moved')
        result[name] = dict(moved=int(moved.sum()),
                            closer=int((moved & (np.abs(pred - y) < np.abs(control - y))).sum()),
                            farther=int((moved & (np.abs(pred - y) > np.abs(control - y))).sum()))
    return result


def main():
    started = time.monotonic()
    source = json.loads((SOURCE_RUN / 'result.json').read_text())
    if source['status'] != 'complete' or source.get('prediction_column') != SOURCE_COLUMN:
        raise ValueError(f'{SOURCE_RUN.name} is not the completed row 33 run')
    dep = departures()
    print(f'2025 departures: {len(dep):,}; twin flight-ID rows: {int(dep.twin.sum())}', flush=True)
    at = datetime.now(timezone.utc)
    run_id = 'v3_' + at.strftime('%Y%m%dT%H%M%S%fZ')
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True)
    record = copy.deepcopy(source)
    moved = {}
    for fold, info in record['folds'].items():
        frames = {seed: pd.read_parquet(SOURCE_RUN / name) for seed, name in info['predictions'].items()}
        for seed, frame in derive(frames, dep).items():
            moved[f'{fold}/{seed}'] = changes(frame, frame.rule_triggered.to_numpy())
            frame.to_parquet(out_dir / info['predictions'][seed], index=False)
    totals = {name: {k: sum(m[name][k] for m in moved.values()) for k in ('moved', 'closer', 'farther')}
              for name in CANDIDATES}
    record.update(run_id=run_id, label='record-fixes', at=at.isoformat(timespec='seconds'),
                  prediction_column=SOURCE_COLUMN, derived_from=dict(run=SOURCE_RUN.name, column=SOURCE_COLUMN),
                  signature={name: sha256(ROOT / name) for name in SOURCES},
                  source_signature=source['signature'], source=source_snapshot(out_dir),
                  record_fixes=dict(constants={k: getattr(record_fixes, k) for k in [
                      'LATE_GAP_SEC', 'LATE_CAP_SEC', 'GAP_CAP_MIN_SEC', 'CAP_EXCLUDED', 'LATE_CAP_EXCLUDED',
                      'TWIN_WINDOW_SEC', 'TWIN_OFFSET_SEC']}, twin_rows_2025=int(dep.twin.sum()),
                      totals=totals, changes=moved),
                  models={}, official_rmse=None, baseline_id=None)
    record['config'] = dict(record['config'], label='record-fixes',
                            candidates=dict(CANDIDATES, control=SOURCE_COLUMN))
    record['report'] = report(record, out_dir)
    record['minutes'] = round((time.monotonic() - started) / 60, 3)
    write_json(out_dir / 'result.json', record)
    append_log(record)
    print(f'Wrote {out_dir.relative_to(ROOT)}; fold/seed rows moved: {json.dumps(totals)}', flush=True)


if __name__ == '__main__':
    main()
