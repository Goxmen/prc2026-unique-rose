"""Validate row 37 on row 34's saved predictions: the in-between guess of dayshift_hedge.py.

Nothing is refitted. The control is the `pred_noid_cap` column (row 34, No-ID-cap) of
run v3_20260919T104157403241Z; the candidate `pred_dayshift` applies
dayshift_hedge.hedge to it, with the airline and format rates and the wrong-date taxi
read from the labels of each fold's own training window (train_start..train_end, by
takeoff time) only. Score the pair with

    .venv/bin/python VALIDATION/compare.py --run <this run> \
        --prediction-column pred_dayshift --baseline-prediction-column pred_noid_cap

Run: .venv/bin/python VALIDATION/dayshift_hedge_run.py
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
import dayshift_hedge
from records import RUNS, append_log, sha256, source_snapshot, write_json
from reporting import report

SOURCE_RUN = RUNS / 'v3_20260919T104157403241Z'
SOURCE_COLUMN = 'pred_noid_cap'
COLUMN = 'pred_dayshift'
KEEP = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan', 'arrival_history_available',
        'arrival_history_days_available', 'raw_flight_id_missing', 'fold', 'month', 'seed', 'rule_triggered']
SOURCES = ['VALIDATION/dayshift_hedge.py', 'VALIDATION/dayshift_hedge_run.py']


def departures():
    frame = duckdb.sql(
        "SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, FLIGHT_mvt, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt, "
        "TAXITIME_SEC_mvt AS y FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'",
        params=[sorted(glob.glob(TRAIN_GLOB))]).df()
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('2025 departures must have unique movement IDs')
    return frame.set_index('mvt_id')


def window_history(dep, info):
    """dayshift_hedge.history from the fold's training window only."""
    start = pd.Timestamp(info['train_start'], tz='UTC')
    end = pd.Timestamp(info['train_end'], tz='UTC') + pd.Timedelta(days=1)
    t = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True)
    train = dep[(t >= start) & (t < end)]
    return dayshift_hedge.history(train, train.y.to_numpy(float))


def derive(frame, dep, hist):
    raw = dep.reindex(frame.mvt_id)
    if raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError('Saved predictions include movements missing from the 2025 departures')
    if not np.array_equal(pd.to_datetime(frame.mvt_time, utc=True).to_numpy(),
                          pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).to_numpy()):
        raise ValueError('Saved takeoff times differ from the raw departures')
    out = frame.loc[:, KEEP + [SOURCE_COLUMN]].copy()
    out[COLUMN] = dayshift_hedge.hedge(frame[SOURCE_COLUMN].to_numpy(float), raw.reset_index(), hist)
    moved = out[COLUMN].to_numpy() != out[SOURCE_COLUMN].to_numpy()
    if (moved & ~out.rule_triggered.to_numpy()).any():
        raise ValueError('The hedge moved a row the Rome rule did not fire on')
    p = dayshift_hedge.weights(raw.reset_index(), hist)
    kind = dayshift_hedge.kinds(raw.reset_index(), out.y.to_numpy(float))
    rows = [dict(mvt_id=int(r.mvt_id), flight=raw.FLIGHT_mvt.iloc[i], p=round(float(p[i]), 3),
                 control=round(float(getattr(r, SOURCE_COLUMN)), 1), candidate=round(float(getattr(r, COLUMN)), 1),
                 y=float(r.y), kind=kind[i])
            for i, r in zip(np.flatnonzero(moved), out[moved].itertuples())]
    return out, rows


def main():
    started = time.monotonic()
    source = json.loads((SOURCE_RUN / 'result.json').read_text())
    if source['status'] != 'complete' or source['config']['candidates'].get('noid_cap') != SOURCE_COLUMN:
        raise ValueError(f'{SOURCE_RUN.name} is not the completed rows 34-36 run')
    dep = departures()
    at = datetime.now(timezone.utc)
    run_id = 'v3_' + at.strftime('%Y%m%dT%H%M%S%fZ')
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True)
    record = copy.deepcopy(source)
    histories, moved = {}, {}
    for fold, info in record['folds'].items():
        hist = window_history(dep, info)
        histories[fold] = hist
        for seed, name in info['predictions'].items():
            frame, rows = derive(pd.read_parquet(SOURCE_RUN / name), dep, hist)
            moved[f'{fold}/{seed}'] = rows
            frame.to_parquet(out_dir / name, index=False)
    record.update(run_id=run_id, label='dayshift-hedge', at=at.isoformat(timespec='seconds'),
                  prediction_column=COLUMN, derived_from=dict(run=SOURCE_RUN.name, column=SOURCE_COLUMN),
                  signature={name: sha256(ROOT / name) for name in SOURCES},
                  source_signature=source['signature'], source=source_snapshot(out_dir),
                  dayshift_hedge=dict(constants={k: getattr(dayshift_hedge, k) for k in [
                      'MIN_GAP_SEC', 'PLAN_MATCH_SEC', 'WRONG_DATE_MAX_TAXI_SEC', 'PRIOR_WEIGHT', 'TAXI_FALLBACK_SEC']},
                      histories=histories, moved=moved),
                  models={}, official_rmse=None, baseline_id=None)
    record['config'] = dict(record['config'], label='dayshift-hedge',
                            candidates={'dayshift': COLUMN, 'control': SOURCE_COLUMN})
    record['report'] = report(record, out_dir)
    record['minutes'] = round((time.monotonic() - started) / 60, 3)
    write_json(out_dir / 'result.json', record)
    append_log(record)
    flights = {r['mvt_id']: r for rows in moved.values() for r in rows}
    print(f'Wrote {out_dir.relative_to(ROOT)}; {sum(map(len, moved.values()))} fold/seed rows moved, '
          f'{len(flights)} distinct flights', flush=True)


if __name__ == '__main__':
    main()
