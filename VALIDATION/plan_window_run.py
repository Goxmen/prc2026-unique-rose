"""Validate the flight-plan window (plan_window.py) on row 30's saved predictions.

Nothing is refitted. The window is applied to the `pred_stand` column of run
v3_20260918T151436598224Z (row 30, Stand-clock) on every fold and seed, and the
result is written as a derived v3 run with two columns: `pred_stand` (row 30,
unchanged) and `pred_window` (row 30 inside the window). Score the pair with

    .venv/bin/python VALIDATION/compare.py --run <this run> \
        --prediction-column pred_window --baseline-prediction-column pred_stand

Before writing anything it checks the rule itself on every 2025 departure with a
flight ID: no training label may fall outside its window.

Run: .venv/bin/python VALIDATION/plan_window_run.py
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
import plan_window
from records import RUNS, append_log, sha256, source_snapshot, write_json
from reporting import report

SOURCE_RUN = RUNS / 'v3_20260918T151436598224Z'
SOURCE_COLUMN = 'pred_stand'
COLUMN = 'pred_window'
KEEP = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan', 'arrival_history_available',
        'arrival_history_days_available', 'raw_flight_id_missing', 'fold', 'month', 'seed', 'rule_triggered']
SOURCES = ['VALIDATION/plan_window.py', 'VALIDATION/plan_window_run.py']


def departures():
    """Flight ID, takeoff, LOBT and label of every 2025 departure, one row per movement ID."""
    frame = duckdb.sql(
        "SELECT MVT_ID_mvt AS mvt_id, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt, LOBT_flt, TAXITIME_SEC_mvt AS y "
        "FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'", params=[sorted(glob.glob(TRAIN_GLOB))]).df()
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('2025 departures must have unique movement IDs')
    return frame.set_index('mvt_id')


def check_rule(dep):
    """Every 2025 label of a departure with a flight ID lies inside its window."""
    low, high = plan_window.window(dep)
    y = dep.y.to_numpy(float)
    bounded = np.isfinite(low)
    outside = int(((y < low) | (y > high))[bounded].sum())
    if outside:
        raise ValueError(f'{outside} training labels fall outside the flight-plan window')
    centre = (low + plan_window.MARGIN_SEC)[bounded]
    return dict(departures=len(dep), bounded=int(bounded.sum()), outside=outside,
                max_abs_label_minus_centre_min=round(float(np.abs(y[bounded] - centre).max() / 60), 2))


def derive(frame, dep):
    """One saved fold/seed file with row 30's column and the windowed column."""
    raw = dep.reindex(frame.mvt_id)
    if raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError('Saved predictions include movements missing from the 2025 departures')
    if not np.array_equal(pd.to_datetime(frame.mvt_time, utc=True).to_numpy(),
                          pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).to_numpy()):
        raise ValueError('Saved takeoff times differ from the raw departures')
    out = frame.loc[:, KEEP + [SOURCE_COLUMN]].copy()
    out[COLUMN] = plan_window.apply(frame[SOURCE_COLUMN], raw.reset_index())
    return out


def main():
    started = time.monotonic()
    source = json.loads((SOURCE_RUN / 'result.json').read_text())
    if source['status'] != 'complete' or source['config']['candidates'].get('stand') != SOURCE_COLUMN:
        raise ValueError(f'{SOURCE_RUN.name} is not the completed row 30 run')
    dep = departures()
    rule = check_rule(dep)
    print(f"Rule check: {rule['bounded']:,} of {rule['departures']:,} departures bounded, "
          f"{rule['outside']} labels outside, largest |label - centre| {rule['max_abs_label_minus_centre_min']} min",
          flush=True)
    at = datetime.now(timezone.utc)
    run_id = 'v3_' + at.strftime('%Y%m%dT%H%M%S%fZ')
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True)
    record = copy.deepcopy(source)
    changes = {}
    for fold, info in record['folds'].items():
        for seed, name in info['predictions'].items():
            frame = derive(pd.read_parquet(SOURCE_RUN / name), dep)
            moved = frame[COLUMN].to_numpy() != frame[SOURCE_COLUMN].to_numpy()
            closer = (np.abs(frame[COLUMN] - frame.y) < np.abs(frame[SOURCE_COLUMN] - frame.y)).to_numpy()
            if (moved & ~closer).any() or (moved & frame.rule_triggered.to_numpy()).any():
                raise ValueError(f'{fold}/{seed}: a moved prediction did not get closer, or a Rome rule row moved')
            changes[f'{fold}/{seed}'] = dict(rows=len(frame), moved=int(moved.sum()),
                                             raised=int((frame[COLUMN] > frame[SOURCE_COLUMN]).sum()))
            frame.to_parquet(out_dir / name, index=False)
    record.update(run_id=run_id, label='plan-window', at=at.isoformat(timespec='seconds'),
                  prediction_column=COLUMN, derived_from=dict(run=SOURCE_RUN.name, column=SOURCE_COLUMN),
                  signature={name: sha256(ROOT / name) for name in SOURCES},
                  source_signature=source['signature'], source=source_snapshot(out_dir),
                  plan_window=dict(margin_sec=plan_window.MARGIN_SEC, rule_check=rule, changes=changes),
                  models={}, official_rmse=None, baseline_id=None)
    record['config'] = dict(record['config'], label='plan-window',
                            candidates={'window': COLUMN, 'control': SOURCE_COLUMN})
    record['report'] = report(record, out_dir)
    record['minutes'] = round((time.monotonic() - started) / 60, 3)
    write_json(out_dir / 'result.json', record)
    append_log(record)
    moved = sum(c['moved'] for c in changes.values())
    print(f'Wrote {out_dir.relative_to(ROOT)}: {moved} of {sum(c["rows"] for c in changes.values()):,} '
          f'fold/seed rows moved; scores {json.dumps({k: round(v, 4) for k, v in record["report"]["scores"].items() if k in ("R_A", "R_B_long", "R_combined")})}',
          flush=True)


if __name__ == '__main__':
    main()
