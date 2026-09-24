"""Stack v38's recording-artifact corrections on the occupancy run: row 43 with and without occupancy.

Nothing is refitted. The occupancy run already holds row 43's row-34 predictions with
and without the four taxi occupancy inputs; this adds the three corrections that v38
layers on row 34 (twin records, day-shift blend, 12-hour clock anchor, plus the
inactive confidence exception), giving four columns per fold and seed:

    pred34        row 43's model at row 34             (control)
    pred34_occ    the same with the occupancy inputs
    pred37        row 43                                (control)
    pred37_occ    row 43 + occupancy

The corrections come from `recording_artifacts.predict` with the parameters fitted in
run `v3_20260919_artifact_experiments`, exactly as for row 43.

Score with

    .venv/bin/python VALIDATION/compare.py --run <this run> \
        --prediction-column pred37_occ --baseline-prediction-column pred37

Run: .venv/bin/python VALIDATION/occupancy_combined_run.py --source-run <occupancy run>
"""

import argparse
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
import recording_artifacts as corrections
from records import RUNS, append_log, sha256, source_snapshot, write_json
from reporting import report

SOURCE_RUN = RUNS / 'v3_20260919T211914721668Z'
PARAMETER_RUN = RUNS / 'v3_20260919_artifact_experiments'
CONTROLS = {'pred34': ('p_control', 'pred37'), 'pred34_occ': ('p_occ', 'pred37_occ')}
CANDIDATES = {'combined': 'pred37', 'combined_occ': 'pred37_occ'}
KEEP = ['mvt_id', 'mvt_time', 'airport', 'y', 'has_flight_plan', 'arrival_history_available',
        'arrival_history_days_available', 'raw_flight_id_missing', 'fold', 'month', 'seed', 'rule_triggered']
SOURCES = ['VALIDATION/occupancy_combined_run.py', 'VALIDATION/taxi_occupancy.py', 'VALIDATION/recording_artifacts.py',
           'VALIDATION/record_fixes.py', 'VALIDATION/plan_window.py']


def departures():
    """The observable columns the corrections read, for every 2025 departure."""
    frame = duckdb.sql(
        "SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, FLIGHT_mvt, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt, "
        "SCHED_TIME_UTC_mvt, LOBT_flt, AOBT_3_flt FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'",
        params=[sorted(glob.glob(TRAIN_GLOB))]).df()
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('2025 departures must have unique movement IDs')
    # Never used: in every window the handler groups fall back to the pooled day weight.
    for column in corrections.SIGNALS:
        frame[column] = np.nan
    return frame.set_index('mvt_id')


def parameters_by_window(source):
    saved = json.loads((PARAMETER_RUN / 'result.json').read_text())
    parameters = saved['correction_parameters']
    for window, values in parameters.items():
        if values['day']['handler_stratified']:
            raise ValueError(f'{window}: handler groups are stratified, so the signals would matter here')
        late = source['parameters'][window]['late_mean']
        if values['late_mean'] != late:
            raise ValueError(f'{window}: correction late mean {values["late_mean"]} differs from {late}')
    return parameters


def derive(frame, raw, parameters):
    """Row 37 with and without OPDI, from row 34 with and without OPDI."""
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('OOF movements must be present and unique within a fold/seed')
    if not np.array_equal(pd.to_datetime(frame.mvt_time, utc=True).to_numpy(),
                          pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).to_numpy()):
        raise ValueError('Saved takeoff times differ from the raw departures')
    context = raw.reset_index()
    out = frame.loc[:, KEEP + list(CONTROLS)].copy()
    moved = {}
    for control, (probability, column) in CONTROLS.items():
        # Twins are found among this validation month's rows, as in the v38 study.
        predictions, masks = corrections.predict(frame[control].to_numpy(float), context, parameters,
                                                 probability=frame[probability].to_numpy(float))
        out[column] = predictions['pred_combined']
        moved[column] = int((out[column].to_numpy() != frame[control].to_numpy()).sum())
    return out, moved


def main(argv=None):
    global SOURCE_RUN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run', help='occupancy run whose row-34 predictions are corrected '
                                         '(default: the run this script was frozen against)')
    ap.add_argument('--label', default='occupancy-combined')
    args = ap.parse_args(argv)
    if args.source_run:
        SOURCE_RUN = RUNS / args.source_run
    started = time.monotonic()
    source = json.loads((SOURCE_RUN / 'result.json').read_text())
    if source['status'] != 'complete' or source['config']['candidates'].get('occ') != 'pred_occ':
        raise ValueError(f'{SOURCE_RUN.name} is not the completed occupancy run')
    parameters = parameters_by_window(source)
    dep = departures()
    at = datetime.now(timezone.utc)
    run_id = 'v3_' + at.strftime('%Y%m%dT%H%M%S%fZ')
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True)
    moved = {}
    for fold, info in source['folds'].items():
        window = parameters[info['model_id']]
        for seed, name in info['predictions'].items():
            frame = pd.read_parquet(SOURCE_RUN / name)
            part, counts = derive(frame, dep.reindex(frame.mvt_id), window)
            part.to_parquet(out_dir / name, index=False)
            moved[f'{fold}/{seed}'] = counts
        print(f"Derived {fold} ({info['val']})", flush=True)
    totals = {column: sum(m[column] for m in moved.values()) for column in CANDIDATES.values()}
    record = copy.deepcopy(source)
    record.update(run_id=run_id, label=args.label, at=at.isoformat(timespec='seconds'),
                  prediction_column='pred34', models={}, official_rmse=None, baseline_id=None,
                  derived_from=dict(run=SOURCE_RUN.name, columns=list(CONTROLS)),
                  correction_parameters=parameters, correction_parameter_run=PARAMETER_RUN.name,
                  signature={name: sha256(ROOT / name) for name in SOURCES},
                  source_signature=source['signature'], source=source_snapshot(out_dir),
                  rows_moved=dict(totals=totals, by_fold=moved))
    record['config'] = dict(record['config'], label=args.label,
                            candidates=dict(CANDIDATES, control='pred34', opdi='pred34_opdi'),
                            recording_artifact_settings=corrections.SETTINGS)
    record['report'] = report(record, out_dir)
    record['minutes'] = round((time.monotonic() - started) / 60, 3)
    write_json(out_dir / 'result.json', record)
    append_log(record)
    print(f'Wrote {out_dir.relative_to(ROOT)}; fold/seed rows moved: {json.dumps(totals)}', flush=True)


if __name__ == '__main__':
    main()
