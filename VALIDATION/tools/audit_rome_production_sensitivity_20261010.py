"""Describe fixed CatBoost/R20 probability changes; never alter predictions.

The inherited p<0.05 exception can change a multi-hour hard override. Count
boundary crossings before candidate construction, using inputs only. This is
not a hidden-label assessment, a new gate, or a calibration/weight search.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAT = ROOT / 'VALIDATION/runs/v3_20261010_rome_catboost_submit'
OUT = ROOT / 'tmp/research_20261010/rome_production_sensitivity.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    result = dict(source_sha256=sha(Path(__file__)), inputs={}, views={},
                  note='Input-only sensitivity audit; no labels, tuning or prediction changes.')
    for view in ['old', 'final']:
        packet_path = CAT / 'inputs' / f'{view}.pkl'
        record_path = CAT / f'{view}.json'
        record = json.loads(record_path.read_text())
        cat_path = ROOT / record['prediction']['path']
        assert sha(cat_path) == record['prediction']['sha256']
        packet = pd.read_pickle(packet_path)
        frame = pd.read_parquet(cat_path)
        if view == 'old':
            directory = ROOT / 'VALIDATION/runs/v3_20261006_hpo_submit/submission_v54'
            paths = [directory / f'ranking_2026-{m}.parquet' for m in ['01', '07']]
        else:
            paths = [ROOT / 'VALIDATION/runs/v3_20261008_final_submit/work/parts/rome.parquet']
        parent = pd.concat([pd.read_parquet(p, columns=['mvt_id', 'p']) for p in paths], ignore_index=True)
        np.testing.assert_array_equal(frame.mvt_id, packet['ids'])
        np.testing.assert_array_equal(parent.mvt_id, packet['ids'])
        use = np.asarray(packet['use'], bool)
        X = packet['X']
        assert use.sum() == len(X) and len(X.columns) == 68
        old = parent.p.to_numpy(float)[use]
        new = .5 * old + .5 * frame.p.to_numpy(float)[use]
        gap = X.mvt_minus_sched_sec.to_numpy(float)
        band = X.clock_raw_flight_id_missing.eq(1).to_numpy() & (gap >= 14400) & (gap < 43200)
        crossed = band & ((old < .05) != (new < .05))
        month_results = {}
        for month in sorted(X.month.unique()):
            scope = X.month.eq(month).to_numpy()
            month_results[str(int(month))] = dict(rows=int(scope.sum()),
                mean_p_change=float((new-old)[scope].mean()),
                mean_absolute_p_change=float(np.abs(new-old)[scope].mean()),
                exception_band_rows=int((scope & band).sum()),
                parent_exception=int((scope & band & (old < .05)).sum()),
                candidate_exception=int((scope & band & (new < .05)).sum()),
                threshold_crossings=int((scope & crossed).sum()))
        result['views'][view] = dict(months=month_results, crossings=[
            dict(mvt_id=int(packet['ids'][use][i]), month=int(X.month.iloc[i]),
                 gap_sec=float(gap[i]), old_p=float(old[i]), new_p=float(new[i]))
            for i in np.flatnonzero(crossed)])
        for path in [packet_path, record_path, cat_path, *paths]:
            result['inputs'][str(path.relative_to(ROOT))] = sha(path)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['views'], indent=2))


if __name__ == '__main__':
    main()
