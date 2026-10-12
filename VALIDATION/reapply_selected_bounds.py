"""Fixed diagnostic: restore already-selected plan bounds after v79's affine mix.

No monthly recalibration, new rule, fitted threshold, version or upload. Reports
the original Jan/Jul comparison and ranking scope. The recovered-plan exclusion
is nonconvex and cannot promise a squared-error improvement from membership alone.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_delay_reconcile as base
import public_constraints_candidate as constraints
import recovered_plan_exclusion_fixed as exclusion
import tuning_seasonal as ts
from records import sha256, write_json

ROOT = base.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_reapply_selected_bounds'
PARENT = ROOT / 'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'
DESIGN = dict(parent='v79', primary='matched-plan bound followed by recovered-plan exclusion',
    settings='Reuse selected v74 functions, constants, source matches and donor fold restrictions exactly',
    no_new_fit=True, no_monthly_recalibration=True, no_upload=True,
    secondary='Matched bound alone describes the nonconvex exclusion contribution; not a threshold search',
    caveat='Reused development data; nonconvex support does not guarantee squared-error improvement')


def run():
    assert not OUT.exists(), 'Preserve completed experiment'
    parent = json.loads((PARENT / 'manifest.json').read_text())
    assert parent['state'] == 'scored' and parent['score'] == 219.613
    file = ROOT / parent['file']
    assert sha256(file) == parent['sha256'] == parent['remote_sha256']
    paths = [Path(__file__), Path(constraints.__file__), Path(exclusion.__file__),
             PARENT / 'cv.parquet', file, base.OUT / 'rows2025.parquet',
             base.OUT / 'ranking_inputs.parquet', exclusion.OUT / 'matches2025.parquet',
             exclusion.OUT / 'ranking_matches.parquet']
    OUT.mkdir(parents=True)
    write_json(OUT / 'design.json', dict(design=DESIGN, sources=[base.artifact(p) for p in paths]))
    cv = pd.read_parquet(PARENT / 'cv.parquet')
    rows = pd.read_parquet(base.OUT / 'rows2025.parquet',
        columns=['mvt_id', 'y', 'nm_gap', 'has_id']).set_index('mvt_id').loc[cv.mvt_id].reset_index()
    np.testing.assert_array_equal(rows.y, cv.y)
    frame = cv[['mvt_id', 'y', 'month', 'day']].copy()
    frame['parent'] = cv.pred
    frame['matched'] = constraints.matched_bound(rows, cv.pred)
    pred = frame.matched.to_numpy().copy()
    matches = pd.read_parquet(exclusion.OUT / 'matches2025.parquet')
    for month, allowed in base.FOLDS.items():
        ix = np.flatnonzero(cv.month.eq(month))
        pred[ix], _ = exclusion.apply(cv.mvt_id.to_numpy()[ix], pred[ix],
            matches.loc[matches.donor_month.isin(allowed + [month])])
    frame['candidate'] = np.rint(np.maximum(pred, 0))
    frame['matched'] = np.rint(np.maximum(frame.matched, 0))
    frame.to_parquet(OUT / 'cv.parquet', index=False)
    months = {month: {name: float(np.sqrt(np.mean((q[name] - q.y)**2)))
        for name in ['parent', 'matched', 'candidate']} for month, q in frame.groupby('month')}
    paired = ts.paired_bootstrap(frame.day, frame.month,
        (frame.parent - frame.y)**2, (frame.candidate - frame.y)**2)
    submitted = pd.read_parquet(file)
    ranking = pd.read_parquet(base.OUT / 'ranking_inputs.parquet').set_index('mvt_id')
    rank_rows = ranking.loc[submitted.MVT_ID_mvt].reset_index()
    raw = submitted.TAXITIME_SEC_mvt.to_numpy(float)
    bounded = constraints.matched_bound(rank_rows, raw)
    final, audit = exclusion.apply(submitted.MVT_ID_mvt.to_numpy(), bounded,
        pd.read_parquet(exclusion.OUT / 'ranking_matches.parquet'))
    final = np.rint(np.maximum(final, 0))
    rank = pd.DataFrame(dict(mvt_id=submitted.MVT_ID_mvt, parent=raw,
        matched=bounded, candidate=final))
    rank.to_parquet(OUT / 'ranking.parquet', index=False)
    audit.to_parquet(OUT / 'exclusion_audit.parquet', index=False)
    observed = rows.has_id.to_numpy(bool) & np.isfinite(rows.nm_gap.to_numpy(float))
    truth_outside = observed & ((rows.y < np.maximum(0, rows.nm_gap - 3606)) |
        (rows.y > np.maximum(0, rows.nm_gap + 3606)))
    result = dict(design=DESIGN, months=months,
        scores={name: float(np.mean([v[name] for v in months.values()]))
                for name in ['parent', 'matched', 'candidate']}, paired=paired,
        cv_changed=int((frame.candidate != frame.parent).sum()),
        matched_cv_changed=int((frame.matched != frame.parent).sum()),
        held_truth_outside_matched_bound=int(truth_outside.sum()),
        ranking_changed=int((final != raw).sum()),
        ranking_matched_changes=int((bounded != raw).sum()),
        ranking_exclusion_changes=int((final != np.rint(bounded)).sum()),
        ranking_rms_change=float(np.sqrt(np.mean((final - raw)**2))),
        ranking_matched_distance_mse=float(np.mean((bounded - raw)**2)),
        caveat=DESIGN['caveat'], artifacts=[base.artifact(OUT / p) for p in
            ['cv.parquet', 'ranking.parquet', 'exclusion_audit.parquet', 'design.json']])
    write_json(OUT / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
