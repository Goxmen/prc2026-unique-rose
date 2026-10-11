"""Label-informed constant-offset opportunity on retained candidate CV.

This is an optimistic descriptive bound within four fixed offset families.
It is not honest validation, a fitted candidate, or a leaderboard forecast.
No held labels are exported as deployment parameters and no upload is made.
"""
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261011_remaining_offset_bias'
SCORED = ROOT / 'VALIDATION/runs/v3_20261011_daily_main_refinements_cv/scored.parquet'
RESULT = SCORED.with_name('result.json')
RAW = ROOT / 'VALIDATION/runs/v3_20261011_daily_ifr/rows2025.parquet'
GROUPS = dict(all='true', nonrome_missing_id="airport!='LIRF' and not has_id",
    rome_missing_id="airport='LIRF' and not has_id",
    rome_id_1h_to_4h_gap="airport='LIRF' and has_id and gap>3600 and gap<14400")


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def describe():
    assert not (OUT / 'result.json').exists(), 'Preserve completed diagnostic'
    design = dict(groups=GROUPS, sources=[art(p) for p in [Path(__file__), SCORED, RESULT, RAW]],
        label_use='held2025 labels determine descriptive best offsets; not honest validation',
        families='each mask separately; one constant shared across both months',
        objectives=['pooled MSE', 'mean of January/July RMSE'],
        bounds='unrestricted real offset; no rounding or physical constraints',
        no_candidate=True, no_submission=True, no_leaderboard_forecast=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'design.json', design)
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(SCORED)).create_view('p')
        con.read_parquet(str(RAW)).create_view('r')
        rows, unique, errors = con.execute('''select count(*),count(distinct p.mvt_id),sum(case when
            p.y is distinct from r.y or p.month is distinct from r.month or p.day is distinct from r.day
            then 1 else 0 end) from p join r using(mvt_id)''').fetchone()
        assert rows == unique == 344419 and errors == 0
        frame = con.execute('''select p.mvt_id,p.month,p.y,p.combined pred,r.airport,r.has_id,r.gap
            from p join r using(mvt_id)''').fetchdf()
        assert frame.mvt_id.is_unique
        con.register('f', frame)
        error = (frame.pred - frame.y).to_numpy(float)
        months = sorted(frame.month.unique())
        month_rows = [frame.month.eq(month).to_numpy() for month in months]
        baseline = {month: float(np.sqrt(np.mean(error[mask]**2))) for month, mask in zip(months, month_rows)}
        expected = json.loads(RESULT.read_text())['cv']['combined']
        np.testing.assert_allclose(np.mean(list(baseline.values())), expected, rtol=0, atol=1e-10)
        records = []
        for name, expression in GROUPS.items():
            selected = con.execute('select mvt_id,coalesce(' + expression + ',false) selected from f').fetchdf()
            np.testing.assert_array_equal(selected.mvt_id, frame.mvt_id)
            mask = selected.selected.to_numpy()
            summaries = []
            for month, month_mask in zip(months, month_rows):
                scope = mask & month_mask
                summaries.append(dict(month=month, n=int(month_mask.sum()), group_n=int(scope.sum()),
                    error_sum=float(error[scope].sum()), sse=float(np.sum(error[month_mask]**2))))
            def metrics(offset):
                return [np.sqrt((s['sse'] + 2*offset*s['error_sum'] + offset**2*s['group_n']) / s['n'])
                        for s in summaries]
            def derivative(offset):
                return float(np.mean([(s['error_sum'] + offset*s['group_n']) / (s['n'] * rmse)
                    for s, rmse in zip(summaries, metrics(offset))]))
            monthly_offsets = [-s['error_sum'] / s['group_n'] for s in summaries if s['group_n']]
            lo, hi = min(monthly_offsets), max(monthly_offsets)
            optimal_cv = brentq(derivative, lo, hi, xtol=1e-10) if lo < hi else lo
            pooled = -float(error[mask].mean())
            options = {}
            for label, offset in [('pooled_mse', pooled), ('mean_monthly_rmse', optimal_cv)]:
                corrected = error + mask * offset
                direct = [float(np.sqrt(np.mean(corrected[m]**2))) for m in month_rows]
                np.testing.assert_allclose(direct, metrics(offset), rtol=0, atol=1e-10)
                options[label] = dict(offset=offset, mean_cv=float(np.mean(direct)),
                    change=float(np.mean(direct) - expected), months=dict(zip(months, direct)))
            np.testing.assert_allclose(derivative(optimal_cv), 0, rtol=0, atol=1e-12)
            reduction = np.mean(error**2) - np.mean((error + mask*pooled)**2)
            np.testing.assert_allclose(reduction, mask.mean()*pooled**2, rtol=0, atol=1e-9)
            records.append(dict(group=name, n=int(mask.sum()), monthly_own_best_offsets=dict(zip(months, monthly_offsets)),
                options=options, analytic_derivative_at_best=derivative(optimal_cv),
                pooled_mse_reduction=float(reduction)))
    write_json(OUT / 'result.json', dict(design=design, rows=rows, baseline=baseline, cv=expected, groups=records,
        audits=dict(all_truth_months_and_ids_exact=True, expanded_and_rowwise_scores_equal=True,
            pooled_gain_identity=True, mean_rmse_stationary_point=True),
        interpretation='All bounds use held labels and unrestricted offsets; no expected test gain follows.'))
    print(json.dumps(dict(cv=expected, groups=[dict(group=r['group'], n=r['n'],
        offset=r['options']['mean_monthly_rmse']['offset'],
        descriptive_best_change=r['options']['mean_monthly_rmse']['change']) for r in records]), indent=2))


if __name__ == '__main__':
    with threadpool_limits(1): describe()
