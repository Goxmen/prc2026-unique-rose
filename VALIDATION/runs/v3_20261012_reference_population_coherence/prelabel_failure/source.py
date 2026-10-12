"""Fixed source-calibration pilot; no held taxi predictions or candidates.

Does an independently cross-fitted public reference-total discrepancy explain
the measurement error in public taxi means?  This changes source calibration,
not the closed stand-feature or runway-covariance experiments.
"""
from pathlib import Path
import json
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import public_daily_taxi as daily
import public_reference_inverse_features as inverse
from records import write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261012_reference_population_coherence'
DESIGN = dict(
    hypothesis='Public reference reconstruction error identifies composition error in daily taxi totals',
    parent='Existing airport intercept/count-deficit Huber20 source calibration',
    change='One additional reference-error coefficient per airport, same Huber20/ridge0.001',
    reference_error='Leave-one-source-day-out stand-composition reference prediction minus published reference mean',
    reference_fit='Unchanged cached two-month public-only inverse, ridge(10/300)^2; exact PRESS deletion',
    interpretation='Reference times and taxi times share a source cohort; test conditional bias, not independent second evidence',
    evaluation='Leave one calendar month out inside each original outer permitted training window',
    departure_labels='Only each outer allowed training months enter calibration or source-error evaluation',
    no_held_taxi_predictions=True, no_model_fit=True, no_candidate=True, no_upload=True,
    no_search=True, no_daily_gate_change=True,
    gate='Both outer windows improve source mean-error RMSE at least5%, with paired-month bootstrap upper bound below zero',
    caveats='Source reconstruction is transductive but public-only; common source does not prove cohort identity; source gain is not taxi CV')


def fit(c, extra):
    X = daily.matrix(c)
    if extra:
        one = np.column_stack([c.airport.eq(a).to_numpy(float) for a in daily.ZONES])
        X = np.column_stack([X, one*c.reference_error.to_numpy()[:, None]/10.])
    y = (c.truth-c.source_mean).to_numpy()
    fun = lambda b: np.r_[(X@b-y)/20., np.sqrt(.001)*b/20.]
    r = least_squares(fun, np.zeros(X.shape[1]), loss='huber', f_scale=1.)
    assert r.success
    return r.x


def predict(c, coefficient):
    X = daily.matrix(c)
    if len(coefficient) == 30:
        one = np.column_stack([c.airport.eq(a).to_numpy(float) for a in daily.ZONES])
        X = np.column_stack([X, one*c.reference_error.to_numpy()[:, None]/10.])
    return X@coefficient


def reference_cells(view):
    inp = inverse.OUT/f'inputs{view}.parquet'
    observed = pd.read_parquet(inverse.OUT/f'observations{view}.parquet')
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        counts = con.execute('''select airport,runway,day,month,stand,count(*) size from read_parquet(?)
            where stand is not null and runway is not null
            and substr(utc_month,1,4)=substr(month,1,4)
            and ((cast(substr(utc_month,6,2) as integer)-1)%6)=((cast(substr(month,6,2) as integer)-1)%6)
            group by all''', [str(inp)]).df()
    observed['pair'] = (observed.month.str[-2:].astype(int)-1)%6
    parts, checks = [], []
    for (airport, runway, pair), c in observed.groupby(['airport', 'runway', 'pair']):
        c = c.sort_values('day').reset_index(drop=True)
        q = counts[counts.airport.eq(airport)&counts.runway.eq(runway)&counts.day.isin(c.day)]
        A = q.pivot(index='day', columns='stand', values='size').fillna(0).reindex(c.day).fillna(0)
        A = A.to_numpy(float)/c.metadata_n.to_numpy()[:, None]
        np.testing.assert_allclose(A.sum(1), 1., atol=1e-12, rtol=0)
        y, month = c.reference.to_numpy(), c.month.to_numpy()
        pred, _, _, _ = inverse.pilot.solve(A, y, month, np.ones(len(c), bool))
        np.testing.assert_allclose(pred, c.full_fit, atol=1e-8, rtol=0)
        G = np.column_stack([month == m for m in sorted(set(month))]).astype(float)
        P = G@np.linalg.solve(G.T@G, G.T)
        C = A-P@A
        H = P+C@C.T@np.linalg.inv(C@C.T+inverse.pilot.DESIGN['ridge']*np.eye(len(c)))
        np.testing.assert_allclose(H@y, pred, atol=1e-8, rtol=0)
        leverage = np.diag(H)
        assert np.max(leverage) < 1.-1e-10
        deleted = y+(pred-y)/(1.-leverage)
        # Direct held-row refits verify every group's first and last deletion.
        for at in sorted({0, len(c)-1}):
            train = np.arange(len(c)) != at
            brute = inverse.pilot.solve(A, y, month, train)[0][at]
            np.testing.assert_allclose(brute, deleted[at], atol=1e-7, rtol=0)
            checks.append(abs(float(brute-deleted[at])))
        c['reference_error'] = deleted-y
        c['reference_leverage'] = leverage
        parts.append(c)
    f = pd.concat(parts, ignore_index=True)
    assert not f.duplicated(['airport', 'runway', 'day']).any()
    return f, dict(groups=len(parts), direct_deletions=len(checks), max_error=max(checks))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'identity.json').exists(), 'Preserve frozen/completed source pilot'
    paths = [Path(__file__), Path(daily.__file__), Path(inverse.__file__),
        Path(inverse.pilot.__file__), daily.CACHE, daily.SOURCE/'daily_runway.parquet']
    for view in ['2025', 'ranking']:
        paths.extend(inverse.OUT/f'{part}{view}.parquet' for part in ['inputs', 'observations'])
    write_json(OUT/'identity.json', dict(design=DESIGN, sources=[daily.art(p) for p in paths],
        folds=daily.base.FOLDS, frozen_before_any_departure_outcome_read=True))
    refs, check = reference_cells('2025')
    test, test_check = reference_cells('ranking')
    refs.to_parquet(OUT/'reference2025.parquet', index=False)
    test.to_parquet(OUT/'referenceranking.parquet', index=False)
    public = daily.public()
    all_eval, coefficients, summaries = [], [], {}
    for fold, allowed in daily.base.FOLDS.items():
        with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
            train = con.execute('''select mvt_id,airport,local_takeoff local_day,runway,
                strftime(takeoff,'%Y-%m') month,y,stand is not null and runway is not null eligible
                from read_parquet(?) where strftime(takeoff,'%Y-%m') in (select unnest(?))''',
                [str(daily.CACHE), allowed]).df()
        assert set(train.month).issubset(allowed) and fold not in set(train.month)
        good = train.eligible & train.y.between(0, 7200)
        c = train[good].groupby(daily.KEY, as_index=False).agg(truth=('y', 'mean'), n=('y', 'size'))
        c = c.merge(public, on=daily.KEY, how='inner', validate='one_to_one')
        c = c[daily.supported(c)&daily.complete_days(c, allowed)].copy()
        c = c.merge(refs[['airport','runway','day','reference_error','reference_leverage']],
            left_on=daily.KEY, right_on=['airport','day','runway'], how='inner', validate='one_to_one')
        c['month'] = c.local_day.str[:7]
        assert set(c.month).issubset(allowed) and fold not in set(c.month)
        c['residual'] = c.truth-c.source_mean
        pieces = []
        for month in sorted(c.month.unique()):
            fit_rows, held = c[c.month.ne(month)], c[c.month.eq(month)].copy()
            b0, b1 = fit(fit_rows, False), fit(fit_rows, True)
            held['baseline_error'] = held.residual-predict(held, b0)
            held['conditional_error'] = held.residual-predict(held, b1)
            held['fold'] = fold
            pieces.append(held)
            coefficients.append(dict(fold=fold, held_source_month=month,
                fitting_months=sorted(fit_rows.month.unique()), baseline=b0.tolist(), conditional=b1.tolist()))
        e = pd.concat(pieces, ignore_index=True)
        def rmse(x): return float(np.sqrt(np.mean(np.square(x))))
        months = e.groupby('month').agg(n=('n','size'),
            base=('baseline_error',lambda x:float(np.square(x).sum())),
            cand=('conditional_error',lambda x:float(np.square(x).sum())))
        rng = np.random.default_rng(721)
        sampled = rng.integers(0,len(months),size=(10000,len(months)))
        a = months.to_numpy(); total = a[sampled].sum(1)
        delta = np.sqrt(total[:,2]/total[:,0])-np.sqrt(total[:,1]/total[:,0])
        r0, r1 = rmse(e.baseline_error), rmse(e.conditional_error)
        ci = np.quantile(delta,[.025,.975]).tolist()
        summaries[fold] = dict(cells=len(e), months=len(months), baseline_rmse=r0, conditional_rmse=r1,
            relative_improvement=1.-r1/r0, change=r1-r0, paired_month_ci95=ci,
            gate=bool(r1<=.95*r0 and ci[1]<0),
            residual_reference_corr=float(np.corrcoef(e.baseline_error,e.reference_error)[0,1]))
        all_eval.append(e)
    evaluations = pd.concat(all_eval,ignore_index=True)
    evaluations.to_parquet(OUT/'source_evaluation.parquet',index=False)
    write_json(OUT/'coefficients.json',coefficients)
    stats = evaluations.groupby(['fold','airport']).agg(cells=('n','size'),
        baseline_rmse=('baseline_error',lambda x:float(np.sqrt(np.square(x).mean()))),
        conditional_rmse=('conditional_error',lambda x:float(np.sqrt(np.square(x).mean()))))
    stats.to_csv(OUT/'by_airport.csv')
    result = dict(design=DESIGN, windows=summaries, gate=all(x['gate'] for x in summaries.values()),
        training_reference_checks=check, ranking_source_only_checks=test_check,
        ranking_reference_cells=len(test), ranking_labels_never_read=True,
        source_only_no_taxi_cv=True, no_candidate=True,
        artifacts=[daily.art(OUT/p) for p in ['reference2025.parquet','referenceranking.parquet',
            'source_evaluation.parquet','coefficients.json','by_airport.csv']])
    write_json(OUT/'result.json',result)
    print(json.dumps(dict(windows=summaries,gate=result['gate'],checks=check)))


if __name__ == '__main__':
    with threadpool_limits(1): main()
