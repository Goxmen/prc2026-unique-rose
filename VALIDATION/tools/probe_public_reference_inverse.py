"""Label-free pilot: recover stand contrasts from runway-day reference means.

This tests prediction of held PUBLIC reference means, not competition taxi CV.
Reference times change monthly; separate month intercepts absorb common drift.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import public_daily_taxi as source
from records import write_json

OUT = ROOT/'VALIDATION/runs/v3_20261011_public_reference_inverse_pilot'
DESIGN = dict(hypothesis='monthly stand/runway reference contrasts can be recovered from public daily aggregate compositions',
    views=['2025 Jan/Jul metadata', 'original2026 Jan/Jul ranking metadata'],
    source='published daily reference taxi total divided by source count, minutes to seconds',
    metadata='stand/runway present; UTC and local month bothJan/Jul; first local day excluded as incomplete',
    observation='day/runway with20 metadata rows and20 source rows; same0.5..1.2 count-ratio range',
    model='stand fractional composition plus separate local-month intercepts; zero-centered stand ridge',
    ridge=(10./300.)**2, rationale='fixed10s aggregate-noise versus300s stand prior; no tuning',
    public_holdout='calendar day divisible by4; fit all other days; month intercepts fitted on source-training days only',
    comparison='monthly runway mean from the same source-training days',
    no_departure_label_access=True, no_taxi_model_fit=True, no_taxi_cv_claim=True, no_production=True,
    limitation='source cohort differs; stand coefficients underidentified; monthly drift beyond intercept remains')


def solve(A, target, month, train, ridge=DESIGN['ridge']):
    periods = sorted(set(month))
    train = np.asarray(train, bool)
    centered = A[train].copy()
    response = target[train].copy()
    means = {}
    for period in periods:
        selected = month[train] == period
        assert selected.sum() >= 3
        xm, ym = centered[selected].mean(axis=0), float(response[selected].mean())
        centered[selected] -= xm
        response[selected] -= ym
        means[period] = (xm, ym)
    # Dual solve keeps this at most62x62, even with hundreds of stands.
    beta = centered.T@np.linalg.solve(centered@centered.T+ridge*np.eye(train.sum()), response)
    baseline = np.array([means[m][1] for m in month])
    intercept = {m: y-x@beta for m, (x, y) in means.items()}
    prediction = A@beta+np.array([intercept[m] for m in month])
    return prediction, baseline, beta, intercept


def checks():
    rng = np.random.default_rng(41)
    A = rng.dirichlet(np.ones(7), size=24)
    month = np.array(['A']*12+['B']*12)
    beta = np.arange(7)*30.
    target = A@beta+np.where(month == 'A', 300., 350.)
    train = np.arange(24)%4 != 0
    actual, _, coef, intercept = solve(A, target, month, train)
    period = np.column_stack([month == m for m in ['A', 'B']]).astype(float)
    X = np.column_stack([A, period])
    penalty = np.diag([DESIGN['ridge']]*7+[0., 0.])
    direct = np.linalg.solve(X[train].T@X[train]+penalty, X[train].T@target[train])
    np.testing.assert_allclose(actual, X@direct, atol=1e-9, rtol=0)
    poisoned = target.copy(); poisoned[~train] += 1e9
    np.testing.assert_array_equal(solve(A, poisoned, month, train)[0], actual)
    permutation = rng.permutation(24)
    np.testing.assert_allclose(solve(A[permutation], target[permutation], month[permutation], train[permutation])[0], actual[permutation], atol=1e-9, rtol=0)
    return dict(independent_primal_dual_equivalence=True, held_source_targets_cannot_affect_fit=True, permutation=True)


def metadata(view):
    if view == '2025':
        f = pd.read_parquet(source.CACHE, columns=['mvt_id', 'airport', 'takeoff', 'runway', 'stand'])
    else:
        f = pd.read_parquet(ROOT/'data/ranking.parquet',
            columns=['MVT_ID_mvt', 'ADEP_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt', 'STAND_mvt'],
            filters=[('PHASE_mvt', '==', 'DEP')]).rename(columns=dict(MVT_ID_mvt='mvt_id', ADEP_mvt='airport',
                MVT_TIME_UTC_mvt='takeoff', RUNWAY_mvt='runway', STAND_mvt='stand'))
    assert f.mvt_id.is_unique
    takeoff = pd.to_datetime(f.takeoff, utc=True)
    f = f.loc[takeoff.dt.month.isin([1, 7])].copy()
    f['day'] = ''
    for airport, zone in source.ZONES.items():
        mask = f.airport.eq(airport)
        f.loc[mask, 'day'] = pd.to_datetime(f.loc[mask, 'takeoff'], utc=True).dt.tz_convert(zone).dt.strftime('%Y-%m-%d')
    f = f[f.day.str[5:7].isin(['01', '07']) & f.stand.notna() & f.runway.notna()].copy()
    assert f.day.ne('').all()
    return f[['mvt_id', 'airport', 'day', 'runway', 'stand']]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'result.json').exists(), 'Preserve completed pilot'
    assert source.read(source.REPORT/'public_daily_taxi_source_audit.json')['status'] == 'pass'
    identity = dict(design=DESIGN, checks=checks(), sources=[source.art(p) for p in [Path(__file__),
        source.CACHE, ROOT/'data/ranking.parquet', source.SOURCE/'daily_runway.parquet',
        ROOT/'tmp/research_20261011/daily_taxi_methodology.pdf']])
    write_json(OUT/'identity.json', identity)
    published = pd.read_parquet(source.SOURCE/'daily_runway.parquet')
    published['reference'] = 60*published.ref_min/published.n
    observations, groups, coefficients = [], [], []
    for view in ['2025', '2026']:
        f = metadata(view)
        counts = f.groupby(['airport', 'runway', 'day', 'stand'], observed=True, as_index=False).size()
        totals = counts.groupby(['airport', 'runway', 'day'], as_index=False)['size'].sum().rename(columns={'size': 'metadata_n'})
        cells = totals.merge(published[['airport', 'runway', 'day', 'n', 'reference']],
            on=['airport', 'runway', 'day'], how='inner', validate='one_to_one')
        keep = cells.metadata_n.ge(20) & cells.n.ge(20) & (cells.n/cells.metadata_n).between(.5, 1.2) & cells.day.str[-2:].ne('01')
        cells = cells[keep].copy()
        for (airport, runway), c in cells.groupby(['airport', 'runway']):
            c = c.sort_values('day').reset_index(drop=True)
            c['month'] = c.day.str[:7]
            c['held'] = c.day.str[-2:].astype(int)%4 == 0
            per_month = c.groupby('month').agg(held=('held', 'sum'), n=('held', 'size'))
            if len(c) < 20 or (per_month.held < 2).any() or ((per_month.n-per_month.held) < 3).any():
                continue
            q = counts[counts.airport.eq(airport) & counts.runway.eq(runway)]
            design = q.pivot(index='day', columns='stand', values='size').fillna(0).reindex(c.day).fillna(0)
            design = design.loc[:, design.sum() > 0]
            A = design.to_numpy(float)/c.metadata_n.to_numpy()[:, None]
            np.testing.assert_allclose(A.sum(axis=1), 1., atol=1e-12)
            target, month = c.reference.to_numpy(), c.month.to_numpy()
            train = ~c.held.to_numpy()
            prediction, baseline, beta, intercept = solve(A, target, month, train)
            full, _, all_beta, all_intercept = solve(A, target, month, np.ones(len(c), bool))
            c['prediction'], c['baseline'], c['full_fit'], c['view'] = prediction, baseline, full, view
            observations.append(c)
            e0, e1 = (baseline-target)**2, (prediction-target)**2
            held = ~train
            groups.append(dict(view=view, airport=airport, runway=runway, cells=len(c), held_cells=int(held.sum()),
                stands=A.shape[1], held_baseline_rmse=float(np.sqrt(e0[held].mean())),
                held_candidate_rmse=float(np.sqrt(e1[held].mean())),
                full_fit_rmse=float(np.sqrt(np.mean((full-target)**2))),
                contrast_std=float(np.std(all_beta)), min_reference=float(min(all_intercept.values())+all_beta.min()),
                max_reference=float(max(all_intercept.values())+all_beta.max())))
            for stand, coefficient in zip(design.columns, all_beta):
                for month_name, level in all_intercept.items():
                    coefficients.append(dict(view=view, airport=airport, runway=runway, stand=stand,
                        month=month_name, reference=float(level+coefficient)))
    obs = pd.concat(observations, ignore_index=True)
    by = {}
    for view, q in obs[obs.held].groupby('view'):
        by[view] = dict(cells=len(q), baseline_rmse=float(np.sqrt(np.mean((q.baseline-q.reference)**2))),
            candidate_rmse=float(np.sqrt(np.mean((q.prediction-q.reference)**2))))
    obs.to_parquet(OUT/'observations.parquet', index=False)
    pd.DataFrame(coefficients).to_parquet(OUT/'reference_estimates.parquet', index=False)
    pd.DataFrame(groups).to_csv(OUT/'groups.csv', index=False)
    result = dict(design=DESIGN, identity=source.art(OUT/'identity.json'), public_holdout=by,
        total_groups=len(groups), only_external_reference_targets=True, no_competition_score=True,
        artifacts=[source.art(OUT/p) for p in ['observations.parquet', 'reference_estimates.parquet', 'groups.csv']])
    write_json(OUT/'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
