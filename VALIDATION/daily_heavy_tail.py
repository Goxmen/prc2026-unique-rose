"""Fixed heavier-tail ordinary aggregate in the aligned daily posterior.

Keep the retained priors, date assignments, source-noise parameters, message
algorithm and 41-node grid. Replace Gaussian aggregate weights with t5 weights,
deterministically matched to the same discrete variance. No outcome-based
airport gate, scale search or new taxi model. This is reused-development work.
"""
import argparse
import itertools
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_offblock_alignment as aligned
from records import write_json

old, base = aligned.posterior, aligned.base
OUT = base.ROOT / 'VALIDATION/runs/v3_20261011_daily_heavy_tail'
Z = np.linspace(-old.DESIGN['ordinary_grid_sd'], old.DESIGN['ordinary_grid_sd'],
                old.DESIGN['ordinary_grid_points'])
GAUSSIAN = -.5*Z**2
TARGET_VARIANCE = float(np.exp(GAUSSIAN)@Z**2/np.exp(GAUSSIAN).sum())


def t5_weights():
    lo, hi = .01, 10.
    for _ in range(60):
        scale = (lo+hi)/2
        logw = -3*np.log1p((Z/scale)**2/5)
        weights = np.exp(logw)
        variance = weights@Z**2/weights.sum()
        if variance < TARGET_VARIANCE:
            lo = scale
        else:
            hi = scale
    return logw, scale


STUDENT, SCALE = t5_weights()
DESIGN = dict(parent='estimated off-block-date posterior on serialized original v70',
    change='t5 ordinary aggregate weights on unchanged 41-node grid, matched discrete variance',
    df=5, scale_in_original_sd=SCALE, discrete_variance=TARGET_VARIANCE,
    grid=old.DESIGN, no_new_taxi_fit=True, no_scale_or_weight_search=True,
    source_parameters='reuse allowed-training off-block parameters exactly',
    scope='same input-assigned interior dates; same fallback predictions',
    interpretation='Shared heavy-tail scale for aggregate ordinary errors; fixed old linear allocation',
    evaluation='motivated by prior-screen residuals on reused Jan/Jul development')


def condition(d, p, nm, nm_gap, airport, prm, total, dn, logprior=STUDENT):
    long = d > 3600
    ordinary, active = ~long, (~long)&(d > p)
    prior_ordinary = float(np.maximum(0, d[ordinary]-p[ordinary]).sum())
    n = int(active.sum())
    indvar, common = old.DESIGN['ordinary_sigma']**2, old.DESIGN['common_day_sigma']**2
    shortvar = n*indvar+n*n*common
    offsets = Z*max(np.sqrt(shortvar), 1.)
    rows = np.flatnonzero(long)
    supports = [old.support(d[i], p[i], nm[i], nm_gap[i], airport, prm) for i in rows]
    size = 1+sum(int(s[2].max()) for s in supports)
    totals = np.arange(size)*old.DESIGN['grid_sec']
    measurement_sd = np.sqrt(prm['sigma']**2+(prm['kappa']*dn)**2)
    z = (total-prm['bias']-prior_ordinary-totals[:, None]-offsets[None, :])/measurement_sd
    joint = logprior[None, :]-2*np.log1p(z*z/3.)
    joint = np.exp(joint-joint.max())
    likelihood = joint.sum(axis=1)
    prior = np.zeros(size)
    prior[0] = 1.
    forward = []
    for ys, weights, contributions in supports:
        forward.append(prior)
        nxt = np.zeros(size)
        for weight, contribution in zip(weights, contributions):
            nxt[contribution:] += weight*prior[:size-contribution]
        prior = nxt/nxt.sum()
    post_offset = (prior[:, None]*joint).sum(axis=0)
    assert post_offset.sum() > 0
    expected_offset = float(post_offset@offsets/post_offset.sum())
    result = p.copy()
    if n:
        result[active] -= (indvar+n*common)/max(shortvar, 1)*expected_offset
        result[ordinary&~active] -= .25*n*common/max(shortvar, 1)*expected_offset
    backward = likelihood/likelihood.max()
    for j in range(len(rows)-1, -1, -1):
        ys, weights, contributions = supports[j]
        factors = np.array([forward[j][:size-c]@backward[c:] for c in contributions])
        posterior = weights*factors
        assert posterior.sum() > 0
        result[rows[j]] = posterior@ys/posterior.sum()
        nxt = np.zeros(size)
        for weight, contribution in zip(weights, contributions):
            nxt[:size-contribution] += weight*backward[contribution:]
        backward = nxt/max(nxt.max(), 1e-300)
    return np.maximum(result, 0), dict(offset=expected_offset, grid_size=size,
        long_rows=len(rows), active=n, prior_ordinary=prior_ordinary, measurement_sd=measurement_sd)


def checks():
    weights = np.exp(STUDENT)
    np.testing.assert_allclose(weights@Z/weights.sum(), 0, atol=1e-15)
    np.testing.assert_allclose(weights@Z**2/weights.sum(), TARGET_VARIANCE, rtol=0, atol=1e-14)
    assert weights[0]/weights.sum() > np.exp(GAUSSIAN[0])/np.exp(GAUSSIAN).sum()
    prm = dict(spread=[-100., 0., 100.], sigma=120., kappa=600., tau=800., bias=30.)
    d, p = np.array([1000., 500., 5000., 7000.]), np.array([800., 900., 2000., 4000.])
    nm, gap = np.zeros(4, bool), np.full(4, np.nan)
    original, _ = old.condition_day(d, p, nm, gap, 'LIRF', prm, 4500., 0.)
    replay, _ = condition(d, p, nm, gap, 'LIRF', prm, 4500., 0., GAUSSIAN)
    np.testing.assert_allclose(replay, original, rtol=0, atol=1e-10)
    actual, _ = condition(d, p, nm, gap, 'LIRF', prm, 4500., 0.)
    # Independent enumeration, including each ordinary offset node.
    supports = [old.support(d[i], p[i], nm[i], gap[i], 'LIRF', prm) for i in [2, 3]]
    offsets = Z*np.sqrt(old.DESIGN['ordinary_sigma']**2+old.DESIGN['common_day_sigma']**2)
    numer = np.zeros(3)
    denominator = 0.
    states = 0
    for j, k, at in itertools.product(range(len(supports[0][0])), range(len(supports[1][0])), range(len(Z))):
        c = (supports[0][2][j]+supports[1][2][k])*old.DESIGN['grid_sec']
        residual = (4500-prm['bias']-200-c-offsets[at])/prm['sigma']
        weight = supports[0][1][j]*supports[1][1][k]*weights[at]*(1+residual**2/3)**-2
        denominator += weight
        numer += weight*np.array([offsets[at], supports[0][0][j], supports[1][0][k]])
        states += 1
    offset, y0, y1 = numer/denominator
    common, indvar = old.DESIGN['common_day_sigma']**2, old.DESIGN['ordinary_sigma']**2
    expected = np.maximum([p[0]-offset, p[1]-.25*common/(indvar+common)*offset, y0, y1], 0)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
    order = np.array([3, 1, 0, 2])
    shuffled, _ = condition(d[order], p[order], nm[order], gap[order], 'LIRF', prm, 4500., 0.)
    np.testing.assert_allclose(shuffled, actual[order], rtol=0, atol=1e-8)
    for dd, pp in [(np.array([1000., 500.]), p[:2]), (d[2:], p[2:])]:
        candidate, _ = condition(dd, pp, np.zeros(len(dd), bool), np.full(len(dd), np.nan), 'LIRF', prm, 4500., 0., GAUSSIAN)
        control, _ = old.condition_day(dd, pp, np.zeros(len(dd), bool), np.full(len(dd), np.nan), 'LIRF', prm, 4500., 0.)
        np.testing.assert_allclose(candidate, control, rtol=0, atol=1e-10)
    return dict(variance_matched=True, centered=True, gaussian_replay=True, no_long_and_no_ordinary=True,
        enumerated_states=states, enumeration_max_error=float(abs(actual-expected).max()), permutation=True)


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = aligned.read(aligned.OUT / 'audit.json')
    assert audit['status'] == 'pass'
    aligned.guard()
    value = dict(design=DESIGN, checks=checks(), source=base.artifact(__file__),
        helpers=[base.artifact(m.__file__) for m in [old, aligned, base]],
        aligned_result=base.artifact(aligned.OUT / 'result.json'), aligned_audit=base.artifact(aligned.OUT / 'audit.json'),
        rows=base.artifact(aligned.OUT / 'rows2025.parquet'), prior=base.artifact(aligned.OUT / 'scored.parquet'))
    path = OUT / 'identity.json'
    if path.exists():
        assert aligned.read(path) == value
    else:
        write_json(path, value)
    return value


def screen():
    guard()
    assert not (OUT / 'result.json').exists(), 'Preserve completed screen'
    f = pd.read_parquet(aligned.OUT / 'scored.parquet')
    raw = pd.read_parquet(aligned.OUT / 'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    parameters = aligned.read(aligned.OUT / 'result.json')['parameters']
    source = base.public(2025).set_index(['airport', 'day'])
    candidate = f.pred.to_numpy(float).copy()
    logs = []
    for month in base.FOLDS:
        ix = np.flatnonzero(f.month.eq(month))
        incoming = f.parent.iloc[ix].to_numpy(float)
        rows = aligned.inference_rows(raw.iloc[ix].reset_index(), incoming)
        np.testing.assert_array_equal(rows.day, f.assigned_day.iloc[ix])
        positions = np.flatnonzero(f.new_scope.iloc[ix])
        selected = rows.iloc[positions].reset_index(drop=True)
        for (airport, day), local in selected.groupby(['airport', 'day']).indices.items():
            at = ix[positions[local]]
            r = selected.iloc[local]
            pub = source.loc[(airport, day)]
            arguments = (r.gap.to_numpy(float), f.parent.iloc[at].to_numpy(float),
                r.has_id.to_numpy(bool), r.nm_gap.to_numpy(float), airport, parameters[month]['offblock'][airport],
                float(pub.total), float(pub.n_public-len(r)))
            replay, _ = condition(*arguments, logprior=GAUSSIAN)
            np.testing.assert_allclose(replay, f.pred.iloc[at], rtol=0, atol=1e-8)
            candidate[at], info = condition(*arguments)
            logs.append(dict(month=month, airport=airport, day=day, rows=len(r), **info))
    f['offblock_gaussian'] = f.pred
    f['pred'] = candidate
    np.testing.assert_array_equal(f.loc[~f.new_scope, 'pred'], f.loc[~f.new_scope, 'offblock_gaussian'])
    assert np.isfinite(candidate).all() and (candidate >= 0).all()
    f.to_parquet(OUT / 'scored.parquet', index=False)
    pd.DataFrame(logs).to_parquet(OUT / 'days.parquet', index=False)
    import tuning_seasonal as ts
    columns = ['old_all_departures', 'control', 'offblock_gaussian', 'pred']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m, g in f.groupby('month')}
    value = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([s[c] for s in scores.values()])) for c in columns},
        paired={c: ts.paired_bootstrap(f.day, f.month, (f[c]-f.y)**2, (f.pred-f.y)**2) for c in columns[:-1]},
        identity=base.artifact(OUT / 'identity.json'), predictions=base.artifact(OUT / 'scored.parquet'),
        fallback_exact=True, gaussian_control_replayed_all_scope_rows=True,
        scope_rows=int(f.new_scope.sum()), interpretation='Exploratory fixed follow-up; no production or upload selected')
    write_json(OUT / 'result.json', value)
    print(value, flush=True)


def audit():
    guard()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    result = aligned.read(OUT / 'result.json')
    assert result['identity'] == base.artifact(OUT / 'identity.json')
    assert result['predictions'] == base.artifact(OUT / 'scored.parquet')
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(OUT / 'scored.parquet')).create_view('candidate')
        con.read_parquet(str(aligned.OUT / 'scored.parquet')).create_view('prior')
        con.read_parquet(str(aligned.OUT / 'rows2025.parquet')).create_view('truth')
        count, unique, errors = con.execute('''select count(*),count(distinct c.mvt_id),
            sum(case when c.y is distinct from t.y or c.day is distinct from t.day or
            c.month is distinct from t.month or c.assigned_day is distinct from p.assigned_day or
            c.new_scope is distinct from p.new_scope or c.offblock_gaussian is distinct from p.pred or
            c.control is distinct from p.control or c.old_all_departures is distinct from p.old_all_departures or
            (not c.new_scope and c.pred is distinct from p.pred) then 1 else 0 end)
            from candidate c join prior p using(mvt_id) join truth t using(mvt_id)''').fetchone()
        assert count == unique == 344419 and errors == 0
        for column in result['cv']:
            metrics = con.execute(f'select month,sqrt(avg(pow({column}-y,2))) as value from candidate group by month').fetchall()
            for month, value in metrics:
                np.testing.assert_allclose(value, result['scores'][month][column], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in metrics]), result['cv'][column], atol=1e-8, rtol=0)
    value = dict(status='pass', rows=count, truth_calendar_controls_and_fallbacks_exact=True,
        independent_sql_scores=True, result=base.artifact(OUT / 'result.json'), source=base.artifact(__file__),
        scope='Fixed quadrature preflight, all scoped Gaussian replays, independent SQL truth/control/scores')
    write_json(OUT / 'audit.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'screen', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
