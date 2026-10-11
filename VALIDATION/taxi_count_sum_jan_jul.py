"""Fixed count/sum recording-mode screen, replacing Rome mixture daily shifts."""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import public_daily_taxi_quality as quality
import taxi_count_sum_posterior as posterior
from records import write_json

daily = quality.original
ROOT = daily.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_taxi_count_sum'
TAU = daily.parent.original.tail.heavy.old.OUT/'screen.json'
art, read, verify = daily.art, daily.read, daily.verify
DESIGN = dict(parent='selected production-rounded daily-quality refinement',
    change='Rome ambiguous two-mode rows use joint truncated source count/sum inference instead of uniform daily shift',
    scope='stand/runway present, existing monthly quality gate and public supported runway-day cells',
    prior='refined parent before daily quality correction; copy share from inherited tau and scheduled gap',
    ordinary_rows='retain selected daily-quality prediction exactly',
    source_calibration='allowed-training complete local days only; omitted count and sum versus intercept and input count/100',
    fit='separate Huber losses scaled by training-target MAD; no held-label fitting',
    residual_covariance='centered residuals/sqrt(input count); retain cells within three MADs in both coordinates',
    settings=posterior.SETTINGS, no_weight_or_parameter_search=True, no_taxi_model_fit=True,
    no_production=True, no_upload=True, limitation='Gaussian moment approximation for other modes; reused development')


def guard():
    assert read(quality.OUT/'audit.json')['status'] == 'pass'
    paths = [Path(__file__), Path(posterior.__file__), Path(daily.__file__), Path(quality.__file__),
        Path(daily.constraints.__file__), Path(daily.constraints.exclusion.__file__),
        quality.OUT/'scored.parquet', quality.OUT/'result.json', quality.OUT/'audit.json',
        daily.CACHE, daily.SOURCE/'daily_runway.parquet', TAU, daily.base.OUT/'rows2025.parquet',
        daily.constraints.exclusion.OUT/'matches2025.parquet']
    value = dict(design=DESIGN, checks=posterior.checks(), sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(path, value)
    return value


def calibrate(rows, source, allowed):
    assert set(rows.month).issubset(allowed)
    rows = rows[rows.airport.eq('LIRF') & rows.eligible].copy()
    rows['valid'] = rows.y.between(0, 7200)
    rows['contribution'] = np.where(rows.valid, rows.y, 0.)
    cells = rows.groupby(daily.KEY, as_index=False).agg(n=('y', 'size'), valid=('valid', 'sum'), total=('contribution', 'sum'))
    cells = cells.merge(source, on=daily.KEY, how='inner', validate='one_to_one')
    keep = (cells.valid.ge(20) & cells.source_n.ge(20) & (cells.source_n/cells.valid).between(.5, 1.2)
            & daily.complete_days(cells, allowed))
    cells = cells[keep].reset_index(drop=True)
    assert len(cells) >= 100
    X = np.column_stack([np.ones(len(cells)), cells.n/100.])
    target = np.column_stack([cells.valid-cells.source_n, cells.total-cells.source_n*cells.source_mean])
    scale = np.maximum(1.4826*np.median(abs(target-np.median(target, axis=0)), axis=0), [1., 60.])
    coefficients = []
    for k in range(2):
        solution = least_squares(lambda b: (X@b-target[:, k])/scale[k],
            np.array([np.median(target[:, k]), 0.]), loss='huber', f_scale=1.)
        assert solution.success
        coefficients.append(solution.x)
    coefficients = np.asarray(coefficients).T
    residual = target-X@coefficients
    normalized = residual/np.sqrt(cells.n.to_numpy())[:, None]
    center = np.median(normalized, axis=0)
    mad = np.maximum(1.4826*np.median(abs(normalized-center), axis=0), [1e-6, 1e-3])
    selected = np.max(abs(normalized-center)/mad, axis=1) <= 3
    assert selected.sum() >= 100
    covariance = np.cov(normalized[selected].T)
    assert np.linalg.eigvalsh(covariance).min() > 0
    cells['omitted_count'], cells['omitted_sum'] = target.T
    cells['residual_count'], cells['residual_sum'] = residual.T
    cells['covariance_selected'] = selected
    return dict(coefficients=coefficients.tolist(), covariance_per_row=covariance.tolist(),
        allowed=allowed, cells=len(cells), covariance_cells=int(selected.sum()),
        huber_scale=scale.tolist(), normalized_mad=mad.tolist()), cells


def screen():
    guard()
    assert not (OUT/'result.json').exists(), 'Preserve completed screen'
    rows, source = daily.rows(), daily.public()
    f = pd.read_parquet(quality.OUT/'scored.parquet')
    ids = f.mvt_id.to_numpy()
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    inputs = rows.drop(columns='y').set_index('mvt_id').loc[ids].reset_index()
    inputs['gap'] = raw.gap
    np.testing.assert_array_equal(raw.y, f.y)
    source_index = source.set_index(daily.KEY)
    candidate = f.refined_quality.to_numpy(float).copy()
    touched = np.zeros(len(f), bool)
    parameters, calibration, logs, details = {}, [], [], []
    old_parameters = read(TAU)['parameters']
    for month, allowed in daily.base.FOLDS.items():
        parameters[month], c = calibrate(rows[rows.month.isin(allowed)], source, allowed)
        calibration.append(c.assign(fold=month))
        ix = np.flatnonzero(f.month.eq(month) & f.airport.eq('LIRF') & f.eligible & f.quality_eligible)
        selected = inputs.iloc[ix].reset_index(drop=True)
        for key, local in selected.groupby(daily.KEY).indices.items():
            if key not in source_index.index:
                continue
            at = ix[local]
            pub = source_index.loc[key]
            incoming = f.refined.iloc[at].to_numpy(float)
            gap = selected.iloc[local].gap.to_numpy(float)
            adjusted, info, detail = posterior.condition(incoming, gap, old_parameters[month]['LIRF']['tau'],
                float(pub.source_n), float(pub.source_n*pub.source_mean), parameters[month])
            if info['active']:
                active_local = np.array([d['at'] for d in detail], int)
                candidate[at[active_local]] = adjusted[active_local]
                touched[at[active_local]] = True
                for d in detail:
                    details.append(dict(mvt_id=ids[at[d['at']]], month=month, **d))
            logs.append(dict(fold=month, airport=key[0], local_day=key[1], runway=key[2], rows=len(at),
                source_n=float(pub.source_n), source_sum=float(pub.source_n*pub.source_mean), **info))
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    constrained = daily.constrain(raw, ids, f.month, candidate, matches)
    np.testing.assert_array_equal(daily.constrain(raw, ids, f.month, f.refined_quality.to_numpy(), matches), f.refined_quality)
    np.testing.assert_array_equal(constrained[~touched], f.refined_quality.to_numpy()[~touched])
    f['touched'] = touched
    f['candidate_unconstrained'], f['candidate'] = candidate, constrained
    names = ['refined', 'refined_quality', 'candidate']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in names} for m, g in f.groupby('month')}
    paired = {name: daily.parent.original.airport.parent.ts.paired_bootstrap(f.day, f.month,
        (f[name]-f.y)**2, (f.candidate-f.y)**2) for name in names[:-1]}
    f.to_parquet(OUT/'scored.parquet', index=False)
    pd.concat(calibration, ignore_index=True).to_parquet(OUT/'calibration.parquet', index=False)
    pd.DataFrame(logs).to_parquet(OUT/'cells.parquet', index=False)
    pd.DataFrame(details).to_parquet(OUT/'probabilities.parquet', index=False)
    write_json(OUT/'parameters.json', parameters)
    result = dict(design=DESIGN, identity=art(OUT/'identity.json'), scores=scores,
        cv={c: float(np.mean([v[c] for v in scores.values()])) for c in names}, paired=paired,
        touched=int(touched.sum()), changed=int(f.candidate.ne(f.refined_quality).sum()),
        artifacts=[art(OUT/p) for p in ['scored.parquet', 'calibration.parquet', 'cells.parquet', 'probabilities.parquet', 'parameters.json']])
    write_json(OUT/'result.json', result)
    print(result, flush=True)


def audit():
    guard()
    assert not (OUT/'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT/'result.json')
    for artifact in result['artifacts']:
        verify(artifact)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('c', OUT/'scored.parquet'), ('p', quality.OUT/'scored.parquet'),
                           ('r', daily.CACHE), ('cal', OUT/'calibration.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, u, errors = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.refined is distinct from p.refined
            or c.refined_quality is distinct from p.refined_quality or c.month is distinct from p.month
            or c.day is distinct from p.day or (not c.touched and c.candidate!=p.refined_quality)
            or (c.touched and (c.airport!='LIRF' or not c.eligible or not c.quality_eligible))
            then 1 else 0 end) from c join p using(mvt_id) join r using(mvt_id)''').fetchone()
        assert n == u == 344419 and errors == 0
        for fold, allowed in daily.base.FOLDS.items():
            actual = con.execute('''select cal.local_day,cal.runway,cal.n,cal.valid,cal.total,
                count(*) as sql_n,sum(case when r.y between 0 and 7200 then 1 else 0 end) sql_valid,
                sum(case when r.y between 0 and 7200 then r.y else 0 end) sql_total
                from cal join r on cal.airport=r.airport and cal.local_day=r.local_takeoff and cal.runway=r.runway
                where cal.fold=? and strftime(r.takeoff AT TIME ZONE 'UTC','%Y-%m') in (select unnest(?))
                and r.stand is not null and r.runway is not null group by all''', [fold, allowed]).df()
            assert len(actual) == read(OUT/'parameters.json')[fold]['cells']
            for column in ['n', 'valid', 'total']:
                np.testing.assert_array_equal(actual[column], actual['sql_'+column])
        for col in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({col}-y,2))) from c group by month').fetchall()
            for m, v in values:
                np.testing.assert_allclose(v, result['scores'][m][col], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][col], atol=1e-8, rtol=0)
    f = pd.read_parquet(OUT/'scored.parquet')
    detail = pd.read_parquet(OUT/'probabilities.parquet').set_index('mvt_id')
    assert len(detail) == int(f.touched.sum()) and detail.index.is_unique
    from scipy.special import expit, logit
    independent = expit(logit(detail.old_probability)+detail.loglik_one-detail.loglik_zero)
    np.testing.assert_array_equal(independent, detail.new_probability)
    ids = f.mvt_id.to_numpy()
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(daily.constrain(raw, ids, f.month, f.candidate_unconstrained.to_numpy(), matches), f.candidate)
    report = dict(status='pass', rows=n, identity=art(OUT/'identity.json'), result=art(OUT/'result.json'),
        independent_sql_parents_truth_membership_calibration_scores=True, unchanged_outside_scope=True,
        all_posterior_odds_and_constraints_replay=True, source=art(__file__))
    write_json(OUT/'audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'screen', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
