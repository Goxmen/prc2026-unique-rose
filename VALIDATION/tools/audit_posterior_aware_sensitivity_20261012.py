"""Independent saved-output/source algebra audit; never evaluates a posterior likelihood."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import hashlib
import time
import resource
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import joint_public_moments as primitives

RUN = ROOT/'VALIDATION/runs/v3_20261012_posterior_aware_sensitivity'
OLD = ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments'
SOURCE_SHA = '5c808fe09a0662b69769177e0a6023f3e79725ddc73288c2e8d0347c7a14042b'
IDENTITY_SHA = '39f94845db85ca61cf414cbcd785155be7f96b4759976e2000662fefad3281b0'
RESULT_SHA = '26d3e06c6e6ff04b296be622c90c892f9d2120c455c234ca203db85af28afe20'
INPUTS = ['mvt_id','month','gap','prior','eligible','quality','runway',
          'local_day','assigned_day','fltrule','delay_scope']


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(record):
    assert art(ROOT/record['path']) == record, record['path']


def near(actual, expected, atol=1e-8, rtol=1e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def check_matrix(matrix, reported, threshold=None):
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = singular[0]*1e-10 if threshold is None else threshold
    assert np.isfinite(matrix).all()
    assert reported['rows'] == matrix.shape[0] and reported['columns'] == 4
    near(reported['singular_values'], singular, atol=2e-7)
    near(reported['threshold'], threshold, atol=1e-15)
    assert reported['rank'] == int((singular > threshold).sum())
    return threshold


def month_audit(month, raw, targets, source_rows, retained, taxi, delay, params, summary):
    p = raw.prior.to_numpy(float)
    d = raw.gap.to_numpy(float)
    tau = params['support']['tau']
    means, covs, probability = primitives.normal_moments(p, d)
    support = {}
    for i in np.flatnonzero(d > 3600):
        y, w, _ = primitives.prior.support(d[i], p[i], False, np.nan, 'LIRF', params['support'])
        inside = (y >= 0) & (y <= 7200)
        values = np.stack([inside.astype(float), y*inside, np.clip(d[i]-y, 0, None)], axis=1)
        mu = (w[:, None]*values).sum(axis=0)
        centered = values-mu
        means[i] = mu
        covs[i] = np.einsum('s,sa,sb->ab', w, centered, centered)
        support[i] = (y, w, values)
    taxi_mask = raw.eligible.to_numpy(bool) & raw.quality.to_numpy(bool)
    delay_mask = ~raw.fltrule.eq('V').to_numpy() & raw.delay_scope.to_numpy(bool)
    taxi_groups = {}
    for (day, runway), group in raw.loc[taxi_mask].groupby(['local_day','runway']):
        idx = group.index.to_numpy()
        if (day, runway) not in taxi.index:
            continue
        n = means[idx, 0].sum()
        src = taxi.loc[(day, runway)]
        if len(idx) >= 20 and src.source_n >= 20 and n >= 20 and .5 <= src.source_n/n <= 1.2:
            taxi_groups[(day, runway)] = (idx, n)
    delay_groups = {}
    for day, group in raw.loc[delay_mask].groupby('assigned_day'):
        idx = group.index.to_numpy()
        if day in delay.index and len(idx) >= 20:
            src = delay.loc[day]
            if np.isfinite(src.total+src.n_public):
                delay_groups[day] = idx
    eligible = taxi_mask & delay_mask & (d > 3600) & (d < 50400) & (p > tau) & (p < d)
    indices = []
    pair_keys = []
    for i in np.flatnonzero(eligible):
        tkey = (raw.local_day.iloc[i], raw.runway.iloc[i])
        dkey = raw.assigned_day.iloc[i]
        if tkey in taxi_groups and dkey in delay_groups:
            indices.append(i)
            if (tkey, dkey) not in pair_keys:
                pair_keys.append((tkey, dkey))
    indices = np.asarray(indices)
    np.testing.assert_array_equal(targets.mvt_id, raw.mvt_id.iloc[indices])
    np.testing.assert_array_equal(retained.mvt_id, raw.mvt_id)
    np.testing.assert_array_equal(retained.prior, p)
    touched = np.zeros(len(raw), bool)
    touched[indices] = True
    np.testing.assert_array_equal(retained.touched, touched)
    reconstructed = p.copy()
    reconstructed[indices] = targets.posterior.to_numpy()
    replay = float(np.max(np.abs(reconstructed-retained.joint.to_numpy())))
    near(replay, summary['zero_tilt_max_difference'], atol=0, rtol=0)
    assert replay <= 1e-8
    for key in ['gap','prior','local_day','runway','assigned_day']:
        np.testing.assert_array_equal(targets[key], raw[key].iloc[indices])
    scale = np.maximum(d[indices]-tau, 60.)
    copy = np.clip((p[indices]-tau)/(d[indices]-tau), 1e-6, 1-1e-6)
    a = np.log(d[indices]/3600.)
    b = np.log(copy)-np.log1p(-copy)
    x = np.column_stack([np.ones(len(indices)), a, b, a*b])
    near(targets[[f'x{k}' for k in range(4)]], x)
    near(targets.scale, scale)
    scalar = targets.posterior_variance.to_numpy()/scale
    near(targets.scalar_derivative, scalar)
    gradient = scalar[:,None]*x
    near(targets[[f'gradient{k}' for k in range(4)]], gradient)
    assert (targets[['prior_variance','posterior_variance','finite_difference_abs']] >= 0).all().all()
    # These are immutable producer FD receipts; +/-epsilon means were not exported.
    assert float(targets.finite_difference_abs.max()) == summary['maximum_finite_difference_abs']
    assert float(targets.finite_difference_abs.max()) < 2e-5
    source_grad = np.zeros((len(raw),3,4))
    for j, i in enumerate(indices):
        y, w, values = support[i]
        mu = float(w@y)
        variance = float(w@((y-mu)**2))
        near(targets.prior_variance.iloc[j], variance, atol=1e-6)
        # Independent expectation formula, with fixed prior support.
        derivative = values.T@(w*(y-mu))/scale[j]
        source_grad[i] = np.outer(derivative, x[j])
    near(targets.variance_ratio, targets.posterior_variance/targets.prior_variance)
    threshold = check_matrix(gradient, summary['output_jacobian'])
    centered = x.copy()
    for _, rows in targets.groupby(['local_day','runway','assigned_day']).indices.items():
        centered[rows] -= x[rows].mean(axis=0)
    check_matrix(centered*scalar[:,None], summary['within_source_pair_contrasts'], threshold)
    q = summary['quantile_levels']
    near(np.quantile(targets.variance_ratio, q), summary['posterior_prior_variance_ratio'])
    near(np.quantile(abs(targets.posterior-targets.prior), q), summary['posterior_movement_abs'])
    assert summary['input_rows'] == len(raw)
    assert summary['supported_targets'] == len(indices)
    assert summary['source_pairs'] == len(pair_keys)
    assert summary['local_days'] == targets.local_day.nunique()
    assert summary['assigned_days'] == targets.assigned_day.nunique()
    assert summary['nonzero_target_derivatives'] == np.count_nonzero(scalar)
    assert not source_rows.duplicated(['month','kind','day','runway']).any()
    source_lookup = source_rows.set_index(['kind','day','runway'])
    expected_keys = set()
    source_values = {}
    for tkey, dkey in pair_keys:
        ti, _ = taxi_groups[tkey]
        di = delay_groups[dkey]
        for coordinate, kind in [(0,'count'),(1,'taxi_sum')]:
            key = (kind,*tkey)
            source_values[key] = source_grad[ti,coordinate].sum(axis=0)
            expected_keys.add(key)
        key = ('delay_sum',dkey,'')
        source_values[key] = source_grad[di,2].sum(axis=0)
        expected_keys.add(key)
    assert set(source_lookup.index) == expected_keys
    assert summary['unique_source_coordinates'] == len(expected_keys)
    for key, values in source_values.items():
        near(source_lookup.loc[key, [f'j{k}' for k in range(4)]], values, atol=2e-6)
    source_matrix = np.vstack([source_values[tuple(row)] for row in source_rows[['kind','day','runway']].itertuples(index=False,name=None)])
    units = np.where(source_rows.kind.eq('count'),1.,3600.)
    check_matrix(source_matrix/units[:,None], summary['unique_source_mean_jacobian_fixed_units'])
    whitened = []
    bias = []
    for tkey, dkey in pair_keys:
        ti, taxi_n = taxi_groups[tkey]
        di = delay_groups[dkey]
        # Direct rowwise projection retains shared-flight cross covariance.
        select = np.zeros((len(raw),3))
        select[ti,:2] = 1
        select[di,2] = 1
        covariance = np.einsum('iab,ia,ib->ab', covs, select, select)
        covariance[:2,:2] += len(ti)*np.asarray(params['taxi']['covariance_per_row']) + np.diag([1.,3600.])
        count_delta = float(delay.loc[dkey].n_public-len(di))
        covariance[2,2] += 3*(params['delay']['sigma']**2+(params['delay']['kappa']*count_delta)**2)
        common = np.array([0.,taxi_n,-np.sum(probability[di]*(d[di]<=3600))])
        covariance += 225*np.outer(common,common)
        chol = np.linalg.cholesky(covariance)
        jacobian = np.stack([source_values[('count',*tkey)],source_values[('taxi_sum',*tkey)],source_values[('delay_sum',dkey,'')]])
        whitened.append(np.linalg.solve(chol,jacobian))
        bias.append(np.linalg.solve(chol,np.eye(3)))
    white = np.concatenate(whitened)
    nuisance = np.concatenate(bias)
    threshold = check_matrix(white, summary['local_pair_whitened_descriptive'])
    # Independent least-squares projection rather than producer's QR expression.
    residual = white-nuisance@np.linalg.lstsq(nuisance,white,rcond=None)[0]
    report = summary['local_pair_whitened_after_three_constant_source_biases']
    check_matrix(residual, report, threshold)
    near(np.linalg.norm(residual)/np.linalg.norm(white), report['frobenius_norm_ratio'])
    return dict(input_rows=len(raw),supported_targets=len(indices),source_pairs=len(pair_keys),
                unique_source_coordinates=len(expected_keys),zero_replay_max_difference=replay,
                saved_finite_difference_max_error=float(targets.finite_difference_abs.max()),
                all_features_derivative_prior_moments_source_jacobians_and_ranks_verified=True,
                whitening='independent rowwise source projection plus least-squares nuisance removal')


def main():
    started = time.monotonic()
    output = RUN/'independent_audit.json'
    assert not output.exists(), 'Preserve completed audit'
    identity = read(RUN/'identity.json')
    result = read(RUN/'result.json')
    assert art(RUN/'identity.json')['sha256'] == IDENTITY_SHA
    assert art(RUN/'result.json')['sha256'] == RESULT_SHA
    assert art(ROOT/'VALIDATION/posterior_aware_sensitivity.py')['sha256'] == SOURCE_SHA
    assert result['identity'] == art(RUN/'identity.json')
    assert result['design'] == identity['design']
    for record in [*identity['files'],*result['outputs']]:
        verify(record)
    assert result['labels_loaded'] is False and result['models_fitted'] is False and result['cv_scored'] is False
    assert pq.ParquetFile(OLD/'inputs.parquet').schema_arrow.names == INPUTS
    raw = pd.read_parquet(OLD/'inputs.parquet',columns=INPUTS)
    targets = pd.read_parquet(RUN/'targets.parquet')
    sources = pd.read_parquet(RUN/'source_jacobians.parquet')
    retained = pd.read_parquet(OLD/'predictions.parquet')
    assert raw.mvt_id.is_unique and targets.mvt_id.is_unique and retained.mvt_id.is_unique
    assert len(raw) == 26528 and set(raw.month) == {'2025-01','2025-07'}
    assert set(targets.month) == set(sources.month) == set(raw.month)
    assert np.isfinite(targets.select_dtypes('number')).all().all()
    assert np.isfinite(sources.select_dtypes('number')).all().all()
    taxi = pd.read_parquet(OLD/'taxi_source.parquet').set_index(['local_day','runway'])
    delay = pd.read_parquet(OLD/'delay_source.parquet').set_index('day')
    assert taxi.index.is_unique and delay.index.is_unique
    params = read(OLD/'parameters.json')
    months = {}
    with threadpool_limits(limits=1):
        for month in ['2025-01','2025-07']:
            mraw = raw[raw.month.eq(month)].reset_index(drop=True)
            keep = retained.set_index('mvt_id').loc[mraw.mvt_id].reset_index()
            assert keep.month.eq(month).all()
            months[month] = month_audit(month,mraw,targets[targets.month.eq(month)].reset_index(drop=True),
                sources[sources.month.eq(month)].reset_index(drop=True),keep,taxi,delay,params[month],result['months'][month])
    report = dict(status='pass',source=art(Path(__file__)),identity=art(RUN/'identity.json'),result=art(RUN/'result.json'),
        files_verified=len(identity['files'])+len(result['outputs']),months=months,
        seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        labels_loaded=False,posterior_likelihood_evaluations=0,fits=0,
        reused_primitives='Original frozen support construction and ordinary Gaussian moments only. No producer grouping, derivative, matrix-summary, whitening or posterior code called.',
        finite_difference_limit='Signed perturbed posterior means were not exported. Audited recorded per-target/max finite-difference errors and immutable reviewed source; did not independently rerun posterior finite differences.',
        conclusion='Algebraic sensitivity confirmed only. No predictive-gain or global-identifiability evidence; honest train-peer packets remain unavailable. No training authorization implied.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(audit=art(output),months=months,seconds=report['seconds'])))


if __name__ == '__main__':
    main()
