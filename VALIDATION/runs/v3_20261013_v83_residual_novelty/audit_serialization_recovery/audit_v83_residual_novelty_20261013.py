"""Independent input and result audit of one fixed v83 residual certificate.

Uses saved SVD factors, never another decomposition/rank rule or an optimizer.
Only projected old-ranking predictions are decoded. Long-double moment checks
and Decimal score-interval extrema are independent of the producer reductions.
No candidate, taxi CV labels, model inference, or final-portability claim.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261013_v83_residual_novelty'
OLD = ROOT / 'VALIDATION/runs/v3_20261013_portable_feedback_geometry'
STUDENT = ROOT / 'VALIDATION/runs/v3_20261013_v80_teacher_distillation'
SOURCE = ROOT / 'VALIDATION/v83_residual_novelty_20261013.py'
DESIGN = ROOT / 'VALIDATION/reports/v83_residual_novelty_design_20261013.json'
MANIFEST = OUT / 'identity.json'


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return {'path': str(path.relative_to(ROOT)), 'sha256': h.hexdigest()}


def verify(record):
    path = (ROOT / record['path']).resolve()
    assert path.is_relative_to(ROOT) and art(path) == {k: record[k] for k in ['path', 'sha256']}
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def projected(path, columns):
    # Explicit prediction-only allowlist; no all-column fallback or held truth.
    allowed = {'mvt_id', 'MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'raw', 'submitted', 'v82'}
    assert columns and all(c in allowed or (c.startswith('v') and c[1:].isdigit()) for c in columns)
    assert '/cv/' not in str(path) and Path(path).name != 'cv_basis.parquet'
    return pd.read_parquet(path, columns=columns)


def aligned(frame, key, ids):
    assert len(frame) == frame[key].nunique() == len(ids)
    assert frame[key].notna().all() and set(frame[key]) == set(ids)
    return frame.set_index(key).loc[ids]


def close(a, b, atol=1e-8, rtol=1e-11):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def decimal_interval(coefficients, centers, constant):
    """Exact extrema for fixed float64 coefficients and decimal public scores."""
    with localcontext() as context:
        context.prec = 70
        lo = hi = Decimal.from_float(float(constant))
        half = Decimal('0.00005')
        for coefficient, center in zip(coefficients, centers):
            c = Decimal.from_float(float(coefficient))
            score = Decimal(str(center))
            lower, upper = (score - half) ** 2, (score + half) ** 2
            lo += c * (lower if c >= 0 else upper)
            hi += c * (upper if c >= 0 else lower)
        return float(lo), float(hi)


def audit():
    m, result = read(MANIFEST), read(OUT / 'result.json')
    assert m['source'] == art(SOURCE) and m['auditor'] == art(__file__) and m['design'] == art(DESIGN)
    pinned = {r['path']: r['sha256'] for r in m['sources']}
    assert len(pinned) == len(m['sources'])
    for record in m['sources']:
        verify(record)
    for field in ['source', 'design', 'auditor', 'review', 'fixture', 'basis', 'candidate', 'student_prediction', 'student_model']:
        assert pinned[m[field]['path']] == m[field]['sha256']
    assert result['identity'] == art(MANIFEST)
    allocation = read(verify(result['allocation']))
    assert allocation['authority'] == '/root' and allocation['identity'] == art(MANIFEST)
    assert allocation['state'] == 'released_for_single_residual_information_certificate'
    started = read(OUT / 'started.json')
    assert started['identity'] == art(MANIFEST) and started['allocation'] == result['allocation']
    old_identity, old_result, old_audit = (read(OLD / name) for name in ['identity.json', 'result.json', 'audit.json'])
    assert old_audit['status'] == 'pass' and old_result['identity'] == old_audit['identity'] == art(OLD / 'identity.json')
    assert old_audit['result'] == art(OLD / 'result.json') and m['basis'] in old_result['outputs']
    assert read(OLD / 'closed.json')['gate_pass'] is False
    old_scores = read(OLD / 'solve_started.json')
    assert old_scores['identity'] == old_result['identity']
    live = read(verify(old_scores['live_snapshot']))['rows']
    live_by = {r['filename']: r for r in live if r['teamName'] == 'unique-rose'}
    for version in m['versions']:
        item = live_by[f'unique-rose_v{version}.parquet']
        assert item['usedPairs'] == 344841 and item['score'] == old_scores['scores'][str(version)] == m['scores'][str(version)]
    for rec in old_scores['scored_manifests'].values():
        verify(rec)
    official = read(STUDENT / 'submission_v83/official_result.json')
    submission = read(STUDENT / 'submission_v83/manifest.json')
    assert official['status'] == 'Succeeded' and official['used_pairs'] == 344841 and official['file'] == 'unique-rose_v83.parquet'
    assert submission['state'] == 'scored' and submission['official_result'] == official
    assert submission['score'] == official['score'] == m['scores']['83']
    assert submission['sha256'] == submission['remote_sha256'] == m['candidate']['sha256']
    prod, prod_audit = read(STUDENT / 'production/result.json'), read(STUDENT / 'production/audit.json')
    assert prod_audit['status'] == 'pass' and prod_audit['result'] == art(STUDENT / 'production/result.json')
    assert prod['identity'] == prod_audit['identity'] == art(STUDENT / 'identity.json')
    assert prod['model'] == m['student_model']
    for key, field in [('candidate', 'candidate'), ('prediction', 'student_prediction')]:
        assert prod['views']['old'][key] == prod_audit['views']['old'][key] == m[field]
    versions = m['versions']
    assert versions == old_identity['versions'] and len(versions) == len(set(versions)) == 54
    assert set(m['scores']) == {str(v) for v in versions + [83]} and min(m['scores'].values()) == m['scores']['80']
    f = projected(verify(m['basis']), ['mvt_id'] + [f'v{v}' for v in versions])
    assert len(f) == f.mvt_id.nunique() == 344841 and f.mvt_id.notna().all()
    ids = f.mvt_id.to_numpy()
    P = f[[f'v{v}' for v in versions]].to_numpy(np.float64)
    c = aligned(projected(verify(m['candidate']), ['MVT_ID_mvt', 'TAXITIME_SEC_mvt']), 'MVT_ID_mvt', ids)
    p83 = c.TAXITIME_SEC_mvt.to_numpy(np.float64)
    student = aligned(projected(verify(m['student_prediction']), ['mvt_id', 'v82', 'raw', 'submitted']), 'mvt_id', ids)
    for a in [P, p83]:
        assert np.isfinite(a).all() and (a >= 0).all() and (a <= 2**52).all() and np.array_equal(a, np.rint(a))
    np.testing.assert_array_equal(student.submitted, p83)
    np.testing.assert_array_equal(student.v82, P[:, versions.index(82)])
    model = read(verify(m['student_model']))
    assert len(model['columns']) == len(model['beta']) == 28 and len(set(model['columns'])) == 28
    weights = np.zeros(54)
    for v, coefficient in zip(model['columns'], model['beta']):
        weights[versions.index(v)] = coefficient
    weights[versions.index(82)] = 1 - sum(model['beta'])
    assert abs(weights.sum() - 1) < 1e-12 and weights.min() >= -3 and weights.max() <= 4
    raw = np.einsum('ij,j->i', P, weights)
    close(raw, student.raw, atol=1e-7, rtol=0)
    assert result['original_affine_student_feasible'] is True
    with np.load(verify(result['decomposition']), allow_pickle=False) as storage:
        saved = {key: storage[key] for key in storage.files}
    np.testing.assert_array_equal(saved['mvt_id'], ids)
    np.testing.assert_array_equal(saved['raw_student_weights'], weights)
    for key, values in saved.items():
        assert np.isfinite(values).all(), key
    n = len(ids); active = [v for v in versions if v != 64]
    assert result['rows'] == n and result['columns'] == 54 and result['active_versions'] == active
    assert result['reference_version'] == 64 and result['baseline_version'] == 80
    D = P[:, [versions.index(v) for v in active]] - P[:, versions.index(64), None]
    delta = p83 - P[:, versions.index(64)]
    A, z = D / np.sqrt(n), delta / np.sqrt(n)
    alpha, s, Vt = saved['alpha'], saved['singular_values'], saved['right_vectors']
    assert alpha.shape == s.shape == (53,) and Vt.shape == (53, 53)
    assert (s >= 0).all() and np.all(s[:-1] >= s[1:])
    eps = np.finfo(np.float64).eps
    tau = eps * max(n, 53) * s[0]
    keep = s > tau
    close(result['rank_tolerance'], tau, atol=0, rtol=1e-14)
    assert result['rank'] == int(keep.sum()) and keep.any()
    close(Vt @ Vt.T, np.eye(53), atol=1e-12, rtol=0)
    # Verify stored full right-factor geometry, without another SVD or rank rule.
    gram = A.T @ A
    factor_error = float(np.max(abs(gram @ Vt.T - Vt.T * (s * s)[None, :])))
    factor_bound = float(128 * eps * max(n, 53) * max(1., s[0] ** 2))
    assert factor_error <= factor_bound
    coordinates = saved['left_coordinates']
    assert coordinates.shape == (53,)
    close(alpha, Vt[keep].T @ (coordinates[keep] / s[keep]), atol=1e-8, rtol=1e-11)
    coordinate_error = float(np.max(abs(Vt @ (A.T @ z) - s * coordinates)))
    coordinate_bound = float(128 * eps * max(n, 53) * max(1., s[0] * np.linalg.norm(z)))
    assert coordinate_error <= coordinate_bound
    assert float(coordinates @ coordinates) <= float(z @ z) + coordinate_bound
    close(result['discarded_singular_max'], float(s[~keep].max(initial=0)))
    close(result['discarded_left_coordinate_energy'], float(np.sum(coordinates[~keep] ** 2)))
    close(result['original_box_discarded_direction_RMS_bound'], float(4 * np.sqrt(53) * s[~keep].max(initial=0)))
    projected_delta = D @ alpha
    close(saved['projected_delta'], projected_delta, atol=1e-6, rtol=1e-11)
    close(saved['residual'], delta - projected_delta, atol=1e-6, rtol=1e-11)
    residual = delta - projected_delta
    r = residual / np.sqrt(n)
    q = float(residual @ residual / n)
    correction = float(D[:, active.index(80)] @ residual / n)
    colnorm = np.einsum('ij,ij->j', D, D) / n
    z2 = float(delta @ delta / n)
    coeff = np.r_[-alpha / 2, (alpha.sum() - 1) / 2, .5]
    score_versions = active + [64, 83]
    centers = [m['scores'][str(v)] for v in score_versions]
    constant = float(-.5 * (z2 - alpha @ colnorm) + correction)
    high = (np.asarray(centers) + .00005) ** 2
    bound = 128 * eps * max(n, 53)
    def gamma(length):
        value = length * eps / (1 - length * eps)
        assert 0 <= value < .01
        return value
    g, b = gamma(2 * n + 4), gamma(2 * len(active) + 4)
    row_error = 2 * b * (abs(delta) + abs(D) @ abs(alpha))
    delta_rms = 2 * float(np.sqrt(row_error @ row_error / n))
    envelope_q = float(2 * (g * q + 2 * np.sqrt(q) * delta_rms + delta_rms ** 2))
    mean_abs_product = float(np.mean(abs(D[:, active.index(80)] * residual)))
    product_error = float(np.mean(abs(D[:, active.index(80)]) * row_error))
    correction_error = 2 * (g * mean_abs_product + product_error)
    constant_error = (.5 * (2 * g * z2 + abs(alpha) @ (2 * g * colnorm)) + correction_error
                      + 2 * gamma(2 * len(active) + 10) * (1 + z2 + abs(alpha) @ colnorm + abs(correction)))
    coefficient_error = gamma(2 * len(active) + 4) * (1 + abs(alpha).sum())
    interval_error = 2 * gamma(2 * (len(active) + 2) + 10) * (1 + abs(constant) + abs(coeff) @ high)
    envelope_h = float(2 * (constant_error + coefficient_error * high[-2] + interval_error))
    forward = dict(gamma_rows=g, gamma_columns=b, reconstruction_error_RMS_bound=delta_rms,
                   max_row_reconstruction_error=float(row_error.max(initial=0)),
                   constant_error=float(constant_error), reference_coefficient_error=float(coefficient_error),
                   interval_evaluation_error=float(interval_error), correction_error=float(correction_error),
                   mean_abs_correction_product=mean_abs_product,
                   exact_integer_input_differences=True, alpha_treated_as_fixed=True)
    for key, value in forward.items():
        if isinstance(value, bool):
            assert result['forward_bounds'][key] is value
        else:
            close(result['forward_bounds'][key], value)
    orth = float(np.max(abs(A.T @ r)))
    orth_bound = float(bound * s[0] * (1 + np.linalg.norm(z) + s[0] * np.linalg.norm(alpha)))
    # Independent wider-precision reduction treats the saved float64 alpha as exact.
    wide_alpha = alpha.astype(np.longdouble)
    wide_q = np.longdouble(0); wide_correction = np.longdouble(0)
    wide_z2 = np.longdouble(0); wide_colnorm = np.zeros(53, dtype=np.longdouble)
    for begin in range(0, n, 8192):
        block = D[begin:begin + 8192].astype(np.longdouble)
        dz = delta[begin:begin + 8192].astype(np.longdouble)
        residual = dz - block @ wide_alpha
        wide_q += np.sum(residual * residual, dtype=np.longdouble)
        wide_correction += np.sum(block[:, active.index(80)] * residual, dtype=np.longdouble)
        wide_z2 += np.sum(dz * dz, dtype=np.longdouble)
        wide_colnorm += np.sum(block * block, axis=0, dtype=np.longdouble)
    wide_q /= n; wide_correction /= n; wide_z2 /= n; wide_colnorm /= n
    wide_constant = -.5 * (wide_z2 - wide_alpha @ wide_colnorm) + wide_correction
    wide_coeff = np.r_[-wide_alpha / 2, (wide_alpha.sum() - 1) / 2, np.longdouble('.5')]
    wide_h = wide_constant + wide_coeff @ np.array(centers, dtype=np.longdouble) ** 2
    assert abs(float(wide_q) - q) <= envelope_q
    h = float(constant + coeff @ np.array(centers) ** 2)
    assert abs(float(wide_h) - h) <= envelope_h
    assert score_versions == result['score_versions']
    close(coeff, result['score_coefficients'], atol=0, rtol=0)
    for field, value in [('residual_MSE', q), ('residual_RMS', np.sqrt(q)), ('h80_center', h),
                         ('cross_moment_constant', constant), ('h80_nonorthogonality_correction', correction),
                         ('q_numeric_envelope', envelope_q), ('h_numeric_envelope', envelope_h),
                         ('orthogonality_max', orth), ('orthogonality_bound', orth_bound),
                         ('retained_condition', s[0] / s[keep][-1])]:
        close(result[field], value)
    lo_exact, hi_exact = decimal_interval(coeff, centers, constant)
    lo, hi = result['h80_interval']
    assert lo <= lo_exact - envelope_h + 1e-8 and hi >= hi_exact + envelope_h - 1e-8
    close([lo, hi], [lo_exact - envelope_h, hi_exact + envelope_h])
    rank_gap = bool(s[keep].min() > 2 * tau and (not (~keep).any() or s[~keep].max() < tau / 2))
    assert result['rank_gap_pass'] == rank_gap and result['original_score_half_width'] == .00005
    precision = bool(q > 100 * envelope_q and orth <= orth_bound and rank_gap)
    excludes_zero = bool(lo > 0 or hi < 0)
    minimum = min(abs(lo), abs(hi)) if excludes_zero else 0.
    reduction = minimum ** 2 / (q + envelope_q) if q + envelope_q > 0 else 0.
    maximum_rmse = m['scores']['80'] + .00005
    consistent = bool(reduction <= maximum_rmse ** 2)
    headroom = maximum_rmse - np.sqrt(max(0., maximum_rmse ** 2 - reduction)) if consistent else 0.
    close(result['minimum_abs_cross_moment'], minimum)
    close(result['conservative_oracle_MSE_reduction'], reduction)
    close(result['conservative_oracle_RMSE_headroom'], headroom)
    gate = dict(precision=precision, excludes_zero=excludes_zero, score_moment_consistent=consistent,
                minimum_half_second=headroom >= .5, passed=bool(precision and excludes_zero and consistent and headroom >= .5))
    assert result['gate'] == gate
    upper = None
    if q > envelope_q:
        optimistic_reduction = max(abs(lo), abs(hi)) ** 2 / (q - envelope_q)
        minimum_rmse = max(0., m['scores']['80'] - .00005)
        upper = float(min(maximum_rmse, np.sqrt(optimistic_reduction)) if optimistic_reduction > minimum_rmse ** 2
                      else minimum_rmse - np.sqrt(minimum_rmse ** 2 - optimistic_reduction))
    if upper is None:
        assert result['optimistic_oracle_RMSE_headroom'] is None
    else:
        close(result['optimistic_oracle_RMSE_headroom'], upper)
    decision = ('material_numerical_residual_information' if gate['passed'] else
                'insufficient_numerical_precision_or_consistency' if not precision or not consistent else
                'no_half_second_oracle_headroom_within_fixed_intervals' if upper is not None and upper < .5 else
                'insufficient_rounding_robust_evidence')
    assert result['decision'] == decision
    assert 'not certified exact mathematical rank' in result['interpretation']
    assert all(result[k] is True for k in ['no_candidate', 'no_optimization', 'no_CV_or_taxi_labels', 'no_upload'])
    assert 'missing final' in result['final_portability']
    write(OUT / 'independent_audit.json', dict(status='pass', source=art(__file__), identity=art(MANIFEST),
          result=art(OUT / 'result.json'), decomposition=result['decomposition'], rows=n, rank=result['rank'],
          source_and_actual_score_bindings=True, all_ID_and_raw_student_checks=True,
          SVD_verification='Stored right factors and fixed cutoff verified; no second decomposition/rank method',
          right_factor_max_error=factor_error, right_factor_bound=factor_bound,
          left_coordinate_equation_error=coordinate_error, left_coordinate_equation_bound=coordinate_bound,
          discarded_coordinate_caveat='Near-null left coordinates are basis-dependent; only moment/equation consistency is checked, not an exact mathematical null-space certificate.',
          longdouble_q=float(wide_q), longdouble_h80=float(wide_h),
          numeric_q_difference=abs(float(wide_q)-q), numeric_h_difference=abs(float(wide_h)-h),
          decimal_official_interval_before_numeric=[lo_exact, hi_exact], gate=gate,
          scope='Original-ranking numerical-tangent residual information only; no candidate, optimizer, taxi labels or final transfer validation.'))
    print(json.dumps({'status': 'pass', 'audit': art(OUT / 'independent_audit.json'), 'gate': gate}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
