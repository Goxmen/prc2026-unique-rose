"""Label-free portability diagnostic for v80's unavailable historical basis.

This is not a candidate and does not recover an exact model. No target column
is read. Nested affine weights are expanded while all nonlinear stage deltas
and rounding residuals remain explicit. Fixed ridge predicts only the missing
weighted residual from the fifteen already-portable version vectors.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_affine_recovery_diagnostic'
READY = [17, 18, 19, 20, 21, 38, 45, 46, 54, 57, 58, 64, 66, 68, 70]
GEOMETRY = ROOT / 'VALIDATION/runs/v3_20261012_target205_geometry'
V71 = ROOT / 'VALIDATION/runs/v3_20261011_physical_affine/submission_v71'
V73 = ROOT / 'VALIDATION/runs/v3_20261011_expanded_affine/submission_v73'
V79 = ROOT / 'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'
DESIGN = dict(
    scope='Descriptive model portability; no candidate, score optimization or final prediction',
    target='Effective missing weighted version residual, anchored at portable v38',
    inputs='All fifteen previously audited portable integer versions; differences from v38',
    ridge='No intercept; solve (X.T X/n + I) beta = X.T residual/n; one fixed ridge=1 second squared',
    tests=['Train Jan2026, hold out Jul2026', 'Train Jul2026, hold out Jan2026',
           'Train both ranking months2026, inspect historical Jan/Jul2025 fidelity'],
    no_label_columns=True, no_gate_or_hyperparameter_search=True, no_final_inference=True,
    retained_nonlinearity='Exact stage differences and separate v71/v73/v79 serialization residuals',
    caveats=['Existing CV versions3-9 use the archived original27 implementation.',
             'Historical CV fidelity is not taxi-out performance or independent model validation.',
             'Replacing the missing affine term alone does not replay changed nonlinear public corrections.',
             'Approximated missing models remain approximations; old forecasts cannot be copied to jittered final inputs.'])


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_paths():
    paths = [Path(__file__), GEOMETRY/'prepared.json', GEOMETRY/'ranking_basis.parquet',
             GEOMETRY/'cv_basis.parquet', V71/'geometry.json', V73/'geometry.json',
             V79/'manifest.json', ROOT/'data/ranking.parquet',
             ROOT/'VALIDATION/runs/v3_20261012_final_legacy_audit/audit.json',
             ROOT/'VALIDATION/runs/v3_20261011_physical_candidate/single/cv/predictions.parquet',
             ROOT/'VALIDATION/runs/v3_20261011_calls_only_candidate/submission_v68/cv.parquet']
    paths += [ROOT/f'submissions/unique-rose_v{v}.parquet' for v in [66, 68, 71, 73, 79, 80]]
    return paths


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'identity.json').exists()
    write(OUT/'identity.json', dict(design=DESIGN, sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths()}))


def stats(values):
    z = np.asarray(values, float)
    return dict(n=len(z), mean=float(z.mean()), rms=float(np.sqrt(np.mean(z*z))),
                absolute_quantiles={str(q):float(np.quantile(abs(z),q)) for q in [.5,.9,.99,.999,1.]})


def weights(path):
    return {int(k):float(v) for k,v in read(path)['weights'].items()}


def aligned(path, ids, idcol='MVT_ID_mvt', col='TAXITIME_SEC_mvt'):
    f = pd.read_parquet(path, columns=[idcol,col]).set_index(idcol)
    assert f.index.is_unique and set(f.index)==set(ids)
    return f.loc[ids,col].to_numpy(float)


def fit(X, y):
    gram = X.T@X/len(X) + np.eye(X.shape[1])
    rhs = X.T@y/len(X)
    beta = np.linalg.solve(gram,rhs)
    np.testing.assert_allclose(gram@beta,rhs,atol=1e-7,rtol=1e-10)
    return beta


def run():
    identity = read(OUT/'identity.json')
    assert identity['design']==DESIGN and not (OUT/'result.json').exists()
    for p,h in identity['sources'].items():
        assert sha(ROOT/p)==h,p
    assert read(ROOT/'VALIDATION/runs/v3_20261012_final_legacy_audit/audit.json')['status']=='pass'
    w71,w73,w79 = weights(V71/'geometry.json'),weights(V73/'geometry.json'),weights(V79/'manifest.json')
    leaf = sorted(w71)
    assert set(w73)==set(leaf)|{72}
    assert set(w79)==(set(leaf)-{66,68})|{72,74,76,77,78}
    a73 = sum(w79[v] for v in [74,76,77,78])
    a71 = w79[72] + a73*w73[72]
    effective = {v:w79.get(v,0)+a73*w73[v]+a71*w71[v] for v in leaf}
    assert abs(sum(effective.values())-1)<1e-9
    missing = sorted(set(leaf)-set(READY))
    prepared = read(GEOMETRY/'prepared.json')
    assert prepared['ranking']['sha256']==sha(GEOMETRY/'ranking_basis.parquet')
    assert prepared['cv']['sha256']==sha(GEOMETRY/'cv_basis.parquet')
    ranking = pd.read_parquet(GEOMETRY/'ranking_basis.parquet')
    ids = ranking.mvt_id.to_numpy()
    assert len(ids)==344841 and ranking.mvt_id.is_unique
    for v in [66,68,71,73,79,80]:
        ranking[f'v{v}'] = aligned(ROOT/f'submissions/unique-rose_v{v}.parquet',ids)
    metadata = pd.read_parquet(ROOT/'data/ranking.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','ADEP_mvt'])
    metadata = metadata.set_index('MVT_ID_mvt').loc[ids]
    month = pd.to_datetime(metadata.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m').to_numpy()
    assert set(month)=={'2026-01','2026-07'}
    raw71 = sum(w*ranking[f'v{v}'].to_numpy() for v,w in w71.items())
    raw73 = sum(w*ranking[f'v{v}'].to_numpy() for v,w in w73.items())
    raw79 = sum(w*ranking[f'v{v}'].to_numpy() for v,w in w79.items())
    # Keep these residuals rather than pretending a rounded affine expression
    # is itself one exact linear combination. The source builders independently
    # verified the original stage rounding including half-integer ties.
    e71,e73,e79 = (ranking[f'v{v}'].to_numpy()-raw for v,raw in [(71,raw71),(73,raw73),(79,raw79)])
    delta = {str(v):ranking[f'v{v}'].to_numpy()-ranking[f'v{p}'].to_numpy()
             for v,p in [(72,71),(74,73),(76,74),(77,76),(78,77),(80,79)]}
    coefficients = {'72':a71,'74':a73,'76':sum(w79[v] for v in [76,77,78]),
                    '77':w79[77]+w79[78],'78':w79[78],'80':1.}
    reconstructed = sum(effective[v]*ranking[f'v{v}'].to_numpy() for v in leaf)
    reconstructed += a71*e71+a73*e73+e79
    reconstructed += sum(coefficients[k]*d for k,d in delta.items())
    error = float(np.max(abs(reconstructed-ranking.v80.to_numpy())))
    assert error < 1e-7,error
    anchor = ranking.v38.to_numpy()
    targets = {v:effective[v]*(ranking[f'v{v}'].to_numpy()-anchor) for v in missing}
    teacher = sum(targets.values())
    predictors = [v for v in READY if v!=38]
    X = np.column_stack([ranking[f'v{v}'].to_numpy()-anchor for v in predictors])
    monthly = {}
    for training,held in [('2026-01','2026-07'),('2026-07','2026-01')]:
        tr,te = month==training,month==held
        beta = fit(X[tr],teacher[tr])
        monthly[training+'->'+held] = dict(training_rows=int(tr.sum()),held_rows=int(te.sum()),
            beta={str(v):float(b) for v,b in zip(predictors,beta)},
            zero_residual_baseline=stats(teacher[te]),ridge_residual_error=stats(X[te]@beta-teacher[te]))
    beta = fit(X,teacher)
    # No y column is read, even in the historical diagnostic.
    cv = pd.read_parquet(GEOMETRY/'cv_basis.parquet',columns=['mvt_id','month','day','airport',*[f'v{v}' for v in leaf if v not in [66,68]]])
    assert len(cv)==344419 and cv.mvt_id.is_unique
    cv['v66'] = aligned(ROOT/'VALIDATION/runs/v3_20261011_physical_candidate/single/cv/predictions.parquet',cv.mvt_id.to_numpy(),'mvt_id','submitted')
    cv['v68'] = aligned(ROOT/'VALIDATION/runs/v3_20261011_calls_only_candidate/submission_v68/cv.parquet',cv.mvt_id.to_numpy(),'mvt_id','submitted')
    cv_anchor = cv.v38.to_numpy()
    cv_teacher = sum(effective[v]*(cv[f'v{v}'].to_numpy()-cv_anchor) for v in missing)
    cv_X = np.column_stack([cv[f'v{v}'].to_numpy()-cv_anchor for v in predictors])
    cv_pred = cv_X@beta
    historical = {m:dict(zero_residual_baseline=stats(cv_teacher[ix]),ridge_residual_error=stats(cv_pred[ix]-cv_teacher[ix]))
                  for m in ['2025-01','2025-07'] for ix in [cv.month.eq(m).to_numpy()]}
    output = dict(design=DESIGN,identity_sha256=sha(OUT/'identity.json'),
        leaf_versions=leaf,portable_versions=READY,missing_versions=missing,
        effective_leaf_weights={str(v):effective[v] for v in leaf},
        stage_delta_coefficients=coefficients,rounding_coefficients={'71':a71,'73':a73,'79':1.},
        exact_v80_reconstruction_max_error=error,
        missing_aggregate=stats(teacher),missing_weight_sum=sum(effective[v] for v in missing),
        missing_weight_l1=sum(abs(effective[v]) for v in missing),
        individual_missing_effects={str(v):dict(weight=effective[v],**stats(targets[v])) for v in missing},
        monthly_transfer_fidelity=monthly,all_ranking_training_fidelity=stats(X@beta-teacher),
        all_ranking_beta={str(v):float(b) for v,b in zip(predictors,beta)},
        historical_prediction_only_fidelity=historical,
        candidate=False,taxi_out_performance_not_evaluated=True,no_final_prediction=True)
    write(OUT/'result.json',output)
    pd.DataFrame(dict(mvt_id=ids,month=month,missing_teacher=teacher,ridge_prediction=X@beta)).to_parquet(OUT/'ranking_fidelity.parquet',index=False)
    pd.DataFrame(dict(mvt_id=cv.mvt_id,month=cv.month,missing_teacher=cv_teacher,ridge_prediction=cv_pred)).to_parquet(OUT/'historical_fidelity.parquet',index=False)
    print(json.dumps(dict(missing=len(missing),reconstruction_error=error,
        missing_rms=output['missing_aggregate']['rms'],
        transfer_rms={k:v['ridge_residual_error']['rms'] for k,v in monthly.items()},
        historical_rms={k:v['ridge_residual_error']['rms'] for k,v in historical.items()})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
