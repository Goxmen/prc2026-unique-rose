"""Independent dual optimization, arithmetic and label audit for fixed weighting."""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
from records import sha256, write_json

OUT = ROOT/'VALIDATION/runs/v3_20261012_heteroscedastic_score_conditioning'
PARENT = ROOT/'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'


def read(path):
    return json.loads(path.read_text())


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def verify(record):
    path = ROOT/record['path']
    assert sha256(path) == record['sha256'], path
    return path


def main():
    assert not (OUT/'audit.json').exists()
    result = read(OUT/'result.json')
    identity = read(verify(result['identity']))
    for record in identity['files']:
        verify(record)
    rank = pd.read_parquet(verify(identity['ranking']))
    cv = pd.read_parquet(verify(identity['cv']))
    ranked = pd.read_parquet(verify(result['ranking']))
    scored = pd.read_parquet(verify(result['validation']))
    assert len(rank) == len(ranked) == 344841 and len(cv) == len(scored) == 344419
    assert rank.mvt_id.is_unique and cv.mvt_id.is_unique
    np.testing.assert_array_equal(rank.mvt_id,ranked.mvt_id)
    pd.testing.assert_frame_equal(cv[['mvt_id','month','day','airport','y']],
                                 scored[['mvt_id','month','day','airport','y']],check_exact=True)
    versions = [v for v in identity['versions'] if v!=78]
    members = [64,70,72,74,76,77,78]
    assert identity['design']['uncertainty_members'] == members
    d = np.column_stack([rank[f'v{v}'].to_numpy(float)-rank.v78 for v in versions])
    scores = identity['official_scores']
    s = np.array([scores[str(v)] for v in versions])
    g = np.array([(np.dot(d[:,j],d[:,j])/len(rank)+scores['78']**2-s[j]**2)/2
                  for j in range(len(versions))])
    radii = (s+scores['78'])*.00005+2.5e-9
    x = np.column_stack([rank[f'v{v}'] for v in members]).astype(float)
    variance = np.sum((x-x.mean(axis=1)[:,None])**2,axis=1)/7
    a = float(variance.mean())
    raw_weight = np.clip(1+variance/a,None,16)
    b = float(raw_weight.mean())
    assert abs(a-result['variance_scale']) < 1e-8 and abs(b-result['weight_scale']) < 1e-12
    w = raw_weight/b
    dual_checks = {}
    for name,weight in [('equal',np.ones(len(rank))),('weighted',w)]:
        h = np.einsum('ij,i,ik->jk',d,weight,d,optimize=True)/len(rank)+np.eye(len(versions))
        # Fenchel dual: minimize ||L^-1(g-z)||^2 with each z in [-radius,+radius].
        # This implementation does not import or call the primal optimizer.
        inverse_cholesky = solve_triangular(np.linalg.cholesky(h),np.eye(len(versions)),lower=True)
        fit = lsq_linear(inverse_cholesky,inverse_cholesky@g,
                         bounds=(-radii,radii),method='bvls',tol=1e-12,max_iter=1000)
        assert fit.success,fit.message
        beta = np.linalg.solve(h,g-fit.x)
        saved = np.array([result['solutions'][name]['coefficients'][str(v)] for v in versions])
        primal = .5*saved@h@saved-g@saved+radii@abs(saved)
        dual = -.5*(g-fit.x)@beta
        gap = float(primal-dual)
        assert -1e-6 < gap < 1e-5, (name,gap)
        np.testing.assert_allclose(saved,beta,atol=2e-3,rtol=0)
        prediction = rank.v78.to_numpy(float)+weight*np.sum(d*saved,axis=1)
        np.testing.assert_allclose(prediction,ranked[name+'_raw'],atol=1e-8,rtol=0)
        np.testing.assert_array_equal(np.rint(np.maximum(prediction,0)),ranked[name])
        held_d = np.column_stack([cv[f'v{v}'].to_numpy(float)-cv.v78 for v in versions])
        if name=='weighted':
            hx = np.column_stack([cv[f'v{v}'] for v in members]).astype(float)
            hv = np.sum((hx-hx.mean(axis=1)[:,None])**2,axis=1)/7
            hw = np.clip(1+hv/a,None,16)/b
        else:
            hw = np.ones(len(cv))
        hp = cv.v78.to_numpy(float)+hw*np.sum(held_d*saved,axis=1)
        np.testing.assert_array_equal(np.rint(np.maximum(hp,0)),scored[name])
        dual_checks[name] = dict(duality_gap=gap,coefficient_max_difference=float(abs(saved-beta).max()),
                                independent_dual_success=True)
    parent = pd.read_parquet(PARENT/'cv.parquet').set_index('mvt_id').loc[scored.mvt_id]
    np.testing.assert_array_equal(parent.pred,scored.v79)
    with duckdb.connect(config={'threads':1}) as con:
        con.register('s',scored)
        for name in ['v79','equal','weighted']:
            for month,value in con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from s group by month').fetchall():
                assert abs(value-result['months'][month][name])<1e-8
    assert result['leaderboard_prediction'] is None and result['no_version_or_upload']
    write_json(OUT/'audit.json',dict(status='pass',result=art(OUT/'result.json'),
        identity=art(OUT/'identity.json'),source=art(Path(__file__).resolve()),
        rows=344419,ranking_rows=344841,dual_optimization=dual_checks,
        exact_parent_labels_ids_integer_predictions=True,independent_SQL_scores=True,
        no_claim_of_known_weighted_leaderboard_loss=True))
    print('PASS: dual optimum, all-row predictions, parent truth and SQL scores')


if __name__=='__main__':
    with threadpool_limits(1):main()
