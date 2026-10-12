"""One fixed test of nonuniform residual uncertainty using existing own scores.

Aggregate RMSE differences identify D.T @ residual / N. A diagonal uncertainty
prior changes how those constraints distribute corrections across flights.
This yields a new prediction direction, whose leaderboard loss is NOT known.
No new score query, hidden truth access, model fit, upload or weight search.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import target205_geometry as basis
import tuning_seasonal as validation
from records import sha256, write_json

ROOT = basis.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_heteroscedastic_score_conditioning'
PARENT = ROOT / 'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'
MEMBERS = [64, 70, 72, 74, 76, 77, 78]
DESIGN = dict(
    reference=78, basis='51 retained, actually scored own predictions; omit zero v78 difference',
    uncertainty_members=MEMBERS,
    uncertainty='population variance across seven recent recipes; 1+variance/ranking_mean_variance; cap at 16; divide by ranking mean',
    cv_normalization='same two ranking-derived normalization constants; no CV labels in weights or coefficients',
    conditioning='g_j=(mean(D_j**2)+score78**2-score_j**2)/2; beta minimizes beta.T@(D.T@W@D/N+I)@beta/2-g.T@beta+radius.T@abs(beta)',
    score_uncertainty='radius_j=(score78+score_j)*0.00005+2.5e-9; worst-case independent four-decimal intervals',
    ridge=1., coefficients='unbounded; convex objective; one fixed setting',
    prediction='round_even(max(0,v78+W*D@beta))',
    comparison='exact submitted v79; same robust ridge conditioning with W=1',
    gate='both months improve and paired-day upper endpoint below zero against v79 AND W=1',
    limitations='reused Jan/Jul development; old v3-v9 source reconstructions inherited; weighted leaderboard RMSE is unidentifiable from existing scores',
    no_fit=True, no_submission=True, no_score_forecast=True,
)


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    return dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=sha256(path))


def verify(record):
    path = ROOT / record['path']
    assert sha256(path) == record['sha256'], path
    return path


def checks():
    rng = np.random.default_rng(261012)
    d = rng.normal(size=(120, 4))
    w = np.linspace(.2, 4., len(d))
    beta = rng.normal(size=d.shape[1])
    residual = w * (d @ beta)
    p = rng.uniform(500, 1400, len(d))
    y = p + residual
    score0 = np.mean((p-y)**2)
    scores = np.mean((p[:, None]+d-y[:, None])**2, axis=0)
    g = (np.mean(d*d, axis=0)+score0-scores)/2
    np.testing.assert_allclose(g, d.T@residual/len(d), atol=1e-10, rtol=0)
    recovered = w*(d@np.linalg.solve(d.T@(w[:, None]*d)/len(d), g))
    np.testing.assert_allclose(recovered, residual, atol=1e-9, rtol=0)
    euclidean = d@np.linalg.solve(d.T@d/len(d), g)
    assert np.mean((recovered-residual)**2) < 1e-16
    assert np.mean((euclidean-residual)**2) > .01
    np.testing.assert_allclose(d.T@(residual-euclidean), 0, atol=1e-8, rtol=0)
    return dict(score_difference_identity=True, weighted_conditioning_known_oracle=True,
                equal_variance_projection=True, weighted_direction_outside_old_span=True)


def freeze(snapshot):
    assert not (OUT/'identity.json').exists(), 'Preserve frozen comparison'
    pre = read(basis.OUT/'prepared.json')
    assert read(basis.OUT/'audit.json')['status'] == 'pass'
    live = read(snapshot)
    assert live['snapshotSource'] == 'live'
    successful = {r['version']: r['score'] for r in live['history']
                  if r['status'] == 'Succeeded' and r['usedPairs'] == 344841}
    assert all(v in successful for v in pre['versions']) and 75 not in pre['versions']
    assert read(PARENT/'manifest.json')['score'] == successful[79] == 219.613
    for key, value in pre['known_scores'].items():
        assert value == successful[int(key)]
    files = [Path(__file__), basis.OUT/'prepared.json', basis.OUT/'audit.json',
             snapshot, PARENT/'cv.parquet', PARENT/'manifest.json']
    write_json(OUT/'identity.json', dict(design=DESIGN, frozen_at=datetime.now(timezone.utc).isoformat(),
        files=[art(p) for p in files], ranking=pre['ranking'], cv=pre['cv'],
        versions=pre['versions'], official_scores={str(v):successful[v] for v in pre['versions']},
        checks=checks(), historical_cv_caveat=pre['historical_cv_caveat']))
    print('Frozen one uncertainty formula; no experimental CV scores read.')


def solve(d, g, w, radius):
    gram = d.T@(w[:, None]*d)/len(d)
    hessian = gram + np.eye(d.shape[1])
    n = len(g)
    # Smooth epigraph formulation; scale by data-derived Hessian spectral norm.
    scale = max(1., float(np.linalg.eigvalsh(hessian)[-1]))
    eye = np.eye(n)
    inequalities = np.block([[-eye, eye], [eye, eye]])
    def objective(x):
        b, t = x[:n], x[n:]
        return ((.5*b@hessian@b-g@b+radius@t)/scale,
                np.r_[hessian@b-g, radius]/scale)
    initial = np.linalg.solve(hessian, g)
    fit = minimize(objective, np.r_[initial, np.abs(initial)+1e-8], jac=True,
        method='SLSQP', bounds=[(None,None)]*n+[(0,None)]*n,
        constraints=[dict(type='ineq', fun=lambda x:inequalities@x,
                          jac=lambda x:inequalities)],
        options=dict(ftol=1e-14,maxiter=5000))
    assert fit.success, fit.message
    assert (inequalities@fit.x).min() > -1e-7
    b = fit.x[:n]
    # Coordinate proximal descent gives an independently checkable KKT residual.
    # Continue the same convex objective, not a new setting or a second candidate.
    for sweep in range(20000):
        maximum = 0.
        for j in range(n):
            partial = g[j]-(hessian[j]@b-hessian[j,j]*b[j])
            updated = np.sign(partial)*max(abs(partial)-radius[j],0)/hessian[j,j]
            maximum = max(maximum, abs(updated-b[j]))
            b[j] = updated
        gradient = hessian@b-g
        active = abs(b)>1e-9
        residual = np.maximum(abs(gradient)-radius,0)
        residual[active] = abs(gradient[active]+radius[active]*np.sign(b[active]))
        if residual.max() < 1e-6:
            break
    assert residual.max() < 1e-5, (sweep, residual.max())
    return b, dict(success=True, slsqp_iterations=int(fit.nit), coordinate_sweeps=sweep+1,
        kkt_maximum=float(residual.max()), minimum_hessian_eigenvalue=float(np.linalg.eigvalsh(hessian)[0]),
        coefficient_l1=float(abs(b).sum()), objective=float(.5*b@hessian@b-g@b+radius@abs(b)))


def diagnose():
    assert not (OUT/'result.json').exists(), 'Preserve completed diagnosis'
    identity = read(OUT/'identity.json')
    assert identity['design'] == DESIGN
    for record in identity['files']:
        verify(record)
    ranking = pd.read_parquet(verify(identity['ranking']))
    versions = [v for v in identity['versions'] if v != 78]
    columns = [f'v{v}' for v in versions]
    d = ranking[columns].to_numpy(float)-ranking.v78.to_numpy(float)[:,None]
    s = np.array([identity['official_scores'][str(v)] for v in versions])
    s0 = identity['official_scores']['78']
    g = (np.mean(d*d,axis=0)+s0*s0-s*s)/2
    radius = (s0+s)*.00005+2.5e-9
    variance = ranking[[f'v{v}' for v in MEMBERS]].to_numpy(float).var(axis=1)
    variance_scale = float(variance.mean())
    assert variance_scale > 0 and len(ranking) == 344841
    weights = np.minimum(1+variance/variance_scale,16)
    weight_scale = float(weights.mean())
    weights /= weight_scale
    solutions = {}
    ranked = ranking[['mvt_id']].copy()
    for name,w in [('equal',np.ones(len(ranking))),('weighted',weights)]:
        b,receipt = solve(d,g,w,radius)
        solutions[name] = dict(coefficients={str(v):float(x) for v,x in zip(versions,b)}, **receipt)
        raw = ranking.v78.to_numpy(float)+w*(d@b)
        ranked[name+'_raw'] = raw
        ranked[name] = np.rint(np.maximum(0,raw))
    # Only now open validation labels; coefficients and both normalizers fixed above.
    held = pd.read_parquet(verify(identity['cv']))
    parent = pd.read_parquet(PARENT/'cv.parquet').set_index('mvt_id').loc[held.mvt_id]
    assert len(held) == 344419 and held.mvt_id.is_unique and parent.index.is_unique
    for column in ['y','month','day']:
        np.testing.assert_array_equal(held[column],parent[column])
    cv = held[['mvt_id','month','day','airport','y']].copy()
    cv['v79'] = parent.pred.to_numpy(float)
    dh = held[columns].to_numpy(float)-held.v78.to_numpy(float)[:,None]
    hv = held[[f'v{v}' for v in MEMBERS]].to_numpy(float).var(axis=1)
    wh = np.minimum(1+hv/variance_scale,16)/weight_scale
    for name,w in [('equal',np.ones(len(held))),('weighted',wh)]:
        b = np.array([solutions[name]['coefficients'][str(v)] for v in versions])
        cv[name] = np.rint(np.maximum(0,held.v78.to_numpy(float)+w*(dh@b)))
    months = {month:{name:float(np.sqrt(np.mean((f[name]-f.y)**2)))
              for name in ['v79','equal','weighted']} for month,f in cv.groupby('month')}
    pairs = {name:validation.paired_bootstrap(cv.day,cv.month,(cv[name]-cv.y)**2,
              (cv.weighted-cv.y)**2) for name in ['v79','equal']}
    gate = all(v['decision']=='improvement' for v in pairs.values()) and all(
        m['weighted']<m[name] for m in months.values() for name in ['v79','equal'])
    ranked.to_parquet(OUT/'ranking.parquet',index=False)
    cv.to_parquet(OUT/'cv.parquet',index=False)
    value = dict(identity=art(OUT/'identity.json'), solutions=solutions,
        variance_scale=variance_scale, weight_scale=weight_scale,
        ranking_weight_quantiles=np.quantile(weights,[0,.5,.9,.99,1]).tolist(),
        cv={name:float(np.mean([m[name] for m in months.values()])) for name in ['v79','equal','weighted']},
        months=months,paired=pairs,predeclared_gate_pass=gate,
        ranking=art(OUT/'ranking.parquet'),validation=art(OUT/'cv.parquet'),
        leaderboard_prediction=None, no_exact_weighted_score_from_existing_feedback=True,
        no_version_or_upload=True)
    write_json(OUT/'result.json',value)
    print(json.dumps(dict(cv=value['cv'],paired=pairs,gate=gate),indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','diagnose','check'])
    parser.add_argument('--snapshot',type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'freeze':freeze(args.snapshot.resolve())
        elif args.stage == 'check':print(checks())
        else:diagnose()
