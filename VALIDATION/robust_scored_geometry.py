"""Read-only affine score geometry with explicit score-rounding uncertainty.

Compare the retained [-3,4]/ridge1 design with one fixed convex objective:
minimize worst-case MSE over four-decimal score intervals, with sum(w)=1.
Absolute coefficients pay their actual score-uncertainty radius. No coefficient
box, ridge, new prediction direction, hidden label, upload or version exists.
All subsequent serialization error is measured separately and bounded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import scored_submission_geometry as geometry
from records import sha256,write_json

ROOT=geometry.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_robust_scored_geometry'
PREPARED=ROOT/'VALIDATION/runs/v3_20261011_daily_models_geometry/prepared.json'
DESIGN=dict(parent='same retained51-vector basis, restricted to already successful official scores',
    primary='convex worst-case affine MSE under independent four-decimal score intervals',
    constraints='sum(weights)=1; epigraph t>=abs(weights); no box or arbitrary ridge',
    comparison='same available basis with retained [-3,4] bounds and ridge1',
    no_new_directions=True,no_score_imputation=True,no_fits=True,no_upload=True,
    no_candidate=True,cv='report reused Jan/Jul separately; selection uses official feedback',
    rounding='measure exact RMS change from raw to nonnegative integer predictions; triangle bound')


def art(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def verify(record):
    path=ROOT/record['path'];assert sha256(path)==record['sha256'];return path


def read(path):return json.loads(Path(path).read_text())


def solve(P,scores,versions,initial):
    at=versions.index(64);D=P-P[:,at,None];G=D.T@D/len(P)
    mse0=scores[at]**2;c=(scores**2-mse0-np.diag(G))/2
    radius=np.abs(scores)*1e-4+2.5e-9
    n=len(versions); eye=np.eye(n)
    inequality=np.block([[-eye,eye],[eye,eye]])
    equality=np.r_[np.ones(n),np.zeros(n)]
    def fun(x):
        w,t=x[:n],x[n:]
        value=mse0+2*c@w+w@G@w+radius@t
        gradient=np.r_[2*(c+G@w),radius]
        return float(value/10000),gradient/10000
    result=minimize(fun,np.r_[initial,np.abs(initial)+1e-8],jac=True,method='SLSQP',
        bounds=[(None,None)]*n+[(0,None)]*n,
        constraints=[dict(type='eq',fun=lambda x:equality@x-1,jac=lambda x:equality),
                     dict(type='ineq',fun=lambda x:inequality@x,jac=lambda x:inequality)],
        options=dict(ftol=1e-12,maxiter=5000))
    w=result.x[:n]
    assert result.success,result.message
    assert abs(w.sum()-1)<1e-8 and (inequality@result.x).min()>-1e-7
    raw=P@w
    mse=float(mse0+2*c@w+w@G@w)
    expanded=float(w@(scores**2)-w@np.mean(P*P,axis=0)+np.mean(raw*raw))
    np.testing.assert_allclose(mse,expanded,atol=1e-5,rtol=0)
    uncertainty=float(np.abs(w)@radius)
    # Conservative accumulation allowance; both algebraic forms also agree.
    scale=float(abs(mse0)+2*np.abs(c)@np.abs(w)+np.abs(w)@np.abs(G)@np.abs(w))
    numerical=64*np.finfo(float).eps*len(P)*max(scale,1.)
    assert mse+uncertainty+numerical>=0
    interval=[float(np.sqrt(max(0,mse-uncertainty-numerical))),float(np.sqrt(mse+uncertainty+numerical))]
    saved=np.rint(np.maximum(raw,0));distance=float(np.sqrt(np.mean((saved-raw)**2)))
    return dict(weights={str(v):float(a) for v,a in zip(versions,w)},
        projected_rmse=float(np.sqrt(max(mse,0))),mse=mse,score_mse_radius=uncertainty,
        numerical_mse_allowance=numerical,raw_rmse_interval=interval,
        serialization_rms=distance,serialized_rmse_interval=[max(0,interval[0]-distance),interval[1]+distance],
        coefficient_l1=float(np.abs(w).sum()),minimum_weight=float(w.min()),maximum_weight=float(w.max()),
        negative_predictions=int((raw<0).sum()),minimum_prediction=float(raw.min()),maximum_prediction=float(raw.max()),
        optimizer=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            epigraph_slack=float(np.max(result.x[n:]-np.abs(w)))),
        independent_expanded_mse=expanded),raw


def checks():
    rng=np.random.default_rng(81731);y=rng.uniform(300,1800,4096)
    common=rng.normal(0,100,len(y));epsilon=rng.normal(0,.3,len(y))
    P=np.column_stack([y+common,y+.9*common+epsilon,y+common+rng.normal(0,20,len(y))])
    s=np.sqrt(np.mean((P-y[:,None])**2,axis=0));versions=[64,70,74]
    result,raw=solve(P,np.round(s,4),versions,np.array([1.,0.,0.]))
    assert result['maximum_weight']>4
    actual=float(np.sqrt(np.mean((raw-y)**2)));lo,hi=result['raw_rmse_interval']
    assert lo<=actual<=hi
    serialized=np.rint(np.maximum(raw,0));actual_serialized=float(np.sqrt(np.mean((serialized-y)**2)))
    lo,hi=result['serialized_rmse_interval'];assert lo<=actual_serialized<=hi
    # Any feasible known blend's actual loss must lie in its own score bound.
    w=np.array([1.2,-.6,.4]); estimate=float(w@np.round(s,4)**2-w@np.mean(P*P,axis=0)+np.mean((P@w)**2))
    rad=float(np.abs(w)@(np.abs(np.round(s,4))*1e-4+2.5e-9))
    np.testing.assert_array_less(abs(estimate-np.mean((P@w-y)**2)),rad)
    return dict(known_target_oracle=True,optimal_coefficient_outside_old_box=True,
        score_rounding_and_actual_serialization_bounds=True,signed_feasible_weight_identity=True)


def diagnose(snapshot):
    assert not (OUT/'result.json').exists(),'Preserve completed diagnosis'
    pre=read(PREPARED);live=read(snapshot)
    assert read(PREPARED.with_name('audit.json'))['status']=='pass'
    assert live['snapshotSource']=='live',live['snapshotSource']
    successful={r['version']:r for r in live['history'] if r['status']=='Succeeded' and r['usedPairs']==344841}
    versions=[v for v in pre['versions'] if v in successful]
    missing=[v for v in pre['versions'] if v not in successful]
    assert set(missing)<={75,76,77} and 64 in versions
    scores=np.array([successful[v]['score'] for v in versions],float)
    for v,s in zip(versions,scores):
        if str(v) in pre['known_scores']:assert abs(s-pre['known_scores'][str(v)])<1e-8
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(design=DESIGN,source=art(__file__),prepared=art(PREPARED),snapshot=art(snapshot),
        available_versions=versions,unscored_excluded=missing,checks=checks(),
        official_scores={str(v):float(s) for v,s in zip(versions,scores)},historical_cv_caveat=pre['historical_cv_caveat'])
    write_json(OUT/'design.json',design)
    ranking=pd.read_parquet(verify(pre['ranking']));held=pd.read_parquet(verify(pre['cv']))
    columns=[f'v{v}' for v in versions];P=ranking[columns].to_numpy(float)
    assert len(P)==344841 and len(held)==344419
    baseline,base_raw=geometry.solve(P,scores,versions,-3.,4.,1.)
    initial=np.array([baseline['weights'][str(v)] for v in versions])
    robust,raw=solve(P,scores,versions,initial)
    cv=held[['mvt_id','month','day','airport','y']].copy();cv['v74']=held.v74
    output=ranking[['mvt_id']].copy()
    for name,solution,prediction in [('bounded',baseline,base_raw),('robust',robust,raw)]:
        w=np.array([solution['weights'][str(v)] for v in versions])
        cv[name]=np.rint(np.maximum(held[columns].to_numpy(float)@w,0))
        output[name+'_raw']=prediction;output[name]=np.rint(np.maximum(prediction,0))
    month_scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['v74','bounded','robust']}
        for m,g in cv.groupby('month')}
    cv.to_parquet(OUT/'diagnostic_cv.parquet',index=False);output.to_parquet(OUT/'diagnostic_ranking.parquet',index=False)
    value=dict(design=art(OUT/'design.json'),bounded=baseline,robust=robust,
        current_best=float(live['team']['bestScore']),checked_at=live['updatedAt'],months=month_scores,
        cv={c:float(np.mean([v[c] for v in month_scores.values()])) for c in ['v74','bounded','robust']},
        available_versions=versions,unscored_excluded=missing,
        ranking=art(OUT/'diagnostic_ranking.parquet'),validation=art(OUT/'diagnostic_cv.parquet'),
        eligible_for_upload=False,next='Independent final/source/schema/CV audit required before any candidate; no file reserved')
    write_json(OUT/'result.json',value)
    print(dict(bounded=baseline['projected_rmse'],robust=robust,cv=value['cv']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['check','diagnose'])
    p.add_argument('--snapshot',type=Path);a=p.parse_args()
    with threadpool_limits(1):
        if a.stage=='check':print(checks())
        else:diagnose(a.snapshot.resolve())
