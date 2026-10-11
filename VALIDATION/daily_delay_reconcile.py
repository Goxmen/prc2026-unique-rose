"""Condition retained predictions on public EUROCONTROL daily delay totals.

Independent implementation of a lead from il-miscusi/prc-2026-gracious-tornado
(GPLv3), research/final/levers/daily.py, read 2026-10-11. No source copied.
This first screen uses analytic Gaussian/mixture moments, not its discrete
forward-backward posterior. Every fitted source-mismatch parameter and held
taxi mean uses only the outer fold's allowed training labels. Prior residual
scales are fixed before scoring; held-out labels are read for scoring only.
EUROCONTROL inputs remain unmodified and are not redistributed.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.special import ndtr
from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VALIDATION/runs/v3_20261011_daily_delay'
SOURCE = ROOT / 'data/external/eurocontrol_performance_20261011'
PARENT = ROOT / 'VALIDATION/runs/v3_20261011_physical_affine/submission_v71'
FOLDS = {'2025-01': [f'2025-{m:02d}' for m in range(3, 12)],
         '2025-07': ['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
DESIGN = dict(parent='v71', ordinary_sigma=200., common_day_sigma=15.,
              long_threshold=3600., copy_airport='LIRF', copy_threshold=3600.,
              copy_max_gap=14*3600., noise_floor=60., mismatch_scale_floor=600.,
              negative_output='clip zero', calibration='allowed outer training labels only',
              screen='fixed analytic Gaussian moments; no hyperparameter scan')

def artifact(p):
    return dict(path=str(Path(p).relative_to(ROOT)), sha256=sha256(p))

def load_rows(paths, labelled=True):
    cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt',
          'SCHED_TIME_UTC_mvt','FLIGHT_ID_mvt','LOBT_flt']
    if labelled: cols += ['TAXITIME_SEC_mvt']
    frames=[]
    for path in paths:
        f=pd.read_parquet(path,columns=cols,filters=[('PHASE_mvt','==','DEP')])
        out=pd.DataFrame(dict(mvt_id=f.MVT_ID_mvt.to_numpy(),airport=f.ADEP_mvt.to_numpy(),
            day=f.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d').to_numpy(),
            month=f.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').to_numpy(),
            gap=(f.MVT_TIME_UTC_mvt-f.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy(),
            has_id=f.FLIGHT_ID_mvt.notna().to_numpy(),
            nm_gap=(f.MVT_TIME_UTC_mvt-f.LOBT_flt).dt.total_seconds().to_numpy()))
        if labelled: out['y']=f.TAXITIME_SEC_mvt.to_numpy(float)
        frames.append(out)
    out=pd.concat(frames,ignore_index=True)
    assert out.mvt_id.is_unique and out.gap.notna().all()
    return out

def public(year):
    f=pd.read_csv(SOURCE/f'all_pre_departure_delays_{year}.csv')
    f=f.rename(columns={'APT_ICAO':'airport','FLT_DATE':'day','FLT_DEP_IFR_2':'n_public'})
    f['total']=60*f.DLY_ALL_PRE_2
    assert not f.duplicated(['airport','day']).any()
    return f[['airport','day','n_public','total']]

def aggregate_truth(rows, source):
    f=rows.assign(contribution=np.maximum(0,rows.gap-rows.y))
    g=f.groupby(['airport','day'],as_index=False).agg(n=('mvt_id','size'),truth=('contribution','sum'))
    g=g.merge(source,on=['airport','day'],how='left',validate='one_to_one')
    g['dn']=g.n_public-g.n;g['eps']=g.total-g.truth
    return g

def robust_scale(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return float(1.4826*np.median(np.abs(x-np.median(x)))) if len(x) else 0.

def fit_noise(rows, source):
    g=aggregate_truth(rows,source);params={}
    for airport,q in g.groupby('airport'):
        q=q.dropna(subset=['eps','dn']); same=q[q.dn==0]
        base=same if len(same)>=10 else q
        bias=float(base.eps.median()); sigma=max(robust_scale(base.eps),DESIGN['noise_floor'])
        bad=q[q.dn!=0]
        kappa=max(DESIGN['mismatch_scale_floor'],float(np.median(q.total/q.n)))
        if len(bad)>=10:
            kappa=max(kappa,float(np.sqrt(np.median(np.maximum((bad.eps-bias)**2-sigma**2,0)/bad.dn**2))))
        held=rows[(rows.airport==airport)&(rows.gap>3600)&(rows.y<rows.gap-600)&rows.y.between(30,10800)]
        tau=float(held.y.mean()) if len(held)>50 else 1200.
        params[airport]=dict(bias=bias,sigma=sigma,kappa=kappa,tau=tau,days=len(q),same_count_days=len(same),
            held_rows=len(held),eps_abs_median=float(q.eps.abs().median()))
    return params

def moments(gap, mean, sigma):
    """E[C], Var(C), Cov(Y,C), C=max(0,gap-Y), Y Gaussian."""
    z=(gap-mean)/sigma
    phi=np.exp(-.5*z*z)/np.sqrt(2*np.pi); prob=ndtr(z)
    ec=(gap-mean)*prob+sigma*phi
    ec2=((gap-mean)**2+sigma*sigma)*prob+(gap-mean)*sigma*phi
    return ec,np.maximum(ec2-ec*ec,0),-sigma*sigma*prob,prob

def apply(rows, pred, source, params):
    pred=np.asarray(pred,float); out=pred.copy(); logs=[]
    indexed=source.set_index(['airport','day'])
    for (airport,day),ix in rows.groupby(['airport','day']).indices.items():
        if (airport,day) not in indexed.index: continue
        src=indexed.loc[(airport,day)]
        if not np.isfinite(src.total) or not np.isfinite(src.n_public): continue
        par=params[airport];d=rows.iloc[ix].gap.to_numpy(float);p=pred[ix]
        sig=np.full(len(ix),DESIGN['ordinary_sigma'])
        ec,vc,cov,prob=moments(d,p,sig)
        # Rome's long rows have two recording modes; infer prior mixture share
        # from the incoming mean, with its held location fitted on training only.
        mix=(airport==DESIGN['copy_airport'])&(d>DESIGN['copy_threshold'])&(d<DESIGN['copy_max_gap'])&(p<d)&(p>par['tau'])
        if np.any(mix):
            pi=np.clip((p[mix]-par['tau'])/(d[mix]-par['tau']),0,.999)
            tau=(p[mix]-pi*d[mix])/(1-pi)
            eh,vh,ch,_=moments(d[mix],tau,sig[mix])
            ec[mix]=(1-pi)*eh
            vc[mix]=(1-pi)*(vh+eh*eh)-ec[mix]**2
            cov[mix]=(1-pi)*(ch+tau*eh)-p[mix]*ec[mix]
        dn=float(src.n_public-len(ix))
        measurement_var=par['sigma']**2+(par['kappa']*dn)**2
        # A fixed common day residual accounts for correlated ordinary errors.
        loading=prob.copy();loading[mix]=0
        common=DESIGN['common_day_sigma']**2
        totalvar=float(vc.sum()+common*loading.sum()**2+measurement_var)
        residual=float(src.total-par['bias']-ec.sum())
        correction=(cov-common*loading*loading.sum())/totalvar*residual
        out[ix]=np.maximum(0,p+correction)
        logs.append(dict(airport=airport,day=day,n=len(ix),dn=dn,public_total=src.total,
            prior_total=float(ec.sum()),residual=residual,variance=totalvar,
            max_abs_change=float(np.max(np.abs(correction))),mean_change=float(correction.mean()),mixture_rows=int(mix.sum())))
    assert np.isfinite(out).all()
    return out,pd.DataFrame(logs)

def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/'data').glob('training_2025-*.parquet'))
    rows=load_rows(paths);source=public(2025);g=aggregate_truth(rows,source)
    g.to_parquet(OUT/'daily_truth_audit.parquet',index=False)
    stats=[]
    for a,q in g.groupby('airport'):
        same=q[q.dn==0]; valid=q.eps.notna()
        stats.append(dict(airport=a,days=len(q),available=int(valid.sum()),same_count_days=len(same),
            eps_abs_median=float(q.eps.abs().median()),same_eps_abs_median=float(same.eps.abs().median()),
            same_eps_abs_p99=float(same.eps.abs().quantile(.99)),max_count_difference=float(q.dn.abs().max())))
    rank=load_rows([ROOT/'data/ranking.parquet'],False)
    coverage=rank.groupby(['airport','day'],as_index=False).size().merge(public(2026),on=['airport','day'],how='left')
    assert coverage.total.notna().all(),coverage.loc[coverage.total.isna()]
    record=dict(design=DESIGN,source=artifact(__file__),raw_inputs=[artifact(p) for p in paths],
        external_manifest=artifact(SOURCE/'manifest.json'),statistics=stats,ranking_days=len(coverage),ranking_covered=len(coverage),
        inspiration='https://github.com/il-miscusi/prc-2026-gracious-tornado',
        rights='EUROCONTROL attributed; inputs retained unmodified locally; no data redistribution')
    write_json(OUT/'identity.json',record);rows.to_parquet(OUT/'rows2025.parquet',index=False)
    rank.to_parquet(OUT/'ranking_inputs.parquet',index=False)
    print(json.dumps(stats,indent=2),flush=True)

def screen():
    rows=pd.read_parquet(OUT/'rows2025.parquet'); cv=pd.read_parquet(PARENT/'cv.parquet');source=public(2025)
    assert cv.mvt_id.is_unique
    aligned=rows.set_index('mvt_id').loc[cv.mvt_id].reset_index()
    np.testing.assert_array_equal(aligned.y,cv.y);np.testing.assert_array_equal(aligned.day,cv.day)
    prediction=cv.pred.to_numpy(float).copy();logs=[];parameters={}
    for month,allowed in FOLDS.items():
        tr=rows[rows.month.isin(allowed)];ix=np.flatnonzero(cv.month.eq(month));parameters[month]=fit_noise(tr,source)
        assert month not in set(tr.month)
        prediction[ix],log=apply(aligned.iloc[ix].reset_index(drop=True),prediction[ix],source,parameters[month])
        logs.append(log)
    frame=cv[['mvt_id','month','day','airport','y']].copy();frame['parent']=cv.pred;frame['pred']=prediction
    frame.to_parquet(OUT/'screen.parquet',index=False);pd.concat(logs).to_parquet(OUT/'screen_days.parquet',index=False)
    results={}
    for month,q in frame.groupby('month'):
        results[month]=dict(parent=float(np.sqrt(np.mean((q.parent-q.y)**2))),candidate=float(np.sqrt(np.mean((q.pred-q.y)**2))))
    scores=dict(parent=float(np.mean([r['parent'] for r in results.values()])),candidate=float(np.mean([r['candidate'] for r in results.values()])))
    write_json(OUT/'screen.json',dict(design=DESIGN,months=results,scores=scores,parameters=parameters,
        source=artifact(__file__),parent=artifact(PARENT/'cv.parquet')))
    print(json.dumps(dict(months=results,scores=scores),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','screen']);a=p.parse_args()
    globals()[a.stage]()
