"""Discrete long-flight posterior with a robust public daily-delay likelihood.

Original implementation inspired by the credited public approach in
daily_delay_reconcile.py. This arm replaces the failed Gaussian regression
with exact discrete messages for long flights and a linear ordinary-flight
aggregate. Training quantiles and source noise use only allowed fold labels.
Fixed settings; no gate chosen using held-out outcomes.
"""
import argparse
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import daily_delay_reconcile as base
from records import write_json

OUT=base.ROOT/'VALIDATION/runs/v3_20261011_daily_delay_posterior'
DESIGN=dict(base.DESIGN,method='60-second discrete messages; t3 measurement likelihood; Gaussian ordinary sum',
    grid_sec=60.,ordinary_sigma=150.,common_day_sigma=15.,held_quantiles=15,
    ordinary_grid_points=41,ordinary_grid_sd=6.,copy_broad_share=.05,
    long_prior='held quantiles centered at incoming mean; Rome adds copy and broad components',
    max_correction=None)

def fit(rows,source):
    params=base.fit_noise(rows,source)
    for a,p in params.items():
        m=rows.airport.eq(a)&(rows.gap>3600)&(rows.y<rows.gap-600)&rows.y.between(30,10800)
        y=rows.loc[m,'y'].to_numpy(float)
        q=np.quantile(y,np.linspace(.025,.975,DESIGN['held_quantiles']))
        p['spread']=(q-q.mean()).tolist()
    return params

def tilt(values,weights,target):
    # Information projection preserves the exact incoming mean wherever feasible.
    if target<=values.min():return np.eye(1,len(values),int(values.argmin())).ravel()
    if target>=values.max():return np.eye(1,len(values),int(values.argmax())).ravel()
    z=(values-target)/max(np.ptp(values),1);lw=np.log(np.maximum(weights,1e-300));lo,hi=-128.,128.
    for _ in range(55):
        mid=(lo+hi)/2;w=np.exp(lw+mid*z-logsumexp(lw+mid*z))
        if w@z>0:hi=mid
        else:lo=mid
    return w

def support(d,p,nm,nm_gap,airport,prm):
    spread=np.asarray(prm['spread']);pi=0.
    mixture=airport=='LIRF' and 3600<d<14*3600 and prm['tau']<p<d
    if mixture:pi=float(np.clip((p-prm['tau'])/(d-prm['tau']),0,.999))
    loc=(p-pi*d)/(1-pi)
    held=np.maximum(0,loc+spread)
    values=held;weights=np.full(len(held),1/len(held))
    if mixture:
        broad=np.linspace(30,max(d,30),9)
        values=np.r_[held,d,broad]
        weights=np.r_[np.full(len(held),(1-pi)*.95/len(held)),pi*.95,np.full(len(broad),.05/len(broad))]
    # Preserve prior means, and avoid forcing an unmatched join bound onto a
    # prior outside it. The original-file join tolerance is 3606 seconds.
    if nm and np.isfinite(nm_gap):
        lo=max(0,nm_gap-3606);hi=nm_gap+3606
        if lo<=p<=hi:
            keep=(values>=lo)&(values<=hi)
            if keep.any():values=values[keep];weights=weights[keep]
    values=np.r_[values,p];weights=np.r_[weights,1e-8]
    weights=tilt(values,weights,p)
    np.testing.assert_allclose(values@weights,p,atol=1e-5,rtol=1e-8)
    c=np.rint(np.maximum(0,d-values)/DESIGN['grid_sec']).astype(int)
    return values,weights,c

def condition_day(d,p,nm,nm_gap,airport,prm,total,dn):
    long=d>3600;ordinary=~long; active=ordinary&(d>p)
    # Linearize the ordinary contribution at the mean: unlike the first screen,
    # this does not add a positive-part Jensen offset on every short flight.
    prior_ordinary=float(np.maximum(0,d[ordinary]-p[ordinary]).sum())
    n=int(active.sum());indvar=DESIGN['ordinary_sigma']**2
    common=DESIGN['common_day_sigma']**2
    shortvar=n*indvar+n*n*common
    sd=max(np.sqrt(shortvar),1.)
    offsets=np.linspace(-DESIGN['ordinary_grid_sd'],DESIGN['ordinary_grid_sd'],DESIGN['ordinary_grid_points'])*sd
    logprior=-.5*(offsets/sd)**2
    rows=np.flatnonzero(long)
    supports=[support(d[i],p[i],nm[i],nm_gap[i],airport,prm) for i in rows]
    size=1+sum(int(x[2].max()) for x in supports)
    totals=np.arange(size)*DESIGN['grid_sec']
    measurement_sd=np.sqrt(prm['sigma']**2+(prm['kappa']*dn)**2)
    z=(total-prm['bias']-prior_ordinary-totals[:,None]-offsets[None,:])/measurement_sd
    loglike=logprior[None,:]-2*np.log1p(z*z/3.)
    scale=loglike.max();jointlike=np.exp(loglike-scale)
    likelihood=jointlike.sum(axis=1)
    prior=np.zeros(size);prior[0]=1.;forward=[]
    for ys,weights,contributions in supports:
        forward.append(prior);nxt=np.zeros(size)
        for w,c in zip(weights,contributions):nxt[c:]+=w*prior[:size-c]
        prior=nxt/nxt.sum()
    post_offset=(prior[:,None]*jointlike).sum(axis=0)
    assert post_offset.sum()>0
    expected_offset=float(post_offset@offsets/post_offset.sum())
    result=p.copy()
    if n:result[active]-=(indvar+n*common)/max(shortvar,1)*expected_offset
    # Common-day information also reaches early ordinary flights at a modest,
    # fixed 0.25 loading. The narrow independent term only reaches active rows.
    if n:result[ordinary&~active]-=.25*n*common/max(shortvar,1)*expected_offset
    backward=likelihood/likelihood.max()
    for j in range(len(rows)-1,-1,-1):
        ys,weights,contributions=supports[j]
        factors=np.array([forward[j][:size-c]@backward[c:] for c in contributions])
        posterior=weights*factors
        assert posterior.sum()>0
        result[rows[j]]=posterior@ys/posterior.sum()
        nxt=np.zeros(size)
        for w,c in zip(weights,contributions):nxt[:size-c]+=w*backward[c:]
        backward=nxt/max(nxt.max(),1e-300)
    return np.maximum(result,0),dict(offset=expected_offset,grid_size=size,long_rows=len(rows),active=n,
        prior_ordinary=prior_ordinary,measurement_sd=measurement_sd)

def apply(rows,pred,source,params):
    pred=np.asarray(pred,float);out=pred.copy();logs=[];indexed=source.set_index(['airport','day']);start=time.time()
    for k,((airport,day),ix) in enumerate(rows.groupby(['airport','day']).indices.items()):
        if (airport,day) not in indexed.index:continue
        src=indexed.loc[(airport,day)]
        if not np.isfinite(src.total) or not np.isfinite(src.n_public):continue
        f=rows.iloc[ix];dn=float(src.n_public-len(ix))
        out[ix],info=condition_day(f.gap.to_numpy(float),pred[ix],f.has_id.to_numpy(bool),f.nm_gap.to_numpy(float),airport,params[airport],float(src.total),dn)
        logs.append(dict(airport=airport,day=day,n=len(ix),dn=dn,**info,max_change=float(np.max(np.abs(out[ix]-pred[ix])))))
        if k%100==0:print(f'{k} days / {time.time()-start:.1f}s',flush=True)
    return out,pd.DataFrame(logs)

def screen():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'screen.json').exists(),'Preserve completed screen'
    write_json(OUT/'design.json',dict(design=DESIGN,source=base.artifact(__file__),helpers=base.artifact(base.__file__)))
    rows=pd.read_parquet(base.OUT/'rows2025.parquet');cv=pd.read_parquet(base.PARENT/'cv.parquet');source=base.public(2025)
    aligned=rows.set_index('mvt_id').loc[cv.mvt_id].reset_index()
    np.testing.assert_array_equal(aligned.y,cv.y);out=cv.pred.to_numpy(float).copy();parameters={};logs=[]
    for month,allowed in base.FOLDS.items():
        ix=np.flatnonzero(cv.month.eq(month));parameters[month]=fit(rows[rows.month.isin(allowed)],source)
        out[ix],log=apply(aligned.iloc[ix].reset_index(drop=True),out[ix],source,parameters[month]);logs.append(log)
    f=cv[['mvt_id','month','day','airport','y']].copy();f['parent']=cv.pred;f['pred']=out
    f.to_parquet(OUT/'screen.parquet',index=False);pd.concat(logs).to_parquet(OUT/'days.parquet',index=False)
    results={m:dict(parent=float(np.sqrt(np.mean((q.parent-q.y)**2))),candidate=float(np.sqrt(np.mean((q.pred-q.y)**2)))) for m,q in f.groupby('month')}
    scores={name:float(np.mean([v[name] for v in results.values()])) for name in ['parent','candidate']}
    write_json(OUT/'screen.json',dict(design=DESIGN,months=results,scores=scores,parameters=parameters,
        source=base.artifact(__file__),helpers=base.artifact(base.__file__),parent=base.artifact(base.PARENT/'cv.parquet')))
    print(results,scores,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen']);a=p.parse_args();globals()[a.stage]()
