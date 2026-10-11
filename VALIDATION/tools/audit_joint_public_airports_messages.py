"""Check native-airport support, cohort gates and scalar source messages."""
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import joint_public_airports_cv as run


def audit():
    identity=run.guard();out=run.OUT/'complete_audit.json';assert not out.exists()
    for name in ['input_audit.json','audit.json']:
        record=run.read(run.OUT/name);assert record['status']=='pass'
        assert record['identity']==run.art(run.OUT/'identity.json');run.verify(record['source'])
    receipt=run.read(run.OUT/'inference.json')
    for a in receipt['artifacts']:run.verify(a)
    rows=pd.read_parquet(run.OUT/'inputs.parquet')
    saved=pd.read_parquet(run.OUT/'predictions.parquet').set_index('mvt_id')
    parameters=run.read(run.OUT/'parameters.json')
    ts=pd.read_parquet(run.OUT/'taxi_source.parquet').set_index(['airport','local_day','runway'])
    ds=pd.read_parquet(run.OUT/'delay_source.parquet').set_index(['airport','day'])
    model=run.model.base;summaries=[];total_checked=0
    for (month,airport),original in rows.groupby(['month','airport']):
        r=original.reset_index(drop=True);p=saved.loc[r.mvt_id].reset_index(drop=True)
        prm=parameters[month][airport]
        v,c,prob=model.normal_moments(r.prior.to_numpy(),r.gap.to_numpy());supports={}
        for i in np.flatnonzero(r.gap.gt(3600)):
            values,weights,_=model.prior.support(float(r.gap.iloc[i]),float(r.prior.iloc[i]),False,np.nan,airport,prm['support'])
            # Native support consists only of retained quantiles plus the mean.
            expected=np.r_[np.maximum(0,r.prior.iloc[i]+np.asarray(prm['support']['spread'])),r.prior.iloc[i]]
            np.testing.assert_array_equal(values,expected)
            np.testing.assert_allclose(values@weights,r.prior.iloc[i],atol=1e-5,rtol=1e-8)
            valid=(values>=0)&(values<=7200)
            w=np.column_stack([valid,np.where(valid,values,0),np.maximum(r.gap.iloc[i]-values,0)])
            v[i]=weights@w;c[i]=(w-v[i]).T@(weights[:,None]*(w-v[i]));supports[i]=(values,weights,w)
        expected=np.zeros(len(r),bool)
        taxi_groups={};delay_groups={}
        for i,a in enumerate(r.itertuples()):
            if not (a.eligible and a.quality and a.delay_scope and a.fltrule!='V'
                    and 3600<a.gap<50400 and prm['support']['tau']<a.prior<a.gap):continue
            tk=(airport,a.local_day,a.runway);dk=(airport,a.assigned_day)
            if tk not in ts.index or dk not in ds.index:continue
            if tk not in taxi_groups:
                taxi_groups[tk]=(r.eligible&r.quality&r.local_day.eq(a.local_day)&r.runway.eq(a.runway)).to_numpy()
            if dk not in delay_groups:
                delay_groups[dk]=(r.delay_scope&~r.fltrule.eq('V')&r.assigned_day.eq(a.assigned_day)).to_numpy()
            tm,dm=taxi_groups[tk],delay_groups[dk];pub,delay=ts.loc[tk],ds.loc[dk]
            count=math.fsum(v[j,0] for j in np.flatnonzero(tm))
            expected[i]=(tm.sum()>=20 and pub.source_n>=20 and count>=20
                         and .5<=pub.source_n/count<=1.2 and dm.sum()>=20
                         and np.isfinite(delay.total+delay.n_public))
        np.testing.assert_array_equal(expected,p.touched)
        active=np.flatnonzero(expected)
        chosen=np.unique(np.r_[active[np.linspace(0,len(active)-1,min(12,len(active)),dtype=int)],
            np.flatnonzero(expected&r.local_day.ne(r.assigned_day))[:4]])
        worst=0.
        for i in chosen:
            a=r.iloc[i];tk=(airport,a.local_day,a.runway);dk=(airport,a.assigned_day)
            tm,dm=taxi_groups[tk],delay_groups[dk];union=np.flatnonzero(tm|dm)
            assert tm[i] and dm[i]
            mean=np.zeros(3);cov=np.zeros((3,3))
            for k in range(3):
                mean[k]=math.fsum(v[j,k] for j in union if j!=i and (tm[j] if k<2 else dm[j]))
                for l in range(3):
                    cov[k,l]=math.fsum(c[j,k,l] for j in union if j!=i
                        and (tm[j] if k<2 else dm[j]) and (tm[j] if l<2 else dm[j]))
            n=int(tm.sum());pub,delay=ts.loc[tk],ds.loc[dk]
            coef=np.asarray(prm['taxi']['coefficients']);bias=coef[0]+n/100*coef[1]
            observed=np.array([pub.source_n+bias[0],pub.source_n*pub.source_mean+bias[1],delay.total-prm['delay']['bias']])
            cov[:2,:2]+=n*np.asarray(prm['taxi']['covariance_per_row'])+np.diag([1.,3600.])
            dn=delay.n_public-int(dm.sum())
            cov[2,2]+=3*(prm['delay']['sigma']**2+(prm['delay']['kappa']*dn)**2)
            count=math.fsum(v[j,0] for j in np.flatnonzero(tm))
            load=math.fsum(prob[j] for j in np.flatnonzero(dm) if r.gap.iloc[j]<=3600)
            loading=np.array([0.,count,-load]);cov+=225*np.outer(loading,loading)
            values,weights,contrib=supports[i];actual=[]
            for dims in [[2],[0,1,2]]:
                delta=observed[dims]-mean[dims]-contrib[:,dims]
                block=.6*cov[np.ix_(dims,dims)];scales=np.sqrt(np.diag(block))
                ll=multivariate_t.logpdf(delta/scales,shape=block/np.outer(scales,scales),df=5)
                lw=np.log(np.maximum(weights,1e-300))+ll;post=np.exp(lw-logsumexp(lw))
                actual.append(float(values@post))
            reference=[p.dep_only.iloc[i],p.joint.iloc[i]]
            np.testing.assert_allclose(actual,reference,atol=2e-6,rtol=0)
            worst=max(worst,float(np.max(abs(np.asarray(actual)-reference))))
        total_checked+=len(r)
        summaries.append(dict(month=month,airport=airport,rows=len(r),active=len(active),sampled=len(chosen),max_error=worst))
        print(summaries[-1],flush=True)
    assert total_checked==317891 and len(summaries)==18
    run.write(out,dict(status='pass',identity=run.art(run.OUT/'identity.json'),source=run.art(__file__),
        input_audit=run.art(run.OUT/'input_audit.json'),score_audit=run.art(run.OUT/'audit.json'),
        inference=run.art(run.OUT/'inference.json'),result=run.art(run.OUT/'result.json'),
        all_native_supports_and_scope_gates_checked=True,scalar_message_checks=summaries,
        message_audit_pending=False,reviewer='root; retained subagents unavailable on account quota'))


if __name__=='__main__':
    with threadpool_limits(1):audit()
