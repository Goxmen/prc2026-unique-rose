"""Scalar cohort/peer selection and SciPy enumeration audit for explicit peers."""
import math
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import joint_public_peer_cv as run


def audit():
    run.guard();assert not (run.OUT/'message_audit.json').exists()
    rows=pd.read_parquet(run.OUT/'inputs.parquet').reset_index(drop=True)
    saved=pd.read_parquet(run.OUT/'predictions.parquet').set_index('mvt_id').loc[rows.mvt_id]
    parameters=run.read(run.OUT/'parameters.json')
    ts=pd.read_parquet(run.OUT/'taxi_source.parquet').set_index(['local_day','runway'])
    ds=pd.read_parquet(run.OUT/'delay_source.parquet').set_index('day')
    with duckdb.connect(config={'threads':1}) as con:
        con.register('i',rows);con.register('p',saved.reset_index())
        n,bad=con.execute('''select count(*),sum(case when i.month!=j.month or j.gap<=3600
            or i.mvt_id=j.mvt_id or not((j.eligible and j.quality and i.local_day=j.local_day and i.runway=j.runway)
                or (j.delay_scope and j.fltrule!='V' and i.assigned_day=j.assigned_day)) then 1 else 0 end)
            from p join i using(mvt_id) join i j on p.peer_id=j.mvt_id where p.peer_id>=0''').fetchone()
        assert n==int(saved.peer_id.ge(0).sum()) and bad==0
    summaries=[];base=run.model.base
    for month,ix in rows.groupby('month').indices.items():
        r=rows.iloc[ix].reset_index(drop=True);p=saved.iloc[ix].reset_index(drop=True);prm=parameters[month]
        means,covs,prob=base.normal_moments(r.prior.to_numpy(),r.gap.to_numpy());supports={}
        for j in np.flatnonzero(r.gap.gt(3600)):
            values,weights,_=base.prior.support(float(r.gap.iloc[j]),float(r.prior.iloc[j]),False,np.nan,'LIRF',prm['support'])
            vectors=np.column_stack([(values>=0)&(values<=7200),np.where((values>=0)&(values<=7200),values,0),np.maximum(r.gap.iloc[j]-values,0)])
            mean=np.array([math.fsum(weights*vectors[:,k]) for k in range(3)])
            cov=np.array([[math.fsum(weights*(vectors[:,k]-mean[k])*(vectors[:,l]-mean[l])) for l in range(3)] for k in range(3)])
            means[j],covs[j]=mean,cov;supports[j]=(values,weights,vectors)
        active=np.flatnonzero(p.peer_id.ge(0))
        chosen=np.unique(np.r_[active[np.linspace(0,len(active)-1,min(64,len(active)),dtype=int)],
            active[r.local_day.iloc[active].ne(r.assigned_day.iloc[active]).to_numpy()][:16]])
        maxerror=0.;maxshare=0.
        for i in chosen:
            a=r.iloc[i]
            tm=(r.eligible&r.quality&r.local_day.eq(a.local_day)&r.runway.eq(a.runway)).to_numpy()
            dm=(r.delay_scope&~r.fltrule.eq('V')&r.assigned_day.eq(a.assigned_day)).to_numpy()
            mask=np.column_stack([tm,tm,dm]).astype(float);union=np.flatnonzero(tm|dm)
            pm=means*mask;pc=covs*mask[:,:,None]*mask[:,None,:]
            mean=np.array([math.fsum(pm[j,k] for j in union if j!=i) for k in range(3)])
            cov=np.array([[math.fsum(pc[j,k,l] for j in union if j!=i) for l in range(3)] for k in range(3)])
            count=int(tm.sum());src=ts.loc[(a.local_day,a.runway)];delay=ds.loc[a.assigned_day]
            coeff=np.asarray(prm['taxi']['coefficients']);bias=coeff[0]+count/100*coeff[1]
            observed=np.array([src.source_n+bias[0],src.source_n*src.source_mean+bias[1],delay.total-prm['delay']['bias']])
            cov[:2,:2]+=count*np.asarray(prm['taxi']['covariance_per_row'])+np.diag([1.,3600.])
            dn=delay.n_public-int(dm.sum());cov[2,2]+=3*(prm['delay']['sigma']**2+(prm['delay']['kappa']*dn)**2)
            loading=np.array([0.,math.fsum(means[j,0] for j in np.flatnonzero(tm)),
                -math.fsum(prob[j] for j in np.flatnonzero(dm) if r.gap.iloc[j]<=3600)])
            cov+=225*np.outer(loading,loading)
            sd=np.sqrt(np.diag(cov));inverse=np.linalg.inv(cov/np.outer(sd,sd))
            peers=[j for j in union if j!=i and r.gap.iloc[j]>3600]
            shares={j:float(np.trace(inverse@(pc[j]/np.outer(sd,sd)))) for j in peers}
            chosen_j=min(peers,key=lambda j:(-round(shares[j],12),int(r.mvt_id.iloc[j])))
            assert int(r.mvt_id.iloc[chosen_j])==p.peer_id.iloc[i]
            maxshare=max(maxshare,abs(shares[chosen_j]-p.peer_variance_share.iloc[i]))
            np.testing.assert_allclose(shares[chosen_j],p.peer_variance_share.iloc[i],atol=1e-10,rtol=0)
            j=chosen_j;cov-=pc[j];mean-=pm[j]
            values,weights,vectors=supports[i];_,pw,pv=supports[j];pv=pv*mask[j]
            posterior_means=[]
            for dims in [[2],[0,1,2]]:
                block=.6*cov[np.ix_(dims,dims)];scale=np.sqrt(np.diag(block));correlation=block/np.outer(scale,scale)
                logmass=[]
                for k in range(len(values)):
                    residual=observed[dims]-mean[dims]-vectors[k,dims]-pv[:,dims]
                    density=multivariate_t.logpdf(residual/scale,shape=correlation,df=5)
                    logmass.append(np.log(max(weights[k],1e-300))+logsumexp(np.log(np.maximum(pw,1e-300))+density))
                post=np.exp(logmass-logsumexp(logmass));posterior_means.append(math.fsum(values*post))
            expected=np.array([p.dep_only.iloc[i],p.joint.iloc[i]])
            np.testing.assert_allclose(posterior_means,expected,rtol=0,atol=2e-7)
            maxerror=max(maxerror,float(np.max(abs(np.array(posterior_means)-expected))))
        summaries.append(dict(month=month,sampled_rows=len(chosen),max_posterior_error=maxerror,max_peer_share_error=maxshare))
    report=dict(status='pass',identity=run.art(run.OUT/'identity.json'),prediction=run.art(run.OUT/'predictions.parquet'),
        source=run.art(__file__),all_peer_memberships_sql=True,peer_rows=n,
        scalar_cohorts_covariances_and_peer_selection=True,independent_scipy_discrete_enumeration=summaries)
    run.write(run.OUT/'message_audit.json',report)
    score=run.read(run.OUT/'audit.json');assert score['status']=='pass' and score['message_audit_pending']
    score.update(message_audit_pending=False,message_audit=run.art(run.OUT/'message_audit.json'))
    run.write(run.OUT/'audit.json',score)
    print(dict(status='pass',peer_rows=n,summaries=summaries),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):audit()
