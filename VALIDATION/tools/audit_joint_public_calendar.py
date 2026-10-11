"""Separate all-row score and scalar uncertain-calendar message review."""
from datetime import date,datetime,timezone
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
import joint_public_calendar_cv as run


def messages():
    model=run.model;rows=pd.read_parquet(run.OUT/'inputs.parquet')
    saved=pd.read_parquet(run.OUT/'predictions.parquet').set_index('mvt_id')
    parameters=run.read(run.OUT/'parameters.json')
    taxi=pd.read_parquet(run.OUT/'taxi_source.parquet').set_index(['local_day','runway'])
    source=pd.read_parquet(run.OUT/'delay_source.parquet');summaries=[]
    for month,r in rows.groupby('month'):
        r=r.reset_index(drop=True);p=saved.loc[r.mvt_id].reset_index(drop=True);prm=parameters[month]
        keep=[day.startswith(month) and 1<int(day[-2:])<pd.Timestamp(day).days_in_month for day in source.day]
        ds=source.loc[keep].sort_values('day').reset_index(drop=True)
        days=np.array([(date.fromisoformat(s)-date(1970,1,1)).days for s in ds.day])
        lookup={int(day):j for j,day in enumerate(days)}
        v,c,coverage,derivative,supports=model.moments(r,days,prm['support'])
        dep=~r.fltrule.eq('V').to_numpy();pos=np.flatnonzero(dep)
        counts=np.array([math.fsum(coverage[j,k] for j in pos) for k in range(len(days))])
        checked_states=0;expected=np.zeros(len(r),bool)
        for i,(values,weights,w,assigned) in supports.items():
            independent=np.array([(datetime.fromtimestamp(float(r.takeoff_epoch.iloc[i]-val),timezone.utc).date()-date(1970,1,1)).days for val in values])
            np.testing.assert_array_equal(assigned,independent);checked_states+=len(values)
            possible=np.unique(independent)
            expected[i]=bool(p.touched.iloc[i] and all(int(d) in lookup and counts[lookup[int(d)]]>=20
                and np.isfinite(ds.total.iloc[lookup[int(d)]]+ds.n_public.iloc[lookup[int(d)]]) for d in possible))
        np.testing.assert_array_equal(expected,p.calendar_touched)
        active=np.flatnonzero(expected)
        chosen=np.unique(np.r_[active[np.linspace(0,len(active)-1,min(24,len(active)),dtype=int)],
            np.flatnonzero(expected&p.calendar_days.gt(1))[:24]])
        worst=0.
        for i in chosen:
            a=r.iloc[i];values,weights,w,assigned=supports[i]
            jd=np.array([lookup[int(day)] for day in np.unique(assigned)]);dims=np.r_[0,1,jd+2]
            tm=(r.eligible&r.quality&r.local_day.eq(a.local_day)&r.runway.eq(a.runway)).to_numpy()
            assert tm[i] and dep[i] and len(jd)==p.calendar_days.iloc[i]
            masks=[tm,tm,*[dep]*len(jd)];mean=np.zeros(len(dims));cov=np.zeros((len(dims),len(dims)))
            for k,dim in enumerate(dims):
                mean[k]=math.fsum(v[j,dim] for j in np.flatnonzero(masks[k]) if j!=i)
                for l,other in enumerate(dims):
                    cov[k,l]=math.fsum(c[j,dim,other] for j in np.flatnonzero(masks[k]&masks[l]) if j!=i)
            n=int(tm.sum());pub=taxi.loc[(a.local_day,a.runway)];coef=np.asarray(prm['taxi']['coefficients'])
            bias=coef[0]+n/100*coef[1]
            observed=np.r_[pub.source_n+bias[0],pub.source_n*pub.source_mean+bias[1],ds.total.to_numpy()[jd]-prm['delay']['bias']]
            cov[:2,:2]+=n*np.asarray(prm['taxi']['covariance_per_row'])+np.diag([1.,3600.])
            for k,j in enumerate(jd):
                dn=ds.n_public.iloc[j]-counts[j]
                cov[k+2,k+2]+=3*(prm['delay']['sigma']**2+(prm['delay']['kappa']*dn)**2)
            loading=np.r_[0.,math.fsum(v[j,0] for j in np.flatnonzero(tm)),
                [math.fsum(derivative[j,k] for j in pos) for k in jd]]
            cov+=225*np.outer(loading,loading);actual=[]
            for subset in [np.arange(2,len(dims)),np.arange(len(dims))]:
                block=.6*cov[np.ix_(subset,subset)];scales=np.sqrt(np.diag(block))
                residual=observed[subset]-mean[subset]-w[:,dims[subset]]
                likelihood=multivariate_t.logpdf(residual/scales,shape=block/np.outer(scales,scales),df=5)
                lw=np.log(np.maximum(weights,1e-300))+likelihood;post=np.exp(lw-logsumexp(lw))
                actual.append(float(values@post))
            reference=[p.dep_only.iloc[i],p.joint.iloc[i]]
            np.testing.assert_allclose(actual,reference,atol=2e-6,rtol=0)
            worst=max(worst,float(np.max(abs(np.asarray(actual)-reference))))
        summaries.append(dict(month=month,rows=len(r),all_support_states_checked=checked_states,
            all_scope_gates_checked=True,sampled_messages=len(chosen),max_message_error=worst))
        print(summaries[-1],flush=True)
    return summaries


def audit():
    import daily_main_refinements_cv as selected
    identity=run.guard();out=run.OUT/'audit.json';assert not out.exists()
    result=run.read(run.OUT/'result.json');run.verify(result['scored'])
    for a in identity['sources']:run.verify(a)
    receipt=run.read(run.OUT/'inference.json');assert receipt['identity']==run.art(run.OUT/'identity.json')
    for a in receipt['artifacts']:run.verify(a)
    with duckdb.connect(config={'threads':1}) as con:
        paths={'f':run.OUT/'scored.parquet','p':run.OUT/'predictions.parquet','i':run.OUT/'inputs.parquet',
            'old':run.parent.OUT/'scored.parquet','op':run.parent.OUT/'predictions.parquet','raw':selected.daily.base.OUT/'rows2025.parquet'}
        for name,path in paths.items():con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct f.mvt_id),sum(case when
            f.y is distinct from raw.y or f.day is distinct from raw.day or f.month is distinct from raw.month
            or f.control is distinct from old.candidate or f.v77 is distinct from old.control
            or f.increment!=coalesce(p.increment,0) or f.old_increment!=coalesce(op.increment,0)
            or f.calendar_touched!=coalesce(p.calendar_touched,false)
            or ((not f.calendar_touched or raw.airport!='LIRF') and f.candidate!=f.control) then 1 else 0 end)
            from f join old using(mvt_id) join raw using(mvt_id) left join p using(mvt_id) left join op using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        bad=con.execute('''select count(*) from p join i using(mvt_id) join op using(mvt_id)
            where p.prior!=i.prior or p.touched!=op.touched or p.increment!=p.joint-p.dep_only
            or abs(p.old_increment-op.increment)>1e-7 or (p.calendar_touched and not p.touched)
            or (not p.calendar_touched and p.increment!=p.old_increment)''').fetchone()[0];assert bad==0
        for name in ['control','candidate']:
            scores=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from f group by month').fetchall()
            for month,value in scores:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in scores]),result['cv'][name],atol=1e-8,rtol=0)
    f=pd.read_parquet(run.OUT/'scored.parquet');ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    for increment,column in [('old_increment','control'),('increment','candidate')]:
        actual=selected.daily.constrain(raw,ids,f.month,f.v77.to_numpy()+.5*f[increment].to_numpy(),matches)
        np.testing.assert_array_equal(actual,f[column])
    checks=messages()
    run.write(out,dict(status='pass',identity=run.art(run.OUT/'identity.json'),result=run.art(run.OUT/'result.json'),source=run.art(__file__),
        rows=n,all_raw_truth_v78_parent_increment_fallback_and_sql_scores=True,final_constraints_replayed=True,
        scalar_message_checks=checks,independent_final_audit_pending=False,
        reviewer='root; retained subagents unavailable on account quota'))


if __name__=='__main__':
    with threadpool_limits(1):audit()
