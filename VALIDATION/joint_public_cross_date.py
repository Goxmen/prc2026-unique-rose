"""Additional cross-date taxi observations after the selected same-date message.

Only shared flights couple the added dates to the old delay coordinate. Each
new date has its own reporting-error loading; old message covariance is exact.
Sequential capped-score Schur updates are a Gaussian moment approximation.
"""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

import joint_public_other_runway as same

base=same.base
INPUTS=same.pilot.INPUTS
SETTINGS=dict(parent=same.SETTINGS,
    additional='all eligible other-local-date taxi cells intersecting assigned-day delay cohort; complete taxi memberships',
    covariance='shared-flight moments only across dates; independent15-second reporting loading for each added local date; exact old same-date block',
    update='same-date robust Schur update followed by separate cross-date robust Schur update; unchanged t5 target likelihood',
    target_exclusion='target is absent from both extra-source blocks; remove its full original3D contribution once',
    no_date_reassignment=True,no_parameter_search=True,no_covariance_repair=True,
    caveat='Independent cross-date reporting errors are an uncalibrated modeling assumption; not exact Student conditioning')


class Messages:
    def __init__(self,inputs,taxi_source,delay_source,tp,dp,sp):
        rows=inputs[INPUTS].copy().reset_index(drop=True)
        assert rows.mvt_id.is_unique and rows.month.nunique()==1
        self.rows=rows;self.gap=rows.gap.to_numpy(float)
        self.v,self.c,self.prob=base.normal_moments(rows.prior.to_numpy(float),self.gap)
        self.supports={};self.dp=dp
        for i in np.flatnonzero(self.gap>3600):
            y,w,_=base.prior.support(self.gap[i],float(rows.prior.iloc[i]),False,np.nan,'LIRF',sp)
            values=base.contributions(y,self.gap[i]);mean=w@values;center=values-mean
            self.v[i]=mean;self.c[i]=(center.T*w)@center
            self.supports[i]=(y,w,values)
        taxi_ok=(rows.eligible&rows.quality).to_numpy(bool)
        self.delay_ok=(~rows.fltrule.eq('V')&rows.delay_scope).to_numpy(bool)
        pos=np.flatnonzero(taxi_ok)
        groups={key:pos[ix] for key,ix in rows.loc[taxi_ok].groupby(['local_day','runway']).indices.items()}
        source=taxi_source.set_index(['local_day','runway']);self.delay=delay_source.set_index('day')
        assert source.index.is_unique and self.delay.index.is_unique
        self.taxi={};self.cache={}
        for key,ix in groups.items():
            if key not in source.index:continue
            row=source.loc[key];mean=self.v[ix,:2].sum(0);n=mean[0]
            if not(len(ix)>=20 and row.source_n>=20 and n>=20 and .5<=row.source_n/n<=1.2):continue
            bias=np.array([1.,len(ix)/100.])@np.asarray(tp['coefficients'])
            self.taxi[key]=dict(ix=ix,mean=mean,n=n,cov=self.c[ix,:2,:2].sum(0),
                observed=np.array([row.source_n,row.source_n*row.source_mean])+bias,
                noise=len(ix)*np.asarray(tp['covariance_per_row'])+np.diag([1.,3600.]))

    def cohort(self,own,dkey):
        key=(own,dkey)
        if key in self.cache:return self.cache[key]
        di=np.flatnonzero(self.delay_ok&self.rows.assigned_day.eq(dkey).to_numpy())
        src=self.delay.loc[dkey];assert len(di)>=20 and np.isfinite(src.total+src.n_public)
        eligible=sorted(k for k,t in self.taxi.items() if k!=own and np.intersect1d(t['ix'],di).size)
        old=[k for k in eligible if k[0]==own[0]]
        new=[k for k in eligible if k[0]!=own[0]]
        old_size=3+2*len(old);size=old_size+2*len(new)
        blocks=[(own,slice(0,2)),*[(k,slice(3+2*j,5+2*j)) for j,k in enumerate(old+new)]]
        mean=np.zeros(size);obs=np.zeros(size);peer=np.zeros((size,size));noise=np.zeros_like(peer)
        mean[2]=self.v[di,2].sum();obs[2]=src.total-self.dp['bias']
        peer[2,2]=self.c[di,2,2].sum()
        noise[2,2]=3*(self.dp['sigma']**2+(self.dp['kappa']*float(src.n_public-len(di)))**2)
        loading=np.zeros(size);loading[2]=-float(np.sum(self.prob[di]*(self.gap[di]<=3600)))
        other_loadings={};extra_members=set()
        for group,sl in blocks:
            t=self.taxi[group];mean[sl]=t['mean'];obs[sl]=t['observed'];peer[sl,sl]=t['cov'];noise[sl,sl]=t['noise']
            shared=np.intersect1d(t['ix'],di)
            peer[sl,2]=self.c[shared,:2,2].sum(0);peer[2,sl]=peer[sl,2]
            if sl.start<old_size:loading[sl.stop-1]=t['n']
            else:
                other_loadings.setdefault(group[0],np.zeros(size))[sl.stop-1]=t['n']
            if group!=own:extra_members.update(t['ix'].tolist())
        common=225*np.outer(loading,loading)
        for value in other_loadings.values():common+=225*np.outer(value,value)
        full=peer+(noise+common)
        sd=np.sqrt(np.diag(full));np.linalg.cholesky(full/np.outer(sd,sd))
        # Distinct taxi groups share no flights, and added-date noise is independent.
        np.testing.assert_array_equal(full[3:old_size,old_size:],0.)
        np.testing.assert_array_equal(full[:2,old_size:],0.)
        self.cache[key]=dict(mean=mean,observed=obs,cov=full,old_size=old_size,
            own=own,assigned_day=dkey,old_cells=old,new_cells=new,extra_members=extra_members,
            peer=peer,noise=noise,common=common,di=di,blocks=blocks)
        return self.cache[key]

    def condition(self,i):
        row=self.rows.iloc[i];own=(row.local_day,row.runway)
        group=self.cohort(own,row.assigned_day);split=group['old_size']
        assert i not in group['extra_members'] and i in self.taxi[own]['ix'] and i in group['di']
        rawmean=group['mean'][:3]-self.v[i];rawcov=group['cov'][:3,:3]-self.c[i]
        cross0=group['cov'][:3,3:split];c0=group['cov'][3:split,3:split]
        r0=group['observed'][3:split]-group['mean'][3:split]
        oldmean,oldcov,d0=same.robust_condition(rawmean,rawcov,cross0,c0,r0)
        cross1=group['cov'][:3,split:];c1=group['cov'][split:,split:]
        r1=group['observed'][split:]-group['mean'][split:]
        newmean,newcov,d1=same.robust_condition(oldmean,oldcov,cross1,c1,r1)
        # Independent simultaneous inflated-noise block solve checks both stages.
        cross=np.concatenate([cross0,cross1],axis=1)
        extra=block_diag(c0/d0['weight'],c1/d1['weight'])
        residual=np.r_[r0,r1]
        if len(residual):
            direct_mean=rawmean+cross@np.linalg.solve(extra,residual)
            direct_cov=rawcov-cross@np.linalg.solve(extra,cross.T)
        else:direct_mean,direct_cov=rawmean,rawcov
        np.testing.assert_allclose(newmean,direct_mean,atol=1e-7,rtol=1e-13)
        np.testing.assert_allclose(newcov,direct_cov,atol=1e-6,rtol=1e-13)
        return dict(group=group,rawmean=rawmean,rawcov=rawcov,oldmean=oldmean,oldcov=oldcov,
            newmean=newmean,newcov=newcov,old_diagnostic=d0,new_diagnostic=d1,
            direct_mean_max_error=float(abs(newmean-direct_mean).max()),
            direct_cov_max_error=float(abs(newcov-direct_cov).max()))


def checks():
    n=123
    rows=pd.DataFrame(dict(mvt_id=np.arange(n),month='2025-01',
        gap=np.r_[np.full(120,1500.),9000.,10000.,11000.],
        prior=np.r_[np.full(120,1000.),3000.,3500.,4000.],eligible=True,quality=True,
        runway=['A']*40+['B']*40+['C']*40+['A','B','C'],
        local_day=['2025-01-02']*80+['2025-01-03']*40+['2025-01-02']*2+['2025-01-03'],
        assigned_day='2025-01-02',fltrule='I',delay_scope=True))
    taxi=pd.DataFrame(dict(local_day=['2025-01-02']*2+['2025-01-03'],runway=['A','B','C'],
        source_n=[41.,41.,41.],source_mean=[1200.,1300.,1100.]))
    delay=pd.DataFrame(dict(day=['2025-01-02'],total=[80000.],n_public=[123.]))
    tp=dict(coefficients=[[0.,0.],[0.,0.]],covariance_per_row=[[.1,0.],[0.,10000.]])
    dp=dict(bias=0.,sigma=1000.,kappa=100.);sp=dict(tau=1000.,spread=[-100.,0.,100.])
    m=Messages(rows,taxi,delay,tp,dp,sp)
    retained,_=same.apply(rows,taxi,delay,tp,dp,sp)
    for i in [120,121,122]:
        value=m.condition(i);g=value['group'];y,w,v=m.supports[i]
        original=base.posterior_mean(y,w,v,g['observed'][:3],value['rawmean'],value['rawcov'])
        selected=base.posterior_mean(y,w,v,g['observed'][:3],value['oldmean'],value['oldcov'])
        np.testing.assert_allclose(original,retained.old_joint.iloc[i],atol=1e-8,rtol=0)
        np.testing.assert_allclose(selected,retained.new_joint.iloc[i],atol=1e-8,rtol=0)
        assert len(g['new_cells'])>0
    fallback=Messages(rows,taxi.iloc[:2],delay,tp,dp,sp).condition(120)
    np.testing.assert_array_equal(fallback['oldmean'],fallback['newmean'])
    np.testing.assert_array_equal(fallback['oldcov'],fallback['newcov'])
    poisoned=Messages(rows.assign(y=1e15,true_day='1900-01-01'),taxi,delay,tp,dp,sp).condition(120)
    np.testing.assert_array_equal(poisoned['newmean'],m.condition(120)['newmean'])
    return dict(original_and_selected_messages_exact=True,separate_date_reporting_covariance=True,
        no_extra_exact=True,target_excluded_once=True,simultaneous_inflated_block_solve=True,
        positive_definite_without_repair=True,label_poison_invariant=True)
