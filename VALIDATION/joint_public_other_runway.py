"""One fixed robust extra-runway moment update; not exact t conditioning."""
import numpy as np
import pandas as pd

import joint_public_moments as base
import joint_public_other_runway_pilot as pilot

SETTINGS=dict(base.SETTINGS,
    additional='all supported same-local-day other runway taxi cells intersecting the assigned-day delay cohort; full taxi membership',
    conditioning='Gaussian Schur moments with one capped variance-matched t5 score weight; unchanged original 3D t5 target likelihood',
    robust_weight='min(1,(5+k)/(3+q)); q=r_B.T C_BB^-1 r_B, k=extra dimension',
    review_history='unexecuted 5+q denominator corrected to variance-matched 3+q before any robust prediction; no weight scan',
    common_error='one shared15-second outer product over the bounded message, exact original3D marginal',
    comparison='new_joint minus original_joint; original departure-only comparator unchanged',
    transfer='fixed0.5 additional increment on exactv78; separately frozen coefficient-one v79 transport',
    caveat='hybrid approximation; unknown cross-runway source-error covariance and possible prior-error suppression',
    disconnected='exact recovery when the entire C_AB is zero; adding disconnected coordinates beside informative ones may change global q/k/weight',
    no_iteration=True,no_parameter_search=True,no_new_fit=True,no_new_gate=True)


def score_weight(q,dimension):
    assert np.isfinite(q) and q>=0 and dimension>=0
    return min(1.,(5.+dimension)/(3.+q))


def robust_condition(mean,covariance,cross,extra_covariance,innovation):
    if cross.shape[1]==0:
        return mean.copy(),covariance.copy(),dict(weight=1.,q=0.,dimension=0)
    # q is cohort-level and independent of target removal: target has no B value.
    sd=np.sqrt(np.diag(extra_covariance))
    lower=np.linalg.cholesky(extra_covariance/np.outer(sd,sd))
    residual=np.linalg.solve(lower,innovation/sd)
    q=float(residual@residual);weight=score_weight(q,len(innovation))
    if not np.any(cross):
        return mean.copy(),covariance.copy(),dict(weight=weight,q=q,dimension=len(innovation))
    z=np.linalg.solve(lower,cross.T/sd[:,None])
    reduction=z.T@z;shift=z.T@residual
    conditioned_cov=covariance-weight*reduction
    scale=np.sqrt(np.diag(conditioned_cov))
    np.linalg.cholesky(conditioned_cov/np.outer(scale,scale))
    return mean+weight*shift,conditioned_cov,dict(weight=weight,q=q,dimension=len(innovation))


def apply(inputs,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters):
    rows=inputs[pilot.INPUTS].copy().reset_index(drop=True)
    assert rows.mvt_id.is_unique and rows.month.nunique()==1
    old,_=base.apply(rows,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters)
    p=rows.prior.to_numpy(float);gap=rows.gap.to_numpy(float)
    vectors,covs,prob=base.normal_moments(p,gap);supports={}
    for i in np.flatnonzero(gap>3600):
        y,w,_=base.prior.support(gap[i],p[i],False,np.nan,'LIRF',support_parameters)
        value=base.contributions(y,gap[i]);mean=w@value;center=value-mean
        vectors[i]=mean;covs[i]=(center.T*w)@center;supports[i]=(y,w,value)
    taxi_ok=(rows.eligible&rows.quality).to_numpy(bool)
    delay_ok=(~rows.fltrule.eq('V')&rows.delay_scope).to_numpy(bool)
    positions=np.flatnonzero(taxi_ok)
    groups={key:positions[ix] for key,ix in rows.loc[taxi_ok].groupby(['local_day','runway']).indices.items()}
    taxi_source=source_taxi.set_index(['local_day','runway']);delay_source=source_delay.set_index('day')
    assert taxi_source.index.is_unique and delay_source.index.is_unique
    taxi={}
    for key,ix in groups.items():
        if key not in taxi_source.index:continue
        src=taxi_source.loc[key];mean=vectors[ix,:2].sum(0);n=mean[0]
        if not(len(ix)>=20 and src.source_n>=20 and n>=20 and .5<=src.source_n/n<=1.2):continue
        bias=np.array([1.,len(ix)/100.])@np.asarray(taxi_parameters['coefficients'])
        taxi[key]=dict(ix=ix,mean=mean,n=n,cov=covs[ix,:2,:2].sum(0),
            observed=np.array([src.source_n,src.source_n*src.source_mean])+bias,
            noise=len(ix)*np.asarray(taxi_parameters['covariance_per_row'])+np.diag([1.,3600.]))
    result=pd.DataFrame(dict(mvt_id=rows.mvt_id,old_joint=old.joint,new_joint=old.joint,
        delta=0.,touched=old.touched,extra_cells=0,weight=1.,extra_q=0.))
    pairs={};summaries=[]
    for i in np.flatnonzero(old.touched.to_numpy(bool)):
        own=(rows.local_day.iloc[i],rows.runway.iloc[i]);dkey=rows.assigned_day.iloc[i]
        key=(own,dkey)
        if key not in pairs:
            ti=taxi[own]['ix'];di=np.flatnonzero(delay_ok&rows.assigned_day.eq(dkey).to_numpy())
            src=delay_source.loc[dkey];assert len(di)>=20 and np.isfinite(src.total+src.n_public)
            extras=sorted(k for k,t in taxi.items() if k[0]==own[0] and k!=own and np.intersect1d(t['ix'],di).size)
            blocks=[(own,slice(0,2)),*[(k,slice(3+2*j,5+2*j)) for j,k in enumerate(extras)]]
            dim=3+2*len(extras);mean=np.zeros(dim);observed=np.zeros(dim)
            peer=np.zeros((dim,dim));measurement=np.zeros_like(peer);loading=np.zeros(dim)
            mean[2]=vectors[di,2].sum();observed[2]=src.total-delay_parameters['bias']
            peer[2,2]=covs[di,2,2].sum()
            dn=float(src.n_public-len(di))
            measurement[2,2]=3*(delay_parameters['sigma']**2+(delay_parameters['kappa']*dn)**2)
            loading[2]=-float(np.sum(prob[di]*(gap[di]<=3600)))
            extra_members=[]
            for group,sl in blocks:
                t=taxi[group];mean[sl]=t['mean'];observed[sl]=t['observed'];peer[sl,sl]=t['cov']
                measurement[sl,sl]=t['noise'];loading[sl.stop-1]=t['n']
                shared=np.intersect1d(t['ix'],di)
                peer[sl,2]=covs[shared,:2,2].sum(0);peer[2,sl]=peer[sl,2]
                if group!=own:extra_members.extend(t['ix'].tolist())
            full=peer+(measurement+225*np.outer(loading,loading))
            scale=np.sqrt(np.diag(full));np.linalg.cholesky(full/np.outer(scale,scale))
            shared=np.intersect1d(ti,di);control=np.zeros((3,3));noise=np.zeros((3,3))
            control[:2,:2]=taxi[own]['cov'];control[2,2]=covs[di,2,2].sum()
            control[:2,2]=control[2,:2]=covs[shared,:2,2].sum(0)
            noise[:2,:2]=taxi[own]['noise'];noise[2,2]=measurement[2,2]
            original_loading=np.array([0.,taxi[own]['n'],loading[2]])
            noise+=225*np.outer(original_loading,original_loading)
            np.testing.assert_array_equal(full[:3,:3],control+noise)
            pairs[key]=dict(mean=mean,observed=observed,cov=full,extra_members=set(extra_members),
                extra_cells=len(extras),extras=[k[1] for k in extras],targets=0)
        group=pairs[key];group['targets']+=1
        assert i not in group['extra_members']
        # The target's full contribution is in A and exactly zero in B.
        other=group['mean'][:3]-vectors[i];a=group['cov'][:3,:3]-covs[i]
        y,w,value=supports[i]
        control=base.posterior_mean(y,w,value,group['observed'][:3],other,a)
        np.testing.assert_allclose(control,float(old.joint.iloc[i]),atol=1e-8,rtol=0)
        if not group['extra_cells']:continue
        mean,cov,diagnostic=robust_condition(other,a,group['cov'][:3,3:],group['cov'][3:,3:],
            group['observed'][3:]-group['mean'][3:])
        new=base.posterior_mean(y,w,value,group['observed'][:3],mean,cov)
        result.loc[i,['new_joint','delta','extra_cells','weight','extra_q']]=[
            new,new-float(old.joint.iloc[i]),group['extra_cells'],diagnostic['weight'],diagnostic['q']]
        group['diagnostic']=diagnostic
    for (own,dkey),group in pairs.items():
        summaries.append(dict(local_day=own[0],runway=own[1],assigned_day=dkey,
            extra_runways=group['extras'],extra_cells=group['extra_cells'],targets=group['targets'],
            **group.get('diagnostic',dict(weight=1.,q=0.,dimension=0))))
    assert np.isfinite(result[['old_joint','new_joint','delta','weight','extra_q']]).all().all()
    np.testing.assert_array_equal(result.loc[result.extra_cells.eq(0),'new_joint'],result.loc[result.extra_cells.eq(0),'old_joint'])
    assert not result.loc[~result.touched,'delta'].any()
    return result,pd.DataFrame(summaries)


def checks():
    # Derive precision relative to Gaussian covariance from the actual t5 density.
    covariance=np.array([[2.,.3],[.3,3.]]);r=np.array([1.2,-.7]);k=len(r)
    q=float(r@np.linalg.solve(covariance,r));expected=-(5+k)/(3+q)*np.linalg.solve(covariance,r)
    def log_t(value):
        return -.5*(5+k)*np.log1p((value@np.linalg.solve(covariance,value))/3.)
    numerical=[]
    for j in range(k):
        step=np.eye(k)[j]*1e-6;numerical.append((log_t(r+step)-log_t(r-step))/2e-6)
    np.testing.assert_allclose(numerical,expected,atol=1e-9,rtol=0)
    assert score_weight(k+2,k)==1 and score_weight(k+2-1e-8,k)==1
    assert abs(score_weight(k+2+1e-8,k)-1)<1e-8
    a=np.array([[4.,.2,.3],[.2,5.,.4],[.3,.4,6.]]);b=np.array([[.1,.2],[.3,-.1],[.2,.4]])
    m=np.array([1.,2.,3.]);innovation=np.array([2.,-1.]);c=covariance
    for cross,extra,residual in [(np.zeros((3,0)),np.zeros((0,0)),np.zeros(0)),
                                 (np.zeros_like(b),c,np.array([1e12,-1e12]))]:
        value=robust_condition(m,a,cross,extra,residual)
        np.testing.assert_array_equal(value[0],m);np.testing.assert_array_equal(value[1],a)
    previous=None
    for multiplier in [0.,1.,10.,1e3,1e9]:
        residual=innovation*multiplier;mean,schur,d=robust_condition(m,a,b,c,residual)
        inflated=c/d['weight']
        np.testing.assert_allclose(mean,m+b@np.linalg.solve(inflated,residual),atol=1e-13,rtol=0)
        np.testing.assert_allclose(schur,a-b@np.linalg.solve(inflated,b.T),atol=1e-13,rtol=0)
        np.linalg.cholesky(schur)
        bound=np.sqrt(d['q'])*d['weight'];assert bound<=np.sqrt(k+2)+1e-12
        alternate=robust_condition(m,a,b[:,::-1],c[::-1,::-1],residual[::-1])
        np.testing.assert_allclose(mean,alternate[0],atol=1e-13,rtol=0)
        np.testing.assert_allclose(schur,alternate[1],atol=1e-13,rtol=0)
        previous=(mean,schur)
    assert np.linalg.norm(previous[0]-m)<1e-8 and np.linalg.norm(previous[1]-a)<1e-12
    # Synthetic memberships and exact target-only fallback; no real source values.
    n=82;sample=pd.DataFrame(dict(mvt_id=np.arange(n),month='2025-01',
        gap=np.r_[np.full(80,1500.),9000.,10000.],prior=np.r_[np.full(80,1000.),3000.,3500.],
        eligible=True,quality=True,runway=['A']*40+['B']*40+['A','B'],
        local_day='2025-01-02',assigned_day='2025-01-02',fltrule='I',delay_scope=True))
    taxi=pd.DataFrame(dict(local_day=['2025-01-02']*2,runway=['A','B'],source_n=[41.,41.],source_mean=[1200.,1300.]))
    delay=pd.DataFrame(dict(day=['2025-01-02'],total=[50000.],n_public=[82.]))
    tp=dict(coefficients=[[0.,0.],[0.,0.]],covariance_per_row=[[.1,0.],[0.,10000.]])
    dp=dict(bias=0.,sigma=1000.,kappa=100.);sp=dict(tau=1000.,spread=[-100.,0.,100.])
    value,cells=apply(sample,taxi,delay,tp,dp,sp)
    assert value.touched.sum()==2 and value.extra_cells.sum()==2
    other,_=apply(sample.sample(frac=1,random_state=71).assign(y=1e12,true_day='1900-01-01'),taxi,delay,tp,dp,sp)
    left=value.set_index('mvt_id').sort_index();right=other.set_index('mvt_id').sort_index()
    np.testing.assert_allclose(left.delta,right.delta,atol=1e-8,rtol=0)
    fallback,_=apply(sample,taxi.iloc[:1],delay,tp,dp,sp)
    assert not fallback.delta.any()
    return dict(variance_matched_t_score_derivative=True,threshold_continuity=True,
        PSD_without_repair=True,independent_inflated_noise_equivalence=True,
        bounded_and_vanishing_influence=True,no_extra_disconnected_exact=True,
        coordinate_and_row_permutation=True,label_poison_invariant=True,
        synthetic_source_membership_and_target_fallback=True)
