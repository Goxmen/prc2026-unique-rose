"""Integrate the dominant uncertain peer explicitly in the joint public update.

Each target uses one deterministic input-only peer: the largest standardized
variance contribution to its three observed moments. Its complete retained
discrete support replaces that peer's Gaussian moments. Remaining peers and
source noise retain the original approximation, with no iterative conditioning.
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import joint_public_moments as base

SETTINGS=dict(base.SETTINGS,peer_count=1,
    peer_selection='maximum trace(inv(target leave-one-out covariance) times projected peer covariance)',
    peer_support='all retained discrete values and weights for the selected long flight',
    tie_break='smallest movement ID after rounding standardized variance shares to12 decimals',
    comparison='same selected peer is explicit in both joint and departure-only calculations',
    control='exact v78; candidate replays v77 with the new fixed half increment',
    no_new_gate=True,no_iteration=True,no_parameter_search=True)


def mixture_mean(values,weights,vectors,peer_weights,peer_vectors,observed,other,covariance):
    """Marginalize one discrete peer before normalizing the target posterior."""
    residual=observed[None,None,:]-other[None,None,:]-vectors[:,None,:]-peer_vectors[None,:,:]
    ll=base.loglike(residual.reshape(-1,residual.shape[-1]),covariance).reshape(len(values),-1)
    ll=logsumexp(ll+np.log(np.maximum(peer_weights,1e-300))[None,:],axis=1)
    lw=np.log(np.maximum(weights,1e-300))+ll
    return float(values@np.exp(lw-logsumexp(lw)))


def apply(inputs,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters):
    names=['mvt_id','gap','prior','eligible','quality','runway','local_day','assigned_day','fltrule','delay_scope']
    rows=inputs[names].copy().reset_index(drop=True)
    assert rows.mvt_id.is_unique and np.isfinite(rows[['gap','prior']]).all().all()
    p,d=rows.prior.to_numpy(float),rows.gap.to_numpy(float)
    vectors,covs,prob=base.normal_moments(p,d);supports={}
    for i in np.flatnonzero(d>3600):
        values,weights,_=base.prior.support(d[i],p[i],False,np.nan,'LIRF',support_parameters)
        value=base.contributions(values,d[i]);mean=weights@value;center=value-mean
        vectors[i],covs[i]=mean,(center.T*weights)@center
        supports[i]=(values,weights,value)
    taxi_ok=(rows.eligible&rows.quality).to_numpy(bool)
    dep_ok=(~rows.fltrule.eq('V')&rows.delay_scope).to_numpy(bool)
    taxi_positions=np.flatnonzero(taxi_ok);dep_positions=np.flatnonzero(dep_ok)
    taxi_groups={key:taxi_positions[ix] for key,ix in rows.loc[taxi_ok].groupby(['local_day','runway']).indices.items()}
    dep_groups={key:dep_positions[ix] for key,ix in rows.loc[dep_ok].groupby('assigned_day').indices.items()}
    taxi_source=source_taxi.set_index(['local_day','runway']);dep_source=source_delay.set_index('day')
    taxi_info={};dep_info={}
    for key,ix in taxi_groups.items():
        if key not in taxi_source.index:continue
        src=taxi_source.loc[key];mean=vectors[ix,:2].sum(0);n=mean[0]
        if not (len(ix)>=20 and src.source_n>=20 and n>=20 and .5<=src.source_n/n<=1.2):continue
        bias=np.array([1.,len(ix)/100.])@np.asarray(taxi_parameters['coefficients'])
        noise=len(ix)*np.asarray(taxi_parameters['covariance_per_row'])+np.diag([1.,3600.])
        taxi_info[key]=dict(ix=ix,mean=mean,cov=covs[ix,:2,:2].sum(0),noise=noise,
            observed=np.array([src.source_n,src.source_n*src.source_mean])+bias,n=n)
    for key,ix in dep_groups.items():
        if key not in dep_source.index:continue
        src=dep_source.loc[key]
        if len(ix)<20 or not np.isfinite(src.total+src.n_public):continue
        dn=float(src.n_public-len(ix));noise=3*(delay_parameters['sigma']**2+(delay_parameters['kappa']*dn)**2)
        loading=float(np.sum(prob[ix]*(d[ix]<=3600)))
        dep_info[key]=dict(ix=ix,mean=float(vectors[ix,2].sum()),cov=float(covs[ix,2,2].sum()),
            noise=noise,observed=float(src.total-delay_parameters['bias']),loading=loading)
    tau=support_parameters['tau'];active=taxi_ok&dep_ok&(d>3600)&(d<50400)&(p>tau)&(p<d)
    result=pd.DataFrame(dict(mvt_id=rows.mvt_id,prior=p,dep_only=p,joint=p,increment=0.,touched=False,
        moment_dep=p,moment_joint=p,peer_id=np.full(len(rows),-1,dtype=np.int64),peer_variance_share=0.))
    pairs={};summaries=[]
    for i in np.flatnonzero(active):
        tkey=(rows.local_day.iloc[i],rows.runway.iloc[i]);dkey=rows.assigned_day.iloc[i]
        if tkey not in taxi_info or dkey not in dep_info:continue
        key=(tkey,dkey)
        if key not in pairs:
            t,a=taxi_info[tkey],dep_info[dkey];shared=np.intersect1d(t['ix'],a['ix'])
            covariance=np.zeros((3,3));covariance[:2,:2]=t['cov'];covariance[2,2]=a['cov']
            covariance[:2,2]=covariance[2,:2]=covs[shared,:2,2].sum(0)
            noise=np.zeros((3,3));noise[:2,:2]=t['noise'];noise[2,2]=a['noise']
            loading=np.array([0.,t['n'],-a['loading']]);noise+=225*np.outer(loading,loading)
            union=np.union1d(t['ix'],a['ix']);peers=union[d[union]>3600]
            peers=peers[np.argsort(rows.mvt_id.to_numpy()[peers],kind='stable')]
            mask=np.column_stack([np.isin(peers,t['ix']),np.isin(peers,t['ix']),np.isin(peers,a['ix'])]).astype(float)
            projected=covs[peers]*mask[:,:,None]*mask[:,None,:]
            pairs[key]=dict(mean=np.r_[t['mean'],a['mean']],cov=covariance+noise,
                observed=np.r_[t['observed'],a['observed']],peers=peers,mask=mask,projected=projected)
            summaries.append(dict(local_day=tkey[0],runway=tkey[1],assigned_day=dkey,discrete_peers=len(peers)))
        group=pairs[key];cov=group['cov']-covs[i];other=group['mean']-vectors[i]
        values,weights,value=supports[i]
        old_joint=base.posterior_mean(values,weights,value,group['observed'],other,cov)
        old_dep=base.posterior_mean(values,weights,value[:,2:],group['observed'][2:],other[2:],cov[2:,2:])
        scores=np.einsum('ab,nba->n',np.linalg.inv(cov),group['projected'])
        scores[group['peers']==i]=-np.inf
        at=int(np.argmax(np.round(scores,12)));j=int(group['peers'][at])
        joint,dep=old_joint,old_dep
        if scores[at]>1e-12:
            _,peer_weights,peer_value=supports[j];mask=group['mask'][at]
            rest_cov=cov-group['projected'][at];rest_mean=other-vectors[j]*mask
            np.linalg.cholesky(rest_cov)
            joint=mixture_mean(values,weights,value,peer_weights,peer_value*mask,group['observed'],rest_mean,rest_cov)
            dep=mixture_mean(values,weights,value[:,2:],peer_weights,peer_value[:,2:]*mask[2:],
                group['observed'][2:],rest_mean[2:],rest_cov[2:,2:])
            result.loc[i,['peer_id','peer_variance_share']]=[int(rows.mvt_id.iloc[j]),scores[at]]
        result.loc[i,['dep_only','joint','increment','touched','moment_dep','moment_joint']]=[dep,joint,joint-dep,True,old_dep,old_joint]
    assert np.isfinite(result[['prior','dep_only','joint','increment','moment_dep','moment_joint']]).all().all()
    return result,pd.DataFrame(summaries)


def checks():
    from scipy.stats import multivariate_t
    values=np.array([1000.,6500.,10000.]);weights=np.array([.3,.4,.3]);gap=11000.
    vectors=base.contributions(values,gap)
    peer_values=np.array([800.,5000.,9000.]);peer_weights=np.array([.4,.2,.4])
    peer_vectors=base.contributions(peer_values,10000.)
    noise=np.array([[4.,1500.,-100.],[1500.,3e6,-1e6],[-100.,-1e6,4e6]])
    observed=np.array([2.,7000.,12000.]);other=np.array([0.,150.,200.]);worst=0.
    for dims in [[2],[0,1,2]]:
        c=noise[np.ix_(dims,dims)];scale=.6*c;sd=np.sqrt(np.diag(scale))
        independent=[]
        for k in range(len(values)):
            mass=0.
            for j in range(len(peer_values)):
                residual=observed[dims]-other[dims]-vectors[k,dims]-peer_vectors[j,dims]
                mass+=peer_weights[j]*multivariate_t.pdf(residual/sd,shape=scale/np.outer(sd,sd),df=5)
            independent.append(weights[k]*mass)
        expected=values@independent/sum(independent)
        actual=mixture_mean(values,weights,vectors[:,dims],peer_weights,peer_vectors[:,dims],observed[dims],other[dims],c)
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0);worst=max(worst,abs(actual-expected))
        degenerate=mixture_mean(values,weights,vectors[:,dims],[1.],peer_vectors[1:2,dims],observed[dims],other[dims],c)
        baseline=base.posterior_mean(values,weights,vectors[:,dims],observed[dims],other[dims]+peer_vectors[1,dims],c)
        np.testing.assert_allclose(degenerate,baseline,atol=1e-9,rtol=0)
    return dict(independent_discrete_enumeration=True,maximum_error=worst,degenerate_peer_equals_fixed_mean=True)
