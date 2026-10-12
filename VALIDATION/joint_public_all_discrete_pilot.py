"""Source-only numerical pilot for all-discrete long-peer marginal messages.

No validation labels, scores, candidate, production, fitting, or weight search.
Every pilot target retains its exact support. Independent Sobol coordinates
sample every other long peer; a flight's single draw drives all source terms.
The remaining t5 covariance retains only ordinary-flight and source/common noise.
This integrates a local pair of cohorts, not a global factor graph.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

import joint_public_moments as base
import joint_public_moments_cv as parent

ROOT=parent.ROOT
OUT=ROOT/'tmp/research_20261012/error_mechanisms/all_discrete_peers'
PEER=ROOT/'VALIDATION/runs/v3_20261011_joint_public_peer'
SEEDS=[1701,2701]
POWERS=[10,12,14]
DESIGN=dict(base.SETTINGS,
    change='all long-flight discrete supports integrated jointly within each target cohort pair',
    scope='source-only numerical pilot; no labels or scores',
    selection='12 cohorts per month at equally spaced ranks of (long-peer count,localday,runway,assignedday); smallest active movement ID per selected cohort',
    sampling='independent inverse-CDF Sobol dimensions in ascending movement-ID order, all retained support values/weights',
    projection='one sampled flight value jointly drives every taxi/delay component it belongs to',
    target='Rao-Blackwell exact support, replacing its sampled value in each draw',
    remainder='subtract ALL masked long-peer means/covariances including target once; retain original ordinary/source/common15sec covariance',
    scrambles=SEEDS,powers=POWERS,
    convergence='joint/dep/increment two-scramble and nested-N differences; target-marginal likelihood ESS',
    no_new_gate=True,no_fit=True,no_cv_score=True,no_production=True,
    limitation='local marginal integration only; does not jointly condition overlapping cohort factors')


def art(path): return parent.art(path)
def write(path,x): parent.write(path,x)


def integral(values,weights,vectors,other_draws,observed,remainder_mean,remainder_cov,draw_weights=None):
    """Exact target support, weighted Monte Carlo over peer configurations.

    The quadratic expansion avoids an N x support x dimension residual array.
    Optional draw weights are used only by exact-enumeration checks.
    """
    values,weights,vectors=np.asarray(values,float),np.asarray(weights,float),np.asarray(vectors,float)
    other_draws=np.asarray(other_draws,float)
    covariance=np.atleast_2d(np.asarray(remainder_cov,float))
    scale=.6*covariance
    np.linalg.cholesky(scale)
    inverse=np.linalg.inv(scale)
    residual=np.asarray(observed)[None,:]-np.asarray(remainder_mean)[None,:]-other_draws
    ra=residual@inverse
    quadratic=np.einsum('ij,ij->i',ra,residual)[:,None]-2*(ra@vectors.T)
    quadratic+=np.einsum('ij,ij->i',vectors@inverse,vectors)[None,:]
    ll=-.5*np.linalg.slogdet(scale)[1]-.5*(5+vectors.shape[1])*np.log1p(np.maximum(quadratic,0)/5)
    log_target=np.log(np.maximum(weights,1e-300))
    if draw_weights is None: log_draw=np.full(len(other_draws),-np.log(len(other_draws)))
    else:
        dw=np.asarray(draw_weights,float);assert np.all(dw>=0) and dw.sum()>0
        log_draw=np.log(np.maximum(dw/dw.sum(),1e-300))
    target_log_mass=log_target+logsumexp(ll+log_draw[:,None],axis=0)
    post=np.exp(target_log_mass-logsumexp(target_log_mass))
    draw_log_mass=log_draw+logsumexp(ll+log_target[None,:],axis=1)
    normalized=np.exp(draw_log_mass-logsumexp(draw_log_mass))
    return dict(mean=float(values@post),ess=float(1/np.sum(normalized**2)),
        ess_fraction=float(1/(len(normalized)*np.sum(normalized**2))),posterior=post)


def sample_cohort(group,seed,power):
    """Canonical ordering makes row permutation irrelevant to the Sobol law."""
    order=np.argsort(group['peer_ids'],kind='stable')
    ids=np.asarray(group['peer_ids'])[order]
    assert len(np.unique(ids))==len(ids)
    uniform=qmc.Sobol(d=len(ids),scramble=True,seed=seed).random_base2(power)
    total=np.zeros((len(uniform),3),float);target_draw=None
    for dim,position in enumerate(order):
        support=group['supports'][position]
        weights=np.asarray(support[1],float);cdf=np.cumsum(weights/weights.sum());cdf[-1]=1.
        states=np.searchsorted(cdf,uniform[:,dim],side='right')
        draw=np.asarray(support[2])[states]*group['masks'][position]
        total+=draw
        if group['peer_ids'][position]==group['target_id']: target_draw=draw.copy()
    assert target_draw is not None
    return total-target_draw


def guard():
    identity=parent.read(OUT/'identity.json');assert identity['design']==DESIGN
    for item in [*identity['sources'],*identity['inputs'],identity['groups'],identity['selection']]:parent.verify(item)
    return identity


def prepare():
    """Build only the input-selected cohort unions, not every long-flight prior."""
    assert not (OUT/'identity.json').exists(),'Preserve completed preparation'
    parent.guard()
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'design.json',DESIGN)
    inputs=pd.read_parquet(parent.OUT/'inputs.parquet')
    source_taxi=pd.read_parquet(parent.OUT/'taxi_source.parquet').set_index(['local_day','runway'])
    source_delay=pd.read_parquet(parent.OUT/'delay_source.parquet').set_index('day')
    parameters=parent.read(parent.OUT/'parameters.json')
    cells=pd.read_parquet(PEER/'cells.parquet')
    old=pd.read_parquet(parent.OUT/'predictions.parquet',columns=['mvt_id','touched','dep_only','joint']).set_index('mvt_id')
    parent_inference=parent.read(parent.OUT/'inference.json')
    assert parent_inference['identity']==art(parent.OUT/'identity.json')
    for item in parent_inference['artifacts']: parent.verify(item)
    peer_identity=parent.read(PEER/'identity.json');peer_inference=parent.read(PEER/'inference.json')
    assert peer_inference['identity']==art(PEER/'identity.json')
    for name in ['cells.parquet']:
        record=next(x for x in peer_inference['artifacts'] if x['path']==str((PEER/name).relative_to(ROOT)))
        parent.verify(record)
    parent.verify(peer_identity['parent_identity'])
    groups=[];selection=[];started=time.monotonic()
    for month,prm in parameters.items():
        rows=inputs.loc[inputs.month.eq(month)].reset_index(drop=True)
        assert not {'y','true_day','BLOCK_TIME_UTC_mvt'}&set(rows)
        p,d=rows.prior.to_numpy(float),rows.gap.to_numpy(float)
        taxi_ok=(rows.eligible&rows.quality).to_numpy(bool)
        dep_ok=(~rows.fltrule.eq('V')&rows.delay_scope).to_numpy(bool)
        touched=old.loc[rows.mvt_id,'touched'].to_numpy(bool)
        taxi_pos,dep_pos=np.flatnonzero(taxi_ok),np.flatnonzero(dep_ok)
        taxi_groups={key:taxi_pos[ix] for key,ix in rows.loc[taxi_ok].groupby(['local_day','runway']).indices.items()}
        dep_groups={key:dep_pos[ix] for key,ix in rows.loc[dep_ok].groupby('assigned_day').indices.items()}
        tab=cells.loc[cells.month.eq(month)].sort_values(['discrete_peers','local_day','runway','assigned_day']).reset_index(drop=True)
        assert len(tab)>=12
        chosen=tab.iloc[np.rint(np.linspace(0,len(tab)-1,12)).astype(int)]
        vectors,covs,prob=base.normal_moments(p,d);support={};spec=[]
        for row in chosen.itertuples():
            tk=(row.local_day,row.runway);dk=row.assigned_day
            ti,di=taxi_groups[tk],dep_groups[dk]
            target=np.flatnonzero(touched&rows.local_day.eq(row.local_day).to_numpy()&rows.runway.eq(row.runway).to_numpy()&rows.assigned_day.eq(dk).to_numpy())
            assert len(target)>0
            i=int(target[np.argmin(rows.mvt_id.to_numpy()[target])])
            assert taxi_ok[i] and dep_ok[i] and 3600<d[i]<50400 and prm['support']['tau']<p[i]<d[i]
            union=np.union1d(ti,di);peers=union[d[union]>3600]
            peers=peers[np.argsort(rows.mvt_id.to_numpy()[peers],kind='stable')]
            assert len(peers)==row.discrete_peers and i in peers
            spec.append((row,tk,dk,ti,di,i,peers))
        needed=np.unique(np.concatenate([s[-1] for s in spec]))
        for i in needed:
            values,weights,_=base.prior.support(d[i],p[i],False,np.nan,'LIRF',prm['support'])
            value=base.contributions(values,d[i]);mean=weights@value;center=value-mean
            vectors[i],covs[i]=mean,(center.T*weights)@center
            support[i]=(values,weights,value)
        for row,tk,dk,ti,di,i,peers in spec:
            ts,ds=source_taxi.loc[tk],source_delay.loc[dk]
            tmean=vectors[ti,:2].sum(0);n=tmean[0]
            assert len(ti)>=20 and ts.source_n>=20 and n>=20 and .5<=ts.source_n/n<=1.2
            assert len(di)>=20 and np.isfinite(ds.total+ds.n_public)
            bias=np.array([1.,len(ti)/100.])@np.asarray(prm['taxi']['coefficients'])
            tnoise=len(ti)*np.asarray(prm['taxi']['covariance_per_row'])+np.diag([1.,3600.])
            dn=float(ds.n_public-len(di))
            dnoise=3*(prm['delay']['sigma']**2+(prm['delay']['kappa']*dn)**2)
            loading=float(np.sum(prob[di]*(d[di]<=3600)))
            shared=np.intersect1d(ti,di)
            covariance=np.zeros((3,3));covariance[:2,:2]=covs[ti,:2,:2].sum(0)
            covariance[2,2]=covs[di,2,2].sum()
            covariance[:2,2]=covariance[2,:2]=covs[shared,:2,2].sum(0)
            noise=np.zeros((3,3));noise[:2,:2]=tnoise;noise[2,2]=dnoise
            common_loading=np.array([0.,n,-loading]);noise+=225*np.outer(common_loading,common_loading)
            covariance+=noise
            mean=np.r_[tmean,vectors[di,2].sum()]
            observed=np.r_[np.array([ts.source_n,ts.source_n*ts.source_mean])+bias,ds.total-prm['delay']['bias']]
            mask=np.column_stack([np.isin(peers,ti),np.isin(peers,ti),np.isin(peers,di)]).astype(float)
            projected_cov=covs[peers]*mask[:,:,None]*mask[:,None,:]
            projected_mean=vectors[peers]*mask
            remainder_cov=covariance-projected_cov.sum(0)
            remainder_mean=mean-projected_mean.sum(0)
            np.linalg.cholesky(remainder_cov)
            values,weights,value=support[i]
            ctrl_joint=base.posterior_mean(values,weights,value,observed,mean-vectors[i],covariance-covs[i])
            ctrl_dep=base.posterior_mean(values,weights,value[:,2:],observed[2:],mean[2:]-vectors[i,2:],covariance[2:,2:]-covs[i,2:,2:])
            saved=old.loc[rows.mvt_id.iloc[i]]
            np.testing.assert_allclose([ctrl_joint,ctrl_dep],[saved.joint,saved.dep_only],atol=1e-7,rtol=0)
            # Independent remainder construction, summing only unsampled flights.
            ot=ti[d[ti]<=3600];od=di[d[di]<=3600];os=np.intersect1d(ot,od)
            direct=np.zeros((3,3));direct[:2,:2]=covs[ot,:2,:2].sum(0);direct[2,2]=covs[od,2,2].sum()
            direct[:2,2]=direct[2,:2]=covs[os,:2,2].sum(0);direct+=noise
            np.testing.assert_allclose(remainder_cov,direct,atol=1e-4,rtol=1e-10)
            np.testing.assert_allclose(remainder_mean,np.r_[vectors[ot,:2].sum(0),vectors[od,2].sum()],atol=1e-7,rtol=0)
            group=dict(month=month,local_day=tk[0],runway=tk[1],assigned_day=dk,target_id=int(rows.mvt_id.iloc[i]),
                peer_ids=rows.mvt_id.to_numpy()[peers].astype(np.int64),masks=mask,supports=[support[j] for j in peers],
                target_support=support[i],observed=observed,remainder_mean=remainder_mean,remainder_cov=remainder_cov,
                moment_joint=ctrl_joint,moment_dep=ctrl_dep)
            groups.append(group)
            selection.append(dict(month=month,local_day=tk[0],runway=tk[1],assigned_day=dk,target_id=group['target_id'],
                peers=len(peers),states=len(values),remainder_min_eigenvalue=float(np.linalg.eigvalsh(remainder_cov).min()),
                moment_joint=ctrl_joint,moment_dep=ctrl_dep))
        print(month,'selected',len(spec),'distinct supports',len(needed),'seconds',time.monotonic()-started,flush=True)
    assert len(groups)==24
    with (OUT/'groups.pkl').open('wb') as f:pickle.dump(groups,f,protocol=5)
    write(OUT/'selection.json',selection)
    sources=[Path(__file__),Path(base.__file__),Path(base.prior.__file__),Path(base.prior.base.__file__),Path(base.taxi.__file__),Path(parent.__file__)]
    paths=[parent.OUT/n for n in ['inputs.parquet','taxi_source.parquet','delay_source.parquet','parameters.json','identity.json','predictions.parquet','inference.json','input_audit.json']]
    paths.extend(PEER/n for n in ['cells.parquet','identity.json','inference.json'])
    write(OUT/'identity.json',dict(design=DESIGN,sources=[art(p) for p in sources],inputs=[art(p) for p in paths],
        groups=art(OUT/'groups.pkl'),selection=art(OUT/'selection.json'),seconds=time.monotonic()-started,
        checks=dict(no_labels_read=True,selected_original_moments_reproduced=True,remainder_direct_sum_reproduced=True)))


def pilot(budget):
    identity=guard();assert not (OUT/'result.json').exists(),'Preserve completed numerical pilot'
    with (OUT/'groups.pkl').open('rb') as f:groups=pickle.load(f)
    log=OUT/'progress.json'
    if log.exists():
        saved=parent.read(log);assert saved['identity']==art(OUT/'identity.json')
        records=saved['records']
    else: records=[]
    done={(r['target_id'],r['seed']) for r in records};started=time.monotonic()
    for group in groups:
        for seed in SEEDS:
            if (group['target_id'],seed) in done:continue
            before=time.monotonic();draws=sample_cohort(group,seed,max(POWERS));values,weights,vectors=group['target_support']
            results=[]
            for power in POWERS:
                n=2**power
                joint=integral(values,weights,vectors,draws[:n],group['observed'],group['remainder_mean'],group['remainder_cov'])
                dep=integral(values,weights,vectors[:,2:],draws[:n,2:],group['observed'][2:],group['remainder_mean'][2:],group['remainder_cov'][2:,2:])
                results.append(dict(power=power,draws=n,joint=joint['mean'],dep=dep['mean'],increment=joint['mean']-dep['mean'],
                    joint_ess=joint['ess'],dep_ess=dep['ess'],joint_ess_fraction=joint['ess_fraction'],dep_ess_fraction=dep['ess_fraction'],
                    joint_posterior=joint['posterior'].tolist(),dep_posterior=dep['posterior'].tolist()))
            records.append(dict(month=group['month'],target_id=group['target_id'],peer_count=len(group['peer_ids']),seed=seed,
                moment_joint=group['moment_joint'],moment_dep=group['moment_dep'],results=results,seconds=time.monotonic()-before))
            write(log,dict(identity=art(OUT/'identity.json'),records=records))
            print(group['month'],group['target_id'],seed,'peers',len(group['peer_ids']),'seconds',round(time.monotonic()-before,3),flush=True)
            if time.monotonic()-started>=budget:
                print('bounded pilot checkpoint; resume same command',flush=True);return
    assert len(records)==48
    summaries=[]
    for group in groups:
        pair=[r for r in records if r['target_id']==group['target_id']]
        assert {r['seed'] for r in pair}==set(SEEDS)
        rows=[]
        for idx,power in enumerate(POWERS):
            a,b=[r['results'][idx] for r in pair]
            rows.append(dict(power=power,**{f'{name}_scramble_difference':abs(a[name]-b[name]) for name in ['joint','dep','increment']},
                joint_min_ess=min(a['joint_ess'],b['joint_ess']),dep_min_ess=min(a['dep_ess'],b['dep_ess']),
                joint_posterior_l1=float(np.sum(abs(np.asarray(a['joint_posterior'])-b['joint_posterior']))),
                dep_posterior_l1=float(np.sum(abs(np.asarray(a['dep_posterior'])-b['dep_posterior'])))))
        nested=[dict(seed=r['seed'],**{f'{name}_4096_to16384':abs(r['results'][-1][name]-r['results'][-2][name])
            for name in ['joint','dep','increment']}) for r in pair]
        summaries.append(dict(month=group['month'],target_id=group['target_id'],peers=len(group['peer_ids']),scramble=rows,nested=nested))
    write(OUT/'result.json',dict(design=DESIGN,identity=art(OUT/'identity.json'),progress=art(log),summaries=summaries,
        inference_seconds=sum(r['seconds'] for r in records),no_labels_or_scores_read=True))
    print('Numerical pilot complete; no model-performance claim',flush=True)


def checks():
    from scipy.stats import multivariate_t
    values=np.array([1000.,6500.,10000.]);weights=np.array([.3,.4,.3]);vectors=base.contributions(values,11000.)
    pv=[np.array([800.,9000.]),np.array([1000.,5000.,9500.])];pw=[np.array([.4,.6]),np.array([.2,.3,.5])]
    masks=np.array([[1.,1.,0.],[0.,0.,1.]])
    vv=[base.contributions(v,10000.)*mask for v,mask in zip(pv,masks)]
    indices=list(itertools.product(range(2),range(3)))
    draws=np.array([vv[0][a]+vv[1][b] for a,b in indices]);dw=np.array([pw[0][a]*pw[1][b] for a,b in indices])
    covariance=np.array([[4.,1500.,-100.],[1500.,3e6,-1e6],[-100.,-1e6,4e6]])
    observed=np.array([2.,7000.,12000.]);other=np.array([0.,150.,200.]);worst=0.
    for dims in [[2],[0,1,2]]:
        cov=covariance[np.ix_(dims,dims)];v=vectors[:,dims];ds=draws[:,dims]
        independent=[]
        for k in range(len(values)):
            total=sum(w*multivariate_t.pdf(observed[dims]-other[dims]-v[k]-draw,shape=.6*cov,df=5) for w,draw in zip(dw,ds))
            independent.append(weights[k]*total)
        expected=values@independent/sum(independent)
        actual=integral(values,weights,v,ds,observed[dims],other[dims],cov,dw)
        np.testing.assert_allclose(actual['mean'],expected,atol=1e-9,rtol=0);worst=max(worst,abs(actual['mean']-expected))
        degenerate=integral(values,weights,v,ds[1:2],observed[dims],other[dims],cov)
        control=base.posterior_mean(values,weights,v,observed[dims],other[dims]+ds[1],cov)
        np.testing.assert_allclose(degenerate['mean'],control,atol=1e-9,rtol=0)
    group=dict(peer_ids=np.array([91,22,47]),target_id=47,masks=np.array([[1.,1.,0.],[0.,0.,1.],[1.,1.,1.]]),
        supports=[(pv[0],pw[0],base.contributions(pv[0],10000.)),(pv[1],pw[1],base.contributions(pv[1],10000.)),(values,weights,vectors)])
    draw=sample_cohort(group,1701,10);order=[2,0,1]
    perm=dict(group,peer_ids=group['peer_ids'][order],masks=group['masks'][order],supports=[group['supports'][j] for j in order])
    np.testing.assert_array_equal(draw,sample_cohort(perm,1701,10))
    np.testing.assert_array_equal(draw,sample_cohort(group,1701,12)[:1024])
    print(dict(exact_enumeration=True,max_error=worst,degenerate_equals_fixed_mean=True,
        canonical_permutation_exact=True,nested_sobol_exact=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['check','prepare','pilot'])
    parser.add_argument('--budget-seconds',type=float,default=15.)
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=='check':checks()
        elif args.stage=='prepare':prepare()
        else:pilot(args.budget_seconds)
