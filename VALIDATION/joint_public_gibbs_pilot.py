"""One fixed, source-only Gibbs rescue of failed all-peer prior sampling.

No labels, CV scoring, new priors/noise/gates, production or tuning. Systematic
single-site sweeps preserve the exact discrete posterior; they are not claimed
to be reversible as a composition. Target means are Rao-Blackwellized.
"""
import argparse
import ctypes
import itertools
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, rankdata
from threadpoolctl import threadpool_limits

import joint_public_all_discrete_pilot as prior_pilot

ROOT=prior_pilot.ROOT
OUT=ROOT/'tmp/research_20261012/error_mechanisms/gibbs_peers'
KERNEL=Path(__file__).with_name('joint_public_gibbs_kernel.c')
SEEDS=[1701,2701,3701,4701]
BUDGETS=[dict(burn=128,retained=256),dict(burn=256,retained=512)]
METRICS=['rb_mean','rb_zero_delay_probability','target_state_value','taxi_count','taxi_sum','delay_sum','zero_delay_peers','log_likelihood']
DESIGN=dict(prior_pilot.DESIGN,change='exact single-site Gibbs conditional, systematic sorted-ID sweep',
    selection='smallest, median-rank, largest peer-count cohorts of the frozen24-target input pilot; ties sortedmonth/targetID',
    chains=SEEDS,budgets=BUDGETS,
    starts=['independent prior draw','independent prior draw','minimum unmasked delay per peer','maximum unmasked delay per peer'],
    randomness='predeclared independent chain uniforms, reused across separate joint and departure-only kernels',
    target='exact conditional mean/probability at end of each complete sweep',
    monitoring=METRICS+['paired joint-minus-departure RB trace'],
    numerical_success='both budgets all3cohorts: rank/folded split Rhat<=1.01, bulkESS>=400 for nonconstant diagnostics; increment change<=max(1sec,2 combined MCSE); no CV allowed on failure',
    escalation='one pilot; no more sweeps/chains or modified gate if convergence fails',
    no_cv_score=True,no_departure_labels=True)
art,write,read=prior_pilot.art,prior_pilot.write,prior_pilot.parent.read


def kernel():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'kernel_{prior_pilot.parent.sha(KERNEL)[:20]}.so'
    if not path.exists():
        subprocess.run(['cc','-O2','-std=c99','-fPIC','-shared',str(KERNEL),'-lm','-o',str(path)],check=True,timeout=30)
    lib=ctypes.CDLL(str(path));fn=lib.gibbs_chain
    i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int]*4+[i64]+[f64]*9+[i64,f64,i64]
    fn.restype=ctypes.c_int
    return fn,path


def packed(group,view):
    order=np.argsort(group['peer_ids'],kind='stable');ids=np.asarray(group['peer_ids'])[order]
    supports=[group['supports'][i]for i in order];masks=group['masks'][order]
    offsets=np.r_[0,np.cumsum([len(s[0])for s in supports])].astype(np.int64)
    values=np.concatenate([s[0]for s in supports]).astype(float)
    weight=np.concatenate([s[1]for s in supports]).astype(float)
    with np.errstate(divide='ignore'):logprior=np.log(weight)
    vectors=np.concatenate([s[2]*m for s,m in zip(supports,masks)]).astype(float)
    zero=np.concatenate([s[2][:,2]<=60 for s in supports]).astype(float)
    original_delay=np.concatenate([s[2][:,2] for s in supports]).astype(float)
    inverse=np.zeros((3,3),float)
    if view=='joint':inverse[:]=np.linalg.inv(.6*group['remainder_cov']);dims=3
    else:inverse[2,2]=1/(.6*group['remainder_cov'][2,2]);dims=1
    av=vectors@inverse;vav=np.einsum('ij,ij->i',av,vectors)
    return dict(peer_ids=ids,offsets=offsets,values=values,vectors=np.ascontiguousarray(vectors),
        av=np.ascontiguousarray(av),vav=vav,logprior=logprior,zero=zero,delay=original_delay,
        observed=np.asarray(group['observed']-group['remainder_mean'],float),inverse=inverse,dims=dims,
        target=int(np.flatnonzero(ids==group['target_id'])[0]),weights=weight)


def conditional(p,states,j):
    total=p['vectors'][states].sum(0);r=p['observed']-total+p['vectors'][states[j]]
    a,b=p['offsets'][j:j+2];v=p['vectors'][a:b]
    residual=r[None,:]-v
    ll=-.5*(5+p['dims'])*np.log1p(np.maximum(np.einsum('ij,ij->i',residual@p['inverse'],residual),0)/5)
    lw=p['logprior'][a:b]+ll
    return np.exp(lw-logsumexp(lw))


def initial(p,chain,seed):
    rng=np.random.default_rng(seed+101);states=[]
    for j in range(len(p['peer_ids'])):
        a,b=p['offsets'][j:j+2]
        if chain<2:
            cdf=np.cumsum(p['weights'][a:b]/p['weights'][a:b].sum());cdf[-1]=1.
            states.append(a+np.searchsorted(cdf,rng.random(),side='right'))
        else:
            positive=np.flatnonzero(p['weights'][a:b]>0)
            chosen=np.argmin(p['delay'][a:b][positive]) if chain==2 else np.argmax(p['delay'][a:b][positive])
            states.append(a+int(positive[chosen]))
    return np.asarray(states,dtype=np.int64)


def run_chain(fn,p,uniforms,states):
    sweeps,peers=uniforms.shape;trace=np.empty((sweeps,8),float);changes=np.zeros(peers,np.int64)
    status=fn(peers,sweeps,p['target'],p['dims'],p['offsets'],p['values'],p['vectors'],p['av'],p['vav'],
        p['logprior'],p['zero'],p['observed'],p['inverse'],uniforms,states,trace,changes)
    assert status==0 and np.isfinite(trace).all(),status
    return trace,states,changes


def split_diagnostics(draws):
    """Rank/folded split Rhat and bulk ESS with initial-positive paired ACF."""
    draws=np.asarray(draws,float);chains,n=draws.shape;half=n//2
    split=np.concatenate([draws[:,:half],draws[:,-half:]],axis=0)
    def rank_normalize(x):
        return norm.ppf((rankdata(x,method='average').reshape(x.shape)-.375)/(x.size+.25))
    def rhat(x):
        w=float(np.mean(np.var(x,axis=1,ddof=1)));b=float(half*np.var(x.mean(1),ddof=1))
        if w==0:return 1. if b==0 else 1e99
        return float(np.sqrt(((half-1)/half*w+b/half)/w))
    rank=rank_normalize(split);fold=rank_normalize(abs(split-np.median(split)))
    rh=max(rhat(rank),rhat(fold))
    def effective_size(x):
        w=float(np.mean(np.var(x,axis=1,ddof=1)));b=float(half*np.var(x.mean(1),ddof=1))
        varplus=(half-1)/half*w+b/half
        if varplus==0:return float(split.size)
        centered=x-x.mean(axis=1,keepdims=True)
        fft=np.fft.rfft(centered,n=2*half,axis=1)
        acov=np.fft.irfft(fft*np.conj(fft),n=2*half,axis=1)[:,:half]/half
        rho=1-(w-acov.mean(0))/varplus;rho[0]=1.
        pairs=[];last=np.inf
        for t in range(0,half-1,2):
            value=float(rho[t]+rho[t+1])
            if value<0:break
            value=min(value,last);pairs.append(value);last=value
        tau=max(-1+2*sum(pairs),1.)
        return float(min(split.size,split.size/tau))
    ess=effective_size(rank);mean_ess=effective_size(split)
    return dict(mean=float(draws.mean()),sd=float(draws.std(ddof=1)),rhat=rh,bulk_ess=ess,mean_ess=mean_ess,
        mcse=float(draws.std(ddof=1)/np.sqrt(mean_ess)),constant=bool(np.ptp(draws)==0))


def prepare():
    prior_pilot.guard();assert not (OUT/'identity.json').exists()
    with (prior_pilot.OUT/'groups.pkl').open('rb')as f:groups=pickle.load(f)
    groups=sorted(groups,key=lambda g:(len(g['peer_ids']),g['month'],g['target_id']))
    selected=[groups[i]for i in [0,(len(groups)-1)//2,len(groups)-1]]
    fn,compiled=kernel();proof=checks(fn)
    with (OUT/'groups.pkl').open('wb')as f:pickle.dump(selected,f,protocol=5)
    write(OUT/'identity.json',dict(design=DESIGN,source=art(__file__),kernel_source=art(KERNEL),compiled=art(compiled),
        prior_identity=art(prior_pilot.OUT/'identity.json'),prior_groups=art(prior_pilot.OUT/'groups.pkl'),
        prior_rejection=art(prior_pilot.OUT/'assessment.json'),groups=art(OUT/'groups.pkl'),proof=proof,
        selection=[dict(month=g['month'],target_id=g['target_id'],peers=len(g['peer_ids']))for g in selected]))
    print('Gibbs preparation and finite-kernel proof complete',flush=True)


def guard():
    prior_pilot.guard();identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for key in ['source','kernel_source','compiled','prior_identity','prior_groups','prior_rejection','groups']:prior_pilot.parent.verify(identity[key])
    return identity


def pilot(budget_seconds):
    identity=guard();assert not (OUT/'result.json').exists(),'Preserve completed fixed pilot'
    fn,_=kernel();started=time.monotonic();total=max(b['burn']+b['retained']for b in BUDGETS)
    with (OUT/'groups.pkl').open('rb')as f:groups=pickle.load(f)
    allrows=[]
    for group in groups:
        traces={};path=OUT/f"target_{group['target_id']}.npz";receipt=path.with_suffix('.json')
        if receipt.exists():
            saved=read(receipt);assert saved['identity']==art(OUT/'identity.json');prior_pilot.parent.verify(saved['trace'])
            data=np.load(path);traces={v:data[v]for v in ['joint','dep']}
        else:
            before=time.monotonic();extra={}
            for view in ['joint','dep']:
                p=packed(group,view);out=[];changes=[];final=[]
                for chain,seed in enumerate(SEEDS):
                    uniforms=np.random.default_rng(seed).random((total,len(p['peer_ids'])))
                    trace,state,changed=run_chain(fn,p,uniforms,initial(p,chain,seed))
                    out.append(trace);changes.append(changed);final.append(state)
                traces[view]=np.stack(out);extra[view+'_changes']=np.stack(changes);extra[view+'_final_states']=np.stack(final)
            np.savez_compressed(path,**traces,**extra)
            write(receipt,dict(identity=art(OUT/'identity.json'),trace=art(path),seconds=time.monotonic()-before,
                peers=len(group['peer_ids']),target_id=group['target_id']))
            print(group['target_id'],'peers',len(group['peer_ids']),'seconds',round(time.monotonic()-before,3),flush=True)
        bybudget=[]
        for b in BUDGETS:
            take=slice(b['burn'],b['burn']+b['retained'])
            metrics={view:{name:split_diagnostics(trace[:,take,k])for k,name in enumerate(METRICS)}for view,trace in traces.items()}
            metrics['increment']=split_diagnostics(traces['joint'][:,take,0]-traces['dep'][:,take,0])
            bybudget.append(dict(**b,diagnostics=metrics))
        a,b=[r['diagnostics']['increment']for r in bybudget]
        consistency=abs(a['mean']-b['mean']);allowance=max(1.,2*np.hypot(a['mcse'],b['mcse']))
        stable=all(z['constant']or(z['rhat']<=1.01 and z['bulk_ess']>=400)
            for row in bybudget for view in ['joint','dep']for z in row['diagnostics'][view].values())
        stable=stable and all(row['diagnostics']['increment']['rhat']<=1.01 and row['diagnostics']['increment']['bulk_ess']>=400 for row in bybudget)
        allrows.append(dict(month=group['month'],target_id=group['target_id'],peers=len(group['peer_ids']),budgets=bybudget,
            cross_budget_increment_difference=consistency,cross_budget_allowance=allowance,numerically_passes=bool(stable and consistency<=allowance)))
        if time.monotonic()-started>=budget_seconds:
            print('bounded cohort checkpoint; resume same command',flush=True);return
    write(OUT/'result.json',dict(design=DESIGN,identity=art(OUT/'identity.json'),cohorts=allrows,
        traces=[art(OUT/f"target_{g['target_id']}.json")for g in groups],
        numerically_passes=all(r['numerically_passes']for r in allrows),no_labels_or_scores_read=True))
    print('Fixed Gibbs numerical pilot complete',flush=True)


def checks(fn=None):
    if fn is None:fn,_=kernel()
    cov=np.array([[4.,1500.,-100.],[1500.,3e6,-1e6],[-100.,-1e6,4e6]])
    values=[np.array([1000.,8000.]),np.array([500.,4000.,10000.]),np.array([1200.,9500.])]
    weights=[np.array([.4,.6]),np.array([.2,.3,.5]),np.array([.7,.3])]
    group=dict(peer_ids=np.array([31,12,27]),target_id=27,masks=np.array([[1.,1.,0.],[0.,0.,1.],[1.,1.,1.]]),
        supports=[(v,w,prior_pilot.base.contributions(v,11000.))for v,w in zip(values,weights)],
        observed=np.array([2.,7000.,14000.]),remainder_mean=np.array([0.,150.,200.]),remainder_cov=cov)
    worst=0.
    for view in ['joint','dep']:
        p=packed(group,view);states=np.array(list(itertools.product(*[range(a,b)for a,b in zip(p['offsets'][:-1],p['offsets'][1:])])),np.int64)
        lookup={tuple(s):i for i,s in enumerate(states)};mass=[];rb=[]
        for state in states:
            r=p['observed']-p['vectors'][state].sum(0)
            mass.append(p['logprior'][state].sum()-.5*(5+p['dims'])*np.log1p(max(float(r@p['inverse']@r),0)/5))
            a,b=p['offsets'][p['target']:p['target']+2];rb.append(conditional(p,state,p['target'])@p['values'][a:b])
        pi=np.exp(mass-logsumexp(mass));kernels=[]
        for site in range(3):
            transition=np.zeros((len(states),len(states)))
            for i,state in enumerate(states):
                prob=conditional(p,state,site)
                for k,w in enumerate(prob):
                    nxt=state.copy();nxt[site]=p['offsets'][site]+k;transition[i,lookup[tuple(nxt)]]=w
            np.testing.assert_allclose(transition.sum(1),1,atol=1e-14,rtol=0)
            flux=pi[:,None]*transition;np.testing.assert_allclose(flux,flux.T,atol=1e-14,rtol=0)
            kernels.append(transition)
        random_scan=sum(kernels)/3;systematic=kernels[0]@kernels[1]@kernels[2]
        np.testing.assert_allclose(pi@random_scan,pi,atol=1e-14,rtol=0)
        np.testing.assert_allclose(pi@systematic,pi,atol=1e-14,rtol=0)
        exact=float(pi@p['values'][states[:,p['target']]])
        np.testing.assert_allclose(pi@rb,exact,atol=1e-10,rtol=0)
        worst=max(worst,float(np.max(abs(pi@systematic-pi))))
        # Compiler kernel versus separately coded explicit-residual Python steps.
        uniforms=np.random.default_rng(918).random((20,3));start=initial(p,0,17)
        trace,last,_=run_chain(fn,p,uniforms,start.copy());reference=[];state=start.copy()
        for row in uniforms:
            for site,u in enumerate(row):
                prob=conditional(p,state,site);cdf=np.cumsum(prob);cdf[-1]=1.
                state[site]=p['offsets'][site]+np.searchsorted(cdf,u,side='right')
            a,b=p['offsets'][p['target']:p['target']+2]
            reference.append(conditional(p,state,p['target'])@p['values'][a:b])
        np.testing.assert_array_equal(last,state);np.testing.assert_allclose(trace[:,0],reference,atol=1e-9,rtol=0)
    iid=np.random.default_rng(91).normal(size=(4,512));d=split_diagnostics(iid)
    assert d['rhat']<1.02 and d['bulk_ess']>1000
    stuck=np.broadcast_to(np.arange(4.)[:,None],(4,512));assert split_diagnostics(stuck)['rhat']>1.1
    result=dict(single_site_detailed_balance=True,random_scan_detailed_balance=True,systematic_sweep_invariance=True,
        exact_rao_blackwell_moments=True,compiled_vs_explicit_residual_steps=True,max_stationarity_error=worst,
        iid_and_stuck_chain_diagnostics=True)
    print(result,flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['check','prepare','pilot'])
    parser.add_argument('--budget-seconds',type=float,default=15.)
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=='check':checks()
        elif args.stage=='prepare':prepare()
        else:pilot(args.budget_seconds)
