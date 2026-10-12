"""Fixed source-only annealed-particle pilot for coupled discrete public peers.

Unlike independent prior proposals, resampling retains induced dependencies.
The original finite priors and exact t5 likelihood remain unchanged. This is
one numerical feasibility screen, not a taxi-out experiment or candidate.
"""
import argparse
import ctypes
import itertools
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import joint_public_gibbs_pilot as old

ROOT=old.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_joint_public_smc_pilot'
INPUT=ROOT/'tmp/research_20261012/error_mechanisms/gibbs_peers/groups.pkl'
ART=old.art
DESIGN=dict(method='Adaptive likelihood-tempered SMC; systematic resampling; exact tempered Gibbs rejuvenation',
    cohorts='Same frozen7/69/475-peer cohorts as rejected Gibbs pilot',
    particles=[512,2048],replicates=[1701,2701,3701,4701],incremental_ess_fraction=.8,
    sweeps_per_temperature=2,max_temperatures=128,final_sweeps=8,max_runtime_seconds=240,
    target='Original product finite prior times unchanged t5 likelihood; separate joint/departure-only views',
    means='Exact conditional target mean at beta1, averaged over final sweeps',
    precision_gate='Independent replicate SE<=max(1second,1% posterior target SD); cross-budget difference<=3 combined SE; all cohorts and views',
    limitation='Particle ESS is not independent posterior ESS; ancestry and between-replicate error retained. Passing only permits broader numerical assessment, never CV directly.',
    no_departure_labels=True,no_cv=True,no_model_selection=True,no_upload=True,
    failure='Close this fixed screen; no particle/temperature tuning or easier cohort gate')


def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def kernel():
    text=old.KERNEL.read_text()
    changes={
        'int likelihood_dims, double *prob)':'int likelihood_dims, double beta, double *prob)',
        'logprior[j]-.5*(5+likelihood_dims)':'logprior[j]-beta*.5*(5+likelihood_dims)',
        'int target, int likelihood_dims,\n':'int target, int likelihood_dims, double beta,\n',
        'likelihood_dims,prob)':'likelihood_dims,beta,prob)'}
    assert [text.count(k) for k in changes]==[1,1,1,2]
    for a,b in changes.items():text=text.replace(a,b)
    text+='''
int smc_batch(int particles,int peers,int sweeps,int target,int dims,double beta,
 const int64_t *offsets,const double *values,const double *vectors,const double *av,
 const double *vav,const double *lp,const double *zero,const double *observed,
 const double *inverse,const double *uniforms,int64_t *states,double *trace) {
 int64_t *changes=(int64_t*)calloc(peers,sizeof(int64_t)); if(!changes)return 9;
 for(int i=0;i<particles;i++) {
  int status=gibbs_chain(peers,sweeps,target,dims,beta,offsets,values,vectors,av,vav,
   lp,zero,observed,inverse,uniforms+(int64_t)i*sweeps*peers,
   states+(int64_t)i*peers,trace+(int64_t)i*sweeps*8,changes);
  if(status){free(changes);return status;}
 } free(changes);return 0;
}
'''
    source=OUT/'kernel.c'
    if source.exists():assert source.read_text()==text
    else:source.write_text(text)
    binary=OUT/'kernel.so'
    if not binary.exists():subprocess.run(['cc','-O2','-std=c99','-fPIC','-shared',str(source),'-lm','-o',str(binary)],check=True)
    fn=ctypes.CDLL(str(binary)).smc_batch
    i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int]*5+[ctypes.c_double,i64]+[f64]*9+[i64,f64]
    fn.restype=ctypes.c_int
    return fn


def step(fn,p,states,beta,sweeps,uniforms):
    n,peers=states.shape;trace=np.empty((n,sweeps,8),float)
    result=fn(n,peers,sweeps,p['target'],p['dims'],beta,p['offsets'],p['values'],p['vectors'],
        p['av'],p['vav'],p['logprior'],p['zero'],p['observed'],p['inverse'],uniforms,states,trace)
    assert result==0 and np.isfinite(trace).all()
    return trace


def likelihood(p,states):
    r=p['observed']-p['vectors'][states].sum(axis=1)
    return -.5*(5+p['dims'])*np.log1p(np.maximum(np.einsum('ij,ij->i',r@p['inverse'],r),0)/5)


def proof(fn):
    cov=np.array([[4.,1500.,-100.],[1500.,3e6,-1e6],[-100.,-1e6,4e6]])
    values=[np.array([1000.,8000.]),np.array([500.,4000.,10000.]),np.array([1200.,9500.])]
    weights=[np.array([.4,.6]),np.array([.2,.3,.5]),np.array([.7,.3])]
    group=dict(peer_ids=np.array([31,12,27]),target_id=27,masks=np.array([[1.,1.,0.],[0.,0.,1.],[1.,1.,1.]]),
        supports=[(v,w,old.prior_pilot.base.contributions(v,11000.)) for v,w in zip(values,weights)],
        observed=np.array([2.,7000.,14000.]),remainder_mean=np.array([0.,150.,200.]),remainder_cov=cov)
    worst=0.
    for view in ['joint','dep']:
        p=old.packed(group,view)
        states=np.array(list(itertools.product(*[range(a,b) for a,b in zip(p['offsets'][:-1],p['offsets'][1:])])),np.int64)
        lookup={tuple(s):i for i,s in enumerate(states)}
        ll=likelihood(p,states); lp=p['logprior'][states].sum(1)
        for beta in [0.,.125,.5,1.]:
            pi=np.exp(lp+beta*ll-logsumexp(lp+beta*ll));transition=np.eye(len(states))
            def cond(state,j):
                a,b=p['offsets'][j:j+2];trial=np.tile(state,(b-a,1));trial[:,j]=np.arange(a,b)
                z=p['logprior'][a:b]+beta*likelihood(p,trial)
                return np.exp(z-logsumexp(z))
            for j in range(3):
                k=np.zeros((len(states),len(states)))
                for i,state in enumerate(states):
                    for s,w in enumerate(cond(state,j)):
                        nxt=state.copy();nxt[j]=p['offsets'][j]+s;k[i,lookup[tuple(nxt)]]=w
                np.testing.assert_allclose(pi[:,None]*k,(pi[:,None]*k).T,rtol=0,atol=1e-14)
                transition=transition@k
            worst=max(worst,float(np.max(abs(pi@transition-pi))))
            np.testing.assert_allclose(pi@transition,pi,rtol=0,atol=1e-14)
            uniforms=np.random.default_rng(51).random((len(states),2,3));actual=states.copy()
            trace=step(fn,p,actual,beta,2,uniforms)
            for i,state0 in enumerate(states):
                state=state0.copy()
                for sweep in range(2):
                    for j,u in enumerate(uniforms[i,sweep]):
                        cdf=np.cumsum(cond(state,j));cdf[-1]=1.;state[j]=p['offsets'][j]+np.searchsorted(cdf,u,side='right')
                    a,b=p['offsets'][p['target']:p['target']+2]
                    np.testing.assert_allclose(trace[i,sweep,0],cond(state,p['target'])@p['values'][a:b],rtol=0,atol=1e-8)
                np.testing.assert_array_equal(state,actual[i])
            # Exact finite-state importance update, independent of stochastic resampling.
            delta=.03125; q=np.exp(lp+(beta+delta)*ll-logsumexp(lp+(beta+delta)*ll))
            w=pi*np.exp(delta*(ll-ll.max()));w/=w.sum()
            np.testing.assert_allclose(w,q,rtol=0,atol=1e-14)
    return dict(tempered_detailed_balance=True,finite_sweep_invariance=True,
        compiled_explicit_transition_and_RB_parity=True,exact_incremental_importance_identity=True,
        max_invariance_error=worst)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'identity.json').exists()
    prior=old.read(old.OUT/'identity.json');assert prior['groups']==ART(INPUT)
    fn=kernel();checked=proof(fn)
    write(OUT/'identity.json',dict(design=DESIGN,proof=checked,sources=[ART(p) for p in
        [Path(__file__),Path(old.__file__),old.KERNEL,INPUT,old.OUT/'identity.json',OUT/'kernel.c',OUT/'kernel.so']]))
    print(json.dumps(checked),flush=True)


def particles(fn,p,n,seed,deadline):
    rng=np.random.default_rng(seed);states=np.empty((n,len(p['peer_ids'])),np.int64)
    for j,(a,b) in enumerate(zip(p['offsets'][:-1],p['offsets'][1:])):
        cdf=np.cumsum(p['weights'][a:b]/p['weights'][a:b].sum());cdf[-1]=1.
        states[:,j]=a+np.searchsorted(cdf,rng.random(n),side='right')
    ancestry=np.arange(n);beta=0.;levels=[]
    for iteration in range(DESIGN['max_temperatures']):
        if time.monotonic()>deadline:raise TimeoutError('Fixed overall pilot runtime exhausted')
        ll=likelihood(p,states)
        def ess(delta):
            z=delta*ll;w=np.exp(z-logsumexp(z));return 1/np.sum(w*w)
        left,right=0.,1-beta
        if ess(right)<.8*n:
            for _ in range(40):
                mid=(left+right)/2
                if ess(mid)<.8*n:right=mid
                else:left=mid
            delta=left
        else:delta=right
        assert delta>1e-12
        z=delta*ll;w=np.exp(z-logsumexp(z));cdf=np.cumsum(w);cdf[-1]=1.
        take=np.searchsorted(cdf,(rng.random()+np.arange(n))/n,side='right')
        states=np.ascontiguousarray(states[take]);ancestry=ancestry[take];beta=min(1.,beta+delta)
        u=rng.random((n,DESIGN['sweeps_per_temperature'],len(p['peer_ids'])))
        step(fn,p,states,beta,DESIGN['sweeps_per_temperature'],u)
        levels.append(dict(beta=beta,incremental_ess=float(1/np.sum(w*w)),ancestors=len(np.unique(ancestry))))
        if beta==1.:break
    assert beta==1.,'Fixed maximum temperature count exhausted'
    u=rng.random((n,DESIGN['final_sweeps'],len(p['peer_ids'])))
    trace=step(fn,p,states,1.,DESIGN['final_sweeps'],u)
    return dict(mean=float(trace[:,:,0].mean()),copy_probability=float(trace[:,:,1].mean()),
        posterior_target_sd=float(trace[:,:,2].std()),levels=levels,final_ancestors=len(np.unique(ancestry)),
        target_state_diversity=len(np.unique(states[:,p['target']])),source_means=trace[:,-1,3:6].mean(0).tolist()),states


def pilot():
    m=old.read(OUT/'identity.json');assert m['design']==DESIGN and not (OUT/'result.json').exists()
    for a in m['sources']:old.prior_pilot.parent.verify(a)
    fn=kernel();started=time.monotonic();deadline=started+DESIGN['max_runtime_seconds']
    with INPUT.open('rb') as f:groups=pickle.load(f)
    rows=[];failure=None
    try:
        for group in groups:
            for view in ['joint','dep']:
                p=old.packed(group,view);budgets=[]
                for n in DESIGN['particles']:
                    replicates=[]
                    for seed in DESIGN['replicates']:
                        value,states=particles(fn,p,n,seed+n,deadline);replicates.append(value)
                        np.savez_compressed(OUT/f'states_{group["target_id"]}_{view}_{n}_{seed}.npz',states=states)
                    means=np.array([r['mean'] for r in replicates]);se=float(means.std(ddof=1)/np.sqrt(len(means)))
                    budgets.append(dict(particles=n,mean=float(means.mean()),replicate_se=se,
                        posterior_sd=float(np.mean([r['posterior_target_sd'] for r in replicates])),replicates=replicates))
                a,b=budgets;limit=max(1.,.01*b['posterior_sd'])
                consistent=abs(a['mean']-b['mean'])<=3*np.hypot(a['replicate_se'],b['replicate_se'])+1e-8
                row=dict(target_id=group['target_id'],peers=len(p['peer_ids']),view=view,budgets=budgets,
                    precision_limit=limit,passes=bool(b['replicate_se']<=limit and consistent))
                rows.append(row);write(OUT/'progress.json',dict(identity=ART(OUT/'identity.json'),rows=rows))
                print(json.dumps({k:row[k] for k in ['target_id','peers','view','passes']}|dict(mean=b['mean'],se=b['replicate_se'],limit=limit)),flush=True)
    except (TimeoutError,AssertionError) as e:failure=str(e)
    result=dict(identity=ART(OUT/'identity.json'),design=DESIGN,rows=rows,failure=failure,
        runtime_seconds=time.monotonic()-started,numerically_passes=not failure and len(rows)==6 and all(r['passes'] for r in rows),
        no_labels_or_scores_read=True,no_cv_or_candidate=True)
    write(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ['failure','runtime_seconds','numerically_passes']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','pilot'])
    with threadpool_limits(1):globals()[ap.parse_args().stage]()
