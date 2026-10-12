"""One source-only target/site block-Gibbs feasibility test.

Retains the old three source cohorts, starts, chains, budgets and diagnostics.
Each other peer is refreshed jointly with the target, allowing compensating
allocations without crossing a low-probability single-flight intermediate.
This is a new transition kernel, not extra sweeps of the rejected sampler.
"""
import argparse
import itertools
from pathlib import Path
import types

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

import joint_public_gibbs_pilot as single

ROOT = single.ROOT
OUT = ROOT / 'tmp/research_20261012/error_mechanisms/pair_gibbs_peers'
KERNEL = Path(__file__).with_name('joint_public_pair_gibbs_kernel.c')
DESIGN = {**single.DESIGN,
    'change': 'exact target/site pair block at every other site; single target block at target site',
    'distinctness': 'allows compensating simultaneous target/peer changes; original sampler stays closed',
    'blocks': 'fixed ascending peer-ID sweep; pair enumeration site-major then target-minor',
    'proof': 'each block detailed balance and complete fixed-sweep invariance on finite exact example',
    'numerical_success': single.DESIGN['numerical_success'] + '; maximum increment MCSE <=25 seconds at both budgets',
    'precision_gate_seconds': 25.,
    'escalation': 'one distinct kernel pilot; no more sweeps, chains, gate changes or CV on failure',
    'interpretation': 'A numerical pass alone is not a prediction gain or production precision guarantee',
}
art, read, write = single.art, single.read, single.write


def block(p, state, site):
    target = p['target']
    sites = [site] if site == target else [site, target]
    choices = list(itertools.product(*[range(p['offsets'][j], p['offsets'][j+1]) for j in sites]))
    remaining = [j for j in range(len(state)) if j not in sites]
    remainder = p['vectors'][state[remaining]].sum(0)
    masses = []
    for choice in choices:
        r = p['observed'] - remainder - p['vectors'][list(choice)].sum(0)
        masses.append(p['logprior'][list(choice)].sum() - .5*(5+p['dims'])*np.log1p(max(float(r@p['inverse']@r), 0)/5))
    return sites, np.asarray(choices, np.int64), np.exp(masses-logsumexp(masses))


def checks(fn=None):
    if fn is None:
        fn, _ = namespace()['kernel']()
    values = [np.array([1000., 8000.]), np.array([500., 4000., 10000.]), np.array([1200., 9500.])]
    weights = [np.array([.4, .6]), np.array([.2, .3, .5]), np.array([.7, .3])]
    group = dict(peer_ids=np.array([31, 12, 27]), target_id=27,
        masks=np.array([[1.,1.,0.], [0.,0.,1.], [1.,1.,1.]]),
        supports=[(v, w, single.prior_pilot.base.contributions(v, 11000.)) for v, w in zip(values, weights)],
        observed=np.array([2., 7000., 14000.]), remainder_mean=np.array([0., 150., 200.]),
        remainder_cov=np.array([[4.,1500.,-100.], [1500.,3e6,-1e6], [-100.,-1e6,4e6]]))
    worst = 0.
    for view in ['joint', 'dep']:
        p = single.packed(group, view)
        states = np.asarray(list(itertools.product(*[range(a,b) for a,b in zip(p['offsets'][:-1],p['offsets'][1:])])), np.int64)
        lookup = {tuple(s): i for i,s in enumerate(states)}
        mass = []
        for state in states:
            r = p['observed']-p['vectors'][state].sum(0)
            mass.append(p['logprior'][state].sum()-.5*(5+p['dims'])*np.log1p(max(float(r@p['inverse']@r),0)/5))
        pi = np.exp(mass-logsumexp(mass))
        complete = np.eye(len(states))
        for site in range(3):
            transition = np.zeros((len(states),len(states)))
            for i,state in enumerate(states):
                sites, choices, probability = block(p,state,site)
                for choice,w in zip(choices,probability):
                    nxt=state.copy();nxt[sites]=choice;transition[i,lookup[tuple(nxt)]]=w
            np.testing.assert_allclose(transition.sum(1),1,atol=1e-14,rtol=0)
            np.testing.assert_allclose(pi[:,None]*transition,(pi[:,None]*transition).T,atol=1e-14,rtol=0)
            complete=complete@transition
        np.testing.assert_allclose(pi@complete,pi,atol=1e-14,rtol=0)
        worst=max(worst,float(np.max(abs(pi@complete-pi))))
        uniforms=np.random.default_rng(918).random((20,3));start=single.initial(p,0,17)
        actual,final,_=single.run_chain(fn,p,uniforms,start.copy())
        state=start.copy();reference=[]
        for row in uniforms:
            for site,u in enumerate(row):
                sites,choices,probability=block(p,state,site)
                cdf=np.cumsum(probability);cdf[-1]=1.
                state[sites]=choices[np.searchsorted(cdf,u,side='right')]
            a,b=p['offsets'][p['target']:p['target']+2]
            conditional=single.conditional(p,state,p['target']);total=p['vectors'][state].sum(0)
            r=p['observed']-total
            reference.append([conditional@p['values'][a:b],conditional@p['zero'][a:b],
                p['values'][state[p['target']]],*total,p['zero'][state].sum(),
                -.5*(5+p['dims'])*np.log1p(max(float(r@p['inverse']@r),0)/5)])
        np.testing.assert_array_equal(final,state)
        np.testing.assert_allclose(actual,reference,atol=1e-8,rtol=0)
    result=dict(each_block_detailed_balance=True,fixed_sweep_invariant=True,
                compiled_all_trace_columns_match_explicit_residual_enumeration=True,
                max_stationarity_error=worst,no_full_sweep_reversibility_claim=True)
    print(result,flush=True)
    return result


def namespace():
    ns=dict(single.__dict__,OUT=OUT,KERNEL=KERNEL,DESIGN=DESIGN,__file__=__file__,checks=checks)
    for name in ['kernel','prepare','guard','pilot']:
        fn=getattr(single,name)
        ns[name]=types.FunctionType(fn.__code__,ns,name,fn.__defaults__,fn.__closure__)
        assert ns[name].__code__ is fn.__code__
    return ns


def prepare():
    assert not (OUT/'identity.json').exists()
    assert read(single.OUT/'result.json')['numerically_passes'] is False
    namespace()['prepare']()
    identity=read(OUT/'identity.json')
    identity['single_site_source']=art(single.__file__)
    identity['single_site_rejection']=art(single.OUT/'result.json')
    identity['review']='Root static review and exact finite-state proof; no unavailable agent review claimed'
    write(OUT/'identity.json',identity)


def pilot():
    identity=read(OUT/'identity.json')
    for key in ['single_site_source','single_site_rejection']:
        single.prior_pilot.parent.verify(identity[key])
    assert not (OUT/'precision_assessment.json').exists()
    namespace()['pilot'](45.)
    if not (OUT/'result.json').exists():
        return
    result=read(OUT/'result.json')
    precision=max(q['budgets'][b]['diagnostics']['increment']['mcse'] for q in result['cohorts'] for b in [0,1])
    passed=result['numerically_passes'] and precision<=DESIGN['precision_gate_seconds']
    write(OUT/'precision_assessment.json',dict(identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        numerically_passes=bool(passed),mixing_gate=bool(result['numerically_passes']),
        maximum_increment_mcse=precision,maximum_permitted_mcse=25.,
        no_labels_or_scores_read=True,no_cv_on_failure=True))
    print(dict(numerically_passes=bool(passed),maximum_increment_mcse=precision),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','prepare','pilot'])
    with threadpool_limits(1):
        stage=parser.parse_args().stage
        if stage=='check':
            checks()
        else:
            globals()[stage]()
