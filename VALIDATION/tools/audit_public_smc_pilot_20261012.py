"""Independent source/finite-state/likelihood and gate audit; no scoring."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_joint_public_smc_pilot'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def art(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))


def audit():
    assert not (OUT/'audit.json').exists()
    identity=json.loads((OUT/'identity.json').read_text());result=json.loads((OUT/'result.json').read_text())
    assert result['identity']==art(OUT/'identity.json') and result['design']==identity['design']
    for a in identity['sources']:assert sha(ROOT/a['path'])==a['sha256']
    prior=json.loads((ROOT/'tmp/research_20261012/error_mechanisms/gibbs_peers/identity.json').read_text())
    assert prior['source']==art(ROOT/'VALIDATION/joint_public_gibbs_pilot.py')
    assert prior['kernel_source']==art(ROOT/'VALIDATION/joint_public_gibbs_kernel.c')
    groups=pickle.load((ROOT/prior['groups']['path']).open('rb'))
    byid={int(g['target_id']):g for g in groups};files=[];states_count=0;maxerr=0.;gates=[]
    for row in result['rows']:
        g=byid[row['target_id']];order=np.argsort(g['peer_ids'],kind='stable')
        supports=[g['supports'][i] for i in order];masks=g['masks'][order]
        offsets=np.r_[0,np.cumsum([len(s[0]) for s in supports])]
        vectors=np.concatenate([s[2]*m for s,m in zip(supports,masks)])
        weights=np.concatenate([s[1] for s in supports]);observed=g['observed']-g['remainder_mean']
        take=[0,1,2] if row['view']=='joint' else [2]
        scale=(.6*g['remainder_cov'])[np.ix_(take,take)];inv=np.linalg.inv(scale)
        for budget in row['budgets']:
            means=np.array([r['mean'] for r in budget['replicates']]);se=means.std(ddof=1)/np.sqrt(len(means))
            np.testing.assert_allclose([means.mean(),se],[budget['mean'],budget['replicate_se']],atol=1e-12,rtol=0)
            for seed,rep in zip(identity['design']['replicates'],budget['replicates']):
                levels=rep['levels'];assert levels[-1]['beta']==1 and np.all(np.diff([0]+[l['beta'] for l in levels])>0)
                assert all(l['incremental_ess']>=.8*budget['particles']-1e-6 for l in levels)
                p=OUT/f'states_{row["target_id"]}_{row["view"]}_{budget["particles"]}_{seed}.npz'
                states=np.load(p)['states'];assert states.shape==(budget['particles'],len(supports))
                assert ((states>=offsets[:-1])&(states<offsets[1:])).all() and (weights[states]>0).all()
                # Direct source residuals and independently normalized SciPy t5 density.
                residual=(observed-vectors[states].sum(1))[:,take]
                direct=-.5*(5+len(take))*np.log1p(np.einsum('ij,ij->i',residual@inv,residual)/5)
                scipy=multivariate_t.logpdf(residual,loc=np.zeros(len(take)),shape=scale,df=5)
                constant=float(multivariate_t.logpdf(np.zeros(len(take)),loc=np.zeros(len(take)),shape=scale,df=5))
                error=float(np.max(abs(direct-(scipy-constant))));assert error<1e-7;maxerr=max(maxerr,error)
                states_count+=len(states);files.append(art(p))
        a,b=row['budgets'];limit=max(1.,.01*b['posterior_sd'])
        gate=b['replicate_se']<=limit and abs(a['mean']-b['mean'])<=3*np.hypot(a['replicate_se'],b['replicate_se'])+1e-8
        assert gate==row['passes'] and limit==row['precision_limit'];gates.append(bool(gate))
    assert result['numerically_passes']==(not result['failure'] and len(gates)==6 and all(gates))
    value=dict(status='pass',result=art(OUT/'result.json'),identity=art(OUT/'identity.json'),
        sources_unchanged=True,original_prior_kernel_identity_exact=True,particle_states_checked=states_count,
        max_independent_scipy_likelihood_error=maxerr,replicate_and_gate_arithmetic_exact=True,
        numerical_gate_passes=result['numerically_passes'],files=files,
        interpretation='A correct sampler implementation does not establish adequate posterior-mean precision. Fixed source-only gate fails; no CV follows.')
    (OUT/'audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['status','particle_states_checked','max_independent_scipy_likelihood_error','numerical_gate_passes']}))


if __name__=='__main__':
    with threadpool_limits(1):audit()
