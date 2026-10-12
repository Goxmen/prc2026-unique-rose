"""Reconstruct every final chain output with explicit residuals and SciPy t5."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'tmp/research_20261012/error_mechanisms/pair_gibbs_peers'
OUT=ROOT/'VALIDATION/reports/all_discrete_peer_20261012/pair_gibbs_audit.json'


def art(p):
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):
    return json.loads(p.read_text())


def main():
    assert not OUT.exists()
    identity,result,assessment=[read(RUN/name) for name in ['identity.json','result.json','precision_assessment.json']]
    assert result['identity']==assessment['identity']==art(RUN/'identity.json')
    assert assessment['result']==art(RUN/'result.json')
    assert result['no_labels_or_scores_read'] and assessment['no_labels_or_scores_read']
    for key in ['source','kernel_source','compiled','prior_identity','prior_groups','prior_rejection',
                'groups','single_site_source','single_site_rejection']:
        record=identity[key];assert art(ROOT/record['path'])==record
    with (RUN/'groups.pkl').open('rb') as f:
        groups=pickle.load(f)
    checks=[]
    for group in groups:
        trace_path=RUN/f"target_{group['target_id']}.npz"
        receipt=read(trace_path.with_suffix('.json'))
        assert receipt['identity']==art(RUN/'identity.json') and receipt['trace']==art(trace_path)
        data=np.load(trace_path)
        order=np.argsort(group['peer_ids'],kind='stable')
        peer_ids=np.asarray(group['peer_ids'])[order]
        supports=[group['supports'][i] for i in order]
        masks=group['masks'][order]
        offsets=np.r_[0,np.cumsum([len(s[0]) for s in supports])]
        values=np.concatenate([s[0] for s in supports])
        weights=np.concatenate([s[1] for s in supports])
        vectors=np.concatenate([s[2]*m for s,m in zip(supports,masks)])
        zero=np.concatenate([s[2][:,2]<=60 for s in supports])
        target=int(np.flatnonzero(peer_ids==group['target_id'])[0])
        lo,hi=offsets[target:target+2]
        observed=group['observed']-group['remainder_mean']
        for view in ['joint','dep']:
            dims=[0,1,2] if view=='joint' else [2]
            shape=.6*group['remainder_cov'][np.ix_(dims,dims)]
            normalizer=multivariate_t.logpdf(np.zeros(len(dims)),shape=shape,df=5)
            for chain,state in enumerate(data[view+'_final_states']):
                assert np.all((state>=offsets[:-1])&(state<offsets[1:]))
                total=vectors[state].sum(0)
                other=total-vectors[state[target]]
                residual=observed[None,:]-other-vectors[lo:hi]
                with np.errstate(divide='ignore'):
                    lw=np.log(weights[lo:hi])+multivariate_t.logpdf(residual[:,dims],shape=shape,df=5)
                probability=np.exp(lw-logsumexp(lw))
                ll=multivariate_t.logpdf((observed-total)[dims],shape=shape,df=5)-normalizer
                expected=np.array([probability@values[lo:hi],probability@zero[lo:hi],
                    values[state[target]],*total,zero[state].sum(),ll])
                actual=data[view][chain,-1]
                np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
                checks.append(dict(target=int(group['target_id']),view=view,chain=chain,
                                   max_absolute_error=float(np.max(abs(actual-expected)))))
    assert len(checks)==24
    maximum=max(q['budgets'][b]['diagnostics']['increment']['mcse'] for q in result['cohorts'] for b in [0,1])
    assert maximum==assessment['maximum_increment_mcse']
    assert assessment['numerically_passes']==(result['numerically_passes'] and maximum<=25.)
    report=dict(status='pass',source=art(Path(__file__)),identity=art(RUN/'identity.json'),
        result=art(RUN/'result.json'),assessment=art(RUN/'precision_assessment.json'),
        checks=checks,maximum_absolute_error=max(r['max_absolute_error'] for r in checks),
        scope='Root independent explicit-residual/SciPy density reconstruction, not a second agent review; '
              'does not repeat chains or claim convergence from arithmetic agreement.',
        no_departure_labels=True,no_cv=True)
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(dict(status='pass',outputs=len(checks),maximum_absolute_error=report['maximum_absolute_error']))


if __name__=='__main__':
    main()
