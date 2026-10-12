"""Frozen, source-only EP toy falsifier; never infer complete real cohorts.

Finite supports remain coupled count/taxi-sum/delay vectors. A Gaussian EP
site on their sum approximates the unchanged t5 likelihood, whose tilted
moments use deterministic Gamma-precision integration. No outcome tuning.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import resource
import time

import numpy as np
from scipy.special import gammaln, logsumexp, roots_genlaguerre
from scipy.integrate import quad_vec
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tmp/research_20261012/error_mechanisms/all_discrete_peers'
OUT = ROOT/'tmp/research_20261012/signal_scout/ep_toy'
DESIGN = dict(
    scope='only frozen 2/3-peer toys; no full real cohort inference, labels, CV, fit or candidate',
    actual_cases='smallest/largest source cohort by(peer_count,month,target_id), target plus smallest-ID1/2 others; omitted peers fixed at their prior means',
    synthetic_cases=['two competing copy assignments with singular aggregate and nonzero orthogonal residual','three cutoff-crossing coupled supports'],
    representation='whiten .6*remainder covariance; exact peer and aggregate affine SVD spans; rank0 deterministic; no covariance floors',
    prior_sites='Gaussian natural parameters, removed before exact finite-support tilt; coupled components never separated',
    likelihood='unchanged df5 Student t; Gamma(2.5,rate2.5) mixture; preserve orthogonal residual evidence',
    baselines=['full discrete enumeration','current prior-peer-moment t5 approximation'],
    orders=['ascending_id','descending_id'], damping=[.25,.5],
    initializations=['prior','soft_lower','soft_upper'],
    initializer='Gaussian moments of original support weights times exp(0/-2/+2 times standardized original y); exact model prior stays original',
    sweeps=64, checkpoints=[32,64], quadrature_nodes=[32,64],
    update='aggregate site first, then each peer in frozen order; fixed natural-parameter damping; no backtracking',
    thresholds=dict(mean_sec=1.,copy_probability=.001,stability_mean_sec=1.,stability_copy_probability=.001,
                    last_sweep_mean_sec=.01,last_sweep_copy_probability=.00001,
                    density_log_error=1e-9,quadrature_whitened_mean=1e-6,quadrature_whitened_covariance=1e-6),
    quadrature_audit='each case/view original-prior aggregate cavity; fixed quad_vec reference epsabs1e-11/epsrel1e-11/limit200, both32/64nodes must pass',
    resource=dict(threads=1,wall_seconds=60,incremental_workspace_mib=256),
    decision='reject this fixed approximation if any run is invalid/unconverged or any evidence threshold fails; no repair, tuning or budget extension',
    interpretation='numerical evidence thresholds do not prove every biased approximation scientifically useless',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def contributions(y,gap):
    y=np.asarray(y,float);valid=(y>=0)&(y<=7200)
    return np.column_stack([valid,np.where(valid,y,0.),np.maximum(gap-y,0.)])


def freeze():
    """Only extract fixed tiny cases; do not run any posterior computation."""
    identity=json.loads((SOURCE/'identity.json').read_text())
    assert sha(SOURCE/'groups.pkl')==identity['groups']['sha256']
    with (SOURCE/'groups.pkl').open('rb') as f:groups=pickle.load(f)
    groups=sorted(groups,key=lambda g:(len(g['peer_ids']),g['month'],g['target_id']))
    cases=[]
    for g in [groups[0],groups[-1]]:
        target=list(g['peer_ids']).index(g['target_id'])
        others=sorted((i for i in range(len(g['peer_ids'])) if i!=target),key=lambda i:g['peer_ids'][i])
        for size in [2,3]:
            chosen=[target,*others[:size-1]];omitted=[i for i in range(len(g['peer_ids'])) if i not in chosen]
            fixed=np.asarray(g['remainder_mean'],float).copy()
            for i in omitted:
                _,w,v=g['supports'][i];fixed+=(np.asarray(w)@np.asarray(v))*g['masks'][i]
            peers=[]
            for i in chosen:
                y,w,v=g['supports'][i]
                peers.append(dict(id=int(g['peer_ids'][i]),values=np.asarray(y).tolist(),weights=np.asarray(w).tolist(),
                    vectors=(np.asarray(v)*g['masks'][i]).tolist(),copy=(np.asarray(v)[:,2]<=60).tolist()))
            cases.append(dict(name=f'actual_{len(g["peer_ids"])}_{size}',source_target=int(g['target_id']),
                source_month=g['month'],source_peer_count=len(g['peer_ids']),peers=peers,
                observed=(np.asarray(g['observed'])-fixed).tolist(),covariance=np.asarray(g['remainder_cov']).tolist()))
    synthetic=[
        ('competing_copy',[[900.,10000.],[900.,10000.]],[[.3,.7],[.7,.3]],[10000.,10000.],
         [1.2,900.,9100.],np.diag([.04,150.**2,150.**2])),
        ('cutoff_crossing',[[7000.,7199.,7201.,11000.],[800.,7200.,7201.,12000.],[1000.,7000.,13000.]],
         [[.2,.25,.25,.3],[.25,.25,.2,.3],[.4,.3,.3]],[11000.,12000.,13000.],
         [2.,14500.,12000.],np.array([[.5,30.,-20.],[30.,1e6,-2e5],[-20.,-2e5,2e6]])),
    ]
    for name,ys,ws,gaps,observed,cov in synthetic:
        peers=[]
        for i,(y,w,gap) in enumerate(zip(ys,ws,gaps)):
            v=contributions(y,gap)
            peers.append(dict(id=i+1,values=y,weights=w,vectors=v.tolist(),copy=(v[:,2]<=60).tolist()))
        cases.append(dict(name=name,peers=peers,observed=observed,covariance=cov.tolist()))
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SOURCE/'identity.json',SOURCE/'groups.pkl',SOURCE/'selection.json']}
    receipt=dict(design=DESIGN,sources=sources,cases=cases)
    if (OUT/'design.json').exists():assert json.loads((OUT/'design.json').read_text())==receipt
    else:write(OUT/'design.json',receipt)
    print('Frozen six tiny cases; no posterior computation.',flush=True)


def finite_case(case,dims):
    peers=[]
    for p in case['peers']:
        w=np.asarray(p['weights'],float);w=w/w.sum();keep=w>0
        peers.append(dict(id=p['id'],y=np.asarray(p['values'],float)[keep],w=w[keep],
            g=np.asarray(p['vectors'],float)[keep][:,dims],copy=np.asarray(p['copy'],bool)[keep]))
    return peers,np.asarray(case['observed'],float)[dims],np.asarray(case['covariance'],float)[np.ix_(dims,dims)]


def moments(x,w):
    mean=w@x;center=x-mean
    return mean,(center.T*w)@center


def summarize(peers,post):
    return dict(mean=[float(p['y']@w) for p,w in zip(peers,post)],
                copy=[float(p['copy']@w) for p,w in zip(peers,post)])


def exact_and_moment(peers,observed,cov):
    ix=np.array(list(itertools.product(*[range(len(p['w'])) for p in peers])),dtype=np.int32)
    total=sum(p['g'][ix[:,i]] for i,p in enumerate(peers))
    lp=sum(np.log(p['w'][ix[:,i]]) for i,p in enumerate(peers))
    residual=observed-total;scale=.6*cov;inv=np.linalg.inv(scale)
    ll=-.5*np.linalg.slogdet(scale)[1]-.5*(5+len(observed))*np.log1p(np.einsum('ni,ij,nj->n',residual,inv,residual)/5)
    independent=multivariate_t.logpdf(residual,shape=scale,df=5)
    density_error=float(np.max(abs((ll-ll[0])-(independent-independent[0]))))
    mass=np.exp(lp+ll-logsumexp(lp+ll))
    exact=[np.bincount(ix[:,i],weights=mass,minlength=len(p['w'])) for i,p in enumerate(peers)]
    prior=[moments(p['g'],p['w']) for p in peers];approx=[]
    for i,p in enumerate(peers):
        other=sum((m for j,(m,c) in enumerate(prior) if j!=i),start=np.zeros(len(observed)))
        vc=cov+sum((c for j,(m,c) in enumerate(prior) if j!=i),start=np.zeros_like(cov))
        lw=np.log(p['w'])+multivariate_t.logpdf(observed-other-p['g'],shape=.6*vc,df=5)
        approx.append(np.exp(lw-logsumexp(lw)))
    return summarize(peers,exact),summarize(peers,approx),density_error


def basis(matrix):
    """Exact numerical affine span; never add artificial variance."""
    _,sing,vt=np.linalg.svd(matrix,full_matrices=False)
    tol=np.finfo(float).eps*max(matrix.shape)*(sing[0] if len(sing) else 0.)
    b=vt[:np.sum(sing>tol)].T.copy()
    for k in range(b.shape[1]):
        if b[np.argmax(abs(b[:,k])),k]<0:b[:,k]*=-1
    return b


def inverse_pd(a):
    a=(a+a.T)/2
    np.linalg.cholesky(a)
    return np.linalg.solve(a,np.eye(len(a)))


def setup(peers,observed,cov):
    chol=np.linalg.cholesky(.6*cov)
    white=[np.linalg.solve(chol,p['g'].T).T for p in peers]
    anchors=[p['w']@g for p,g in zip(peers,white)]
    local=[basis(g-a) for g,a in zip(white,anchors)]
    aggregate=basis(np.concatenate([b.T for b in local],axis=0))
    y=np.linalg.solve(chol,observed)-sum(anchors)
    yp=aggregate.T@y;perp=y-aggregate@yp
    sizes=[b.shape[1] for b in local];ends=np.cumsum([0,*sizes]);slices=[slice(ends[i],ends[i+1]) for i in range(len(peers))]
    a=np.concatenate([aggregate.T@b for b in local],axis=1)
    z=[(g-anchor)@b for g,anchor,b in zip(white,anchors,local)]
    return dict(z=z,a=a,slices=slices,yp=yp,perp2=float(perp@perp),original_dim=len(observed))


def t_tilt(mean,cov,y,perp2,original_dim,nodes):
    """Gaussian cavity times EXACT t5 mixture, approximated only by quadrature."""
    x,w=roots_genlaguerre(nodes,1.5);lam=x/2.5
    precision=inverse_pd(cov);h=precision@mean;eye=np.eye(len(mean))
    mus=[];vs=[];lw=[]
    for rate,weight in zip(lam,w):
        v=inverse_pd(precision+rate*eye);mu=v@(h+rate*y)
        evidence=cov+eye/rate;res=y-mean
        loge=-.5*(len(mean)*np.log(2*np.pi)+np.linalg.slogdet(evidence)[1]+res@np.linalg.solve(evidence,res))
        loge+=.5*(original_dim-len(mean))*np.log(rate/(2*np.pi))-.5*rate*perp2
        lw.append(np.log(weight)-gammaln(2.5)+loge);mus.append(mu);vs.append(v)
    lw=np.asarray(lw);pw=np.exp(lw-logsumexp(lw));mus=np.asarray(mus)
    m=pw@mus;center=mus-m
    v=np.einsum('n,nij->ij',pw,np.asarray(vs))+(center.T*pw)@center
    return m,(v+v.T)/2


def quadrature_reference(mean,cov,y,perp2,original_dim):
    """Independent adaptive integral of Gaussian conditional first/raw moments."""
    r=len(mean);inv=inverse_pd(cov);h=inv@mean;eye=np.eye(r)
    # A finite reference density rescales the integral only; it cancels from
    # every normalized moment and is not a likelihood approximation.
    diff=y-mean;wide=cov+eye
    logscale=gammaln((5+original_dim)/2)-gammaln(2.5)-original_dim/2*np.log(5*np.pi)
    logscale-=.5*np.linalg.slogdet(wide)[1]+(5+original_dim)/2*np.log1p((diff@np.linalg.solve(wide,diff)+perp2)/5)
    def integrand(rate):
        if rate<=0:return np.zeros(1+r+r*r)
        v=np.linalg.solve(inv+rate*eye,eye);m=v@(h+rate*y)
        c=cov+eye/rate;diff=y-mean
        lg=2.5*np.log(2.5)-gammaln(2.5)+1.5*np.log(rate)-2.5*rate
        lg-=.5*(r*np.log(2*np.pi)+np.linalg.slogdet(c)[1]+diff@np.linalg.solve(c,diff))
        lg+=.5*(original_dim-r)*np.log(rate/(2*np.pi))-.5*rate*perp2
        return np.exp(lg-logscale)*np.r_[1.,m,(v+np.outer(m,m)).ravel()]
    value,error,info=quad_vec(integrand,0,np.inf,epsabs=1e-11,epsrel=1e-11,limit=200,full_output=True)
    assert info.success and value[0]>0,'adaptive quadrature reference failed'
    m=value[1:1+r]/value[0];v=value[1+r:].reshape(r,r)/value[0]-np.outer(m,m)
    return m,v,float(error)


def quadrature_audit(peers,observed,cov):
    s=setup(peers,observed,cov);a=s['a'];n=a.shape[1]
    if not n:return dict(deterministic=True,pass_all=True)
    m=np.zeros(n);v=np.zeros((n,n))
    for p,z,sl in zip(peers,s['z'],s['slices']):
        mm,vv=moments(z,p['w']);m[sl]=mm;v[sl,sl]=vv
    m=a@m;v=a@v@a.T
    expected,expected_v,error=quadrature_reference(m,v,s['yp'],s['perp2'],s['original_dim'])
    records=[]
    for nodes in DESIGN['quadrature_nodes']:
        actual,actual_v=t_tilt(m,v,s['yp'],s['perp2'],s['original_dim'],nodes)
        records.append(dict(nodes=nodes,mean_error=float(np.max(abs(actual-expected))),
            covariance_error=float(np.max(abs(actual_v-expected_v)))))
    passed=all(r['mean_error']<=1e-6 and r['covariance_error']<=1e-6 for r in records)
    return dict(rank=len(m),observed_dimension=len(observed),orthogonal_residual_squared=s['perp2'],
        reference_integration_error=error,records=records,pass_all=passed)


def run_ep(peers,observed,cov,order,damping,initial,nodes,budget):
    s=setup(peers,observed,cov);z,a,slices=s['z'],s['a'],s['slices'];n=a.shape[1];r=a.shape[0]
    if not n:return dict(checkpoints={str(k):summarize(peers,[p['w'] for p in peers]) for k in DESIGN['checkpoints']},last_mean_change=0.,last_copy_change=0.)
    ts=[];hs=[]
    for p,x in zip(peers,z):
        if not x.shape[1]:ts.append(np.zeros((0,0)));hs.append(np.zeros(0));continue
        y=p['y'];sd=np.sqrt(p['w']@((y-p['w']@y)**2));sign={'prior':0,'soft_lower':-2,'soft_upper':2}[initial]
        lw=np.log(p['w'])+(sign*(y-p['w']@y)/sd if sd else 0)
        w=np.exp(lw-logsumexp(lw));m,v=moments(x,w);t=inverse_pd(v);ts.append(t);hs.append(t@m)
    tt=np.zeros((r,r));ht=np.zeros(r)
    def global_gaussian(skip_peer=None,skip_t=False):
        q=np.zeros((n,n));h=np.zeros(n)
        for i,sl in enumerate(slices):
            if i!=skip_peer:q[sl,sl]+=ts[i];h[sl]+=hs[i]
        if not skip_t:q+=a.T@tt@a;h+=a.T@ht
        v=inverse_pd(q);return v@h,v
    def peer_tilt(i):
        if not z[i].shape[1]:return peers[i]['w'],None,None,None,None
        m,v=global_gaussian(skip_peer=i);sl=slices[i];m=m[sl];v=v[sl,sl];prec=inverse_pd(v)
        diff=z[i]-m;lw=np.log(peers[i]['w'])-.5*np.einsum('ni,ij,nj->n',diff,prec,diff)
        w=np.exp(lw-logsumexp(lw));mt,vt=moments(z[i],w)
        return w,mt,vt,m,v
    def output():return summarize(peers,[peer_tilt(i)[0] for i in range(len(peers))])
    visit=sorted(range(len(peers)),key=lambda i:peers[i]['id'],reverse=order=='descending_id')
    checkpoints={};previous=None;change_mean=change_copy=None
    for sweep in range(1,DESIGN['sweeps']+1):
        budget()
        m,v=global_gaussian(skip_t=True);m=a@m;v=a@v@a.T
        mt,vt=t_tilt(m,v,s['yp'],s['perp2'],s['original_dim'],nodes)
        pt,p=inverse_pd(vt),inverse_pd(v)
        tt=(1-damping)*tt+damping*(pt-p);ht=(1-damping)*ht+damping*(pt@mt-p@m)
        global_gaussian()
        for i in visit:
            if not z[i].shape[1]:continue
            _,mt,vt,m,v=peer_tilt(i);pt,p=inverse_pd(vt),inverse_pd(v)
            ts[i]=(1-damping)*ts[i]+damping*(pt-p);hs[i]=(1-damping)*hs[i]+damping*(pt@mt-p@m)
            global_gaussian()
        current=output()
        if previous is not None:
            change_mean=float(np.max(abs(np.asarray(current['mean'])-previous['mean'])))
            change_copy=float(np.max(abs(np.asarray(current['copy'])-previous['copy'])))
        previous=current
        if sweep in DESIGN['checkpoints']:checkpoints[str(sweep)]=current
    return dict(checkpoints=checkpoints,last_mean_change=change_mean,last_copy_change=change_copy)


def errors(actual,expected):
    return {k:abs(np.asarray(actual[k])-expected[k]).tolist() for k in ['mean','copy']}


def test():
    receipt=json.loads((OUT/'design.json').read_text());assert receipt['design']==DESIGN
    for path,h in receipt['sources'].items():assert sha(ROOT/path)==h,path
    assert not (OUT/'result.json').exists(),'Preserve numerical decision'
    started=time.monotonic();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    def budget():
        if time.monotonic()-started>=60:raise TimeoutError('fixed60second budget exhausted')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss>256*1024:raise MemoryError('fixed256MiB incremental workspace exceeded')
    records=[];baselines=[];passed=True;failure=None
    try:
        for case in receipt['cases']:
            for view,dims in [('joint',[0,1,2]),('departure',[2])]:
                budget();peers,obs,cov=finite_case(case,dims);exact,moment,density=exact_and_moment(peers,obs,cov)
                assert density<=DESIGN['thresholds']['density_log_error'],density
                baselines.append(dict(case=case['name'],view=view,peer_ids=[p['id'] for p in peers],exact=exact,
                    prior_moment=moment,prior_moment_absolute_errors=errors(moment,exact),density_log_error=density))
                try:
                    qa=quadrature_audit(peers,obs,cov)
                except (np.linalg.LinAlgError,FloatingPointError,ValueError,AssertionError) as e:
                    qa=dict(pass_all=False,error=f'{type(e).__name__}: {e}')
                baselines[-1]['quadrature_audit']=qa;passed &= qa['pass_all']
                for order,initial,damping,nodes in itertools.product(DESIGN['orders'],DESIGN['initializations'],DESIGN['damping'],DESIGN['quadrature_nodes']):
                    record=dict(case=case['name'],view=view,order=order,initial=initial,damping=damping,nodes=nodes)
                    try:
                        ep=run_ep(peers,obs,cov,order,damping,initial,nodes,budget)
                        record.update(status='completed',**ep)
                        record['absolute_errors']={k:errors(v,exact) for k,v in ep['checkpoints'].items()}
                        accurate=all(max(e['mean'])<=1 and max(e['copy'])<=.001 for e in record['absolute_errors'].values())
                        converged=ep['last_mean_change']<=.01 and ep['last_copy_change']<=.00001
                        record.update(accuracy_pass=accurate,convergence_pass=converged);passed &= accurate and converged
                    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as e:
                        record.update(status='invalid',error=f'{type(e).__name__}: {e}');passed=False
                    records.append(record)
                complete=[r for r in records if r['case']==case['name'] and r['view']==view and r['status']=='completed']
                outputs=[q for rr in complete for q in rr['checkpoints'].values()]
                if outputs:
                    spread={key:np.ptp(np.array([o[key] for o in outputs]),axis=0).tolist() for key in ['mean','copy']}
                    stable=max(spread['mean'])<=1 and max(spread['copy'])<=.001
                    baselines[-1].update(all_configuration_spread=spread,stability_pass=stable);passed &= stable
                else:passed=False
                print(case['name'],view,'complete',len(complete),'of24',flush=True)
    except (TimeoutError,MemoryError) as e:
        failure=f'{type(e).__name__}: {e}';passed=False
    result=dict(design_sha256=sha(OUT/'design.json'),source_sha256=sha(__file__),decision='pass_toy_evidence_only' if passed else 'reject_fixed_EP_toy',
        baselines=baselines,records=records,resource_failure=failure,seconds=time.monotonic()-started,
        peak_incremental_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss)/1024,
        no_real_cohort_inference=True,no_labels_or_scores=True,interpretation=DESIGN['interpretation'])
    write(OUT/'result.json',result)
    print(result['decision'],round(result['seconds'],3),'seconds',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','test'])
    with threadpool_limits(1),np.errstate(over='raise',invalid='raise',divide='raise'):
        globals()[parser.parse_args().stage]()
