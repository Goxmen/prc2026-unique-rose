"""Independent arithmetic checks for daily-delay conditioning."""
import itertools
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import daily_delay_posterior as model
from records import write_json

def run():
    base=model.base
    for d,p,s in [(1000,800,200),(1000,1200,200),(1000,800,50),(0,200,200),(10000,800,200)]:
        c,v,k,_=base.moments(np.array([d]),np.array([p]),np.array([s]));z=(d-p)/s
        ec=quad(lambda t:max(0,d-p-s*t)*norm.pdf(t),-12,min(z,12))[0]
        vc=quad(lambda t:max(0,d-p-s*t)**2*norm.pdf(t),-12,min(z,12))[0]-ec*ec
        cov=quad(lambda t:s*t*max(0,d-p-s*t)*norm.pdf(t),-12,min(z,12))[0]
        np.testing.assert_allclose([c[0],v[0],k[0]],[ec,vc,cov],rtol=1e-8,atol=1e-7)
    prm=dict(spread=[-200.,0.,200.],tau=1000.,sigma=1e12,kappa=600.,bias=0.)
    d=np.array([800.,1000.,5000.,8000.]);p=np.array([600.,1200.,1500.,6000.]);nm=np.zeros(4,bool);ng=np.full(4,np.nan)
    q,_=model.condition_day(d,p,nm,ng,'LIRF',prm,2000.,0)
    np.testing.assert_allclose(p,q,atol=1e-7,rtol=1e-9)
    perm=np.array([2,0,3,1]);prm['sigma']=60.
    q,_=model.condition_day(d,p,nm,ng,'LIRF',prm,6000.,0)
    r,_=model.condition_day(d[perm],p[perm],nm[perm],ng[perm],'LIRF',prm,6000.,0)
    np.testing.assert_allclose(r,q[perm],atol=1e-8,rtol=1e-9)
    for pp in [0,50,1000,3000,7000,12000,90000]:
        ys,w,c=model.support(8000.,pp,False,np.nan,'LIRF',prm)
        np.testing.assert_allclose(ys@w,pp,atol=1e-5)
    # Enumerate every state for two ambiguous flights; this is independent of
    # the production forward/backward convolution recursion.
    sup=[model.support(d[i],p[i],False,np.nan,'LIRF',prm) for i in [2,3]]
    var=model.DESIGN['ordinary_sigma']**2+model.DESIGN['common_day_sigma']**2
    offsets=np.linspace(-6,6,41)*np.sqrt(var)
    states=[];weights=[]
    for (i,j),offset in itertools.product(itertools.product(range(len(sup[0][0])),range(len(sup[1][0]))),offsets):
        contribution=200+(sup[0][2][i]+sup[1][2][j])*60+offset
        likelihood=(1+((6000-contribution)/60)**2/3)**-2
        weight=sup[0][1][i]*sup[1][1][j]*np.exp(-.5*offset**2/var)*likelihood
        weights.append(weight);states.append([sup[0][0][i],sup[1][0][j],offset])
    w=np.array(weights);means=w@np.array(states)/w.sum()
    expected=p.copy();expected[2:]=means[:2];expected[0]-=means[2]
    expected[1]-=.25*model.DESIGN['common_day_sigma']**2/var*means[2]
    np.testing.assert_allclose(q,expected,atol=1e-8,rtol=1e-10)
    report=dict(passed=True,gaussian_quadrature_cases=5,prior_mean_cases=7,
        no_information_identity=True,permutation_invariance=True,enumerated_states=len(states),
        discrete_posterior_max_error=float(np.max(abs(q-expected))),source=base.artifact(model.__file__),
        helper=base.artifact(base.__file__),audit_source=base.artifact(__file__))
    write_json(model.OUT/'numerical_audit.json',report);print(report)

if __name__=='__main__':run()
