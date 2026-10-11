"""Fixed joint taxi-count/taxi-sum/departure-delay moment messages.

Distinct local takeoff/runway and UTC off-block groups retain their own rows.
Shared rows contribute covariance to both sources. No labels or fitted taxi
model enter inference. This is an approximate message with Gaussian peer
moments, not exact joint discrete inference across all flights.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr,logsumexp

import taxi_count_sum_posterior as taxi
import daily_delay_posterior as prior

SETTINGS=dict(ordinary_sigma=200.,common_day_sigma=15.,cutoff=7200.,df=5,
    prior='retained v70 means, original training quantiles/copy/broad support; no public-input networks',
    likelihood='variance-matched multivariate t5 with peer moments and shared-flight covariance',
    measurement='retained taxi2 covariance plus retained departure t3 scale converted to variance',
    calendars='own local takeoff runway cell; own once-estimated UTC off-block day',
    comparison='joint posterior mean minus matched departure-only moment posterior mean',
    transfer='one fixed half incremental correction through current v77 constraints',
    common_error='one shared15-second mean perturbation across the selected two cohorts',
    scope='existing ambiguous Rome gap gate, original monthly quality gate, supported sources',
    no_iteration=True,no_parameter_search=True,no_new_fit=True)


def bounded_moments(mean,upper,sigma):
    upper=np.maximum(np.asarray(upper,float),0.)
    a,b=-mean/sigma,(upper-mean)/sigma
    pa,pb=np.exp(-a*a/2)/np.sqrt(2*np.pi),np.exp(-b*b/2)/np.sqrt(2*np.pi)
    mass=ndtr(b)-ndtr(a)
    first=mean*mass+sigma*(pa-pb)
    second=(mean*mean+sigma*sigma)*mass+2*mean*sigma*(pa-pb)+sigma*sigma*(a*pa-b*pb)
    return mass,first,second


def normal_moments(mean,gap):
    mean,gap=np.asarray(mean,float),np.asarray(gap,float)
    sigma=SETTINGS['ordinary_sigma']
    n,s,s2=bounded_moments(mean,np.full(len(mean),7200.),sigma)
    z=(gap-mean)/sigma;phi=np.exp(-z*z/2)/np.sqrt(2*np.pi);prob=ndtr(z)
    d=(gap-mean)*prob+sigma*phi
    d2=((gap-mean)**2+sigma*sigma)*prob+(gap-mean)*sigma*phi
    mass,first,second=bounded_moments(mean,np.minimum(7200.,gap),sigma)
    cross_n=gap*mass-first;cross_s=gap*first-second
    v=np.column_stack([n,s,d]);c=np.zeros((len(mean),3,3))
    c[:,0,0]=n*(1-n);c[:,1,1]=np.maximum(s2-s*s,0);c[:,2,2]=np.maximum(d2-d*d,0)
    c[:,0,1]=c[:,1,0]=s*(1-n)
    c[:,0,2]=c[:,2,0]=cross_n-n*d
    c[:,1,2]=c[:,2,1]=cross_s-s*d
    return v,c,prob


def contributions(values,gap):
    valid=(values>=0)&(values<=7200)
    return np.column_stack([valid,np.where(valid,values,0.),np.maximum(gap-values,0.)]).astype(float)


def loglike(residual,covariance):
    residual=np.asarray(residual,float);covariance=np.atleast_2d(covariance)
    scale=.6*covariance
    sign,det=np.linalg.slogdet(scale);assert sign>0 and np.isfinite(det)
    mahal=np.einsum('ij,ij->i',residual,np.linalg.solve(scale,residual.T).T)
    return -.5*det-.5*(5+residual.shape[1])*np.log1p(np.maximum(mahal,0)/5)


def posterior_mean(values,weights,vectors,observed,other_mean,covariance):
    lw=np.log(np.maximum(weights,1e-300))+loglike(observed[None,:]-other_mean[None,:]-vectors,covariance)
    post=np.exp(lw-logsumexp(lw));return float(values@post)


def apply(inputs,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters):
    names=['mvt_id','gap','prior','eligible','quality','runway','local_day','assigned_day','fltrule','delay_scope']
    rows=inputs[names].copy().reset_index(drop=True)
    assert rows.mvt_id.is_unique and np.isfinite(rows[['gap','prior']]).all().all()
    p,d=rows.prior.to_numpy(float),rows.gap.to_numpy(float)
    vectors,covs,prob=normal_moments(p,d)
    supports={}
    for i in np.flatnonzero(d>3600):
        values,weights,_=prior.support(d[i],p[i],False,np.nan,'LIRF',support_parameters)
        value=contributions(values,d[i]);mean=weights@value;center=value-mean
        covariance=(center.T*weights)@center
        vectors[i],covs[i]=mean,covariance
        supports[i]=(values,weights,value)
    taxi_ok=rows.eligible.to_numpy(bool)&rows.quality.to_numpy(bool)
    dep_ok=~rows.fltrule.eq('V').to_numpy()&rows.delay_scope.to_numpy(bool)
    taxi_groups=rows.loc[taxi_ok].groupby(['local_day','runway']).indices
    # GroupBy.indices on a filtered frame is positional within that frame.
    taxi_positions=np.flatnonzero(taxi_ok);dep_positions=np.flatnonzero(dep_ok)
    taxi_groups={key:taxi_positions[ix] for key,ix in taxi_groups.items()}
    dep_groups={key:dep_positions[ix] for key,ix in rows.loc[dep_ok].groupby('assigned_day').indices.items()}
    taxi_source=source_taxi.set_index(['local_day','runway']);dep_source=source_delay.set_index('day')
    taxi_info={};dep_info={}
    for key,ix in taxi_groups.items():
        if key not in taxi_source.index:continue
        src=taxi_source.loc[key];mean=vectors[ix,:2].sum(0);n=mean[0]
        if not (len(ix)>=20 and src.source_n>=20 and n>=20 and .5<=src.source_n/n<=1.2):continue
        bias=np.array([1.,len(ix)/100.])@np.asarray(taxi_parameters['coefficients'])
        noise=len(ix)*np.asarray(taxi_parameters['covariance_per_row'])+np.diag([1.,60.**2])
        taxi_info[key]=dict(ix=ix,mean=mean,cov=covs[ix,:2,:2].sum(0),noise=noise,
            observed=np.array([src.source_n,src.source_n*src.source_mean])+bias,n=n)
    for key,ix in dep_groups.items():
        if key not in dep_source.index:continue
        src=dep_source.loc[key]
        if len(ix)<20 or not np.isfinite(src.total+src.n_public):continue
        dn=float(src.n_public-len(ix))
        # The retained likelihood is Student t3 with scale sigma; its finite
        # variance is three times scale squared. Both new comparisons use it.
        noise=3*(delay_parameters['sigma']**2+(delay_parameters['kappa']*dn)**2)
        loading=float(np.sum(prob[ix]*(d[ix]<=3600)))
        dep_info[key]=dict(ix=ix,mean=float(vectors[ix,2].sum()),cov=float(covs[ix,2,2].sum()),
            noise=noise,observed=float(src.total-delay_parameters['bias']),loading=loading)
    tau=support_parameters['tau']
    active=(taxi_ok&dep_ok&(d>3600)&(d<50400)&(p>tau)&(p<d))
    result=pd.DataFrame(dict(mvt_id=rows.mvt_id,prior=p,dep_only=p,joint=p,increment=0.,touched=False))
    summaries=[];pairs={}
    for i in np.flatnonzero(active):
        tkey=(rows.local_day.iloc[i],rows.runway.iloc[i]);dkey=rows.assigned_day.iloc[i]
        if tkey not in taxi_info or dkey not in dep_info:continue
        key=(tkey,dkey)
        if key not in pairs:
            t,a=taxi_info[tkey],dep_info[dkey]
            shared=np.intersect1d(t['ix'],a['ix'])
            covariance=np.zeros((3,3));covariance[:2,:2]=t['cov'];covariance[2,2]=a['cov']
            covariance[:2,2]=covariance[2,:2]=covs[shared,:2,2].sum(0)
            noise=np.zeros((3,3));noise[:2,:2]=t['noise'];noise[2,2]=a['noise']
            loading=np.array([0.,t['n'],-a['loading']])
            noise+=15.**2*np.outer(loading,loading)
            pairs[key]=dict(mean=np.r_[t['mean'],a['mean']],covariance=covariance+noise,
                observed=np.r_[t['observed'],a['observed']])
            summaries.append(dict(local_day=tkey[0],runway=tkey[1],assigned_day=dkey,
                taxi_rows=len(t['ix']),delay_rows=len(a['ix']),shared_rows=len(shared),
                taxi_count_mean=t['mean'][0],taxi_sum_mean=t['mean'][1],delay_mean=a['mean'],
                covariance_eigen_min=float(np.linalg.eigvalsh(covariance+noise).min())))
        group=pairs[key];cov=group['covariance']-covs[i];other=group['mean']-vectors[i]
        values,weights,value=supports[i]
        joint=posterior_mean(values,weights,value,group['observed'],other,cov)
        dep=posterior_mean(values,weights,value[:,2:],group['observed'][2:],other[2:],cov[2:,2:])
        result.loc[i,['dep_only','joint','increment','touched']]=[dep,joint,joint-dep,True]
    assert np.isfinite(result[['prior','dep_only','joint','increment']]).all().all()
    assert not result.loc[~result.touched,'increment'].any()
    return result,pd.DataFrame(summaries)


def checks():
    from scipy.integrate import quad
    from scipy.stats import norm,multivariate_t
    worst=0.
    for mean,gap in [(0.,-100.),(1000.,1400.),(7190.,8000.),(9000.,9500.)]:
        v,c,_=normal_moments([mean],[gap])
        functions=[lambda y:float(0<=y<=7200),lambda y:y if 0<=y<=7200 else 0.,lambda y:max(gap-y,0)]
        points=sorted(set([mean-12*200,0.,7200.,gap,mean+12*200]))
        def integral(fn):
            return sum(quad(lambda y:fn(y)*norm.pdf(y,loc=mean,scale=200),a,b,epsabs=1e-6)[0]
                for a,b in zip(points[:-1],points[1:]))
        expected=np.array([integral(f) for f in functions])
        cv=np.array([[integral(lambda y:f(y)*g(y)) for g in functions] for f in functions])-np.outer(expected,expected)
        np.testing.assert_allclose(v[0],expected,rtol=1e-8,atol=1e-6)
        np.testing.assert_allclose(c[0],cv,rtol=1e-8,atol=1e-5)
        worst=max(worst,float(np.max(abs(c[0]-cv))))
    cov=np.array([[4.,1000.,-100.],[1000.,2e6,-1e6],[-100.,-1e6,3e6]])
    residual=np.array([[1.,300.,-500.],[-2.,-1000.,1000.]])
    for dims in [[2],[0,1,2]]:
        cc=cov[np.ix_(dims,dims)];rr=residual[:,dims]
        expected=multivariate_t.logpdf(rr,shape=.6*cc,df=5)
        actual=loglike(rr,cc)
        np.testing.assert_allclose(actual-actual[0],expected-expected[0],atol=1e-12,rtol=0)
    return dict(independent_three_moment_quadrature=True,max_covariance_error=worst,
        independent_one_and_three_dimensional_student_density=True)
