"""Marginalize uncertain UTC off-block days in the joint source message.

Departure totals on different days are mutually exclusive contributions from
one flight. Their negative cross-covariance is retained. Ordinary normal moments
are integrated analytically across midnight; long priors remain discrete.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

import joint_public_moments as base

SETTINGS=dict(base.SETTINGS,
    change='integrate each support state into its own UTC off-block day',
    peers='all non-VFR flights in the held takeoff month contribute probability to interior UTC days',
    moments='analytic normal interval moments; exact discrete long-flight calendar moments',
    source_counts='expected membership count replaces hard day count in retained noise formula',
    common_error='ordinary mean perturbation includes probability flow across midnight',
    target_scope='old touched Rome rows only; every support day must be interior and source-supported',
    fallback='original joint increment when any candidate day is unsupported',
    calendar_noise='same retained per-day variance; no new fitted calibration',
    common_derivative_step_sec=.25,no_new_fit=True,no_iteration=True)


def interval_moments(mean,lo,hi,sigma=200.):
    """Unnormalized zeroth, first and second normal moments on (lo, hi]."""
    mean,lo,hi=np.broadcast_arrays(np.asarray(mean,float),np.asarray(lo,float),np.asarray(hi,float))
    valid=hi>lo;hi=np.maximum(hi,lo)
    a,b=(lo-mean)/sigma,(hi-mean)/sigma
    pa,pb=np.exp(-a*a/2)/np.sqrt(2*np.pi),np.exp(-b*b/2)/np.sqrt(2*np.pi)
    # Opposite-tail CDF form avoids cancellation when both bounds are positive.
    mass=np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a))
    first=mean*mass+sigma*(pa-pb)
    second=(mean*mean+sigma*sigma)*mass+2*mean*sigma*(pa-pb)+sigma*sigma*(a*pa-b*pb)
    return tuple(np.where(valid,x,0.) for x in [mass,first,second])


def day_numbers(takeoff,values):
    return np.floor((takeoff-np.asarray(values,float))/86400).astype(np.int64)


def moments(rows,days,parameters):
    mean=rows.prior.to_numpy(float);gap=rows.gap.to_numpy(float);epoch=rows.takeoff_epoch.to_numpy(float)
    n,k=len(rows),len(days);old_v,old_c,_=base.normal_moments(mean,gap)
    v=np.zeros((n,k+2));c=np.zeros((n,k+2,k+2));coverage=np.zeros((n,k));derivative=np.zeros((n,k))
    v[:,:2]=old_v[:,:2];c[:,:2,:2]=old_c[:,:2,:2]
    second=np.zeros((n,k));cross=np.zeros((n,2,k))
    for j,day in enumerate(days):
        lo=epoch-(day+1)*86400;hi=epoch-day*86400
        coverage[:,j]=interval_moments(mean,lo,hi)[0]
        a,b,d=interval_moments(mean,lo,np.minimum(hi,gap))
        v[:,j+2]=gap*a-b;second[:,j]=gap*gap*a-2*gap*b+d
        a,b,d=interval_moments(mean,np.maximum(lo,0),np.minimum(np.minimum(hi,gap),7200))
        cross[:,0,j]=gap*a-b;cross[:,1,j]=gap*b-d
        step=SETTINGS['common_derivative_step_sec'];values=[]
        for shift in [-step,step]:
            a,b,_=interval_moments(mean+shift,lo,np.minimum(hi,gap));values.append(gap*a-b)
        derivative[:,j]=(values[1]-values[0])/(2*step)
    c[:,2:,2:]=-v[:,2:,None]*v[:,None,2:]
    for j in range(k):c[:,j+2,j+2]+=second[:,j]
    c[:,:2,2:]=cross-v[:,:2,None]*v[:,None,2:]
    c[:,2:,:2]=c[:,:2,2:].transpose(0,2,1)
    lookup={int(d):i for i,d in enumerate(days)};supports={}
    for i in np.flatnonzero(gap>3600):
        values,weights,_=base.prior.support(gap[i],mean[i],False,np.nan,'LIRF',parameters)
        assigned=day_numbers(epoch[i],values);w=np.zeros((len(values),k+2))
        valid=(values>=0)&(values<=7200);w[:,0]=valid;w[:,1]=np.where(valid,values,0)
        coverage[i]=0
        for day in np.unique(assigned):
            if int(day) not in lookup:continue
            j=lookup[int(day)];at=assigned==day
            w[at,j+2]=np.maximum(gap[i]-values[at],0);coverage[i,j]=weights[at].sum()
        v[i]=weights@w;center=w-v[i];c[i]=(center.T*weights)@center
        supports[i]=(values,weights,w,assigned)
    derivative[gap>3600]=0
    assert np.isfinite(v).all() and np.isfinite(c).all() and np.isfinite(derivative).all()
    return v,c,coverage,derivative,supports


def apply(inputs,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters):
    names=['mvt_id','month','takeoff_epoch','gap','prior','eligible','quality','runway','local_day','assigned_day','fltrule','delay_scope']
    rows=inputs[names].copy().reset_index(drop=True);assert rows.month.nunique()==1
    result,_=base.apply(rows,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters)
    result['old_dep_only']=result.dep_only;result['old_joint']=result.joint;result['old_increment']=result.increment
    result['calendar_touched']=False;result['calendar_days']=0
    dates=pd.to_datetime(source_delay.day,utc=True)
    safe=(dates.dt.strftime('%Y-%m')==rows.month.iloc[0])&(dates.dt.day>1)&(dates.dt.day<dates.dt.days_in_month)
    ds=source_delay.loc[safe].copy();ds['number']=ds.day.to_numpy(dtype='datetime64[D]').astype(np.int64)
    ds=ds.set_index('number').sort_index();days=ds.index.to_numpy();lookup={int(d):i for i,d in enumerate(days)}
    v,c,coverage,derivative,supports=moments(rows,days,support_parameters)
    dep=~rows.fltrule.eq('V').to_numpy();taxi=(rows.eligible&rows.quality).to_numpy()
    count=coverage[dep].sum(0);dep_mean=v[dep,2:].sum(0);dep_cov=c[dep,2:,2:].sum(0)
    dep_load=derivative[dep].sum(0)
    supported=(count>=20)&np.isfinite(ds.total.to_numpy()+ds.n_public.to_numpy())
    dn=ds.n_public.to_numpy()-count
    dep_noise=3*(delay_parameters['sigma']**2+(delay_parameters['kappa']*dn)**2)
    ts=source_taxi.set_index(['local_day','runway']);groups={};pairs={};summaries=[]
    for i in np.flatnonzero(result.touched):
        values,weights,w,assigned=supports[i]
        possible=np.unique(assigned)
        if any(int(d) not in lookup or not supported[lookup[int(d)]] for d in possible):continue
        jd=np.array([lookup[int(d)] for d in possible]);dims=np.r_[0,1,jd+2]
        tk=(rows.local_day.iloc[i],rows.runway.iloc[i]);key=(tk,tuple(jd))
        if tk not in groups:
            tm=taxi&rows.local_day.eq(tk[0]).to_numpy()&rows.runway.eq(tk[1]).to_numpy()
            groups[tk]=tm
        tm=groups[tk]
        if key not in pairs:
            n=int(tm.sum());src=ts.loc[tk]
            bias=np.array([1.,n/100])@np.asarray(taxi_parameters['coefficients'])
            observed=np.r_[np.array([src.source_n,src.source_n*src.source_mean])+bias,
                ds.total.to_numpy()[jd]-delay_parameters['bias']]
            mean=np.r_[v[tm,:2].sum(0),dep_mean[jd]]
            cov=np.zeros((len(dims),len(dims)));cov[:2,:2]=c[tm,:2,:2].sum(0)
            cov[2:,2:]=dep_cov[np.ix_(jd,jd)]
            cross=c[tm&dep,:2,2:].sum(0)[:,jd];cov[:2,2:]=cross;cov[2:,:2]=cross.T
            cov[:2,:2]+=n*np.asarray(taxi_parameters['covariance_per_row'])+np.diag([1.,3600.])
            cov[2:,2:]+=np.diag(dep_noise[jd])
            loading=np.r_[0.,v[tm,0].sum(),dep_load[jd]];cov+=225*np.outer(loading,loading)
            pairs[key]=(observed,mean,cov)
            summaries.append(dict(local_day=tk[0],runway=tk[1],offblock_days=','.join(ds.day.iloc[jd]),
                taxi_rows=n,expected_departures=count[jd].tolist(),target_days=len(jd)))
        observed,mean,cov=pairs[key];cc=cov-c[i][np.ix_(dims,dims)];other=mean-v[i,dims]
        joint=base.posterior_mean(values,weights,w[:,dims],observed,other,cc)
        depart=base.posterior_mean(values,weights,w[:,jd+2],observed[2:],other[2:],cc[2:,2:])
        result.loc[i,['dep_only','joint','increment','calendar_touched','calendar_days']]=[depart,joint,joint-depart,True,len(jd)]
    assert np.isfinite(result[['dep_only','joint','increment']]).all().all()
    np.testing.assert_array_equal(result.loc[~result.calendar_touched,'increment'],result.loc[~result.calendar_touched,'old_increment'])
    return result,pd.DataFrame(summaries)


def checks():
    from scipy.integrate import quad
    from scipy.stats import norm
    worst=0.
    for mean,lo,hi in [(1000.,900.,1100.),(1000.,-20.,800.),(7190.,7000.,7200.),(1000.,1100.,1000.)]:
        actual=interval_moments(mean,lo,hi)
        expected=[quad(lambda y:y**p*norm.pdf(y,loc=mean,scale=200),lo,hi,epsabs=1e-7)[0] if hi>lo else 0. for p in range(3)]
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=1e-10)
        worst=max(worst,float(np.max(abs(np.asarray(actual)-expected))))
    midnight=pd.Timestamp('2025-07-03T00:00:00Z').timestamp()
    d=int(midnight//86400)
    np.testing.assert_array_equal(day_numbers(midnight+600,[599,600,601]),[d,d,d-1])
    rows=pd.DataFrame(dict(gap=[1400.,5000.],prior=[1000.,2000.],takeoff_epoch=[midnight+1000,midnight+1000]))
    prm=dict(tau=1000.,spread=[-100.,0.,100.]);days=np.array([d-1,d])
    v,c,coverage,derivative,support=moments(rows,days,prm)
    old_v,old_c,_=base.normal_moments([1000.],[1400.])
    reduce=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1]])
    np.testing.assert_allclose(reduce@v[0],old_v[0],atol=1e-8,rtol=1e-12)
    np.testing.assert_allclose(reduce@c[0]@reduce.T,old_c[0],atol=1e-7,rtol=1e-10)
    np.testing.assert_allclose(coverage.sum(1),1.,atol=1e-12)
    assert c[0,2,3]<0 and all(np.linalg.eigvalsh(x).min()>-1e-6 for x in c)
    values,weights,w,assigned=support[1]
    np.testing.assert_allclose(w[:,2:].sum(1),np.maximum(5000-values,0),atol=0,rtol=0)
    assert len(np.unique(assigned))==2
    np.testing.assert_allclose(derivative[0].sum(),-ndtr((1400-1000)/200),atol=2e-7,rtol=0)
    functions=[lambda y:float(0<=y<=7200),lambda y:y if 0<=y<=7200 else 0.,
        lambda y:max(1400-y,0) if y>1000 else 0.,lambda y:max(1400-y,0) if y<=1000 else 0.]
    breaks=[-1400.,0.,1000.,1400.,3400.,7200.]
    def integral(fn):
        return sum(quad(lambda y:fn(y)*norm.pdf(y,loc=1000,scale=200),a,b,epsabs=1e-7)[0]
            for a,b in zip(breaks[:-1],breaks[1:]))
    independent=np.array([integral(fn) for fn in functions])
    independent_cov=np.array([[integral(lambda y:f(y)*g(y)) for g in functions] for f in functions])-np.outer(independent,independent)
    np.testing.assert_allclose(v[0],independent,atol=1e-8,rtol=1e-10)
    np.testing.assert_allclose(c[0],independent_cov,atol=2e-7,rtol=1e-10)
    # Enough ordinary support on either side of midnight for a two-day message.
    n=82;sample=pd.DataFrame(dict(mvt_id=np.arange(n),month='2025-07',takeoff_epoch=midnight+1000,
        gap=np.r_[np.full(80,1500.),9000.,10000.],prior=np.r_[np.full(40,900.),np.full(40,1100.),3500.,4500.],
        eligible=True,quality=True,runway='09',local_day='2025-07-03',
        assigned_day=['2025-07-03']*40+['2025-07-02']*42,fltrule='I',delay_scope=True))
    ts=pd.DataFrame(dict(local_day=['2025-07-03'],runway=['09'],source_n=[82.],source_mean=[1300.]))
    ds=pd.DataFrame(dict(day=['2025-07-02','2025-07-03'],total=[30000.,20000.],n_public=[42.,40.]))
    tp=dict(coefficients=[[0.,0.],[0.,0.]],covariance_per_row=[[.1,0.],[0.,10000.]])
    dp=dict(bias=0.,sigma=1000.,kappa=100.)
    actual,_=apply(sample,ts,ds,tp,dp,prm)
    assert actual.calendar_touched.sum()==2 and actual.calendar_days.max()==2
    shuffled=sample.sample(frac=1,random_state=81).assign(y=1e12,true_day='1900-01-01')
    other,_=apply(shuffled,ts,ds,tp,dp,prm)
    a=actual.set_index('mvt_id').sort_index();b=other.set_index('mvt_id').sort_index()
    np.testing.assert_array_equal(a.calendar_touched,b.calendar_touched)
    np.testing.assert_allclose(a.increment,b.increment,atol=1e-8,rtol=0)
    fallback,_=apply(sample,ts,ds.iloc[:1],tp,dp,prm)
    assert not fallback.calendar_touched.any()
    np.testing.assert_array_equal(fallback.increment,fallback.old_increment)
    return dict(interval_quadrature_max_error=worst,midnight_inclusive=True,
        normal_split_recombines_original_moments=True,exclusive_day_negative_covariance=True,
        discrete_support_contribution_conserved=True,common_derivative_sum_checked=True,
        independent_calendar_covariance_quadrature=True,two_day_message_and_unsupported_fallback=True,
        row_permutation_and_label_poison_invariant=True)
