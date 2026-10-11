"""Reuse the joint moment update with each non-Rome airport's native prior.

Only the support's airport argument changes; the retained calibrated source
likelihood, cohorts, three-moment arithmetic and fixed gate are unchanged.
The Rome copy-mode component is never enabled for another airport.
"""
import types

import numpy as np
import pandas as pd

import joint_public_moments as base

SETTINGS=dict(base.SETTINGS,
    scope='all nine non-Rome airports; same eligibility, quality and long-gap/prior gate',
    prior='each airport retains its original training quantiles; Rome copy component is not transferred',
    parent='v78 unchanged on Rome; fixed half incremental correction on other airports',
    calibration='same count/sum calibration function per native airport and fold; own timezone retained',
    no_new_gate=True,no_airport_selection=True)


def apply(airport,inputs,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters):
    def support(d,p,nm,nm_gap,ignored_airport,parameters):
        return base.prior.support(d,p,nm,nm_gap,airport,parameters)
    namespace=dict(base.apply.__globals__,prior=types.SimpleNamespace(support=support))
    function=types.FunctionType(base.apply.__code__,namespace,base.apply.__name__,base.apply.__defaults__,base.apply.__closure__)
    assert function.__code__ is base.apply.__code__
    return function(inputs,source_taxi,source_delay,taxi_parameters,delay_parameters,support_parameters)


def calibrate(airport,rows,source,allowed):
    import taxi_count_sum_jan_jul as original
    code=original.calibrate.__code__
    assert sum(isinstance(c,str) and c=='LIRF' for c in code.co_consts)==1
    constants=tuple(airport if isinstance(c,str) and c=='LIRF' else c for c in code.co_consts)
    new=code.replace(co_consts=constants)
    assert new.co_code==code.co_code and new.co_names==code.co_names
    function=types.FunctionType(new,original.calibrate.__globals__,original.calibrate.__name__,original.calibrate.__defaults__,original.calibrate.__closure__)
    parameters,cells=function(rows,source,allowed)
    assert cells.airport.eq(airport).all()
    return parameters,cells


def checks():
    rows=pd.DataFrame(dict(mvt_id=np.arange(22),gap=np.r_[np.full(20,1500.),9000.,10000.],
        prior=np.r_[np.full(20,1000.),3500.,4500.],eligible=True,quality=True,runway='09',
        local_day='2025-06-02',assigned_day='2025-06-02',fltrule='I',delay_scope=True))
    taxi=pd.DataFrame(dict(local_day=['2025-06-02'],runway=['09'],source_n=[22.],source_mean=[1300.]))
    delay=pd.DataFrame(dict(day=['2025-06-02'],total=[20000.],n_public=[22.]))
    tp=dict(coefficients=[[0.,0.],[0.,0.]],covariance_per_row=[[.1,0.],[0.,10000.]])
    dp=dict(bias=0.,sigma=1000.,kappa=100.)
    sp=dict(tau=1000.,spread=[-100.,0.,100.])
    old,_=base.apply(rows,taxi,delay,tp,dp,sp)
    rome,_=apply('LIRF',rows,taxi,delay,tp,dp,sp)
    pd.testing.assert_frame_equal(rome,old,check_exact=True)
    other,_=apply('EGLL',rows,taxi,delay,tp,dp,sp)
    assert int(other.touched.sum())==2
    for i in [20,21]:
        values,_,_=base.prior.support(rows.gap.iloc[i],rows.prior.iloc[i],False,np.nan,'EGLL',sp)
        assert values.min()>=rows.prior.iloc[i]-100 and values.max()<=rows.prior.iloc[i]+100
        assert values.min()<=other.joint.iloc[i]<=values.max()
        assert values.min()<=other.dep_only.iloc[i]<=values.max()
    np.testing.assert_array_equal(other.loc[~other.touched,'increment'],0)
    return dict(unchanged_Rome_reference_exact=True,native_non_Rome_support_bounds=True,
        Rome_copy_component_not_transferred=True,ordinary_rows_unchanged=True)
