"""Frozen Rome binary/multiclass calibration; see docs/ROME_CALIBRATION.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

import plan_window
import record_fixes
import rome_hybrid
import rome_signals as signals
import stand_day_run as row30

SETTINGS = dict(calibration_prior=20.0, minimum_calibration_rows=20,
                probability_floor=1e-8, inverse_temperature_bounds=[.25,4.],
                bias_bounds=[-8.,8.], day_prior_rows=20, day_gap_sec=50400,
                day_min_taxi_sec=60, day_max_taxi_sec=7200, day_seconds=86400)
EXTRA = row30.CLASSIFIERS['stand']
COLUMNS = dict(binary='pred_binary_cal', three='pred_three_cal',
               binary_uncal='pred_binary_uncal', binary_cal_hard='pred_binary_cal_hard',
               three_uncal='pred_three_uncal')


def outcomes(y, gap, classes=3):
    y, gap = np.asarray(y,float), np.asarray(gap,float)
    if y.shape != gap.shape or not np.isfinite(y).all():
        raise ValueError('Training outcomes need aligned finite labels')
    label=np.zeros(len(y),dtype=np.int32)
    if classes==3:
        label[(y-86400>=1)&(y-86400<=7200)]=2
    elif classes!=2:
        raise ValueError('Expected two or three classes')
    label[np.isfinite(gap)&(np.abs(y-gap)<=60)]=1
    return label


def probabilities(booster, X, classes, threads):
    p=np.asarray(booster.predict(X,num_threads=threads),float)
    if classes==2:
        p=np.column_stack([1-p,p])
    if p.shape!=(len(X),classes):
        raise ValueError('Invalid classifier probabilities')
    validate_probabilities(p)
    return p


def validate_probabilities(p):
    if (p.ndim!=2 or p.shape[1] not in (2,3) or not np.isfinite(p).all()
            or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-8,rtol=0)):
        raise ValueError('Expected finite, nonnegative probabilities summing to one')


def binary_schedule_scope(p, gap):
    """The saved parent assigns zero schedule probability when the clock is missing."""
    p=np.asarray(p,float).copy()
    validate_probabilities(p)
    if p.shape!=(len(gap),2):raise ValueError('Binary clock scope is misaligned')
    p[~np.isfinite(gap)]=[1.,0.]
    return p


def fit_classifier(X, y, gap, seed, classes, threads=6, rounds=300):
    label=outcomes(y,gap,classes)
    params=dict(rome_hybrid.CLASSIFIER_PARAMS,seed=seed,bagging_seed=seed+1,
                feature_fraction_seed=seed+2,num_threads=threads)
    if classes==3:
        params.update(objective='multiclass',metric='multi_logloss',num_class=3)
    model=signals._train(params,X,label,rounds,f'rome-{classes}class/{seed}')
    return model,dict(counts=np.bincount(label,minlength=classes).tolist(),
                      features=X.columns.tolist(),params=params,rounds=rounds)


def calibration_objective(theta, logp, labels):
    k=logp.shape[1]
    bias=np.r_[0.,theta[1:]]
    z=theta[0]*logp+bias
    q=softmax(z,axis=1)
    delta=q.copy();delta[np.arange(len(labels)),labels]-=1
    diff=theta-np.r_[1.,np.zeros(k-1)]
    lam=SETTINGS['calibration_prior']
    loss=float(np.sum(logsumexp(z,axis=1)-z[np.arange(len(labels)),labels])+.5*lam*(diff@diff))
    grad=np.r_[np.sum(delta*logp),np.sum(delta,axis=0)[1:]]+lam*diff
    return loss,grad


def fit_calibrator(p, labels, noid):
    p=np.asarray(p,float);labels=np.asarray(labels,int);noid=np.asarray(noid,bool)
    validate_probabilities(p)
    if labels.shape!=(len(p),) or noid.shape!=(len(p),):
        raise ValueError('Calibration rows are not aligned')
    k=p.shape[1]
    if ((labels<0)|(labels>=k)).any():
        raise ValueError('Invalid calibration class')
    logp=np.log(np.clip(p,SETTINGS['probability_floor'],1))
    groups={}
    for name,mask in [('id_present',~noid),('id_missing',noid)]:
        theta=np.r_[1.,np.zeros(k-1)]
        before=calibration_objective(theta,logp[mask],labels[mask])[0]
        status='identity: insufficient rows'
        if mask.sum()>=SETTINGS['minimum_calibration_rows']:
            fit=minimize(calibration_objective,theta,args=(logp[mask],labels[mask]),
                         method='L-BFGS-B',jac=True,
                         bounds=[SETTINGS['inverse_temperature_bounds']]+[SETTINGS['bias_bounds']]*(k-1),
                         options={'maxiter':250,'ftol':1e-12,'gtol':1e-7})
            if not fit.success or not np.isfinite(fit.fun) or fit.fun>before+1e-6:
                raise ValueError(f'Calibration optimizer failed: {fit.message}')
            theta=fit.x;status=str(fit.message)
        groups[name]=dict(theta=theta.tolist(),rows=int(mask.sum()),
                          class_counts=np.bincount(labels[mask],minlength=k).tolist(),
                          objective_before=before,
                          objective_after=calibration_objective(theta,logp[mask],labels[mask])[0],status=status)
    return dict(classes=k,settings=SETTINGS.copy(),groups=groups)


def calibrate(p, noid, fitted):
    p=np.asarray(p,float);noid=np.asarray(noid,bool)
    validate_probabilities(p)
    if p.shape!=(len(noid),fitted['classes']) or fitted['settings']!=SETTINGS:
        raise ValueError('Calibration shape/settings mismatch')
    out=np.empty_like(p)
    logp=np.log(np.clip(p,SETTINGS['probability_floor'],1))
    for name,mask in [('id_present',~noid),('id_missing',noid)]:
        t=np.asarray(fitted['groups'][name]['theta'])
        out[mask]=softmax(t[0]*logp[mask]+np.r_[0.,t[1:]],axis=1)
    return out


def day_eligible(raw, gap):
    return (raw.ADEP_mvt.eq('LIRF').to_numpy() & raw.FLIGHT_ID_mvt.isna().to_numpy()
            & (np.asarray(gap)>=SETTINGS['day_gap_sec']) & (signals.local_days(raw)==1))


def constrain_day(p, eligible, day_count):
    p=np.asarray(p,float)
    eligible=np.asarray(eligible,bool)
    validate_probabilities(p)
    if p.shape[1]!=3 or not np.isfinite(day_count) or day_count<0 or eligible.shape!=(len(p),):
        raise ValueError('Expected three class probabilities and nonnegative support')
    q=p.copy()
    day=np.where(eligible,q[:,2]*day_count/(day_count+SETTINGS['day_prior_rows']),0.)
    nond=np.maximum(q[:,:2],SETTINGS['probability_floor'])
    q[:,:2]=nond/nond.sum(axis=1,keepdims=True)*(1-day[:,None])
    q[:,2]=day
    return q


def mixture(ordinary, gap, p, raw, late_mean):
    """Observable fields only; target/block timestamps cannot affect prediction."""
    ordinary=np.asarray(ordinary,float);gap=np.asarray(gap,float);p=np.asarray(p,float)
    validate_probabilities(p)
    if p.shape[0]!=len(ordinary) or gap.shape!=ordinary.shape or len(raw)!=len(ordinary):
        raise ValueError('Mixture components are misaligned')
    other=ordinary.copy()
    late=signals.late_rows(gap,raw)
    if late_mean is not None and np.isfinite(late_mean):
        other[late]=np.minimum(late_mean,np.maximum(gap[late]-3600,60))
    sched=np.where(np.isfinite(gap),gap,other)
    components=np.column_stack([other,sched])
    if p.shape[1]==3:
        day=86400+np.clip(ordinary,SETTINGS['day_min_taxi_sec'],SETTINGS['day_max_taxi_sec'])
        components=np.column_stack([components,day])
    if p.shape!=components.shape or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid mixture weights')
    out=np.maximum(np.sum(p*components,axis=1),0)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite mixture')
    return out


def postprocess(control, rome_prediction, raw, rome):
    pred=np.asarray(control,float).copy()
    pred[rome]=rome_prediction
    pred=record_fixes.noid_cap(plan_window.apply(pred,raw),raw)
    if not np.array_equal(pred[~rome],np.asarray(control)[~rome]):
        raise ValueError('Non-Rome predictions changed')
    return pred
