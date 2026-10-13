"""Frozen-message target support tilts for an honest cross-fitted prior family.

No labels enter message construction. The learned posterior estimator is
D(0) + (J(theta)-D(theta))/2, with both likelihoods and D(0) frozen. This is a
target-only discriminative estimator, not a joint posterior reoptimization.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

import posterior_aware_sensitivity as gate

SCALE = 3600.
RIDGE = .001
OPTIMIZER = dict(maxiter=200, ftol=1e-12, gtol=1e-8)
INPUT_COLUMNS = gate.INPUT_COLUMNS


def messages(inputs, source_taxi, source_delay, parameters):
    # This whitelist deliberately excludes every target/recorded block column.
    rows = inputs[INPUT_COLUMNS].copy().reset_index(drop=True)
    p,d,v,c,supports,ti,di,pairs,targets = gate.moments_and_groups(
        rows, source_taxi, source_delay, parameters)
    n = len(targets)
    width = max((len(supports[i][0]) for i,k in targets), default=1)
    out = {name: np.zeros((n,width), float) for name in ['values','statistic','joint_like','dep_like']}
    out['log_weight'] = np.full((n,width), -np.inf)
    out['features'] = np.empty((n,4), float)
    out['scale'] = np.empty(n, float)
    out['positions'] = np.array([i for i,k in targets], np.int64)
    out['mvt_id'] = rows.mvt_id.to_numpy()[out['positions']]
    out['prior'] = p[out['positions']]
    tau = parameters['support']['tau']
    for at,(i,key) in enumerate(targets):
        values,weights,value = supports[i]
        group = pairs[key]
        covariance = group['covariance']-c[i]
        other = group['mean']-v[i]
        residual = group['observed'][None,:]-other[None,:]-value
        out['joint_like'][at,:len(values)] = gate.original.loglike(residual,covariance)
        out['dep_like'][at,:len(values)] = gate.original.loglike(residual[:,2:],covariance[2:,2:])
        out['values'][at,:len(values)] = values
        out['log_weight'][at,:len(values)] = np.log(np.maximum(weights,1e-300))
        scale = max(d[i]-tau,60.)
        out['scale'][at] = scale
        out['statistic'][at,:len(values)] = (values-p[i])/scale
        z = np.clip((p[i]-tau)/(d[i]-tau),1e-6,1-1e-6)
        a,b = np.log(d[i]/3600.),np.log(z)-np.log1p(-z)
        out['features'][at] = [1.,a,b,a*b]
    # Zero means are retained once; the baseline never becomes a gradient path.
    out['dep_zero'] = moment(out,np.zeros(4),'dep')[0]
    out['joint_zero'] = moment(out,np.zeros(4),'joint')[0]
    return out


def moment(packet, theta, variant):
    assert variant in ['prior','joint','dep']
    eta = packet['features'] @ np.asarray(theta,float)
    lw = packet['log_weight'] + eta[:,None]*packet['statistic']
    if variant != 'prior':
        lw = lw+packet[variant+'_like']
    weights = np.exp(lw-logsumexp(lw,axis=1)[:,None])
    values = packet['values']
    mean = np.sum(weights*values,axis=1)
    variance = np.sum(weights*(values-mean[:,None])**2,axis=1)
    gradient = (variance/packet['scale'])[:,None]*packet['features']
    assert np.isfinite(mean).all() and np.isfinite(gradient).all()
    return mean,gradient,variance


def estimate(packet, theta, objective):
    if objective == 'prior':
        mean,gradient,_ = moment(packet,theta,'prior')
        return mean,gradient
    assert objective == 'posterior'
    joint,jg,_ = moment(packet,theta,'joint')
    dep,dg,_ = moment(packet,theta,'dep')
    return packet['dep_zero']+.5*(joint-dep),.5*(jg-dg)


def objective(theta, packet, y, kind):
    prediction,gradient = estimate(packet,theta,kind)
    residual = (prediction-y)/SCALE
    loss = float(np.mean(residual**2)+RIDGE*np.sum(theta**2))
    jac = 2*np.mean(residual[:,None]*gradient/SCALE,axis=0)+2*RIDGE*theta
    return loss,jac


def optimize(packet, y, kind):
    y = np.asarray(y,float)
    assert y.shape == packet['prior'].shape and len(y)>0 and np.isfinite(y).all()
    before,gradient = objective(np.zeros(4),packet,y,kind)
    result = minimize(objective,np.zeros(4),args=(packet,y,kind),jac=True,
                      method='L-BFGS-B',options=OPTIMIZER)
    receipt = dict(objective=kind,rows=len(y),success=bool(result.success),message=str(result.message),
        iterations=int(result.nit),evaluations=int(result.nfev),initial_loss=before,
        final_loss=float(result.fun),theta=result.x.tolist(),gradient=result.jac.tolist(),
        initial_gradient=gradient.tolist(),settings=OPTIMIZER,ridge=RIDGE,scale=SCALE)
    # Caller writes this receipt even on failure, then stops without retries.
    return receipt


def concatenate(packets):
    assert len(packets)>0
    width = max(p['values'].shape[1] for p in packets)
    result = {}
    for key in packets[0]:
        arrays = []
        for p in packets:
            a = p[key]
            if a.ndim == 2 and key != 'features' and a.shape[1]<width:
                fill = -np.inf if key=='log_weight' else 0.
                a = np.pad(a,((0,0),(0,width-a.shape[1])),constant_values=fill)
            arrays.append(a)
        result[key] = np.concatenate(arrays)
    assert len(np.unique(result['mvt_id']))==len(result['mvt_id'])
    return result


def predictions(rows, packet, theta):
    result = rows[['mvt_id','month','prior']].copy().reset_index(drop=True)
    result['dep_only'] = result.prior
    result['joint'] = result.prior
    result['increment'] = 0.
    result['touched'] = False
    ix = packet['positions']
    np.testing.assert_array_equal(result.mvt_id.iloc[ix],packet['mvt_id'])
    joint = moment(packet,theta,'joint')[0]
    dep = moment(packet,theta,'dep')[0]
    result.loc[ix,['dep_only','joint','increment','touched']] = pd.DataFrame(
        dict(dep_only=dep,joint=joint,increment=joint-dep,touched=True),index=ix)
    return result


def diagnostics(packet):
    zero = np.zeros(4)
    _,jg,jv = moment(packet,zero,'joint')
    _,dg,dv = moment(packet,zero,'dep')
    gradient = .5*(jg-dg)
    epsilon = 1e-5
    worst = 0.
    # Frozen log likelihoods, support and D0 for both signs of every difference.
    for k in range(4):
        shift = np.eye(4)[k]*epsilon
        fd = (estimate(packet,shift,'posterior')[0]-estimate(packet,-shift,'posterior')[0])/(2*epsilon)
        np.testing.assert_allclose(fd,gradient[:,k],atol=2e-5,rtol=2e-5)
        worst = max(worst,float(np.max(abs(fd-gradient[:,k]),initial=0.)))
    singular = np.linalg.svd(gradient,compute_uv=False)
    ratio = abs(jv-dv)/np.maximum(jv+dv,1e-30)
    q = [0.,.1,.5,.9,1.]
    return dict(rows=len(jv),incremental_derivative_rank=int(np.sum(singular>singular[0]*1e-10)) if len(singular) else 0,
        singular_values=singular.tolist(),cancellation_abs_difference_over_sum=np.quantile(ratio,q).tolist() if len(jv) else [],
        per_row_gradient_norm_quantiles=np.quantile(np.linalg.norm(gradient,axis=1),q).tolist() if len(jv) else [],
        signed_variance_difference_quantiles=np.quantile(jv-dv,q).tolist() if len(jv) else [],
        finite_difference_max_abs=worst,nonzero_rows=int(np.sum(np.linalg.norm(gradient,axis=1)>1e-8)),
        target_only_fixed_messages=True,no_predictive_gain_claim=True)


def checks():
    # Different source likelihoods make the variance-difference derivative
    # nontrivial; a shared likelihood must cancel identically in this estimator.
    v = np.array([[300.,1100.,6500.],[500.,1400.,9000.]])
    w = np.array([[.4,.4,.2],[.25,.5,.25]])
    p = dict(values=v,log_weight=np.log(w),features=np.array([[1.,.2,-.8,-.16],[1.,.8,.2,.16]]),
        scale=np.array([5000.,8000.]),prior=np.sum(v*w,axis=1),
        joint_like=np.array([[0.,-.4,-2.],[-1.,0.,-.2]]),dep_like=np.array([[-.2,0.,-.4],[0.,-.7,-.1]]))
    p['statistic']=(v-p['prior'][:,None])/p['scale'][:,None]
    p['dep_zero']=moment(p,np.zeros(4),'dep')[0]
    result=diagnostics(p)
    y=np.array([700.,5000.]);theta=np.array([.2,-.1,.1,.05])
    for kind in ['prior','posterior']:
        value,jac=objective(theta,p,y,kind)
        fd=np.array([(objective(theta+np.eye(4)[k]*1e-5,p,y,kind)[0]-
                      objective(theta-np.eye(4)[k]*1e-5,p,y,kind)[0])/2e-5 for k in range(4)])
        np.testing.assert_allclose(jac,fd,atol=1e-9,rtol=1e-6)
    same=dict(p,dep_like=p['joint_like'].copy())
    same['dep_zero']=moment(same,np.zeros(4),'dep')[0]
    for t in [np.zeros(4),theta,np.array([5.,-3.,2.,1.])]:
        mean,gradient=estimate(same,t,'posterior')
        np.testing.assert_array_equal(mean,same['dep_zero'])
        np.testing.assert_array_equal(gradient,np.zeros((2,4)))
    return dict(incremental_derivative=result,loss_analytic_gradient=True,
                identical_source_likelihood_cancels_exactly=True,fixed_baseline_no_gradient=True)
