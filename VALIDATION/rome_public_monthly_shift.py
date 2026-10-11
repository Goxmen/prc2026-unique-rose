"""Infer a fixed airline/month Rome probability shift from public daily totals.

This differs from independent day-level flight conditioning: one regularized
logit shift per airline is shared over the month. Inference accepts only public
aggregates, flight inputs, retained probabilities and mode predictions. No
departure target enters inference. No hyperparameter or strength search.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

SETTINGS = dict(logit_prior_sd=.5, likelihood='Student t3 with fixed incoming variance',
    ordinary_sigma=150., common_day_sigma=15., optimizer='L-BFGS-B from zero',
    optimizer_gtol=1e-7, optimizer_ftol=1e-12, optimizer_maxiter=500,
    grouping='one airline shift shared across input-assigned interior off-block dates',
    uncertainty='source/count mismatch plus independent recording modes and ordinary aggregate',
    scope='Rome classifier probabilities strictly between zero and one; explicit VFR excluded',
    source_dates='takeoff minus frozen incoming primary prediction; no regrouping after update',
    no_parameter_or_weight_search=True)
FIELDS = ['mvt_id', 'airline', 'day', 'p', 'c0', 'c1', 'eligible']


def prepare(frame, source, prm):
    f = frame[FIELDS].copy()  # departure labels cannot enter the optimizer
    assert f.mvt_id.is_unique and np.isfinite(f[['p', 'c0', 'c1']]).all().all()
    assert f.p.between(0, 1).all() and f[['c0', 'c1']].ge(0).all().all()
    assert not source.day.duplicated().any()
    f['airline'] = f.airline.fillna('__unknown__').astype(str)
    days = sorted(set(f.loc[f.eligible, 'day']) & set(source.day))
    sources = source.set_index('day').loc[days]
    assert np.isfinite(sources[['total', 'n_public']]).all().all()
    daymap = {day: k for k, day in enumerate(days)}
    observed = f.eligible.to_numpy(bool) & f.day.isin(days).to_numpy()
    active = observed & f.p.gt(0).to_numpy() & f.p.lt(1).to_numpy() & f.c0.ne(f.c1).to_numpy()
    airlines = sorted(set(f.loc[active, 'airline']))
    airlinemap = {name: k for k, name in enumerate(airlines)}
    d = f.loc[observed, 'day'].map(daymap).to_numpy(int)
    p = f.p.to_numpy(float)
    c0, dc = f.c0.to_numpy(float), (f.c1-f.c0).to_numpy(float)
    baseline = np.bincount(d, weights=(c0+p*dc)[observed], minlength=len(days))
    n = np.bincount(d, minlength=len(days))
    latent = np.bincount(d, weights=(p*(1-p)*dc**2)[observed], minlength=len(days))
    dn = sources.n_public.to_numpy(float)-n
    variance = (prm['sigma']**2+(prm['kappa']*dn)**2+latent+
        n*SETTINGS['ordinary_sigma']**2+n**2*SETTINGS['common_day_sigma']**2)
    return dict(frame=f, days=days, airlines=airlines, active=active, baseline=baseline,
        target=sources.total.to_numpy(float)-prm['bias'], sd=np.sqrt(variance), count=n,
        dn=dn, latent_variance=latent,
        active_days=f.loc[active, 'day'].map(daymap).to_numpy(int),
        active_airlines=f.loc[active, 'airline'].map(airlinemap).to_numpy(int),
        p=p[active], logit=logit(p[active]), dc=dc[active])


def objective(theta, data):
    p = expit(data['logit']+theta[data['active_airlines']])
    change = np.bincount(data['active_days'], weights=(p-data['p'])*data['dc'],
                         minlength=len(data['days']))
    z = (data['target']-data['baseline']-change)/data['sd']
    prior_precision = 1/SETTINGS['logit_prior_sd']**2
    value = .5*prior_precision*(theta@theta)+np.sum(2*np.log1p(z*z/3))
    derivative = (-4*z/(3+z*z)/data['sd'])[data['active_days']]
    gradient = prior_precision*theta+np.bincount(data['active_airlines'],
        weights=derivative*p*(1-p)*data['dc'], minlength=len(theta))
    return float(value), gradient


def infer(frame, source, prm):
    data = prepare(frame, source, prm)
    theta = np.zeros(len(data['airlines']))
    initial, _ = objective(theta, data)
    if len(theta):
        result = minimize(objective, theta, args=(data,), jac=True, method='L-BFGS-B',
            options=dict(gtol=SETTINGS['optimizer_gtol'], ftol=SETTINGS['optimizer_ftol'],
                maxiter=SETTINGS['optimizer_maxiter'], maxls=50))
        assert result.success, result.message
        theta = result.x
        terminal, grad = objective(theta, data)
        assert terminal <= initial+1e-9 and np.isfinite(theta).all()
        iterations, gradient = int(result.nit), float(np.max(abs(grad)))
    else:
        terminal, iterations, gradient = initial, 0, 0.
    p = frame.p.to_numpy(float).copy()
    updated = expit(data['logit']+theta[data['active_airlines']])
    # Preserve exact identity if the optimizer took no step.
    if np.any(theta):
        p[data['active']] = updated
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_array_equal(p[~data['active']], frame.p.to_numpy()[~data['active']])
    log = dict(airlines=dict(zip(data['airlines'], map(float, theta))),
        initial_objective=initial, final_objective=terminal, iterations=iterations,
        max_abs_gradient=gradient, active_rows=int(data['active'].sum()),
        max_probability_change=float(np.max(abs(p-frame.p))),
        max_abs_logit_shift=float(np.max(abs(theta))) if len(theta) else 0.)
    day = pd.DataFrame(dict(day=data['days'], n=data['count'], dn=data['dn'],
        baseline=data['baseline'], target=data['target'], sd=data['sd'],
        latent_variance=data['latent_variance']))
    day['adapted'] = data['baseline']+np.bincount(data['active_days'],
        weights=(p[data['active']]-data['p'])*data['dc'], minlength=len(data['days']))
    return p, log, day


def checks():
    f = pd.DataFrame(dict(mvt_id=np.arange(6), airline=['A','B','A','B','C',None],
        day=['2025-01-02']*2+['2025-01-03']*4, p=[.2,.6,.3,.5,0,1],
        c0=[4100.,3100.,5100.,2100.,100.,0.], c1=0., eligible=[True]*5+[False]))
    source = pd.DataFrame(dict(day=['2025-01-02','2025-01-03'], total=[5000.,5200.], n_public=[2,3]))
    prm = dict(bias=20., sigma=100., kappa=600.)
    data = prepare(f, source, prm)
    np.testing.assert_allclose(data['baseline'], [.8*4100+.4*3100, .7*5100+.5*2100+100])
    # An independently enumerated two-airline/day derivative by finite differences.
    theta = np.array([.2,-.3])
    value, grad = objective(theta, data)
    h = 1e-5
    finite = np.array([(objective(theta+np.eye(2)[k]*h,data)[0]-
        objective(theta-np.eye(2)[k]*h,data)[0])/(2*h) for k in range(2)])
    np.testing.assert_allclose(grad, finite, atol=1e-8, rtol=0)
    p, log, days = infer(f, source, prm)
    poisoned, _, _ = infer(f.assign(y=1e12, true_day='1900-01-01', BLOCK_TIME_UTC_mvt='2099-01-01'),source,prm)
    np.testing.assert_array_equal(p,poisoned)
    order=np.array([5,3,1,4,0,2])
    shuffled,_,_=infer(f.iloc[order],source.iloc[::-1],prm)
    np.testing.assert_allclose(shuffled,p[order],atol=1e-12,rtol=0)
    aligned=source.copy();aligned['total']=data['baseline']+prm['bias']
    unchanged,_,_=infer(f,aligned,prm)
    np.testing.assert_array_equal(unchanged,f.p)
    absent=f.assign(eligible=False)
    unchanged,_,_=infer(absent,source,prm)
    np.testing.assert_array_equal(unchanged,f.p)
    increased=source.copy();increased['total']=data['baseline']+prm['bias']+1000
    down,_,_=infer(f,increased,prm)
    assert (down[data['active']]<f.p.to_numpy()[data['active']]).all()
    shifts=logit(p[data['active']])-logit(f.p.to_numpy()[data['active']])
    np.testing.assert_allclose(shifts[[0,2]],log['airlines']['A'],atol=1e-15,rtol=0)
    np.testing.assert_allclose(shifts[[1,3]],log['airlines']['B'],atol=1e-15,rtol=0)
    return dict(hand_daily_sums=True,analytic_gradient_error=float(abs(grad-finite).max()),
        finite_difference_gradient=True, label_and_true_date_isolation=True,
        row_source_order_invariant=True, matched_source_identity=True, empty_scope_identity=True,
        higher_public_total_reduces_copy_probability=True, airline_shift_shared_across_days=True)


if __name__ == '__main__':
    print(checks())
