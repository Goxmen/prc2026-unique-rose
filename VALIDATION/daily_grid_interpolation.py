"""Deterministic interpolation on the existing daily posterior's minute grid.

Each physical delay state contributes to its two adjacent grid nodes, with
linear weights. This preserves its expected delay instead of rounding it to
one node. The physical taxi state and its prior mass stay unchanged. This is
a numerical approximation, not a claim that the continuous likelihood is exact.
"""
import itertools

import numpy as np

import daily_heavy_tail as heavy

old = heavy.old
SETTINGS = dict(grid_sec=60., method='linear adjacent-node mass allocation',
    ordinary_prior='original Gaussian', physical_states='original support and prior masses',
    no_new_parameters=True, no_random_sampling=True,
    limitation='Preserves first moments; introduces at most900 seconds-squared of grid variance per physical state')


def interpolate_states(values, weights, gap):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    exact = np.maximum(0., gap-values)/SETTINGS['grid_sec']
    lower = np.floor(exact).astype(np.int64)
    fraction = exact-lower
    expanded_values = np.repeat(values, 2)
    expanded_weights = (weights[:, None]*np.column_stack([1-fraction, fraction])).ravel()
    nodes = np.column_stack([lower, lower+1]).ravel()
    keep = expanded_weights > 0
    return expanded_values[keep], expanded_weights[keep], nodes[keep]


def support(d, p, nm, nm_gap, airport, prm):
    values, weights, _ = old.support(d, p, nm, nm_gap, airport, prm)
    return interpolate_states(values, weights, d)


def condition(d, p, nm, nm_gap, airport, prm, total, dn, interpolate=True):
    long = d > 3600
    ordinary, active = ~long, (~long)&(d > p)
    prior_ordinary = float(np.maximum(0, d[ordinary]-p[ordinary]).sum())
    n = int(active.sum())
    indvar, common = old.DESIGN['ordinary_sigma']**2, old.DESIGN['common_day_sigma']**2
    shortvar = n*indvar+n*n*common
    offsets = heavy.Z*max(np.sqrt(shortvar), 1.)
    rows = np.flatnonzero(long)
    originals = [old.support(d[i], p[i], nm[i], nm_gap[i], airport, prm) for i in rows]
    supports = [interpolate_states(s[0],s[1],d[i]) for i,s in zip(rows,originals)] if interpolate else originals
    size = 1+sum(int(s[2].max()) for s in supports)
    totals = np.arange(size)*SETTINGS['grid_sec']
    measurement_sd = np.sqrt(prm['sigma']**2+(prm['kappa']*dn)**2)
    z = (total-prm['bias']-prior_ordinary-totals[:, None]-offsets[None, :])/measurement_sd
    joint = heavy.GAUSSIAN[None, :]-2*np.log1p(z*z/3.)
    joint = np.exp(joint-joint.max())
    likelihood = joint.sum(axis=1)
    prior = np.zeros(size)
    prior[0] = 1.
    forward = []
    for ys, weights, contributions in supports:
        forward.append(prior)
        nxt = np.zeros(size)
        for weight, contribution in zip(weights, contributions):
            nxt[contribution:] += weight*prior[:size-contribution]
        prior = nxt/nxt.sum()
    post_offset = (prior[:, None]*joint).sum(axis=0)
    assert post_offset.sum() > 0
    expected_offset = float(post_offset@offsets/post_offset.sum())
    result = p.copy()
    if n:
        result[active] -= (indvar+n*common)/max(shortvar, 1)*expected_offset
        result[ordinary&~active] -= .25*n*common/max(shortvar, 1)*expected_offset
    backward = likelihood/likelihood.max()
    for j in range(len(rows)-1, -1, -1):
        ys, weights, contributions = supports[j]
        factors = np.array([forward[j][:size-c]@backward[c:] for c in contributions])
        posterior = weights*factors
        assert posterior.sum() > 0
        result[rows[j]] = posterior@ys/posterior.sum()
        nxt = np.zeros(size)
        for weight, contribution in zip(weights, contributions):
            nxt[:size-contribution] += weight*backward[contribution:]
        backward = nxt/max(nxt.max(), 1e-300)
    errors=[]
    for i,(ys,w,c) in zip(rows,originals):
        errors.append(float(w@(SETTINGS['grid_sec']*c-np.maximum(0,d[i]-ys))))
    return np.maximum(result, 0), dict(offset=expected_offset, grid_size=size,
        long_rows=len(rows), active=n, prior_ordinary=prior_ordinary, measurement_sd=measurement_sd,
        original_grid_mean_error_sec=float(sum(errors)))


def checks():
    cases=[([0.,30.,60.,90.,120.],[.1,.2,.3,.1,.3],120.),
           ([0.,6000.,9999.],[.2,.3,.5],5432.1),([17.],[1.],31.)]
    for values,weights,gap in cases:
        y,w,c=interpolate_states(values,weights,gap)
        np.testing.assert_allclose(w.sum(),1.,atol=1e-15)
        np.testing.assert_allclose(w@y,np.asarray(weights)@values,atol=1e-12)
        np.testing.assert_allclose(w@(60*c),np.asarray(weights)@np.maximum(0,gap-np.asarray(values)),atol=1e-12)
        for value,mass in zip(values,weights):
            np.testing.assert_allclose(w[y==value].sum(),mass,atol=1e-15)
            contribution=max(0,gap-value)
            var=float(w[y==value]@((c[y==value]*60-contribution)**2)/mass)
            assert var <= 900+1e-10
    prm=dict(spread=[-101.,0.,99.],sigma=120.,kappa=600.,tau=800.,bias=30.)
    d,p=np.array([1000.,500.,5000.,7000.]),np.array([800.,900.,2000.,4000.])
    nm,gap=np.zeros(4,bool),np.full(4,np.nan)
    actual,_=condition(d,p,nm,gap,'LIRF',prm,4500.,0.)
    supports=[support(d[i],p[i],nm[i],gap[i],'LIRF',prm) for i in [2,3]]
    offsets=heavy.Z*np.sqrt(old.DESIGN['ordinary_sigma']**2+old.DESIGN['common_day_sigma']**2)
    numer=np.zeros(3);denominator=0.;count=0
    for j,k,at in itertools.product(range(len(supports[0][0])),range(len(supports[1][0])),range(len(offsets))):
        contribution=(supports[0][2][j]+supports[1][2][k])*60
        residual=(4500-prm['bias']-200-contribution-offsets[at])/prm['sigma']
        weight=supports[0][1][j]*supports[1][1][k]*np.exp(heavy.GAUSSIAN[at])*(1+residual**2/3)**-2
        denominator+=weight
        numer+=weight*np.array([offsets[at],supports[0][0][j],supports[1][0][k]])
        count+=1
    offset,y0,y1=numer/denominator
    indvar,common=old.DESIGN['ordinary_sigma']**2,old.DESIGN['common_day_sigma']**2
    expected=np.maximum([p[0]-offset,p[1]-.25*common/(indvar+common)*offset,y0,y1],0.)
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
    perm=np.array([3,1,0,2]);shuffled,_=condition(d[perm],p[perm],nm[perm],gap[perm],'LIRF',prm,4500.,0.)
    np.testing.assert_allclose(shuffled,actual[perm],atol=1e-8,rtol=0)
    for select in [np.arange(4),np.array([0,1]),np.array([2,3])]:
        args=(d[select],p[select],nm[select],gap[select],'LIRF',prm,4500.,0.)
        control,_=heavy.condition(*args,logprior=heavy.GAUSSIAN)
        replay,_=condition(*args,interpolate=False)
        np.testing.assert_allclose(replay,control,atol=1e-10,rtol=0)
    return dict(first_moments_exact=True,physical_masses_exact=True,variance_bounded=True,
        enumerated_states=count,enumeration_max_error=float(abs(expected-actual).max()),
        row_permutation=True,old_control_replay=True)
