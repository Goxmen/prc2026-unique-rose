"""Leave-one-out Rome recording-mode inference from public taxi counts/sums.

Only a fixed, label-free prior and public measurements enter inference. The
publication excludes taxi times above two hours, which changes both moments.
"""
import numpy as np
from scipy.special import expit, logit, ndtr

SETTINGS = dict(cutoff=7200., ordinary_sigma=200., common_day_sigma=15.,
    count_noise_floor=1., sum_noise_floor=60., df=5,
    copy_gap_min=3600., copy_gap_max=14*3600.,
    likelihood='variance-matched bivariate Student t5; Gaussian moments for other flights',
    allocation='change ambiguous copy-mode means only; simultaneous leave-one-out updates',
    fallback='unchanged incoming prediction; no repeated conditioning')


def normal_moments(mean, sigma=200.):
    """Moments of [1(0<=Y<=7200), Y*1(0<=Y<=7200)] for Gaussian Y."""
    mean = np.asarray(mean, float)
    a, b = -mean/sigma, (SETTINGS['cutoff']-mean)/sigma
    pa, pb = np.exp(-a*a/2)/np.sqrt(2*np.pi), np.exp(-b*b/2)/np.sqrt(2*np.pi)
    n = ndtr(b)-ndtr(a)
    total = mean*n+sigma*(pa-pb)
    second = (mean*mean+sigma*sigma)*n+2*mean*sigma*(pa-pb)+sigma*sigma*(a*pa-b*pb)
    vector = np.column_stack([n, total])
    covariance = np.zeros((len(mean), 2, 2))
    covariance[:, 0, 0] = n*(1-n)
    covariance[:, 0, 1] = covariance[:, 1, 0] = total*(1-n)
    covariance[:, 1, 1] = np.maximum(second-total*total, 0)
    return vector, covariance


def mixture_moments(mean, gap, tau):
    mean, gap = np.asarray(mean, float), np.asarray(gap, float)
    assert np.isfinite(mean).all() and np.isfinite(gap).all()
    active = ((gap > SETTINGS['copy_gap_min']) & (gap < SETTINGS['copy_gap_max'])
              & (mean > tau) & (mean < gap))
    vector, covariance = normal_moments(mean, SETTINGS['ordinary_sigma'])
    rows = np.flatnonzero(active)
    probability = (mean[rows]-tau)/(gap[rows]-tau)
    zero, cov_zero = normal_moments(np.full(len(rows), tau), SETTINGS['ordinary_sigma'])
    one = np.column_stack([gap[rows] <= SETTINGS['cutoff'],
                           np.where(gap[rows] <= SETTINGS['cutoff'], gap[rows], 0)]).astype(float)
    v = (1-probability[:, None])*zero+probability[:, None]*one
    delta = one-zero
    cov = ((1-probability[:, None, None])*cov_zero
           + (probability*(1-probability))[:, None, None]*delta[:, :, None]*delta[:, None, :])
    vector[rows], covariance[rows] = v, cov
    return dict(vector=vector, covariance=covariance, rows=rows, probability=probability,
                zero=zero, one=one, cov_zero=cov_zero)


def log_likelihood(residual, covariance):
    # t5 has covariance (5/3)*scale; use scale=3/5*the supplied covariance.
    scale = covariance*(SETTINGS['df']-2)/SETTINGS['df']
    sign, determinant = np.linalg.slogdet(scale)
    assert sign > 0 and np.isfinite(determinant)
    mahal = float(residual@np.linalg.solve(scale, residual))
    return -.5*determinant-.5*(SETTINGS['df']+2)*np.log1p(mahal/SETTINGS['df'])


def condition(mean, gap, tau, source_n, source_sum, parameters):
    mean = np.asarray(mean, float)
    gap = np.asarray(gap, float)
    result = mean.copy()
    moments = mixture_moments(mean, gap, tau)
    vector, covariance = moments['vector'], moments['covariance']
    expected_n = float(vector[:, 0].sum())
    supported = (len(mean) >= 20 and source_n >= 20 and expected_n >= 20
                 and .5 <= source_n/expected_n <= 1.2)
    if not supported or not len(moments['rows']):
        return result, dict(supported=supported, active=0, expected_n=expected_n), []
    x = np.array([1., len(mean)/100.])
    bias = x@np.asarray(parameters['coefficients'])
    observed = np.array([source_n, source_sum])+bias
    prior = vector.sum(axis=0)
    noise = len(mean)*np.asarray(parameters['covariance_per_row'])
    noise += np.diag([SETTINGS['count_noise_floor']**2, SETTINGS['sum_noise_floor']**2])
    # Conservatively allow common ordinary timing error, including held components.
    noise[1, 1] += (SETTINGS['common_day_sigma']*expected_n)**2
    aggregate_covariance = covariance.sum(axis=0)+noise
    details = []
    for k, i in enumerate(moments['rows']):
        other_mean = prior-vector[i]
        other_covariance = aggregate_covariance-covariance[i]
        residual_zero = observed-other_mean-moments['zero'][k]
        residual_one = observed-other_mean-moments['one'][k]
        score_zero = log_likelihood(residual_zero, other_covariance+moments['cov_zero'][k])
        score_one = log_likelihood(residual_one, other_covariance)
        probability = float(expit(logit(moments['probability'][k])+score_one-score_zero))
        result[i] = (1-probability)*tau+probability*gap[i]
        details.append(dict(at=int(i), old_probability=float(moments['probability'][k]),
            new_probability=probability, loglik_zero=float(score_zero), loglik_one=float(score_one)))
    np.testing.assert_array_equal(result[np.setdiff1d(np.arange(len(mean)), moments['rows'])],
                                  mean[np.setdiff1d(np.arange(len(mean)), moments['rows'])])
    assert np.isfinite(result).all() and (result >= 0).all()
    return result, dict(supported=True, active=len(moments['rows']), expected_n=expected_n,
        expected_sum=float(prior[1]), adjusted_n=float(observed[0]), adjusted_sum=float(observed[1]),
        bias_count=float(bias[0]), bias_sum=float(bias[1]),
        noise_count_sd=float(np.sqrt(noise[0, 0])), noise_sum_sd=float(np.sqrt(noise[1, 1])),
        max_change=float(np.max(abs(result-mean)))), details


def checks():
    from scipy.integrate import quad
    from scipy.stats import multivariate_t, norm
    errors = []
    for mean in [0., 1000., 7190., 9000.]:
        v, c = normal_moments([mean])
        integrals = [quad(lambda y: y**k*norm.pdf(y, loc=mean, scale=200.), 0, 7200,
                          epsabs=1e-7)[0] for k in range(3)]
        expected = np.array([[integrals[0]*(1-integrals[0]), integrals[1]*(1-integrals[0])],
                             [integrals[1]*(1-integrals[0]), integrals[2]-integrals[1]**2]])
        np.testing.assert_allclose(v[0], integrals[:2], atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(c[0], expected, atol=1e-7, rtol=1e-9)
        errors.append(float(np.max(abs(c[0]-expected))))
    covariance = np.array([[9., 1200.], [1200., 2e6]])
    a, b = np.array([2., 800.]), np.array([-1., 1400.])
    independent = (multivariate_t.logpdf(a, shape=.6*covariance, df=5)
                   - multivariate_t.logpdf(b, shape=.6*covariance, df=5))
    np.testing.assert_allclose(log_likelihood(a, covariance)-log_likelihood(b, covariance), independent, atol=1e-12)
    mean = np.r_[np.full(23, 1000.), 4000., 5000.]
    gap = np.r_[np.full(23, 1400.), 6000., 9000.]
    prm = dict(coefficients=[[1., 1000.], [0., 0.]], covariance_per_row=[[.04, 30.], [30., 40000.]])
    out, log, details = condition(mean, gap, 1000., 24., 26000., prm)
    order = np.random.default_rng(82).permutation(len(mean))
    shuffled, _, _ = condition(mean[order], gap[order], 1000., 24., 26000., prm)
    np.testing.assert_allclose(shuffled, out[order], atol=1e-9, rtol=0)
    assert log['active'] == 2 and len(details) == 2
    np.testing.assert_array_equal(out[:23], mean[:23])
    no_source, log, _ = condition(mean, gap, 1000., 3., 2000., prm)
    np.testing.assert_array_equal(no_source, mean)
    assert not log['supported']
    return dict(quadrature_max_error=max(errors), independent_student_density=True,
        permutation=True, unchanged_ordinary_rows=True, unsupported_exact_fallback=True)
