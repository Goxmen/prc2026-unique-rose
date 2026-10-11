"""Use retained broad taxi supports inside the count/sum moment posterior."""
import numpy as np
from scipy.special import logsumexp

import daily_delay_posterior as retained
import taxi_count_sum_posterior as original

SETTINGS = dict(original.SETTINGS,
    long_prior='unchanged retained15 training quantiles plus copy and5% broad support; mean-preserving tilt',
    source_uncertainty='same completed count/sum calibration; no refit',
    prior_join_bounds='not applied inside support; original final matched/recovered bounds still applied',
    allocation='same ambiguous Rome mixture scope; other long rows contribute broad uncertainty only')


def support(mean, gap, parameters):
    values, weights, _ = retained.support(gap, mean, False, np.nan, 'LIRF', parameters)
    valid = (values >= 0) & (values <= SETTINGS['cutoff'])
    vector = np.column_stack([valid.astype(float), np.where(valid, values, 0)])
    average = weights@vector
    centered = vector-average
    covariance = (centered.T*weights)@centered
    np.testing.assert_allclose(values@weights, mean, atol=1e-5, rtol=1e-8)
    return values, weights, vector, average, covariance


def condition(mean, gap, tau, source_n, source_sum, parameters):
    mean, gap = np.asarray(mean, float), np.asarray(gap, float)
    result = mean.copy()
    active = ((gap > SETTINGS['copy_gap_min']) & (gap < SETTINGS['copy_gap_max'])
              & (mean > tau) & (mean < gap))
    vector, covariance = original.normal_moments(mean, SETTINGS['ordinary_sigma'])
    supports = {}
    for i in np.flatnonzero(gap > SETTINGS['copy_gap_min']):
        supports[i] = support(mean[i], gap[i], parameters['held_parameters'])
        vector[i], covariance[i] = supports[i][3:]
    expected_n = float(vector[:, 0].sum())
    supported = (len(mean) >= 20 and source_n >= 20 and expected_n >= 20
                 and .5 <= source_n/expected_n <= 1.2)
    if not supported or not active.any():
        return result, dict(supported=supported, active=0, expected_n=expected_n), []
    bias = np.array([1., len(mean)/100.])@np.asarray(parameters['coefficients'])
    observed = np.array([source_n, source_sum])+bias
    prior = vector.sum(axis=0)
    noise = len(mean)*np.asarray(parameters['covariance_per_row'])
    noise += np.diag([SETTINGS['count_noise_floor']**2, SETTINGS['sum_noise_floor']**2])
    noise[1, 1] += (SETTINGS['common_day_sigma']*expected_n)**2
    aggregate_covariance = covariance.sum(axis=0)+noise
    details = []
    for i in np.flatnonzero(active):
        values, weights, contributions, _, _ = supports[i]
        other_mean = prior-vector[i]
        other_covariance = aggregate_covariance-covariance[i]
        ll = np.array([original.log_likelihood(observed-other_mean-v, other_covariance) for v in contributions])
        log_weights = np.where(weights > 0, np.log(np.maximum(weights, 1e-300)), -np.inf)
        posterior = np.exp(log_weights+ll-logsumexp(log_weights+ll))
        result[i] = values@posterior
        details.append(dict(at=int(i), prior_mean=float(values@weights), posterior_mean=float(result[i]),
            prior_variance=float(weights@(values-mean[i])**2),
            posterior_sd=float(np.sqrt(posterior@(values-result[i])**2))))
    np.testing.assert_array_equal(result[~active], mean[~active])
    assert np.isfinite(result).all() and (result >= 0).all()
    return result, dict(supported=True, active=int(active.sum()), expected_n=expected_n,
        expected_sum=float(prior[1]), adjusted_n=float(observed[0]), adjusted_sum=float(observed[1]),
        bias_count=float(bias[0]), bias_sum=float(bias[1]),
        noise_count_sd=float(np.sqrt(noise[0, 0])), noise_sum_sd=float(np.sqrt(noise[1, 1])),
        max_change=float(np.max(abs(result-mean)))), details


def checks():
    from scipy.stats import multivariate_t
    prm = dict(tau=1000., spread=[-500., -200., 0., 200., 500.])
    parameters = dict(coefficients=[[1., 1000.], [0., 0.]],
        covariance_per_row=[[.04, 30.], [30., 40000.]], held_parameters=prm)
    mean = np.r_[np.full(24, 1000.), 4000.]
    gap = np.r_[np.full(24, 1400.), 9000.]
    actual, info, _ = condition(mean, gap, 1000., 24., 26000., parameters)
    # Independent exact sum over this flight's discrete states; others retain Gaussian moments.
    values, weights, vectors, average, covariance = support(mean[-1], gap[-1], prm)
    ordinary, cv = original.normal_moments(mean[:24])
    observed = np.array([25., 27000.])
    expected_n = ordinary[:, 0].sum()+average[0]
    noise = 25*np.asarray(parameters['covariance_per_row'])+np.diag([1., 60.**2])
    noise[1, 1] += (15*expected_n)**2
    shape = .6*(cv.sum(axis=0)+noise)
    likelihood = np.array([multivariate_t.pdf(observed-ordinary.sum(axis=0)-v, shape=shape, df=5) for v in vectors])
    expected = (weights*likelihood)@values/(weights*likelihood).sum()
    np.testing.assert_allclose(actual[-1], expected, atol=1e-9, rtol=0)
    order = np.random.default_rng(27).permutation(len(mean))
    permuted, _, _ = condition(mean[order], gap[order], 1000., 24., 26000., parameters)
    np.testing.assert_allclose(permuted, actual[order], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(actual[:24], mean[:24])
    assert info['active'] == 1
    return dict(independent_discrete_state_enumeration=True, permutation=True,
        mean_preserving_support=True, unchanged_ordinary_rows=True)
