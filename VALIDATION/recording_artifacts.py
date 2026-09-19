"""Small recording-artifact corrections, fitted only inside each training window.

These experiments layer onto row 34. Their gates and regularization are fixed
before the new control's validation scores are examined. Prediction uses an
explicit observable-column allowlist; departure labels and block stamps never
enter it. See docs/RECORDING_ARTIFACT_EXPERIMENTS.md for the frozen recipes.
"""

import numpy as np
import pandas as pd

import plan_window
import record_fixes

DAY = 86400.0
SETTINGS = dict(clock_agreement_sec=1800, clock_fit_min_sec=7200,
                clock_prior_rows=25, day_min_gap_sec=14 * 3600,
                day_prior_rows=5, handler_threshold=0.5,
                handler_min_arrivals=20, handler_min_training_dates=5,
                rule_max_gap_sec=12 * 3600, rule_max_probability=0.05)
RAW = ['ADEP_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
       'LOBT_flt', 'AOBT_3_flt']
SIGNALS = ['handler_arr_sched_share', 'handler_arr_n']
CANDIDATES = {
    'twins': 'pred_twins',
    'clock12': 'pred_clock12',
    'clock2': 'pred_clock2',
    'day_shrunk': 'pred_day_shrunk',
    'day_handler': 'pred_day_handler',
    'rule_confidence': 'pred_rule_confidence',
    'combined': 'pred_combined',
    'rule_shrunk': 'pred_rule_shrunk',
    'combined_midgap': 'pred_combined_midgap',
}


def observable(frame):
    """No target, block time or arbitrary extra feature is accepted by the model."""
    out = frame.loc[:, RAW + SIGNALS].copy()
    for name in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt']:
        out[name] = pd.to_datetime(out[name], utc=True)
    return out


def clocks(raw):
    movement = raw['MVT_TIME_UTC_mvt']
    return {name: (movement - raw[col]).dt.total_seconds().to_numpy(float)
            for name, col in [('schedule', 'SCHED_TIME_UTC_mvt'), ('lobt', 'LOBT_flt'),
                              ('aobt3', 'AOBT_3_flt')]}


def clock_mask(raw, gaps, threshold):
    return (raw.FLIGHT_ID_mvt.notna().to_numpy()
            & np.isfinite(gaps['lobt']) & np.isfinite(gaps['aobt3'])
            & (np.minimum(gaps['lobt'], gaps['aobt3']) >= threshold)
            & (np.abs(gaps['lobt'] - gaps['aobt3']) <= SETTINGS['clock_agreement_sec']))


def day_mask(raw, gaps):
    local_day = lambda c: raw[c].dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
    days = (local_day('MVT_TIME_UTC_mvt') - local_day('SCHED_TIME_UTC_mvt')).dt.days.to_numpy(float)
    return (raw.ADEP_mvt.eq('LIRF').to_numpy() & raw.FLIGHT_ID_mvt.isna().to_numpy()
            & (gaps['schedule'] >= SETTINGS['day_min_gap_sec']) & (days == 1))


def rule_band(raw, gaps):
    return (raw.ADEP_mvt.eq('LIRF').to_numpy() & raw.FLIGHT_ID_mvt.isna().to_numpy()
            & (gaps['schedule'] >= 14400) & (gaps['schedule'] < SETTINGS['rule_max_gap_sec']))


def _weight(step, residual, prior=0.0, scale=None):
    """Ridge least squares on the actual correction step, constrained to a blend."""
    if not len(step):
        return float(prior)
    penalty = SETTINGS['day_prior_rows'] * (float(np.median(step ** 2)) if scale is None else scale)
    denominator = float(step @ step) + penalty
    if denominator <= 0:
        return float(prior)
    return float(np.clip((step @ residual + penalty * prior) / denominator, 0, 1))


def fit(training, late_mean):
    """Fit bounded residuals and day weights on unique, in-window training rows."""
    if training.mvt_id.isna().any() or training.mvt_id.duplicated().any():
        raise ValueError('Fit requires unique training movements')
    if late_mean is None or not np.isfinite(late_mean):
        raise ValueError('A finite training-window late-taxi mean is required')
    raw = observable(training)
    y = training.y.to_numpy(float)
    if not np.isfinite(y).all() or (y < 1).any() or (y > 200000).any():
        raise ValueError('Use the control model\'s retained training-label range')
    gaps = clocks(raw)
    eligible = clock_mask(raw, gaps, SETTINGS['clock_fit_min_sec'])
    # Residuals are relative to the actual plan clock, never the schedule.
    residual = np.clip(y[eligible] - gaps['aobt3'][eligible], -3660, 3660)
    prior = SETTINGS['clock_prior_rows']
    pooled = float(residual.sum() / (len(residual) + prior))
    airports = raw.ADEP_mvt.to_numpy()[eligible]
    by_airport = {str(ap): float((residual[airports == ap].sum() + prior * pooled)
                                / ((airports == ap).sum() + prior)) for ap in np.unique(airports)}
    mask = day_mask(raw, gaps)
    step = DAY + late_mean - gaps['schedule'][mask]
    res = y[mask] - gaps['schedule'][mask]
    pooled_weight = _weight(step, res)
    dates = raw.MVT_TIME_UTC_mvt.dt.normalize().to_numpy()[mask]
    share = raw.handler_arr_sched_share.to_numpy(float)[mask]
    count = raw.handler_arr_n.to_numpy(float)[mask]
    supported = np.isfinite(share) & (count >= SETTINGS['handler_min_arrivals'])
    weights, support, groups = {}, {}, {}
    scale = float(np.median(step ** 2)) if len(step) else 0
    for name, group in [('low', share < SETTINGS['handler_threshold']),
                        ('high', share >= SETTINGS['handler_threshold'])]:
        group = group & supported
        n_dates = len(np.unique(dates[group]))
        support[name] = dict(rows=int(group.sum()), dates=n_dates)
        groups[name] = group
    stratified = all(s['dates'] >= SETTINGS['handler_min_training_dates'] for s in support.values())
    for name, group in groups.items():
        weights[name] = (_weight(step[group], res[group], prior=pooled_weight, scale=scale)
                         if stratified else pooled_weight)
    midgap = rule_band(raw, gaps)
    rule_step = np.minimum(late_mean, np.maximum(gaps['schedule'][midgap] - 3600, 60)) - gaps['schedule'][midgap]
    rule_weight = _weight(rule_step, y[midgap] - gaps['schedule'][midgap])
    return dict(settings=SETTINGS.copy(), late_mean=float(late_mean),
                clock=dict(rows=int(eligible.sum()), pooled_residual_sec=pooled,
                           airport_residual_sec=by_airport),
                day=dict(rows=int(mask.sum()), dates=len(np.unique(dates)),
                         pooled_weight=pooled_weight, handler_weights=weights, handler_support=support,
                         handler_stratified=stratified),
                rule=dict(rows=int(midgap.sum()), dates=int(raw.loc[midgap, 'MVT_TIME_UTC_mvt'].dt.normalize().nunique()),
                          weight=rule_weight))


def predict(control, frame, parameters, probability, twin_mask=None):
    """Apply fixed candidate interventions; returns candidates plus observable masks.

    `probability` is the already-fitted Rome classifier's output. `twin_mask`
    must be calculated from only the exposed movement context, with no labels.
    """
    raw = observable(frame)
    n = len(raw)
    control, probability = np.asarray(control, float), np.asarray(probability, float)
    if control.shape != (n,) or probability.shape != (n,) or not np.isfinite(control).all():
        raise ValueError('Control/probability vectors must align with finite predictions')
    if parameters['settings'] != SETTINGS:
        raise ValueError('Fitted correction settings differ from the implementation')
    gaps = clocks(raw)
    cp = parameters['clock']
    offset = np.array([cp['airport_residual_sec'].get(str(ap), cp['pooled_residual_sec'])
                       for ap in raw.ADEP_mvt], float)
    clock_target = plan_window.apply(gaps['aobt3'] + offset, raw)
    masks, predictions = {}, {}
    for name, hours in [('clock12', 12), ('clock2', 2)]:
        masks[name] = clock_mask(raw, gaps, hours * 3600)
        predictions[name] = np.where(masks[name], clock_target, control)
    masks['day'] = day_mask(raw, gaps)
    dp = parameters['day']
    weights = np.full(n, dp['pooled_weight'])
    count = raw.handler_arr_n.to_numpy(float)
    share = raw.handler_arr_sched_share.to_numpy(float)
    support = np.isfinite(share) & (count >= SETTINGS['handler_min_arrivals'])
    for name, group in [('low', share < SETTINGS['handler_threshold']),
                        ('high', share >= SETTINGS['handler_threshold'])]:
        weights[support & group] = dp['handler_weights'][name]
    step = DAY + parameters['late_mean'] - gaps['schedule']
    predictions['day_shrunk'] = np.where(masks['day'], gaps['schedule'] + dp['pooled_weight'] * step, control)
    predictions['day_handler'] = np.where(masks['day'], gaps['schedule'] + weights * step, control)
    masks['rule_band'] = rule_band(raw, gaps)
    masks['rule_confidence'] = (
        masks['rule_band']
        & np.isfinite(probability) & (probability >= 0) & (probability < SETTINGS['rule_max_probability']))
    late = np.minimum(parameters['late_mean'], np.maximum(gaps['schedule'] - 3600, 60))
    predictions['rule_confidence'] = np.where(
        masks['rule_confidence'], probability * gaps['schedule'] + (1 - probability) * late, control)
    predictions['rule_shrunk'] = np.where(
        masks['rule_band'], gaps['schedule'] + parameters['rule']['weight'] * (late - gaps['schedule']), control)
    if twin_mask is None:
        twin_mask = record_fixes.twin_rows(frame)
    masks['twins'] = np.asarray(twin_mask, bool)
    if masks['twins'].shape != (n,):
        raise ValueError('Twin mask must align with predictions')
    predictions['twins'] = record_fixes.twin_records(control, frame, masks['twins'])
    combined = predictions['twins'].copy()
    # The Rome regimes are disjoint; the ID-linked clock rule has last priority.
    for name, maskname in [('day_handler', 'day'), ('rule_confidence', 'rule_confidence'), ('clock12', 'clock12')]:
        mask = masks[maskname]
        combined[mask] = predictions[name][mask]
    predictions['combined'] = combined
    combined_midgap = combined.copy()
    mask = masks['rule_band']
    combined_midgap[mask] = predictions['rule_shrunk'][mask]
    predictions['combined_midgap'] = combined_midgap
    if any(not np.isfinite(p).all() or (p < 0).any() for p in predictions.values()):
        raise ValueError('Correction produced a nonfinite or negative prediction')
    return {CANDIDATES[name]: predictions[name] for name in CANDIDATES}, masks
