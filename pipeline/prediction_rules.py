"""Fixed, observable-input prediction rules shared by validation and submission."""

import numpy as np
import pandas as pd

RULE_INPUTS = ['ADEP_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
ROME_THRESHOLD_SECONDS = 21600


def apply_rule(predictions, raw, rule='off'):
    """Apply the registered rule to original movement fields, never recovered IDs.

    Inputs and predictions have the same positional order. Targets and plan
    fields are excluded before evaluating eligibility. Return fresh predictions
    and an audit aligned to the caller's index; never mutate either input.
    """
    pred = np.asarray(predictions, dtype=float).copy()
    if pred.ndim != 1 or len(pred) != len(raw) or not np.isfinite(pred).all():
        raise ValueError('Expected one finite prediction per input row')
    if rule not in ['off', 'rome_6h']:
        raise ValueError(f'Unknown prediction rule: {rule}')
    if rule == 'off':
        return pred, pd.DataFrame({'rule_triggered': False, 'rule_changed': False,
                                   'rule_gap_sec': np.nan}, index=raw.index)
    visible = raw.loc[:, RULE_INPUTS]
    gap = (pd.to_datetime(visible.MVT_TIME_UTC_mvt, utc=True)
           - pd.to_datetime(visible.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    triggered = (visible.ADEP_mvt.eq('LIRF') & visible.FLIGHT_ID_mvt.isna()
                 & gap.ge(ROME_THRESHOLD_SECONDS)).fillna(False).to_numpy(bool)
    changed = triggered & (pred != gap.to_numpy())
    pred[triggered] = gap.to_numpy()[triggered]
    return pred, pd.DataFrame({'rule_triggered': triggered, 'rule_changed': changed,
                               'rule_gap_sec': gap}, index=raw.index)
