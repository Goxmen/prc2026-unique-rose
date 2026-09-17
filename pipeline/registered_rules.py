"""Selected rule variants, preserving the frozen six-hour implementation."""

import numpy as np
import pandas as pd

from prediction_rules import RULE_INPUTS, apply_rule as fixed_rule

RULE_FILES = {
    'off': [],
    'rome_6h': ['pipeline/prediction_rules.py'],
    'rome_4h': ['pipeline/prediction_rules.py', 'pipeline/registered_rules.py'],
}


def apply_rule(predictions, raw, rule='off'):
    if rule != 'rome_4h':
        return fixed_rule(predictions, raw, rule)
    pred = np.asarray(predictions, dtype=float).copy()
    if pred.ndim != 1 or len(pred) != len(raw) or not np.isfinite(pred).all():
        raise ValueError('Expected one finite prediction per input row')
    visible = raw.loc[:, RULE_INPUTS]
    gap = (pd.to_datetime(visible.MVT_TIME_UTC_mvt, utc=True)
           - pd.to_datetime(visible.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    triggered = (visible.ADEP_mvt.eq('LIRF') & visible.FLIGHT_ID_mvt.isna()
                 & gap.ge(14400)).fillna(False).to_numpy(bool)
    changed = triggered & (pred != gap.to_numpy())
    pred[triggered] = gap.to_numpy()[triggered]
    return pred, pd.DataFrame({'rule_triggered': triggered, 'rule_changed': changed,
                               'rule_gap_sec': gap}, index=raw.index)
