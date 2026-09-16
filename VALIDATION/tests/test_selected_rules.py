"""Four-hour boundary and agreement with the already-scored threshold sweep."""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'pipeline'))
from registered_rules import apply_rule
from prediction_rules import apply_rule as six_hour
from tools.sweep_rule_thresholds import eligible_gaps, threshold_loss
from reporting import Loss


class SelectedRuleTests(unittest.TestCase):
    def test_sweep_equivalence_and_raw_eligibility(self):
        t = pd.Timestamp('2025-07-01', tz='UTC')
        raw = pd.DataFrame({'ADEP_mvt': ['LIRF']*6+['LFPG','LIRF'],
                            'FLIGHT_ID_mvt': [np.nan]*5+[123., np.nan, np.nan],
                            'MVT_TIME_UTC_mvt': t,
                            'SCHED_TIME_UTC_mvt': [t-pd.Timedelta(seconds=x) for x in
                                                  [14399,14400,14401,21599,21600,30000,30000,30000]],
                            'ADEP_flt': 'LIRF', 'taxi_sec': -1e20})
        raw.loc[7, 'SCHED_TIME_UTC_mvt'] = pd.NaT
        prediction, target = np.arange(8)+500., np.arange(8)+600.
        out, audit = apply_rule(prediction, raw, 'rome_4h')
        np.testing.assert_array_equal(out, [500,14400,14401,21599,21600,505,506,507])
        loss = Loss(pd.DataFrame({'y': target}), ((prediction-target)**2)[None,:], (42,))
        swept, mask, _ = threshold_loss(loss, eligible_gaps(raw), 4)
        np.testing.assert_array_equal(swept.sq[0], (out-target)**2)
        np.testing.assert_array_equal(mask, audit.rule_triggered)
        np.testing.assert_array_equal(apply_rule(prediction, raw, 'rome_6h')[0],
                                      six_hour(prediction, raw, 'rome_6h')[0])
        np.testing.assert_array_equal(prediction, np.arange(8)+500.)


if __name__ == '__main__':
    unittest.main()
