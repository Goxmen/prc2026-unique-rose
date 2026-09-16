"""Check sensitivity scoring against explicit per-origin prediction overrides."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reporting import COVERAGE, Loss, group_loss
from tools.sweep_rule_thresholds import eligible_gaps, threshold_loss


class ThresholdSweepTests(unittest.TestCase):
    def test_raw_eligibility_inclusive_threshold_and_no_label_dependency(self):
        t = pd.Timestamp('2025-07-02', tz='UTC')
        raw = pd.DataFrame({'ADEP_mvt': ['LIRF']*4+['LFPG', 'LIRF'],
            'FLIGHT_ID_mvt': [np.nan, np.nan, np.nan, 10, np.nan, np.nan],
            'MVT_TIME_UTC_mvt': t, 'SCHED_TIME_UTC_mvt':
                [t-pd.Timedelta(seconds=s) for s in [14399, 14400, 21600, 30000, 30000]]+[pd.NaT]},
                index=pd.Index(range(6), name='mvt_id'))
        original = Loss(pd.DataFrame({'y': [-10, 0, 25000, 100, 100, 100]}, index=raw.index),
                        np.ones((2, 6)), (42, 142))
        gaps = eligible_gaps(raw)
        candidate, mask, _ = threshold_loss(original, gaps.iloc[::-1], 4)
        self.assertEqual(mask.tolist(), [False, True, True, False, False, False])
        np.testing.assert_array_equal(candidate.sq[:, mask], [[14400**2, 3400**2]]*2)
        self.assertEqual(threshold_loss(original, gaps, 6)[1].tolist(), [False, False, True, False, False, False])
        raw['y'], raw['BLOCK_TIME_UTC_mvt'], raw['ADEP_flt'] = -1e20, t, 'LIRF'
        pd.testing.assert_series_equal(gaps, eligible_gaps(raw))
        np.testing.assert_array_equal(original.sq, np.ones((2, 6)))

    def test_override_after_origin_averaging_matches_explicit_predictions(self):
        meta = pd.DataFrame({'mvt_time': pd.to_datetime(['2025-12-01', '2025-12-02'], utc=True),
                             'airport': 'LIRF', 'y': [30000., 100.], 'has_flight_plan': False,
                             **{key: 0 for key in COVERAGE}}, index=pd.Index([10, 20], name='mvt_id'))
        predictions = [np.array([[100., 100.], [200., 200.]]), np.array([[300., 300.], [400., 400.]])]
        before, after = {}, {}
        for i, pred in enumerate(predictions):
            before[str(i)] = Loss(meta, (pred-meta.y.to_numpy())**2, (42, 142))
            changed = pred.copy()
            changed[:, 0] = 25000
            after[str(i)] = Loss(meta, (changed-meta.y.to_numpy())**2, (42, 142))
        grouped = group_loss(before, ['0', '1'])
        candidate, _, _ = threshold_loss(grouped, pd.Series({10: 25000., 20: np.nan}), 6)
        explicit = group_loss(after, ['0', '1'])
        np.testing.assert_array_equal(candidate.sq, explicit.sq)
        # Averaging predictions before squaring would incorrectly erase seed/origin variance.
        self.assertNotEqual(float(candidate.sq[:, 1].mean()), (250.-100.)**2)


if __name__ == '__main__':
    unittest.main()
