"""Check weighting and paired calendar aggregation independently of fitted models."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import score_disruption_experiment as score
from reporting import Loss


class DisruptionScoringTests(unittest.TestCase):
    def loss(self, squares):
        meta = pd.DataFrame({'mvt_time': pd.to_datetime(['2025-11-01', '2025-11-01', '2025-11-02'], utc=True)},
                            index=pd.Index([100, 101, 102], name='mvt_id'))
        return Loss(meta, np.asarray(squares, float), (42, 142))

    def test_weighted_score_averages_losses_not_predictions(self):
        loss = self.loss([[1, 9, 25], [9, 25, 49]])
        weights = np.array([1., 2., 3.])
        results = score.levels({'R_A': loss, 'R_B_long': loss}, {'R_A': weights, 'R_B_long': weights})
        expected = np.sqrt((5 * 1 + 17 * 2 + 37 * 3) / 6)
        self.assertAlmostEqual(results['weighted']['R_combined']['score'], expected)

    def test_daily_weighted_bootstrap_uses_each_draws_denominator(self):
        loss = self.loss([[1, 9, 25], [9, 25, 49]])
        weights = np.array([1., 2., 3.])
        calendar = pd.date_range('2025-11-01', periods=2, tz='UTC')
        draws = np.array([[2, 0], [0, 2], [1, 1]])
        observed = score.bootstrap_mse(loss, weights, calendar, draws)
        np.testing.assert_allclose(observed, [39 / 3, 37, 150 / 6])

    def test_shared_ids_combine_before_concentration(self):
        left = self.loss([[1, 1, 1], [1, 1, 1]])
        right = self.loss([[2, 3, 4], [2, 3, 4]])
        groups = {arm: {'R_A': loss, 'R_B_long': loss} for arm, loss in [('control', left), ('candidate', right)]}
        result, rows = score.concentration(groups, {'R_A': np.ones(3), 'R_B_long': np.ones(3)})
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(result['total_mse_change'], 2)
        self.assertAlmostEqual(result['gross_increase'], 2)


if __name__ == '__main__':
    unittest.main()
