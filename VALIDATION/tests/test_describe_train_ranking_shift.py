"""Guard the descriptive comparison's denominators, novelty and reference ranges."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from describe_train_ranking_shift import (MISSING, MONTHS, TARGETS,
                                         category_comparison, category_counts,
                                         compare_scalars, reference_summary)


class DescriptiveShiftTests(unittest.TestCase):
    def test_missing_is_a_distribution_change_but_not_a_new_stand(self):
        result = category_comparison({'A': 10}, {'A': 5, MISSING: 4, 'B': 1})
        self.assertEqual(result['tv_pp'], 50)
        self.assertEqual(result['missing_pct'], 40)
        self.assertEqual(result['new_share_pct'], 10)

    def test_new_share_is_directional_and_uses_target_rows(self):
        a, b = {'A': 90, 'B': 10}, {'B': 1}
        forward, backward = category_comparison(a, b), category_comparison(b, a)
        self.assertEqual(forward['tv_pp'], backward['tv_pp'])
        self.assertEqual(forward['new_share_pct'], 0)
        self.assertEqual(backward['new_share_pct'], 90)

    def test_category_counts_keep_nulls_and_do_not_invent_absent_categories(self):
        values = pd.Series(pd.Categorical(['A', None, 'A'], categories=['A', 'B'])).astype('string')
        self.assertEqual(category_counts(values), {'A': 2, MISSING: 1})

    def test_reference_ties_and_unavailable_values(self):
        result = reference_summary([0., 1., 2., np.nan, None], 1.)
        self.assertEqual(result['n'], 3)
        self.assertAlmostEqual(result['fraction_smaller'], 1/3)
        self.assertEqual(result['max'], 2.)
        self.assertIsNone(reference_summary([None, np.nan], 1.)['max'])

    def test_matches_each_target_to_its_own_month_and_keeps_denominators(self):
        values = list(range(12)) + [20, 7]
        rows = [dict(month=m, airport='X', feature='no_flight_id', statistic='rate_pct',
                     value=v, n_valid=90, n_total=100, unit='percentage points')
                for m, v in zip(MONTHS + TARGETS, values)]
        result = compare_scalars(pd.DataFrame(rows)).set_index('target_month')
        self.assertEqual(result.at['2026-01', 'delta'], 20)
        self.assertEqual(result.at['2026-07', 'delta'], 1)
        self.assertEqual(result.at['2026-01', 'adjacent_n'], 11)
        self.assertEqual(result.at['2026-01', 'all_pairs_n'], 66)
        self.assertEqual(result.at['2026-01', 'adjacent_max'], 1)
        self.assertEqual(result.at['2026-01', 'all_pairs_max'], 11)
        self.assertEqual(result.at['2026-01', 'target_n_valid'], 90)
        self.assertEqual(result.at['2026-01', 'target_n_total'], 100)


if __name__ == '__main__':
    unittest.main()
