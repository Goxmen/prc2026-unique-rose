import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import callsign_lgb_submit as runner


class ComponentReplacementTest(unittest.TestCase):
    def setUp(self):
        self.parts = pd.DataFrame(dict(mvt_id=[20, 10], lgb=[100., 500.],
                                       airport=[200., 800.], catboost=[300., 1100.]))

    def test_replace_only_main_equal_mean(self):
        source = self.parts.copy(deep=True)
        old, new = runner.replace_main(self.parts, [20, 10],
                                      pd.DataFrame(dict(mvt_id=[20, 10], pred=[400., 200.])))
        np.testing.assert_array_equal(old, [200., 800.])
        np.testing.assert_array_equal(new, [300., 700.])
        pd.testing.assert_frame_equal(self.parts, source, check_exact=True)

    def test_control_reproduces_mean_and_views_are_separate(self):
        replacement = self.parts[['mvt_id', 'lgb']].rename(columns={'lgb': 'pred'})
        old, same = runner.replace_main(self.parts, [20, 10], replacement)
        np.testing.assert_array_equal(old, same)
        other_view = self.parts.copy()
        other_view['airport'] += 12
        _, shifted = runner.replace_main(other_view, [20, 10], replacement)
        np.testing.assert_array_equal(shifted, same + 4)

    def test_misaligned_duplicate_and_nonfinite_rejected(self):
        for ids, values in [([10, 20], [100., 500.]), ([20, 10], [np.nan, 500.]),
                            ([20, 10], [-1., 500.])]:
            with self.assertRaises((AssertionError, ValueError)):
                runner.replace_main(self.parts, [20, 10], pd.DataFrame(dict(mvt_id=ids, pred=values)))
        repeated = self.parts.copy()
        repeated['mvt_id'] = [20, 20]
        with self.assertRaises(ValueError):
            runner.replace_main(repeated, [20, 20], pd.DataFrame(dict(mvt_id=[20, 20], pred=[1., 2.])))


if __name__ == '__main__':
    unittest.main()
