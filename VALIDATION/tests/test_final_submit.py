"""Category widening and exact input comparison used by the final-test-set runner."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / 'pipeline'),
                str(Path(__file__).resolve().parents[1] / 'tools')]
import final_submit as fs


class FinalSubmitTests(unittest.TestCase):
    def test_widen_keeps_values_and_unions_categories(self):
        a = pd.DataFrame({'airline': pd.Categorical(['AZA', 'DLH']), 'x': [1., 2.]})
        b = pd.DataFrame({'airline': pd.Categorical(['KLM', 'AZA']), 'x': [3., np.nan]})
        a2, b2 = fs.widen([a.copy(), b.copy()])
        self.assertEqual(list(a2.airline.astype(str)), ['AZA', 'DLH'])
        self.assertEqual(list(b2.airline.astype(str)), ['KLM', 'AZA'])
        self.assertEqual(set(a2.airline.cat.categories), {'AZA', 'DLH', 'KLM'})
        self.assertTrue(a2.airline.dtype == b2.airline.dtype)
        both = pd.concat([a2, b2], ignore_index=True)
        self.assertIsInstance(both.airline.dtype, pd.CategoricalDtype)

    def test_same_frame_is_exact_and_treats_missing_as_equal(self):
        t = pd.to_datetime(['2026-02-01 00:00:05', None], utc=True)
        a = pd.DataFrame({'c': pd.Categorical(['A', None], categories=['A', 'B']), 'x': [1.5, np.nan], 't': t})
        b = pd.DataFrame({'c': pd.Categorical(['A', None], categories=['B', 'A']), 'x': [1.5, np.nan], 't': t})
        self.assertTrue(fs.same_frame(a, b))
        c = b.copy()
        c.loc[0, 'x'] = 1.5 + 1e-9
        self.assertFalse(fs.same_frame(a, c))
        d = b.copy()
        d['t'] = t + pd.Timedelta(seconds=12)
        self.assertFalse(fs.same_frame(a, d))
        self.assertFalse(fs.same_frame(a, b[['x', 'c', 't']]))


if __name__ == '__main__':
    unittest.main()
