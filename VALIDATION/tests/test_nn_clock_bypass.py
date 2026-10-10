"""Physical eligibility and exact parent-preserving initialization tests."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nn_clock_bypass_jan_jul as runner


class ClockBypassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_source_provenance_and_independent_clock_masks(self):
        frame = pd.DataFrame({name: [100., 100., 100., -1., 10801., np.nan, 100., 100., np.inf]
                              for name in runner.CLOCKS})
        frame['clock_aobt3_available'] = [1, 1, 0, 1, 1, 1, 1, 1, 1]
        frame['clock_raw_aobt3_missing'] = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        frame['ab_matched'] = [1, 1, 1, 1, 1, 1, 0, 1, 1]
        frame['ab_stand_n'] = [2, 2, 2, 2, 2, 2, 2, 0, 2]
        frame['op_matched'] = [1, 1, 1, 1, 1, 1, 0, 1, 1]
        original = frame.copy(deep=True)
        clocks, masks = runner.physical_clocks(frame)
        expected = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1],
                             [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                             [1, 0, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(masks, expected)
        self.assertTrue(np.isfinite(clocks).all())
        self.assertTrue((clocks[~masks] == 0).all())
        self.assertEqual(clocks.dtype, np.float32)
        pd.testing.assert_frame_equal(frame, original)

    def test_inclusive_limits_without_clipping_invalid_values(self):
        frame = pd.DataFrame({name: [0., 10800., -0.00001, 10800.00001] for name in runner.CLOCKS})
        frame['clock_aobt3_available'] = 1
        frame['clock_raw_aobt3_missing'] = 0
        frame['ab_matched'] = 1
        frame['ab_stand_n'] = 1
        frame['op_matched'] = 1
        clocks, mask = runner.physical_clocks(frame)
        self.assertTrue(mask[:2].all())
        self.assertFalse(mask[2:].any())
        np.testing.assert_array_equal(clocks[:2], frame.loc[:1, runner.CLOCKS].to_numpy(np.float32))

    def test_initial_mean_gradients_rng_and_missing_fallback(self):
        checks = runner.self_check()
        self.assertTrue(all(value is True for key, value in checks.items() if key != 'device'))


if __name__ == '__main__':
    unittest.main()
