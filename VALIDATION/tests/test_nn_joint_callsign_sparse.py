"""New joint features must preserve the original model inputs and update code."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nn_joint_callsign_sparse_jan_jul as joint


class JointSparseTests(unittest.TestCase):
    def test_feature_join_preserves_categories_index_and_independent_missingness(self):
        X = pd.DataFrame({f'num_{i}': [1., np.nan, 3.] for i in range(154)}, index=[91, 20, 45])
        X['airport'] = pd.Categorical(['B', 'A', 'B'], categories=['unused', 'B', 'A'])
        X['clock_raw_aobt3_missing'] = [1., 0., 1.]
        table = pd.DataFrame({name: np.array([100., np.nan, 300.], dtype=np.float32)
                              for name in joint.FEATURES}, index=[10, 20, 30])
        result = joint.join_features(X, np.array([30, 20, 10]), table)
        pd.testing.assert_frame_equal(result[X.columns], X, check_exact=True)
        for name in joint.FEATURES:
            np.testing.assert_equal(result[name].to_numpy(), [300., np.nan, 100.])
            self.assertEqual(result[name].dtype, np.dtype('float32'))
        self.assertEqual(result.shape, (3, 162))
        with self.assertRaises(ValueError):
            joint.join_features(X, np.array([30, 20, 999]), table)
        with self.assertRaises(ValueError):
            joint.join_features(X.drop(columns='num_0'), np.array([30, 20, 10]), table)

    def test_training_bytecode_and_original_namespace_remain_unchanged(self):
        original = joint.base.train
        original_output = original.__globals__['OUT']
        clone = joint.training_function()
        self.assertIs(clone.__code__, original.__code__)
        self.assertIsNot(clone.__globals__, original.__globals__)
        differences = [key for key in clone.__globals__
                       if clone.__globals__[key] is not original.__globals__[key]]
        self.assertEqual(differences, ['OUT'])
        self.assertEqual(clone.__globals__['OUT'], joint.OUT / 'runtime')
        self.assertEqual(original.__globals__['OUT'], original_output)
        self.assertIs(joint.base.train, original)
        self.assertEqual(joint.STOP, {'2025-01': 21, '2025-07': 17})


if __name__ == '__main__':
    unittest.main()
