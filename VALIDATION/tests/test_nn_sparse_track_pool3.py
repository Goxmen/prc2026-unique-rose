"""Continuation must change only member seeds and use the frozen inference arms."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nn_sparse_track_pool3 as pool


class SparsePoolContinuationTests(unittest.TestCase):
    def test_only_three_seed_literals_and_no_parent_mutation(self):
        before_function = pool.base.train
        before_namespace = dict(pool.base.train.__globals__)
        result = pool.check_specialization()
        self.assertEqual(result['seed_substitutions'], 3)
        self.assertTrue(result['all_other_training_ast_exact'])
        self.assertTrue(result['original_globals_unchanged'])
        self.assertIs(pool.base.train, before_function)
        self.assertEqual(before_namespace.keys(), pool.base.train.__globals__.keys())
        for key, value in before_namespace.items():
            self.assertIs(value, pool.base.train.__globals__[key])

    def test_fixed_half_and_symmetric_availability_fallback(self):
        parent = np.array([100., 200., 300., 400.])
        direct = np.array([130., 160., 330., 370.])
        candidate = np.array([90., 250., 340., 360.])
        calls = np.array([120., 190., 320., 410.])
        available = np.array([False, False, True, True])
        arms = pool.neural_arms(parent, direct, candidate, calls, available)
        self.assertEqual(list(arms), pool.ARMS)
        np.testing.assert_array_equal(arms['half_sparse3_v58'], .5*(parent+candidate))
        np.testing.assert_array_equal(arms['half_direct3_v58'], .5*(parent+direct))
        for label in ['available_half_sparse3_v58', 'available_half_direct3_v58']:
            np.testing.assert_array_equal(arms[label][~available], parent[~available])
        np.testing.assert_array_equal(arms['available_half_sparse3_v58'][available],
                                      arms['half_sparse3_v58'][available])
        np.testing.assert_array_equal(arms['available_half_direct3_v58'][available],
                                      arms['half_direct3_v58'][available])
        np.testing.assert_array_equal(arms['half_sparse3_calls15'], .5*(candidate+calls))
        np.testing.assert_array_equal(arms['half_direct3_calls15'], .5*(direct+calls))

    def test_original_k0_artifact_names_preserved(self):
        prediction, model, record = pool.paths('2025-01', 1)
        self.assertEqual(prediction.name, '2025-01_s42_k1.parquet')
        self.assertEqual(model.name, '2025-01_s42_k1.pt')
        self.assertEqual(record.name, '2025-01_s42_k1.json')
        self.assertNotEqual(record, pool.OUT/'networks'/'2025-01_s42.json')
        self.assertEqual(pool.DESIGN['new_members'], [1, 2])
        self.assertEqual(pool.DESIGN['epochs'], {'2025-01': 21, '2025-07': 17})


if __name__ == '__main__':
    unittest.main()
