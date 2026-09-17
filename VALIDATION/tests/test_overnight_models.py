"""Intervention isolation, routing, holdout safety and ordinary model replay."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modeling
import overnight_models as om
from registered_rules import apply_rule
from VALIDATION.tests.test_clock_models import clock_rows
from VALIDATION.tests.test_v3 import arrivals


class OvernightModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = clock_rows(pd.date_range('2025-01-01', periods=900, freq='h', tz='UTC'),
                             600 + np.arange(900) % 300)
        cls.arr = arrivals(cls.raw.MVT_TIME_UTC_mvt - pd.Timedelta(hours=1))
        cls.X, cls.y, cls.categories = modeling.training_matrix(cls.raw, cls.arr, om.BASE_CONFIG)

    def fit(self, X, y, raw, recipe):
        with contextlib.redirect_stderr(io.StringIO()):
            return om.fit_model(X, y, raw, recipe, seed=42, rounds=4, threads=2)

    def test_exact_feature_interventions_and_no_accidental_target_input(self):
        self.assertEqual(self.X.columns.tolist(), om.ORIGINAL_FEATURES + om.CLOCK_FEATURES)
        expected = {'control': 27, 'no_month': 26, 'no_history': 22,
                    'clock_features': 45, 'small_trees': 27, 'rule_routing': 27}
        dirty = self.X.assign(taxi_sec=1e12, BLOCK_TIME_UTC_mvt=0)
        for recipe, size in expected.items():
            selected = om.select_features(self.X, recipe)
            self.assertEqual(selected.shape, (len(self.raw), size))
            pd.testing.assert_frame_equal(selected, om.select_features(dirty, recipe))
        self.assertNotIn('month', om.select_features(self.X, 'no_month'))
        self.assertFalse(set(modeling.HISTORY_FEATURES) & set(om.select_features(self.X, 'no_history')))
        self.assertEqual(om.model_params('control', 42),
                         {**om.model_params('small_trees', 42), 'num_leaves': 255})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            om.select_features(self.X, 'typo')

    def test_routing_boundary_uses_original_observables_only(self):
        raw = self.raw.iloc[:7].copy()
        raw['FLIGHT_ID_mvt'] = np.nan
        raw['SCHED_TIME_UTC_mvt'] = raw.MVT_TIME_UTC_mvt - pd.to_timedelta(
            [14399, 14400, 14401, 20000, 20000, 20000, 20000], unit='s')
        raw.loc[3, 'FLIGHT_ID_mvt'] = 123.
        raw.loc[4, 'ADEP_mvt'] = 'LFPG'
        raw.loc[5, 'SCHED_TIME_UTC_mvt'] = pd.NaT
        raw.loc[6, 'MVT_TIME_UTC_mvt'] = pd.NaT
        expected = [True, False, False, True, True, True, True]
        np.testing.assert_array_equal(om.training_mask(raw, 'rule_routing'), expected)
        changed = raw.assign(taxi_sec=-1e30, BLOCK_TIME_UTC_mvt=pd.Timestamp('1900-01-01'))
        np.testing.assert_array_equal(om.training_mask(changed, 'rule_routing'), expected)
        for recipe in om.RECIPES:
            if recipe != 'rule_routing':
                self.assertTrue(om.training_mask(raw, recipe).all())

    def test_control_matches_shared_baseline_fit(self):
        control, history = self.fit(self.X, self.y, self.raw, 'control')
        X27, y27, _ = modeling.training_matrix(self.raw, self.arr,
                                               dict(om.BASE_CONFIG, clock_features=False))
        pd.testing.assert_frame_equal(X27, om.select_features(self.X, 'control'))
        with contextlib.redirect_stderr(io.StringIO()):
            reference = modeling.fit_booster(X27, y27, 42, 4,
                                            dict(om.BASE_CONFIG, threads=2), 'reference')
        np.testing.assert_array_equal(om.predict_model(control, self.X, 'control', threads=2),
                                      modeling.predict(reference, X27, threads=2))
        self.assertEqual(len(history['training']['rmse']), 4)

    def test_routed_fit_replay_and_all_evaluation_rows_retained(self):
        raw = self.raw.copy()
        raw.loc[:24, 'FLIGHT_ID_mvt'] = np.nan
        raw.loc[:24, 'SCHED_TIME_UTC_mvt'] = raw.loc[:24, 'MVT_TIME_UTC_mvt'] - pd.Timedelta(hours=4)
        X, y, _ = modeling.training_matrix(raw, self.arr, om.BASE_CONFIG)
        mask = om.training_mask(raw, 'rule_routing')
        self.assertEqual(int((~mask).sum()), 25)
        routed, _ = self.fit(X, y, raw, 'rule_routing')
        manual, _ = self.fit(X.loc[mask], y[mask], raw.loc[mask], 'control')
        predicted = om.predict_model(routed, X, 'rule_routing', threads=2)
        np.testing.assert_array_equal(predicted, om.predict_model(manual, X, 'control', threads=2))
        self.assertEqual(len(predicted), len(raw))
        final, audit = apply_rule(predicted, raw, 'rome_4h')
        self.assertEqual(len(final), len(raw))
        np.testing.assert_array_equal(final[audit.rule_triggered], np.full(25, 14400.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.txt'
            routed.save_model(str(path))
            loaded = lgb.Booster(model_file=str(path))
            np.testing.assert_array_equal(predicted, om.predict_model(loaded, X, 'rule_routing', threads=2))
            with self.assertRaisesRegex(ValueError, 'feature identity'):
                om.predict_model(loaded, X, 'clock_features', threads=2)

    def test_heldout_labels_do_not_change_any_candidate_inputs(self):
        vd = clock_rows(pd.date_range('2025-07-01', periods=5, freq='h', tz='UTC'))
        va = arrivals(vd.MVT_TIME_UTC_mvt - pd.Timedelta(hours=1))
        altered = vd.assign(taxi_sec=-1e20, BLOCK_TIME_UTC_mvt=pd.Timestamp('1900-01-01', tz='UTC'))
        original = modeling.validation_matrix(self.raw, self.arr, vd, va, '2025-06-30',
                                             self.categories, om.BASE_CONFIG)
        other = modeling.validation_matrix(self.raw, self.arr, altered, va, '2025-06-30',
                                          self.categories, om.BASE_CONFIG)
        for recipe in om.RECIPES:
            pd.testing.assert_frame_equal(om.select_features(original, recipe),
                                          om.select_features(other, recipe))

    def test_bad_alignment_and_empty_routing_rejected(self):
        with self.assertRaisesRegex(ValueError, 'index alignment'):
            self.fit(self.X, self.y, self.raw.iloc[::-1], 'rule_routing')
        with self.assertRaisesRegex(ValueError, 'targets differ'):
            self.fit(self.X, self.y[::-1], self.raw, 'control')
        raw = self.raw.assign(FLIGHT_ID_mvt=np.nan)
        raw['SCHED_TIME_UTC_mvt'] = raw.MVT_TIME_UTC_mvt - pd.Timedelta(hours=4)
        with self.assertRaisesRegex(ValueError, 'No training rows'):
            self.fit(self.X, self.y, raw, 'rule_routing')


if __name__ == '__main__':
    unittest.main()
