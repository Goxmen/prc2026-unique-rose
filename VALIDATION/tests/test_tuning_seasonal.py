"""Guard the seasonal tuning search: fixed trial list, exact control parameters, scoring."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / 'pipeline')]
import overnight_models
import tuning_seasonal as ts


class TuningSeasonalTest(unittest.TestCase):
    def test_trials_are_fixed_and_start_with_control(self):
        a, b = ts.sample_settings(), ts.sample_settings()
        self.assertEqual(a, b)
        self.assertEqual(a[0], ts.CONTROL)
        self.assertEqual(len(a), ts.STAGE_A['trials'] + 1)
        self.assertEqual(len({tuple(sorted(s.items())) for s in a}), len(a))
        for s in a[1:]:
            self.assertTrue(all(s[k] in ts.SPACE[k] for k in ts.SPACE))

    def test_control_parameters_are_the_production_ones(self):
        for seed in (42, 142):
            self.assertEqual(ts.lgb_params(ts.CONTROL, seed, 0.05),
                             overnight_models.model_params('control', seed, ts.THREADS))

    def test_overrides_reach_the_parameters(self):
        settings = dict(ts.CONTROL, num_leaves=511, cat_smooth=50.0)
        params = ts.lgb_params(settings, 42, 0.02)
        self.assertEqual((params['num_leaves'], params['cat_smooth'], params['learning_rate']), (511, 50.0, 0.02))
        self.assertEqual(params['objective'], 'regression')

    def test_checkpoints_include_control_rounds(self):
        self.assertIn(ts.CONTROL_ROUNDS, ts.checkpoints(ts.STAGE_A['rounds']))
        self.assertEqual(ts.checkpoints(1000), [200, 400, 600, 800, 1000])

    def test_scores_split_by_month_and_flight_id(self):
        y = np.array([100., 100., 100., 100.])
        pred = np.array([110., 100., 130., 100.])
        has_id = np.array([True, False, True, True])
        month = np.array(['2025-01', '2025-01', '2025-07', '2025-07'])
        out = ts.scores(y, pred, has_id, month)
        self.assertAlmostEqual(out['all_2025-01'], np.sqrt(100 / 2))
        self.assertAlmostEqual(out['all_2025-07'], np.sqrt(900 / 2))
        self.assertAlmostEqual(out['all'], (np.sqrt(100 / 2) + np.sqrt(900 / 2)) / 2)
        self.assertAlmostEqual(out['id_2025-01'], 10.0)
        self.assertAlmostEqual(out['id'], (10.0 + np.sqrt(900 / 2)) / 2)
        one = ts.scores(y[:2], pred[:2], has_id[:2], month[:2])
        self.assertAlmostEqual(one['all'], np.sqrt(100 / 2))

    def test_folds_are_the_requested_design(self):
        self.assertEqual(ts.FOLDS['2025-01'], ['2025-03', '2025-04', '2025-05', '2025-06', '2025-07',
                                               '2025-08', '2025-09', '2025-10', '2025-11'])
        self.assertEqual(ts.FOLDS['2025-07'], ['2025-01', '2025-02', '2025-03', '2025-09', '2025-10',
                                               '2025-11', '2025-12'])
        self.assertEqual((ts.fold_freeze('2025-01'), ts.fold_freeze('2025-07')), ('2025-11-30', '2025-12-31'))

    def test_bootstrap_signs_and_null(self):
        rng = np.random.default_rng(0)
        day = np.repeat([f'2025-01-{d:02d}' for d in range(1, 31)], 50)
        month = np.full(len(day), '2025-01')
        se = rng.exponential(1000., len(day))
        same = ts.paired_bootstrap(day, month, se, se)
        self.assertEqual((same['change'], same['ci95']), (0.0, [0.0, 0.0]))
        better = ts.paired_bootstrap(day, month, se, se * 0.8)
        self.assertLess(better['change'], 0)
        self.assertEqual(better['decision'], 'improvement')

    def test_recipes_differ_only_where_the_ledger_says(self):
        r = ts.RECIPES
        self.assertEqual(r['v39']['ordinary'], r['v40']['ordinary'])
        self.assertEqual((r['v39']['exclude_rule_rows'], r['v40']['exclude_rule_rows']), (False, True))
        self.assertEqual(r['v41']['ordinary'], r['v40']['ordinary'] + list(ts.taxi_occupancy.FEATURES))
        self.assertEqual(r['v41']['classifier'], r['v40']['classifier'] + list(ts.taxi_occupancy.FEATURES))
        self.assertEqual(r['v42']['ordinary'], r['v40']['ordinary'] + list(ts.features.FEATURES))
        self.assertEqual(r['v43']['ordinary'], r['v42']['ordinary'] + list(ts.sf.FEATURES))
        for name in ('v42', 'v43'):
            self.assertEqual(r[name]['classifier'], r['v40']['classifier'])
        self.assertEqual(ts.LINEAGE, [('v39', 'v40'), ('v40', 'v41'), ('v40', 'v42'), ('v42', 'v43')])

    def test_stored_columns_cover_each_recipe(self):
        for names in (['v39', 'v40', 'v41'], ['v42', 'v43']):
            cols = ts.recipe_columns(names)
            self.assertEqual(len(cols), len(set(cols)))
            for name in names:
                needed = ts.rome_hybrid.regressor_features(ts.KIND, ts.RECIPES[name]['ordinary'])
                self.assertTrue(set(needed + ts.RECIPES[name]['classifier']) <= set(cols))

    def test_ledger_rows_match_official_scores(self):
        rows = ts.ledger_rows()
        self.assertEqual(set(rows), set(ts.RECIPES))
        self.assertEqual(rows['v43']['lb'], 274.766)
        self.assertEqual(rows['v40']['weighted'], 264.3397)


if __name__ == '__main__':
    unittest.main()
