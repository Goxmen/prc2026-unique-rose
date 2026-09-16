"""Clock isolation, residual reconstruction, saved predictions and linked recipes."""

import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from VALIDATION.tests.test_v3 import departures, arrivals, config
import harness
import modeling
import submit
from clock_features import clock_features
from prediction_rules import apply_rule
from records import MODEL_FILES, ROOT, sha256


def clock_rows(times, targets=None):
    frame = departures(times, targets)
    frame['FLIGHT_ID_mvt'] = 1.
    for name in ['raw_flight_id_missing', 'raw_plan_block_missing', 'raw_aobt3_missing']:
        frame[name] = False
    return frame


class ClockModelTests(unittest.TestCase):
    def test_fallback_sign_missingness_and_label_isolation(self):
        raw = clock_rows(pd.date_range('2025-07-01', periods=7, tz='UTC'))
        raw.loc[0, 'SCHED_TIME_UTC_mvt'] -= pd.Timedelta(minutes=10)
        for row, missing in [(1, ['AOBT_3_flt']), (2, ['AOBT_3_flt', 'EOBT_1_flt']),
                             (3, ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt']),
                             (4, ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt']),
                             (5, ['AOBT_3_flt', 'EOBT_1_flt', 'IOBT_flt', 'LOBT_flt', 'SCHED_TIME_UTC_mvt'])]:
            raw.loc[row, missing] = pd.NaT
        raw.loc[6, 'AOBT_3_flt'] = raw.loc[6, 'MVT_TIME_UTC_mvt'] + pd.Timedelta(minutes=1)
        raw.loc[0, 'raw_aobt3_missing'] = True  # recovered availability must retain raw provenance
        f = clock_features(raw)
        np.testing.assert_array_equal(f.clock_baseline_source, [0, 1, 2, 3, 4, 5, 0])
        np.testing.assert_array_equal(f.clock_baseline_sec, [1200]*5 + [0, 0])
        self.assertEqual(f.clock_aobt3_minus_schedule_sec.iloc[0], 600)
        self.assertEqual(f.clock_span_sec.iloc[0], 600)
        self.assertEqual(f.clock_available_count.iloc[5], 0)
        self.assertTrue(np.isnan(f.clock_span_sec.iloc[5]))
        self.assertEqual(f.clock_aobt3_available.iloc[0], 1)
        self.assertEqual(f.clock_raw_aobt3_missing.iloc[0], 1)
        raw['taxi_sec'], raw['BLOCK_TIME_UTC_mvt'] = -1e20, pd.Timestamp('2030-01-01')
        pd.testing.assert_frame_equal(f, clock_features(raw))

    def test_residual_fit_and_negative_reconstruction(self):
        raw = clock_rows(pd.date_range('2025-01-01', periods=450, freq='h', tz='UTC'), [500]*450)
        arr = arrivals(raw.MVT_TIME_UTC_mvt-pd.Timedelta(hours=1))
        cfg = config(clock_features=True, target_mode='clock_residual')
        X, y, _ = modeling.training_matrix(raw, arr, cfg)
        direct, yd, _ = modeling.training_matrix(raw, arr, config(clock_features=True))
        pd.testing.assert_frame_equal(X, direct)
        np.testing.assert_array_equal(y, yd)
        legacy, _, _ = modeling.training_matrix(raw, arr, config())
        pd.testing.assert_frame_equal(X[legacy.columns], legacy)
        with contextlib.redirect_stderr(io.StringIO()):
            model = modeling.fit_booster(X, y, 42, 2, cfg, 'residual check')
        np.testing.assert_allclose(model.predict(X), -700.)
        np.testing.assert_allclose(modeling.predict(model, X, target_mode='clock_residual'), 500.)
        fake = argparse.Namespace(params={'num_threads': 1}, predict=lambda X, **kw: np.array([-500., -1500.]))
        V = pd.DataFrame({'clock_baseline_sec': [1000., 1000.]})
        np.testing.assert_array_equal(modeling.predict(fake, V, target_mode='clock_residual'), [500., 0.])

    def test_harness_saved_model_and_rule_order(self):
        dep = clock_rows(pd.date_range('2025-01-01', '2025-07-31 12:00', freq='12h'))
        val_mask = dep.MVT_TIME_UTC_mvt.dt.month.eq(7)
        rule_index = dep.index[val_mask][0]
        dep.loc[rule_index, 'FLIGHT_ID_mvt'] = np.nan
        dep.loc[rule_index, 'raw_flight_id_missing'] = True
        dep.loc[rule_index, 'SCHED_TIME_UTC_mvt'] = dep.loc[rule_index, 'MVT_TIME_UTC_mvt']-pd.Timedelta(hours=6)
        dep.loc[dep.index[val_mask][-2:], 'taxi_sec'] = [-10, 0]
        arr = arrivals(dep.MVT_TIME_UTC_mvt-pd.Timedelta(hours=1))
        cfg = config(clock_features=True, target_mode='clock_residual', prediction_rule='rome_6h')
        cfg.update(seeds=[42], checkpoints=[1])
        args = harness.parse_args(['--folds', 'A1', '--clock-features', '--target-mode', 'clock_residual'])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, windows = harness.plan(args)
            folds, models = harness.execute(dep, arr, windows, cfg, Path(temp))
            saved = pd.read_parquet(Path(temp)/folds['A1']['predictions']['42'])
            td, ta = modeling.period_rows(dep, '2025-01-01', '2025-06-30'), modeling.period_rows(arr, '2025-01-01', '2025-06-30')
            vd, va = modeling.month_rows(dep, '2025-07'), modeling.month_rows(arr, '2025-07')
            _, _, cats = modeling.training_matrix(td, ta, cfg)
            V = modeling.validation_matrix(td, ta, vd, va, '2025-06-30', cats, cfg)
            changed = vd.assign(taxi_sec=1e10, BLOCK_TIME_UTC_mvt=pd.Timestamp('2030-01-01'))
            pd.testing.assert_frame_equal(V, modeling.validation_matrix(td, ta, changed, va, '2025-06-30', cats, cfg))
            model = lgb.Booster(model_file=str(Path(temp)/models['M1']['seeds']['42']['model']))
            raw_prediction = modeling.predict(model, V, target_mode='clock_residual', threads=2)
            expected, _ = apply_rule(raw_prediction, vd, 'rome_6h')
            np.testing.assert_allclose(saved.model_pred, raw_prediction)
            np.testing.assert_allclose(saved.pred, expected)
            np.testing.assert_array_equal(saved.y, vd.taxi_sec)
            self.assertEqual(saved.pred.iloc[0], 21600.)
            self.assertEqual(saved.pred_round_1.iloc[0], 21600.)
            with patch.object(harness, 'fit_booster', side_effect=AssertionError('must reuse fit')):
                repeated, _ = harness.execute(dep, arr, windows, cfg, Path(temp),
                                              resume_progress={'models': models, 'folds': {}})
            again = pd.read_parquet(Path(temp)/repeated['A1']['predictions']['42'])
            np.testing.assert_allclose(saved.pred, again.pred)

    def test_linked_residual_recipe_cannot_change_target_or_features(self):
        run = {'run_id': 'T2', 'protocol': 'v3', 'status': 'complete',
               'config': config(clock_features=True, target_mode='clock_residual', prediction_rule='rome_6h'),
               'source': {'files': {p: sha256(ROOT/p) for p in MODEL_FILES}},
               'rule_source': {'pipeline/prediction_rules.py': sha256(ROOT/'pipeline/prediction_rules.py')}}
        args = argparse.Namespace(run_id='T2', seed=None, **dict.fromkeys(submit.DEFAULTS))
        with patch.object(submit, 'read_run', return_value=(run, Path('.'))):
            cfg, _, _ = submit.resolve_config(args)
            self.assertEqual(cfg['target_mode'], 'clock_residual')
            self.assertTrue(cfg['clock_features'])
            args.target_mode = 'direct'
            with self.assertRaisesRegex(ValueError, 'differs'):
                submit.resolve_config(args)
            args.target_mode, args.clock_features = None, False
            with self.assertRaisesRegex(ValueError, 'differs'):
                submit.resolve_config(args)


if __name__ == '__main__':
    unittest.main()
