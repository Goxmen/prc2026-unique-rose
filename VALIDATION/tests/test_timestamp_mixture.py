"""Mixture outcome boundaries, observable inference, replay and recipe guards."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from VALIDATION.tests.test_clock_models import clock_rows
from VALIDATION.tests.test_v3 import arrivals, config
import harness
import modeling
import submit
from prediction_rules import apply_rule
from records import MODEL_FILES, ROOT, sha256
from timestamp_mixture import MixtureBooster, training_classes, validate_recipe


class TimestampMixtureTests(unittest.TestCase):
    def test_training_outcome_boundaries_and_priority(self):
        y = [7200, 7201, 9000, 9000, 86400, 86401, 93600, 93601, 87000, 150000]
        X = pd.DataFrame({'mvt_minus_sched_sec': [7200, 7261, 8939, np.nan, 1200,
                                                1200, 1200, 1200, 87000, 1200]})
        np.testing.assert_array_equal(training_classes(X, y), [0, 1, 0, 0, 0, 2, 2, 0, 1, 0])

    def test_probabilities_weight_components_and_missing_schedule(self):
        gate = argparse.Namespace(predict=lambda X, **kw: np.array([[.8, .15, .05], [.5, .5, 0]]))
        ordinary = argparse.Namespace(predict=lambda X, **kw: np.array([600., -10.]))
        model = MixtureBooster(gate, ordinary, dict(ordinary=10, schedule=1, day=1), 2)
        X = pd.DataFrame({'mvt_minus_sched_sec': [100000., np.nan]}, index=[4, 9])
        audit = []
        expected = [.8*600 + .15*100000 + .05*87000, 0]
        np.testing.assert_allclose(modeling.predict(model, X, audit_out=audit), expected)
        self.assertEqual(audit[0].index.tolist(), [4, 9])
        self.assertEqual(audit[0].mixture_day_sec.iloc[1], 86400)
        self.assertEqual(audit[0].mixture_schedule_sec.iloc[1], 0)
        with self.assertRaisesRegex(ValueError, 'residual'):
            modeling.predict(model, X, target_mode='clock_residual')
        gate.predict = lambda X, **kw: np.zeros((len(X), 3))
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            modeling.predict(model, X)

    def test_harness_bundle_replay_checkpoint_rule_and_hidden_labels(self):
        dep = clock_rows(pd.date_range('2025-01-01', '2025-07-31 12:00', freq='12h'))
        dep.loc[0, 'taxi_sec'] = 86401
        dep.loc[1, 'taxi_sec'] = 30000
        dep.loc[1, 'SCHED_TIME_UTC_mvt'] = dep.loc[1, 'MVT_TIME_UTC_mvt']-pd.Timedelta(seconds=30000)
        july = dep.index[dep.MVT_TIME_UTC_mvt.dt.month.eq(7)]
        dep.loc[july[-2:], 'taxi_sec'] = [-10, 0]
        dep.loc[july[0], 'FLIGHT_ID_mvt'] = np.nan
        dep.loc[july[0], 'raw_flight_id_missing'] = True
        dep.loc[july[0], 'SCHED_TIME_UTC_mvt'] = dep.loc[july[0], 'MVT_TIME_UTC_mvt']-pd.Timedelta(hours=6)
        arr = arrivals(dep.MVT_TIME_UTC_mvt-pd.Timedelta(hours=1))
        cfg = config(clock_features=True, model_mode='timestamp_mixture', target_mode='direct',
                     prediction_rule='rome_6h')
        cfg.update(seeds=[42], checkpoints=[1])
        args = harness.parse_args(['--folds', 'A1', '--clock-features', '--model-mode', 'timestamp_mixture'])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, windows = harness.plan(args)
            folds, models = harness.execute(dep, arr, windows, cfg, Path(temp))
            saved = pd.read_parquet(Path(temp)/folds['A1']['predictions']['42'])
            td, ta = modeling.period_rows(dep, '2025-01-01', '2025-06-30'), modeling.period_rows(arr, '2025-01-01', '2025-06-30')
            vd, va = modeling.month_rows(dep, '2025-07'), modeling.month_rows(arr, '2025-07')
            X, y, cats = modeling.training_matrix(td, ta, cfg)
            V = modeling.validation_matrix(td, ta, vd, va, '2025-06-30', cats, cfg)
            changed = vd.assign(taxi_sec=150000, BLOCK_TIME_UTC_mvt=pd.Timestamp('2030-01-01'))
            pd.testing.assert_frame_equal(V, modeling.validation_matrix(td, ta, changed, va, '2025-06-30', cats, cfg))
            details = models['M1']['seeds']['42']
            self.assertEqual(details['components']['class_counts'],
                             {'ordinary': len(y)-2, 'schedule': 1, 'day': 1})
            curve = json.loads((Path(temp)/details['training_loss']).read_text())
            self.assertEqual(len(curve['components']['gate']['training']['multi_logloss']), 2)
            self.assertEqual(len(curve['components']['ordinary']['training']['rmse']), 2)
            model = modeling.load_booster(Path(temp)/details['model'], cfg)
            # Outcome construction is prohibited during all inference paths.
            with patch('timestamp_mixture.training_classes', side_effect=AssertionError('training only')):
                expected = modeling.predict(model, V, threads=2)
                checkpoint = modeling.predict(model, V, threads=2, num_iteration=1)
            np.testing.assert_allclose(saved.model_pred, expected)
            np.testing.assert_allclose(saved.pred, apply_rule(expected, vd, 'rome_6h')[0])
            np.testing.assert_allclose(saved.pred_round_1, apply_rule(checkpoint, vd, 'rome_6h')[0])
            np.testing.assert_array_equal(saved.y, vd.taxi_sec)
            self.assertEqual(saved.pred.iloc[0], 21600)
            np.testing.assert_allclose(saved.filter(like='mixture_p_').sum(axis=1), 1)
            with patch.object(harness, 'fit_booster', side_effect=AssertionError('must reuse fits')):
                again, _ = harness.execute(dep, arr, windows, cfg, Path(temp),
                                            resume_progress={'models': models, 'folds': {}})
            pd.testing.assert_frame_equal(saved, pd.read_parquet(Path(temp)/again['A1']['predictions']['42']))

    def test_linked_recipe_guards_and_absent_rare_classes(self):
        cfg = config(clock_features=True, model_mode='timestamp_mixture', target_mode='direct')
        for changes in [{'clock_features': False}, {'target_mode': 'clock_residual'}, {'early_stopping_rounds': 10}]:
            with self.assertRaises(ValueError):
                validate_recipe(dict(cfg, **changes))
        run = {'run_id': 'M1', 'protocol': 'v3', 'status': 'complete', 'config': cfg,
               'source': {'files': {p: sha256(ROOT/p) for p in MODEL_FILES}}}
        args = argparse.Namespace(run_id='M1', seed=None, **dict.fromkeys(submit.DEFAULTS))
        with patch.object(submit, 'read_run', return_value=(run, Path('.'))):
            resolved, _, _ = submit.resolve_config(args)
            self.assertEqual(resolved['model_mode'], 'timestamp_mixture')
            args.model_mode = 'single'
            with self.assertRaisesRegex(ValueError, 'differs'):
                submit.resolve_config(args)
        raw = clock_rows(pd.date_range('2025-01-01', periods=450, freq='h', tz='UTC'), [500]*450)
        arr = arrivals(raw.MVT_TIME_UTC_mvt-pd.Timedelta(hours=1))
        X, y, _ = modeling.training_matrix(raw, arr, cfg)
        with contextlib.redirect_stderr(io.StringIO()):
            model = modeling.fit_booster(X, y, 42, 2, cfg, 'absent rare classes')
        self.assertTrue(np.isfinite(modeling.predict(model, X)).all())
        self.assertEqual(model.training_summary['class_counts'], dict(ordinary=450, schedule=0, day=0))


if __name__ == '__main__':
    unittest.main()
