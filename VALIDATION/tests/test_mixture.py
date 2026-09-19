"""Checks for the Rome schedule-substitution mixture (candidate M1)."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modeling
import mixture
import reporting
from VALIDATION.tests.test_v3 import departures, arrivals, config


class Stub:
    def __init__(self, values):
        self.values = np.asarray(values, float)
        self.params = {'num_threads': 1}

    def predict(self, X, num_iteration=None, num_threads=None):
        return self.values[X.index.to_numpy()]


def synthetic_departures():
    """LIRF rows scheduled 20 min before takeoff; every fifth label is that gap."""
    dep = departures(pd.date_range("2025-01-01", "2025-07-31 12:00", freq="6h"))
    substituted = np.arange(len(dep)) % 5 == 0
    dep["taxi_sec"] = np.where(substituted, 1200, 700 + np.arange(len(dep)) % 120)
    dep.loc[np.arange(len(dep)) % 7 == 0, "ADEP_mvt"] = "EGLL"
    dep["raw_flight_id_missing"] = np.arange(len(dep)) % 3 == 0
    dep["raw_plan_block_missing"] = dep.raw_flight_id_missing
    return dep


class MixtureTests(unittest.TestCase):
    def test_substitution_label_uses_label_and_schedule_at_lirf_only(self):
        X = pd.DataFrame({mixture.GAP: [1200., 1200., 1200., 1200.]})
        frame = pd.DataFrame({"ADEP_mvt": ["LIRF", "LIRF", "LIRF", "EGLL"]})
        s = mixture.substituted(X, [1200, 1260, 1261, 1200], frame)
        self.assertEqual(s.tolist(), [True, True, False, False])

    def test_prediction_columns_blend_only_lirf_and_keep_rule_definition(self):
        V = pd.DataFrame({mixture.GAP: [30000., 30000., 30000., 900., np.nan]})
        frame = pd.DataFrame({"ADEP_mvt": ["LIRF", "LIRF", "EGLL", "LIRF", "LIRF"],
                              "raw_flight_id_missing": [True, False, True, True, True],
                              "raw_plan_block_missing": [True, False, True, True, True]})
        base = np.array([1000., 1000., 1000., 1000., 1000.])
        stubs = {"substitution": Stub([0.5, 0.25, 0.9, 1.0, 0.9]),
                 "ordinary": Stub([800., 800., 800., 800., 800.])}
        out = mixture.predict_components(stubs, V, frame, base)
        np.testing.assert_allclose(out["p_substituted"], [0.5, 0.25, 0, 1.0, 0])
        np.testing.assert_allclose(out["pred_mix_base"], [15500., 8250., 1000., 900., 1000.])
        np.testing.assert_allclose(out["pred_mix_ordinary"], [15400., 8100., 1000., 900., 1000.])
        np.testing.assert_allclose(out["pred_rule6h"], [30000., 1000., 1000., 1000., 1000.])

    def test_components_save_load_and_predict_without_refitting(self):
        # The historical Rome mixture is a standalone component API. The
        # harness's current timestamp_mixture mode is a different experiment.
        dep = synthetic_departures()
        arr = arrivals(dep.MVT_TIME_UTC_mvt - pd.Timedelta(hours=1))
        td = modeling.period_rows(dep, '2025-01-01', '2025-06-30')
        ta = modeling.period_rows(arr, '2025-01-01', '2025-06-30')
        vd, va = modeling.month_rows(dep, '2025-07'), modeling.month_rows(arr, '2025-07')
        cfg = config()
        cfg.update(seeds=[42], threads=1)
        X, y, cats = modeling.training_matrix(td, ta, cfg)
        V = modeling.validation_matrix(td, ta, vd, va, '2025-06-30', cats, cfg)
        base = np.full(len(vd), 1000.0)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), patch.object(mixture, 'CLASSIFIER_ROUNDS', 4):
            directory = Path(temp)
            boosters, details = mixture.fit_components(
                X, y, td, 42, cfg['rounds'], cfg, 'synthetic mixture', directory / 'components', directory)
            first = mixture.predict_components(boosters, V, vd, base)
            frame = modeling.prediction_frame(vd, V, '2025-01-01', '2025-06-30', '2025-07')
            frame = frame.assign(seed=42, fold='A1', pred=base, **first)
            path = directory / 'oof_A1_seed42.parquet'
            frame.to_parquet(path, index=False)
            frame = pd.read_parquet(path)
            self.assertTrue(set(mixture.COLUMNS) <= set(frame.columns))
            self.assertTrue(np.isfinite(frame[["pred"] + mixture.COLUMNS]).all().all())
            positive = int((td.taxi_sec.eq(1200) & td.ADEP_mvt.eq('LIRF')).sum())
            self.assertEqual(details['classifier_positives'], positive)
            self.assertEqual(details['classifier_train_rows'], int(td.ADEP_mvt.eq('LIRF').sum()))
            self.assertEqual(details['ordinary_train_rows'], len(td) - positive)
            for relative in [*details['models'].values(), details['ordinary_training_loss']]:
                self.assertTrue((directory / relative).is_file())
            self.assertTrue(frame.loc[frame.airport.ne("LIRF"), "p_substituted"].eq(0).all())
            folds = {'A1': dict(val='2025-07', predictions={'42': path.name})}
            record = {"config": cfg, "folds": folds, "prediction_column": "pred_mix_ordinary"}
            loss = reporting.load_losses(record, temp)["A1"]
            ordered = frame.sort_values("mvt_id")
            np.testing.assert_allclose(loss.sq[0], (ordered.y - ordered.pred_mix_ordinary) ** 2)
            with patch.object(mixture, 'fit_booster', side_effect=AssertionError('must reuse ordinary fit')), \
                 patch.object(mixture, 'fit_components', side_effect=AssertionError('must reuse components')), \
                 patch.object(mixture, 'substituted', side_effect=AssertionError('training labels only')):
                loaded = mixture.load_components(details, directory, threads=1)
                again = mixture.predict_components(loaded, V, vd, base)
                changed = vd.assign(taxi_sec=199999, BLOCK_TIME_UTC_mvt=pd.Timestamp('1900-01-01', tz='UTC'))
                without_target = mixture.predict_components(loaded, V, changed, base)
            for column in mixture.COLUMNS:
                np.testing.assert_allclose(first[column], again[column], rtol=0, atol=1e-10)
                np.testing.assert_array_equal(again[column], without_target[column])

    def test_component_fit_requires_both_rome_recording_classes(self):
        X = pd.DataFrame({mixture.GAP: [1200.0, 1200.0]})
        with patch.object(mixture, 'fit_booster', side_effect=AssertionError('must reject before fitting')):
            for airports, labels in [(['LIRF', 'LIRF'], [1200, 1200]),
                                     (['LIRF', 'LIRF'], [700, 800]),
                                     (['EGLL', 'EGLL'], [1200, 700])]:
                raw = pd.DataFrame({'ADEP_mvt': airports})
                with self.subTest(airports=airports, labels=labels), self.assertRaisesRegex(ValueError, 'both substituted and ordinary'):
                    mixture.fit_components(X, np.array(labels), raw, 42, 2, config(), 'invalid',
                                           Path('unused'), Path('.'))


if __name__ == "__main__":
    unittest.main()
