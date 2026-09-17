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
import harness
import mixture
import reporting
from VALIDATION.tests.test_v3 import departures, arrivals, config


class Stub:
    def __init__(self, values):
        self.values = np.asarray(values, float)

    def predict(self, X, num_iteration=None):
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

    def test_harness_saves_components_and_resume_reuses_them(self):
        dep = synthetic_departures()
        arr = arrivals(dep.MVT_TIME_UTC_mvt - pd.Timedelta(hours=1))
        args = harness.parse_args(["--folds", "A1", "--mixture", "lirf-schedule"])
        cfg = config(mixture="lirf-schedule")
        cfg["seeds"] = [42]
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            _, windows = harness.plan(args)
            folds, models = harness.execute(dep, arr, windows, cfg, Path(temp))
            frame = pd.read_parquet(Path(temp) / folds["A1"]["predictions"]["42"])
            self.assertTrue(set(mixture.COLUMNS) <= set(frame.columns))
            self.assertTrue(np.isfinite(frame[["pred"] + mixture.COLUMNS]).all().all())
            details = models["M1"]["seeds"]["42"]["mixture"]
            self.assertEqual(details["classifier_positives"], int((dep.taxi_sec.eq(1200) & dep.ADEP_mvt.eq("LIRF")
                                                                 & (dep.MVT_TIME_UTC_mvt < "2025-07-01")).sum()))
            self.assertTrue(frame.loc[frame.airport.ne("LIRF"), "p_substituted"].eq(0).all())
            record = {"config": cfg, "folds": folds, "prediction_column": "pred_mix_ordinary"}
            loss = reporting.load_losses(record, temp)["A1"]
            ordered = frame.sort_values("mvt_id")
            np.testing.assert_allclose(loss.sq[0], (ordered.y - ordered.pred_mix_ordinary) ** 2)
            with patch.object(harness, "fit_booster", side_effect=AssertionError("must reuse base fit")), \
                 patch.object(mixture, "fit_components", side_effect=AssertionError("must reuse components")):
                resumed, _ = harness.execute(dep, arr, windows, cfg, Path(temp),
                                             resume_progress={"models": models, "folds": {}})
            again = pd.read_parquet(Path(temp) / resumed["A1"]["predictions"]["42"])
            for column in ["pred"] + mixture.COLUMNS:
                np.testing.assert_allclose(frame[column], again[column], rtol=0, atol=1e-10)

    def test_mixture_rejects_checkpoints(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            harness.parse_args(["--mixture", "lirf-schedule", "--rounds", "10", "--checkpoints", "5"])


if __name__ == "__main__":
    unittest.main()
