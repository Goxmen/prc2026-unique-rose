"""Stage 0 checks for raw provenance, feature isolation, and round checkpoints."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness
import modeling
import reporting
from movement_data import load_phase, RECOVERY_NAMES, RAW_NAMES
from VALIDATION.tests.test_v3 import departures, arrivals, config


class Stage0Tests(unittest.TestCase):
    def test_loader_preserves_raw_values_and_excludes_hidden_block(self):
        raw = departures(["2025-01-01", "2025-01-02", "2025-01-03"], [-12, 0, 90000])
        raw = raw.rename(columns={"mvt_id": "MVT_ID_mvt", "taxi_sec": "TAXITIME_SEC_mvt"})
        raw["PHASE_mvt"] = "DEP"
        raw["BLOCK_TIME_UTC_mvt"] = raw.MVT_TIME_UTC_mvt - pd.Timedelta(hours=24)
        for name in RECOVERY_NAMES:
            raw[name] = None
        raw["FLIGHT_ID_mvt"] = [1.0, None, 3.0]
        raw["ADEP_flt"] = ["LIRF", None, None]
        with tempfile.TemporaryDirectory() as temp, duckdb.connect() as con:
            path = Path(temp) / "data.parquet"
            raw.to_parquet(path)
            loaded = load_phase(con, str(path), "DEP")
            np.testing.assert_array_equal(loaded.taxi_sec, [-12, 0, 90000])
            np.testing.assert_array_equal(loaded.mvt_id, [0, 1, 2])
            self.assertEqual(loaded.raw_flight_id_missing.tolist(), [False, True, False])
            self.assertEqual(loaded.raw_plan_block_missing.tolist(), [False, True, True])
            self.assertNotIn("BLOCK_TIME_UTC_mvt", loaded)
            self.assertEqual(modeling.observable(loaded).columns.tolist(), RAW_NAMES)
            expected = con.sql("SELECT " + ",".join(RAW_NAMES) + " FROM read_parquet(?)", params=[str(path)]).df()
            pd.testing.assert_frame_equal(modeling.observable(loaded), expected)

    def test_checkpoint_predictions_and_loss_records_match_saved_model(self):
        dep = departures(pd.date_range("2025-01-01", "2025-07-31 12:00", freq="12h"))
        arr = arrivals(dep.MVT_TIME_UTC_mvt - pd.Timedelta(hours=1))
        args = harness.parse_args(["--folds", "A1", "--rounds", "3", "--checkpoints", "2"])
        cfg = config()
        cfg.update(rounds=3, checkpoints=[2], seeds=[42])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            _, windows = harness.plan(args)
            folds, models = harness.execute(dep, arr, windows, cfg, Path(temp))
            frame = pd.read_parquet(Path(temp) / folds["A1"]["predictions"]["42"])
            self.assertTrue(np.isfinite(frame[["pred", "pred_round_2"]]).all().all())
            details = models["M1"]["seeds"]["42"]
            curve = json.loads((Path(temp) / details["training_loss"]).read_text())["rmse"]
            self.assertEqual(len(curve), 3)
            record = {"config": cfg, "folds": folds, "prediction_column": "pred_round_2"}
            loss = reporting.load_losses(record, temp)["A1"]
            ordered = frame.sort_values("mvt_id")
            np.testing.assert_allclose(loss.sq[0], (ordered.y-ordered.pred_round_2)**2)
            self.assertEqual(len(frame), 62)
            resumed_progress = {"models": models, "folds": {}}
            with patch.object(harness, "fit_booster", side_effect=AssertionError("must reuse saved fit")):
                resumed_folds, _ = harness.execute(dep, arr, windows, cfg, Path(temp),
                                                   resume_progress=resumed_progress)
            resumed = pd.read_parquet(Path(temp) / resumed_folds["A1"]["predictions"]["42"])
            np.testing.assert_allclose(frame.pred, resumed.pred, rtol=0, atol=1e-10)
            np.testing.assert_allclose(frame.pred_round_2, resumed.pred_round_2, rtol=0, atol=1e-10)

    def test_parallel_seed_fits_preserve_predictions_and_budget(self):
        dep = departures(pd.date_range("2025-01-01", "2025-07-31 12:00", freq="12h"))
        arr = arrivals(dep.MVT_TIME_UTC_mvt - pd.Timedelta(hours=1))
        args = harness.parse_args(["--folds", "A1", "--fit-workers", "2", "--threads", "4"])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            _, windows = harness.plan(args)
            sequential = config()
            sequential.update(threads=4, seeds=[42, 142, 242])
            parallel = dict(sequential, fit_workers=2)
            paths = [Path(temp) / "sequential", Path(temp) / "parallel"]
            outcomes = [harness.execute(dep, arr, windows, cfg, path)
                        for cfg, path in zip([sequential, parallel], paths)]
            for seed in sequential["seeds"]:
                a = pd.read_parquet(paths[0] / outcomes[0][0]["A1"]["predictions"][str(seed)])
                b = pd.read_parquet(paths[1] / outcomes[1][0]["A1"]["predictions"][str(seed)])
                np.testing.assert_allclose(a.pred, b.pred, rtol=0, atol=1e-10)
            details = outcomes[1][1]["M1"]["seeds"]
            self.assertEqual([details[str(seed)]["fit_threads"] for seed in [42, 142, 242]], [2, 2, 4])


if __name__ == "__main__":
    unittest.main()
