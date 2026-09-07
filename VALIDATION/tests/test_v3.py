"""Regression checks for leakage boundaries, weighting, and the v3 workflow."""

import argparse
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness
import modeling as m
import protocol
import reporting as r
import submit
from features_time import day_index
from log_official import calibration_key
from records import MODEL_FILES, ROOT, sha256


def departures(times, targets=None):
    t = pd.Series(pd.to_datetime(times, utc=True, format="mixed"))
    frame = pd.DataFrame({c: ["x"] * len(t) for c in m.RAW_NAMES})
    for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]:
        frame[c] = t if c == "MVT_TIME_UTC_mvt" else t - pd.Timedelta(minutes=20)
    frame["ADEP_mvt"], frame["ADES_mvt"] = "LIRF", "EGLL"
    frame["RUNWAY_mvt"], frame["STAND_mvt"] = "09", "1"
    frame["taxi_sec"] = targets if targets is not None else np.arange(len(t)) % 120 + 700
    frame["mvt_id"] = np.arange(len(t), dtype=float)
    return frame


def arrivals(times, values=None):
    return pd.DataFrame({"MVT_TIME_UTC_mvt": pd.to_datetime(times, utc=True, format="mixed"),
                         "ADES_mvt": "LIRF", "taxi_sec": 400 if values is None else values})


def config(**changes):
    return dict(max_taxi_sec=200000, lag_train_days=0, fpl_flag=False,
                rounds=2, early_stopping_rounds=0, threads=2, seeds=[42, 142],
                sample=1.0, train_months=6, bootstrap_reps=40, bootstrap_seed=7,
                block_days=1, params=m.PARAMS, **changes)


def loss(month, ids, errors):
    n = len(ids)
    meta = pd.DataFrame({"mvt_time": pd.to_datetime([month + "-01"] * n, utc=True),
                         "airport": ["LIRF"] * n, "y": np.ones(n) * 100,
                         "has_flight_plan": [True] * n, "arrival_history_available": [True] * n,
                         "arrival_history_days_available": [7] * n},
                        index=pd.Index(ids, name="mvt_id"))
    errors = np.asarray(errors, float)
    return r.Loss(meta, errors ** 2, tuple(42 + 100*i for i in range(len(errors))))


class ProtocolTests(unittest.TestCase):
    def test_default_windows_and_specials(self):
        selected, skipped = protocol.select_folds()
        self.assertFalse(skipped)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({protocol.window_key(f, s) for f, s in selected}), 6)
        for fold, start in selected:
            end = pd.Timestamp(fold.train_end)
            self.assertEqual(end.to_period("M").ordinal - start.to_period("M").ordinal, 5)
        special, _ = protocol.select_folds("G7,F1,W1", months=4)
        self.assertEqual([str(start.date()) for _, start in special],
                         ["2025-01-01", "2025-01-01", "2025-02-01"])
        self.assertFalse(special[-1][0].departure_history)

    def test_legacy_overrides_and_unknown_fold(self):
        selected, _ = protocol.select_folds("jul_fresh,jul_stale,apr", months=0)
        self.assertEqual({f.name for f, _ in selected}, {"A1", "S1", "apr"})
        self.assertTrue(all(str(start.date()) == "2025-01-01" for _, start in selected))
        selected, skipped = protocol.select_folds("apr,A1", months=6)
        self.assertEqual(skipped, ["apr"])
        self.assertEqual([f.name for f, _ in selected], ["A1"])
        with self.assertRaisesRegex(ValueError, "unknown fold"):
            protocol.select_folds("typo")


class AvailabilityTests(unittest.TestCase):
    def test_causal_labels_and_window_lower_bound(self):
        dep = departures(["2025-01-31 12:00", "2025-02-01 12:00",
                          "2025-02-02 12:00", "2025-02-02 12:10"], [900, 100, 200, 6000])
        arr = arrivals(["2025-01-31", "2025-02-01"])
        td, ta = m.period_rows(dep, "2025-02-01", "2025-02-28"), m.period_rows(arr, "2025-02-01", "2025-02-28")
        X, _, _ = m.training_matrix(td, ta, config())
        self.assertTrue(np.isnan(X.op_taxi_30d.iloc[0]))
        np.testing.assert_array_equal(X.op_taxi_30d.iloc[1:].to_numpy(), [100, 100])
        np.testing.assert_array_equal(X.op_taxi_30d_n.to_numpy(), [0, 1, 1])

    def test_gap_is_hidden_and_target_does_not_select_events(self):
        train = departures(["2025-06-29 12:00", "2025-06-30 12:00"], [100, 300])
        val = departures(["2025-12-01 12:00", "2025-12-01 12:20"], [99000, 500])
        all_arr = arrivals(["2025-06-30 10:00", "2025-11-30 10:00", "2025-12-01 10:00"], [300, 7000, 400])
        ta = m.month_rows(all_arr, "2025-06")
        va = m.month_rows(all_arr, "2025-12")
        _, _, cats = m.training_matrix(train, ta, config())
        X = m.validation_matrix(train, ta, val, va, "2025-06-30", cats, config())
        self.assertTrue(X.arr_taxiin_7d.isna().all())
        self.assertEqual(X.dep_actual_30m_before.iloc[1], 1)
        np.testing.assert_array_equal(X.op_taxi_30d.to_numpy(), [200, 200])
        changed = val.copy()
        changed["taxi_sec"] = [1, 1e9]
        changed["BLOCK_TIME_UTC_mvt"] = pd.Timestamp("1900-01-01")
        Y = m.validation_matrix(train, ta, changed, va, "2025-06-30", cats, config())
        pd.testing.assert_frame_equal(X, Y)

    def test_unknown_categories_and_winter_history_exclusion(self):
        train = departures(["2025-02-01", "2025-02-02"])
        arr = arrivals(["2025-02-01"])
        X, _, cats = m.training_matrix(train, arr, config(), departure_history=False)
        self.assertFalse(set(m.HISTORY_FEATURES) & set(X))
        val = departures(["2025-01-02"])
        val["AIRCRAFT_OPERATOR_flt"] = "unseen"
        Y = m.validation_matrix(train, arr, val, arrivals(["2025-01-01"]),
                                "2025-12-31", cats, config(), departure_history=False)
        self.assertTrue(Y.operator.isna().all())
        self.assertFalse(set(m.HISTORY_FEATURES) & set(Y))

    def test_inner_round_selection_never_uses_outer_month(self):
        dep = departures([f"2025-{month:02}-15 12:00" for month in range(1, 7)])
        arr = arrivals([f"2025-{month:02}-14" for month in range(1, 7)])
        cfg = config()
        cfg["early_stopping_rounds"] = 3
        calls = []
        def fit(X, y, seed, rounds, config, desc, validation=None):
            calls.append((X.month.tolist(), validation[0].month.tolist()))
            return argparse.Namespace(best_iteration=2)
        with patch.object(m, "fit_booster", fit):
            rounds, detail = m.choose_rounds(dep, arr, 42, cfg, "2025-06-30")
        self.assertEqual(calls, [([1, 2, 3, 4, 5], [6])])
        self.assertEqual(rounds, 2)
        self.assertEqual(detail["train_end"], "2025-05-31")


class ScoringTests(unittest.TestCase):
    def test_december_and_seeds_have_one_weight_per_flight(self):
        losses = {
            "B3": loss("2025-11", [1], [[1], [3]]),
            "B4": loss("2025-12", [2], [[2], [4]]),
            "B7": loss("2025-12", [2], [[6], [8]]),
        }
        group = r.group_loss(losses, ["B3", "B4", "B7"])
        self.assertEqual(len(group.meta), 2)
        self.assertAlmostEqual(r.rmse(group), np.sqrt((5 + 30) / 2))
        opposite = loss("2025-11", [1], [[3], [-3]])
        self.assertEqual(r.rmse(opposite), 3)

    def test_pooling_uses_flights_not_mean_monthly_rmse(self):
        group = r.group_loss({"x": loss("2025-07", [1, 2], [[3, 3]]),
                              "y": loss("2025-08", [3], [[0]])}, ["x", "y"])
        self.assertAlmostEqual(r.rmse(group), np.sqrt(6))

    def test_bootstrap_is_shared_and_reproducible(self):
        one = loss("2025-12", [1], [[2], [4]])
        losses = {"B4": one, "B7": one}
        calendar, weights = r.bootstrap_calendar(losses, 100, 11)
        calendar2, weights2 = r.bootstrap_calendar(losses, 100, 11)
        self.assertTrue(calendar.equals(calendar2))
        np.testing.assert_array_equal(weights, weights2)
        np.testing.assert_array_equal(weights.sum(axis=1), np.repeat(31, 100))
        with np.errstate(invalid="ignore"):
            diff = r.boot_scores(one, calendar, weights) - r.boot_scores(one, calendar, weights)
        result = r.interval(0, diff)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertEqual(result["decision"], "inconclusive")
        for blocks in [2, 7]:
            _, block_weights = r.bootstrap_calendar(losses, 100, 11, blocks)
            np.testing.assert_array_equal(block_weights.sum(axis=1), np.repeat(31, 100))

    def test_alignment_tail_and_partial_scores(self):
        first = loss("2025-12", [1], [[5]])
        other = loss("2025-12", [2], [[5]])
        with self.assertRaisesRegex(ValueError, "align"):
            r.group_loss({"x": first, "y": other}, ["x", "y"])
        first.meta["y"] = 90000
        stats = r.summarize(first)
        self.assertEqual(stats["tail"]["pct_sse"], 100)
        self.assertIsNone(stats["ordinary"]["rmse"])
        self.assertIn("R_A_partial", r.score_groups({"A6": first}))
        self.assertNotIn("R_A", r.score_groups({"A6": first}))


class EndToEndTests(unittest.TestCase):
    def test_all_folds_share_fits_save_every_seed_and_compare(self):
        times = pd.date_range("2025-01-01 12:00", "2025-12-31 12:00", freq="D")
        dep = departures(times)
        dep.loc[dep.index % 31 == 0, "taxi_sec"] = 90000
        dep.loc[dep.index % 31 == 0, "AOBT_3_flt"] = pd.NaT
        arr = arrivals(times - pd.Timedelta(hours=1))
        args = harness.parse_args(["--folds", "A,B,G7,F1,W1,S1"])
        with contextlib.redirect_stdout(io.StringIO()):
            _, windows = harness.plan(args)
        cfg = config()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch.object(harness, "fit_booster", wraps=m.fit_booster) as fitter:
                folds, models = harness.execute(dep, arr, windows, cfg, Path(temp))
            self.assertEqual(len(models), 9)
            self.assertEqual(fitter.call_count, 18)
            self.assertEqual(len(folds), 20)
            self.assertEqual(folds["A1"]["model_id"], folds["B4"]["model_id"])
            self.assertEqual(folds["A1"]["model_id"], folds["S1"]["model_id"])
            self.assertFalse(set(m.HISTORY_FEATURES) & set(models[folds["W1"]["model_id"]]["features"]))
            for info in folds.values():
                self.assertEqual(len(info["predictions"]), 2)
            b4 = pd.read_parquet(Path(temp) / folds["B4"]["predictions"]["42"])
            a6 = pd.read_parquet(Path(temp) / folds["A6"]["predictions"]["42"])
            self.assertFalse(b4.arrival_history_available.iloc[0])
            self.assertTrue(a6.arrival_history_available.iloc[0])
            self.assertEqual(b4.arrival_history_days_available.iloc[0], 0)
            self.assertEqual(a6.arrival_history_days_available.iloc[0], 7)
            record = {"run_id": "fixture", "protocol": protocol.VERSION,
                      "score_version": protocol.SCORE_VERSION, "config": cfg,
                      "folds": folds, "data_identity": [], "ranking_weights": {"jan_fraction": .4429}}
            result = r.report(record, temp, (copy.deepcopy(record), temp))
            self.assertIn("R_combined", result["scores"])
            self.assertNotIn("G7", result["groups"])
            self.assertEqual(len(result["aging"]), 11)
            for change in result["comparison"]["scores"].values():
                self.assertEqual(change["difference"], 0)
                self.assertEqual(change["ci95"], [0, 0])
            bad = copy.deepcopy(record)
            bad["protocol"] = "legacy"
            with self.assertRaisesRegex(ValueError, "incompatible protocol"):
                r.report(record, temp, (bad, temp))

    def test_submission_template_and_configuration(self):
        template = pa.table({"MVT_ID_mvt": pa.array([20., 10.]),
                             "TAXITIME_SEC_mvt": pa.array([None, None], type=pa.int32())})
        out = submit.fill_template(template, [10., 20.], [12.7, -3])
        self.assertTrue(out.schema.equals(template.schema))
        self.assertEqual(out["MVT_ID_mvt"].to_pylist(), [20., 10.])
        self.assertEqual(out["TAXITIME_SEC_mvt"].to_pylist(), [0, 13])
        for ids, preds in [([10., 10.], [1, 2]), ([10.], [1]), ([10., 20.], [np.nan, 1])]:
            with self.assertRaises(ValueError):
                submit.fill_template(template, ids, preds)
        args = argparse.Namespace(run_id="x", seed=None, **dict.fromkeys(submit.DEFAULTS))
        ref = {"run_id": "x", "protocol": "v3", "status": "complete",
               "config": config(), "source": {"files": {name: sha256(ROOT / name) for name in MODEL_FILES}}}
        with patch.object(submit, "read_run", return_value=(ref, Path("."))):
            cfg, seed, _ = submit.resolve_config(args)
            self.assertEqual(cfg["rounds"], 2)
            self.assertEqual(seed, 42)
            args.rounds = 100
            with self.assertRaisesRegex(ValueError, "differs"):
                submit.resolve_config(args)

    def test_official_results_do_not_mix_protocols(self):
        legacy = {"derived": {"pooled_projection_s": 10}}
        modern = {"protocol": "v3", "score_version": protocol.SCORE_VERSION,
                  "full_benchmark": True, "status": "complete", "config": config()}
        self.assertNotEqual(calibration_key(legacy), calibration_key(modern))
        modern["full_benchmark"] = False
        self.assertIsNone(calibration_key(modern))


if __name__ == "__main__":
    unittest.main()

