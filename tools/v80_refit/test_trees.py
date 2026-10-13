"""Small contract tests; no real data preparation, model fitting or inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("v80_trees_tested", Path(__file__).with_name("trees.py"))
trees = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trees)


class TreeContracts(unittest.TestCase):
    def test_recorded_caps_rounds(self):
        caps = {2: 7200, 3: 200000, 5: 10000, 6: 16000, 7: 22000, 9: 50000}
        for v, cap in caps.items():
            c = trees.configuration(v, 4)
            self.assertEqual(c["max_taxi_sec"], cap)
            self.assertEqual(c["rounds"], 3000 if v == 2 else 800)
            self.assertEqual(c["seed"], 42)
        self.assertEqual(set(trees.VERSIONS), {2, 3, 5, 6, 7, 9, 26, 27, 28, 29, 43, 44})

    def test_categories_mask_unseen_and_preserve_order(self):
        X = pd.DataFrame({"a": ["new", "B", "A"], "n": [1, 2, 3]})
        got = trees.apply_categories(X, {"a": ["B", "A"]})
        self.assertEqual(got.a.cat.categories.tolist(), ["B", "A"])
        self.assertTrue(pd.isna(got.a.iloc[0]))
        self.assertEqual(got.a.cat.codes.tolist(), [-1, 0, 1])
        self.assertEqual(X.a.tolist(), ["new", "B", "A"])

    def test_integer_serialization_order_and_clipping(self):
        f = {2: [pd.DataFrame({"mvt_id": [30, 10, 20], "pred": [-8, 2.5, 3.5]})]}
        got = trees.integer_frame([10, 20, 30], f, [2])
        self.assertEqual(got.mvt_id.tolist(), [10, 20, 30])
        self.assertEqual(got.v2.tolist(), [2, 4, 0])
        self.assertTrue(pd.api.types.is_integer_dtype(got.v2))

    def test_retained_model_keeps_original_codes_and_unused_levels(self):
        current = pd.DataFrame({"cat": pd.Categorical(["B", "A", None], categories=["B", "A"]),
                                "numeric": [1., 2., 3.]}, index=[20, 10, 30])
        saved = current.copy()
        saved["cat"] = saved.cat.cat.set_categories(["A", "unused", "B"])
        retained = {"X": saved, "y": np.array([5., 6., 7.])}
        got = trees.retained_training_frame(current, retained["y"], retained)
        self.assertIs(got, saved)
        self.assertEqual(got.cat.cat.codes.tolist(), [2, 0, -1])
        self.assertEqual(current.cat.cat.codes.tolist(), [0, 1, -1])

    def test_retained_encoding_cannot_hide_changed_values_rows_or_targets(self):
        X = pd.DataFrame({"cat": pd.Categorical(["A", "B", None]), "n": [1., 2., 3.]})
        for change in ("category", "missing", "numeric", "index", "target", "dtype", "columns"):
            with self.subTest(change=change):
                saved, y = X.copy(), np.array([4., 5., 6.])
                if change == "category": saved.loc[0, "cat"] = "B"
                if change == "missing": saved.loc[2, "cat"] = "A"
                if change == "numeric": saved.loc[0, "n"] += 1e-9
                if change == "index": saved = saved.iloc[::-1]
                if change == "target": y[0] += 1
                if change == "dtype": saved["cat"] = saved.cat.astype(object)
                if change == "columns": saved = saved[["n", "cat"]]
                with self.assertRaises((AssertionError, ValueError)):
                    trees.retained_training_frame(X, [4., 5., 6.], {"X": saved, "y": y})

    def test_wrong_retained_model_mapping_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trees.save(root / "identity.json", {"test": True})
            source = root / "retained.txt"
            source.write_text("fake model")
            X = pd.DataFrame({"cat": pd.Categorical(["A", "B"], categories=["A", "B"])})
            booster = Mock(pandas_categorical=[["B", "A"]])
            with patch.object(trees.lgb, "Booster", return_value=booster), \
                 patch.object(trees.lgb, "train") as fit:
                with self.assertRaisesRegex(ValueError, "category mapping"):
                    trees.fit_saved(root, "v5", X, [1., 2.], {"num_threads": 1}, 800, [], trees.artifact(source))
                fit.assert_not_called()
            self.assertFalse((root / "models/v5.txt").exists())

    def test_bad_coverage_and_nan_rejected(self):
        for ids, pred in [([1, 1], [2., 3.]), ([1, 3], [2., 3.]), ([1, 2], [2., float("nan")])]:
            with self.assertRaises(ValueError):
                trees.integer_frame([1, 2], {2: [pd.DataFrame({"mvt_id": ids, "pred": pred})]}, [2])
        with self.assertRaises(ValueError):
            trees.integer_frame([1, 1], {}, [])

    def test_artifact_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "data"
            p.write_text("first")
            rec = trees.artifact(p)
            self.assertEqual(trees.verify(rec), p)
            p.write_text("second")
            with self.assertRaises(ValueError):
                trees.verify(rec)

    def test_unrecorded_saved_model_never_refitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trees.save(root / "identity.json", {"test": True})
            (root / "models").mkdir()
            (root / "models/one.txt").write_text("retain me")
            with patch.object(trees.lgb, "train") as fit:
                with self.assertRaisesRegex(ValueError, "never refit silently"):
                    trees.fit_saved(root, "one", pd.DataFrame({"x": [0]}), [1], {"num_threads": 1}, 2, [])
                fit.assert_not_called()

    def test_reload_checks_every_row(self):
        class Model:
            def __init__(self, mismatch=False):
                self.mismatch = mismatch
            def predict(self, x, num_threads):
                arr = x.x.to_numpy(float).copy()
                if self.mismatch and len(arr) and arr[-1] > 20000:
                    arr[-1] += 1
                return arr
        X = pd.DataFrame({"x": np.arange(20003)})
        np.testing.assert_array_equal(trees.predictions_equal(Model(), Model(), X, 1), X.x)
        with self.assertRaises(AssertionError):
            trees.predictions_equal(Model(), Model(True), X, 1)

    def test_plan_does_not_fit_or_read_large_packets(self):
        with patch.object(trees.pd, "read_pickle", side_effect=AssertionError("no packet read")), \
             patch.object(trees.lgb, "train", side_effect=AssertionError("no fit")):
            result = trees.plan()
        self.assertTrue(result["no_fit_or_inference"])
        self.assertTrue(all(p["exists"] for p in result["input_paths"]))


if __name__ == "__main__":
    unittest.main()
