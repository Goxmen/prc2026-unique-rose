"""Synthetic contracts only: no real data, feature preparation or model fitting."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_month_selection as selection


class SplitAndScoreTests(unittest.TestCase):
    def test_four_uniform_cyclic_embargoes(self):
        expected_exclusions = {
            "2025-01": {12, 1, 2}, "2025-02": {1, 2, 3},
            "2025-06": {5, 6, 7}, "2025-07": {6, 7, 8},
        }
        self.assertEqual(set(selection.folds()), set(expected_exclusions))
        for month, excluded in expected_exclusions.items():
            self.assertEqual(selection.folds()[month],
                [f"2025-{m:02d}" for m in range(1, 13) if m not in excluded])
            self.assertEqual(len(selection.folds()[month]), 9)
        self.assertIn("2025-04", selection.folds()["2025-07"])
        self.assertIn("2025-05", selection.folds()["2025-07"])

    def test_ten_fixed_mixtures(self):
        names = list(selection.TREE_COMPONENTS) + list(selection.NEURAL_FAMILIES)
        values = {name: np.array([i, 10+i], dtype=float) for i, name in enumerate(names)}
        mixed = selection.mix_candidates(values)
        self.assertEqual(list(mixed), [f"C{i:02d}" for i in range(1, 11)])
        np.testing.assert_array_equal(mixed["C01"], values["main184"])
        np.testing.assert_array_equal(mixed["C02"], values["airport168"])
        np.testing.assert_array_equal(mixed["C03"], values["cat174"])
        np.testing.assert_allclose(mixed["C04"],
            (values["main184"] + values["airport168"] + values["cat174"]) / 3)
        for candidate, family in [("C05", "calls_ple"), ("C06", "physical_ple"), ("C07", "public_tabm")]:
            np.testing.assert_allclose(mixed[candidate], .71*values[family] + .29*mixed["C04"])
        np.testing.assert_allclose(mixed["C08"],
            .71*(values["calls_ple"] + values["physical_ple"])/2 + .29*mixed["C04"])
        np.testing.assert_allclose(mixed["C09"],
            .71*(values["calls_ple"] + values["physical_ple"] + values["public_tabm"])/3 + .29*mixed["C04"])
        np.testing.assert_allclose(mixed["C10"],
            .71*(values["calls_ple"] + values["public_tabm"])/2 + .29*mixed["C04"])
        for recipe in selection.CANDIDATES.values():
            self.assertAlmostEqual(sum(recipe["weights"].values()), 1.)

    def test_score_uses_every_row(self):
        result = selection.score_arrays([1., 1., 1.], {"C01": [1., 1., 7.]})["C01"]
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["sse"], 36.)
        self.assertAlmostEqual(result["rmse"], np.sqrt(12))

    def test_score_rejects_missing_nonfinite_or_misaligned_rows(self):
        cases = [([], []), ([1., np.nan], [1., 2.]), ([1., 2.], [1.]),
                 ([1., 2.], [1., np.inf]), ([[1., 2.]], [[1., 2.]])]
        for labels, pred in cases:
            with self.subTest(labels=labels, pred=pred), self.assertRaises(ValueError):
                selection.score_arrays(labels, {"C01": pred})

    def test_mean_of_monthly_rmse_is_not_pooled_rmse(self):
        errors, counts = [1., 2., 4., 8.], [1, 2, 3, 4]
        monthly = {}
        for month, error, n in zip(selection.MONTHS, errors, counts):
            monthly[month] = selection.score_arrays(np.zeros(n),
                {c: np.full(n, error) for c in selection.CANDIDATES})
        result = selection.aggregate_metrics(monthly)["C01"]
        self.assertEqual(result["mean4"], 3.75)
        self.assertAlmostEqual(result["pooled4"], np.sqrt(313 / 10))
        self.assertEqual(result["jan_jul"], 4.5)
        self.assertEqual(result["feb_jun"], 3.)
        self.assertEqual(result["n"], 10)
        del monthly["2025-02"]
        with self.assertRaises(ValueError):
            selection.aggregate_metrics(monthly)


class CachedPredictionTests(unittest.TestCase):
    """Exercise corruption/provenance guards without constructing any model."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.month, self.component, self.seed = "2025-01", "main184", 42
        self.ids = np.array([501, 502, 503])
        self.identity = {"path": "synthetic_identity.json", "sha256": "synthetic"}
        self.folder = self.out / "folds" / self.month
        self.folder.mkdir(parents=True)
        inputs = self.folder / "inputs.pkl"
        inputs.write_bytes(b"synthetic input identity only")
        self.input_record = selection.artifact(inputs)
        selection.save_json(self.folder / "input_binding.json",
            dict(identity=self.identity, inputs=self.input_record))
        self.prediction, self.receipt = selection.prediction_paths(
            self.out, self.month, self.component, self.seed)
        self.prediction.parent.mkdir(parents=True)
        pd.DataFrame(dict(mvt_id=self.ids, pred=[10., 20., 30.])).to_parquet(self.prediction, index=False)
        self.checkpoint = self.folder / "checkpoint.bin"
        self.checkpoint.write_bytes(b"synthetic checkpoint, never loaded")
        self.record = dict(identity=self.identity, inputs=self.input_record,
            component=self.component, seed=self.seed,
            prediction=selection.artifact(self.prediction),
            model_receipt={"members": {"EDDF": {"checkpoint": selection.artifact(self.checkpoint)}}})
        self._save()

    def _save(self):
        selection.save_json(self.receipt, self.record)

    def _load(self):
        return selection.load_prediction(self.out, self.month, self.component,
                                         self.seed, self.ids, self.identity)

    def test_valid_prediction_reloads(self):
        np.testing.assert_array_equal(self._load(), [10., 20., 30.])

    def test_changed_run_identity_rejected(self):
        self.record["identity"] = {"path": "other", "sha256": "other"}
        self._save()
        with self.assertRaises(ValueError):
            self._load()

    def test_reordered_ids_rejected_even_when_file_hash_valid(self):
        pd.DataFrame(dict(mvt_id=self.ids[::-1], pred=[10., 20., 30.])).to_parquet(self.prediction, index=False)
        self.record["prediction"] = selection.artifact(self.prediction)
        self._save()
        with self.assertRaises(ValueError):
            self._load()

    def test_other_component_or_seed_rejected(self):
        for key, value in [("component", "main156"), ("seed", 142)]:
            original = copy.deepcopy(self.record)
            self.record[key] = value
            self._save()
            with self.subTest(field=key), self.assertRaises(ValueError):
                self._load()
            self.record = original

    def test_changed_fold_input_binding_rejected(self):
        replacement = self.folder / "other_inputs.pkl"
        replacement.write_bytes(b"different fresh fold")
        selection.save_json(self.folder / "input_binding.json",
            dict(identity=self.identity, inputs=selection.artifact(replacement)))
        with self.assertRaises(ValueError):
            self._load()

    def test_nested_checkpoint_corruption_rejected(self):
        self.checkpoint.write_bytes(b"changed checkpoint")
        with self.assertRaises(ValueError):
            self._load()

    def test_prediction_bytes_corruption_rejected(self):
        self.prediction.write_bytes(b"invalid parquet")
        with self.assertRaises(ValueError):
            self._load()


if __name__ == "__main__":
    unittest.main()
