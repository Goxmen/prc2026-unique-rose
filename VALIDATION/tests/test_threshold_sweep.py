"""Check direct-submission bookkeeping with simulated training and MinIO."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modeling
from tools import sweep_taxi_thresholds as sweep
from records import MODEL_FILES, write_json


class ThresholdSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.submissions = self.root / "submissions"
        self.submissions.mkdir()
        self.manifests = self.root / "sweeps"
        self.manifests.mkdir()
        for name, value in [("SUBMISSIONS", self.submissions), ("SWEEPS", self.manifests)]:
            patcher = patch.object(sweep, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.remote = {}
        self.fits = []
        self.uploads = []
        self.fail_after_upload = False

    def recipe(self):
        return {
            "sweep_id": "test", "seed": 42,
            "config": {key: value for key, value in sweep.DEFAULTS.items()
                       if key != "max_taxi_sec"},
            "source": {name: "source-hash" for name in MODEL_FILES},
            "submissions": sweep.plan({"unique-rose_v4.parquet"}),
        }

    def fake_fit(self, command, **kwargs):
        self.assertNotIn("--run-id", command)
        self.assertEqual(command[command.index("--early-stopping-rounds") + 1], "0")
        self.assertEqual(command[command.index("--rounds") + 1], "800")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertNotIn("--fpl-flag", command)
        self.assertIn("--track-training-loss", command)
        version = int(command[command.index("--version") + 1])
        threshold = int(command[command.index("--max-taxi-sec") + 1])
        self.fits.append((version, threshold))
        path = self.submissions / f"unique-rose_v{version}.parquet"
        # Contents are irrelevant to orchestration; submit.fill_template has real tests.
        path.write_bytes(f"predictions for {threshold}".encode())
        path.with_suffix(".model.txt").write_text("model")
        write_json(path.with_suffix(".training_loss.json"), {
            "file": path.name, "max_taxi_sec": threshold, "seed": 42,
            "rmse": [500 - i / 4 for i in range(800)],
        })
        write_json(path.with_suffix(".meta.json"), {
            "config": dict(self.recipe()["config"], max_taxi_sec=threshold), "seed": 42,
            "selected_rounds": 800, "inner_selection": None, "harness_run_id": None,
            "full_benchmark_validated": False, "freeze": "2025-12-31",
            "model_source": {name: "source-hash" for name in MODEL_FILES},
            "train_rows": 2_084_000,
        })

    def fake_mc(self, operation, *args, binary=False):
        if operation == "cp":
            self.assertEqual(args[0], "--attr")
            self.assertIn("max-taxi-sec=", args[1])
            path, remote_path = Path(args[2]), args[3]
            self.assertNotIn(path.name, self.remote)
            self.assertEqual(remote_path, sweep.BUCKET + path.name)
            self.remote[path.name] = path.read_bytes()
            self.uploads.append(path.name)
            if self.fail_after_upload:
                self.fail_after_upload = False
                raise RuntimeError("connection lost after successful upload")
            return ""
        self.assertEqual(operation, "cat")
        value = self.remote[args[0].removeprefix(sweep.BUCKET)]
        return value if binary else value.decode()

    def run_sweep(self, record, manifest):
        with patch.object(sweep, "remote_files", side_effect=lambda: set(self.remote)), \
             patch.object(sweep, "mc", side_effect=self.fake_mc), \
             patch.object(sweep.subprocess, "run", side_effect=self.fake_fit), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sweep.execute(record, manifest)

    def test_id_allocation_accounts_for_remote_local_and_reserved_ids(self):
        (self.submissions / "unique-rose_v8.model.txt").touch()
        write_json(self.manifests / "older.json", {
            "submissions": [{"file": "unique-rose_v12.parquet"}]})
        items = sweep.plan({"unique-rose_v9.parquet_result.json"})
        self.assertEqual([item["version"] for item in items], list(range(13, 20)))
        self.assertEqual([item["max_taxi_sec"] for item in items], list(sweep.THRESHOLDS))

    def test_complete_sweep_then_resume_never_retrains_or_reuploads(self):
        record = self.recipe()
        manifest = self.manifests / "test.json"
        self.run_sweep(record, manifest)
        self.assertEqual(self.fits, list(zip(range(5, 12), sweep.THRESHOLDS)))
        self.assertEqual(len(self.uploads), 7)
        self.assertGreater(manifest.with_name("test_training_loss.png").stat().st_size, 0)
        saved = json.loads(manifest.read_text())
        self.assertTrue(all(item["state"] == "uploaded" for item in saved["submissions"]))
        self.run_sweep(saved, manifest)
        self.assertEqual(len(self.fits), 7)
        self.assertEqual(len(self.uploads), 7)

    def test_resume_after_ambiguous_upload_uses_hash_and_saved_mapping(self):
        record = self.recipe()
        manifest = self.manifests / "test.json"
        self.fail_after_upload = True
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            self.run_sweep(record, manifest)
        saved = json.loads(manifest.read_text())
        self.assertEqual(saved["submissions"][0]["state"], "trained")
        self.run_sweep(saved, manifest)
        self.assertEqual(len(self.fits), 7)
        self.assertEqual(len(self.uploads), 7)

    def test_remote_collision_is_never_adopted_or_overwritten(self):
        record = self.recipe()
        self.remote["unique-rose_v5.parquet"] = b"someone else's submission"
        with self.assertRaisesRegex(ValueError, "already used remotely"):
            self.run_sweep(record, self.manifests / "test.json")
        self.assertEqual(self.fits, [])
        self.assertEqual(self.uploads, [])

    def test_score_refresh_matches_filenames_and_preserves_pending(self):
        record = self.recipe()
        for item in record["submissions"]:
            item["state"] = "uploaded"
        self.remote["unique-rose_v7.parquet_result.json"] = json.dumps({
            "status": "Succeeded", "score": 390.25, "used_pairs": 344841}).encode()
        self.remote["unique-rose_v5.parquet_result.json"] = json.dumps({
            "status": "Succeeded", "score": 410.75, "used_pairs": 344841}).encode()
        with patch.object(sweep, "mc", side_effect=self.fake_mc):
            sweep.refresh_scores(record, set(self.remote))
        self.assertEqual(record["submissions"][0]["score"], 410.75)
        self.assertIsNone(record["submissions"][1]["score"])
        self.assertEqual(record["submissions"][2]["max_taxi_sec"], 22_000)
        self.assertEqual(record["submissions"][2]["score"], 390.25)

    def test_artifact_with_different_threshold_is_rejected(self):
        record = self.recipe()
        item = record["submissions"][0]
        self.fake_fit(sweep.training_command(record, item))
        wrong = copy.deepcopy(item)
        wrong["max_taxi_sec"] = 90_000
        with self.assertRaisesRegex(ValueError, "mismatched"):
            sweep.check_artifact(record, wrong)


class TrainingLossTests(unittest.TestCase):
    def test_recorded_loss_matches_predictions_without_changing_the_fit(self):
        x = np.linspace(0, 1, 1000)
        features = pd.DataFrame({name: pd.Categorical(["same"] * len(x))
                                 for name in modeling.CATEGORICAL})
        features["signal"] = x
        target = 500 + 1000 * x
        config = {"threads": 1}
        history = {}
        with contextlib.redirect_stderr(io.StringIO()):
            plain = modeling.fit_booster(features, target, 42, 6, config, "plain")
            recorded = modeling.fit_booster(features, target, 42, 6, config, "recorded",
                                             training_history=history)
        np.testing.assert_array_equal(plain.predict(features), recorded.predict(features))
        self.assertEqual(set(history), {"training"})
        self.assertEqual(len(history["training"]["rmse"]), 6)
        expected = [np.sqrt(np.mean((recorded.predict(features, num_iteration=i) - target) ** 2))
                    for i in range(1, 7)]
        np.testing.assert_allclose(history["training"]["rmse"], expected)


if __name__ == "__main__":
    unittest.main()
