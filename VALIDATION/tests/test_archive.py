"""Run lookup and bookkeeping survive the validation-folder cleanup."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import read_run, source_snapshot, write_json
from tools import log_official


class ArchiveTests(unittest.TestCase):
    def test_old_ids_and_paths_resolve_and_current_run_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = root / "runs"
            archive = root / "archive/runs/old"
            runs.mkdir()
            write_json(archive / "result.json", {"run_id": "old", "location": "archive"})
            for request in ["old", runs / "old", runs / "old/result.json", archive]:
                record, directory = read_run(request, runs=runs)
                self.assertEqual(record["location"], "archive")
                self.assertEqual(directory, archive)
            write_json(runs / "old/result.json", {"run_id": "old", "location": "current"})
            record, directory = read_run("old", runs=runs)
            self.assertEqual(record["location"], "current")
            self.assertEqual(directory, runs / "old")
            with self.assertRaises(FileNotFoundError):
                read_run("missing", runs=runs)

    def test_official_score_updates_archived_detail_and_central_ledger(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            log = root / "runs/experiments.jsonl"
            log.parent.mkdir()
            row = {"run_id": "old", "at": "2026-09-01", "derived": {"pooled_projection_s": 400}}
            log.write_text(json.dumps(row) + "\n")
            detail = root / "archive/runs/old/result.json"
            write_json(detail, row)
            log_official.main(["--run", "old", "--official", "350.5",
                               "--submission", "unique-rose_v3.parquet", "--log", str(log)])
            self.assertEqual(json.loads(detail.read_text())["official_rmse"], 350.5)
            self.assertEqual(json.loads(log.read_text())["official_rmse"], 350.5)

    def test_snapshot_includes_relocated_active_tools_and_docs(self):
        with tempfile.TemporaryDirectory() as temp:
            files = source_snapshot(Path(temp))["files"]
            self.assertIn("VALIDATION/tools/log_official.py", files)
            self.assertIn("VALIDATION/docs/PROTOCOL.md", files)
            self.assertIn("VALIDATION/reports/STAGE0.md", files)
            self.assertFalse(any(name.startswith("VALIDATION/archive/") for name in files))
            self.assertNotIn("VALIDATION/benchmark_threads.py", files)


if __name__ == "__main__":
    unittest.main()
