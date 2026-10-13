"""Tiny release checks; no competition data or training is accessed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("reproduce_final", Path(__file__).with_name("reproduce_final.py"))
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class FinalReplayTests(unittest.TestCase):
    def test_constraint_order_and_rounding(self):
        ids = np.arange(1, 9)
        raw = pd.DataFrame(dict(mvt_id=ids, has_id=[True] + [False] * 7,
                                nm_gap=[5000.] + [np.nan] * 7))
        matches = pd.DataFrame(dict(mvt_id=[2, 3, 4, 5], accepted=[True] * 4,
                                    recovered_gap=[4000., 2000., 4000., 4000.]))
        values = np.array([1000., 4000., 1000., 7606., 394., 1.5, 2.5, -5.])
        actual = release.constrain(ids, raw, values, matches)
        np.testing.assert_array_equal(actual, [1394, 394, 5606, 7606, 394, 2, 2, 0])
        np.testing.assert_array_equal(release.constrain(ids, raw, actual, matches), actual)

    def test_alignment_reorders_but_rejects_bad_ids(self):
        data = pd.DataFrame(dict(mvt_id=[2, 1], submitted=[20, 10]))
        np.testing.assert_array_equal(release.align(data, [1, 2]).submitted, [10, 20])
        for ids in ([1, 3], [1]):
            with self.assertRaises(ValueError):
                release.align(data, ids)
        with self.assertRaises(ValueError):
            release.align(pd.DataFrame(dict(mvt_id=[1, 1])), [1, 2])

    def test_hash_corruption_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "packet.json"
            path.write_text("original")
            item = release.record(path, root)
            self.assertEqual(release.checked(root, item), path)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                release.checked(root, item)
            for unsafe in ("../escape", "/tmp/escape"):
                with self.assertRaises(ValueError):
                    release.safe_path(root, unsafe)

    def test_serialization_refuses_nonfinite_and_overflow(self):
        for values in ([np.nan], [np.inf], [float(2 ** 31)]):
            with self.assertRaises(ValueError):
                release.integer_vector(values)
        np.testing.assert_array_equal(release.integer_vector([-1, 0.5, 1.5]), [0, 0, 2])

    def test_semantic_digest_detects_id_and_prediction_changes(self):
        original = release.values_sha256([1, 2], [10, 20])
        self.assertNotEqual(original, release.values_sha256([2, 1], [10, 20]))
        self.assertNotEqual(original, release.values_sha256([1, 2], [10, 21]))

    def test_published_manifest_pin_is_checked_first(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest hash differs"):
                release.verify(folder, "0" * 64)

    def test_incomplete_manifest_cannot_omit_required_packets(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = dict(format="prc-v83-component-replay-v1", rows=release.ROWS,
                            versions=release.VERSIONS, accepted_final_sha256=release.FINAL_SHA,
                            expected_values_sha256=release.VALUES_SHA, files=[])
            (Path(folder) / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Required packet missing"):
                release.verify(folder)


if __name__ == "__main__":
    unittest.main()
