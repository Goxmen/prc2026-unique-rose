"""Small synthetic assembly contracts; no competition inference or training."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("v80_assemble_tested", Path(__file__).with_name("assemble.py"))
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)


class AssemblyContracts(unittest.TestCase):
    def test_ordered_affine_preserves_negative_weights(self):
        matrix = np.array([[1, 10, 100], [100, 10, 1]], dtype=np.int32)
        weights = [-.25, .75, .5]
        np.testing.assert_array_equal(assembly.affine(matrix, weights), [57.25, -17.])
        np.testing.assert_array_equal(assembly.serial(assembly.affine(matrix, weights)), [57, 0])

    def test_affine_rejects_noninteger_basis_and_wrong_width(self):
        with self.assertRaises(AssertionError):
            assembly.affine([[.5, 2]], [1, 0])
        with self.assertRaises(ValueError):
            assembly.affine([[1, 2]], [1])

    def test_rounding_ties_nonnegative_and_nonfinite(self):
        np.testing.assert_array_equal(assembly.serial([-1.5, -.1, .5, 1.5, 2.5, 3.5]), [0, 0, 0, 2, 2, 4])
        for value in [np.nan, np.inf, -np.inf, float(2**32)]:
            with self.assertRaises(ValueError):
                assembly.serial([value])

    def test_signed_transport_is_not_clipped_separately(self):
        # In particular, a negative replacement delta must lower the parent.
        parent = np.array([100, 0, 50])
        v78 = np.array([30, 20, 0])
        tabm78 = np.array([10, 5, 9])
        robust78 = np.array([35, 10, 0])
        np.testing.assert_array_equal(assembly.serial(parent.astype(float) + (tabm78 - v78) + (robust78 - v78)), [85, 0, 59])

    def test_alignment_reorders_without_substitution(self):
        frame = pd.DataFrame(dict(mvt_id=[3, 1, 2], pred=[30, 10, 20]))
        np.testing.assert_array_equal(assembly.align(frame, [1, 2, 3]).pred, [10, 20, 30])
        for bad in [pd.DataFrame(dict(mvt_id=[1, 1, 2])), pd.DataFrame(dict(mvt_id=[1, 2, 4]))]:
            with self.assertRaises(ValueError):
                assembly.align(bad, [1, 2, 3])

    def test_independent_support_edges_and_ties(self):
        ids = np.arange(8)
        rows = pd.DataFrame(dict(has_id=[True, True, False, False, False, False, False, False],
                                 nm_gap=[0, -5000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
        matches = pd.DataFrame(dict(mvt_id=[2, 3, 4, 5, 6], accepted=[True] * 5,
                                    recovered_gap=[4000, 4000, 4000, 1000, 4000]))
        values = [9000, 10, 394, 7606, 4000, 1000, 5000, -9]
        np.testing.assert_array_equal(assembly.independent_bound(ids, rows, values, matches),
                                      [3606, 0, 394, 7606, 394, 4606, 7606, 0])

    def test_independent_support_matches_original_fixture(self):
        import sys
        sys.path.insert(0, str(assembly.ROOT / "VALIDATION"))
        import public_constraints_candidate as original
        rng = np.random.default_rng(74080)
        ids = np.arange(83)
        rows = pd.DataFrame(dict(has_id=rng.integers(0, 2, len(ids)).astype(bool), nm_gap=rng.normal(4000, 7000, len(ids))))
        rows.loc[[0, 4, 9], "nm_gap"] = np.nan
        matches = pd.DataFrame(dict(mvt_id=ids[::2], accepted=True,
                                    recovered_gap=rng.normal(2000, 4000, len(ids[::2])), donor_mvt_id=ids[::2] + 100))
        values = rng.normal(2000, 5000, len(ids))
        matched = original.matched_bound(rows, values)
        expected, _ = original.exclusion.apply(ids, np.maximum(0, matched), matches)
        np.testing.assert_array_equal(assembly.independent_bound(ids, rows, values, matches), expected)

    def test_day_checkpoint_reuses_and_rejects_changed_prior(self):
        temporary_root = assembly.ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="v80_assembly_test_", dir=temporary_root) as temporary:
            folder = Path(temporary)
            assembly.write(folder / "parameters.json", dict(parameters={"AAAA": {}}))
            source = pd.DataFrame(dict(airport=["AAAA", "AAAA"], day=["2026-01-01", "2026-01-02"], total=[100., 200.], n_public=[3., 3.]))
            def condition(gap, prior, *args, **kwargs):
                return prior + (1 if kwargs else 2), dict(offset=0.)
            function = Mock(side_effect=condition)
            c = types.SimpleNamespace(V72=folder, base=types.SimpleNamespace(public=lambda _: source),
                                      heavy=types.SimpleNamespace(condition=function, GAUSSIAN="gaussian"))
            ids = np.array([7, 8, 9])
            rows = pd.DataFrame(dict(airport=["AAAA"] * 3, day=["2026-01-01", "2026-01-01", "2026-01-02"],
                                      gap=[100., 200., 300.], has_id=[False] * 3, nm_gap=[np.nan] * 3))
            prior = np.array([10, 20, 30])
            identity = {"path": "synthetic", "sha256": "1"}
            first = assembly.daily_posterior(folder, identity, ids, rows, prior, c)
            self.assertEqual(function.call_count, 4)
            second = assembly.daily_posterior(folder, identity, ids, rows, prior, c)
            self.assertEqual(function.call_count, 4)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], [12, 22, 32])
            with self.assertRaises(AssertionError):
                assembly.daily_posterior(folder, identity, ids, rows, prior + 1, c)
            with self.assertRaises(ValueError):
                assembly.daily_posterior(folder, {"path": "synthetic", "sha256": "2"}, ids, rows, prior, c)

    def test_monthly_and_daily_audits_match_original_synthetic_cells(self):
        import sys
        sys.path.insert(0, str(assembly.ROOT / "VALIDATION"))
        import monthly_taxi_reconcile as monthly
        import public_daily_taxi as daily
        airports = list(monthly.ZONES)
        ap = airports[0]
        n = 103
        rows = pd.DataFrame(dict(airport=[ap] * n, local_month=["2026-01"] * n,
                                 local_day=["2026-01-10"] * n, runway=["A"] * n,
                                 eligible=[True] * 101 + [False, True]))
        prediction = np.r_[np.linspace(500, 700, 101), 600., 7300.]
        monthly_source = pd.DataFrame(dict(airport=[ap], local_month=["2026-01"],
                                           mean_sec=[800.], invalid=[.01], noref=[.02]))
        coefficients = list(np.linspace(-3, 7, len(airports) + 2))
        expected, _ = monthly.apply(rows, prediction, monthly_source, dict(coefficients=coefficients))
        np.testing.assert_allclose(assembly.independent_monthly(rows, prediction, monthly_source, coefficients, airports), expected, atol=1e-10, rtol=0)
        daily_source = pd.DataFrame(dict(airport=[ap], local_day=["2026-01-10"], runway=["A"], source_n=[95.], source_mean=[800.]))
        params = dict(coefficients=list(np.linspace(-8, 20, 2 * len(airports))), noise={a: 20. for a in airports})
        _, _, expected_shift = daily.apply(rows, prediction, daily_source, params)
        np.testing.assert_allclose(assembly.independent_daily_shift(rows, prediction, daily_source, params, airports), expected_shift, atol=1e-10, rtol=0)
        for source in [monthly_source.assign(invalid=.05), monthly_source.iloc[:0]]:
            expected, _ = monthly.apply(rows, prediction, source, dict(coefficients=coefficients))
            np.testing.assert_allclose(assembly.independent_monthly(rows, prediction, source, coefficients, airports), expected, atol=1e-10, rtol=0)

    def test_artifact_corruption_is_rejected(self):
        temporary_root = assembly.ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="v80_assembly_test_", dir=temporary_root) as temporary:
            path = Path(temporary) / "record.json"
            assembly.write(path, {"complete": True})
            record = assembly.art(path)
            self.assertEqual(assembly.verify(record), path)
            path.write_text("tampered")
            with self.assertRaises(ValueError):
                assembly.verify(record)


if __name__ == "__main__":
    unittest.main()
