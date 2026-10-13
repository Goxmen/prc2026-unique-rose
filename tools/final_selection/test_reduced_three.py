"""Synthetic contracts only: no training, production data or experiment logs."""
from pathlib import Path
import copy
import importlib.util
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("reduced_three", Path(__file__).with_name("reduced_three.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ReducedContracts(unittest.TestCase):
    def test_exact_dependency_graph_and_fixed_seed_budget(self):
        self.assertEqual(runner.SELECTED, ("C05", "C07", "C10"))
        jobs = runner.jobs()
        self.assertEqual(len(jobs), 11)
        self.assertNotIn(("physical_ple", 42), jobs)
        self.assertEqual([s for c, s in jobs if c == "calls_ple"], [42, 1042, 2042])
        self.assertEqual([s for c, s in jobs if c == "public_tabm"], [42])
        self.assertEqual(runner.protocol()["neural_fit_count"], 16)

    def test_candidate_arithmetic(self):
        values = dict(main184=np.array([10., 20.]), airport168=np.array([20., 30.]),
                      cat174=np.array([30., 40.]), calls_ple=np.array([40., 50.]),
                      public_tabm=np.array([60., 70.]))
        result = runner.mix_candidates(values)
        trees = np.array([20., 30.])
        np.testing.assert_allclose(result["C05"], .71 * values["calls_ple"] + .29 * trees)
        np.testing.assert_allclose(result["C07"], .71 * values["public_tabm"] + .29 * trees)
        np.testing.assert_allclose(result["C10"], .5 * (result["C05"] + result["C07"]))

    def test_separate_identity_preserves_original_and_rejects_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original_path = out / "identity.json"
            original_path.write_text('{"historical":"preserved"}\n')
            before = original_path.read_bytes()
            parent = runner.original.artifact(original_path)
            with mock.patch.object(runner.original, "freeze", return_value=parent):
                reduced, original = runner.freeze(out)
                self.assertNotEqual(reduced, original)
                self.assertEqual(original_path.read_bytes(), before)
                self.assertEqual(runner.freeze(out), (reduced, original))
                changed = runner.protocol()
                changed["neural_fit_count"] = 8
                with mock.patch.object(runner, "protocol", return_value=changed):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        runner.freeze(out)

    def test_component_rejects_input_identity_seed_and_protocol_mismatch(self):
        identity = dict(path="identity.json", sha256="abc")
        binding = dict(identity=identity, inputs=dict(path="input.pkl", sha256="def"))
        record = dict(identity=identity, inputs=binding["inputs"], component="public_tabm", seed=42)
        runner.check_component(record, binding, identity, "public_tabm", 42)
        for key, value in (("identity", {}), ("inputs", {}), ("seed", 1042), ("component", "physical_ple")):
            bad = copy.deepcopy(record)
            bad[key] = value
            with self.assertRaises(ValueError):
                runner.check_component(bad, binding, identity, "public_tabm", 42)

    def test_fourfold_gate_before_truth_or_any_parquet_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for month in runner.MONTHS[:-1]:
                folder = runner.destination(out) / "folds" / month
                folder.mkdir(parents=True)
                for name in ("predictions.json", "evaluation.json"):
                    (folder / name).write_text("{}")
            with mock.patch("pandas.read_parquet", side_effect=AssertionError("read too early")):
                with self.assertRaisesRegex(ValueError, "All four"):
                    runner.verify_assemblies(out, {}, {})

    def test_origin_input_exception_is_exact_not_broad(self):
        input_record = dict(path="remote/inputs.pkl", sha256="truthbound")
        model_record = dict(path="remote/model.pt", sha256="modelbound")
        changed_input = dict(path="remote/inputs.pkl", sha256="changed")
        with mock.patch.object(runner.original, "verify") as verify:
            runner.verify_retained_artifacts(dict(input=input_record, model=model_record,
                                                bad_input=changed_input), input_record)
            self.assertEqual(verify.call_args_list, [mock.call(model_record), mock.call(changed_input)])

    def test_cache_hash_failure_is_never_a_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path, receipt = runner.original.prediction_paths(out, "2025-01", "calls_ple", 42)
            receipt.parent.mkdir(parents=True)
            receipt.write_text("{}")
            identity = dict(path="identity", sha256="sha")
            binding = dict(identity=identity, inputs=dict(path="inputs", sha256="sha"))
            record = dict(identity=identity, inputs=binding["inputs"], component="calls_ple", seed=42,
                          prediction=dict(path="prediction", sha256="bad"), model_receipt={})
            with mock.patch.object(runner.original, "read", side_effect=[binding, record]):
                with mock.patch.object(runner.original, "verify", side_effect=ValueError("changed")):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        runner.cached_component(out, "2025-01", "calls_ple", 42, identity)

    def test_aggregate_requires_four_folds_and_preserves_equal_month_weight(self):
        monthly = {m: {c: dict(n=i+1, rmse=float(i+1), sse=float((i+1)**3)) for c in runner.SELECTED}
                   for i, m in enumerate(runner.MONTHS)}
        result = runner.aggregate_metrics(monthly)
        self.assertEqual(result["C10"]["mean4"], 2.5)
        self.assertEqual(result["C10"]["pooled4"], np.sqrt(10.))
        with self.assertRaises(ValueError):
            runner.aggregate_metrics({runner.MONTHS[0]: monthly[runner.MONTHS[0]]})

    def test_host_provenance_keeps_nested_model_hosts(self):
        self.assertEqual(runner.hosts_in(dict(members=[dict(host="GPC"),
                          dict(identity=dict(execution=dict(host="GLAPTOP")))])), {"GPC", "GLAPTOP"})

    def test_assembly_is_isolated_and_worker_failure_never_reads_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), threads=4, device="cuda")
            with mock.patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run:
                with mock.patch.object(Path, "read_text", side_effect=AssertionError("must not inspect log")):
                    with self.assertRaisesRegex(RuntimeError, "Log not inspected"):
                        runner.child(args, "2025-01", "assemble")
                argv = run.call_args.args[0]
                self.assertIn("_assemble", argv)
                self.assertEqual(Path(argv[2]).name, "reduced_three.py")

    def test_compact_fourfold_audit_works_without_remote_pickle_and_rejects_corruption(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            identity, parent = {"path": "reduced", "sha256": "reducedhash"}, {"path": "original", "sha256": "parenthash"}
            runner.original.save_json(out / "shared_input_sources.json", dict(identity=parent, sources=[]))
            for month in runner.MONTHS:
                fold = out / "folds" / month
                fold.mkdir(parents=True)
                inputs = dict(path=str(fold / "inputs.pkl"), sha256="originpicklehash")
                prep = fold / "inputs.json"
                runner.original.save_json(prep, dict(identity=dict(month=month,
                    train_months=runner.original.folds()[month], sources=[])))
                binding = dict(identity=parent, inputs=inputs, preparation=runner.original.artifact(prep))
                runner.original.save_json(fold / "input_binding.json", binding)
                components = []
                for component, seed in runner.jobs():
                    path, receipt = runner.original.prediction_paths(out, month, component, seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(dict(mvt_id=[1, 2], pred=[10., 20.])).to_parquet(path, index=False)
                    runner.original.save_json(receipt, dict(identity=parent, inputs=inputs,
                        component=component, seed=seed, prediction=runner.original.artifact(path),
                        model_receipt=dict(input=inputs)))
                    components.append(runner.original.artifact(receipt))
                target = runner.destination(out) / "folds" / month
                target.mkdir(parents=True)
                prediction, evaluation = target / "predictions.parquet", target / "evaluation.parquet"
                pd.DataFrame(dict(mvt_id=[1, 2], **{c: [10, 20] for c in runner.SELECTED})).to_parquet(prediction, index=False)
                pd.DataFrame(dict(mvt_id=[1, 2], day=[month + "-01"] * 2, taxi_sec=[11., 22.])).to_parquet(evaluation, index=False)
                runner.original.save_json(target / "predictions.json", dict(identity=identity,
                    original_identity=parent, inputs=inputs, components=components, candidates=list(runner.SELECTED),
                    held_rows=2, prediction=runner.original.artifact(prediction)))
                runner.original.save_json(target / "evaluation.json", dict(identity=identity,
                    original_identity=parent, inputs=inputs, preparation=binding["preparation"],
                    candidate_predictions=runner.original.artifact(prediction), rows=2,
                    origin_input_verification=dict(artifact=inputs, verified_sha256=inputs["sha256"], bytes=500, host="remote"),
                    evaluation=runner.original.artifact(evaluation)))
            self.assertEqual(set(runner.verify_assemblies(out, identity, parent)), set(runner.MONTHS))
            self.assertFalse(any((out / "folds" / m / "inputs.pkl").exists() for m in runner.MONTHS))
            (runner.destination(out) / "folds" / runner.MONTHS[-1] / "evaluation.parquet").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Changed or missing"):
                runner.verify_assemblies(out, identity, parent)


if __name__ == "__main__":
    unittest.main()
