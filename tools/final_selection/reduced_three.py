"""Three fixed direct finalists, reusing immutable four-month component workers.

The original source, protocol and receipts stay unchanged. This separate budget
revision keeps callsign seeds 42/1042/2042 and public TabM seed 42 in every fold.
No validation score chooses the seed budget. No upload or automatic selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "VALIDATION"), str(ROOT / "pipeline")]
import final_month_selection as original

MONTHS = original.MONTHS
SELECTED = ("C05", "C07", "C10")
FAMILY_SEEDS = {"calls_ple": (42, 1042, 2042), "public_tabm": (42,)}
NAMES = {"C05": "Callsign PLE (3 seeds) + trees",
         "C07": "Public physical-track TabM (seed42) + trees",
         "C10": "Callsign PLE (3 seeds) / public TabM (seed42) + trees"}


def destination(out):
    return Path(out) / "reduced_three"


def jobs():
    return [(c, 42) for c in original.TREE_COMPONENTS] + [("rome_lgb", 42)] + [
        ("rome_cat", s) for s in (42, 142, 242)] + [
        (family, seed) for family, seeds in FAMILY_SEEDS.items() for seed in seeds]


def protocol():
    return dict(schema="final_month_selection_reduced_three_v1", folds=original.folds(),
        candidates={c: dict(name=NAMES[c], weights=original.CANDIDATES[c]["weights"])
                    for c in SELECTED}, neural_seeds={k: list(v) for k, v in FAMILY_SEEDS.items()},
        neural_fit_count=16, original_neural_fit_count=36, jobs=[list(j) for j in jobs()],
        tree_seed=42, rome_cat_seeds=[42, 142, 242], neural_epochs=21,
        neural_schedule_epochs=24, primary_metric="unweighted mean of four all-row monthly RMSEs",
        rounding="rint(max(final_prediction,0)), after original rules and corrections",
        selection="manual only after all three candidates and four months are complete",
        rationale="Owner requested fewer contenders and two machines before score review; drop physical PLE and extra TabM seeds",
        seed_budget="callsign three retained, TabM seed42 only; identical budget in all four folds",
        evidence="reused 2025 development data; no independent final-test claim",
        historical_versions=False, uploads=False,
        limitation="Direct refits omit historical affine/public posterior/distillation layers; not v80/v83 replays")


def freeze(out):
    out = Path(out)
    if not (out / "identity.json").is_file():
        raise ValueError("Copy or preserve the original run identity before using this revision.")
    original_identity = original.freeze(out)
    path = destination(out) / "identity.json"
    spec = dict(protocol=protocol(), original_identity=original_identity,
                wrapper_source=original.artifact(Path(__file__)))
    if path.exists():
        if original.read(path) != spec:
            raise ValueError("Reduced protocol/source changed; preserve existing results.")
    else:
        original.save_json(path, spec)
    return original.artifact(path), original_identity


def check_binding(binding, identity):
    if binding["identity"] != identity:
        raise ValueError("Fold belongs to a different original run.")


def check_component(record, binding, identity, component, seed):
    check_binding(binding, identity)
    if record["identity"] != identity or record["inputs"] != binding["inputs"]:
        raise ValueError("Component belongs to different source/input identity.")
    if record["component"] != component or record["seed"] != seed:
        raise ValueError("Component or seed differs from requested fixed job.")


def hosts_in(value):
    hosts = set()
    if isinstance(value, dict):
        if isinstance(value.get("host"), str):
            hosts.add(value["host"])
        for item in value.values():
            hosts.update(hosts_in(item))
    elif isinstance(value, list):
        for item in value:
            hosts.update(hosts_in(item))
    return hosts


def cached_component(out, month, component, seed, original_identity):
    _, receipt = original.prediction_paths(out, month, component, seed)
    if not receipt.exists():
        return False
    binding = original.read(out / "folds" / month / "input_binding.json")
    record = original.read(receipt)
    check_component(record, binding, original_identity, component, seed)
    original.verify(record["prediction"])
    original.verify_artifacts(record["model_receipt"])
    return True


def child(args, month, stage, component=None, seed=42):
    # Deliberately never reads a log, even when a worker fails.
    logs = destination(args.out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    name = f"{month}_{stage}" + (f"_{component}_s{seed}" if component else "")
    path = logs / (name + ".log")
    if stage == "assemble":
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "_assemble",
                   "--out", str(args.out), "--month", month,
                   "--threads", str(args.threads), "--device", args.device]
    else:
        command = [sys.executable, "-u", str(ROOT / "VALIDATION/final_month_selection.py"),
                   "_worker", "--out", str(args.out), "--month", month, "--stage", stage,
                   "--threads", str(args.threads), "--device", args.device]
    if component:
        command += ["--component", component, "--seed", str(seed)]
    original.log(f"{name}; log: {path}")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS="1", TORCH_DISABLE_NATIVE_JIT="1")
    with path.open("a") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{name} exited {completed.returncode}; preserved log: {path}. Log not inspected.")


def mix_candidates(values):
    import numpy as np
    result = {}
    for candidate in SELECTED:
        weights = original.CANDIDATES[candidate]["weights"]
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-12, rtol=0):
            raise ValueError("Invalid fixed candidate weights.")
        result[candidate] = sum(weight * np.asarray(values[name], dtype=float)
                                for name, weight in weights.items())
    return result


def assemble(out, month, identity, original_identity, threads=4):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    import tuning_seasonal as ts
    folder = out / "folds" / month
    binding = original.read(folder / "input_binding.json")
    check_binding(binding, original_identity)
    original.verify(binding["preparation"])
    packet_path = original.verify(binding["inputs"])
    data = pd.read_pickle(packet_path)
    ids = np.asarray(data["ids"])
    if len(np.unique(ids)) != len(ids) or np.intersect1d(ids, data["train_ids"]).size:
        raise ValueError("Duplicate or training-overlapping held IDs.")
    with threadpool_limits(threads):
        values = {c: original.load_prediction(out, month, c, 42, ids, original_identity)
                  for c in original.TREE_COMPONENTS}
        for family, seeds in FAMILY_SEEDS.items():
            values[family] = np.mean([original.load_prediction(out, month, family, seed, ids,
                                    original_identity) for seed in seeds], axis=0)
        p = .5 * original.load_prediction(out, month, "rome_lgb", 42, ids, original_identity)
        p += .5 * np.mean([original.load_prediction(out, month, "rome_cat", seed, ids,
                           original_identity) for seed in (42, 142, 242)], axis=0)
        if ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid Rome probability.")
        frame = pd.DataFrame(dict(mvt_id=ids))
        for candidate, ordinary in mix_candidates(values).items():
            pred = ts.final_prediction(ordinary, p, data["core_V"], data["raw"],
                                      data["meta"]["parameters"], data["meta"]["corrections"])
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite predictions for {candidate}.")
            frame[candidate] = np.rint(np.maximum(pred, 0)).astype("int64")
    components = [original.artifact(original.prediction_paths(out, month, c, s)[1]) for c, s in jobs()]
    provenance = {f"{c}_s{s}": sorted(hosts_in(original.read(original.prediction_paths(out, month, c, s)[1])["model_receipt"]))
                  for c, s in jobs()}
    target = destination(out) / "folds" / month
    target.mkdir(parents=True, exist_ok=True)
    path, receipt = target / "predictions.parquet", target / "predictions.json"
    spec = dict(identity=identity, original_identity=original_identity, inputs=binding["inputs"],
                components=components, candidates=list(SELECTED), held_rows=len(frame),
                component_hosts=provenance)
    if receipt.exists():
        old = original.read(receipt)
        if {k: old[k] for k in spec} != spec:
            raise ValueError("Existing reduced assembly uses another protocol/input/component.")
        pd.testing.assert_frame_equal(pd.read_parquet(original.verify(old["prediction"])), frame,
                                      check_exact=True)
    else:
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        original.save_json(receipt, dict(spec, prediction=original.artifact(path), assembly_host=platform.node()))
    # Compact evaluation data travels with returned folds; the original 4GB
    # pickle remains on its training host. Export happens only AFTER all fixed
    # component predictions and all three final candidate arrays are committed.
    evaluation = pd.DataFrame(dict(mvt_id=ids,
        day=pd.to_datetime(data["raw"].MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy(),
        taxi_sec=data["raw"].taxi_sec.to_numpy(float)))
    if evaluation.day.isna().any() or not np.isfinite(evaluation.taxi_sec).all():
        raise ValueError("Invalid evaluation labels or UTC dates.")
    evaluation_path, evaluation_receipt = target / "evaluation.parquet", target / "evaluation.json"
    evaluation_spec = dict(identity=identity, original_identity=original_identity,
        inputs=binding["inputs"], preparation=binding["preparation"],
        candidate_predictions=original.artifact(path), rows=len(ids),
        origin_input_verification=dict(artifact=binding["inputs"], bytes=packet_path.stat().st_size,
            host=platform.node(), method="Full file SHA256 verified before deserializing on origin host",
            verified_sha256=binding["inputs"]["sha256"]))
    if evaluation_receipt.exists():
        old = original.read(evaluation_receipt)
        if {k: old[k] for k in evaluation_spec} != evaluation_spec:
            raise ValueError("Existing evaluation export has different provenance.")
        pd.testing.assert_frame_equal(pd.read_parquet(original.verify(old["evaluation"])),
                                      evaluation, check_exact=True)
    else:
        temporary = evaluation_path.with_suffix(".parquet.tmp")
        evaluation.to_parquet(temporary, index=False)
        temporary.replace(evaluation_path)
        original.save_json(evaluation_receipt, dict(evaluation_spec,
                            evaluation=original.artifact(evaluation_path)))
    original.log(f"{month}: three reduced finalists and compact evaluation fixed; no validation score read.")


def require_all_predictions(out):
    missing = [month for month in MONTHS if not
               all((destination(out) / "folds" / month / name).is_file()
                   for name in ("predictions.json", "evaluation.json"))]
    if missing:
        raise ValueError("All four reduced predictions must exist before any truth scoring; missing " + ", ".join(missing))


def verify_retained_artifacts(value, origin_input):
    """Verify returned artifacts; full input is attested separately on its host.

    The sole allowed omission is the exact path AND SHA256 of the originating
    input pickle. This never skips a checkpoint/model/source or arbitrary file.
    """
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            if value != origin_input:
                original.verify(value)
        else:
            for item in value.values():
                verify_retained_artifacts(item, origin_input)
    elif isinstance(value, list):
        for item in value:
            verify_retained_artifacts(item, origin_input)


def verify_assemblies(out, identity, original_identity):
    # Entire first pass is metadata/predictions only. Never load a truth packet
    # until every fold, candidate and component binding has been validated.
    require_all_predictions(out)
    import numpy as np
    import pandas as pd
    records = {}
    shared = original.read(out / "shared_input_sources.json")
    if shared["identity"] != original_identity:
        raise ValueError("Shared inputs have a different original run identity.")
    for month in MONTHS:
        folder = out / "folds" / month
        binding = original.read(folder / "input_binding.json")
        check_binding(binding, original_identity)
        preparation = original.read(original.verify(binding["preparation"]))
        if (preparation["identity"]["month"] != month or
                preparation["identity"]["train_months"] != original.folds()[month]):
            raise ValueError("Preparation violates the frozen held month or calendar buffer.")
        if preparation["identity"]["sources"] != shared["sources"]:
            raise ValueError("Four folds have different source data.")
        reduced = original.read(destination(out) / "folds" / month / "predictions.json")
        if (reduced["identity"] != identity or reduced["original_identity"] != original_identity or
                reduced["inputs"] != binding["inputs"] or reduced["candidates"] != list(SELECTED)):
            raise ValueError("Mixed reduced identity, inputs or candidates.")
        expected = [original.artifact(original.prediction_paths(out, month, c, s)[1]) for c, s in jobs()]
        if reduced["components"] != expected:
            raise ValueError("Reduced assembly components differ from the fixed dependency graph.")
        for item, (component, seed) in zip(reduced["components"], jobs()):
            record = original.read(original.verify(item))
            check_component(record, binding, original_identity, component, seed)
            verify_retained_artifacts(record["model_receipt"], binding["inputs"])
            original.verify(record["prediction"])
        frame = pd.read_parquet(original.verify(reduced["prediction"]))
        if list(frame.columns) != ["mvt_id", *SELECTED] or len(frame) != reduced["held_rows"]:
            raise ValueError("Incomplete reduced candidate arrays.")
        if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any() or not np.isfinite(frame[list(SELECTED)].to_numpy(float)).all():
            raise ValueError("Invalid prediction IDs or candidate values.")
        evaluation = original.read(destination(out) / "folds" / month / "evaluation.json")
        if (evaluation["identity"] != identity or evaluation["original_identity"] != original_identity or
                evaluation["inputs"] != binding["inputs"] or evaluation["preparation"] != binding["preparation"] or
                evaluation["candidate_predictions"] != reduced["prediction"] or evaluation["rows"] != len(frame)):
            raise ValueError("Compact evaluation provenance differs from prediction/input bindings.")
        origin = evaluation["origin_input_verification"]
        if (origin["artifact"] != binding["inputs"] or origin["verified_sha256"] != binding["inputs"]["sha256"] or
                origin["bytes"] <= 0 or not origin["host"]):
            raise ValueError("Invalid origin-host full-input verification.")
        original.verify(evaluation["evaluation"])
        records[month] = (binding, reduced, frame, evaluation)
    return records


def aggregate_metrics(monthly):
    import numpy as np
    if set(monthly) != set(MONTHS):
        raise ValueError("All four folds are required.")
    result = {}
    for candidate in SELECTED:
        parts = [monthly[m][candidate] for m in MONTHS]
        result[candidate] = dict(name=NAMES[candidate],
            monthly={m: monthly[m][candidate]["rmse"] for m in MONTHS},
            mean4=float(np.mean([p["rmse"] for p in parts])),
            pooled4=float(np.sqrt(sum(p["sse"] for p in parts) / sum(p["n"] for p in parts))),
            jan_jul=float(np.mean([monthly[m][candidate]["rmse"] for m in (MONTHS[0], MONTHS[3])])),
            feb_jun=float(np.mean([monthly[m][candidate]["rmse"] for m in (MONTHS[1], MONTHS[2])])),
            n=sum(p["n"] for p in parts))
    return result


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def report(out):
    import numpy as np
    import pandas as pd
    identity, original_identity = freeze(out)
    records = verify_assemblies(out, identity, original_identity)
    monthly, paired_frames = {}, []
    for month in MONTHS:
        binding, reduced, frame, evaluation_record = records[month]
        data = pd.read_parquet(original.verify(evaluation_record["evaluation"]))
        if list(data.columns) != ["mvt_id", "day", "taxi_sec"] or not np.array_equal(frame.mvt_id, data.mvt_id):
            raise ValueError("Held-label/prediction alignment mismatch.")
        if data.mvt_id.duplicated().any() or data.day.isna().any():
            raise ValueError("Invalid compact evaluation IDs/day.")
        truth = data.taxi_sec.to_numpy(float)
        monthly[month] = original.score_arrays(truth, {c: frame[c].to_numpy(float) for c in SELECTED})
        errors = pd.DataFrame({c: np.square(frame[c].to_numpy(float) - truth) for c in SELECTED})
        errors["month"] = month
        errors["day"] = data.day.to_numpy()
        paired_frames.append(errors)
        del data
    metrics = aggregate_metrics(monthly)
    order = sorted(metrics, key=lambda c: (metrics[c]["mean4"], c))
    import tuning_seasonal as ts
    errors = pd.concat(paired_frames, ignore_index=True)
    paired = {c: ts.paired_bootstrap(errors.day, errors.month, errors.C10, errors[c],
                                   reps=2000, seed=20261013) for c in SELECTED}
    result = dict(identity=identity, original_identity=original_identity, protocol=protocol(),
                  metrics=metrics, monthly=monthly, order=order,
                  predictions={m: records[m][1] for m in MONTHS}, paired_vs_C10=paired,
                  evaluation_packets={m: records[m][3] for m in MONTHS},
                  input_verification="Original full pickle hashed on origin host; collector report verifies compact export, original bindings and retained model artifacts",
                  selected_model=None, uploads=False,
                  uncertainty="2000 paired UTC-day draws within months; exploratory; no multiplicity correction",
                  note="Manual selection pending; callsign three seeds, public TabM one seed; not v80/v83 replays.")
    target = destination(out)
    original.save_json(target / "scores.json", result)
    lines = ["Candidate  Jan       Feb       Jun       Jul       Mean4     Pooled4   Jan/Jul   Feb/Jun"]
    rows = []
    for c in order:
        d = metrics[c]
        values = [d["monthly"][m] for m in MONTHS] + [d[k] for k in ("mean4", "pooled4", "jan_jul", "feb_jun")]
        lines.append(f"{c:9s}  " + "  ".join(f"{v:8.4f}" for v in values))
        rows.append(dict(candidate=c, name=d["name"], **d["monthly"],
                         **{k: d[k] for k in ("mean4", "pooled4", "jan_jul", "feb_jun")}))
    atomic_text(target / "scores.csv", pd.DataFrame(rows).to_csv(index=False))
    summary = "\n".join(lines) + "\n"
    atomic_text(target / "scores.txt", summary)
    print(summary, end="")
    print("No final model selected or uploaded.")


def run(args):
    with original.run_lock(args.out):
        identity, original_identity = freeze(args.out)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the unchanged component workers.")
        original.log("Three finalists; callsign3 / TabM1; original fits reused; no submissions.")
        for month in args.folds:
            child(args, month, "prepare")
            for component, seed in jobs():
                if cached_component(args.out, month, component, seed, original_identity):
                    original.log(f"{month}_{component}_s{seed}: verified retained component.")
                else:
                    child(args, month, "fit", component, seed)
            # A short-lived process owns the 4GB packet and correction objects.
            # Python allocator arenas must not accumulate in the supervisor and
            # overlap the next fold's heavy fit on the 16GB laptop.
            child(args, month, "assemble")
        if all((destination(args.out) / "folds" / m / "predictions.json").is_file() for m in MONTHS):
            report(args.out)
        else:
            original.log("Partial reduced run preserved; report waits for all four folds.")


def status(out):
    # Metadata only: no logs, model loading, source hashing or label access.
    result = {"host": platform.node(), "out": str(out), "folds": {}}
    for month in MONTHS:
        result["folds"][month] = dict(
            prepared=(out / "folds" / month / "input_binding.json").is_file(),
            component_receipts=[f"{c}_s{s}" for c, s in jobs()
                                if original.prediction_paths(out, month, c, s)[1].is_file()],
            reduced_predictions=(destination(out) / "folds" / month / "predictions.json").is_file())
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "report", "plan", "status", "_assemble"))
    parser.add_argument("--out", type=Path, default=original.DEFAULT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--folds", nargs="+", choices=MONTHS, default=list(MONTHS))
    parser.add_argument("--month", choices=MONTHS)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT / "VALIDATION/runs"):
        parser.error("--out must stay inside this repository's VALIDATION/runs directory")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16")
    if args.command == "plan":
        print(json.dumps(protocol(), indent=2))
    elif args.command == "status":
        status(args.out)
    elif args.command == "run":
        run(args)
    elif args.command == "_assemble":
        if args.month is None:
            parser.error("Internal assembly requires --month")
        identity, original_identity = freeze(args.out)
        assemble(args.out, args.month, identity, original_identity, args.threads)
    else:
        with original.run_lock(args.out):
            report(args.out)


if __name__ == "__main__":
    main()
