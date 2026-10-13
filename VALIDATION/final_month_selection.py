"""Four embargoed 2025 folds and ten frozen direct model recipes.

Setup does not execute this module. The user explicitly launches ``run``.
No full-year fit, ranking prediction, network request, or submission is made.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "VALIDATION/runs/v3_20261013_final_month_selection"
MONTHS = ("2025-01", "2025-02", "2025-06", "2025-07")
NEURAL_SEEDS = (42, 1042, 2042)
TREE_COMPONENTS = ("main184", "airport168", "cat174")
NEURAL_FAMILIES = ("calls_ple", "physical_ple", "public_tabm")

# Each entry is the ordinary regression part. Every candidate receives exactly
# the same fold-fitted Rome probability mixture, rules and correction functions.
CANDIDATES = {
    "C01": {"name": "Daily/public main LightGBM, 184 inputs", "weights": {"main184": 1.0}},
    "C02": {"name": "Public per-airport LightGBM, 168", "weights": {"airport168": 1.0}},
    "C03": {"name": "Public CatBoost, 174 inputs", "weights": {"cat174": 1.0}},
    "C04": {"name": "Three-tree ensemble", "weights": {
        "main184": 1 / 3, "airport168": 1 / 3, "cat174": 1 / 3}},
    "C05": {"name": "Callsign PLE + trees", "weights": {
        "calls_ple": 0.71, "main184": 0.29 / 3, "airport168": 0.29 / 3, "cat174": 0.29 / 3}},
    "C06": {"name": "Physical-track PLE + trees", "weights": {
        "physical_ple": 0.71, "main184": 0.29 / 3, "airport168": 0.29 / 3, "cat174": 0.29 / 3}},
    "C07": {"name": "Public physical-track TabM + trees", "weights": {
        "public_tabm": 0.71, "main184": 0.29 / 3, "airport168": 0.29 / 3, "cat174": 0.29 / 3}},
    "C08": {"name": "Callsign / physical PLE + trees", "weights": {
        "calls_ple": 0.355, "physical_ple": 0.355,
        "main184": 0.29 / 3, "airport168": 0.29 / 3, "cat174": 0.29 / 3}},
    "C09": {"name": "Three neural families + trees", "weights": {
        "calls_ple": 0.71 / 3, "physical_ple": 0.71 / 3, "public_tabm": 0.71 / 3,
        "main184": 0.29 / 3, "airport168": 0.29 / 3, "cat174": 0.29 / 3}},
    "C10": {"name": "Callsign PLE / public TabM + trees", "weights": {
        "calls_ple": 0.355, "public_tabm": 0.355,
        "main184": 0.29 / 3, "airport168": 0.29 / 3, "cat174": 0.29 / 3}},
}


def folds():
    """Cyclic calendar embargo, preserving the requested December/January edge."""
    result = {}
    for month in MONTHS:
        center = int(month[-2:])
        excluded = {(center - 2) % 12 + 1, center, center % 12 + 1}
        result[month] = [f"2025-{m:02d}" for m in range(1, 13) if m not in excluded]
    return result


def protocol():
    return dict(schema="final_month_selection_v1", folds=folds(), candidates=CANDIDATES,
        tree_seed=42, rome_cat_seeds=[42, 142, 242], neural_seeds=list(NEURAL_SEEDS),
        neural_epochs=21, neural_schedule_epochs=24,
        primary_metric="unweighted mean of four all-row monthly RMSEs",
        secondary_metric="pooled all-row RMSE across four months",
        rounding="rint(max(final_prediction,0)), after rules and corrections",
        selection="manual after all ten candidates and four months are complete",
        label_policy="fold training months only for all fitting and preprocessing",
        evidence="reused 2025 development data; not an independent final-test estimate",
        historical_versions=False, uploads=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=digest(path))


def verify(record):
    path = ROOT / record["path"]
    if not path.is_file() or digest(path) != record["sha256"]:
        raise ValueError(f"Changed or missing artifact: {path}")
    return path


def verify_artifacts(value):
    """Validate nested model receipts, without loading a model or predicting."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            verify(value)
        else:
            for nested in value.values():
                verify_artifacts(nested)
    elif isinstance(value, list):
        for nested in value:
            verify_artifacts(nested)


def read(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def log(message):
    print(time.strftime("[%H:%M:%S] ") + message, flush=True)


def source_identity():
    # Pin helper implementations as well as the new adapters. Historical scripts
    # are not edited or executed; changing source refuses to mix resumed fits.
    paths = sorted(set((ROOT / "VALIDATION").glob("*.py")) |
                   set((ROOT / "pipeline").glob("*.py")))
    return [artifact(p) for p in paths]


def freeze(out):
    path = out / "identity.json"
    track_manifest = ROOT / "data/derived/sparse_track_sequence_20261010/full.json"
    spec = dict(protocol=protocol(), sources=source_identity(), track_cache_manifest=artifact(track_manifest))
    if path.exists():
        if read(path) != spec:
            raise ValueError("Protocol/source changed since this run began; preserve existing artifacts.")
    else:
        save_json(path, spec)
    return artifact(path)


@contextmanager
def run_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This validation already has an active runner.") from exc
        stream.write(str(os.getpid()))
        stream.flush()
        yield


def jobs():
    return [(c, 42) for c in TREE_COMPONENTS] + [("rome_lgb", 42)] + [
        ("rome_cat", s) for s in (42, 142, 242)] + [
        (family, s) for family in NEURAL_FAMILIES for s in NEURAL_SEEDS]


def child(args, month, stage, component=None, seed=42):
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    name = f"{month}_{stage}" + (f"_{component}_s{seed}" if component else "")
    path = logs / (name + ".log")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "_worker",
               "--out", str(args.out), "--month", month, "--stage", stage,
               "--threads", str(args.threads), "--device", args.device]
    if component:
        command += ["--component", component, "--seed", str(seed)]
    log(f"{name}; log: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS="1", TORCH_DISABLE_NATIVE_JIT="1")
    with path.open("a") as stream:
        done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if done.returncode:
        tail = "\n".join(path.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"{name} failed ({done.returncode}). Saved work remains resumable.\n{tail}")


def prediction_paths(out, month, component, seed):
    path = out / "folds" / month / "components" / f"{component}_s{seed}.parquet"
    return path, path.with_suffix(".json")


def load_prediction(out, month, component, seed, ids, identity):
    import numpy as np
    import pandas as pd
    path, receipt = prediction_paths(out, month, component, seed)
    record = read(receipt)
    binding = read(out / "folds" / month / "input_binding.json")
    if record["identity"] != identity or binding["identity"] != identity:
        raise ValueError(f"Prediction belongs to another run: {path}")
    if record["component"] != component or record["seed"] != seed:
        raise ValueError(f"Wrong component/seed receipt: {path}")
    if record["inputs"] != binding["inputs"]:
        raise ValueError(f"Prediction uses different fold inputs: {path}")
    verify_artifacts(record["model_receipt"])
    frame = pd.read_parquet(verify(record["prediction"]))
    if not np.array_equal(frame.mvt_id.to_numpy(), ids):
        raise ValueError(f"Prediction IDs/order mismatch: {path}")
    result = frame.pred.to_numpy(float)
    if result.shape != (len(ids),) or not np.isfinite(result).all():
        raise ValueError(f"Non-finite/missing predictions: {path}")
    return result


def mix_candidates(values):
    import numpy as np
    result = {}
    for key, recipe in CANDIDATES.items():
        if not np.isclose(sum(recipe["weights"].values()), 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"Weights do not sum to one: {key}")
        result[key] = sum(w * np.asarray(values[name], dtype=float)
                          for name, w in recipe["weights"].items())
    return result


def worker(args):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    identity = freeze(args.out)
    folder = args.out / "folds" / args.month
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(args.threads):
        if args.stage == "prepare":
            from final_month_inputs import prepare_fold
            prepare_fold(folder, args.month, folds()[args.month], args.threads)
            # Models consume this verified, complete packet; no old CV matrices.
            packet = folder / "inputs.pkl"
            preparation = folder / "inputs.json"
            shared_sources = read(preparation)["identity"]["sources"]
            common_path = args.out / "shared_input_sources.json"
            common = dict(identity=identity, sources=shared_sources)
            if common_path.exists() and read(common_path) != common:
                raise ValueError("Source data changed between folds; refusing to mix them.")
            if not common_path.exists():
                save_json(common_path, common)
            save_json(folder / "input_binding.json", dict(identity=identity, inputs=artifact(packet),
                                                          preparation=artifact(preparation)))
            return
        binding = read(folder / "input_binding.json")
        if binding["identity"] != identity:
            raise ValueError("Fold inputs belong to another protocol/source.")
        verify(binding["preparation"])
        packet = verify(binding["inputs"])
        data = pd.read_pickle(packet)
        data["input_identity"] = binding["inputs"]
        data["_input_artifact"] = dict(path=str(packet.resolve()), sha256=binding["inputs"]["sha256"])
        ids = np.asarray(data["ids"])
        if len(np.unique(ids)) != len(ids) or np.intersect1d(ids, data["train_ids"]).size:
            raise ValueError("Duplicate held IDs or training/held overlap.")
        if args.stage == "fit":
            path, receipt = prediction_paths(args.out, args.month, args.component, args.seed)
            if receipt.exists():
                record = read(receipt)
                if record["inputs"] != binding["inputs"]:
                    raise ValueError("Cached prediction uses other inputs.")
                load_prediction(args.out, args.month, args.component, args.seed, ids, identity)
                log("Verified existing predictions; no fit repeated.")
                return
            # Each adapter adds its component/member path; the common directory
            # also shares train-only neural encodings across seeds/families.
            model_out = folder / "models"
            if args.component in NEURAL_FAMILIES:
                from final_month_neural import fit_neural_member
                pred, details = fit_neural_member(args.component, data, model_out,
                    seed=args.seed, threads=args.threads, device=args.device)
            else:
                from final_month_models import fit_component
                pred, details = fit_component(args.component, data, model_out,
                    seed=args.seed, threads=args.threads)
            pred = np.asarray(pred, dtype=float)
            if pred.shape != (len(ids),) or not np.isfinite(pred).all():
                raise ValueError("Model did not predict every held-out row finitely.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".parquet.tmp")
            pd.DataFrame(dict(mvt_id=ids, pred=pred)).to_parquet(temporary, index=False)
            temporary.replace(path)
            save_json(receipt, dict(identity=identity, inputs=binding["inputs"],
                component=args.component, seed=args.seed, prediction=artifact(path), model_receipt=details))
            return
        import tuning_seasonal as ts
        values = {c: load_prediction(args.out, args.month, c, 42, ids, identity) for c in TREE_COMPONENTS}
        for family in NEURAL_FAMILIES:
            values[family] = np.mean([load_prediction(args.out, args.month, family, s, ids, identity)
                                     for s in NEURAL_SEEDS], axis=0)
        p = 0.5 * load_prediction(args.out, args.month, "rome_lgb", 42, ids, identity)
        p += 0.5 * np.mean([load_prediction(args.out, args.month, "rome_cat", s, ids, identity)
                           for s in (42, 142, 242)], axis=0)
        if ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid Rome mixture probability.")
        frame = pd.DataFrame(dict(mvt_id=ids))
        for candidate, ordinary in mix_candidates(values).items():
            pred = ts.final_prediction(ordinary, p, data["core_V"], data["raw"],
                                      data["meta"]["parameters"], data["meta"]["corrections"])
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite final candidate: {candidate}")
            frame[candidate] = np.rint(np.maximum(pred, 0)).astype("int64")
        path = folder / "predictions.parquet"
        receipt = folder / "predictions.json"
        if receipt.exists():
            old = read(receipt)
            if old["identity"] != identity or old["inputs"] != binding["inputs"]:
                raise ValueError("Existing candidate predictions belong to another run.")
            pd.testing.assert_frame_equal(pd.read_parquet(verify(old["prediction"])), frame, check_exact=True)
        else:
            frame.to_parquet(path, index=False)
            save_json(receipt, dict(identity=identity, inputs=binding["inputs"], prediction=artifact(path),
                components=[artifact(prediction_paths(args.out, args.month, c, s)[1]) for c, s in jobs()],
                candidates=list(CANDIDATES), held_rows=len(frame)))
        log("All ten candidates serialized; no outer-validation label used for fitting or stopping.")


def score_arrays(y, predictions):
    """Pure scoring contract, tested separately without executing a fit."""
    import numpy as np
    y = np.asarray(y, float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("All held labels must be finite; cannot silently drop rows.")
    result = {}
    for key, pred in predictions.items():
        pred = np.asarray(pred, float)
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid predictions for {key}")
        sse = float(np.square(pred - y).sum())
        result[key] = dict(n=len(y), sse=sse, rmse=float(np.sqrt(sse / len(y))))
    return result


def aggregate_metrics(monthly):
    import numpy as np
    if set(monthly) != set(MONTHS):
        raise ValueError("All four folds are required before candidate selection.")
    result = {}
    for candidate in CANDIDATES:
        parts = [monthly[m][candidate] for m in MONTHS]
        result[candidate] = dict(name=CANDIDATES[candidate]["name"],
            monthly={m: monthly[m][candidate]["rmse"] for m in MONTHS},
            mean4=float(np.mean([p["rmse"] for p in parts])),
            pooled4=float(np.sqrt(sum(p["sse"] for p in parts) / sum(p["n"] for p in parts))),
            jan_jul=float(np.mean([monthly[m][candidate]["rmse"] for m in (MONTHS[0], MONTHS[3])])),
            feb_jun=float(np.mean([monthly[m][candidate]["rmse"] for m in (MONTHS[1], MONTHS[2])])),
            n=sum(p["n"] for p in parts))
    return result


def report(out):
    import numpy as np
    import pandas as pd
    identity = freeze(out)
    monthly, bindings, paired_frames = {}, {}, []
    for month in MONTHS:
        folder = out / "folds" / month
        binding, prediction = read(folder / "input_binding.json"), read(folder / "predictions.json")
        if prediction["identity"] != identity or binding["identity"] != identity:
            raise ValueError("Mixed validation runs in report.")
        if prediction["inputs"] != binding["inputs"]:
            raise ValueError("Candidate predictions and held labels have different inputs.")
        preparation = read(verify(binding["preparation"]))
        if preparation["identity"]["sources"] != read(out / "shared_input_sources.json")["sources"]:
            raise ValueError("Four folds were built from different source data.")
        for record in prediction["components"]:
            component = read(verify(record))
            if component["identity"] != identity or component["inputs"] != binding["inputs"]:
                raise ValueError("Assembled component belongs to different inputs.")
            verify_artifacts(component["model_receipt"])
            verify(component["prediction"])
        frame = pd.read_parquet(verify(prediction["prediction"]))
        # All models/predictions are already fixed before outer truth is scored.
        data = pd.read_pickle(verify(binding["inputs"]))
        if not np.array_equal(frame.mvt_id.to_numpy(), data["ids"]):
            raise ValueError("Held-label/prediction alignment mismatch.")
        monthly[month] = score_arrays(data["raw"].taxi_sec.to_numpy(float),
                                     {c: frame[c].to_numpy(float) for c in CANDIDATES})
        errors = pd.DataFrame({c: np.square(frame[c].to_numpy(float) - data["raw"].taxi_sec.to_numpy(float))
                               for c in CANDIDATES})
        errors["month"] = month
        errors["day"] = pd.to_datetime(data["raw"].MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
        paired_frames.append(errors)
        bindings[month] = prediction
        del data, frame
    metrics = aggregate_metrics(monthly)
    order = sorted(metrics, key=lambda c: (metrics[c]["mean4"], c))
    import tuning_seasonal as ts
    errors = pd.concat(paired_frames, ignore_index=True)
    paired = {c: ts.paired_bootstrap(errors.day, errors.month, errors.C10, errors[c],
                                    reps=2000, seed=20261013) for c in CANDIDATES}
    result = dict(identity=identity, protocol=protocol(), metrics=metrics, monthly=monthly,
                  order=order, predictions=bindings, paired_vs_C10=paired, selected_model=None,
                  uncertainty="2000 paired UTC-day bootstrap draws within months; exploratory, no multiplicity correction",
                  note="Await user model selection. New direct refits, not historical submission scores.")
    save_json(out / "scores.json", result)
    lines = ["Candidate  Jan       Feb       Jun       Jul       Mean4     Pooled4   Jan/Jul   Feb/Jun"]
    rows = []
    for c in order:
        d = metrics[c]
        values = [d["monthly"][m] for m in MONTHS] + [d[k] for k in ("mean4", "pooled4", "jan_jul", "feb_jun")]
        lines.append(f"{c:9s}  " + "  ".join(f"{v:8.4f}" for v in values))
        rows.append(dict(candidate=c, name=d["name"], **d["monthly"],
                         **{k: d[k] for k in ("mean4", "pooled4", "jan_jul", "feb_jun")}))
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    summary = "\n".join(lines) + "\n"
    (out / "scores.txt").write_text(summary)
    print(summary, end="")
    print("All scores are seconds; lower is better. No final model selected or uploaded.")


def run(args):
    with run_lock(args.out):
        freeze(args.out)
        # Fail before spending hours on CPU trees if the required GPU/backend or
        # an input-only track cache is unavailable. No model is constructed here.
        import torch
        import catboost
        import lightgbm
        from final_month_neural import TRACK_DIRECTORY
        if not torch.cuda.is_available():
            raise RuntimeError("This frozen suite requires CUDA for CatBoost and the neural pools.")
        for name in ("full.json", "full_index.parquet", "full_points.bin"):
            if not (TRACK_DIRECTORY / name).is_file():
                raise FileNotFoundError(TRACK_DIRECTORY / name)
        log("Four folds, ten recipes; shared components fit sequentially. No submissions.")
        for month in args.folds:
            child(args, month, "prepare")
            for component, seed in jobs():
                child(args, month, "fit", component, seed)
            child(args, month, "assemble")
        if all((args.out / "folds" / m / "predictions.json").exists() for m in MONTHS):
            report(args.out)
        else:
            log("Partial run saved. All four folds are required for the selection table.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "report", "plan", "_worker"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--folds", nargs="+", choices=MONTHS, default=list(MONTHS))
    parser.add_argument("--month", choices=MONTHS)
    parser.add_argument("--stage", choices=("prepare", "fit", "assemble"))
    parser.add_argument("--component", choices=tuple(c for c, _ in jobs()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT / "VALIDATION/runs"):
        parser.error("--out must stay inside this repository's VALIDATION/runs directory")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16")
    if args.command == "plan":
        print(json.dumps(protocol(), indent=2))
    elif args.command == "run":
        run(args)
    elif args.command == "report":
        report(args.out)
    else:
        if args.month is None or args.stage is None or (args.stage == "fit" and args.component is None):
            parser.error("Internal worker requires month/stage/component")
        worker(args)


if __name__ == "__main__":
    main()
