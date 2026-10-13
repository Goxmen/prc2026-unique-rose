"""Recover the missing tree members of the fixed v80 recipe on full 2025.

This is a fresh refit, not an exact reconstruction claim. Models already fitted
on the same full-year inputs are retained. Version 3 and 5--9 use the archived
original-27 implementation, because their exact historical source is absent.
Old-view differences are recorded, never optimized or used to select settings.
No network access or submission is performed by this worker.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "VALIDATION"), str(ROOT / "pipeline")]
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

VERSIONS = (2, 3, 5, 6, 7, 9, 26, 27, 28, 29, 43, 44)
EARLY = (2, 3, 5, 6, 7, 9)
HYBRID = (26, 27, 28, 29)
DEFAULT = ROOT / "VALIDATION/runs/v3_20261013_v80_refit_final"
RUNS = ROOT / "VALIDATION/runs"
EARLY_RUN = RUNS / "v3_20261012_final_missing_early_reconstruct"
STAND_RUN = RUNS / "v3_20261012_final_missing_stand_reconstruct"
COORD_RUN = RUNS / "v3_20261012_final_missing_coordinates_reconstruct"
FINAL_RUN = RUNS / "v3_20261008_final_submit"
IDEAS = RUNS / "v3_20261003_ideas_submit/work/inputs.pkl"
SOURCE = ROOT / "VALIDATION/archive/runs/v3_20260909T161759260709Z/source/VALIDATION/modeling.py"
SOURCE_SHA = "5eda81a506acb91e1baba28e369fac379bf7bfbbed31680deaf7474da8ed920d"
CAVEAT = ("v3/v5-v9 original modeling.py is not retained; archived original27 "
          "implementation with recorded feature order, filter, rounds and seeds. "
          "Fresh refit differences are expected and disclosed; no historical-byte gate.")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    p = Path(path).absolute()
    return {"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            "sha256": sha(p)}


def resolve(record):
    p = Path(record["path"])
    return p if p.is_absolute() else ROOT / p


def verify(record):
    p = resolve(record)
    if sha(p) != record["sha256"]:
        raise ValueError(f"Changed artifact: {p}")
    return p


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)


def save_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def backend():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Archived original27 source changed")
    spec = importlib.util.spec_from_file_location("v80_refit_original27", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(version, threads):
    """Recorded full-year specification; no tuning and no early stopping."""
    if version == 2:
        return {"max_taxi_sec": 7200, "rounds": 3000, "seed": 42,
                "threads": threads, "features": 18, "category_order": "sorted"}
    meta = read(ROOT / f"submissions/unique-rose_v{version}.meta.json")
    cfg = meta["config"] if "config" in meta else {k: meta[k] for k in ("max_taxi_sec", "rounds", "fpl_flag")}
    cfg = dict(cfg, threads=threads, seed=42)
    if cfg["fpl_flag"] or cfg["rounds"] != 800:
        raise ValueError(f"Unexpected early recipe v{version}")
    return cfg


def input_contract():
    """Small receipts locate and hash-bind all large packets before execution."""
    v4 = read(EARLY_RUN / "v4/preflight.json")
    stand = read(STAND_RUN / "preflight.json")
    coords = read(COORD_RUN / "identity.json")
    wx = read(RUNS / "v3_20261008_open_meteo_submit/work/prepared.json")
    records = [v4["training"], stand["training"],
               {"path": str(IDEAS.relative_to(ROOT)), "sha256": wx["original_inputs_sha256"]}]
    for view in ("old", "final"):
        rec = read(FINAL_RUN / f"inputs_{view}.json")
        records.append({"path": str((FINAL_RUN / f"work/inputs_{view}.pkl").relative_to(ROOT)),
                        "sha256": rec["sha256"]})
    records += [read(EARLY_RUN / "v5/trained.json")["model"],
                read(EARLY_RUN / "v5/preflight.json")["training"]]
    classifier = next(r for r in coords["sources"]
                      if r["path"].endswith("models/refit2025__classifier__42.txt"))
    records.append(classifier)
    # training_ids independently rereads these original labels/membership.
    for item in coords["sources"]:
        name = item["path"]
        if "data/training_2025-" in name or name.endswith(("training_membership.parquet", "submission_v46.json")):
            records.append(item)
    return records


def plan(threads=4):
    paths = input_contract()
    return dict(versions=list(VERSIONS), fresh_early=[2, 3, 6, 7, 9],
                retained_early_model=[5], fresh_hybrid_ordinary=list(HYBRID),
                fresh_hybrid_classifier=[26, 29], fresh_coordinate_ordinary=[43, 44],
                retained_coordinate_classifier=True, threads=threads,
                input_paths=[dict(path=p["path"], exists=resolve(p).exists()) for p in paths],
                source_caveat=CAVEAT, no_fit_or_inference=True,
                required_files=[p["path"] for p in paths])


def freeze(out, threads):
    records = input_contract()
    # Every packet is verified before any fit; sources are frozen separately.
    for item in records:
        verify(item)
    sources = [Path(__file__), SOURCE]
    sources += sorted((ROOT / "VALIDATION").glob("*.py"))
    sources += sorted((ROOT / "pipeline").glob("*.py"))
    evidence = [EARLY_RUN / "v4/preflight.json", EARLY_RUN / "v5/trained.json",
                STAND_RUN / "preflight.json", COORD_RUN / "identity.json"]
    evidence += [ROOT / f"submissions/unique-rose_v{v}.meta.json" for v in EARLY if v != 2]
    evidence += [ROOT / "data/derived/scored_submission_cache_20261011/manifest.json"]
    # These are the historical calibration/feature records actually read later.
    coordinate_identity = read(COORD_RUN / "identity.json")
    evidence += [ROOT / item["path"] for item in coordinate_identity["sources"]
                 if item["path"].endswith(("submission_v43.json", "submission_v44.json", "categories.json"))]
    evidence += [RUNS / "v3_20260918T120331999764Z/submission_v29.json"]
    for view in ("old", "final"):
        evidence += [FINAL_RUN / f"inputs_{view}.json",
                     ROOT / "data" / ("submitting.parquet" if view == "old" else "final_submitting.parquet")]
    ident = dict(versions=list(VERSIONS), threads=threads, input_artifacts=records,
                 sources=[artifact(p) for p in sources], evidence=[artifact(p) for p in evidence],
                 early={str(v): configuration(v, threads) for v in EARLY},
                 caveat=CAVEAT, python=sys.version, lightgbm=lgb.__version__,
                 host=os.uname().nodename, fit_scope="all eligible 2025 only",
                 model_selection="none; fixed v80 components and original ensemble weights")
    path = out / "identity.json"
    if path.exists():
        if read(path) != ident:
            raise ValueError("Tree refit source/input/config identity changed")
    else:
        save(path, ident)
    return ident


def categorical_values(frame):
    return {c: frame[c].cat.categories.tolist() for c in frame
            if isinstance(frame[c].dtype, pd.CategoricalDtype)}


def apply_categories(frame, cats):
    frame = frame.copy()
    for name, values in cats.items():
        items = frame[name].astype(object)
        frame[name] = pd.Categorical(items.where(items.isin(values)), categories=values)
    return frame


def fit_saved(out, name, X, y, params, rounds, inputs, reuse=None):
    """Save models before inference. A completed model is never fitted twice."""
    folder = out / "models"
    model, receipt = folder / f"{name}.txt", folder / f"{name}.json"
    cfg = dict(name=name, columns=list(X), rows=len(y), params=params, rounds=rounds,
               categories=categorical_values(X), inputs=inputs,
               identity_sha256=sha(out / "identity.json"))
    if receipt.exists():
        rec = read(receipt)
        if rec["config"] != cfg:
            raise ValueError(f"Model configuration changed: {name}")
        verify(rec["model"])
        return lgb.Booster(model_file=str(model))
    if model.exists() or model.with_suffix(".tmp").exists():
        raise ValueError(f"Unrecorded saved model {model}; preserve and inspect, never refit silently")
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"trees: {'reuse' if reuse else 'fit'} {name} ({rounds} rounds, {len(y)} rows)", flush=True)
    if reuse:
        src = verify(reuse)
        shutil.copyfile(src, model)
        booster = lgb.Booster(model_file=str(model))
    else:
        save(folder / f"{name}.started.json", dict(config=cfg, at=time.time()))
        booster = lgb.train(params, lgb.Dataset(X, np.asarray(y, float),
                            categorical_feature=list(cfg["categories"])), num_boost_round=rounds)
        temp = model.with_suffix(".tmp")
        booster.save_model(str(temp))
        temp.replace(model)
    if booster.feature_name() != list(X) or booster.current_iteration() != rounds:
        raise ValueError(f"Saved model shape/rounds differ: {name}")
    # Reload immediately against the fitted object on deterministic train samples.
    sample = X.iloc[np.linspace(0, len(X) - 1, min(4096, len(X)), dtype=int)]
    reloaded = lgb.Booster(model_file=str(model))
    np.testing.assert_array_equal(booster.predict(sample, num_threads=params["num_threads"]),
                                  reloaded.predict(sample, num_threads=params["num_threads"]))
    save(receipt, dict(config=cfg, model=artifact(model), retained_model=reuse,
                       seconds=time.monotonic()-started, reload_sample_exact=True,
                       runtime=dict(host=os.uname().nodename, lightgbm=lgb.__version__)))
    return booster


def fit_early(out, threads):
    import overnight_models as om
    from common import CATEGORICAL
    back = backend()
    packet = pd.read_pickle(EARLY_RUN / "v4/training.pkl")
    base, y = packet["X"][om.ORIGINAL_FEATURES], np.asarray(packet["y"], float)
    if len(y) != 2084659 or len(base.columns) != 27:
        raise ValueError("Original full-year 27-column training population differs")
    source = artifact(EARLY_RUN / "v4/training.pkl")
    for version in EARLY:
        cfg = configuration(version, threads)
        keep = (y >= 1) & (y <= cfg["max_taxi_sec"])
        X = base.loc[keep].copy()
        if version == 2:
            X = X.iloc[:, :18]
            for c in CATEGORICAL:
                X[c] = X[c].astype(object).astype("category")
            params = dict(back.PARAMS, seed=42, num_threads=threads)
        else:
            X, _ = back.fit_categories(X)
            meta = read(ROOT / f"submissions/unique-rose_v{version}.meta.json")
            if len(X) != meta["train_rows"] or list(X) != meta.get("features", om.ORIGINAL_FEATURES):
                raise ValueError(f"v{version} recorded feature order or row count differs")
            params = dict(back.PARAMS, seed=42, bagging_seed=43,
                          feature_fraction_seed=44, num_threads=threads)
        reuse = read(EARLY_RUN / "v5/trained.json")["model"] if version == 5 else None
        if reuse:
            # Retained v5 was fitted at eight threads. Keep its actual provenance.
            params["num_threads"] = 8
            retained = pd.read_pickle(EARLY_RUN / "v5/training.pkl")
            pd.testing.assert_frame_equal(X, retained["X"], check_exact=True)
            np.testing.assert_array_equal(y[keep], retained["y"])
            del retained
        model = fit_saved(out, f"v{version}", X, y[keep], params, cfg["rounds"], [source], reuse)
        del X, model
        gc.collect()
    del base, packet, y
    gc.collect()


def fit_hybrids(out, threads):
    import overnight_models as om
    import rome_hybrid as rh
    import rome_signals as signals
    import rome_hybrid_submit as recipes
    import rome_signals_submit as v29
    packet = pd.read_pickle(STAND_RUN / "training.pkl")
    X, y, raw = packet["X"], np.asarray(packet["y"], float), packet["raw"]
    if not X.index.equals(raw.index) or len(y) != 2084659:
        raise ValueError("Hybrid full-year row alignment differs")
    np.testing.assert_array_equal(y, raw.taxi_sec.to_numpy(float))
    s = rh.substituted(X, y, raw)
    lirf = raw.ADEP_mvt.eq("LIRF").to_numpy()
    if int((~s).sum()) != 2051353 or lirf.sum() != 160704:
        raise ValueError("Hybrid ordinary/classifier populations differ")
    source = artifact(STAND_RUN / "training.pkl")
    for v in HYBRID:
        extra = recipes.RECIPES[v]["extra"] if v < 29 else v29.ORDINARY
        columns = rh.regressor_features("clock", extra)
        model = fit_saved(out, f"v{v}_ordinary", X.loc[~s, columns], y[~s],
                          om.model_params("control", 42, threads), 800, [source])
        del model
        gc.collect()
    for v, extra in [(26, ()), (29, v29.CLASSIFIER)]:
        A = signals.classifier_matrix(X.loc[lirf], raw.loc[lirf], "clock", extra)
        params = dict(rh.CLASSIFIER_PARAMS, seed=42, bagging_seed=43,
                      feature_fraction_seed=44, num_threads=threads)
        model = fit_saved(out, f"v{v}_classifier", A, s[lirf].astype(float), params,
                          rh.CLASSIFIER_ROUNDS, [source])
        del model, A
        gc.collect()
    actual = signals.fit_parameters(X, y, raw)
    recorded = read(STAND_RUN / "preflight.json")["parameters"]
    prior = read(v29.MANIFEST)["model"]["parameters"]
    if actual != recorded or actual != prior:
        raise ValueError("Original full-year Rome late calibration differs")
    save(out / "hybrid_parameters.json", dict(parameters=actual, source=source))
    del packet, X, y, raw
    gc.collect()


def fit_coordinates(out, threads):
    import final_missing_coordinates_reconstruct as coords
    identity = read(COORD_RUN / "identity.json")
    packet = pd.read_pickle(IDEAS)
    columns = identity["ordinary_features"]
    X, y = packet["Xn"][columns], np.asarray(packet["yn"], float)
    if len(X) != 2051327 or len(columns) != 112:
        raise ValueError("v43/v44 full-year ordinary population differs")
    # This derives and checks raw IDs/labels against the already audited membership.
    ids = coords.parent.training_ids(packet)
    if len(ids) != len(y):
        raise ValueError("v43/v44 target membership differs")
    source = artifact(IDEAS)
    categories = read(coords.FOLDERS[43] / "categories.json")
    for name, values in categories.items():
        if name in X and X[name].cat.categories.tolist() != values:
            raise ValueError(f"Coordinate category order differs: {name}")
    for v in (43, 44):
        params = dict(identity["params"][str(v)], num_threads=threads)
        model = fit_saved(out, f"v{v}_ordinary", X, y, params,
                          800 if v == 43 else 5000, [source])
        del model
        gc.collect()
    del packet, X, y
    gc.collect()


def predictions_equal(model, replay, X, threads):
    pieces = []
    for start in range(0, len(X), 20000):
        part = X.iloc[start:start+20000]
        a = model.predict(part, num_threads=threads)
        b = replay.predict(part, num_threads=threads)
        np.testing.assert_array_equal(a, b)
        pieces.append(a)
    return np.concatenate(pieces) if pieces else np.empty(0)


def integer_frame(ids, frames, versions):
    ids = np.asarray(ids)
    if pd.Index(ids).has_duplicates:
        raise ValueError("Template IDs are not unique")
    out = pd.DataFrame({"mvt_id": ids})
    for version in versions:
        frame = pd.concat(frames[version], ignore_index=True)
        if frame.mvt_id.duplicated().any() or len(frame) != len(ids) or set(frame.mvt_id) != set(ids):
            raise ValueError(f"v{version} ID coverage differs")
        pred = frame.set_index("mvt_id").loc[ids, "pred"].to_numpy(float)
        if not np.isfinite(pred).all():
            raise ValueError(f"v{version} has nonfinite predictions")
        rounded = np.rint(np.maximum(pred, 0))
        if np.any(rounded > np.iinfo(np.int32).max):
            raise ValueError(f"v{version} prediction exceeds final template range")
        out[f"v{version}"] = rounded.astype(np.int64)
    return out


def infer_view(out, view, threads):
    import rome_hybrid as rh
    import rome_signals as signals
    import rome_hybrid_submit as recipes
    import rome_signals_submit as v29
    import final_missing_coordinates_reconstruct as coords
    destination = out / "components" / f"{view}.parquet"
    receipt = destination.with_suffix(".json")
    if receipt.exists():
        record = read(receipt)
        if record["identity_sha256"] != sha(out / "identity.json"):
            raise ValueError("Inference identity changed")
        verify(record["prediction"])
        for rec in record["model_receipts"]:
            verify(rec)
        return record
    if destination.exists():
        raise ValueError(f"Unrecorded predictions: {destination}")
    data = pd.read_pickle(FINAL_RUN / f"work/inputs_{view}.pkl")
    template = ROOT / "data" / ("submitting.parquet" if view == "old" else "final_submitting.parquet")
    ids = pd.read_parquet(template, columns=["MVT_ID_mvt"]).MVT_ID_mvt.to_numpy()
    if len(ids) != (344841 if view == "old" else 670790):
        raise ValueError("Unexpected template row count")
    frames = {v: [] for v in VERSIONS}
    model_pairs, records = {}, {}
    for path in sorted((out / "models").glob("*.json")):
        if path.name.endswith(".started.json"):
            continue
        rec = read(path)
        model_path = verify(rec["model"])
        records[path.stem] = rec
        model_pairs[path.stem] = (lgb.Booster(model_file=str(model_path)),
                                  lgb.Booster(model_file=str(model_path)))
    classifier_source = next(r for r in read(COORD_RUN / "identity.json")["sources"]
                             if r["path"].endswith("models/refit2025__classifier__42.txt"))
    clf_path = verify(classifier_source)
    coord_classifier = (lgb.Booster(model_file=str(clf_path)), lgb.Booster(model_file=str(clf_path)))
    coordinate_records = coords.originals()
    late_mean = read(out / "hybrid_parameters.json")["parameters"]["late_mean"]
    for month, item in data["months"].items():
        V, raw = item["V"], item["raw"]
        if not V.index.equals(raw.index):
            raise ValueError("Ranking raw/feature alignment differs")
        print(f"trees: predict {view} {month}", flush=True)
        for v in EARLY:
            name = f"v{v}"
            cfg = records[name]["config"]
            B = apply_categories(V[cfg["columns"]], cfg["categories"])
            pred = np.maximum(predictions_equal(*model_pairs[name], B, threads), 0)
            frames[v].append(pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), pred=pred)))
        probs = {}
        for v, extra in [(26, ()), (29, v29.CLASSIFIER)]:
            lirf = raw.ADEP_mvt.eq("LIRF").to_numpy() & np.isfinite(V[rh.GAP].to_numpy(float))
            p = np.zeros(len(raw))
            if lirf.any():
                B = signals.classifier_matrix(V.loc[lirf], raw.loc[lirf], "clock", extra)
                name = f"v{v}_classifier"
                B = apply_categories(B, records[name]["config"]["categories"])
                p[lirf] = predictions_equal(*model_pairs[name], B, threads)
            probs[v] = p
        for v in HYBRID:
            name = f"v{v}_ordinary"
            cfg = records[name]["config"]
            B = apply_categories(V[cfg["columns"]], cfg["categories"])
            ordinary = np.maximum(predictions_equal(*model_pairs[name], B, threads), 0)
            p = probs[26 if v < 29 else 29]
            pred = signals.combine(ordinary, p, V, raw,
                                   late_mean=late_mean if v == 29 else None)["final"]
            frames[v].append(pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), pred=pred)))
        # Preserve exactly the coordinate recipe's retained 68-input classifier,
        # training-only corrections, rule routing and final rounding order.
        p = signals.substitution_probability(coord_classifier[0], V, raw, "clock", threads,
                                              coords.original.cv.CLASSIFIER)
        np.testing.assert_array_equal(p, signals.substitution_probability(
            coord_classifier[1], V, raw, "clock", threads, coords.original.cv.CLASSIFIER))
        for v in (43, 44):
            name = f"v{v}_ordinary"
            cfg = records[name]["config"]
            B = apply_categories(V[cfg["columns"]], cfg["categories"])
            ordinary = np.maximum(predictions_equal(*model_pairs[name], B, threads), 0)
            prior = coordinate_records[v]["training"]
            pred = signals.combine(ordinary, p, V, raw, late_mean=prior["parameters"]["late_mean"])["final"]
            frame, _ = coords.original.v42.parent.apply_corrections(pred, raw, V, prior["corrections"], p)
            frames[v].append(frame[["mvt_id", "pred"]])
    frame = integer_frame(ids, frames, VERSIONS)
    save_frame(destination, frame)
    diagnostics = {}
    if view == "old":
        cache = read(ROOT / "data/derived/scored_submission_cache_20261011/manifest.json")
        for v in VERSIONS:
            prior = cache[str(v)]
            path = ROOT / prior["file"]
            if sha(path) != prior["sha256"]:
                raise ValueError(f"Historical v{v} predictions changed")
            old = pd.read_parquet(path).set_index("MVT_ID_mvt").loc[ids, "TAXITIME_SEC_mvt"].to_numpy(float)
            delta = frame[f"v{v}"].to_numpy(float) - old
            diagnostics[str(v)] = dict(changed_rows=int(np.count_nonzero(delta)),
                rms_difference=float(np.sqrt(np.mean(delta ** 2))),
                max_difference=float(np.max(np.abs(delta))),
                used_for_selection=False)
    record = dict(identity_sha256=sha(out / "identity.json"), prediction=artifact(destination),
                  rows=len(frame), versions=list(VERSIONS), view=view, all_model_reload_predictions_exact=True,
                  finite_nonnegative_integer=True, template_order_exact=True,
                  model_receipts=[artifact(p) for p in sorted((out / "models").glob("*.json"))
                                  if not p.name.endswith(".started.json")],
                  retained_coordinate_classifier=classifier_source, diagnostic_old_comparison=diagnostics)
    save(receipt, record)
    del data, frame, frames, model_pairs, coord_classifier
    gc.collect()
    return record


def run(out, threads):
    out = Path(out) / "trees"
    freeze(out, threads)
    result_path = out / "result.json"
    if result_path.exists():
        result = read(result_path)
        if result["identity"]["sha256"] != sha(out / "identity.json"):
            raise ValueError("Completed tree identity changed")
        for item in result["predictions"].values():
            verify(item)
        for item in result["models"]:
            verify(item)
        print("trees: completed artifacts verified; nothing refitted", flush=True)
        return
    with threadpool_limits(threads):
        fit_early(out, threads)
        fit_hybrids(out, threads)
        fit_coordinates(out, threads)
        reports = {v: infer_view(out, v, threads) for v in ("old", "final")}
    save(result_path, dict(status="pass", identity=artifact(out / "identity.json"),
                          predictions={v: reports[v]["prediction"] for v in reports},
                          reports={v: artifact(out / "components" / f"{v}.json") for v in reports},
                          models=[artifact(p) for p in sorted((out / "models").glob("*.txt"))],
                          versions=list(VERSIONS), caveat=CAVEAT, no_upload=True,
                          no_historical_exactness_claim=True, no_new_hyperparameter_selection=True))
    print("trees: all 12 original recipe components complete", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16")
    if args.action == "plan":
        print(json.dumps(plan(args.threads), indent=2))
    else:
        run(args.out, args.threads)


if __name__ == "__main__":
    main()
