"""Fresh fold-scoped tree and Rome components for final-month model selection.

This module has no executable entry point. Nothing trains on import. All ordinary
models consume the new fold matrices supplied by final_month_inputs; historical
prediction caches and fitted models are never used. Hyperparameters are frozen
from the selected LB1/P04/K07/R20 and public Rome CatBoost recipes.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd


COMPONENTS = ("main156", "main162", "main174", "main184", "airport168",
              "cat174", "rome_lgb", "rome_cat")
GROUPS = {"main156": "combo156", "main162": "calls162", "main174": "public174",
          "main184": "daily184", "airport168": "airport168", "cat174": "public174"}
ROME_CAT_ROUNDS = {"2025-01": 1182, "2025-02": 1182,
                   "2025-06": 1401, "2025-07": 1401}
PARAMETER_PROVENANCE = {
    "main": "v3_20261011_public_daily_taxi_lgb/predictions/2025-01_s42.json",
    "airport": "v3_20261011_public_airport_trees/members/2025-01_EDDF_s42.json",
    "cat": "v3_20261011_catboost_full_inputs/predictions/2025-01_s42.json",
    "rome_cat": "v3_20261011_rome_public_delay_cat/predictions/2025-01_s42.json",
    "rome_lgb": "v3_20261005_hpo_jan_jul/rome.json:trials.R20",
}


def component_config(component, month, seed=42, threads=4):
    """Return a JSON-compatible recipe without reading data or creating a model."""
    if component not in COMPONENTS or month not in ROME_CAT_ROUNDS:
        raise ValueError((component, month))
    if threads < 1 or seed < 0:
        raise ValueError("Positive threads and nonnegative seed required")
    if component in ("cat174", "rome_cat"):
        classifier = component == "rome_cat"
        params = dict(loss_function="Logloss" if classifier else "RMSE",
            depth=6 if classifier else 8, learning_rate=.05 if classifier else .1,
            l2_leaf_reg=10. if classifier else 3, random_seed=seed,
            task_type="GPU", devices="0", thread_count=threads,
            gpu_ram_part=.65 if classifier else .9, allow_writing_files=False,
            verbose=500 if classifier else 1000)
        if classifier:
            params["eval_metric"] = "Logloss"
        else:
            params.update(border_count=254, bootstrap_type="Bernoulli", subsample=.7)
        return dict(component=component, family="catboost", classifier=classifier,
            params=params, rounds=ROME_CAT_ROUNDS[month] if classifier else 10000,
            stopping="fixed; no held-target access", seed=seed,
            columns=80 if classifier else 174,
            seasonal_rounds_note="Jan/Feb inherit 1182; Jun/Jul inherit 1401" if classifier else None)
    classifier = component == "rome_lgb"
    params = dict(objective="binary" if classifier else "regression",
        metric="binary_logloss" if classifier else "rmse", learning_rate=.02,
        num_leaves=127, min_data_in_leaf=50 if classifier else 200,
        feature_fraction=.6 if classifier else .7, bagging_fraction=.8,
        bagging_freq=1, verbosity=-1, seed=seed, bagging_seed=seed+1,
        feature_fraction_seed=seed+2, num_threads=threads)
    if not classifier:
        params.update(lambda_l2=100., path_smooth=0., cat_smooth=100., cat_l2=10.,
            min_data_per_group=100, max_cat_threshold=8)
    if component == "airport168":
        params.update(num_leaves=63, min_data_in_leaf=50, lambda_l2=10., max_cat_threshold=16)
    return dict(component=component, family="lightgbm", classifier=classifier,
        params=params, rounds=1700 if classifier else 5000 if component == "airport168" else 10000,
        columns=68 if classifier else int(component[-3:]), seed=seed,
        stopping="fixed; no held-target access")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=_sha(path))


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _ids_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _cats(frame):
    return [c for c in frame if isinstance(frame[c].dtype, pd.CategoricalDtype)]


def _cat_frame(frame, cats):
    result = frame.copy()
    for c in cats:
        result[c] = result[c].astype(object).where(result[c].notna(), "__missing__").astype(str)
    return result


def _columns(data, component):
    groups = data["feature_groups"]
    if component == "airport168" and "airport168" not in groups:
        return list(groups["combo156"]) + [c for c in groups["public174"] if c not in groups["calls162"]]
    return list(groups[GROUPS[component]])


def validate_fold_membership(data):
    """Check the new cyclic one-month embargo independently of the builder."""
    month = data["meta"]["month"]
    number = int(month[-2:])
    excluded = {number, (number-2) % 12+1, number % 12+1}
    allowed = {f"2025-{m:02d}" for m in range(1, 13) if m not in excluded}
    if set(data["meta"]["train_months"]) != allowed:
        raise ValueError("Input manifest does not use the requested one-month embargo")
    for key in ("train_raw", "core_raw"):
        actual = set(pd.to_datetime(data[key].MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m"))
        if actual != allowed:
            raise ValueError(f"Unexpected training-month membership: {key}")
    held = set(pd.to_datetime(data["raw"].MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m"))
    if held != {month}:
        raise ValueError("Held rows do not match the requested month")
    np.testing.assert_array_equal(data["train_raw"].mvt_id, data["train_ids"])
    np.testing.assert_array_equal(data["raw"].mvt_id, data["ids"])
    np.testing.assert_array_equal(data["core_raw"].mvt_id, data["core_ids"])
    if np.intersect1d(data["core_ids"], data["ids"]).size:
        raise ValueError("Classifier training overlaps held IDs")


def _rome_data(data, public):
    import rome_hybrid
    import rome_signals
    import tuning_seasonal as ts
    X, raw = data["core_X"], data["core_raw"]
    V, held = data["core_V"], data["raw"]
    mask = raw.ADEP_mvt.eq("LIRF").to_numpy()
    use = held.ADEP_mvt.eq("LIRF").to_numpy() & np.isfinite(V[rome_hybrid.GAP].to_numpy(float))
    target = rome_hybrid.substituted(X.loc[mask], np.asarray(data["core_y"])[mask], raw.loc[mask]).astype(float)
    if set(np.unique(target)) != {0., 1.}:
        raise ValueError("Rome training must contain both recording outcomes")
    A = rome_signals.classifier_matrix(X.loc[mask], raw.loc[mask], "clock", ts.CLASSIFIER)
    B = rome_signals.classifier_matrix(V.loc[use], held.loc[use], "clock", ts.CLASSIFIER)
    if A.shape[1] != 68 or list(A) != list(B):
        raise ValueError("Original R20 68-column contract changed")
    if public:
        extra_a, extra_b = data["core_public"].loc[mask], data["held_public"].loc[use]
        if extra_a.shape[1] != 12 or list(extra_a) != list(extra_b):
            raise ValueError("Expected twelve public Rome features")
        A = pd.concat([A, extra_a.set_axis(A.index)], axis=1)
        B = pd.concat([B, extra_b.set_axis(B.index)], axis=1)
    return A, target, B, np.asarray(data["core_ids"])[mask], np.asarray(data["ids"])[use], use


def _fit_one(X, y, V, train_ids, held_ids, output, config, input_artifact):
    """Persist the trained model before inference; resume only verified artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cats = _cats(X)
    if list(X) != list(V) or len(X.columns) != config["columns"]:
        raise ValueError("Training/held feature contract differs")
    if len(set(train_ids)) != len(train_ids) or len(set(held_ids)) != len(held_ids):
        raise ValueError("Nonunique movement IDs")
    if np.intersect1d(train_ids, held_ids).size or len(y) != len(X):
        raise ValueError("Training/held overlap or target misalignment")
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite training targets")
    source = _artifact(__file__)
    contract = dict(config=config, source=source, input=input_artifact,
        columns=list(X), categoricals=cats, train_rows=len(X), held_rows=len(V),
        train_ids_sha256=_ids_digest(train_ids), held_ids_sha256=_ids_digest(held_ids))
    contract_path = output / "contract.json"
    if contract_path.exists():
        if _read(contract_path) != contract:
            raise ValueError(f"Existing fit contract changed: {output}")
    else:
        if any(output.iterdir()):
            raise ValueError(f"Unrecorded outputs: {output}")
        _write(contract_path, contract)
    suffix = ".cbm" if config["family"] == "catboost" else ".txt"
    model_path = output / ("model" + suffix)
    model_receipt = output / "checkpoint.json"
    prediction_path = output / "predictions.parquet"
    done_path = output / "result.json"
    if done_path.exists():
        result = _read(done_path)
        if (result["contract"] != _artifact(contract_path) or
                result["model"] != _artifact(model_path) or
                result["prediction"] != _artifact(prediction_path) or
                result["checkpoint"] != _artifact(model_receipt)):
            raise ValueError("Completed model/prediction provenance changed")
        frame = pd.read_parquet(prediction_path)
        np.testing.assert_array_equal(frame.mvt_id, held_ids)
        return frame.pred.to_numpy(float), result
    if config["family"] == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        cls = CatBoostClassifier if config["classifier"] else CatBoostRegressor
        X, V = _cat_frame(X, cats), _cat_frame(V, cats)
        def load():
            model = cls()
            model.load_model(str(model_path))
            return model
        def predict(model):
            pool = Pool(V, cat_features=cats)
            return (model.predict_proba(pool, thread_count=config["params"]["thread_count"])[:, 1]
                    if config["classifier"] else model.predict(pool, thread_count=config["params"]["thread_count"]))
        def fit():
            model = cls(iterations=config["rounds"], **config["params"])
            model.fit(Pool(X, y, cat_features=cats))
            if model.tree_count_ != config["rounds"]:
                raise ValueError("Unexpected CatBoost stopping")
            return model
    else:
        import lightgbm as lgb
        def load():
            return lgb.Booster(model_file=str(model_path))
        def predict(model):
            return model.predict(V, num_threads=config["params"]["num_threads"])
        def fit():
            model = lgb.train(config["params"], lgb.Dataset(X, y, categorical_feature=cats),
                              num_boost_round=config["rounds"])
            if model.current_iteration() != config["rounds"]:
                raise ValueError("Unexpected LightGBM stopping")
            return model
    started = time.monotonic()
    if model_receipt.exists():
        checkpoint = _read(model_receipt)
        if checkpoint["contract"] != _artifact(contract_path) or checkpoint["model"] != _artifact(model_path):
            raise ValueError("Saved checkpoint changed")
        model = load()
    else:
        if model_path.exists() or (output / "fit_started.json").exists():
            raise ValueError(f"Interrupted/unrecorded fit; inspect before repeating: {output}")
        _write(output / "fit_started.json", dict(contract=_artifact(contract_path),
               host=platform.node(), unix_time=time.time()))
        model = fit()
        temporary = model_path.with_suffix(model_path.suffix + ".tmp")
        model.save_model(str(temporary))
        temporary.replace(model_path)
        _write(model_receipt, dict(contract=_artifact(contract_path), model=_artifact(model_path),
            host=platform.node(), fit_seconds=time.monotonic()-started,
            saved_before_held_inference=True))
    # Empty airport-held subsets are allowed; the checkpoint is still retained.
    raw = np.asarray(predict(model), float) if len(V) else np.empty(0, float)
    restored = load()
    replay = np.asarray(predict(restored), float) if len(V) else np.empty(0, float)
    np.testing.assert_array_equal(raw, replay)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite model output")
    if config["classifier"]:
        if not ((raw >= 0) & (raw <= 1)).all():
            raise ValueError("Invalid probability")
        prediction = raw
    else:
        prediction = np.maximum(raw, 0).astype(np.float32).astype(float)
    frame = pd.DataFrame(dict(mvt_id=held_ids, pred=prediction))
    if prediction_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(prediction_path), frame, check_exact=True)
    else:
        temporary = prediction_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(prediction_path)
    result = dict(contract=_artifact(contract_path), model=_artifact(model_path),
        prediction=_artifact(prediction_path), checkpoint=_artifact(model_receipt),
        prediction_reload_exact=True, finite=True, rows=len(V),
        host=platform.node(), elapsed_seconds=time.monotonic()-started)
    _write(done_path, result)
    del model, restored
    gc.collect()
    return prediction, result


def fit_component(component, data, out, seed=42, threads=4):
    """Return (full-held prediction, receipt); caller owns all final scoring.

    ``data['_input_artifact']`` must describe the fresh inputs.pkl with an absolute
    path and SHA-256. The caller binds that cache to its frozen fold manifest.
    """
    artifact = data["_input_artifact"]
    if _artifact(artifact["path"]) != artifact:
        raise ValueError("Fresh input cache changed")
    validate_fold_membership(data)
    month = data["meta"]["month"]
    config = component_config(component, month, seed, threads)
    out = Path(out) / f"{component}_s{seed}"
    if component.startswith("rome_"):
        X, y, V, train_ids, held_ids, use = _rome_data(data, component == "rome_cat")
        values, receipt = _fit_one(X, y, V, train_ids, held_ids, out, config, artifact)
        result = np.zeros(len(data["ids"]), float)
        result[use] = values
        return result, dict(component=component, seed=seed, held_rows=len(result),
                            applicable_rows=int(use.sum()), member=receipt)
    columns = _columns(data, component)
    X, V = data["X"].loc[:, columns], data["V"].loc[:, columns]
    if component != "airport168":
        return _fit_one(X, np.asarray(data["y"], float), V, data["train_ids"],
                        data["ids"], out, config, artifact)
    train_airport = data["train_raw"].ADEP_mvt.to_numpy()
    held_airport = data["raw"].ADEP_mvt.to_numpy()
    if set(held_airport) - set(train_airport):
        raise ValueError("Held airport has no training population")
    result = np.full(len(V), np.nan)
    members = {}
    for airport in sorted(set(train_airport)):
        train_mask, held_mask = train_airport == airport, held_airport == airport
        prediction, receipt = _fit_one(X.loc[train_mask], np.asarray(data["y"])[train_mask],
            V.loc[held_mask], np.asarray(data["train_ids"])[train_mask],
            np.asarray(data["ids"])[held_mask], out / airport, config, artifact)
        result[held_mask] = prediction
        members[airport] = receipt
    if not np.isfinite(result).all():
        raise ValueError("Airport models did not cover every held row")
    return result, dict(component=component, seed=seed, rows=len(result), members=members)
