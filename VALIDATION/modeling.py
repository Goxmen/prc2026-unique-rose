"""Shared v3 feature availability, categorical fitting, and LightGBM training."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import CATEGORICAL, RAW_COLS, build_features  # noqa: E402
from features_time import TimeFeatures, STALE_ABLE, day_index, fpl_flags, _naive  # noqa: E402
from protocol import TAIL_SEC  # noqa: E402

PARAMS = {
    "objective": "regression", "metric": "rmse", "learning_rate": 0.05,
    "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.9,
    "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
}
RAW_NAMES = [c.strip() for c in RAW_COLS.split(",")]
HISTORY_FEATURES = list(STALE_ABLE) + ["op_taxi_30d_n"]


def period_rows(df, start, end):
    t = _naive(df["MVT_TIME_UTC_mvt"])
    return df.loc[(t >= pd.Timestamp(start)) & (t < pd.Timestamp(end) + pd.Timedelta(days=1))]


def month_rows(df, month):
    p = pd.Period(month, "M")
    return period_rows(df, p.start_time, p.end_time.normalize())


def observable(df):
    """Physically remove hidden labels and movement block times at the boundary."""
    return df.loc[:, RAW_NAMES]


def context(history_dep, history_arr, eval_dep=None, eval_arr=None, departure_history=True):
    """Labels belong to the training window; context sees only supplied frames."""
    dep = history_dep.loc[history_dep.taxi_sec.between(1, TAIL_SEC)]
    if not departure_history:
        dep = dep.iloc[:0]
    arr = history_arr if eval_arr is None else pd.concat([history_arr, eval_arr], ignore_index=True)
    events = observable(history_dep)
    if eval_dep is not None:
        events = pd.concat([events, observable(eval_dep)], ignore_index=True)
    return TimeFeatures(dep, arr.loc[arr.taxi_sec.between(1, TAIL_SEC)],
                        dep_events=events, arr_events=arr)


def matrix(df, tf, *, freeze_day, lag_days=0, fpl_flag=False, departure_history=True):
    raw = observable(df)
    time = tf.transform(raw, freeze_day=freeze_day, lag_days=lag_days)
    if not departure_history:
        time = time.drop(columns=HISTORY_FEATURES)
    parts = [build_features(raw), time]
    if fpl_flag:
        parts.append(fpl_flags(raw))
    return pd.concat(parts, axis=1)


def fit_categories(X):
    X = X.copy()
    categories = {}
    for c in CATEGORICAL:
        categories[c] = pd.Index(pd.unique(X[c].dropna()))
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X, categories


def apply_categories(X, categories):
    X = X.copy()
    for c, cats in categories.items():
        X[c] = pd.Categorical(X[c].where(X[c].isin(cats)), categories=cats)
    return X


def as_categorical(X_tr, X_va):
    X_tr, cats = fit_categories(X_tr)
    return X_tr, apply_categories(X_va, cats)


def training_matrix(dep, arr, config, departure_history=True):
    keep = dep.taxi_sec.between(1, config["max_taxi_sec"])
    if not keep.any():
        raise ValueError("no training rows survive the training-label filter")
    tf = context(dep, arr, departure_history=departure_history)
    X = matrix(dep.loc[keep], tf, freeze_day=None, lag_days=config["lag_train_days"],
               fpl_flag=config["fpl_flag"], departure_history=departure_history)
    X, cats = fit_categories(X)
    return X, dep.loc[keep, "taxi_sec"].to_numpy(float), cats


def validation_matrix(dep, arr, val_dep, val_arr, freeze, cats, config, departure_history=True):
    tf = context(dep, arr, val_dep, val_arr, departure_history)
    freeze_day = int((pd.Timestamp(freeze) - pd.Timestamp("2025-01-01")).days)
    X = matrix(val_dep, tf, freeze_day=freeze_day, fpl_flag=config["fpl_flag"],
               departure_history=departure_history)
    return apply_categories(X, cats)


def fit_booster(X, y, seed, rounds, config, desc, validation=None, training_history=None):
    params = dict(PARAMS, seed=seed, bagging_seed=seed + 1,
                  feature_fraction_seed=seed + 2, num_threads=config.get("threads", 0))
    train = lgb.Dataset(X, y, categorical_feature=CATEGORICAL)
    with tqdm(total=rounds, desc=desc, unit="rnd", leave=False) as bar:
        def progress(env):
            for name, metric, value, *_ in env.evaluation_result_list or []:
                if name == "training" and metric == "rmse":
                    bar.set_postfix(train_rmse=f"{value:.2f}", refresh=False)
            bar.update(env.iteration + 1 - bar.n)

        callbacks = [progress]
        kwargs = {}
        if training_history is not None:
            # Measure the training set itself; this introduces no holdout split.
            kwargs.update(valid_sets=[train], valid_names=["training"])
            callbacks.append(lgb.record_evaluation(training_history))
        if validation is not None:
            vx, vy = validation
            kwargs.setdefault("valid_sets", []).append(lgb.Dataset(vx, vy, reference=train))
            if training_history is not None:
                kwargs["valid_names"].append("validation")
            callbacks.append(lgb.early_stopping(config["early_stopping_rounds"], verbose=False))
        return lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)


def choose_rounds(dep, arr, seed, config, train_end, departure_history=True):
    if not config["early_stopping_rounds"]:
        return config["rounds"], None
    month = pd.Timestamp(train_end).strftime("%Y-%m")
    cutoff = pd.Period(month, "M").start_time - pd.Timedelta(days=1)
    start = _naive(dep.MVT_TIME_UTC_mvt).min().normalize()
    inner_dep, inner_arr = period_rows(dep, start, cutoff), period_rows(arr, start, cutoff)
    va, va_arr = month_rows(dep, month), month_rows(arr, month)
    if inner_dep.empty or va.empty:
        raise ValueError("inner round selection requires at least two populated training months")
    X, y, cats = training_matrix(inner_dep, inner_arr, config, departure_history)
    V = validation_matrix(inner_dep, inner_arr, va, va_arr, cutoff, cats, config, departure_history)
    booster = fit_booster(X, y, seed, config["rounds"], config, f"inner {month}/{seed}",
                          validation=(V, va.taxi_sec.to_numpy(float)))
    rounds = booster.best_iteration or config["rounds"]
    return rounds, {"train_end": str(cutoff.date()), "validation_month": month,
                    "rounds": rounds, "train_rows": len(y), "validation_rows": len(va)}


def predict(booster, X, num_iteration=None):
    return np.clip(np.concatenate([booster.predict(X.iloc[i:i + 20_000], num_iteration=num_iteration)
                                  for i in range(0, len(X), 20_000)]), 0, None)


def prediction_frame(val, X, train_start, train_end, month):
    t = pd.to_datetime(val.MVT_TIME_UTC_mvt, utc=True)
    days = t.dt.tz_localize(None).dt.normalize()
    p = pd.Period(month, "M")
    available = np.zeros(len(val), dtype=np.int8)
    for lag in range(1, 8):
        prev = days - pd.Timedelta(days=lag)
        available += (((prev >= pd.Timestamp(train_start)) & (prev <= pd.Timestamp(train_end))) |
                      ((prev >= p.start_time) & (prev <= p.end_time))).to_numpy(np.int8)
    return pd.DataFrame({
        "mvt_id": val.mvt_id.to_numpy(), "mvt_time": t.to_numpy(),
        "airport": val.ADEP_mvt.to_numpy(), "y": val.taxi_sec.to_numpy(float),
        "has_flight_plan": val.AOBT_3_flt.notna().to_numpy(),
        "arrival_history_available": X.arr_taxiin_7d.notna().to_numpy(),
        "arrival_history_days_available": available,
    })
