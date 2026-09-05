"""Train the LightGBM taxi-out model on raw challenge columns.

The round count is fixed (--rounds); there is no early stopping, so nothing
about the model is tuned on the holdout. The Nov+Dec 2025 holdout RMSE is
reported for reference only — the leaderboard is the real validation. The
saved booster is refit on all of 2025.

Run: .venv/bin/python pipeline/train.py
"""

import argparse
import json
from datetime import datetime, timezone

import lightgbm as lgb
from tqdm import tqdm

from common import (
    CATEGORICAL,
    MAX_TAXI_SEC,
    MIN_TAXI_SEC,
    MODELS,
    RAW_COLS,
    TRAIN_GLOB,
    build_features,
    connect,
    rmse,
)

HOLDOUT_START = "2025-11-01"  # validation = Nov + Dec 2025

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
}
NUM_ROUNDS = 3000


def load_departures(con):
    return con.sql(
        f"""
        SELECT TAXITIME_SEC_mvt AS taxi_sec, {RAW_COLS}
        FROM read_parquet('{TRAIN_GLOB}')
        WHERE PHASE_mvt = 'DEP'
          AND TAXITIME_SEC_mvt BETWEEN {MIN_TAXI_SEC} AND {MAX_TAXI_SEC}
        """
    ).df()


def progress_bar(total, desc):
    """LightGBM callback pair: a tqdm bar over boosting rounds."""
    bar = tqdm(total=total, desc=desc, unit="round")

    def cb(env):
        bar.update(env.iteration + 1 - bar.n)
        if env.evaluation_result_list:
            _, metric, value, _ = env.evaluation_result_list[0]
            bar.set_postfix_str(f"holdout {metric} {value:.1f} s")

    return bar, cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="raw", help="model name: models/lgbm_<tag>.txt")
    ap.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    ap.add_argument("--no-holdout", action="store_true",
                    help="skip the Jan–Oct fit and its Nov+Dec report; just refit on all 2025")
    args = ap.parse_args()

    con = connect()
    print("loading 2025 departures…")
    df = load_departures(con)
    val_mask = (df["MVT_TIME_UTC_mvt"] >= HOLDOUT_START).to_numpy()
    print(f"departures: {len(df):,} (train {(~val_mask).sum():,} | holdout {val_mask.sum():,})")

    X, y = build_features(df), df["taxi_sec"]
    categories = {}
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
        categories[c] = X[c].cat.categories.tolist()

    score = None
    if not args.no_holdout:
        dtrain = lgb.Dataset(X[~val_mask], y[~val_mask], categorical_feature=CATEGORICAL)
        dval = lgb.Dataset(X[val_mask], y[val_mask], reference=dtrain)
        bar, cb = progress_bar(args.rounds, "fit Jan–Oct")
        booster = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=args.rounds,
            valid_sets=[dval],
            valid_names=["holdout"],
            callbacks=[cb],
        )
        bar.close()
        pred = booster.predict(X[val_mask])
        score = rmse(y[val_mask], pred)
        print(f"holdout RMSE (Nov+Dec 2025), reference only: {score:.1f} s")

        per_airport = (
            X[val_mask]
            .assign(taxi_sec=y[val_mask], pred=pred)
            .groupby("airport", observed=True)
            .apply(lambda g: rmse(g["taxi_sec"], g["pred"]), include_groups=False)
        )
        for airport, r in per_airport.sort_values().items():
            print(f"  {airport}: {r:7.1f} s")

    # Refit on the full year for the artifact used to predict Jan/Jul 2026.
    bar, cb = progress_bar(args.rounds, "refit 2025")
    final = lgb.train(PARAMS, lgb.Dataset(X, y, categorical_feature=CATEGORICAL),
                      num_boost_round=args.rounds, callbacks=[cb])
    bar.close()

    MODELS.mkdir(exist_ok=True)
    model_path = MODELS / f"lgbm_{args.tag}.txt"
    final.save_model(str(model_path))
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": len(df),
        "target_filter_sec": [MIN_TAXI_SEC, MAX_TAXI_SEC],
        "holdout_start": HOLDOUT_START,
        "holdout_rmse_sec": round(score, 1) if score is not None else None,
        "num_rounds": args.rounds,
        "params": PARAMS,
        "categories": categories,
    }
    model_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {model_path.relative_to(model_path.parents[1])} (+ .meta.json, refit on all 2025)")


if __name__ == "__main__":
    main()
