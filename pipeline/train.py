"""Train the baseline taxi-out model: per-airport mean with a global fallback.

Validation is a simple time holdout — fit on Jan–Oct 2025, score Nov–Dec 2025 —
mirroring the fact that the evaluation months (Jan/Jul 2026) lie after training.
The saved artifact is refit on all of 2025.

Run: .venv/bin/python pipeline/train.py
"""

import json
from datetime import datetime, timezone

from common import MAX_TAXI_SEC, MIN_TAXI_SEC, MODELS, TRAIN_GLOB, connect, rmse

HOLDOUT_START = "2025-11-01"  # validation = Nov + Dec 2025
MODEL_PATH = MODELS / "baseline_airport_mean.json"


def load_departures(con):
    return con.sql(
        f"""
        SELECT ADEP_mvt AS airport,
               MVT_TIME_UTC_mvt AS mvt_time,
               TAXITIME_SEC_mvt AS taxi_sec
        FROM read_parquet('{TRAIN_GLOB}')
        WHERE PHASE_mvt = 'DEP'
          AND TAXITIME_SEC_mvt BETWEEN {MIN_TAXI_SEC} AND {MAX_TAXI_SEC}
        """
    ).df()


def fit(df):
    """Return the baseline 'model': per-airport means plus a global fallback."""
    return {
        "global_mean": float(df["taxi_sec"].mean()),
        "airport_mean": df.groupby("airport")["taxi_sec"].mean().to_dict(),
    }


def predict(model, airports):
    return airports.map(model["airport_mean"]).fillna(model["global_mean"])


def main():
    con = connect()

    print("loading 2025 departures…")
    df = load_departures(con)
    val_mask = df["mvt_time"] >= HOLDOUT_START
    tr, val = df[~val_mask], df[val_mask]
    print(f"departures: {len(df):,} (train {len(tr):,} | holdout {len(val):,})")

    model = fit(tr)
    score = rmse(val["taxi_sec"], predict(model, val["airport"]))
    naive = rmse(val["taxi_sec"], model["global_mean"])
    print(f"holdout RMSE (Nov+Dec 2025): {score:.1f} s   [global-mean ref: {naive:.1f} s]")

    per_airport = (
        val.assign(pred=predict(model, val["airport"]))
        .groupby("airport")
        .apply(lambda g: rmse(g["taxi_sec"], g["pred"]), include_groups=False)
    )
    for airport, r in per_airport.sort_values().items():
        print(f"  {airport}: {r:7.1f} s")

    # Refit on the full year for the artifact used to predict Jan/Jul 2026.
    final = fit(df)
    final["meta"] = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": len(df),
        "target_filter_sec": [MIN_TAXI_SEC, MAX_TAXI_SEC],
        "holdout_start": HOLDOUT_START,
        "holdout_rmse_sec": round(score, 1),
    }
    MODELS.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(f"saved {MODEL_PATH.relative_to(MODEL_PATH.parents[1])} (refit on all 2025)")


if __name__ == "__main__":
    main()
