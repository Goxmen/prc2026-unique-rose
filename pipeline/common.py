"""Shared paths and helpers for the PRC 2026 pipeline."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"

TRAIN_GLOB = str(DATA / "training_2025-*.parquet")
RANKING = DATA / "ranking.parquet"
TEMPLATE = DATA / "submitting.parquet"

TEAM = "unique-rose"

# Training-only target filter: drop physically implausible taxi-out times so
# they don't skew the fitted means. The evaluation rows are never filtered.
MIN_TAXI_SEC = 1
MAX_TAXI_SEC = 7200  # 2 h; drops ~900 of 2.08 M departure rows


# Raw columns the model consumes. Everything here is populated for DEP rows in
# ranking.parquet too (BLOCK_TIME_UTC_mvt is not, and must never appear: for
# departures it *is* the target's other endpoint).
RAW_COLS = """
    ADEP_mvt, ADES_mvt, AIRCRAFT_TYPE_mvt, RUNWAY_mvt, STAND_mvt,
    FLIGHT_RULE_mvt, MARKET_SEGMENT_flt, FLIGHT_TYPE_flt, WK_TBL_CAT_flt,
    AIRCRAFT_OPERATOR_flt, MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt,
    LOBT_flt, IOBT_flt, EOBT_1_flt, AOBT_3_flt
"""

CATEGORICAL = [
    "airport", "dest", "aircraft_type", "runway", "stand", "flight_rule",
    "market_segment", "flight_type", "wake_cat", "operator",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix from the RAW_COLS columns; same code path for training and
    ranking rows. Missing values stay NaN (LightGBM handles them natively)."""
    t = df["MVT_TIME_UTC_mvt"]
    X = pd.DataFrame(
        {
            "airport": df["ADEP_mvt"],
            "dest": df["ADES_mvt"],
            "aircraft_type": df["AIRCRAFT_TYPE_mvt"],
            # runway/stand IDs repeat across airports; scope them
            "runway": df["ADEP_mvt"] + ":" + df["RUNWAY_mvt"],
            "stand": df["ADEP_mvt"] + ":" + df["STAND_mvt"],
            "flight_rule": df["FLIGHT_RULE_mvt"],
            "market_segment": df["MARKET_SEGMENT_flt"],
            "flight_type": df["FLIGHT_TYPE_flt"],
            "wake_cat": df["WK_TBL_CAT_flt"],
            "operator": df["AIRCRAFT_OPERATOR_flt"],
            "hour": t.dt.hour,
            "dow": t.dt.dayofweek,
            "month": t.dt.month,
        },
        index=df.index,
    )
    # Takeoff minus each off-block estimate ≈ taxi-out plus that estimate's
    # error. AOBT_3 (actual off-block, flown trajectory) is the strong one;
    # the organizers left all of these populated in ranking.parquet.
    for name, col in [
        ("sched", "SCHED_TIME_UTC_mvt"),
        ("lobt", "LOBT_flt"),
        ("iobt", "IOBT_flt"),
        ("eobt1", "EOBT_1_flt"),
        ("aobt3", "AOBT_3_flt"),
    ]:
        X[f"mvt_minus_{name}_sec"] = (t - df[col]).dt.total_seconds()
    return X


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection in UTC with a terminal progress bar on long queries."""
    con = duckdb.connect()
    con.sql("SET TimeZone='UTC'")
    con.sql("SET enable_progress_bar=true")
    return con


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
