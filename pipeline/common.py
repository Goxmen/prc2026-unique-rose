"""Shared paths and helpers for the PRC 2026 pipeline."""

from pathlib import Path

import duckdb
import numpy as np

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
