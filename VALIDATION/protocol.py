"""Fold definitions and configuration for validation protocol v3."""

from dataclasses import dataclass, replace

import pandas as pd

VERSION = "v3"
SCORE_VERSION = "v3-flight-loss-day-bootstrap-1"
TAIL_SEC = 7200
MAX_TAXI_SEC = 200_000


@dataclass(frozen=True)
class Fold:
    name: str
    role: str
    train_end: str
    val: str
    freeze: str
    fixed_months: int | None = None
    departure_history: bool = True

    @property
    def gap(self):
        return pd.Period(self.val, "M").ordinal - pd.Period(self.train_end, "M").ordinal


def monthly(name, role, train_end, val, **kwargs):
    return Fold(name, role, train_end, val, train_end, **kwargs)


A_FOLDS = [monthly(f"A{i}", "A", str(pd.Period(f"2025-{i+5:02}", "M").end_time.date()),
                   f"2025-{i+6:02}") for i in range(1, 7)]
B_FOLDS = [monthly(f"B{i}", "B", end, val) for i, (end, val) in enumerate([
    ("2025-06-30", "2025-09"), ("2025-06-30", "2025-10"),
    ("2025-06-30", "2025-11"), ("2025-06-30", "2025-12"),
    ("2025-07-31", "2025-10"), ("2025-07-31", "2025-11"),
    ("2025-07-31", "2025-12"), ("2025-08-31", "2025-11"),
    ("2025-08-31", "2025-12"), ("2025-09-30", "2025-12"),
], 1)]
FOLDS = A_FOLDS + B_FOLDS + [
    monthly("G7", "diagnostic", "2025-05-31", "2025-12", fixed_months=5),
    monthly("F1", "diagnostic", "2025-11-30", "2025-12", fixed_months=11),
    monthly("W1", "retrospective", "2025-12-31", "2025-01",
            fixed_months=11, departure_history=False),
    replace(A_FOLDS[0], name="S1", role="feature_staleness", freeze="2025-01-31"),
    monthly("apr", "legacy", "2025-03-31", "2025-04"),
]
MAIN_NAMES = [f.name for f in A_FOLDS + B_FOLDS]
DEFAULT_FOLDS = ",".join(MAIN_NAMES)
ALIASES = dict(zip(["jul_fresh", "aug", "sep", "oct", "nov", "dec_jan26"],
                   [f.name for f in A_FOLDS]))
ALIASES["jul_stale"] = "S1"


def train_start(fold, months):
    months = fold.fixed_months if fold.fixed_months is not None else months
    if months == 0:
        return pd.Timestamp("2025-01-01")
    return pd.Timestamp(fold.train_end).replace(day=1) - pd.DateOffset(months=months - 1)


def select_folds(spec=DEFAULT_FOLDS, months=6):
    if months < 0:
        raise ValueError("--train-months must be nonnegative (0 means expanding)")
    names = []
    by_name = {f.name.lower(): f for f in FOLDS}
    for token in (spec or DEFAULT_FOLDS).split(","):
        token = token.strip().lower()
        expanded = {"a": [f.name for f in A_FOLDS], "b": [f.name for f in B_FOLDS],
                    "main": MAIN_NAMES, "all": [f.name for f in FOLDS if f.name != "apr"]}.get(token)
        for name in expanded or [ALIASES.get(token, token)]:
            if name.lower() not in by_name:
                raise ValueError(f"unknown fold: {name}; choose from {', '.join(f.name for f in FOLDS)}")
            names.append(by_name[name.lower()].name)
    selected, skipped = [], []
    for fold in FOLDS:
        if fold.name not in names:
            continue
        start = train_start(fold, months)
        if start < pd.Timestamp("2025-01-01"):
            skipped.append(fold.name)
        else:
            selected.append((fold, start))
    if not selected:
        raise ValueError("no runnable folds within 2025")
    return selected, skipped


def window_key(fold, start):
    return (str(start.date()), fold.train_end, fold.departure_history)
