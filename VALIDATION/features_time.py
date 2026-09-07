"""Time-dependent features, carrying an explicit availability contract.

Every feature here belongs to one of three classes, and the class alone decides
whether the feature has to be frozen when we validate:

  STALE_ABLE  Built from past *departure taxi-outs*. In the 2026 ranking months
              the freshest such label we will ever own is 2025-12-31 (February
              to June were never released and January's are blanked because they
              are the other half of the test). So in any fold these must be
              frozen at the training cutoff for validation rows — never computed
              at the row's own date.

  FRESH_ROLL  Built from past *arrival taxi-ins*. Arrival rows keep both
              BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt in ranking.parquet
              (verified: 152,871 Jan + 191,822 Jul arrivals, all populated), so
              these stay current in the test months. Always causal, never frozen.

  FRESH_NOW  Movement counts in a time window around the row's own off-block,
              from timestamps that survive in ranking.parquet. No target
              involved, so nothing to freeze.

The freeze is the whole point of the harness, so it is a property of the data,
not a flag someone remembers to pass: `transform` refuses to run without an
explicit decision about the cutoff.
"""

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2025-01-01")

# 30-day windows on operator/runway/airport-hour; stands are sparser, so 90.
STALE_ABLE = {
    "op_taxi_30d": 30,
    "rwy_taxi_30d": 30,
    "aphour_taxi_30d": 30,
    "stand_taxi_90d": 90,
}
FRESH_ROLL = ["arr_taxiin_7d"]
FRESH_NOW = ["dep_actual_30m_before", "arr_actual_30m_before", "dep_sched_60m_around"]

TIME_FEATURES = list(STALE_ABLE) + ["op_taxi_30d_n"] + FRESH_ROLL + FRESH_NOW
FPL_FEATURES = ["has_flight_plan"]


def _naive(s: pd.Series) -> pd.Series:
    """DuckDB hands back tz-aware or naive depending on the column type."""
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if s.dt.tz is not None else s


def day_index(ts: pd.Series) -> np.ndarray:
    return ((_naive(ts) - EPOCH).dt.days).to_numpy()


def obt_ref(df: pd.DataFrame) -> pd.Series:
    """Best off-block estimate that survives in ranking.parquet for DEP rows.

    BLOCK_TIME_UTC_mvt is blanked there, so we anchor on the flown-trajectory
    actual off-block and fall back through the filed estimates.
    """
    out = _naive(df["AOBT_3_flt"])
    for col in ("LOBT_flt", "IOBT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"):
        out = out.fillna(_naive(df[col]))
    return out.fillna(_naive(df["MVT_TIME_UTC_mvt"]))


def fpl_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Whether this movement matched a Network Manager flight plan.

    Not a time feature, but it belongs to the same availability contract: the
    _flt columns are populated for 98.5% of DEP rows in ranking.parquet, so this
    is knowable at prediction time. It matters because a missing match is where
    the corrupt labels live — NULL AOBT_3_flt covers 82% of the ~24 h rows but
    only 1% of normal ones, and at LIRF it lifts the rate from 0.03% to 29.5%.
    Today that signal only reaches the model as a NaN buried inside
    mvt_minus_aobt3_sec; this makes it something a tree can split on directly.
    """
    return pd.DataFrame(
        {"has_flight_plan": df["AOBT_3_flt"].notna().astype("int8")}, index=df.index
    )


def _prefix(codes, days, values, n_codes, n_days):
    """Per-key daily sums/counts, prefix-summed so any window is one subtraction."""
    keep = codes >= 0
    s = np.zeros((n_codes, n_days))
    c = np.zeros((n_codes, n_days))
    np.add.at(s, (codes[keep], days[keep]), values[keep])
    np.add.at(c, (codes[keep], days[keep]), 1.0)
    zero = np.zeros((n_codes, 1))
    return (np.concatenate([zero, np.cumsum(s, 1)], 1),
            np.concatenate([zero, np.cumsum(c, 1)], 1))


class _Key:
    """String key -> dense integer code; anything unseen becomes -1 (NaN feature)."""

    def __init__(self, values):
        self.index = pd.Index(pd.unique(pd.Series(values).dropna()))

    def encode(self, values):
        return self.index.get_indexer(pd.Series(values).to_numpy())

    def __len__(self):
        return len(self.index)


class TimeFeatures:
    """Fit the aggregate tables once on all of 2025, then serve any cutoff.

    Building on the full year is safe because a window sum never reads past its
    cutoff, and every cutoff we ask for is at or before the fold's training end.
    It means one build for the whole harness instead of one per fold.
    """

    def __init__(self, dep: pd.DataFrame, arr: pd.DataFrame, n_days: int | None = None,
                 *, dep_events: pd.DataFrame | None = None, arr_events: pd.DataFrame | None = None):
        """dep / arr        labelled movements -> the taxi-out and taxi-in windows.
        dep_events / arr_events  movement timestamps only, for the intra-day counts.

        The two are separate because they have different horizons at prediction
        time. Departure taxi-out labels stop at 2025-12-31 forever. But arrival
        taxi-ins and every movement timestamp exist inside the 2026 ranking
        months, so at submission time you pass the union of 2025 and ranking for
        `arr` and the `*_events` frames while `dep` stays 2025-only. Default them
        to dep/arr and you get the training-year behaviour the harness uses.
        """
        dep_events = dep if dep_events is None else dep_events
        arr_events = arr if arr_events is None else arr_events
        self._span = max(day_index(f["MVT_TIME_UTC_mvt"]).max()
                         for f in (dep, arr, dep_events, arr_events) if len(f))
        self.n_days = int(n_days or self._span + 1)
        dep_day = day_index(dep["MVT_TIME_UTC_mvt"])
        arr_day = day_index(arr["MVT_TIME_UTC_mvt"])

        self.k_op = _Key(dep["AIRCRAFT_OPERATOR_flt"])
        self.k_rwy = _Key(dep["ADEP_mvt"] + ":" + dep["RUNWAY_mvt"])
        self.k_stand = _Key(dep["ADEP_mvt"] + ":" + dep["STAND_mvt"])
        self.k_aphour = _Key(dep["ADEP_mvt"] + ":" + _naive(dep["MVT_TIME_UTC_mvt"]).dt.hour.astype(str))
        self.k_ap = _Key(pd.concat([dep["ADEP_mvt"], arr["ADES_mvt"],
                                    dep_events["ADEP_mvt"], arr_events["ADES_mvt"]], ignore_index=True))

        y = dep["taxi_sec"].to_numpy(dtype=float)
        self.tbl = {
            "op_taxi_30d": _prefix(self.k_op.encode(dep["AIRCRAFT_OPERATOR_flt"]), dep_day, y, len(self.k_op), self.n_days),
            "rwy_taxi_30d": _prefix(self.k_rwy.encode(dep["ADEP_mvt"] + ":" + dep["RUNWAY_mvt"]), dep_day, y, len(self.k_rwy), self.n_days),
            "stand_taxi_90d": _prefix(self.k_stand.encode(dep["ADEP_mvt"] + ":" + dep["STAND_mvt"]), dep_day, y, len(self.k_stand), self.n_days),
            "aphour_taxi_30d": _prefix(
                self.k_aphour.encode(dep["ADEP_mvt"] + ":" + _naive(dep["MVT_TIME_UTC_mvt"]).dt.hour.astype(str)),
                dep_day, y, len(self.k_aphour), self.n_days),
        }
        self.tbl["arr_taxiin_7d"] = _prefix(
            self.k_ap.encode(arr["ADES_mvt"]), arr_day,
            arr["taxi_sec"].to_numpy(dtype=float), len(self.k_ap), self.n_days)

        # Sorted event times per airport for the intra-day window counts. Both
        # sides are movement timestamps that ranking.parquet still carries.
        self._dep_obt = self._by_airport(dep_events["ADEP_mvt"], obt_ref(dep_events))
        self._arr_land = self._by_airport(arr_events["ADES_mvt"], _naive(arr_events["MVT_TIME_UTC_mvt"]))
        self._dep_sched = self._by_airport(dep_events["ADEP_mvt"], _naive(dep_events["SCHED_TIME_UTC_mvt"]))

    @staticmethod
    def _by_airport(airport, times):
        t64 = times.to_numpy("datetime64[ns]")
        ns, ok = t64.astype("int64"), ~np.isnat(t64)
        ap_arr = airport.to_numpy()
        return {ap: np.sort(ns[idx][ok[idx]])
                for ap, idx in pd.Series(ap_arr).groupby(ap_arr).indices.items()}

    def _counts(self, airport, anchor, table, lo_min, hi_min, exclude_self):
        """Events in the same airport within [t+lo_min, t+hi_min] minutes of each
        row's anchor. Ties at exactly t are inside the window, so `exclude_self`
        removes the row's own event (and only that one) when anchor and table are
        the same set of movements."""
        ns = anchor.to_numpy("datetime64[ns]").astype("int64")
        lo, hi = int(lo_min * 60e9), int(hi_min * 60e9)
        out = np.zeros(len(ns), dtype=np.float32)
        ap_arr = airport.to_numpy()
        for ap, idx in pd.Series(ap_arr).groupby(ap_arr).indices.items():
            events = table.get(ap)
            if events is None or len(events) == 0:
                continue
            t = ns[idx]
            n = (np.searchsorted(events, t + hi, "right")
                 - np.searchsorted(events, t + lo, "left")).astype(np.float32)
            out[idx] = n - 1.0 if exclude_self else n
        return np.maximum(out, 0)

    def _window(self, name, codes, cut):
        cs, cc = self.tbl[name]
        cut = np.clip(cut, -1, self.n_days - 1)
        hi = cut + 1
        lo = np.maximum(hi - STALE_ABLE.get(name, 7), 0)
        bad = codes < 0
        safe = np.where(bad, 0, codes)
        s = cs[safe, hi] - cs[safe, lo]
        c = cc[safe, hi] - cc[safe, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where((c > 0) & ~bad, s / np.maximum(c, 1), np.nan)
        return mean, np.where(bad, 0, c)

    def transform(self, df: pd.DataFrame, *, freeze_day, lag_days: int = 0) -> pd.DataFrame:
        """Time features for `df`.

        freeze_day : int  -> STALE_ABLE windows end on that day index. Use for
                             validation/test rows: the cutoff is the fold's
                             training end, never the row's own date.
                     None -> causal, window ends the day before each row. Use for
                             training rows.
        lag_days   : additionally push the causal cutoff back this many days.
                     Only meaningful with freeze_day=None; this is the knob for
                     the "should training features be stale too?" experiment.
        """
        d = day_index(df["MVT_TIME_UTC_mvt"])
        cut = np.full(len(df), int(freeze_day)) if freeze_day is not None else d - 1 - lag_days

        hour = _naive(df["MVT_TIME_UTC_mvt"]).dt.hour.astype(str)
        op_mean, op_n = self._window("op_taxi_30d", self.k_op.encode(df["AIRCRAFT_OPERATOR_flt"]), cut)
        out = pd.DataFrame(index=df.index)
        out["op_taxi_30d"] = op_mean
        out["op_taxi_30d_n"] = op_n
        out["rwy_taxi_30d"] = self._window("rwy_taxi_30d", self.k_rwy.encode(df["ADEP_mvt"] + ":" + df["RUNWAY_mvt"]), cut)[0]
        out["stand_taxi_90d"] = self._window("stand_taxi_90d", self.k_stand.encode(df["ADEP_mvt"] + ":" + df["STAND_mvt"]), cut)[0]
        out["aphour_taxi_30d"] = self._window("aphour_taxi_30d", self.k_aphour.encode(df["ADEP_mvt"] + ":" + hour), cut)[0]

        # Arrival taxi-in never freezes: those labels are present in the test
        # months, so a causal window is what deployment actually gives us.
        out["arr_taxiin_7d"] = self._window("arr_taxiin_7d", self.k_ap.encode(df["ADEP_mvt"]), d - 1)[0]

        anchor = obt_ref(df)
        out["dep_actual_30m_before"] = self._counts(df["ADEP_mvt"], anchor, self._dep_obt, -30, 0, True)
        out["arr_actual_30m_before"] = self._counts(df["ADEP_mvt"], anchor, self._arr_land, -30, 0, False)
        # Scheduled, not actual: demand around your slot is knowable in advance.
        out["dep_sched_60m_around"] = self._counts(df["ADEP_mvt"], anchor, self._dep_sched, -30, 30, False)
        return out
