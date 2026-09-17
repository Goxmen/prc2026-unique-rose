"""Real-time runway queue inputs from supplied movement timestamps.

Each departure is described by the traffic around its own takeoff time:
other departures' takeoff time, runway and AOBT3, and arrivals' landing time and
runway. All of these are present for the Jan/Jul 2026 ranking rows. Departure
block times and taxi-out labels are never read: only the columns in `DEP_COLUMNS`
and `ARR_COLUMNS` are taken from the event frames.

Every window excludes the takeoff instant itself, so a departure never counts
itself, and the result does not depend on whether the described rows are part
of the event frames. Runways are scoped by airport; a missing runway gives
missing runway inputs rather than a pooled "unknown runway".

PAST_FEATURES use only movements before takeoff:

    q_rwy_prev_takeoff_gap_sec  seconds since the previous takeoff on the same runway
    q_rwy_takeoffs_prev30       takeoffs on the same runway in the previous 30 minutes
    q_apt_takeoffs_prev30       takeoffs at the airport in the previous 30 minutes
    q_rwy_others_gap_prev60     mean takeoff-minus-AOBT3 of departures that took off
                                from the same runway in the previous 60 minutes
    q_apt_others_gap_prev60     the same at airport level
    q_queue_ahead_since_aobt3   airport takeoffs between this flight's AOBT3 and its
                                takeoff (missing without AOBT3 or if AOBT3 is later)
    q_rwy_landings_prev30       landings on the same runway in the previous 30 minutes

FUTURE_FEATURES also use movements after takeoff, which the ranking file supplies:

    q_rwy_next_takeoff_gap_sec  seconds until the next takeoff on the same runway
    q_rwy_takeoffs_next30       takeoffs on the same runway in the next 30 minutes
    q_rwy_others_gap_pm30       mean takeoff-minus-AOBT3 on the same runway, 30 minutes
                                either side
    q_apt_others_gap_pm30       the same at airport level

Takeoff-minus-AOBT3 means only use gaps within [0, MAX_GAP_SEC]; that filter
uses AOBT3, never a label.
"""

import numpy as np
import pandas as pd

from features_time import _naive

DEP_COLUMNS = ['ADEP_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt']
ARR_COLUMNS = ['ADES_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt']
PAST_FEATURES = [
    'q_rwy_prev_takeoff_gap_sec', 'q_rwy_takeoffs_prev30', 'q_apt_takeoffs_prev30',
    'q_rwy_others_gap_prev60', 'q_apt_others_gap_prev60', 'q_queue_ahead_since_aobt3',
    'q_rwy_landings_prev30',
]
FUTURE_FEATURES = [
    'q_rwy_next_takeoff_gap_sec', 'q_rwy_takeoffs_next30', 'q_rwy_others_gap_pm30', 'q_apt_others_gap_pm30',
]
ALL_FEATURES = PAST_FEATURES + FUTURE_FEATURES
MAX_GAP_SEC = 7200
SEC = 1_000_000_000
MIN30, MIN60 = 30 * 60 * SEC, 60 * 60 * SEC
SETTINGS = dict(anchor='takeoff time', max_gap_sec=MAX_GAP_SEC, past=PAST_FEATURES, future=FUTURE_FEATURES,
                dep_columns=DEP_COLUMNS, arr_columns=ARR_COLUMNS)


def _ns(values):
    t = _naive(values).to_numpy('datetime64[ns]')
    return t.astype('int64'), ~np.isnat(t)


def _runway_key(airport, runway):
    key = airport.astype(object) + ':' + runway.astype(object)
    return np.array(key.where(airport.notna() & runway.notna()), dtype=object)


def _groups(keys):
    """Row indices per non-missing key."""
    keys = np.asarray(keys, dtype=object)
    present = np.flatnonzero(pd.notna(keys))
    return {key: present[idx] for key, idx in pd.Series(keys[present]).groupby(keys[present]).indices.items()}


class _Events:
    """Event times sorted within each key, with running sums of an optional value."""

    def __init__(self, keys, times, ok, values=None):
        keys = np.asarray(keys, dtype=object)
        ok = ok & pd.notna(keys)
        codes, uniques = pd.factorize(keys[ok])
        order = np.lexsort((times[ok], codes))
        self.t = times[ok][order]
        bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
        self.segment = {key: (int(bounds[i]), int(bounds[i + 1])) for i, key in enumerate(uniques)}
        if values is not None:
            v = np.asarray(values, float)[ok][order]
            valid = np.isfinite(v)
            self.sum = np.r_[0.0, np.cumsum(np.where(valid, v, 0.0))]
            self.count = np.r_[0, np.cumsum(valid)]

    def locate(self, groups, n, t, side):
        """Insertion position of each row's time inside its key's segment, plus the segment.

        Rows with a missing key or a key without events get position and segment -1.
        """
        pos = np.full(n, -1, dtype=np.int64)
        seg = np.full((n, 2), -1, dtype=np.int64)
        for key, rows in groups.items():
            if key not in self.segment:
                continue
            start, stop = self.segment[key]
            pos[rows] = start + np.searchsorted(self.t[start:stop], t[rows], side)
            seg[rows] = (start, stop)
        return pos, seg


def _span(events, groups, t, lo, hi, include_lo, include_hi):
    a, _ = events.locate(groups, len(t), t + lo, 'left' if include_lo else 'right')
    b, _ = events.locate(groups, len(t), t + hi, 'right' if include_hi else 'left')
    return a, b


def _count(events, groups, t, lo, hi, include_lo, include_hi, missing):
    a, b = _span(events, groups, t, lo, hi, include_lo, include_hi)
    return np.where(missing, np.nan, np.where(a < 0, 0.0, b - a))


def _mean(events, groups, t, windows, missing):
    total, count = np.zeros(len(t)), np.zeros(len(t))
    for lo, hi, include_lo, include_hi in windows:
        a, b = _span(events, groups, t, lo, hi, include_lo, include_hi)
        known = a >= 0
        a, b = np.maximum(a, 0), np.maximum(b, 0)
        total += np.where(known, events.sum[b] - events.sum[a], 0.0)
        count += np.where(known, events.count[b] - events.count[a], 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(missing | (count == 0), np.nan, total / np.maximum(count, 1))


def queue_features(raw, dep_events, arr_events, features=ALL_FEATURES):
    """Queue inputs aligned to `raw` departures, from the supplied event frames only."""
    wanted = set(features)
    if wanted - set(ALL_FEATURES):
        raise ValueError(f'Unknown queue inputs: {sorted(wanted - set(ALL_FEATURES))}')
    dep = dep_events.loc[:, DEP_COLUMNS]
    arr = arr_events.loc[:, ARR_COLUMNS]
    q = raw.loc[:, DEP_COLUMNS]
    n = len(q)

    q_t, q_ok = _ns(q['MVT_TIME_UTC_mvt'])
    q_aobt, q_aobt_ok = _ns(q['AOBT_3_flt'])
    apt_keys = np.array(q['ADEP_mvt'], dtype=object)
    apt_keys[~q_ok] = None
    rwy_keys = _runway_key(q['ADEP_mvt'], q['RUNWAY_mvt'])
    rwy_keys[~q_ok] = None
    apt, rwy = _groups(apt_keys), _groups(rwy_keys)
    no_apt, no_rwy = pd.isna(apt_keys), pd.isna(rwy_keys)

    d_t, d_ok = _ns(dep['MVT_TIME_UTC_mvt'])
    d_aobt, d_aobt_ok = _ns(dep['AOBT_3_flt'])
    gap = np.where(d_aobt_ok, (d_t - d_aobt) / SEC, np.nan)
    gap = np.where((gap >= 0) & (gap <= MAX_GAP_SEC), gap, np.nan)
    by_apt = _Events(dep['ADEP_mvt'].to_numpy(object), d_t, d_ok, gap)
    by_rwy = _Events(_runway_key(dep['ADEP_mvt'], dep['RUNWAY_mvt']), d_t, d_ok, gap)
    a_t, a_ok = _ns(arr['MVT_TIME_UTC_mvt'])
    landings = _Events(_runway_key(arr['ADES_mvt'], arr['RUNWAY_mvt']), a_t, a_ok)

    out = {}
    if 'q_rwy_prev_takeoff_gap_sec' in wanted:
        j, seg = by_rwy.locate(rwy, n, q_t, 'left')
        has = (j >= 0) & (j > seg[:, 0])
        prev = by_rwy.t[np.clip(j - 1, 0, max(len(by_rwy.t) - 1, 0))] if len(by_rwy.t) else np.zeros(n)
        out['q_rwy_prev_takeoff_gap_sec'] = np.where(has, (q_t - prev) / SEC, np.nan)
    if 'q_rwy_next_takeoff_gap_sec' in wanted:
        j, seg = by_rwy.locate(rwy, n, q_t, 'right')
        has = (j >= 0) & (j < seg[:, 1])
        nxt = by_rwy.t[np.clip(j, 0, max(len(by_rwy.t) - 1, 0))] if len(by_rwy.t) else np.zeros(n)
        out['q_rwy_next_takeoff_gap_sec'] = np.where(has, (nxt - q_t) / SEC, np.nan)
    if 'q_rwy_takeoffs_prev30' in wanted:
        out['q_rwy_takeoffs_prev30'] = _count(by_rwy, rwy, q_t, -MIN30, 0, True, False, no_rwy)
    if 'q_rwy_takeoffs_next30' in wanted:
        out['q_rwy_takeoffs_next30'] = _count(by_rwy, rwy, q_t, 0, MIN30, False, True, no_rwy)
    if 'q_apt_takeoffs_prev30' in wanted:
        out['q_apt_takeoffs_prev30'] = _count(by_apt, apt, q_t, -MIN30, 0, True, False, no_apt)
    if 'q_rwy_others_gap_prev60' in wanted:
        out['q_rwy_others_gap_prev60'] = _mean(by_rwy, rwy, q_t, [(-MIN60, 0, True, False)], no_rwy)
    if 'q_apt_others_gap_prev60' in wanted:
        out['q_apt_others_gap_prev60'] = _mean(by_apt, apt, q_t, [(-MIN60, 0, True, False)], no_apt)
    if 'q_rwy_others_gap_pm30' in wanted:
        out['q_rwy_others_gap_pm30'] = _mean(by_rwy, rwy, q_t, [(-MIN30, 0, True, False), (0, MIN30, False, True)], no_rwy)
    if 'q_apt_others_gap_pm30' in wanted:
        out['q_apt_others_gap_pm30'] = _mean(by_apt, apt, q_t, [(-MIN30, 0, True, False), (0, MIN30, False, True)], no_apt)
    if 'q_queue_ahead_since_aobt3' in wanted:
        a, _ = by_apt.locate(apt, n, np.where(q_aobt_ok, q_aobt, q_t), 'left')
        b, _ = by_apt.locate(apt, n, q_t, 'left')
        valid = q_aobt_ok & (q_aobt <= q_t) & ~no_apt
        out['q_queue_ahead_since_aobt3'] = np.where(valid, np.where(a < 0, 0.0, b - a), np.nan)
    if 'q_rwy_landings_prev30' in wanted:
        out['q_rwy_landings_prev30'] = _count(landings, rwy, q_t, -MIN30, 0, True, False, no_rwy)
    return pd.DataFrame({name: out[name] for name in ALL_FEATURES if name in wanted}, index=raw.index).astype('float32')


def add_queue(X, raw, dep_events, arr_events, features=ALL_FEATURES):
    if not X.index.equals(raw.index):
        raise ValueError('Matrix and raw movement rows must align before adding queue inputs')
    overlap = set(X.columns) & set(features)
    if overlap:
        raise ValueError(f'Queue inputs already present: {sorted(overlap)}')
    return pd.concat([X, queue_features(raw, dep_events, arr_events, features)], axis=1)
