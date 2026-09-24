"""Taxi occupancy inputs: how many other aircraft are on the tarmac right now.

The runway queue inputs of `runway_queue.py` measure *flow* -- takeoffs per half
hour, gap since the previous takeoff. Flow turns around when an airport genuinely
jams, because fewer aircraft get off the ground. These inputs measure *stock*
instead: at a given instant, how many other departures have left their stand
(`AOBT_3_flt`) but have not yet taken off.

A departure occupies the tarmac over [AOBT_3, takeoff). Only intervals with both
timestamps present and a duration within [0, MAX_GAP_SEC] count, the same filter
`runway_queue.py` applies to its takeoff-minus-AOBT3 means; that filter uses
AOBT_3, never a label. Both columns are present for the Jan/Jul 2026 ranking rows.
Departure block times and taxi-out labels are never read.

Two anchors and two scopes give four inputs:

    o_apt_at_aobt3    others taxiing at the airport when this flight left its stand
    o_rwy_at_aobt3    the same on this flight's own runway
    o_apt_at_takeoff  others taxiing at the airport when this flight took off
    o_rwy_at_takeoff  the same on this flight's own runway

The `_at_aobt3` pair is fixed before this flight's taxi begins. The `_at_takeoff`
pair spans it, so it is confounded with the target in the same way as
`q_queue_ahead_since_aobt3`; keep the two pairs separable.

A flight never counts itself: at its own takeoff its interval has already closed,
and at its own off-block it is subtracted explicitly. Runways are scoped by
airport, and a missing runway gives missing runway inputs rather than a pooled
"unknown runway". A row without a usable `AOBT_3_flt` gets missing `_at_aobt3`
inputs; its `_at_takeoff` inputs are still defined.
"""

import numpy as np
import pandas as pd

from runway_queue import DEP_COLUMNS, MAX_GAP_SEC, SEC, _Events, _groups, _ns, _runway_key

FEATURES = ['o_apt_at_aobt3', 'o_rwy_at_aobt3', 'o_apt_at_takeoff', 'o_rwy_at_takeoff']
AOBT3_FEATURES = ['o_apt_at_aobt3', 'o_rwy_at_aobt3']
TAKEOFF_FEATURES = ['o_apt_at_takeoff', 'o_rwy_at_takeoff']
SETTINGS = dict(anchors=['AOBT_3', 'takeoff'], scopes=['airport', 'runway'],
                max_gap_sec=MAX_GAP_SEC, features=FEATURES, dep_columns=DEP_COLUMNS)


def _before(events, groups, n, t, side):
    """Number of this key's events at or before each row's time; 0 for unknown keys."""
    pos, seg = events.locate(groups, n, t, side)
    return np.where(pos < 0, 0, pos - np.maximum(seg[:, 0], 0))


def occupancy_features(raw, dep_events, features=FEATURES):
    """Occupancy inputs aligned to `raw` departures, from the supplied event frame only."""
    wanted = set(features)
    if wanted - set(FEATURES):
        raise ValueError(f'Unknown occupancy inputs: {sorted(wanted - set(FEATURES))}')
    dep = dep_events.loc[:, DEP_COLUMNS]
    q = raw.loc[:, DEP_COLUMNS]
    n = len(q)

    # Other departures' taxi intervals, filtered to plausible durations.
    d_t, d_ok = _ns(dep['MVT_TIME_UTC_mvt'])
    d_aobt, d_aobt_ok = _ns(dep['AOBT_3_flt'])
    dur = np.where(d_ok & d_aobt_ok, (d_t - d_aobt) / SEC, np.nan)
    usable = np.isfinite(dur) & (dur >= 0) & (dur <= MAX_GAP_SEC)
    d_apt = dep['ADEP_mvt'].to_numpy(object)
    d_rwy = _runway_key(dep['ADEP_mvt'], dep['RUNWAY_mvt'])
    starts = {'apt': _Events(d_apt, d_aobt, usable), 'rwy': _Events(d_rwy, d_aobt, usable)}
    ends = {'apt': _Events(d_apt, d_t, usable), 'rwy': _Events(d_rwy, d_t, usable)}

    # This flight's own anchors and keys.
    q_t, q_ok = _ns(q['MVT_TIME_UTC_mvt'])
    q_aobt, q_aobt_ok = _ns(q['AOBT_3_flt'])
    own_dur = np.where(q_ok & q_aobt_ok, (q_t - q_aobt) / SEC, np.nan)
    own_usable = np.isfinite(own_dur) & (own_dur >= 0) & (own_dur <= MAX_GAP_SEC)
    apt_keys = np.array(q['ADEP_mvt'], dtype=object)
    apt_keys[~q_ok] = None
    rwy_keys = _runway_key(q['ADEP_mvt'], q['RUNWAY_mvt'])
    rwy_keys[~q_ok] = None
    groups = {'apt': _groups(apt_keys), 'rwy': _groups(rwy_keys)}
    missing = {'apt': pd.isna(apt_keys), 'rwy': pd.isna(rwy_keys)}

    out = {}
    for scope in ('apt', 'rwy'):
        for anchor in ('aobt3', 'takeoff'):
            name = f'o_{scope}_at_{anchor}'
            if name not in wanted:
                continue
            t = q_aobt if anchor == 'aobt3' else q_t
            # Started at or before t, minus already taken off at or before t.
            occ = (_before(starts[scope], groups[scope], n, t, 'right')
                   - _before(ends[scope], groups[scope], n, t, 'right')).astype(float)
            if anchor == 'aobt3':
                # At its own off-block a flight sits inside its own interval, unless that
                # interval is empty (AOBT_3 equal to takeoff), where the two events cancel.
                occ -= own_usable & (own_dur > 0)
                gone = missing[scope] | ~q_aobt_ok
            else:
                gone = missing[scope] | ~q_ok
            out[name] = np.where(gone, np.nan, occ)
    return pd.DataFrame({name: out[name] for name in FEATURES if name in wanted},
                        index=raw.index).astype('float32')


def add_occupancy(X, raw, dep_events, features=FEATURES):
    if not X.index.equals(raw.index):
        raise ValueError('Matrix and raw movement rows must align before adding occupancy inputs')
    overlap = set(X.columns) & set(features)
    if overlap:
        raise ValueError(f'Occupancy inputs already present: {sorted(overlap)}')
    return pd.concat([X, occupancy_features(raw, dep_events, features)], axis=1)
