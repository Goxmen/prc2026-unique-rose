"""In-between guess for the one Rome rule cell that is a genuine mixture.

The fixed four-hour rule predicts `takeoff - schedule` for every LIRF departure
without a flight ID that took off at least four hours late. That is the right
answer when the off-block was recorded as the scheduled time, and four hours of
error when the off-block is real and the flight was simply late. Splitting the
eligible rows by delay and by `rome_signals.mangled` shows the rule is not facing
one population but four, and only one of them is a coin worth hedging (2025
labels, `abs(y - gap) <= PLAN_MATCH_SEC` counts as a planned time typed in):

    delay      mangled   rows   planned time typed in
    4-6 h      yes         15   15  (100%)
    4-6 h      no          49   35  ( 71%)
    6-14 h     yes         20   20  (100%)
    6-14 h     no          23   22  ( 96%)

Below 14 h a mangled flight number means the record is corrupt, 35 times out of
35, so the rule stays hard there; at 6-14 h a clean number is corrupt 96% of the
time, so the rule stays hard there too. The clean 4-6 h cell is the exception:
roughly seven rows in ten are corrupt, and under squared error the best single
guess for a weighted coin is the probability-weighted mean, not either face.
An eligible row's prediction becomes

    p * (takeoff - schedule) + (1 - p) * taxi

where p is the corrupt share among the labelled rows of this cell in the
training window, shrunk toward 0.5 with weight PRIOR_WEIGHT so a short window
cannot drive it to 0 or 1, and taxi is the median label of the cell's real
off-block rows (TAXI_FALLBACK_SEC if the window holds none).

Example: RYR68SF left Rome for Malaga on 28 July 2025, scheduled 16:45, airborne
21:00 - four and a quarter hours late. The rule predicts a 15,301 s taxi; the
plane pushed back at 20:47 and taxied 723 s. Its neighbour two hours earlier,
WMTMT6AN, is the other face of the same coin: scheduled 19:30, airborne 23:38,
label 14,939 s, exactly the delay.

This narrows an existing rule; it never widens it. Rows outside the cell, and
every airport other than LIRF, keep the prediction they already had.
"""

import numpy as np
import pandas as pd

import rome_signals

AIRPORT = 'LIRF'
MIN_GAP_SEC = rome_signals.RULE_GAP_SEC          # 4 h: where the rule starts firing
MAX_GAP_SEC = 6 * 3600                           # 6 h: above this a clean number is 96% corrupt
PLAN_MATCH_SEC = 120                             # label within 2 min of the delay = typed in
NORMAL_MAX_SEC = 2400                            # a real taxi-out at Rome, generously bounded
PRIOR_WEIGHT = 2.0
TAXI_FALLBACK_SEC = 1144.0                       # median real taxi of the cell on all of 2025


def _gap(raw):
    return (pd.to_datetime(raw['MVT_TIME_UTC_mvt'], utc=True)
            - pd.to_datetime(raw['SCHED_TIME_UTC_mvt'], utc=True)).dt.total_seconds().to_numpy(float)


def eligible(raw):
    """LIRF, no flight ID, four to six hours late, flight number not mangled."""
    gap = _gap(raw)
    return ((raw['ADEP_mvt'].to_numpy(object) == AIRPORT) & raw['FLIGHT_ID_mvt'].isna().to_numpy()
            & (gap >= MIN_GAP_SEC) & (gap < MAX_GAP_SEC) & ~rome_signals.mangled(raw['FLIGHT_mvt']))


def kinds(raw, y):
    """'planned time', 'real off-block' or None for each row, from its label."""
    y = np.asarray(y, float)
    gap = _gap(raw)
    return np.select([np.abs(y - gap) <= PLAN_MATCH_SEC, y <= NORMAL_MAX_SEC],
                     ['planned time', 'real off-block'], None)


def history(raw, y):
    """Counts the hedge needs, from labelled rows (training window only)."""
    keep = eligible(raw)
    kind = kinds(raw.loc[keep], np.asarray(y, float)[keep])
    known = pd.notna(kind)
    planned = kind[known] == 'planned time'
    taxi = np.asarray(y, float)[keep][known][~planned]
    return dict(rows=int(len(planned)), planned=int(planned.sum()),
                taxi_sec=float(np.median(taxi)) if len(taxi) else TAXI_FALLBACK_SEC)


def weights(raw, hist):
    """p per row (NaN where the row is not eligible or there is no history)."""
    out = np.full(len(raw), np.nan)
    keep = eligible(raw)
    if not keep.any() or not hist['rows']:
        return out
    out[keep] = (hist['planned'] + PRIOR_WEIGHT * 0.5) / (hist['rows'] + PRIOR_WEIGHT)
    return out


def hedge(pred, raw, hist):
    """Replace each eligible row's prediction with the probability-weighted mean of its two faces."""
    pred = np.asarray(pred, float).copy()
    p = weights(raw, hist)
    m = np.isfinite(p)
    pred[m] = p[m] * _gap(raw)[m] + (1 - p[m]) * hist['taxi_sec']
    return pred
