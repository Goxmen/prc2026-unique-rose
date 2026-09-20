"""In-between guess for Rome rule rows whose label may carry the wrong date.

A LIRF departure without a flight ID that took off at least 14 h after schedule, on
a later local (Europe/Rome) day, gets one of two labels in 2025, and nothing visible
tells which:

    planned time typed in   recorded off-block = schedule, label = takeoff - schedule
    wrong date              recorded off-block = actual local clock time on the
                            schedule's date, label = days * 86400 + taxi

Example, two Jet2 departures on the night of 24-25 January 2025, both about 16 h late
from stands 612 and 613: EXSXS3AD was recorded leaving at Fri 12:30:03 (its schedule;
label 56,633 s), EXSXS1GT at Fri 04:24:01 (the real clock time, Friday's date; label
87,177 s). Both planes really taxied for about 15 minutes.

The four-hour Rome rule predicts the first label. Under squared error the best single
guess between two possible labels is their probability-weighted mean, so an eligible
row's prediction becomes

    p * (days * 86400 + taxi) + (1 - p) * (takeoff - schedule)

where p is the wrong-date share among the training labels of eligible rows, shrunk
level by level with the same weight W = PRIOR_WEIGHT, so a handful of rows never
pushes it to 0 or 1:

    overall = (wrong + W * 0.5) / (rows + W)
    format  = (wrong with this format + W * overall) / (rows with this format + W)
    p       = (wrong for this airline + W * format) / (rows for this airline + W)
    taxi    = median of label - days * 86400 over the wrong-date rows (TAXI_FALLBACK_SEC if none)

Format is mangled or clean (rome_signals.mangled); the airline is
rome_signals.airline_code of FLIGHT_mvt. Rows labelled neither way are ignored. On all
of 2025: 11 of 19 wrong date; mangled 9 of 14, clean 2 of 5; WMT 0 of 2, ITY 3 of 4.
"""

import numpy as np
import pandas as pd

import rome_signals

AIRPORT = 'LIRF'
MIN_GAP_SEC = rome_signals.DAYSHIFT_GAP_SEC
DAY_SEC = rome_signals.DAY_SEC
PLAN_MATCH_SEC = 120
WRONG_DATE_MAX_TAXI_SEC = 3600
PRIOR_WEIGHT = 2.0
TAXI_FALLBACK_SEC = 900.0


def _gap(raw):
    return (pd.to_datetime(raw['MVT_TIME_UTC_mvt'], utc=True)
            - pd.to_datetime(raw['SCHED_TIME_UTC_mvt'], utc=True)).dt.total_seconds().to_numpy(float)


def eligible(raw):
    """LIRF, no flight ID, at least 14 h late and on a later local day: the rows that can go either way."""
    return ((raw['ADEP_mvt'].to_numpy(object) == AIRPORT) & raw['FLIGHT_ID_mvt'].isna().to_numpy()
            & (_gap(raw) >= MIN_GAP_SEC) & (rome_signals.local_days(raw) >= 1))


def kinds(raw, y):
    """'wrong date', 'planned time' or None for each row, from its label."""
    y = np.asarray(y, float)
    gap, days = _gap(raw), rome_signals.local_days(raw)
    taxi = y - DAY_SEC * days
    wrong = (taxi >= 0) & (taxi <= WRONG_DATE_MAX_TAXI_SEC)
    plan = np.abs(y - gap) <= PLAN_MATCH_SEC
    return np.select([wrong, plan], ['wrong date', 'planned time'], None)


def history(raw, y):
    """Counts the hedge needs, from labelled rows (training window only)."""
    keep = eligible(raw)
    kind = kinds(raw.loc[keep], np.asarray(y, float)[keep])
    known = pd.notna(kind)
    sub = raw.loc[keep].loc[known]
    wrong = kind[known] == 'wrong date'
    fmt = rome_signals.mangled(sub['FLIGHT_mvt'])
    airline = rome_signals.airline_code(sub['FLIGHT_mvt']).to_numpy(object)
    taxi = (np.asarray(y, float)[keep][known] - DAY_SEC * rome_signals.local_days(sub))[wrong]
    count = lambda m: [int(wrong[m].sum()), int(m.sum())]
    return dict(rows=int(len(wrong)), wrong_date=int(wrong.sum()),
                format={'mangled': count(fmt), 'clean': count(~fmt)},
                airline={str(a): count(airline == a) for a in pd.unique(airline[pd.notna(airline)])},
                taxi_sec=float(np.median(taxi)) if wrong.any() else TAXI_FALLBACK_SEC)


def _shrink(counts, prior):
    k, n = counts
    return (k + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT)


def weights(raw, hist):
    """p per row (NaN where the row is not eligible or there is no history)."""
    out = np.full(len(raw), np.nan)
    keep = eligible(raw)
    if not keep.any() or not hist['rows']:
        return out
    sub = raw.loc[keep]
    fmt = rome_signals.mangled(sub['FLIGHT_mvt'])
    airline = rome_signals.airline_code(sub['FLIGHT_mvt']).to_numpy(object)
    overall = _shrink([hist['wrong_date'], hist['rows']], 0.5)
    p = np.empty(len(sub))
    for i in range(len(sub)):
        base = _shrink(hist['format']['mangled' if fmt[i] else 'clean'], overall)
        own = hist['airline'].get(str(airline[i]), [0, 0]) if pd.notna(airline[i]) else [0, 0]
        p[i] = _shrink(own, base)
    out[keep] = p
    return out


def hedge(pred, raw, hist):
    """Replace each eligible row's prediction with the probability-weighted mean of its two possible labels."""
    pred = np.asarray(pred, float).copy()
    p = weights(raw, hist)
    m = np.isfinite(p)
    gap, days = _gap(raw), rome_signals.local_days(raw)
    pred[m] = p[m] * (DAY_SEC * days[m] + hist['taxi_sec']) + (1 - p[m]) * gap[m]
    return pred
