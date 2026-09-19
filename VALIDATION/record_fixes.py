"""Two recording faults that bound or fix a prediction after the model; no label is read.

NO-ID CAP. A departure keeps its flight ID only when its recorded off-block is within
an hour of its flight plan's LOBT (see plan_window.py), so a departure without one was
recorded leaving its stand more than an hour away from the plan. Rows without a flight
ID carry no plan, so the schedule stands in for it. Over the 18,705 validation rows
(seed 42) of departures outside Rome with no flight ID that took off more than 75
minutes after schedule, 99.0% have label <= (takeoff - schedule) - 3,600 s; LTFM is
the exception at 92.4%. Two ceilings follow, applied only where they lower a prediction:

    gap > 75 min, not LIRF, not LTFM:  pred <= gap - 3600  (so at least 15 min)
    gap > 30 min, not LIRF:            pred <= gap

where gap = takeoff - schedule. Rome keeps its own no-ID model (rome_signals.py).
Example, ranking row NJE278K at LSZH stand 103: scheduled 11:50, takeoff 13:30
(gap 6,015 s); v33 predicted 13,089 s, the cap allows at most 2,415 s.

TWIN RECORDS. LFPG sometimes logs one takeoff twice: two departures with the same
airport and flight number, takeoffs seconds apart, usually from different stands (a
tow). The first stand's off-block is near the plan, so that record keeps the flight ID
and a long label; the real pushback came more than an hour after LOBT, so the other
record loses the ID and has an ordinary label. In 2025 there are 110 such clusters
(107 at LFPG); in the 102 two-row clusters with exactly one flight ID, the ID row's
label minus (takeoff - LOBT) has median -5.5 min (10th-90th percentile -29 to +18),
and it exceeds the other row's label in 97% of pairs. The ID row's prediction becomes

    takeoff - LOBT - TWIN_OFFSET_SEC

and the no-ID row keeps its prediction. TWIN_OFFSET_SEC was read from all of 2025
(half-year medians 244 and 419 s); with no offset the validation gain is almost the same.
Example, 2025 SIA335 at LFPG: takeoffs 10:55:58 (stand Z04, no ID, label 956 s) and
10:56:07 (stand U16, flight ID, off-block 09:40:03 = schedule, label 4,564 s).
"""

import numpy as np
import pandas as pd

LATE_GAP_SEC = 4500
LATE_CAP_SEC = 3600
GAP_CAP_MIN_SEC = 1800
CAP_EXCLUDED = ('LIRF',)
LATE_CAP_EXCLUDED = ('LIRF', 'LTFM')
TWIN_WINDOW_SEC = 180
TWIN_OFFSET_SEC = 330


def _seconds(later, earlier):
    return (pd.to_datetime(later, utc=True) - pd.to_datetime(earlier, utc=True)).dt.total_seconds().to_numpy(float)


def noid_cap(pred, raw):
    """Lower each no-ID prediction outside Rome to its ceiling; needs ADEP, FLIGHT_ID, takeoff and schedule."""
    pred = np.asarray(pred, float)
    gap = _seconds(raw['MVT_TIME_UTC_mvt'], raw['SCHED_TIME_UTC_mvt'])
    noid = raw['FLIGHT_ID_mvt'].isna().to_numpy()
    airport = raw['ADEP_mvt'].to_numpy(object)
    ceiling = np.full(len(pred), np.inf)
    late = noid & ~np.isin(airport, LATE_CAP_EXCLUDED) & (gap > LATE_GAP_SEC)
    ceiling[late] = gap[late] - LATE_CAP_SEC
    delayed = noid & ~np.isin(airport, CAP_EXCLUDED) & (gap > GAP_CAP_MIN_SEC)
    ceiling[delayed] = np.minimum(ceiling[delayed], gap[delayed])
    return np.minimum(pred, ceiling)


def twin_rows(raw):
    """Boolean mask of the flight-ID row in each two-record cluster with exactly one flight ID and an LOBT.

    `raw` must hold every departure of the set (twins are found among its rows), with
    ADEP_mvt, FLIGHT_mvt, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt and LOBT_flt.
    """
    t = pd.DataFrame({'airport': raw['ADEP_mvt'].to_numpy(object), 'flight': raw['FLIGHT_mvt'].to_numpy(object),
                      'time': pd.to_datetime(raw['MVT_TIME_UTC_mvt'], utc=True).to_numpy(),
                      'has_id': raw['FLIGHT_ID_mvt'].notna().to_numpy(), 'row': np.arange(len(raw))})
    t = t[pd.notna(t.flight)].sort_values(['airport', 'flight', 'time'], kind='mergesort')
    step = t.groupby(['airport', 'flight'], sort=False).time.diff().dt.total_seconds()
    t['cluster'] = (~step.le(TWIN_WINDOW_SEC)).cumsum()
    size = t.groupby('cluster').row.transform('size')
    ids = t.groupby('cluster').has_id.transform('sum')
    mask = np.zeros(len(raw), bool)
    mask[t.row[(size == 2) & (ids == 1) & t.has_id].to_numpy()] = True
    return mask & raw['LOBT_flt'].notna().to_numpy()


def twin_records(pred, raw, mask=None):
    """Set the flight-ID row of each twin pair to takeoff - LOBT - TWIN_OFFSET_SEC (never below zero).

    `mask` is twin_rows() computed on a larger set that contains these rows; without it
    the twins are found among `raw` itself.
    """
    pred = np.asarray(pred, float).copy()
    mask = twin_rows(raw) if mask is None else np.asarray(mask, bool)
    target = _seconds(raw['MVT_TIME_UTC_mvt'], raw['LOBT_flt']) - TWIN_OFFSET_SEC
    pred[mask] = np.maximum(target[mask], 0.0)
    return pred
