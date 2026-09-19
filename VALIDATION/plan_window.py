"""Flight-plan window: a bound on every prediction for a departure with a flight ID.

A flight ID links a movement record to a Network Manager flight plan, and the link
is only made when the recorded off-block is within an hour of the plan's last
off-block time, `LOBT_flt`. Over all 2,062,623 departures with a flight ID in the
2025 training files, |recorded off-block - LOBT| never exceeds 60.1 minutes, at
every one of the ten airports. The label is takeoff minus recorded off-block, so
for each of those flights it lies in

    takeoff - LOBT +- MARGIN_SEC

Takeoff and LOBT are both supplied for ranking departures. A prediction outside
the window moves to its nearest edge, which can only bring it closer to the label.
MARGIN_SEC is the largest deviation seen plus about one minute.

This is post-processing: the models are unchanged and no label is read. Rows
without a flight ID or without LOBT keep their prediction, so the four-hour Rome
rule's rows (which never carry a flight ID) never move.

Example, ranking row ITY340 (LIRF -> LFMN): scheduled 2026-07-16 19:35 UTC, LOBT
21:44, takeoff in the record 2026-07-17 21:51. The label must lie between 23.1 h
and 25.1 h; v30 predicted 3.6 h.
"""

import numpy as np
import pandas as pd

MARGIN_SEC = 3660


def window(raw):
    """Lowest and highest possible label per row; -inf/inf where there is no flight ID or LOBT."""
    centre = (pd.to_datetime(raw['MVT_TIME_UTC_mvt'], utc=True)
              - pd.to_datetime(raw['LOBT_flt'], utc=True)).dt.total_seconds().to_numpy(float)
    bounded = raw['FLIGHT_ID_mvt'].notna().to_numpy() & np.isfinite(centre)
    low = np.where(bounded, centre - MARGIN_SEC, -np.inf)
    high = np.where(bounded, centre + MARGIN_SEC, np.inf)
    return low, high


def apply(pred, raw):
    """Move each prediction into its window; labels are positive, so never below zero."""
    low, high = window(raw)
    return np.maximum(np.clip(np.asarray(pred, float), low, high), 0.0)
