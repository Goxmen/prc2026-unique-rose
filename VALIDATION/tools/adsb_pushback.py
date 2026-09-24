"""Estimate off-block from ADSB.lol ground traces.

The recorded off-block is when the aircraft starts moving off its stand. In a
ground trace that is the *end* of the last long stationary period the aircraft
spent parked. Two earlier detectors failed on exactly this point:

* "first ground point before takeoff" finds the *start* of that period, which is
  the previous rotation's arrival taxi-in, not this departure.
* "last stationary stretch before takeoff" finds the runway queue, because
  aircraft stop repeatedly while waiting to line up.

So the rule here is: take the last stationary period that is long enough to be a
turnaround *and* sits on an apron rather than a taxiway, and report when it
ended. The apron test is what separates parked from queueing; see adsb_geometry.

A trace file covers one aircraft for a whole UTC day worldwide, so points are
first clipped to a radius around the departure airport.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from adsb_geometry import AirportGround, metres

AIRPORT_XY = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}

# tuning, all revisited in the sweep at the bottom of validate_pushback.py
AIRPORT_RADIUS_M = 6_000.0   # clip the worldwide trace to this airport
LOOKBACK_S = 6 * 3600        # how far before takeoff to look for the stand
STATIONARY_M = 60.0          # points within this of the run's anchor count as stopped
MIN_PARK_S = 600.0           # a turnaround is at least this long
OWN_STAND_M = 120.0          # distance to the movement record's own stand
STAND_NEAR_M = 200.0         # a stop this close to any tagged stand counts as parked


@dataclass
class Pushback:
    offblock: float | None       # epoch seconds
    method: str                  # which branch produced it
    park_start: float | None     # when the aircraft settled on the stand
    park_points: int
    on_apron: bool
    own_stand: bool              # the parked position matches STAND_mvt
    ground_points: int           # points at this airport in the window


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (epoch, lat, lon, is_ground) for one aircraft-day."""
    with gzip.open(path, "rb") as fh:
        doc = json.load(fh)
    base = doc["timestamp"]
    tr = doc.get("trace") or []
    if not tr:
        return (np.empty(0),) * 4
    t = np.fromiter((p[0] for p in tr), dtype=np.float64, count=len(tr)) + base
    lat = np.fromiter((p[1] for p in tr), dtype=np.float64, count=len(tr))
    lon = np.fromiter((p[2] for p in tr), dtype=np.float64, count=len(tr))
    grd = np.fromiter((p[3] == "ground" for p in tr), dtype=bool, count=len(tr))
    return t, lat, lon, grd


def extract(t, lat, lon, grd, takeoff: float, ground: AirportGround,
            stand_ref: str | None = None,
            lookback_s: float = LOOKBACK_S,
            stationary_m: float = STATIONARY_M,
            min_park_s: float = MIN_PARK_S,
            stand_near_m: float = STAND_NEAR_M) -> Pushback:
    """Find when this departure left its stand."""
    alat, alon = AIRPORT_XY[ground.icao]
    near = grd & (t >= takeoff - lookback_s) & (t <= takeoff + 120) \
         & (metres(lat, lon, alat, alon) <= AIRPORT_RADIUS_M)
    n = int(near.sum())
    if n == 0:
        return Pushback(None, "no_ground_points", None, 0, False, False, 0)

    t, lat, lon = t[near], lat[near], lon[near]
    order = np.argsort(t)
    t, lat, lon = t[order], lat[order], lon[order]
    # "parked" means on a tagged apron OR close to a tagged stand: OSM apron
    # coverage is incomplete, and at Zurich the stops it misses sit a median
    # 120 m from a known stand
    apron = ground.on_apron(lat, lon)
    parked_here = apron | ground.near_stand(lat, lon, stand_near_m)
    sp = ground.stand_point(stand_ref)

    # group consecutive points that stay within stationary_m of the run's anchor
    runs = []
    i = 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and metres(lat[i], lon[i], lat[j + 1], lon[j + 1]) <= stationary_m:
            j += 1
        runs.append((i, j))
        i = j + 1

    def describe(a: int, b: int, method: str) -> Pushback:
        # off-block lies between the last parked fix and the first fix that moved
        end = t[b] if b + 1 >= len(t) else (t[b] + t[b + 1]) / 2.0
        own = False
        if sp is not None:
            own = bool(metres(lat[a], lon[a], sp[0], sp[1]) <= OWN_STAND_M)
        return Pushback(float(end), method, float(t[a]), b - a + 1,
                        bool(apron[a]), own, n)

    # the last run long enough to be a turnaround, at a parking position
    for a, b in reversed(runs):
        if t[b] - t[a] >= min_park_s and parked_here[a]:
            return describe(a, b, "parked")

    # No long parked run. At Zurich that is about half of all departures: the
    # transponder is off through the turnaround and comes on at pushback, so the
    # trace simply begins on the stand. Take the first parked fix.
    where = np.flatnonzero(parked_here)
    if len(where):
        k = int(where[0])
        run = next((r for r in runs if r[0] <= k <= r[1]), (k, k))
        return describe(run[0], run[1], "transponder_on")
    return Pushback(None, "never_parked", None, 0, False, False, n)


def to_row(p: Pushback) -> dict:
    return asdict(p)
