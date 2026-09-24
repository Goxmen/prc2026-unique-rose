"""Airport ground geometry from OpenStreetMap, for deciding where an aircraft sat.

The point of this module is one distinction: an aircraft stopped on an *apron* is
parked, an aircraft stopped on a *taxiway* is queueing. Earlier pushback detectors
failed because they could not tell those apart -- "last stationary stretch before
takeoff" locks onto the runway hold, because aircraft stop repeatedly while queuing.

Polygons come from Overpass. Point-in-polygon is ray casting in numpy rather than
shapely, which is not a dependency of this project.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OSM_DIR = REPO / "data/external/adsb"
OVERPASS = "https://overpass-api.de/api/interpreter"
AGENT = "PRC-DataChallenge-2026/1.0 (research; taxi-time prediction)"

# metres per degree, near enough at European latitudes for distances of a few
# hundred metres; latitude is applied to the longitude term by the caller
M_PER_DEG_LAT = 111_320.0


def download(icao: str, lat: float, lon: float, half_deg: float = 0.05) -> Path:
    """Fetch aprons, parking positions, taxiways and runways around one airport."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    bbox = f"{lat - half_deg},{lon - half_deg * 1.5},{lat + half_deg},{lon + half_deg * 1.5}"
    query = f"""[out:json][timeout:180];
(
  way["aeroway"="apron"]({bbox});
  relation["aeroway"="apron"]({bbox});
  way["aeroway"="parking_position"]({bbox});
  node["aeroway"="parking_position"]({bbox});
  way["aeroway"="taxiway"]({bbox});
  way["aeroway"="runway"]({bbox});
);
out geom;"""
    req = Request(OVERPASS, data=urlencode({"data": query}).encode(),
                  headers={"User-Agent": AGENT})
    with urlopen(req, timeout=240) as resp:
        body = resp.read()
    dest = OSM_DIR / f"osm_{icao}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def _ring(geom) -> np.ndarray:
    return np.array([[p["lat"], p["lon"]] for p in geom], dtype=np.float64)


class AirportGround:
    """Apron polygons and stand reference points for one airport."""

    def __init__(self, icao: str):
        self.icao = icao
        data = json.loads((OSM_DIR / f"osm_{icao}.json").read_text())
        self.aprons: list[np.ndarray] = []
        self.stands: dict[str, np.ndarray] = {}

        for el in data["elements"]:
            tags = el.get("tags", {})
            kind = tags.get("aeroway")
            if kind == "apron":
                if el["type"] == "way" and "geometry" in el:
                    self.aprons.append(_ring(el["geometry"]))
                elif el["type"] == "relation":
                    for mem in el.get("members", []):
                        if mem.get("role") == "outer" and "geometry" in mem:
                            self.aprons.append(_ring(mem["geometry"]))
            elif kind == "parking_position":
                ref = tags.get("ref")
                if not ref:
                    continue
                if el["type"] == "way" and "geometry" in el:
                    pts = _ring(el["geometry"])
                elif el["type"] == "node":
                    pts = np.array([[el["lat"], el["lon"]]])
                else:
                    continue
                # a stand way is a short centreline; its mean is the parked position
                self.stands[ref.strip().upper()] = pts.mean(axis=0)

        self.aprons = [a for a in self.aprons if len(a) >= 4]
        if not self.aprons:
            raise ValueError(f"no apron polygons for {icao}")
        # one bounding box per polygon, to skip most ray casts
        self._bbox = np.array([[a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()]
                               for a in self.aprons])
        self._stand_xy = (np.array(list(self.stands.values()), dtype=np.float64)
                          if self.stands else None)

    def on_apron(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Boolean mask: is each point inside any apron polygon?"""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        inside = np.zeros(lat.shape, dtype=bool)
        for poly, (la0, la1, lo0, lo1) in zip(self.aprons, self._bbox):
            cand = ~inside & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
            if not cand.any():
                continue
            inside[cand] = _ray_cast(lat[cand], lon[cand], poly)
        return inside

    def near_stand(self, lat: np.ndarray, lon: np.ndarray, radius_m: float) -> np.ndarray:
        """Boolean mask: is each point within radius_m of any tagged parking position?

        OSM apron coverage is patchy -- at Zurich only 205 of 254 stands fall
        inside a tagged apron -- so proximity to a stand is a second, independent
        way of recognising a parked aircraft.
        """
        if self._stand_xy is None:
            return np.zeros(np.shape(lat), dtype=bool)
        lat = np.asarray(lat, dtype=np.float64).reshape(-1, 1)
        lon = np.asarray(lon, dtype=np.float64).reshape(-1, 1)
        d = metres(lat, lon, self._stand_xy[:, 0], self._stand_xy[:, 1])
        return d.min(axis=1) <= radius_m

    def stand_point(self, ref: str | None) -> np.ndarray | None:
        if not ref:
            return None
        return self.stands.get(str(ref).strip().upper())


def _ray_cast(lat: np.ndarray, lon: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over points."""
    y, x = lat, lon
    py, px = poly[:, 0], poly[:, 1]
    qy, qx = np.roll(py, -1), np.roll(px, -1)
    inside = np.zeros(y.shape, dtype=bool)
    for i in range(len(poly)):
        y0, x0, y1, x1 = py[i], px[i], qy[i], qx[i]
        if y0 == y1:
            continue
        crosses = (y0 > y) != (y1 > y)
        if not crosses.any():
            continue
        xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < xin)
    return inside


def metres(lat1, lon1, lat2, lon2):
    """Local flat-earth distance in metres; fine at taxiway scale."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    mlat = np.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * M_PER_DEG_LAT
    dx = (lon2 - lon1) * M_PER_DEG_LAT * np.cos(mlat)
    return np.hypot(dx, dy)
