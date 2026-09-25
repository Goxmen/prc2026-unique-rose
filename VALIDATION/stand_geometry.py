"""Stand and runway-end coordinates for the ten challenge airports (idea 34).

Two public maps are matched to the stand names in `STAND_mvt`:

* X-Plane Scenery Gateway `apt.dat`, the recommended scenery per airport
  (GPL v2 or later). Row 1300 is a stand, row 100 a runway with both ends.
* OpenStreetMap `aeroway=parking_position` (ODbL), fetched by
  `tools/adsb_geometry.download`. It fills stands the Gateway lacks, and where
  both maps have a stand their positions check each other.

The maps and the movement data spell stands differently ("H01" / "H1",
"Gate E 58" / "E58", "K08-(E)" / "K8"), so every name goes through `canon`.
A stand then matches by one of three rules, tried in order:

* ``exact``      the same canonical name;
* ``substands``  the map splits it into lettered halves ("210" -> "210A",
                 "210B"), whose positions are averaged;
* ``parent``     the map only has the stand without its last letter
                 ("151A" -> "151").

Exact matches from either map are preferred to approximate ones from either.
Downloads: `tools/fetch_stand_maps.py`. Table and checks:
`tools/build_stand_coords.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XPLANE_DIR = REPO / 'data/external/stand_maps/xplane'
OSM_DIR = REPO / 'data/external/adsb'           # shared with the ADS-B probe
OUT_DIR = REPO / 'data/external/stand_maps'
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']

# names that are not a stand: no position can be right for them
NOT_A_STAND = {'UNKNOWN'}
# two map entries with one canonical name are one stand only if this close
SAME_STAND_M = 150.0
M_PER_DEG_LAT = 111_320.0
RULES = ('exact', 'substands', 'parent')
SOURCES = ('xplane', 'osm')

# Where one name means two places, the map that agrees with the data wins.
# LSZH 101-105: 98% business-aviation departures (Citations, one operator shared
# with 106-112), so they are the general-aviation centre stands. OSM puts them
# there; X-Plane's "Stand 101-105" is an airline apron 1.3 km away (its
# "GAC 101-105" agree with OSM). Checked 2026-09-25.
# EDDM 103-105: unused in 2025, new in 2026, and 151-270 m apart in the maps.
# Their 2026 arrivals taxi in for 415-428 s (median), like OSM's neighbours
# 110-112 (364-485 s) and unlike X-Plane's 175/181/182 (306-357 s). X-Plane's
# 106/107 have no OSM position and stay, flagged in the report.
SOURCE_ORDER = {**{('LSZH', k): ('osm', 'xplane') for k in ('101', '102', '103', '104', '105')},
                **{('EDDM', k): ('osm', 'xplane') for k in ('103', '104', '105')}}

_WORDS = re.compile(r'^(GATE|STAND|RAMP|PARKING|POSITION|POS)\b\s*')
_SIZE_CODE = re.compile(r'-\([A-F]\)$')          # LTFM: "K08-(E)"
_RANGE = re.compile(r'\d\s+-\s+\d')              # LEBL: "Ramp 1 - 2" is two stands
_LEADING_ZEROS = re.compile(r'(?<!\d)0+(?=\d)')


def canon(name) -> str | None:
    """One spelling per stand: upper case, no label words, spaces or leading zeros."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).upper().strip()
    if not s or _RANGE.search(s):
        return None
    s = _SIZE_CODE.sub('', s)
    s = _WORDS.sub('', s)
    s = re.sub(r'[\s_]+', '', s)
    s = _LEADING_ZEROS.sub('', s)
    return s or None


def metres(lat1, lon1, lat2, lon2):
    """Local flat-earth distance in metres; fine within one airport."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    mlat = np.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * M_PER_DEG_LAT
    dx = (lon2 - lon1) * M_PER_DEG_LAT * np.cos(mlat)
    return np.hypot(dx, dy)


def parse_apt(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stands (row 1300) and land-runway ends (row 100) from one apt.dat."""
    stands, ends = [], []
    for line in text.splitlines():
        if line.startswith('1300 '):
            f = line.split(None, 6)
            if len(f) == 7:
                stands.append((f[6].strip(), float(f[1]), float(f[2])))
        elif line.startswith('100 '):
            f = line.split()
            # 8 runway fields, then 9 per end: designator lat lon displaced ...
            for i in (8, 17):
                ends.append((f[i], float(f[i + 1]), float(f[i + 2]), float(f[i + 3])))
    return (pd.DataFrame(stands, columns=['name', 'lat', 'lon']),
            pd.DataFrame(ends, columns=['runway', 'lat', 'lon', 'displaced_m']))


def parse_osm(data: dict) -> pd.DataFrame:
    """Tagged parking positions; a stand drawn as a line sits at its mean point."""
    rows = []
    for el in data.get('elements', []):
        tags = el.get('tags', {})
        if tags.get('aeroway') != 'parking_position' or not tags.get('ref'):
            continue
        if el['type'] == 'node':
            lat, lon = el['lat'], el['lon']
        elif el['type'] == 'way' and el.get('geometry'):
            lat = float(np.mean([p['lat'] for p in el['geometry']]))
            lon = float(np.mean([p['lon'] for p in el['geometry']]))
        else:
            continue
        for ref in str(tags['ref']).split(';'):
            rows.append((ref.strip(), lat, lon))
    return pd.DataFrame(rows, columns=['name', 'lat', 'lon'])


def index(stands: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Canonical name -> position. Duplicates farther apart than
    SAME_STAND_M are two different places and the name is dropped."""
    out = {}
    df = stands.assign(key=stands['name'].map(canon)).dropna(subset=['key'])
    for key, g in df.groupby('key', sort=True):
        lat, lon = g['lat'].mean(), g['lon'].mean()
        if metres(g['lat'], g['lon'], lat, lon).max() <= SAME_STAND_M:
            out[key] = (float(lat), float(lon))
    return out


def match(key: str | None, idx: dict, rule: str):
    """Position of canonical stand `key` under one rule, or None."""
    if not key:
        return None
    if rule == 'exact':
        return idx.get(key)
    if rule == 'substands':
        if not key[-1].isdigit():
            return None
        pts = [idx[k] for k in idx
               if len(k) == len(key) + 1 and k.startswith(key) and k[-1].isalpha()]
        if not pts:
            return None
        lat, lon = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
        if metres([p[0] for p in pts], [p[1] for p in pts], lat, lon).max() > SAME_STAND_M:
            return None
        return float(lat), float(lon)
    if rule == 'parent':
        if len(key) > 1 and key[-1].isalpha() and key[-2].isdigit():
            return idx.get(key[:-1])
        return None
    raise ValueError(rule)


def locate(stand, maps: dict[str, dict], airport: str | None = None) -> dict:
    """Best position for one raw stand name, given {'xplane': idx, 'osm': idx}."""
    key = canon(stand)
    if key is None or key in NOT_A_STAND:
        return {'key': key, 'lat': np.nan, 'lon': np.nan, 'source': None, 'rule': None}
    for rule in RULES:
        for source in SOURCE_ORDER.get((airport, key), SOURCES):
            pos = match(key, maps.get(source, {}), rule)
            if pos is not None:
                return {'key': key, 'lat': pos[0], 'lon': pos[1],
                        'source': source, 'rule': rule}
    return {'key': key, 'lat': np.nan, 'lon': np.nan, 'source': None, 'rule': None}


def load_maps(icao: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Both maps' stand indexes, plus raw X-Plane stands and runway ends."""
    xp_stands, ends = parse_apt((XPLANE_DIR / f'{icao}.dat').read_text(encoding='latin-1'))
    osm_path = OSM_DIR / f'osm_{icao}.json'
    osm_stands = (parse_osm(json.loads(osm_path.read_text())) if osm_path.exists()
                  else pd.DataFrame(columns=['name', 'lat', 'lon']))
    maps = {'xplane': index(xp_stands), 'osm': index(osm_stands)}
    return maps, xp_stands, osm_stands, ends
