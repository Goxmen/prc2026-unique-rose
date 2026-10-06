"""Idea 4: taxi-route length and turning from each stand to each take-off runway end (OpenStreetMap).

For each airport, OpenStreetMap ways tagged aeroway=taxiway, taxilane or runway within 6 km of the
aerodrome reference point (ODbL; fetched once from Overpass, cached under
data/external/osm_taxi/) form a graph: way vertices are nodes, consecutive vertices are edges
weighted by length (runway edges count double, so routes prefer taxiways). Stands (public stand
map) and take-off runway ends (`runway_ends.parquet`, the points behind `sc_rwy_end_m`) are snapped
to the nearest node of the graph's largest connected part. For every stand and runway end:

    rt_path_m      snap distances + shortest route length (m)
    rt_turn_deg    total absolute heading change along the route (segments under 5 m skipped)
    rt_ratio       rt_path_m / straight-line stand-to-runway-end distance

Inputs depend only on stand, runway and the maps, never on times or labels.
Build: .venv/bin/python VALIDATION/taxi_route.py  ->  data/derived/taxi_routes.parquet
"""
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
from download_adsblol_heatmap import AIRPORTS
import stand_features as sf

CACHE = ROOT / 'data/external/osm_taxi'
PATH = ROOT / 'data/derived/taxi_routes.parquet'
OVERPASS = ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter',
            'https://overpass.private.coffee/api/interpreter']
AGENT = 'PRC-2026-unique-rose research (github.com/Goxmen/PRC)'
FEATURES = ['rt_path_m', 'rt_turn_deg', 'rt_ratio']
RADIUS_M, RUNWAY_WEIGHT, MIN_SEG_M = 6000, 2.0, 5.0


def fetch(icao):
    path = CACHE / f'{icao}.json'
    if path.exists():
        return json.loads(path.read_text())
    lat, lon = AIRPORTS[icao]
    query = f"""[out:json][timeout:180];
way["aeroway"~"^(taxiway|taxilane|runway)$"](around:{RADIUS_M},{lat},{lon});
out geom;"""
    body, errors = None, []
    for attempt in range(6):                     # public servers are busy at times: rotate and back off
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            req = Request(url, data=urlencode({'data': query}).encode(), headers={'User-Agent': AGENT})
            with urlopen(req, timeout=300) as resp:
                body = resp.read()
            break
        except OSError as e:
            errors.append(f'{url}: {e}')
            time.sleep(20 * (attempt + 1))
    if body is None:
        raise RuntimeError(f'{icao}: Overpass failed: {errors}')
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    time.sleep(5)
    return json.loads(body)


def xy(lat, lon, ref):
    k = np.cos(np.radians(ref[0]))
    return (np.asarray(lon) - ref[1]) * 111_320 * k, (np.asarray(lat) - ref[0]) * 110_540


def graph(data, ref):
    G, pos = nx.Graph(), {}
    for el in data['elements']:
        if el.get('type') != 'way' or 'geometry' not in el:
            continue
        kind = el.get('tags', {}).get('aeroway')
        nodes, geom = el['nodes'], el['geometry']
        x, y = xy([g['lat'] for g in geom], [g['lon'] for g in geom], ref)
        for n, a, b in zip(nodes, x, y):
            pos[n] = (float(a), float(b))
        for i in range(len(nodes) - 1):
            d = float(np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]))
            w = d * (RUNWAY_WEIGHT if kind == 'runway' else 1.0)
            if not G.has_edge(nodes[i], nodes[i + 1]) or G[nodes[i]][nodes[i + 1]]['weight'] > w:
                G.add_edge(nodes[i], nodes[i + 1], weight=w, length=d)
    biggest = max(nx.connected_components(G), key=len)
    G = G.subgraph(biggest).copy()
    return G, {n: pos[n] for n in G.nodes}


def turning(points):
    """Total absolute heading change (degrees) along a polyline, skipping segments under MIN_SEG_M."""
    p = np.asarray(points)
    seg = np.diff(p, axis=0)
    keep = np.hypot(seg[:, 0], seg[:, 1]) >= MIN_SEG_M
    seg = seg[keep]
    if len(seg) < 2:
        return 0.0
    h = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
    d = (np.diff(h) + 180) % 360 - 180
    return float(np.abs(d).sum())


def routes(icao, coords, ends):
    ref = AIRPORTS[icao]
    G, pos = graph(fetch(icao), ref)
    nodes = np.array(list(pos))
    nxy = np.array([pos[n] for n in nodes])

    def snap(lat, lon):
        x, y = xy(lat, lon, ref)
        d = np.hypot(nxy[:, 0] - x, nxy[:, 1] - y)
        i = int(d.argmin())
        return nodes[i], float(d[i]), (float(x), float(y))

    st = coords[coords.airport == icao]
    snapped = {r.stand: snap(r.lat, r.lon) for r in st.itertuples()}
    rows = []
    for e in ends[ends.airport == icao].itertuples():
        src, src_d, src_xy = snap(e.lat, e.lon)
        dist, paths = nx.single_source_dijkstra(G, src, weight='weight')
        for stand, (node, d, sxy) in snapped.items():
            if node not in paths:
                continue
            path = paths[node]
            length = float(sum(G[a][b]['length'] for a, b in zip(path[:-1], path[1:])))
            total = d + length + src_d
            straight = float(np.hypot(sxy[0] - src_xy[0], sxy[1] - src_xy[1]))
            rows.append(dict(airport=icao, stand=stand, runway=e.runway, rt_path_m=total,
                             rt_turn_deg=turning([sxy] + [pos[n] for n in path[::-1]] + [src_xy]),
                             rt_ratio=total / straight if straight > 0 else np.nan, stand_snap_m=d, end_snap_m=src_d))
    return pd.DataFrame(rows), dict(nodes=G.number_of_nodes(), edges=G.number_of_edges())


def build(path=PATH):
    coords, ends, _ = sf.load()
    parts, info = [], {}
    for icao in sorted(AIRPORTS):
        started = time.monotonic()
        part, info[icao] = routes(icao, coords, ends)
        parts.append(part)
        print(f'{icao}: {info[icao]["nodes"]:,} nodes, {len(part):,} stand-runway routes, median '
              f'{part.rt_path_m.median():.0f} m, ratio {part.rt_ratio.median():.2f}, stand snap median '
              f'{part.stand_snap_m.median():.0f} m ({time.monotonic() - started:.0f}s)', flush=True)
    table = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    print(f'Wrote {path}: {len(table):,} routes')


def load(path=PATH):
    t = pd.read_parquet(path)
    return t.set_index(t.airport + ':' + t.stand + ':' + t.runway)[FEATURES]


def add(X, raw_airport, raw_stand, raw_runway, table):
    k = (pd.Series(np.asarray(raw_airport, dtype=object)).astype('string') + ':'
         + pd.Series(np.asarray(raw_stand, dtype=object)).astype('string') + ':'
         + pd.Series(np.asarray(raw_runway, dtype=object)).astype('string'))
    part = table.reindex(k.to_numpy(object))
    part.index = X.index
    return pd.concat([X, part], axis=1)


if __name__ == '__main__':
    build()
