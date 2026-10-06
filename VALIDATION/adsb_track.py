"""More detail from each departure's ADSB.lol ground track (requested 2026-10-06).

`adsblol.py` already gives the track's timing (first ground point, first movement, end of the last
stop, stand exit), its first speed and how far its first point is from the stand and the runway.
This adds the shape of the track. The points are the same as `adsblol.py` uses: the matched
aircraft's ground segment (`adsblol.match`), up to takeoff + 60 s. "Taxi points" are those at least
2 min before takeoff, which leaves out the take-off roll.

TRACK_FEATURES (missing when the departure is not seen on the ground):

    ab_path_m          distance travelled along the segment (m); steps implying > 100 m/s are dropped
    ab_straight_m      straight-line distance from the segment's first point to its last (m)
    ab_max_gs          top ground speed over the taxi points (kt)
    ab_moving_share    share of taxi points moving at 5 kt or more
    ab_stop_sec        time standing still (< 1 kt) between taxi points at most 2 min apart (s)
    ab_turn_deg        total change of direction over taxi steps of at least 20 m (degrees)
    ab_first_bearing   direction of the first move: first point to the first point 30 m away (degrees)
    ab_first_x_m       first point east of the airport's centre (mean of its runway ends, m)
    ab_first_y_m       first point north of the airport's centre (m)
    ab_last_rwy_m      last ground point's distance to the start of the take-off runway (m)

Build once: `.venv/bin/python VALIDATION/adsb_track.py` (writes `adsb_track_features.parquet`
next to the other ADSB.lol tables). Block times and taxi labels are never read.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import adsblol as ab
import stand_geometry as sg

TRACK_PATH = ab.HEATMAP / 'adsb_track_features.parquet'
TRACK_FEATURES = ['ab_path_m', 'ab_straight_m', 'ab_max_gs', 'ab_moving_share', 'ab_stop_sec', 'ab_turn_deg',
                  'ab_first_bearing', 'ab_first_x_m', 'ab_first_y_m', 'ab_last_rwy_m']
ROLL_MS, MAX_STEP_MPS, STILL_DT_MS, TURN_STEP_M, FIRST_MOVE_M = 120_000, 100.0, 120_000, 20.0, 30.0


def bearing(lat1, lon1, lat2, lon2):
    mlat = np.radians((lat1 + lat2) / 2.0)
    return np.degrees(np.arctan2((lon2 - lon1) * np.cos(mlat), lat2 - lat1)) % 360


def track_day(day, deps, tables):
    pos, calls = ab.load_day(day)
    if pos is None:
        return None
    out = deps[['mvt_id', 'airport', 'to_ms', 'cs', 'runway_key']].copy()
    pos = pos[pos.ts <= out.to_ms.max() + ab.AFTER_MS]
    _, g, _, hit = ab.match(out[['mvt_id', 'airport', 'to_ms', 'cs']], pos, calls)
    _, e, origin = tables
    epos = e.assign(k=e.airport + ':' + e.runway).set_index('k')
    pts = hit[['mvt_id', 'seg', 'to_ms']].merge(g[['seg', 'ts', 'lat', 'lon', 'gs']], on='seg')
    pts = pts[pts.ts <= pts.to_ms + ab.AFTER_MS].sort_values(['mvt_id', 'ts'], kind='stable').reset_index(drop=True)
    res = out[['mvt_id']].copy()
    if pts.empty:
        return res.assign(**{c: np.nan for c in TRACK_FEATURES})
    lat, lon = pts.lat.to_numpy(float) / 1e6, pts.lon.to_numpy(float) / 1e6
    ts, gs, mid = pts.ts.to_numpy(float), pts.gs.to_numpy(float) / 10, pts.mvt_id.to_numpy()
    taxi = ts <= pts.to_ms.to_numpy(float) - ROLL_MS
    same = np.r_[False, mid[1:] == mid[:-1]]
    step = np.r_[0.0, sg.metres(lat[:-1], lon[:-1], lat[1:], lon[1:])]
    dt = np.r_[0.0, np.diff(ts)]
    ok = same & (step <= MAX_STEP_MPS * np.maximum(dt, 1) / 1000)
    pts['step'] = np.where(ok, step, 0.0)
    # still time: both ends of the step are taxi points below 1 kt and close in time
    still = (gs >= 0) & (gs < 1) & taxi                  # -1 in the heatmap means no speed
    pts['still'] = np.where(same & np.r_[False, still[:-1]] & still & (dt <= STILL_DT_MS), dt / 1000, 0.0)
    # turning: bearing of each taxi step of at least TURN_STEP_M, change between consecutive such steps
    long_step = same & taxi & (step >= TURN_STEP_M) & ok
    b = np.full(len(pts), np.nan)
    b[long_step] = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])[long_step[1:]]
    tb = pd.DataFrame({'mvt_id': mid[long_step], 'b': b[long_step]})
    tb['d'] = tb.groupby('mvt_id').b.diff()
    tb['d'] = np.abs((tb.d + 180) % 360 - 180)
    first = pts.groupby('mvt_id').first()
    last = pts.groupby('mvt_id').last()
    f_lat, f_lon = first.lat.to_numpy(float) / 1e6, first.lon.to_numpy(float) / 1e6
    # first move: first point at least FIRST_MOVE_M from the segment's first point
    d0 = sg.metres(lat, lon, pts.mvt_id.map(first.lat).to_numpy(float) / 1e6,
                   pts.mvt_id.map(first.lon).to_numpy(float) / 1e6)
    away = pts[d0 >= FIRST_MOVE_M].groupby('mvt_id').first()
    tp = pts[taxi]
    agg = pd.DataFrame({
        'ab_path_m': pts.groupby('mvt_id').step.sum(),
        'ab_straight_m': pd.Series(sg.metres(f_lat, f_lon, last.lat.to_numpy(float) / 1e6,
                                             last.lon.to_numpy(float) / 1e6), index=first.index),
        'ab_max_gs': tp.groupby('mvt_id').gs.max() / 10,
        'ab_moving_share': (tp.gs >= ab.MOVING_GS).groupby(tp.mvt_id).mean(),
        'ab_stop_sec': pts.groupby('mvt_id').still.sum(),
        'ab_turn_deg': tb.groupby('mvt_id').d.sum(),
    })
    agg['ab_first_bearing'] = pd.Series(bearing(f_lat[first.index.get_indexer(away.index)],
                                                f_lon[first.index.get_indexer(away.index)],
                                                away.lat.to_numpy(float) / 1e6, away.lon.to_numpy(float) / 1e6),
                                        index=away.index)
    ap = out.set_index('mvt_id').airport.reindex(first.index)
    olat, olon = ap.map(origin.lat).to_numpy(float), ap.map(origin.lon).to_numpy(float)
    agg['ab_first_x_m'] = pd.Series((f_lon - olon) * sg.M_PER_DEG_LAT * np.cos(np.radians(olat)), index=first.index)
    agg['ab_first_y_m'] = pd.Series((f_lat - olat) * sg.M_PER_DEG_LAT, index=first.index)
    rk = out.set_index('mvt_id').runway_key.reindex(first.index)
    agg['ab_last_rwy_m'] = pd.Series(sg.metres(last.lat.to_numpy(float) / 1e6, last.lon.to_numpy(float) / 1e6,
                                               rk.map(epos.lat).to_numpy(float), rk.map(epos.lon).to_numpy(float)),
                                     index=first.index)
    seen = res.mvt_id.isin(first.index).to_numpy()
    agg = agg.reindex(res.mvt_id)
    for c in TRACK_FEATURES:
        v = agg[c].to_numpy(float)
        if c in ('ab_path_m', 'ab_stop_sec', 'ab_turn_deg'):
            v = np.where(seen & np.isnan(v), 0.0, v)      # seen, but no qualifying step: zero, not missing
        res[c] = v
    return res


def build(path=TRACK_PATH, workers=8, days=None):
    import stand_features as sf
    deps = ab.departures()
    all_days = sorted(d for d in deps.day.unique() if (ab.HEATMAP / 'positions' / f'{d}.parquet').exists())
    days = days or all_days
    groups = {d: g for d, g in deps.groupby('day') if d in set(days)}
    tables = sf.load()
    with ProcessPoolExecutor(workers) as pool:
        parts = [p for p in pool.map(track_day, days, [groups[d] for d in days], [tables] * len(days), chunksize=4)
                 if p is not None]
    table = deps[['mvt_id']].merge(pd.concat(parts, ignore_index=True), on='mvt_id', how='left')
    table = table[['mvt_id'] + TRACK_FEATURES].sort_values('mvt_id').reset_index(drop=True)
    if path is not None:
        tmp = path.with_suffix('.parquet.tmp')
        table.to_parquet(tmp, index=False)
        tmp.replace(path)
    return table, days


def load(path=TRACK_PATH):
    return pd.read_parquet(path).set_index('mvt_id')


if __name__ == '__main__':
    started = time.monotonic()
    table, days = build()
    print(f'Wrote {TRACK_PATH} ({len(table):,} departures over {len(days)} days) in '
          f'{(time.monotonic() - started) / 60:.1f} min')
