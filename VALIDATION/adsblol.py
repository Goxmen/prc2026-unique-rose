"""ADSB.lol inputs: what the open heatmap saw of each departure and of the airport around it.

Source: `data/external/adsblol_heatmap/` from `VALIDATION/tools/download_adsblol_heatmap.py`:
every aircraft within 20 km of the ten airports, on the ground or at most 10,000 ft,
sampled every 10 s, with callsigns. ADSB.lol globe_history archives (CC0 / ODbL);
declare as an external dataset.

A departure is matched to a heatmap aircraft whose callsign equals its `CALLSIGN_flt`
(or `FLIGHT_mvt` when that is missing) and was seen within [takeoff - 4 h,
takeoff + 10 min]. Among those, the aircraft whose last ground point at the departure
airport is closest before takeoff + 60 s wins, if it is within 15 min of takeoff.
Only positions up to takeoff + 60 s are used. Block times and taxi labels are never read.

OWN_FEATURES (missing when unmatched or never seen on the ground):

    ab_matched         1 if the callsign matched an aircraft, else 0
    ab_seg_sec         takeoff - first ground point of the departure's ground segment
                       (segments break where an aircraft goes unheard for more than 10 min)
    ab_move_sec        takeoff - first point of that segment moving at 5 kt or more
    ab_still_end_sec   takeoff - end of the segment's last stationary stretch (< 1 kt) of 3 min or more
    ab_seg_n           ground points in the segment
    ab_first_gs        ground speed (kt) at the segment's first point; high = first heard already taxiing
    ab_last_gnd_sec    takeoff - last ground point (how well the heatmap aligns with the recorded takeoff)
    ab_prev_gap_sec    how long the aircraft went unheard on the ground before the segment, capped at 24 h

AIRPORT_FEATURES, from every aircraft in the airport's box (not only matched ones):

    ab_moving_10/30    mean distinct aircraft per minute moving on the ground (>= 5 kt), 10/30 min before takeoff
    ab_ground_10       mean distinct aircraft per minute on the ground, 10 min before takeoff
    ab_low_air_10      mean distinct aircraft per minute airborne at most 3,000 ft, 10 min before takeoff

NEIGHBOUR_FEATURES, from the airport's other departures (never the row itself):

    ab_nb_seg_mean_60  mean ab_seg_sec of matched departures that took off in the 60 min before
    ab_nb_seg_n_60     how many of them have ab_seg_sec

STAND_FEATURES (second table, `adsb_stand_features.parquet`), from the matched aircraft's ground
points within [takeoff - 4 h, takeoff + 60 s] and the public stand map (`stand_features.load`)
for `STAND_mvt` and `RUNWAY_mvt`:

    ab_stand_min_m     closest approach to the departure's own stand (m)
    ab_stand_n         ground points within 100 m of the own stand
    ab_stand_exit_sec  takeoff - last point within 100 m of the own stand (the pushback witness)
    ab_exit_gap_sec    segment start - that last stand point; large means the stand point is the
                       previous arrival, not this departure
    ab_seg_stand_m     distance from the segment's first point to the own stand (late detection)
    ab_seg_rwy_m       distance from the segment's first point to the start of the take-off runway
    ab_nb_exit_mean_60 mean ab_stand_exit_sec of the airport's departures in the 60 min before
                       whose stand point belongs to the departure (ab_exit_gap_sec <= 600)
    ab_nb_exit_n_60    how many of them

COVERAGE_FEATURES (derived in `load_all`):

    ab_cov_day         share of the airport's departures that UTC day seen on the ground (ab_seg_sec
                       present). Receiver coverage comes and goes (LEMD: none in January-February
                       2025, about 30% in July-August); without it, zero traffic counts from a blind
                       receiver look like an empty airport.

Build once: `.venv/bin/python VALIDATION/adsblol.py`; models read it through `add_adsblol`.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
import glob
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import stand_geometry as sg

ROOT = Path(__file__).resolve().parents[1]
HEATMAP = ROOT / 'data/external/adsblol_heatmap'
FEATURES_PATH = HEATMAP / 'adsb_features.parquet'
STAND_PATH = HEATMAP / 'adsb_stand_features.parquet'
TRAIN_GLOB = ROOT / 'data/training_*.parquet'
RANKING = ROOT / 'data/ranking.parquet'
GROUND, MOVING_GS, STILL_GS = -123, 50, 10   # heatmap encodings: ground flag, 0.1 kt units
LOW_AIR_ALT = 3000 // 25
SEGMENT_GAP_MS, STILL_MIN_MS = 10 * 60_000, 3 * 60_000
LOOKBACK_MS, MATCH_TOL_MS, AFTER_MS = 4 * 3_600_000, 15 * 60_000, 60_000
OWN_FEATURES = ['ab_matched', 'ab_seg_sec', 'ab_move_sec', 'ab_still_end_sec', 'ab_seg_n', 'ab_first_gs',
                'ab_last_gnd_sec', 'ab_prev_gap_sec']
AIRPORT_FEATURES = ['ab_moving_10', 'ab_moving_30', 'ab_ground_10', 'ab_low_air_10']
NEIGHBOUR_FEATURES = ['ab_nb_seg_mean_60', 'ab_nb_seg_n_60']
ADSB_FEATURES = OWN_FEATURES + AIRPORT_FEATURES + NEIGHBOUR_FEATURES
STAND_FEATURES = ['ab_stand_min_m', 'ab_stand_n', 'ab_stand_exit_sec', 'ab_exit_gap_sec', 'ab_seg_stand_m',
                  'ab_seg_rwy_m', 'ab_nb_exit_mean_60', 'ab_nb_exit_n_60']
COVERAGE_FEATURES = ['ab_cov_day']
STAND_R_M, OWN_EXIT_GAP_SEC = 100.0, 600.0


def epoch_ms(series):
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    return s.astype('datetime64[ms]').astype('int64')


def departures():
    cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'STAND_mvt',
            'RUNWAY_mvt']
    files = sorted(glob.glob(str(TRAIN_GLOB))) + [str(RANKING)]
    dep = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    dep = dep[dep.PHASE_mvt == 'DEP'].drop_duplicates('MVT_ID_mvt')
    out = pd.DataFrame({'mvt_id': dep.MVT_ID_mvt.to_numpy(), 'airport': dep.ADEP_mvt.astype(str).to_numpy(),
                        'to_ms': epoch_ms(dep.MVT_TIME_UTC_mvt).to_numpy()})
    out['cs'] = (dep.CALLSIGN_flt.fillna(dep.FLIGHT_mvt).fillna('').astype(str)
                 .str.replace(' ', '', regex=False).str.upper().to_numpy())
    out['day'] = pd.to_datetime(out.to_ms, unit='ms').dt.strftime('%Y-%m-%d')
    out['stand_key'] = out.airport + ':' + dep.STAND_mvt.astype('string').fillna('').to_numpy()
    out['runway_key'] = out.airport + ':' + dep.RUNWAY_mvt.astype('string').fillna('').to_numpy()
    return out


def read_day(day):
    path = HEATMAP / 'positions' / f'{day}.parquet'
    if not path.exists():
        return None, None
    pos = pd.read_parquet(path)
    pos = pd.DataFrame({'ts': epoch_ms(pos.ts).to_numpy(), 'airport': pos.airport.astype(str).to_numpy(),
                        'hex': pos.hex.to_numpy(), 'lat': pos.lat.to_numpy(), 'lon': pos.lon.to_numpy(),
                        'alt': pos.alt.to_numpy(), 'gs': pos.gs.to_numpy()})
    calls = pd.read_parquet(HEATMAP / 'callsigns' / f'{day}.parquet')
    calls['cs'] = calls.callsign.astype(str).str.strip().str.upper()
    return pos, calls[['hex', 'cs', 'first_ms', 'last_ms']]


def segments(ground):
    """Ground points sorted per aircraft and airport, with segment and stationary-stretch summaries."""
    g = ground.sort_values(['hex', 'airport', 'ts'], kind='stable').reset_index(drop=True)
    same = (g.hex.to_numpy()[1:] == g.hex.to_numpy()[:-1]) & (g.airport.to_numpy()[1:] == g.airport.to_numpy()[:-1])
    gap = np.diff(g.ts.to_numpy())
    new = np.r_[True, ~same | (gap > SEGMENT_GAP_MS)]
    g['seg'] = np.cumsum(new)
    g['prev_gap'] = np.r_[np.nan, np.where(same, gap, np.nan)].astype(float)
    seg = g.groupby('seg').agg(seg_start=('ts', 'min'), seg_n=('ts', 'size'))
    seg['first_gs'] = g.groupby('seg').gs.first() / 10
    seg['lat0'], seg['lon0'] = g.groupby('seg').lat.first(), g.groupby('seg').lon.first()
    seg['prev_gap'] = g.groupby('seg').prev_gap.first()
    seg['move'] = g[g.gs >= MOVING_GS].groupby('seg').ts.min()
    still = g.gs.between(0, STILL_GS - 1).to_numpy()
    run = np.cumsum(np.r_[True, (still[1:] != still[:-1]) | new[1:]])
    s = g.loc[still].assign(run=run[still]).groupby('run').agg(seg=('seg', 'first'), a=('ts', 'min'), b=('ts', 'max'))
    s = s[s.b - s.a >= STILL_MIN_MS]
    seg['still_end'] = s.groupby('seg').b.max()
    return g, seg


def airport_minutes(pos):
    """Distinct aircraft per airport-minute: moving on the ground, on the ground, low airborne."""
    m = pos.assign(minute=pos.ts // 60_000)
    ground = m.alt == GROUND
    kinds = {'moving': ground & (m.gs >= MOVING_GS), 'ground': ground, 'low_air': ~ground & (m.alt <= LOW_AIR_ALT)}
    return {k: m[mask].groupby(['airport', 'minute']).hex.nunique() for k, mask in kinds.items()}


def window_mean(counts, airport, minute, width):
    """Mean of per-minute counts over [minute - width, minute - 1]; minutes without points count 0."""
    out = np.full(len(minute), np.nan)
    for ap in np.unique(airport):
        rows = np.flatnonzero(airport == ap)
        c = counts.xs(ap, level='airport') if ap in counts.index.get_level_values(0) else pd.Series(dtype=float)
        lo = int(minute[rows].min()) - width - 1
        hi = int(minute[rows].max()) + 1
        dense = np.zeros(hi - lo + 1)
        c = c[(c.index >= lo) & (c.index <= hi)]
        dense[c.index.to_numpy() - lo] = c.to_numpy()
        cum = np.r_[0, np.cumsum(dense)]
        end = minute[rows] - lo
        out[rows] = (cum[end] - cum[end - width]) / width
    return out


def load_day(day):
    """The day's positions and callsigns, plus the previous day's last hours for the lookback."""
    pos, calls = read_day(day)
    if pos is None:
        return None, None
    prev_pos, prev_calls = read_day(str(date.fromisoformat(day) - timedelta(days=1)))
    day_ms = int(pd.Timestamp(day).value // 1_000_000)
    if prev_pos is not None:
        pos = pd.concat([prev_pos[prev_pos.ts >= day_ms - LOOKBACK_MS - 30 * 60_000], pos], ignore_index=True)
        calls = pd.concat([prev_calls, calls], ignore_index=True)
    return pos, calls


def match(out, pos, calls):
    """Callsign candidates, ground segments and each departure's matched last ground point."""
    cand = out.merge(calls, on='cs')
    cand = cand[(cand.first_ms <= cand.to_ms + 10 * 60_000) & (cand.last_ms >= cand.to_ms - LOOKBACK_MS)]
    cand = cand[['mvt_id', 'airport', 'to_ms', 'hex']].drop_duplicates()
    g, seg = segments(pos[pos.alt == GROUND])
    cand = cand.assign(t=cand.to_ms + AFTER_MS).sort_values('t')
    hit = pd.merge_asof(cand, g[['ts', 'hex', 'airport', 'seg']].sort_values('ts'), left_on='t', right_on='ts',
                        by=['hex', 'airport'], direction='backward', tolerance=MATCH_TOL_MS + AFTER_MS)
    hit = hit.dropna(subset=['ts']).sort_values(['mvt_id', 'ts']).drop_duplicates('mvt_id', keep='last')
    return cand, g, seg, hit.join(seg, on='seg')


def build_day(day, deps):
    """Own and airport features for one day's departures (takeoff on `day`, UTC)."""
    pos, calls = load_day(day)
    if pos is None:
        return None
    out = deps[['mvt_id', 'airport', 'to_ms', 'cs']].copy()
    pos = pos[pos.ts <= out.to_ms.max() + AFTER_MS]
    cand, g, seg, hit = match(out, pos, calls)
    out['ab_matched'] = out.mvt_id.isin(cand.mvt_id).astype(float)
    h = hit.set_index('mvt_id').reindex(out.mvt_id)
    to = out.to_ms.to_numpy(float)
    out['ab_seg_sec'] = (to - h.seg_start.to_numpy(float)) / 1000
    out['ab_move_sec'] = (to - h.move.to_numpy(float)) / 1000
    out['ab_still_end_sec'] = (to - h.still_end.to_numpy(float)) / 1000
    out['ab_seg_n'] = h.seg_n.to_numpy(float)
    out['ab_first_gs'] = h.first_gs.to_numpy(float)
    out['ab_last_gnd_sec'] = (to - h.ts.to_numpy(float)) / 1000
    out['ab_prev_gap_sec'] = np.minimum(h.prev_gap.to_numpy(float) / 1000, 86_400)
    counts = airport_minutes(pos)
    minute, airport = (out.to_ms // 60_000).to_numpy(), out.airport.to_numpy()
    for name, kind, width in (('ab_moving_10', 'moving', 10), ('ab_moving_30', 'moving', 30),
                              ('ab_ground_10', 'ground', 10), ('ab_low_air_10', 'low_air', 10)):
        out[name] = window_mean(counts[kind], airport, minute, width)
    return out.drop(columns='cs')


def neighbours(table):
    """Mean ab_seg_sec of the airport's departures that took off in the 60 min before."""
    table = table.sort_values(['airport', 'to_ms', 'mvt_id']).reset_index(drop=True)
    mean, count = np.full(len(table), np.nan), np.zeros(len(table))
    for _, rows in table.groupby('airport').indices.items():
        t = table.to_ms.to_numpy()[rows]
        v = table.ab_seg_sec.to_numpy()[rows]
        has = ~np.isnan(v)
        cs, cn = np.r_[0, np.cumsum(np.where(has, v, 0))], np.r_[0, np.cumsum(has)]
        lo = np.searchsorted(t, t - 3_600_000, 'left')
        hi = np.searchsorted(t, t, 'left')   # strictly before: never the row itself
        n = cn[hi] - cn[lo]
        count[rows] = n
        mean[rows] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    table['ab_nb_seg_mean_60'], table['ab_nb_seg_n_60'] = mean, count
    return table


def build_features(path=FEATURES_PATH, workers=8):
    deps = departures()
    days = sorted(d for d in deps.day.unique() if (HEATMAP / 'positions' / f'{d}.parquet').exists())
    groups = {d: g for d, g in deps.groupby('day') if d in set(days)}
    with ProcessPoolExecutor(workers) as pool:
        parts = [p for p in pool.map(build_day, days, [groups[d] for d in days], chunksize=4) if p is not None]
    table = neighbours(pd.concat(parts, ignore_index=True))
    # Every departure gets a row; days without a heatmap (or no takeoff time) stay missing, not 0.
    table = deps[['mvt_id', 'airport', 'to_ms']].merge(table.drop(columns=['airport', 'to_ms']), on='mvt_id', how='left')
    table = table[['mvt_id', 'airport', 'to_ms'] + ADSB_FEATURES].sort_values('mvt_id').reset_index(drop=True)
    tmp = path.with_suffix('.parquet.tmp')
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return table, days


def stand_day(day, deps, tables):
    """Stand-exit and position inputs for one day's departures (STAND_FEATURES before neighbours)."""
    pos, calls = load_day(day)
    if pos is None:
        return None
    out = deps[['mvt_id', 'airport', 'to_ms', 'cs', 'stand_key', 'runway_key']].copy()
    pos = pos[pos.ts <= out.to_ms.max() + AFTER_MS]
    _, g, _, hit = match(out[['mvt_id', 'airport', 'to_ms', 'cs']], pos, calls)
    c, e, _ = tables
    spos = c.assign(k=c.airport + ':' + c.stand).set_index('k')
    epos = e.assign(k=e.airport + ':' + e.runway).set_index('k')
    out['slat'], out['slon'] = out.stand_key.map(spos.lat), out.stand_key.map(spos.lon)
    rlat, rlon = out.runway_key.map(epos.lat), out.runway_key.map(epos.lon)
    h = hit[['mvt_id', 'hex', 'seg_start', 'lat0', 'lon0']].merge(out[['mvt_id', 'airport', 'to_ms', 'slat', 'slon']],
                                                                     on='mvt_id')
    pts = h[['mvt_id', 'hex', 'airport', 'to_ms', 'slat', 'slon']].merge(g[['hex', 'airport', 'ts', 'lat', 'lon']],
                                                                          on=['hex', 'airport'])
    pts = pts[(pts.ts >= pts.to_ms - LOOKBACK_MS) & (pts.ts <= pts.to_ms + AFTER_MS)]
    pts['dist'] = sg.metres(pts.lat.to_numpy() / 1e6, pts.lon.to_numpy() / 1e6, pts.slat.to_numpy(), pts.slon.to_numpy())
    near = pts[pts.dist <= STAND_R_M]
    agg = pd.DataFrame({'min_m': pts.groupby('mvt_id').dist.min(), 'n': near.groupby('mvt_id').size(),
                        'exit': near.groupby('mvt_id').ts.max()}).reindex(out.mvt_id)
    hh = h.set_index('mvt_id').reindex(out.mvt_id)
    to = out.to_ms.to_numpy(float)
    has_stand = out.slat.notna().to_numpy() & hh.hex.notna().to_numpy()
    res = out[['mvt_id', 'airport', 'to_ms']].copy()
    res['ab_stand_min_m'] = np.where(has_stand, agg.min_m.to_numpy(float), np.nan)
    res['ab_stand_n'] = np.where(has_stand, agg.n.fillna(0).to_numpy(float), np.nan)
    res['ab_stand_exit_sec'] = (to - agg.exit.to_numpy(float)) / 1000
    res['ab_exit_gap_sec'] = (hh.seg_start.to_numpy(float) - agg.exit.to_numpy(float)) / 1000
    lat0, lon0 = hh.lat0.to_numpy(float) / 1e6, hh.lon0.to_numpy(float) / 1e6
    res['ab_seg_stand_m'] = sg.metres(lat0, lon0, out.slat.to_numpy(float), out.slon.to_numpy(float))
    res['ab_seg_rwy_m'] = sg.metres(lat0, lon0, rlat.to_numpy(float), rlon.to_numpy(float))
    return res


def prior_mean(table, value, valid, name):
    """Mean of `value` over the airport's departures in the 60 min before (strictly), where `valid`."""
    table = table.sort_values(['airport', 'to_ms', 'mvt_id']).reset_index(drop=True)
    mean, count = np.full(len(table), np.nan), np.zeros(len(table))
    ok = valid(table).to_numpy() & table[value].notna().to_numpy()
    for _, rows in table.groupby('airport').indices.items():
        t, v, has = table.to_ms.to_numpy()[rows], table[value].to_numpy()[rows], ok[rows]
        cs, cn = np.r_[0, np.cumsum(np.where(has, v, 0))], np.r_[0, np.cumsum(has)]
        lo, hi = np.searchsorted(t, t - 3_600_000, 'left'), np.searchsorted(t, t, 'left')
        n = cn[hi] - cn[lo]
        count[rows], mean[rows] = n, np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    table[f'{name}_mean_60'], table[f'{name}_n_60'] = mean, count
    return table


def build_stand_features(path=STAND_PATH, workers=8):
    import stand_features as sf
    deps = departures()
    days = sorted(d for d in deps.day.unique() if (HEATMAP / 'positions' / f'{d}.parquet').exists())
    groups = {d: g for d, g in deps.groupby('day') if d in set(days)}
    tables = sf.load()
    with ProcessPoolExecutor(workers) as pool:
        parts = [p for p in pool.map(stand_day, days, [groups[d] for d in days], [tables] * len(days), chunksize=4)
                 if p is not None]
    table = prior_mean(pd.concat(parts, ignore_index=True), 'ab_stand_exit_sec',
                       lambda t: t.ab_exit_gap_sec <= OWN_EXIT_GAP_SEC, 'ab_nb_exit')
    table = deps[['mvt_id']].merge(table.drop(columns=['airport', 'to_ms']), on='mvt_id', how='left')
    table = table[['mvt_id'] + STAND_FEATURES].sort_values('mvt_id').reset_index(drop=True)
    tmp = path.with_suffix('.parquet.tmp')
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return table, days


def load_features(path=FEATURES_PATH):
    return pd.read_parquet(path).set_index('mvt_id')


def load_all():
    """Both tables and the daily coverage share, one row per departure."""
    table = load_features().join(load_features(STAND_PATH), how='left')
    day = table.to_ms // 86_400_000
    seen = table.ab_seg_sec.notna().astype(float).where(table.ab_matched.notna())
    table['ab_cov_day'] = seen.groupby([table.airport, day]).transform('mean')
    return table


def add_adsblol(X, raw, table, features=ADSB_FEATURES):
    """Attach the ADSB.lol inputs to a model matrix whose rows align with `raw`."""
    if not X.index.equals(raw.index):
        raise ValueError('Matrix and raw movement rows must align before adding ADSB.lol inputs')
    overlap = set(X.columns) & set(features)
    if overlap:
        raise ValueError(f'ADSB.lol inputs already present: {sorted(overlap)}')
    missing = ~raw['mvt_id'].isin(table.index)
    if missing.any():
        raise ValueError(f'{int(missing.sum())} departures missing from the ADSB.lol feature table')
    values = table.loc[raw['mvt_id'].to_numpy(), list(features)].to_numpy(float)
    return pd.concat([X, pd.DataFrame(values, index=X.index, columns=list(features))], axis=1)


if __name__ == '__main__':
    if sys.argv[1:] == ['stand']:
        table, days = build_stand_features()
        print(f'Wrote {STAND_PATH} ({len(table):,} departures over {len(days)} days)')
    else:
        table, days = build_features()
        print(f'Wrote {FEATURES_PATH} ({len(table):,} departures over {len(days)} days)')
    sys.exit(0)
