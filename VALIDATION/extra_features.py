"""Inputs suggested by other teams' write-ups (2026-10-03): arrival taxi-in durations, peer clocks,
runway configuration and departure bursts.

Built once for every departure in `training_*.parquet` and `ranking.parquet` from movement fields
only. Departure taxi-out labels and block times are never read: departures contribute their takeoff
time, runway, stand and flight-plan AOBT_3 (an input column, present in the ranking file);
arrivals contribute landing time, runway, stand and taxi-in duration (supplied for ranking
arrivals too). Every event pool is the union of 2025 and the ranking months, so a row near a month
edge sees whatever neighbours exist.

ARRIVAL_FEATURES (idea 1; OpenAir):
    sa_last_taxiin_sec    taxi-in of the last arrival at this stand whose in-block (landing +
                          taxi-in) is strictly before takeoff
    sa_prev_taxiin_sec    taxi-in of the arrival at this stand before that one
    ra_last_taxiin_sec    taxi-in of the last arrival that landed on this departure's runway before
                          takeoff
    ra_last_land_age_sec  takeoff minus that landing

PEER_FEATURES (idea 2; Phoenix). A departure's clock proxy is takeoff minus its AOBT_3, kept when
0 < proxy <= 3 h. Statistics are over the airport's other departures (never the row itself):
    pc_apt_mean_pm30, pc_apt_std_pm30, pc_apt_n_pm30   takeoff within 30 min either side
    pc_apt_mean_next30                                 takeoff in the 30 min after
    pc_rwy_mean_pm30, pc_rwy_n_pm30                    same runway, within 30 min either side
    pc_zone_mean_pm60, pc_zone_n_pm60                  stand in the same 300 m map cell
                                                       (public stand map), within 60 min
    pc_rwy_prev_proxy, pc_rwy_next_proxy               proxy of the previous / next departure on the
                                                       runway that has one, if within 30 min
    pc_own_minus_apt                                   own proxy minus pc_apt_mean_pm30

RUNWAY_FEATURES (idea 5; smart-jigsaw):
    rc_burst_prev, rc_burst_next   departures before / after this one on the runway in an unbroken
                                   chain of takeoffs less than 120 s apart
    rc_rwy_streak_sec              takeoff minus the first takeoff of this runway's current streak
                                   (a streak breaks at a 45 min gap between its takeoffs)
    rc_dep_rwys_60, rc_arr_rwys_60 runways used at least twice by departures / arrivals at the
                                   airport in the 60 min before takeoff
    rc_dep_set_60, rc_arr_set_60   those runway sets as categories

Build: .venv/bin/python VALIDATION/extra_features.py  ->  data/derived/extra_features.parquet
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]

PATH = ROOT / 'data/derived/extra_features.parquet'
ARRIVAL_FEATURES = ['sa_last_taxiin_sec', 'sa_prev_taxiin_sec', 'ra_last_taxiin_sec', 'ra_last_land_age_sec']
PEER_FEATURES = ['pc_apt_mean_pm30', 'pc_apt_std_pm30', 'pc_apt_n_pm30', 'pc_apt_mean_next30', 'pc_rwy_mean_pm30',
                 'pc_rwy_n_pm30', 'pc_zone_mean_pm60', 'pc_zone_n_pm60', 'pc_rwy_prev_proxy', 'pc_rwy_next_proxy',
                 'pc_own_minus_apt']
RUNWAY_FEATURES = ['rc_burst_prev', 'rc_burst_next', 'rc_rwy_streak_sec', 'rc_dep_rwys_60', 'rc_arr_rwys_60',
                   'rc_dep_set_60', 'rc_arr_set_60']
CATEGORICAL = ['rc_dep_set_60', 'rc_arr_set_60']
GROUPS = dict(arrtaxi=ARRIVAL_FEATURES, peers=PEER_FEATURES, rwycfg=RUNWAY_FEATURES)
S = 1_000_000_000          # ns per second
PROXY_MAX, BURST_GAP, STREAK_GAP, ZONE_M = 3 * 3600, 120, 45 * 60, 300


def ns(series):
    t = pd.to_datetime(series, utc=True)
    ok = t.notna().to_numpy()
    return np.where(ok, t.dt.tz_convert(None).astype('datetime64[ns]').astype('int64'), 0), ok


def key(*parts):
    """'a:b:...' per position (missing if any part is missing); parts are aligned by position."""
    as_str = lambda p: pd.Series(np.asarray(p, dtype=object)).astype('string')
    k = as_str(parts[0])
    for p in parts[1:]:
        k = k + ':' + as_str(p)
    return k.to_numpy(object)


def arrival_features(dep, arr):
    """sa_* by stand (in-block before takeoff) and ra_* by runway (landing before takeoff)."""
    t, t_ok = ns(dep.MVT_TIME_UTC_mvt)
    land, l_ok = ns(arr.MVT_TIME_UTC_mvt)
    taxi = pd.to_numeric(arr.taxi_sec, errors='coerce').to_numpy(float)
    good = l_ok & np.isfinite(taxi) & (taxi >= 0)
    out = pd.DataFrame(index=dep.index, columns=ARRIVAL_FEATURES, dtype=float)
    d = pd.DataFrame({'row': np.arange(len(dep)), 't': t, 'stand': key(dep.ADEP_mvt, dep.STAND_mvt),
                      'rwy': key(dep.ADEP_mvt, dep.RUNWAY_mvt)})[t_ok]
    a = pd.DataFrame({'inb': land + (np.where(good, taxi, 0) * S).astype(np.int64), 'land': land, 'taxi': taxi,
                      'stand': key(arr.ADES_mvt, arr.STAND_mvt), 'rwy': key(arr.ADES_mvt, arr.RUNWAY_mvt)})[good]
    s = a.dropna(subset=['stand']).sort_values(['stand', 'inb'], kind='mergesort')
    s['prev_taxi'] = s.groupby('stand', sort=False).taxi.shift()
    m = pd.merge_asof(d.dropna(subset=['stand']).sort_values('t'), s.sort_values('inb')[['inb', 'stand', 'taxi', 'prev_taxi']],
                      left_on='t', right_on='inb', by='stand', allow_exact_matches=False, direction='backward')
    out.iloc[m.row.to_numpy(), 0] = m.taxi.to_numpy(float)
    out.iloc[m.row.to_numpy(), 1] = m.prev_taxi.to_numpy(float)
    r = a.dropna(subset=['rwy']).sort_values('land')[['land', 'rwy', 'taxi']]
    m = pd.merge_asof(d.dropna(subset=['rwy']).sort_values('t'), r, left_on='t', right_on='land', by='rwy',
                      allow_exact_matches=False, direction='backward')
    out.iloc[m.row.to_numpy(), 2] = m.taxi.to_numpy(float)
    out.iloc[m.row.to_numpy(), 3] = (m.t - m.land).to_numpy(float) / S
    return out


def window_stats(group, t, v, lo, hi):
    """Per row: sum, sum of squares and count of valid v over other rows of its group with
    t_j in [t_i + lo, t_i + hi] (seconds), excluding the row itself."""
    n = len(t)
    s1, s2, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    valid = np.isfinite(v)
    order = np.lexsort((t, group))
    g, tt, vv, ok = group[order], t[order], np.where(valid, v, 0)[order], valid[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        ts_ = tt[a:b]
        c1 = np.r_[0, np.cumsum(vv[a:b])]
        c2 = np.r_[0, np.cumsum(vv[a:b] ** 2)]
        c0 = np.r_[0, np.cumsum(ok[a:b])]
        i = np.searchsorted(ts_, ts_ + lo, side='left')      # t is in seconds here
        j = np.searchsorted(ts_, ts_ + hi, side='right')
        idx = order[a:b]
        s1[idx], s2[idx], cnt[idx] = c1[j] - c1[i], c2[j] - c2[i], c0[j] - c0[i]
    # Remove the row itself where its own time lies in its window.
    inside = (lo <= 0) & (hi >= 0) & valid
    s1[inside] -= v[inside]
    s2[inside] -= v[inside] ** 2
    cnt[inside] -= 1
    return s1, s2, cnt


def mean_std(s1, s2, cnt):
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    return mean, np.sqrt(np.clip(var, 0, None))


def codes(values):
    return pd.factorize(pd.Series(values, dtype=object).fillna('\x00missing'))[0]


def peer_features(dep, stand_xy):
    t, t_ok = ns(dep.MVT_TIME_UTC_mvt)
    aobt, a_ok = ns(dep.AOBT_3_flt)
    proxy = np.where(t_ok & a_ok, (t - aobt) / S, np.nan)
    proxy = np.where((proxy > 0) & (proxy <= PROXY_MAX), proxy, np.nan)
    ts_ = t // S
    apt = codes(dep.ADEP_mvt.astype(str).to_numpy())
    rwy_key = key(dep.ADEP_mvt, dep.RUNWAY_mvt)
    rwy = codes(rwy_key)
    out = pd.DataFrame(index=dep.index)
    mean, std = mean_std(*window_stats(apt, ts_, proxy, -1800, 1800))
    out['pc_apt_mean_pm30'], out['pc_apt_std_pm30'] = mean, std
    out['pc_apt_n_pm30'] = window_stats(apt, ts_, proxy, -1800, 1800)[2]
    out['pc_apt_mean_next30'] = mean_std(*window_stats(apt, ts_, proxy, 1, 1800))[0]
    s1, s2, c = window_stats(rwy, ts_, proxy, -1800, 1800)
    has_rwy = pd.notna(rwy_key)
    out['pc_rwy_mean_pm30'] = np.where(has_rwy, mean_std(s1, s2, c)[0], np.nan)
    out['pc_rwy_n_pm30'] = np.where(has_rwy, c, np.nan)
    sk = key(dep.ADEP_mvt, dep.STAND_mvt)
    xy = stand_xy.reindex(sk)
    zone = np.where(xy.x.notna(), key(dep.ADEP_mvt, (xy.x // ZONE_M).astype('Int64').astype('string').to_numpy(),
                                      (xy.y // ZONE_M).astype('Int64').astype('string').to_numpy()), None)
    zc = codes(zone)
    s1, s2, c = window_stats(zc, ts_, proxy, -3600, 3600)
    has_zone = pd.notna(zone)
    out['pc_zone_mean_pm60'] = np.where(has_zone, mean_std(s1, s2, c)[0], np.nan)
    out['pc_zone_n_pm60'] = np.where(has_zone, c, np.nan)
    # Previous / next departure on the runway that has a proxy, within 30 min.
    f = pd.DataFrame({'row': np.arange(len(dep)), 't': ts_, 'rwy': rwy_key, 'p': proxy})[t_ok & has_rwy]
    peers = f.dropna(subset=['p']).sort_values('t')
    left = f.sort_values('t')
    for name, direction in (('pc_rwy_prev_proxy', 'backward'), ('pc_rwy_next_proxy', 'forward')):
        m = pd.merge_asof(left, peers[['t', 'rwy', 'p']].rename(columns={'p': 'q', 't': 'tq'}), left_on='t',
                          right_on='tq', by='rwy', allow_exact_matches=False, direction=direction,
                          tolerance=1800)
        col = np.full(len(dep), np.nan)
        col[m.row.to_numpy()] = m.q.to_numpy(float)
        out[name] = col
    out['pc_own_minus_apt'] = proxy - out.pc_apt_mean_pm30.to_numpy()
    return out


def chain(group, t, gap):
    """Length of the unbroken run of earlier / later events within `gap` seconds, per row."""
    order = np.lexsort((t, group))
    g, tt = group[order], t[order]
    close = np.r_[False, (g[1:] == g[:-1]) & (np.diff(tt) < gap * S)]
    prev = np.zeros(len(t), int)
    for i in range(1, len(t)):
        prev[i] = prev[i - 1] + 1 if close[i] else 0
    nxt = np.zeros(len(t), int)
    for i in range(len(t) - 2, -1, -1):
        nxt[i] = nxt[i + 1] + 1 if close[i + 1] else 0
    out_prev, out_next = np.empty(len(t)), np.empty(len(t))
    out_prev[order], out_next[order] = prev, nxt
    return out_prev, out_next


def runway_sets(apt_q, t_q, events_apt, events_rwy, events_t, window, min_uses=2):
    """For each query: distinct runways at its airport used >= min_uses times in [t - window, t)."""
    n = len(t_q)
    count = np.zeros(n)
    labels = np.full(n, '', dtype=object)
    for ap in np.unique(apt_q):
        q = np.flatnonzero(apt_q == ap)
        e = events_apt == ap
        for r in sorted(set(events_rwy[e]) - {None}):
            times = np.sort(events_t[e & (events_rwy == r)])
            used = (np.searchsorted(times, t_q[q], 'left') - np.searchsorted(times, t_q[q] - window * S, 'left')) >= min_uses
            count[q] += used
            labels[q] = np.where(used, labels[q] + ',' + r, labels[q])
    labels = np.array([s.lstrip(',') or 'none' for s in labels], dtype=object)
    return count, labels


def runway_features(dep, arr):
    t, t_ok = ns(dep.MVT_TIME_UTC_mvt)
    rwy_key = key(dep.ADEP_mvt, dep.RUNWAY_mvt)
    has = t_ok & pd.notna(rwy_key)
    grp = codes(np.where(has, rwy_key, None))
    out = pd.DataFrame(index=dep.index)
    prev, nxt = chain(grp, t, BURST_GAP)
    out['rc_burst_prev'] = np.where(has, prev, np.nan)
    out['rc_burst_next'] = np.where(has, nxt, np.nan)
    # Streak start: first takeoff after the last >= 45 min gap on this runway.
    order = np.lexsort((t, grp))
    g, tt = grp[order], t[order]
    new = np.r_[True, (g[1:] != g[:-1]) | (np.diff(tt) >= STREAK_GAP * S)]
    start = np.maximum.accumulate(np.where(new, np.arange(len(tt)), 0))
    streak = np.empty(len(t))
    streak[order] = (tt - tt[start]) / S
    out['rc_rwy_streak_sec'] = np.where(has, streak, np.nan)
    apt = dep.ADEP_mvt.astype(str).to_numpy(object)
    drwy = dep.RUNWAY_mvt.astype(object).where(dep.RUNWAY_mvt.notna(), None).to_numpy(object)
    out['rc_dep_rwys_60'], out['rc_dep_set_60'] = runway_sets(apt, t, apt[t_ok], drwy[t_ok], t[t_ok], 3600)
    la, l_ok = ns(arr.MVT_TIME_UTC_mvt)
    aapt = arr.ADES_mvt.astype(str).to_numpy(object)
    arwy = arr.RUNWAY_mvt.astype(object).where(arr.RUNWAY_mvt.notna(), None).to_numpy(object)
    out['rc_arr_rwys_60'], out['rc_arr_set_60'] = runway_sets(apt, t, aapt[l_ok], arwy[l_ok], la[l_ok], 3600)
    for c in CATEGORICAL:
        out[c] = np.where(t_ok, out[c].to_numpy(object), None)
    return out


def build(path=PATH):
    import overnight_run
    import stand_features as sf
    started = time.monotonic()
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    cols_d = ['mvt_id', 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt']
    cols_a = ['ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'taxi_sec']
    deps = pd.concat([dep[cols_d], rank[cols_d]], ignore_index=True)        # departure labels never taken
    arrs = pd.concat([arr[cols_a], rank_arr[cols_a]], ignore_index=True)    # arrival taxi-in is an input
    if deps.mvt_id.duplicated().any():
        raise ValueError('Duplicate departure IDs across training and ranking')
    coords, _, origin = sf.load()
    o = origin.reindex(coords.airport)
    k = np.cos(np.radians(o.lat.to_numpy()))
    stand_xy = pd.DataFrame({'x': (coords.lon.to_numpy() - o.lon.to_numpy()) * 111_320 * k,
                             'y': (coords.lat.to_numpy() - o.lat.to_numpy()) * 110_540},
                            index=pd.Index(key(coords.airport, coords.stand)))
    parts = [arrival_features(deps, arrs)]
    print(f'arrival inputs {time.monotonic() - started:.0f}s', flush=True)
    parts.append(peer_features(deps, stand_xy))
    print(f'peer inputs {time.monotonic() - started:.0f}s', flush=True)
    parts.append(runway_features(deps, arrs))
    print(f'runway inputs {time.monotonic() - started:.0f}s', flush=True)
    table = pd.concat(parts, axis=1)
    table.insert(0, 'mvt_id', deps.mvt_id.to_numpy())
    for c in CATEGORICAL:
        table[c] = table[c].astype('string')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.parquet.tmp')
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f'Wrote {path}: {len(table):,} departures in {time.monotonic() - started:.0f}s', flush=True)
    print(table.drop(columns=['mvt_id'] + CATEGORICAL).describe().T[['count', 'mean', '50%', 'max']].round(1).to_string())


def load(path=PATH):
    t = pd.read_parquet(path).set_index('mvt_id')
    for c in CATEGORICAL:
        t[c] = t[c].astype('category')
    return t


def add(X, ids, table, features):
    """Append `features` for rows whose movement IDs are `ids` (aligned to X)."""
    part = table.reindex(np.asarray(ids))[features]
    part.index = X.index
    if part.index.has_duplicates:
        raise ValueError('Duplicate rows')
    return pd.concat([X, part], axis=1)


if __name__ == '__main__':
    build()
