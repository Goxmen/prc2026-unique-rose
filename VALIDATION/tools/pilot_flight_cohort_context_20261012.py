"""Four fixed input-only cohort-coverage dates; no model or label loading.

Reads compact scalar caches only. Neighbors retain paired original-clock and
own physical witnesses; no peer raw points or precomputed predictions enter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp/research_20261012/model_diversity'
DAYS = ['2025-01-10', '2025-07-10', '2026-01-10', '2026-07-10']
K, WINDOW_MS, ALIAS_MS = 16, 1800000, 120000
COLS = ['mvt_id', 'airport', 'stand', 'runway', 'to_ms', 'clock_ms', 'hex',
        'stand_lat', 'stand_lon', 'count', 'raw_count', 'raw_ground_count']
SOURCES = {
    '2025': ['data/derived/sparse_track_sequence_20261010/full_index.parquet',
             'data/external/opdi/opdi_features.parquet',
             'data/external/adsblol_heatmap/adsb_features.parquet',
             'data/external/adsblol_heatmap/adsb_stand_features.parquet'],
    '2026': ['data/derived/sparse_track_submit_20261010/old/full_index.parquet',
             'data/derived/final_tables/old/opdi.parquet',
             'data/derived/final_tables/old/adsb.parquet',
             'data/derived/final_tables/old/adsb_stand.parquet']}


def art(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=h.hexdigest())


def load(con, day):
    paths = [ROOT / p for p in SOURCES[day[:4]]]
    start = int(pd.Timestamp(day, tz='UTC').timestamp() * 1000)
    q = con.execute(f"SELECT {','.join(COLS)} FROM read_parquet(?) "
                    'WHERE to_ms BETWEEN ? AND ? ORDER BY mvt_id',
                    [str(paths[0]), start-WINDOW_MS, start+86400000+WINDOW_MS-1]).df()
    assert q.mvt_id.is_unique and len(q) < 20000
    con.register('allowed_queries', q)
    f = con.execute('''SELECT q.*, op.op_matched, op.op_ground_sec,
        op.op_own_stand_sec, ab.ab_matched, st.ab_stand_n, st.ab_stand_exit_sec
        FROM allowed_queries q
        LEFT JOIN read_parquet(?) op USING(mvt_id)
        LEFT JOIN read_parquet(?) ab USING(mvt_id)
        LEFT JOIN read_parquet(?) st USING(mvt_id) ORDER BY q.mvt_id''',
        list(map(str, paths[1:]))).df()
    con.unregister('allowed_queries')
    assert len(f) == len(q) and f.mvt_id.is_unique
    forbidden = {'y', 'taxi_sec', 'TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt', 'pred'}
    assert not forbidden.intersection(f.columns)
    return f, start


def clocks(f):
    raw = (f.to_ms.to_numpy(float)-f.clock_ms.to_numpy(float, na_value=np.nan))/1000
    values = np.column_stack([raw, f.op_ground_sec, f.op_own_stand_sec,
                              f.ab_stand_exit_sec]).astype(float)
    masks = np.isfinite(values) & (values >= 0) & (values <= 10800)
    masks[:, 1:3] &= f.op_matched.eq(1).to_numpy()[:, None]
    masks[:, 3] &= (f.ab_matched.eq(1) & f.ab_stand_n.gt(0)).to_numpy()
    return values, masks


def neighbors(f):
    """Stable all-query ID-based neighbor lists, plus input-alias exclusions."""
    answer, dropped, available = {}, {}, {}
    for _, g in f.groupby('airport', observed=True):
        g = g.sort_values(['to_ms', 'mvt_id'], kind='stable')
        ids = g.mvt_id.to_numpy(np.int64)
        t = g.to_ms.to_numpy(np.int64)
        h = g.hex.to_numpy(float, na_value=np.nan)
        c = g.clock_ms.to_numpy(float, na_value=np.nan)
        stand = g.stand.astype('string').fillna('').to_numpy(str)
        runway = g.runway.astype('string').fillna('').to_numpy(str)
        for j, ident in enumerate(ids):
            lo = np.searchsorted(t, t[j]-WINDOW_MS, side='left')
            hi = np.searchsorted(t, t[j]+WINDOW_MS, side='right')
            take = np.arange(lo, hi)
            other = ids[take] != ident
            samehex = np.isfinite(h[j]) & (h[take] == h[j]) & (abs(t[take]-t[j]) <= ALIAS_MS)
            signature = (stand[j] != '') & (runway[j] != '') & np.isfinite(c[j])
            alias = (signature & (stand[take] == stand[j]) & (runway[take] == runway[j])
                     & (t[take] == t[j]) & (c[take] == c[j])) | samehex
            dropped[int(ident)] = int(np.sum(other & alias))
            take = take[other & ~alias]
            available[int(ident)] = len(take)
            order = np.lexsort((ids[take], t[take], abs(t[take]-t[j])))
            answer[int(ident)] = ids[take[order[:K]]]
    return answer, dropped, available


def describe(f, start):
    selected, aliases, available = neighbors(f)
    values, valid = clocks(f)
    paired = valid[:, 0] & valid[:, 1:].any(1)
    ownphysical = valid[:, 1:].any(1)
    positions = pd.Series(np.arange(len(f)), index=f.mvt_id)
    t = f.to_ms.to_numpy(np.int64)
    runway = f.runway.astype('string').fillna('').to_numpy(str)
    lat, lon = f.stand_lat.to_numpy(float), f.stand_lon.to_numpy(float)
    rows = []
    for j in np.flatnonzero((t >= start) & (t < start+86400000)):
        ident = int(f.mvt_id.iloc[j]); ix = positions.loc[selected[ident]].to_numpy(int)
        pair = paired[ix]
        same = (runway[j] != '') & (runway[ix] != '') & (runway[ix] == runway[j])
        distance = 6371000*np.hypot(np.deg2rad(lat[ix]-lat[j]),
            np.deg2rad(lon[ix]-lon[j])*np.cos(np.deg2rad((lat[ix]+lat[j])/2)))
        disagreement = valid[ix, 0, None] & valid[ix, 1:] & (abs(values[ix, :1]-values[ix, 1:]) > 120)
        dt = (t[ix]-t[j])/1000
        rows.append(dict(mvt_id=ident, airport=str(f.airport.iloc[j]),
            missing_original_clock=bool(pd.isna(f.clock_ms.iloc[j])),
            query_no_physical=bool(~ownphysical[j]), query_sparse=bool(f['count'].iloc[j] < 2),
            n=len(ix), pool=available[ident], aliases_removed=aliases[ident],
            paired=int(pair.sum()), paired_same_runway=int((pair & same).sum()),
            paired_stand300m=int((pair & (distance <= 300)).sum()),
            paired_disagreement120=int(disagreement.any(1).sum()),
            span_sec=float(np.ptp(dt)) if len(ix) > 1 else 0.,
            furthest_sec=float(abs(dt).max()) if len(ix) else 0.))
    return pd.DataFrame(rows)


def summary(f):
    return dict(rows=len(f), any_neighbor=float(f.n.gt(0).mean()),
        median_neighbors=float(f.n.median()), truncated_share=float(f.pool.gt(K).mean()),
        median_full_pool=float(f.pool.median()), aliases_removed=int(f.aliases_removed.sum()),
        any_paired=float(f.paired.gt(0).mean()), median_paired=float(f.paired.median()),
        paired_same_runway=float(f.paired_same_runway.gt(0).mean()),
        paired_stand300m=float(f.paired_stand300m.gt(0).mean()),
        paired_disagreement120=float(f.paired_disagreement120.gt(0).mean()),
        median_span_sec=float(f.span_sec.median()), median_furthest_sec=float(f.furthest_sec.median()))


def checks():
    f = pd.DataFrame(dict(mvt_id=[1,2,3,4,5,6,7],airport=['A']*6+['B'],
        stand='S',runway='R',to_ms=[0,0,1000,1800000,1800001,-1800000,0],
        clock_ms=[-60000,-60000,-50000,np.nan,np.nan,np.nan,-60000],
        hex=[1,1,2,3,4,5,1]))
    a,_,_=neighbors(f);b,_,_=neighbors(f.sample(frac=1,random_state=91))
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert a[1].tolist()==[3,6,4] and a[7].size==0
    poison=f.assign(y=1e9,BLOCK_TIME_UTC_mvt='2099-01-01')
    z,_,_=neighbors(poison);assert all(np.array_equal(a[k],z[k]) for k in a)
    q=pd.DataFrame(dict(to_ms=[10800000,1000,0],clock_ms=[0,np.nan,1000],
        op_ground_sec=[10800,500,600],op_own_stand_sec=[-1,100,700],op_matched=[1,0,1],
        ab_matched=[1,1,0],ab_stand_n=[1,0,1],ab_stand_exit_sec=[0,500,600]))
    _,v=clocks(q);np.testing.assert_array_equal(v,[[1,1,0,1],[0,0,0,0],[0,1,1,0]])
    return dict(input_only_allowlist=True, label_poisoning_invariance=True, row_order_invariance=True,
        query_and_known_alias_removal=True, inclusive_time_boundaries=True,
        same_airport=True, no_context=True, exact_physical_gates=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    proof=checks()
    if a.check:print(proof);return
    started=time.monotonic();reports=[];OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'cohort_context_pilot.json';assert not dest.exists()
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for day in DAYS:
            f,start=load(con,day);r=describe(f,start)
            reports.append(dict(day=day,all=summary(r),
                strata={name:summary(r.loc[mask]) for name,mask in {
                    'no_original_AOBT':r.missing_original_clock,'no_own_physical':r.query_no_physical,
                    'under2_own_points':r.query_sparse,
                    'missing_AOBT_and_no_own_physical':r.missing_original_clock&r.query_no_physical}.items() if mask.any()},
                airport={apt:summary(g) for apt,g in r.groupby('airport')}))
    sources=[art(ROOT/p) for ps in SOURCES.values() for p in ps]
    value=dict(design=art(OUT/'cohort_context_design.json'),source=art(__file__),sources=sources,
        checks=proof,dates=DAYS,seconds=time.monotonic()-started,reports=reports,
        no_labels_predictions_scores_or_raw_points=True)
    dest.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(seconds=value['seconds'],reports=[dict(day=x['day'],all=x['all'],no_plan=x['strata'].get('no_original_AOBT')) for x in reports]),indent=2))


if __name__=='__main__':main()
