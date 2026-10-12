"""Bounded cached-position pilot; no taxi targets, fitting, inference or downloads."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'data/derived/sparse_track_sequence_20261010'
OUT=ROOT/'VALIDATION/reports/spatial_peer_motion_20261012'
DAYS=['2025-03-10','2025-09-10']
DESIGN=dict(days=DAYS,selection='smallest SHA256(movement-ID)64 eligible flights per airport/day; quarter/middle/three-quarter retained ground points',
    radius_m=300.,past_seconds=600.,negative_time_shift_seconds=21600.,max_peers=16,min_peers=3,
    eligibility='known aircraft/stand; at least3 retained ground points with speed0..60kt,0..7200s before takeoff, sameUTCday and >=200m from stand',
    donors='retained ground positions with observed speed0..60kt, same airport/UTCday; exact raw absolute timestamp, no interpolation',
    exclusion='all same known aircraft and same movement-ID or exact known takeoff/actualclock/stand/runway aliases',
    prediction='median of per-aircraft observed speed medians; equally weighted aircraft; matched count across actual-past,6hour-shift null,spatial-only',
    positive='observed peer motion, not absent-peer imputation; unsupported rows remain unpredicted',
    gate='each date >=10% matched-support coverage; each date MAE at least5% lower than BOTH controls and pooled at least10% lower',
    limits='90seconds,oneCPUthread,noGPU; original two shared-training pilot dates, no date/threshold search',
    no_taxi_labels=True,no_fit=True,no_CV=True,no_download=True,
    interpretation='Source-speed observability/predictability only; no taxi-error headroom or candidate claim')


def art(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def read(path):return json.loads(Path(path).read_text())


def digest_id(value):return hashlib.sha256(str(int(value)).encode()).hexdigest()


def xy(lat,lon,origin_lat,origin_lon):
    return np.column_stack([(lon-origin_lon)*111320*np.cos(np.radians(origin_lat)),(lat-origin_lat)*110540])


def aliases(index):
    valid=index.clock_ms.notna()&index.stand.notna()&index.runway.notna()
    groups={}
    for _,f in index.loc[valid].groupby(['airport','to_ms','clock_ms','stand','runway'],dropna=False):
        ids=set(f.mvt_id.astype(int))
        for key in ids:groups[key]=ids
    return groups


def per_aircraft(frame,distances):
    if frame.empty:return frame.assign(distance=[]).groupby('hex').agg(speed=('speed','median'),distance=('distance','min'),age=('age','min'))
    return frame.assign(distance=distances).groupby('hex').agg(speed=('speed','median'),distance=('distance','min'),age=('age','min'))


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'identity.json').exists(),'Preserve frozen source-only pilot'
    records={d:read(CACHE/f'full_chunks/{d}.json') for d in DAYS}
    sources=[art(__file__),art(CACHE/'full.json')]
    for day,r in records.items():
        for key in ['index','points']:
            actual=art(ROOT/r[key]['path']);assert actual==r[key]
            sources.append(actual)
        sources.append(art(CACHE/f'full_chunks/{day}.json'))
    identity=dict(design=DESIGN,sources=sources,before_any_source_outcomes=True)
    write(OUT/'identity.json',identity)
    all_rows=[];coverage=[];integrity=[];failure=None
    try:
        for day,record in records.items():
            index=pd.read_parquet(ROOT/record['index']['path'])
            assert index.mvt_id.is_unique and index.day.eq(day).all()
            assert not {'y','TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt'}&set(index)
            points=np.fromfile(ROOT/record['points']['path'],dtype=np.int32).reshape(-1,6)
            assert len(points)==record['stored_points']
            base_ms=int(pd.Timestamp(day,tz='UTC').timestamp()*1000)
            alias=aliases(index);frames=[];query=[];outside=0;raw_ground=0
            for q in index.itertuples(index=False):
                if pd.isna(q.hex):continue
                p=points[int(q.start):int(q.start)+int(q.count)]
                assert len(p)==q.count and np.all((p[:,0]>=-60000)&(p[:,0]<=7200000))
                assert np.all((p[:,1]>=-90000000)&(p[:,1]<=90000000)&(p[:,2]>=-180000000)&(p[:,2]<=180000000))
                ts=int(q.to_ms)-p[:,0].astype(np.int64)
                ground=(p[:,3]==-123)&(p[:,4]>=0)&(p[:,4]<=600)&(p[:,0]>=0)
                raw_ground+=int(ground.sum());same=(ts>=base_ms)&(ts<base_ms+86400000)
                outside+=int((ground&~same).sum());take=ground&same
                if not take.any():continue
                raw=p[take];absolute=ts[take]
                coordinate=xy(raw[:,1]/1e6,raw[:,2]/1e6,q.origin_lat,q.origin_lon)
                f=pd.DataFrame(dict(mvt_id=int(q.mvt_id),hex=int(q.hex),airport=q.airport,
                    ts=absolute,x=coordinate[:,0],ycoord=coordinate[:,1],speed=raw[:,4]/10.))
                frames.append(f)
                if np.isfinite(q.stand_lat+q.stand_lon):
                    stand=xy(np.array([q.stand_lat]),np.array([q.stand_lon]),q.origin_lat,q.origin_lon)[0]
                    eligible=np.flatnonzero(np.linalg.norm(coordinate-stand,axis=1)>=200)
                    if len(eligible)>=3:
                        chosen=eligible[np.rint(np.array([.25,.5,.75])*(len(eligible)-1)).astype(int)]
                        rows=f.iloc[np.unique(chosen)].copy();rows['sample_hash']=digest_id(q.mvt_id)
                        query.append(rows)
            donors=pd.concat(frames,ignore_index=True)
            duplicates=int(donors.duplicated(['airport','hex','ts']).sum())
            # Canonical original raw selector already resolves timestamp conflicts.
            assert donors.groupby(['airport','hex','ts'])[['x','ycoord','speed']].nunique().le(1).all().all()
            donors=donors.drop_duplicates(['airport','hex','ts']).reset_index(drop=True)
            queries=pd.concat(query,ignore_index=True)
            chosen=queries[['airport','mvt_id','sample_hash']].drop_duplicates().sort_values('sample_hash').groupby('airport').head(64)
            queries=queries[queries.mvt_id.isin(chosen.mvt_id)].reset_index(drop=True)
            integrity.append(dict(day=day,flights=len(index),known_aircraft=int(index.hex.notna().sum()),
                raw_retained_points=len(points),observed_ground_speed_points=raw_ground,
                prior_UTCday_ground_points_excluded=outside,deduplicated_points=duplicates,
                unique_donor_points=len(donors),eligible_query_flights=len(pd.concat(query).mvt_id.unique()),
                selected_query_flights=queries.mvt_id.nunique(),exact_signature_alias_edges=sum(len(v)-1 for v in alias.values()),
                missing_source_day_sides=int(index.left_source_missing.sum()+index.right_source_missing.sum())))
            for airport,qframe in queries.groupby('airport'):
                pool=donors[donors.airport.eq(airport)].reset_index(drop=True)
                tree=cKDTree(pool[['x','ycoord']].to_numpy())
                support=0
                for q in qframe.itertuples(index=False):
                    if time.monotonic()-started>90:raise TimeoutError('Frozen90second source-pilot bound')
                    nearby=tree.query_ball_point([q.x,q.ycoord],300.,workers=1)
                    f=pool.iloc[nearby].copy()
                    f=f[f.hex.ne(q.hex)&~f.mvt_id.isin(alias.get(int(q.mvt_id),{int(q.mvt_id)}))]
                    assert f.hex.ne(q.hex).all() and f.mvt_id.ne(q.mvt_id).all()
                    dist=np.hypot(f.x-q.x,f.ycoord-q.ycoord).to_numpy()
                    f['age']=(q.ts-f.ts)/1000.
                    actual=f.age.gt(0)&f.age.le(600)
                    shifted=base_ms+((q.ts-base_ms+21600000)%86400000)
                    null_age=(shifted-f.ts)/1000.
                    null=null_age.gt(0)&null_age.le(600)
                    a=per_aircraft(f.loc[actual],dist[actual]);b=per_aircraft(f.loc[null].assign(age=null_age[null]),dist[null])
                    c=per_aircraft(f,dist)
                    n=min(len(a),len(b),len(c),16)
                    item=dict(day=day,airport=airport,mvt_id=int(q.mvt_id),hex=int(q.hex),ts=int(q.ts),
                        observed_speed=float(q.speed),past_peers=len(a),null_peers=len(b),spatial_peers=len(c),matched_peers=n,
                        supported=n>=3)
                    if n>=3:
                        support+=1
                        # Same number of distinct aircraft, so gains cannot be
                        # a prediction from presence/count alone.
                        item.update(dynamic=float(a.sort_values(['age','distance']).head(n).speed.median()),
                            null=float(b.sort_values(['age','distance']).head(n).speed.median()),
                            spatial=float(c.sort_values(['distance','age']).head(n).speed.median()))
                    all_rows.append(item)
                coverage.append(dict(day=day,airport=airport,queries=len(qframe),matched_supported=support,
                    fraction=support/len(qframe),unique_donor_points=len(pool)))
            del points,index,donors,queries,frames,query
    except (AssertionError,TimeoutError) as error:
        failure=repr(error)
    f=pd.DataFrame(all_rows);f.to_parquet(OUT/'source_queries.parquet',index=False)
    scores={}
    if len(f):
        for day,g in f.groupby('day'):
            use=g[g.supported].copy();metrics={}
            for arm in ['dynamic','null','spatial']:
                if arm in use and len(use):
                    error=use[arm]-use.observed_speed
                    metrics[arm]=dict(mae=float(abs(error).mean()),rmse=float(np.sqrt((error**2).mean())))
            scores[day]=dict(queries=len(g),supported=len(use),fraction=len(use)/len(g),metrics=metrics)
    valid=not failure and len(scores)==2 and all(s['fraction']>=.1 and s['metrics'] for s in scores.values())
    gate=False;pooled={}
    if valid:
        use=f[f.supported]
        pooled={a:float(abs(use[a]-use.observed_speed).mean()) for a in ['dynamic','null','spatial']}
        gate=all(s['metrics']['dynamic']['mae']<=.95*s['metrics'][c]['mae'] for s in scores.values() for c in ['null','spatial'])
        gate=bool(gate and all(pooled['dynamic']<=.9*pooled[c] for c in ['null','spatial']))
    result=dict(status='completed' if not failure else 'failed',failure=failure,identity=art(OUT/'identity.json'),
        source_queries=art(OUT/'source_queries.parquet'),design=DESIGN,integrity=integrity,coverage=coverage,
        scores=scores,pooled_MAE=pooled,source_gate=gate,runtime_seconds=time.monotonic()-started,
        no_departure_targets_read=True,no_fitting_or_downloads=True,
        deployment='Past ground position timestamps from other known departures are present in cached public ranking tracks; source dates/windows are checked. A later coverage audit on2026 is required before any feature/model proposal.',
        limitation='Two training dates only; observed speed may not recover recorded off-block time, and route/direction is not map-matched. Same-query/count-matched comparisons address count/presence but do not establish causal congestion.')
    write(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ['status','failure','scores','pooled_MAE','source_gate','runtime_seconds']},indent=2))


if __name__=='__main__':
    with threadpool_limits(1):main()
