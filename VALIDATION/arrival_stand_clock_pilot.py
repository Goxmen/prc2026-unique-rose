"""Bounded arrival stand-stop feasibility pilot; no departure targets or fits.

Select the first independently witnessed stop at the supplied arrival stand,
then compare with supplied arrival in-block. The latter never selects points.
A future departure feature would require both confirmation and supplied in-block
strictly before the query, using only its permitted reference months.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path
import time

import numpy as np
import pandas as pd

import aircraft_rotation as rotation
import stand_features as sf
import stand_geometry as sg

ROOT = rotation.ROOT
OUT = ROOT / 'tmp/research_20261010/signal_scout/arrival_stand_clock'
DAYS = ['2025-01-15', '2025-07-15']
SETTINGS = dict(days=DAYS, window_after_landing_sec=5400, stand_radius_m=100,
    max_stop_speed_deciknot=10, min_approach_speed_deciknot=20,
    min_stop_points=3, min_stop_span_sec=30, max_point_gap_sec=120,
    max_approach_age_sec=300, ground_alt_code=-123,
    selection='earliest consecutive slow/near run with observed preceding outside-stand approach',
    clock_comparison='supplied arrival in-block minus independently selected first stop point',
    causal_availability='max(stop confirmation, supplied arrival in-block); strictly before departure',
    labels='supplied arrivals only, never used to select physical witness',
    identity='unchanged audited unique near-movement aircraft identity',
    source_licensing='existing ADSB.lol cache; X-Plane Gateway GPLv2+ and OSM ODbL maps')


def witness(points, landing_ms, lat, lon):
    """No recorded in-block/taxi-in argument can influence the physical stop."""
    p = points.loc[points.ts_ms.between(landing_ms, landing_ms + 5400000)].copy()
    if len(p) < 4 or not np.isfinite([lat, lon]).all():
        return None
    p = p.sort_values(['ts_ms', 'lat', 'lon', 'alt', 'gs']).drop_duplicates('ts_ms')
    t = p.ts_ms.to_numpy(np.int64)
    dist = sg.metres(p.lat.to_numpy()/1e6, p.lon.to_numpy()/1e6, lat, lon)
    speed = p.gs.to_numpy()
    ground = p.alt.to_numpy() == -123
    slow_near = ground & (speed >= 0) & (speed <= 10) & (dist <= 100)
    run = np.cumsum(np.r_[True, (slow_near[1:] != slow_near[:-1]) | (np.diff(t) > 120000)])
    for key in np.unique(run[slow_near]):
        inds = np.flatnonzero((run == key) & slow_near)
        if len(inds) < 3 or t[inds[-1]]-t[inds[0]] < 30000:
            continue
        first = inds[0]
        # The immediately preceding retained point must show arrival approach,
        # rather than a receiver restarting on an already parked aircraft.
        prev = first-1
        if (prev < 0 or t[first]-t[prev] > 300000 or dist[prev] <= 100
                or not ground[prev] or speed[prev] < 20):
            continue
        confirms = inds[(np.arange(len(inds)) >= 2) & (t[inds]-t[first] >= 30000)]
        confirmation = int(confirms[0])
        return dict(stop_ms=int(t[first]), confirm_ms=int(t[confirmation]),
            approach_ms=int(t[prev]), approach_m=float(dist[prev]),
            stop_m=float(dist[first]), confirm_m=float(dist[confirmation]),
            stop_points=int(confirmation-first+1),
            physical_taxiin_sec=float((t[first]-landing_ms)/1000))
    return None


def run(days=DAYS):
    started = time.monotonic()
    con = rotation.connection()
    identity = rotation.DEFAULT/'identity.parquet'
    # The identity cache contains supplied arrival fields but no departure labels.
    q = con.execute('''SELECT mvt_id,airport,stand,hex,epoch_ms(t)::BIGINT landing_ms,
        epoch_ms(inblock)::BIGINT inblock_ms,taxiin,strftime(t,'%Y-%m-%d') AS "day"
        FROM read_parquet(?) WHERE phase='ARR'
        AND strftime(t,'%Y-%m-%d') IN (SELECT unnest(?)) ORDER BY mvt_id''',
        [str(identity), list(days)]).df()
    geometry = pd.read_parquet(sf.COORDS,columns=['airport','stand','lat','lon','source','rule'])
    geometry = geometry.rename(columns=dict(lat='stand_lat',lon='stand_lon',source='map_source',rule='map_rule'))
    q = q.merge(geometry,on=['airport','stand'],how='left',validate='many_to_one')
    sources, results = set(), []
    for day in days:
        dates = [day, str(date.fromisoformat(day)+timedelta(days=1))]
        paths = [rotation.HEATMAP/'positions'/f'{d}.parquet' for d in dates]
        if not all(p.exists() for p in paths):
            raise ValueError('Pilot requires complete current/following source days')
        sources.update(paths)
        p = con.execute('''SELECT airport,hex,epoch_ms(ts)::BIGINT ts_ms,lat,lon,alt,gs
            FROM read_parquet(?)''', [[str(v) for v in paths]]).df()
        groups = {(a, int(h)): v for (a,h),v in p.groupby(['airport','hex'],sort=False)}
        for row in q.loc[q.day.eq(day)].itertuples():
            if pd.isna(row.hex) or pd.isna(row.stand_lat):
                continue
            pts = groups.get((row.airport,int(row.hex)))
            if pts is None:
                continue
            found = witness(pts, row.landing_ms, row.stand_lat, row.stand_lon)
            if found is not None:
                results.append(dict(mvt_id=row.mvt_id, **found))
    columns = ['mvt_id','stop_ms','confirm_ms','approach_ms','approach_m','stop_m',
               'confirm_m','stop_points','physical_taxiin_sec']
    matches = pd.DataFrame(results, columns=columns)
    out = q.merge(matches,on='mvt_id',how='left',validate='one_to_one')
    out['recorded_minus_physical_sec'] = (out.inblock_ms-out.stop_ms)/1000
    out['available_ms'] = out[['inblock_ms','confirm_ms']].max(axis=1).where(out.stop_ms.notna() & out.inblock_ms.notna())
    OUT.mkdir(parents=True,exist_ok=True)
    table = OUT/'arrivals.parquet'
    out.to_parquet(table,index=False)
    summary=[]
    for (day,airport), rows in out.groupby(['day','airport']):
        lag=rows.recorded_minus_physical_sec.dropna()
        summary.append(dict(day=day,airport=airport,arrivals=len(rows),identified=int(rows.hex.notna().sum()),
            mapped=int(rows.stand_lat.notna().sum()),witnessed=int(rows.stop_ms.notna().sum()),
            lag_quantiles=None if lag.empty else {str(q):float(lag.quantile(q)) for q in [.1,.5,.9]}))
    map_paths=[sf.COORDS]
    report=dict(settings=SETTINGS,days=list(days),rows=len(out),witnessed=int(out.stop_ms.notna().sum()),
        summary=summary,seconds=time.monotonic()-started,
        sources={str(p.relative_to(ROOT)):rotation.sha256(p) for p in sorted(sources|set(map_paths)|{identity})},
        helpers={str(Path(m.__file__).relative_to(ROOT)):rotation.sha256(Path(m.__file__)) for m in [rotation,sg,sf]},
        source_sha256=rotation.sha256(Path(__file__)),table_sha256=rotation.sha256(table))
    rotation.write_json(OUT/'pilot.json',report)
    print(report,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',nargs='+',default=DAYS)
    args=parser.parse_args()
    run(args.days)
