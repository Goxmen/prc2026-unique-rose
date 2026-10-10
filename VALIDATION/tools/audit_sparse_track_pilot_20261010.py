"""Independent raw SQL reconstruction for fixed input-only pilot strata."""
from pathlib import Path
import json
import math
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import sparse_track_sequence as s
from aircraft_rotation import sha256, write_json


def run():
    manifest = json.loads((s.OUT/'pilot.json').read_text())
    assert manifest['source_sha256'] == sha256(Path(s.__file__))
    for name, digest in manifest['outputs'].items():
        assert sha256(s.OUT/name) == digest
    index = pd.read_parquet(s.OUT/'pilot_index.parquet')
    points = np.load(s.OUT/'pilot_points.npy', mmap_mode='r', allow_pickle=False)
    # Independently check every metadata lookup against the static source maps.
    coords = pd.read_parquet(s.sf.COORDS).dropna(subset=['stand', 'lat'])
    ends = pd.read_parquet(s.sf.ENDS)
    origin = ends.groupby('airport')[['lat', 'lon']].mean()
    for prefix, source, key in [('stand', coords, 'stand'), ('runway', ends, 'runway')]:
        merged = index[['airport', key]].merge(source[['airport', key, 'lat', 'lon']],
            on=['airport', key], how='left', validate='many_to_one')
        for axis in ['lat', 'lon']:
            np.testing.assert_allclose(index[f'{prefix}_{axis}'].to_numpy(float),
                                       merged[axis].to_numpy(float), equal_nan=True, atol=0, rtol=0)
    for axis in ['lat', 'lon']:
        np.testing.assert_allclose(index[f'origin_{axis}'], index.airport.map(origin[axis]), atol=0, rtol=0)
    chosen = []
    for day, g in index.groupby('day'):
        strata = [g.hex.isna(), g['count'].eq(1), g['count'].eq(2),
                  g['raw_count'].between(3, 7), g['raw_count'].between(8, 31),
                  g.raw_count.between(32, 128), g.raw_count.gt(128),
                  g.clock_ms.isna() & g['count'].ge(2),
                  (g.to_ms%86400000).lt(7200000) & g['count'].ge(2)]
        for mask in strata:
            chosen.extend(g.loc[mask].head(1).mvt_id.tolist())
    sample = index.loc[index.mvt_id.isin(chosen)].copy()
    con = duckdb.connect()
    con.execute("SET threads=1; SET TimeZone='UTC'; SET memory_limit='1GB'")
    con.register('query', sample[['mvt_id', 'airport', 'hex', 'to_ms']])
    # SQL ordering and window selection is separate from the builder's grouped
    # NumPy searchsorted path. All rows/times are selected without labels.
    reconstructed = con.execute('''WITH matched AS (
        SELECT q.mvt_id,q.to_ms,epoch_ms(p.ts) t,p.lat,p.lon,p.alt,p.gs,
          row_number() OVER (PARTITION BY q.mvt_id,p.ts ORDER BY p.lat,p.lon,p.alt,p.gs) duplicate_rank
        FROM query q JOIN read_parquet(?) p ON q.airport=p.airport AND q.hex=p.hex
        AND epoch_ms(p.ts)>=q.to_ms-7200000 AND epoch_ms(p.ts)<=q.to_ms+60000)
        SELECT mvt_id,to_ms,t,lat,lon,alt,gs FROM matched WHERE duplicate_rank=1 ORDER BY mvt_id,t''',
        [[str(ROOT/path) for path in manifest['positions']]]).df()
    checked_raw = checked_float = 0
    for q in sample.itertuples(index=False):
        observed = reconstructed.loc[reconstructed.mvt_id.eq(q.mvt_id)]
        assert len(observed) == q.raw_count
        if observed.empty:
            assert q.count == 0
            continue
        selected = [math.floor(k*(len(observed)-1)/(s.MAX_POINTS-1)) for k in range(s.MAX_POINTS)] if len(observed)>s.MAX_POINTS else list(range(len(observed)))
        all_rows = observed.to_dict('records')
        expected = []
        for k in selected:
            r = all_rows[k]
            expected.append([q.to_ms-r['t'], r['lat'], r['lon'], r['alt'], r['gs'],
                             r['t']-all_rows[k-1]['t'] if k else -1])
        actual = points[q.start:q.start+q.count]
        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.int32))
        checked_raw += actual.size
        one = sample.loc[sample.mvt_id.eq(q.mvt_id)]
        channels, mask, quality = s.encoded_batch(one, points)
        assert mask.sum() == q.count and not channels[0, q.count:].any()
        for j, row in enumerate(expected):
            dt, latraw, lonraw, alt, speed, gap = row
            lat, lon = latraw/1e6, lonraw/1e6
            stand = np.isfinite(q.stand_lat) and np.isfinite(q.stand_lon)
            runway = np.isfinite(q.runway_lat) and np.isfinite(q.runway_lon)
            clock = pd.notna(q.clock_ms)
            distance = lambda a,b: math.hypot((a-lat)*111320,
                (b-lon)*111320*math.cos(math.radians((lat+a)/2))) / 40000
            manual = [dt/7200000, math.log1p(max(gap, 0)/1000)/math.log1p(7260), gap>=0,
                max(-2, min(2, (lon-q.origin_lon)*111320*math.cos(math.radians(q.origin_lat))/20000)),
                max(-2, min(2, (lat-q.origin_lat)*111320/20000)),
                min(speed/3000, 1) if speed>=0 else 0, speed>=0,
                min(alt*25/10000, 1) if alt>=0 else 0, alt>=0 or alt==-123, alt==-123,
                min(distance(q.stand_lat,q.stand_lon), 1) if stand else 0, stand,
                min(distance(q.runway_lat,q.runway_lon), 1) if runway else 0, runway,
                max(-8, min(8, math.asinh((q.to_ms-dt-q.clock_ms)/1800000)))/8 if clock else 0, clock]
            np.testing.assert_allclose(channels[0,j], manual, rtol=2e-6, atol=2e-7)
            checked_float += len(manual)
    assert points[:,0].min() >= -60000 and points[:,0].max() <= 7200000
    assert np.all((points[:,5] == -1) | ((points[:,5] > 0) & (points[:,5] <= 7260000)))
    result = dict(status='passed', sampled_rows=len(sample), checked_raw_values=checked_raw,
        checked_float_values=checked_float, all_rows=len(index), all_stored_points=len(points),
        checked_geometry_metadata_values=6*len(index),
        sampled_ids=sample.mvt_id.tolist(), source_sha256=sha256(Path(__file__)),
        pilot_manifest_sha256=sha256(s.OUT/'pilot.json'),
        scope='independent SQL time/identity filtering + scalar raw unit/mask/geometry reconstruction')
    write_json(ROOT/'tmp/research_20261010/model_diversity/sparse_track_pilot_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
