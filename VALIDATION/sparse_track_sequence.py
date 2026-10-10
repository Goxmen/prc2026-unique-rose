"""Input-only sparse ADS-B sequences; isolated four-date research pilot.

No labels, block times, fitting, interpolation or identity inference occurs here.
Audited unique near-takeoff identities are reused. The cache is ragged int32 raw
observations plus static/query metadata; numeric model transforms are explicit.
"""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import aircraft_rotation as rotation
import stand_features as sf
import stand_geometry as sg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/derived/sparse_track_sequence_20261010'
IDENTITY = rotation.DEFAULT / 'identity.parquet'
HEATMAP = rotation.HEATMAP
PILOT_DAYS = ['2025-01-10', '2025-03-10', '2025-07-10', '2025-09-10']
MAX_POINTS = 128
RAW_NAMES = ['takeoff_minus_position_ms', 'lat_microdeg', 'lon_microdeg',
             'alt_code', 'gs_deciknot', 'preceding_raw_gap_ms']
CHANNEL_NAMES = ['time_before_takeoff', 'preceding_gap', 'preceding_gap_known',
                 'east', 'north', 'speed', 'speed_known', 'altitude',
                 'altitude_known', 'ground', 'stand_distance', 'stand_known',
                 'runway_distance', 'runway_known', 'actual_clock_offset', 'actual_clock_known']
QUALITY_NAMES = ['log_raw_count', 'retained_slots_fraction', 'span', 'ground_fraction',
                 'actual_clock_known', 'stand_known', 'runway_known',
                 'left_source_missing', 'right_source_missing']
SETTINGS = dict(window_before_sec=7200, window_after_sec=60, max_points=MAX_POINTS,
    minimum_distinct_timestamps=2, days=PILOT_DAYS, labels_used=False,
    identity='audited unique near-takeoff identity; missing and ambiguous remain absent',
    sort=['airport', 'hex', 'ts_ms', 'lat', 'lon', 'alt', 'gs'],
    duplicate_timestamp='retain lexicographically first lat/lon/alt/gs per airport/hex/time',
    selection='floor(linspace(0,n-1,128)); first and last always retained',
    gap='previous raw observation within query window; first=-1; never interpolated',
    ground_alt_code=-123, other_negative_alt='unknown', negative_speed='unknown',
    origin=sf.SETTINGS['origin'], runway_point=sf.SETTINGS['runway_point'],
    point_channels=CHANNEL_NAMES, quality_channels=QUALITY_NAMES,
    bounds=dict(time=[-1/120, 1], gap=[0, 1], east_north=[-2, 2], speed=[0, 1],
                altitude=[0, 1], distance=[0, 1], clock=[-1, 1]),
    scales=dict(time_sec=7200, gap_log_sec=7260, east_north_m=20000,
                speed_kt=300, altitude_ft=10000, distance_m=40000,
                clock='clip(asinh((point-originalAOBT)/1800),-8,8)/8', raw_count_cap=4096),
    padding='right padding, independent boolean point mask, all padded channels zero',
    absent_sequence='0 or 1 points cached honestly; model residual hard-gated off')


def inputs(con, days=PILOT_DAYS):
    """Allowlisted departure inputs; date filter precedes identity join."""
    q = con.execute('''SELECT MVT_ID_mvt::BIGINT mvt_id, ADEP_mvt airport,
        STAND_mvt stand, RUNWAY_mvt runway, epoch_ms(MVT_TIME_UTC_mvt)::BIGINT to_ms,
        epoch_ms(AOBT_3_flt)::BIGINT clock_ms
        FROM read_parquet(?) WHERE PHASE_mvt='DEP'
          AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN (SELECT unnest(?))''',
        [str(ROOT / 'data/training_2025-*.parquet'), list(days)]).df()
    identities = pd.read_parquet(IDENTITY, columns=['mvt_id', 'hex'])
    q = q.merge(identities, on='mvt_id', how='left', validate='one_to_one')
    q['day'] = pd.to_datetime(q.to_ms, unit='ms', utc=True).dt.strftime('%Y-%m-%d')
    return q.sort_values('mvt_id').reset_index(drop=True)


def add_geometry(query, tables=None):
    coords, ends, origin = tables or sf.load()
    result = query.copy()
    stand = coords.assign(key=coords.airport + ':' + coords.stand).set_index('key')
    runway = ends.assign(key=ends.airport + ':' + ends.runway).set_index('key')
    for prefix, field, table in [('stand', 'stand', stand), ('runway', 'runway', runway)]:
        keys = result.airport.astype('string') + ':' + result[field].astype('string')
        for col in ['lat', 'lon']:
            result[f'{prefix}_{col}'] = keys.map(table[col]).to_numpy(float, na_value=np.nan)
    for col in ['lat', 'lon']:
        result[f'origin_{col}'] = result.airport.map(origin[col]).to_numpy(float, na_value=np.nan)
    if result[['origin_lat', 'origin_lon']].isna().any().any():
        raise ValueError('Airport has no static runway-centre origin')
    return result


def canonical_positions(positions):
    if positions.empty:
        return positions
    p = positions.copy()
    if 'ts_ms' not in p:
        # .astype(int64) is resolution-dependent in pandas 3; use explicit ms.
        p['ts_ms'] = pd.to_datetime(p.ts, utc=True).astype('datetime64[ms, UTC]').astype('int64')
    cols = ['airport', 'hex', 'ts_ms', 'lat', 'lon', 'alt', 'gs']
    if p[cols].isna().any().any():
        raise ValueError('Null raw position values require an explicit source representation')
    if not (p.lat.between(-90000000, 90000000).all() and p.lon.between(-180000000, 180000000).all()):
        raise ValueError('Raw latitude/longitude outside geographic bounds')
    return p.sort_values(cols, kind='stable').drop_duplicates(['airport', 'hex', 'ts_ms'])


def select_sequences(query, positions, available_days):
    """Range-search each unique aircraft; bounded per-flight arrays, no large join."""
    p = canonical_positions(positions)
    groups = {}
    if not p.empty:
        for key, frame in p.groupby(['airport', 'hex'], observed=True, sort=False):
            groups[(str(key[0]), int(key[1]))] = frame[['ts_ms', 'lat', 'lon', 'alt', 'gs']].to_numpy(np.int64)
    rows, parts, cursor = [], [], 0
    for q in query.itertuples(index=False):
        start_ms, end_ms = int(q.to_ms)-7200000, int(q.to_ms)+60000
        left_day = pd.Timestamp(start_ms, unit='ms', tz='UTC').strftime('%Y-%m-%d')
        right_day = pd.Timestamp(end_ms, unit='ms', tz='UTC').strftime('%Y-%m-%d')
        seq = np.empty((0, 6), dtype=np.int32)
        total = ground_count = 0
        span_ms = 0
        if pd.notna(q.hex):
            group = groups.get((str(q.airport), int(q.hex)))
            if group is not None:
                lo = np.searchsorted(group[:, 0], start_ms, side='left')
                hi = np.searchsorted(group[:, 0], end_ms, side='right')
                raw = group[lo:hi]
                total = len(raw)
                if total:
                    choose = (np.arange(total) if total <= MAX_POINTS else
                              np.linspace(0, total-1, MAX_POINTS).astype(np.int64))
                    gaps = np.r_[-1, np.diff(raw[:, 0])]
                    selected = raw[choose]
                    seq = np.column_stack([q.to_ms-selected[:, 0], selected[:, 1:], gaps[choose]]).astype(np.int32)
                    span_ms = int(raw[-1, 0]-raw[0, 0])
                    ground_count = int((raw[:, 3] == -123).sum())
        row = q._asdict()
        row.update(start=cursor, count=len(seq), raw_count=total, raw_ground_count=ground_count,
                   span_ms=span_ms, left_source_missing=int(left_day not in available_days),
                   right_source_missing=int(right_day not in available_days))
        rows.append(row)
        parts.append(seq)
        cursor += len(seq)
    points = np.concatenate(parts) if parts else np.empty((0, 6), dtype=np.int32)
    return pd.DataFrame(rows), points


def encoded_batch(index, points):
    """Bounded float32 channels, boolean point mask, and input-only quality."""
    n = len(index)
    channels = np.zeros((n, MAX_POINTS, len(CHANNEL_NAMES)), dtype=np.float32)
    mask = np.zeros((n, MAX_POINTS), dtype=bool)
    quality = np.zeros((n, len(QUALITY_NAMES)), dtype=np.float32)
    for i, q in enumerate(index.itertuples(index=False)):
        raw = points[int(q.start):int(q.start)+int(q.count)].astype(np.float64)
        count = len(raw)
        clock_known = pd.notna(q.clock_ms)
        stand_known = np.isfinite(q.stand_lat) and np.isfinite(q.stand_lon)
        runway_known = np.isfinite(q.runway_lat) and np.isfinite(q.runway_lon)
        quality[i] = [np.log1p(min(q.raw_count, 4096))/np.log1p(4096), count/MAX_POINTS,
                      q.span_ms/7260000, q.raw_ground_count/max(1, q.raw_count),
                      clock_known, stand_known, runway_known, q.left_source_missing, q.right_source_missing]
        if not count:
            continue
        lat, lon = raw[:, 1]/1e6, raw[:, 2]/1e6
        speed_known = raw[:, 4] >= 0
        alt_known = (raw[:, 3] >= 0) | (raw[:, 3] == -123)
        ground = raw[:, 3] == -123
        values = [raw[:, 0]/7200000,
            np.log1p(np.maximum(raw[:, 5], 0)/1000)/np.log1p(7260), raw[:, 5] >= 0,
            np.clip((lon-q.origin_lon)*sg.M_PER_DEG_LAT*np.cos(np.radians(q.origin_lat))/20000, -2, 2),
            np.clip((lat-q.origin_lat)*sg.M_PER_DEG_LAT/20000, -2, 2),
            np.where(speed_known, np.clip(raw[:, 4]/3000, 0, 1), 0), speed_known,
            np.where(alt_known & ~ground, np.clip(raw[:, 3]*25/10000, 0, 1), 0), alt_known, ground,
            np.clip(sg.metres(lat, lon, q.stand_lat, q.stand_lon)/40000, 0, 1) if stand_known else np.zeros(count),
            np.full(count, stand_known),
            np.clip(sg.metres(lat, lon, q.runway_lat, q.runway_lon)/40000, 0, 1) if runway_known else np.zeros(count),
            np.full(count, runway_known),
            np.clip(np.arcsinh((q.to_ms-raw[:, 0]-q.clock_ms)/1800000), -8, 8)/8 if clock_known else np.zeros(count),
            np.full(count, clock_known)]
        channels[i, :count] = np.column_stack(values)
        mask[i, :count] = True
    if not np.isfinite(channels).all() or not np.isfinite(quality).all():
        raise ValueError('Sequence transform produced nonfinite values')
    return channels, mask, quality


def build_pilot(out=OUT):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ['pilot_index.parquet', 'pilot_points.npy', 'pilot.json']):
        raise FileExistsError('Pilot exists: retain its source identity and do not overwrite')
    identity_manifest = json.loads(IDENTITY.with_suffix('.json').read_text())
    if rotation.sha256(IDENTITY) != identity_manifest['identity_sha256']:
        raise ValueError('Audited identity table changed')
    movement_hashes = {str(p.relative_to(ROOT)): rotation.sha256(p)
                       for p in sorted(ROOT.glob('data/training_2025-*.parquet'))}
    identity_movements = {str(Path(v['path']).relative_to(ROOT)): v['sha256']
                          for v in identity_manifest['movement_sources']}
    if movement_hashes != identity_movements:
        raise ValueError('Movement sources differ from those used by the audited identity cache')
    con = rotation.connection()
    query = add_geometry(inputs(con))
    indices, arrays, paths, cursor = [], [], set(), 0
    for day in PILOT_DAYS:
        stamp = date.fromisoformat(day)
        candidates = [HEATMAP / 'positions' / f'{stamp+timedelta(days=d)}.parquet' for d in [-1, 0, 1]]
        present = [p for p in candidates if p.exists()]
        paths.update(present)
        pos = con.execute('''SELECT airport,hex,epoch_ms(ts)::BIGINT ts_ms,lat,lon,alt,gs
            FROM read_parquet(?) WHERE ts>=?::TIMESTAMPTZ-INTERVAL 2 HOUR
            AND ts<?::TIMESTAMPTZ+INTERVAL 1 DAY+INTERVAL 60 SECOND''',
            [[str(p) for p in present], day, day]).df() if present else pd.DataFrame()
        ix, values = select_sequences(query.loc[query.day.eq(day)], pos, {p.stem for p in present})
        ix['start'] += cursor
        cursor += len(values)
        indices.append(ix)
        arrays.append(values)
    index = pd.concat(indices, ignore_index=True)
    points = np.concatenate(arrays)
    if index.mvt_id.duplicated().any() or points.dtype != np.int32:
        raise ValueError('Pilot identity/dtype invariant failed')
    # Full pilot transform is checked in bounded batches, not retained as a dense cache.
    for start in range(0, len(index), 1024):
        encoded_batch(index.iloc[start:start+1024], points)
    ip, pp = out/'pilot_index.parquet', out/'pilot_points.npy'
    index.to_parquet(ip, index=False)
    np.save(pp, points, allow_pickle=False)
    byday = {}
    for day, g in index.groupby('day'):
        byday[day] = dict(rows=len(g), identified=int(g.hex.notna().sum()),
            any_points=int(g['count'].gt(0).sum()), usable_2plus=int(g['count'].ge(2).sum()),
            sparse_2_to_31=int(g['raw_count'].between(2, 31).sum()),
            dense_32plus=int(g['raw_count'].ge(32).sum()),
            truncated_128=int(g.raw_count.gt(128).sum()),
            missing_original_clock_usable=int((g['count'].ge(2) & g.clock_ms.isna()).sum()),
            raw_points=int(g.raw_count.sum()), kept_points=int(g['count'].sum()))
    manifest = dict(settings=SETTINGS, rows=len(index), stored_points=len(points), by_day=byday,
        raw_array_bytes=points.nbytes, elapsed_sec=time.monotonic()-started,
        source_sha256=rotation.sha256(Path(__file__)), identity_sha256=rotation.sha256(IDENTITY),
        identity_manifest_sha256=rotation.sha256(IDENTITY.with_suffix('.json')),
        stand_map_sha256=rotation.sha256(sf.COORDS), runway_map_sha256=rotation.sha256(sf.ENDS),
        movements=movement_hashes,
        positions={str(p.relative_to(ROOT)): rotation.sha256(p) for p in sorted(paths)},
        outputs={ip.name: rotation.sha256(ip), pp.name: rotation.sha256(pp)})
    rotation.write_json(out/'pilot.json', manifest)
    con.close()
    print(json.dumps({k: manifest[k] for k in ['rows', 'stored_points', 'raw_array_bytes', 'elapsed_sec', 'by_day']}, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', required=True)
    args = parser.parse_args()
    build_pilot()
