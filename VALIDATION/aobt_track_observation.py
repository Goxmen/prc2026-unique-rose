"""Input-only ADS-B observations centered on the original reported AOBT.

The existing track family summarizes a departure's whole ground segment. These
six fixed inputs instead describe what the identified aircraft was doing at the
reported actual off-block time. Neither departure block nor target is selected.
Identity is the independently audited, near-takeoff unique identity cache. A
reported clock is not assumed to be the physical off-block time.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

import aircraft_rotation as rotation
import stand_features as sf
import stand_geometry as sg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/derived/aobt_track_observation_20261010'
IDENTITY = ROOT / 'data/derived/aircraft_rotation_20261010/identity.parquet'
HEATMAP = ROOT / 'data/external/adsblol_heatmap'
FEATURES = ['ao_point_offset_sec', 'ao_speed_kt', 'ao_stand_distance_m',
            'ao_on_ground', 'ao_moving_before', 'ao_moving_after']
SETTINGS = dict(features=FEATURES, source_clock='original AOBT_3_flt only',
    query_clock_age_sec=[-180, 14400], nearest_tolerance_sec=120,
    fraction_window_sec=300, moving_threshold_kt=5,
    position_cutoff_sec_after_takeoff=60, tie_break='earlier position',
    fractions='valid-speed ground points; before excludes clock, after includes clock',
    labels_used=False)


def inputs(con):
    """Explicit allowlist: do not load departure target or recorded block."""
    q = con.execute('''SELECT MVT_ID_mvt::BIGINT mvt_id, ADEP_mvt airport,
        STAND_mvt stand, epoch_ms(MVT_TIME_UTC_mvt)::BIGINT to_ms,
        epoch_ms(AOBT_3_flt)::BIGINT clock_ms
        FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',
        [str(ROOT / 'data/training_2025-*.parquet')]).df()
    ids = pd.read_parquet(IDENTITY, columns=['mvt_id', 'hex'])
    q = q.merge(ids, on='mvt_id', how='left', validate='one_to_one')
    q['day'] = pd.to_datetime(q.clock_ms, unit='ms', utc=True).dt.strftime('%Y-%m-%d')
    return q


def eligible(query):
    age = (query.to_ms - query.clock_ms) / 1000
    return query.hex.notna() & query.clock_ms.notna() & age.between(-180, 14400)


def observe(con, query, positions, coords):
    """Pure day calculation; positions may include adjacent calendar days."""
    result = query[['mvt_id']].copy()
    for col in FEATURES:
        result[col] = np.nan
    active = query.loc[eligible(query)].copy()
    if active.empty or positions.empty:
        return result
    con.register('ao_queries', active)
    con.register('ao_positions', positions)
    points = con.execute('''SELECT q.mvt_id,q.clock_ms,p.ts,p.lat,p.lon,p.alt,p.gs,
        epoch_ms(p.ts)-q.clock_ms delta_ms,
        row_number() OVER (PARTITION BY q.mvt_id
          ORDER BY abs(epoch_ms(p.ts)-q.clock_ms),p.ts,p.lat,p.lon) near_rank
        FROM ao_queries q JOIN ao_positions p
          ON q.hex=p.hex AND q.airport=p.airport
        WHERE epoch_ms(p.ts) BETWEEN q.clock_ms-300000 AND q.clock_ms+300000
          AND epoch_ms(p.ts)<=q.to_ms+60000''').df()
    if points.empty:
        return result
    near = points.loc[points.near_rank.eq(1) & points.delta_ms.abs().le(120000)].set_index('mvt_id')
    table = result.set_index('mvt_id')
    table.loc[near.index, 'ao_point_offset_sec'] = near.delta_ms.to_numpy(float) / 1000
    table.loc[near.index, 'ao_speed_kt'] = np.where(near.gs >= 0, near.gs / 10, np.nan)
    table.loc[near.index, 'ao_on_ground'] = np.where(near.alt.notna(), near.alt.eq(-123), np.nan)
    stand_positions = coords.assign(key=coords.airport + ':' + coords.stand).set_index('key')
    qnear = active.set_index('mvt_id').reindex(near.index)
    keys = qnear.airport.astype('string') + ':' + qnear.stand.astype('string')
    lat = keys.map(stand_positions.lat).to_numpy(float)
    lon = keys.map(stand_positions.lon).to_numpy(float)
    distance = sg.metres(near.lat.to_numpy(float) / 1e6,
                         near.lon.to_numpy(float) / 1e6, lat, lon)
    table.loc[near.index, 'ao_stand_distance_m'] = distance
    valid = points.alt.eq(-123) & points.gs.ge(0)
    for col, mask in [('ao_moving_before', points.delta_ms.lt(0)),
                      ('ao_moving_after', points.delta_ms.ge(0))]:
        part = points.loc[valid & mask]
        fraction = part.gs.ge(50).groupby(part.mvt_id).mean()
        table.loc[fraction.index, col] = fraction.to_numpy(float)
    return table.reset_index()


def build(days=None):
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ('features.parquet' if days is None else 'pilot.parquet')
    if path.exists():
        raise ValueError(f'Preserve the existing result; inspect its manifest: {path}')
    con = rotation.connection()
    query = inputs(con)
    coords = sf.load()[0]
    selected = sorted(query.loc[eligible(query), 'day'].dropna().unique()) if days is None else days
    chunks, source_paths = [], set()
    for day in selected:
        q = query.loc[query.day.eq(day)]
        stamp = date.fromisoformat(day)
        paths = [HEATMAP / 'positions' / f'{stamp + timedelta(days=delta)}.parquet'
                 for delta in [-1, 0, 1]]
        paths = [p for p in paths if p.exists()]
        source_paths.update(paths)
        if paths:
            pos = con.execute('''SELECT * FROM read_parquet(?)
                WHERE ts>=?::TIMESTAMPTZ-INTERVAL 5 MINUTE
                  AND ts<?::TIMESTAMPTZ+INTERVAL 1 DAY+INTERVAL 5 MINUTE''',
                [[str(p) for p in paths], day, day]).df()
        else:
            pos = pd.DataFrame()
        chunks.append(observe(con, q, pos, coords))
    all_ids = query[['mvt_id']] if days is None else query.loc[query.day.isin(days), ['mvt_id']]
    values = pd.concat(chunks, ignore_index=True)
    result = all_ids.merge(values, on='mvt_id', how='left', validate='one_to_one')
    if result.mvt_id.duplicated().any():
        raise ValueError('Feature movement IDs are not unique')
    result = result.sort_values('mvt_id').reset_index(drop=True)
    for c in FEATURES:
        result[c] = result[c].astype('float32')
    result.to_parquet(path, index=False)
    manifest = dict(settings=SETTINGS, source_sha256=rotation.sha256(Path(__file__)),
        identity_sha256=rotation.sha256(IDENTITY),
        stand_map_sha256=rotation.sha256(sf.COORDS),
        inputs={str(p.relative_to(ROOT)): rotation.sha256(p)
                for p in sorted(ROOT.glob('data/training_2025-*.parquet'))},
        positions={str(p.relative_to(ROOT)): rotation.sha256(p) for p in sorted(source_paths)},
        selected_days=selected, rows=len(result),
        known={c: int(result[c].notna().sum()) for c in FEATURES},
        feature_sha256=rotation.sha256(path), elapsed_sec=time.monotonic()-started)
    rotation.write_json(path.with_suffix('.json'), manifest)
    con.close()
    print({k: manifest[k] for k in ['rows', 'known', 'elapsed_sec']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--days', nargs='+')
    args = p.parse_args()
    build(args.days)
