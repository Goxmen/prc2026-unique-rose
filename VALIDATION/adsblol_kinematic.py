"""Recover missing ADSB ground witnesses from takeoff time and runway geometry.

The ordinary exact-callsign matches always take precedence. A new match must be
unique in both directions, align with the supplied runway, and fall within 30 s
of an airport offset calibrated on known 2025 aircraft identities. No departure
label or block time is read. Existing local ADSB.lol and stand-map caches suffice.

Calibration dates belong to both Jan/Jul training sets. ``audit`` hides a fixed
10% of known identities on disjoint dates before matching or reserving events.
``build`` requires explicit days or --all-days; it never alters existing caches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import adsblol as ab
import stand_features as sf
import stand_geometry as sg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'tmp/research_20261010/signal_scout/kinematic'
CALIBRATION_DAYS = ['2025-03-10', '2025-09-10']
AUDIT_DAYS = ['2025-01-12', '2025-07-12', '2026-01-12', '2026-02-12',
              '2026-06-12', '2026-07-12']
PARAMETERS = dict(event_gap_ms=90000, event_speed=700, identity_tolerance_ms=90000,
                  match_tolerance_ms=30000, max_heading_deg=25, max_cross_m=200,
                  minimum_calibration=20, aircraft_conflict_ms=120000,
                  min_along_m=-300, max_past_runway_m=2000)
RAW_COLUMNS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt',
               'MVT_TIME_UTC_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'FLIGHT_ID_mvt']
SUMMARY_COLUMNS = ab.OWN_FEATURES + ab.STAND_FEATURES[:6]
CONFIDENCE_COLUMNS = ['ak_recovered', 'ak_timing_residual_sec', 'ak_heading_error_deg',
                      'ak_runway_cross_m']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def departures(ranking):
    frames = [pd.read_parquet(p, columns=RAW_COLUMNS)
              for p in [*sorted((ROOT / 'data').glob('training_2025-*.parquet')), ranking]]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt')
    out = pd.DataFrame({'mvt_id': raw.MVT_ID_mvt.to_numpy(),
                        'airport': raw.ADEP_mvt.astype(str).to_numpy(),
                        'to_ms': ab.epoch_ms(raw.MVT_TIME_UTC_mvt).to_numpy(),
                        'no_id': raw.FLIGHT_ID_mvt.isna().to_numpy(),
                        'runway': raw.RUNWAY_mvt.fillna('').astype(str).to_numpy()})
    out['cs'] = (raw.CALLSIGN_flt.fillna(raw.FLIGHT_mvt).fillna('').astype(str)
                 .str.replace(' ', '', regex=False).str.upper().to_numpy())
    out['day'] = pd.to_datetime(out.to_ms, unit='ms').dt.strftime('%Y-%m-%d')
    out['stand_key'] = out.airport + ':' + raw.STAND_mvt.fillna('').astype(str).to_numpy()
    out['runway_key'] = out.airport + ':' + out.runway
    return out


def takeoff_events(pos):
    p = pos.sort_values(['hex', 'airport', 'ts'], kind='stable').reset_index(drop=True)
    same = p.hex.eq(p.hex.shift()) & p.airport.eq(p.airport.shift())
    delta = p.ts - p.ts.shift()
    keep = (same & p.alt.ne(ab.GROUND) & p.alt.shift().eq(ab.GROUND)
            & delta.between(1, PARAMETERS['event_gap_ms'])
            & p.gs.ge(PARAMETERS['event_speed']))
    out = p.loc[keep, ['hex', 'airport', 'lat', 'lon']].copy()
    out['event_ms'] = (p.ts[keep] + p.ts.shift()[keep]) / 2
    out['heading'] = np.degrees(np.arctan2(
        (out.lon - p.lon.shift()[keep]) * np.cos(np.radians(out.lat / 1e6)),
        out.lat - p.lat.shift()[keep])) % 360
    out = out.reset_index(drop=True)
    out['event_id'] = np.arange(len(out))
    return out


def known_events(reference, events):
    pairs = reference[['mvt_id', 'airport', 'hex', 'to_ms']].merge(events, on=['airport', 'hex'])
    pairs['timing_sec'] = (pairs.event_ms - pairs.to_ms) / 1000
    return pairs[pairs.timing_sec.abs() <= PARAMETERS['identity_tolerance_ms'] / 1000]


def calibrate(deps, days):
    observations = []
    for day in days:
        d = deps[deps.day.eq(day)]
        pos, calls = ab.load_day(day)
        if pos is None or d.empty:
            continue
        _, _, _, hit = ab.match(d[['mvt_id', 'airport', 'to_ms', 'cs']], pos, calls)
        pairs = known_events(hit, takeoff_events(pos))
        pairs = pairs.loc[pairs.groupby('mvt_id').timing_sec.apply(lambda x: x.abs().idxmin())]
        observations.append(pairs[['airport', 'timing_sec']].assign(day=day))
    if not observations:
        raise ValueError('No calibration observations')
    obs = pd.concat(observations, ignore_index=True)
    airports = {}
    for airport, group in obs.groupby('airport'):
        if len(group) >= PARAMETERS['minimum_calibration']:
            median = float(group.timing_sec.median())
            airports[airport] = dict(offset_sec=median, n=len(group),
                                     median_abs_deviation_sec=float((group.timing_sec - median).abs().median()))
    return dict(days=days, parameters=PARAMETERS, airports=airports,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def runway_geometry(airport, runway, ends):
    try:
        number = int(runway[:2])
        suffix = runway[2:]
        reciprocal = f'{(number + 17) % 36 + 1:02d}' + {'L': 'R', 'R': 'L'}.get(suffix, suffix)
        start, end = ends.loc[(airport, runway)], ends.loc[(airport, reciprocal)]
    except (KeyError, ValueError):
        return None
    lat_scale = sg.M_PER_DEG_LAT
    lon_scale = lat_scale * np.cos(np.radians(start.lat))
    dx, dy = (end.lon - start.lon) * lon_scale, (end.lat - start.lat) * lat_scale
    return start, lat_scale, lon_scale, dx, dy, np.degrees(np.arctan2(dx, dy)) % 360


def match_missing(deps, pos, calls, calibration, tables, hide_known=False):
    """Return new identities and the original matches, preserving original precedence."""
    if calibration['parameters'] != PARAMETERS:
        raise ValueError('Calibration settings differ from the frozen matcher')
    _, ground, segments, original = ab.match(deps[['mvt_id', 'airport', 'to_ms', 'cs']], pos, calls)
    events = takeoff_events(pos)
    hidden_ids = set(original.loc[original.mvt_id.astype('int64') % 10 == 0, 'mvt_id']) if hide_known else set()
    reference = original[~original.mvt_id.isin(hidden_ids)]
    reserved = set(known_events(reference, events).event_id)
    events = events[~events.event_id.isin(reserved)]
    query = deps[~deps.mvt_id.isin(reference.mvt_id)]
    ends = tables[1].set_index(['airport', 'runway'])
    candidates = []
    for airport, group in query.groupby('airport'):
        if airport not in calibration['airports']:
            continue
        offset = calibration['airports'][airport]['offset_sec']
        airport_events = events[events.airport.eq(airport)].sort_values('event_ms')
        times = airport_events.event_ms.to_numpy()
        for row in group.itertuples():
            geometry = runway_geometry(airport, row.runway, ends)
            if geometry is None:
                continue
            center = row.to_ms + offset * 1000
            lo, hi = np.searchsorted(times, [center - PARAMETERS['match_tolerance_ms'],
                                             center + PARAMETERS['match_tolerance_ms']])
            near = airport_events.iloc[lo:hi].copy()
            if near.empty:
                continue
            start, lat_scale, lon_scale, dx, dy, heading = geometry
            near['heading_error_deg'] = abs((near.heading - heading + 180) % 360 - 180)
            x, y = (near.lon / 1e6 - start.lon) * lon_scale, (near.lat / 1e6 - start.lat) * lat_scale
            near['runway_length_m'] = np.hypot(dx, dy)
            near['cross_m'] = abs(x * dy - y * dx) / near.runway_length_m
            near['along_m'] = (x * dx + y * dy) / near.runway_length_m
            near = near[(near.heading_error_deg <= PARAMETERS['max_heading_deg'])
                        & (near.cross_m <= PARAMETERS['max_cross_m'])
                        & (near.along_m >= PARAMETERS['min_along_m'])
                        & (near.along_m <= near.runway_length_m + PARAMETERS['max_past_runway_m'])]
            if near.empty:
                continue
            near['mvt_id'], near['to_ms'] = row.mvt_id, row.to_ms
            near['runway'] = row.runway
            near['timing_residual_sec'] = (near.event_ms - center) / 1000
            near['cost'] = near.timing_residual_sec.abs() + 0.2 * near.heading_error_deg
            candidates.append(near)
    columns = ['mvt_id', 'hex', 'airport', 'to_ms', 'event_id', 'timing_residual_sec',
               'heading_error_deg', 'cross_m', 'along_m', 'runway_length_m',
               'runway', 'event_ms', 'lat', 'lon']
    if not candidates:
        return pd.DataFrame(columns=columns), original, ground, segments, hidden_ids
    candidates = (pd.concat(candidates, ignore_index=True).sort_values('cost')
                   .drop_duplicates(['mvt_id', 'hex']).reset_index(drop=True))
    candidates['n_aircraft'] = candidates.groupby('mvt_id').mvt_id.transform('size')
    candidates['n_flights'] = candidates.groupby('event_id').event_id.transform('size')
    accepted = candidates[candidates.n_aircraft.eq(1) & candidates.n_flights.eq(1)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=columns), original, ground, segments, hidden_ids
    # A second nearby event from an already assigned aircraft must not get a new row.
    near_reference = accepted[['mvt_id', 'hex', 'airport', 'to_ms']].merge(
        reference[['hex', 'airport', 'to_ms']], on=['hex', 'airport'], suffixes=('', '_known'))
    near_reference = near_reference[(near_reference.to_ms - near_reference.to_ms_known).abs()
                                     <= PARAMETERS['aircraft_conflict_ms']]
    accepted = accepted[~accepted.mvt_id.isin(near_reference.mvt_id)]
    if accepted.empty:
        return pd.DataFrame(columns=columns), original, ground, segments, hidden_ids
    # Multiple ground/air flag changes cannot allocate one aircraft to nearby departures.
    accepted = accepted.sort_values(['hex', 'airport', 'to_ms'])
    conflict = ((accepted.hex.eq(accepted.hex.shift()) & accepted.airport.eq(accepted.airport.shift()))
                & accepted.to_ms.diff().abs().le(PARAMETERS['aircraft_conflict_ms']))
    reject = set(accepted.index[conflict]) | set(accepted.index[np.r_[conflict.to_numpy()[1:], False]])
    accepted = accepted.loc[~accepted.index.isin(reject), columns]
    if accepted.mvt_id.isin(reference.mvt_id).any() or accepted.event_id.isin(reserved).any():
        raise AssertionError('A kinematic match overwrote an original identity')
    return accepted.reset_index(drop=True), original, ground, segments, hidden_ids


def summarize(deps, matches, ground, segments, tables):
    """The existing own-track and stand summaries, on recovered identities only."""
    out = deps[deps.mvt_id.isin(matches.mvt_id)].copy().reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=['mvt_id', *SUMMARY_COLUMNS, *CONFIDENCE_COLUMNS])
    cand = matches[['mvt_id', 'hex', 'airport', 'to_ms']].assign(t=lambda x: x.to_ms + ab.AFTER_MS).sort_values('t')
    hit = pd.merge_asof(cand, ground[['ts', 'hex', 'airport', 'seg']].sort_values('ts'),
                        left_on='t', right_on='ts', by=['hex', 'airport'], direction='backward',
                        tolerance=ab.MATCH_TOL_MS + ab.AFTER_MS).dropna(subset=['ts'])
    hit = hit.join(segments, on='seg').set_index('mvt_id').reindex(out.mvt_id)
    result = out[['mvt_id']].copy()
    takeoff = out.to_ms.to_numpy(float)
    result['ab_matched'] = 1.0
    for feature, source in [('ab_seg_sec', 'seg_start'), ('ab_move_sec', 'move'),
                             ('ab_still_end_sec', 'still_end'), ('ab_last_gnd_sec', 'ts')]:
        result[feature] = (takeoff - hit[source].to_numpy(float)) / 1000
    result['ab_seg_n'], result['ab_first_gs'] = hit.seg_n.to_numpy(), hit.first_gs.to_numpy()
    result['ab_prev_gap_sec'] = np.minimum(hit.prev_gap.to_numpy(float) / 1000, 86400)
    stands, ends, _ = tables
    stands = stands.assign(k=stands.airport + ':' + stands.stand).set_index('k')
    ends = ends.assign(k=ends.airport + ':' + ends.runway).set_index('k')
    out['slat'], out['slon'] = out.stand_key.map(stands.lat), out.stand_key.map(stands.lon)
    own = out.merge(matches[['mvt_id', 'hex']], on='mvt_id')
    points = own[['mvt_id', 'hex', 'airport', 'to_ms', 'slat', 'slon']].merge(
        ground[['hex', 'airport', 'ts', 'lat', 'lon']], on=['hex', 'airport'])
    points = points[(points.ts >= points.to_ms - ab.LOOKBACK_MS) & (points.ts <= points.to_ms + ab.AFTER_MS)]
    points['distance'] = sg.metres(points.lat.to_numpy() / 1e6, points.lon.to_numpy() / 1e6,
                                    points.slat.to_numpy(), points.slon.to_numpy())
    near = points[points.distance <= ab.STAND_R_M]
    aggregate = pd.DataFrame({'distance': points.groupby('mvt_id').distance.min(),
                              'count': near.groupby('mvt_id').size(),
                              'exit': near.groupby('mvt_id').ts.max()}).reindex(out.mvt_id)
    has_stand = out.slat.notna().to_numpy() & hit.hex.notna().to_numpy()
    result['ab_stand_min_m'] = np.where(has_stand, aggregate.distance.to_numpy(), np.nan)
    result['ab_stand_n'] = np.where(has_stand, aggregate['count'].fillna(0).to_numpy(), np.nan)
    result['ab_stand_exit_sec'] = (takeoff - aggregate.exit.to_numpy(float)) / 1000
    result['ab_exit_gap_sec'] = (hit.seg_start.to_numpy(float) - aggregate.exit.to_numpy(float)) / 1000
    result['ab_seg_stand_m'] = sg.metres(hit.lat0.to_numpy(float) / 1e6, hit.lon0.to_numpy(float) / 1e6,
                                         out.slat.to_numpy(), out.slon.to_numpy())
    result['ab_seg_rwy_m'] = sg.metres(hit.lat0.to_numpy(float) / 1e6, hit.lon0.to_numpy(float) / 1e6,
                                       out.runway_key.map(ends.lat).to_numpy(), out.runway_key.map(ends.lon).to_numpy())
    aligned = matches.set_index('mvt_id').reindex(out.mvt_id)
    result['ak_recovered'] = 1.0
    for feature, source in zip(CONFIDENCE_COLUMNS[1:], ['timing_residual_sec', 'heading_error_deg', 'cross_m']):
        result[feature] = aligned[source].to_numpy()
    return result[['mvt_id', *SUMMARY_COLUMNS, *CONFIDENCE_COLUMNS]]


def add_features(X, ids, table):
    """Fill missing own-track inputs; rows with existing ground witnesses are immutable."""
    aligned = table.set_index('mvt_id').reindex(np.asarray(ids))
    missing = X.ab_seg_sec.isna().to_numpy() & aligned.ab_seg_sec.notna().to_numpy()
    result = X.copy()
    for column in SUMMARY_COLUMNS:
        if column in result:
            result.loc[missing, column] = aligned[column].to_numpy()[missing]
    for column in CONFIDENCE_COLUMNS:
        result[column] = np.where(missing, aligned[column].to_numpy(),
                                   0.0 if column == 'ak_recovered' else np.nan)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=['calibrate', 'audit', 'build'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--ranking', type=Path, default=ROOT / 'data/final_ranking.parquet')
    parser.add_argument('--days', nargs='+')
    parser.add_argument('--all-days', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    deps, tables = departures(args.ranking), sf.load()
    path = args.out / 'calibration.json'
    if args.stage == 'calibrate':
        calibration = calibrate(deps, args.days or CALIBRATION_DAYS)
        write_json(path, calibration)
        print(json.dumps(calibration, indent=2))
        return
    calibration = json.loads(path.read_text())
    days = args.days or (AUDIT_DAYS if args.stage == 'audit' else None)
    if args.all_days:
        if days is not None:
            raise ValueError('Choose explicit days or --all-days')
        days = sorted(deps.day.unique())
    if days is None:
        raise ValueError('A build requires --days or --all-days')
    if args.stage == 'audit' and set(days) & set(calibration['days']):
        raise ValueError('Audit and calibration dates must be disjoint')
    reports, frames, false_matches = [], [], []
    for day in days:
        d = deps[deps.day.eq(day)]
        pos, calls = ab.load_day(day)
        if pos is None or d.empty:
            reports.append(dict(day=day, skipped=True))
            continue
        matches, original, ground, segments, hidden = match_missing(
            d, pos, calls, calibration, tables, hide_known=args.stage == 'audit')
        compared = matches.merge(original[['mvt_id', 'hex']].rename(columns={'hex': 'known_hex'}), on='mvt_id', how='left')
        compared['hidden'] = compared.mvt_id.isin(hidden)
        compared['correct'] = compared.hex.eq(compared.known_hex)
        wrong = compared[compared.hidden & ~compared.correct].assign(day=day)
        false_matches.append(wrong)
        new = matches[~matches.mvt_id.isin(original.mvt_id)]
        reports.append(dict(day=day, rows=len(d), original_matches=len(original),
                            hidden_known=len(hidden), hidden_recovered=int(compared.hidden.sum()),
                            hidden_correct=int((compared.hidden & compared.correct).sum()),
                            new_matches=len(new), new_no_id=int(d.loc[d.mvt_id.isin(new.mvt_id), 'no_id'].sum())))
        frames.append(summarize(d, new, ground, segments, tables).assign(day=day))
        print(json.dumps(reports[-1]), flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(args.out / f'{args.stage}_features.parquet', index=False)
    pd.concat(false_matches, ignore_index=True).to_csv(args.out / f'{args.stage}_false_matches.csv', index=False)
    summary = dict(stage=args.stage, seconds=time.monotonic() - started, days=reports,
                    parameters=PARAMETERS, calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(args.out / f'{args.stage}_report.json', summary)


if __name__ == '__main__':
    main()
