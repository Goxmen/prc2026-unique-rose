"""Stand clock and dirty-day inputs: two observable witnesses of when a plane left its stand.

A departure without a flight ID has no off-block clock at all (no AOBT3, no EOBT):
only its schedule and its takeoff are known. These inputs add two witnesses, both
built from movements that `ranking.parquet` still carries. Departure block times
and taxi-out labels are never read: only the columns in `DEP_COLUMNS` and
`ARR_COLUMNS` are taken from the event frames. Arrival in-block is landing plus
taxi-in, both supplied for ranking arrivals.

STAND_FEATURES, from arrivals at the same airport and stand:

    st_last_arr_sec             seconds from the last in-block at this stand before
                                takeoff to takeoff. If that arrival was another
                                aircraft, this one had already left, so taxi >= it;
                                if it was this aircraft, it cannot have left before
                                it arrived, so taxi <= it.
    st_prev_arr_sec             the same for the arrival before that one
    st_same_type                1 if the last arrival's aircraft type equals this
                                departure's, 0 if not, missing if either is missing
    st_same_airline             the same for the airline read from FLIGHT_mvt
    st_inblock_minus_sched_sec  the last arrival's in-block minus this departure's
                                schedule: positive when a plane parked after the
                                departure was due to leave

DAY_FEATURES, from the airport's UTC day of takeoff (arrivals: UTC day of landing):

    dd_noid_share               share of the day's other departures without a flight ID
    dd_notype_share             share of the day's other departures without an aircraft type
    dd_arr_sched_share          share of the day's arrivals whose in-block equals their
                                schedule within 60 s, i.e. logged rather than measured
    dd_delay_mean_sec           mean takeoff-minus-schedule of the day's other departures
                                with a flight ID, each clipped to [0, 3 h]
    dd_late_share               share of the day's other departures with a flight ID
                                that took off more than 45 min after schedule
    dh_noid_share_pm90          share of the airport's departures without a flight ID
                                within 90 min either side of takeoff, excluding the
                                takeoff instant

"Other" departures exclude the described row when it is part of the event frame
(matched by `mvt_id`), so a row never counts itself and the result does not depend
on whether the described rows were passed as events.
"""

import numpy as np
import pandas as pd

from rome_signals import airline_code
from runway_queue import SEC, _Events, _groups, _mean, _ns

DEP_COLUMNS = ['mvt_id', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'FLIGHT_ID_mvt', 'AIRCRAFT_TYPE_mvt']
ARR_COLUMNS = ['ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'taxi_sec',
               'AIRCRAFT_TYPE_mvt', 'FLIGHT_mvt']
RAW_COLUMNS = ['mvt_id', 'ADEP_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
               'AIRCRAFT_TYPE_mvt', 'FLIGHT_mvt']
STAND_FEATURES = ['st_last_arr_sec', 'st_prev_arr_sec', 'st_same_type', 'st_same_airline',
                  'st_inblock_minus_sched_sec']
DAY_FEATURES = ['dd_noid_share', 'dd_notype_share', 'dd_arr_sched_share', 'dd_delay_mean_sec',
                'dd_late_share', 'dh_noid_share_pm90']
ALL_FEATURES = STAND_FEATURES + DAY_FEATURES
ARR_SCHED_MATCH_SEC = 60
LATE_SEC = 45 * 60
DELAY_CLIP_SEC = 3 * 3600
WINDOW_SEC = 90 * 60
DAY_NS = 86400 * SEC
SETTINGS = dict(stand=STAND_FEATURES, day=DAY_FEATURES, day_scope='airport x UTC day',
                arr_sched_match_sec=ARR_SCHED_MATCH_SEC, late_sec=LATE_SEC, delay_clip_sec=DELAY_CLIP_SEC,
                window_sec=WINDOW_SEC,
                dep_columns=DEP_COLUMNS, arr_columns=ARR_COLUMNS)


def _stand_key(airport, stand):
    key = airport.astype(object) + ':' + stand.astype(object)
    return np.array(key.where(airport.notna() & stand.notna()), dtype=object)


def _in_block(arr):
    t, ok = _ns(arr['MVT_TIME_UTC_mvt'])
    taxi = pd.to_numeric(arr['taxi_sec'], errors='coerce').to_numpy(float)
    ok &= np.isfinite(taxi)
    return np.where(ok, t + np.round(np.where(ok, taxi, 0) * SEC).astype(np.int64), 0), ok


def _match(a, b):
    """1.0 where both are present and equal, 0.0 where both are present and differ, else NaN."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    known = pd.notna(a) & pd.notna(b)
    return np.where(known, (a == b).astype(float), np.nan)


def stand_features(raw, arr_events):
    """STAND_FEATURES aligned to `raw` departures, from the supplied arrivals only."""
    arr = arr_events.loc[:, ARR_COLUMNS]
    q = raw.loc[:, RAW_COLUMNS]
    n = len(q)
    inb, inb_ok = _in_block(arr)
    a_key = _stand_key(arr['ADES_mvt'], arr['STAND_mvt'])
    keep = inb_ok & pd.notna(a_key)
    a = pd.DataFrame({'key': a_key[keep], 'inb': inb[keep],
                      'a_type': arr['AIRCRAFT_TYPE_mvt'].to_numpy(object)[keep],
                      'a_airline': airline_code(arr['FLIGHT_mvt']).to_numpy(object)[keep]})
    a = a.sort_values(['key', 'inb'], kind='mergesort')
    a['prev_inb'] = a.groupby('key', sort=False)['inb'].shift().astype(float)
    t, t_ok = _ns(q['MVT_TIME_UTC_mvt'])
    d_key = _stand_key(q['ADEP_mvt'], q['STAND_mvt'])
    use = t_ok & pd.notna(d_key)
    d = pd.DataFrame({'key': d_key[use], 't': t[use], 'row': np.flatnonzero(use)}).sort_values('t', kind='mergesort')
    # Strictly before takeoff: an in-block at the takeoff instant does not count.
    m = pd.merge_asof(d, a.sort_values('inb', kind='mergesort'), left_on='t', right_on='inb', by='key',
                      allow_exact_matches=False, direction='backward').sort_values('row')
    rows = m['row'].to_numpy()
    last = m['inb'].to_numpy(float)
    sched, sched_ok = _ns(q['SCHED_TIME_UTC_mvt'])
    out = {name: np.full(n, np.nan) for name in STAND_FEATURES}
    out['st_last_arr_sec'][rows] = (m['t'].to_numpy(float) - last) / SEC
    out['st_prev_arr_sec'][rows] = (m['t'].to_numpy(float) - m['prev_inb'].to_numpy(float)) / SEC
    out['st_same_type'][rows] = _match(q['AIRCRAFT_TYPE_mvt'].to_numpy(object)[rows], m['a_type'])
    out['st_same_airline'][rows] = _match(airline_code(q['FLIGHT_mvt']).to_numpy(object)[rows], m['a_airline'])
    out['st_inblock_minus_sched_sec'][rows] = np.where(sched_ok[rows], (last - sched[rows]) / SEC, np.nan)
    return pd.DataFrame(out, index=raw.index)


def _day(t, ok):
    return np.where(ok, t // DAY_NS, -1)


def _lookup(table, airport, day, columns):
    index = pd.MultiIndex.from_arrays([airport, day])
    return table.reindex(index)[columns].to_numpy(float).T


def day_features(raw, dep_events, arr_events):
    """DAY_FEATURES aligned to `raw` departures, from the supplied event frames only."""
    dep = dep_events.loc[:, DEP_COLUMNS]
    arr = arr_events.loc[:, ARR_COLUMNS]
    q = raw.loc[:, RAW_COLUMNS]
    n = len(q)

    t, t_ok = _ns(dep['MVT_TIME_UTC_mvt'])
    sched, sched_ok = _ns(dep['SCHED_TIME_UTC_mvt'])
    noid = dep['FLIGHT_ID_mvt'].isna().to_numpy()
    delay = np.where(t_ok & sched_ok, (t - sched) / SEC, np.nan)
    timed = ~noid & np.isfinite(delay)
    # Each event's contribution to its airport-day sums; a described event's own row is subtracted below.
    parts = pd.DataFrame({'n': 1.0, 'noid': noid.astype(float),
                          'notype': dep['AIRCRAFT_TYPE_mvt'].isna().to_numpy().astype(float),
                          'timed': timed.astype(float),
                          'late': (timed & (np.nan_to_num(delay) > LATE_SEC)).astype(float),
                          'delay': np.where(timed, np.clip(np.nan_to_num(delay), 0, DELAY_CLIP_SEC), 0.0)},
                         index=dep['mvt_id'].to_numpy())
    keyed = parts.assign(airport=dep['ADEP_mvt'].to_numpy(object), day=_day(t, t_ok))[t_ok]
    by_day = keyed.groupby(['airport', 'day'])[list(parts.columns)].sum()

    inb, inb_ok = _in_block(arr)
    a_sched, a_sched_ok = _ns(arr['SCHED_TIME_UTC_mvt'])
    a_t, a_t_ok = _ns(arr['MVT_TIME_UTC_mvt'])
    ok = inb_ok & a_sched_ok & a_t_ok
    a = pd.DataFrame({'airport': arr['ADES_mvt'].to_numpy(object), 'day': _day(a_t, a_t_ok),
                      'match': (np.abs(inb - a_sched) <= ARR_SCHED_MATCH_SEC * SEC).astype(float)})[ok]
    arr_share = a.groupby(['airport', 'day'])['match'].mean().rename('share').to_frame()

    q_t, q_ok = _ns(q['MVT_TIME_UTC_mvt'])
    airport, day = q['ADEP_mvt'].to_numpy(object), _day(q_t, q_ok)
    sums = pd.DataFrame(_lookup(by_day, airport, day, list(parts.columns)).T, columns=parts.columns)
    member = q['mvt_id'].isin(parts.index).to_numpy() & q_ok
    own = pd.DataFrame(0.0, index=range(n), columns=parts.columns)
    own.loc[member] = parts.loc[q['mvt_id'].to_numpy()[member]].to_numpy()
    others = sums - own
    with np.errstate(invalid='ignore', divide='ignore'):
        share = lambda column, base: np.where(others[base] > 0, others[column] / others[base], np.nan)
        out = {'dd_noid_share': share('noid', 'n'), 'dd_notype_share': share('notype', 'n'),
               'dd_arr_sched_share': _lookup(arr_share, airport, day, ['share'])[0],
               'dd_delay_mean_sec': share('delay', 'timed'), 'dd_late_share': share('late', 'timed')}
    events_by_airport = _Events(dep['ADEP_mvt'].to_numpy(object), t, t_ok, noid.astype(float))
    keys = airport.copy()
    keys[~q_ok] = None
    w = WINDOW_SEC * SEC
    out['dh_noid_share_pm90'] = _mean(events_by_airport, _groups(keys), q_t,
                                      [(-w, 0, True, False), (0, w, False, True)], pd.isna(keys))
    frame = pd.DataFrame({name: out[name] for name in DAY_FEATURES}, index=raw.index)
    return frame.where(np.repeat(q_ok[:, None], len(DAY_FEATURES), axis=1))


def stand_day_features(raw, dep_events, arr_events, features=ALL_FEATURES):
    wanted = set(features)
    if wanted - set(ALL_FEATURES):
        raise ValueError(f'Unknown stand/day inputs: {sorted(wanted - set(ALL_FEATURES))}')
    parts = []
    if wanted & set(STAND_FEATURES):
        parts.append(stand_features(raw, arr_events))
    if wanted & set(DAY_FEATURES):
        parts.append(day_features(raw, dep_events, arr_events))
    frame = pd.concat(parts, axis=1)
    return frame.loc[:, [name for name in ALL_FEATURES if name in wanted]].astype('float32')


def add_stand_day(X, raw, dep_events, arr_events, features=ALL_FEATURES):
    if not X.index.equals(raw.index):
        raise ValueError('Matrix and raw movement rows must align before adding stand/day inputs')
    overlap = set(X.columns) & set(features)
    if overlap:
        raise ValueError(f'Stand/day inputs already present: {sorted(overlap)}')
    return pd.concat([X, stand_day_features(raw, dep_events, arr_events, features)], axis=1)
