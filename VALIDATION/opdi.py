"""OPDI inputs: what open ADS-B ground tracks saw of each departure.

OPDI (EUROCONTROL's Open Performance Data Initiative) turns OpenSky Network ADS-B
positions into per-flight events: first seen, leaving a parking position, entering
a taxiway, apron, de-icing pad or runway, take-off. The organiser ruled on Discord
(2026-09-18) that off-block times extracted from open trajectory data are allowed;
OPDI must be declared as an external dataset.

A departure is matched to an OPDI flight at the same departure airport whose
callsign equals its `CALLSIGN_flt` (or `FLIGHT_mvt` when that is missing) and whose
[first seen - 5 min, last seen + 5 min] contains the recorded takeoff; the latest
first seen at or before takeoff wins. Only events at or before takeoff (+30 s) are
used. Departure block times and taxi-out labels are never read.

OWN_FEATURES (missing when the flight is not matched or the event was not seen):

    op_matched              1 if an OPDI flight matched, else 0
    op_fs_ground            1 if OPDI first heard the aircraft inside the airport box
    op_fs_sec               takeoff - first seen, when first seen on the ground before takeoff
    op_own_stand_sec        takeoff - last exit of the departure's own stand (OSM ref = STAND_mvt)
    op_first_stand_sec      takeoff - first exit of any parking position
    op_first_apron_sec      takeoff - first exit of an apron
    op_first_twy_sec        takeoff - first entry to a taxiway
    op_rwy_sec              takeoff - last entry to a runway (time lined up / holding)
    op_deice_sec            seconds spent on de-icing pads (0 if matched and none)
    op_ground_sec           the earliest of the ground observations above: the longest
                            stretch OPDI saw the aircraft moving before takeoff

NEIGHBOUR_FEATURES, from the airport's other departures (never the row itself):

    nb_op_ground_mean_60    mean op_ground_sec of departures that took off in the 60 min before
    nb_op_ground_n_60       how many of them have op_ground_sec
    nb_op_ground_mean_pm30  mean op_ground_sec of departures within 30 min either side
    nb_op_deice_share_60    share of matched departures in the 60 min before that used a de-icing pad

Build the table once (all 2025 departures and the ranking departures) with
`.venv/bin/python VALIDATION/tools/fetch_opdi.py` then
`.venv/bin/python VALIDATION/opdi.py`; models read it through `add_opdi`.
"""

from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPDI_DIR = ROOT / 'data' / 'external' / 'opdi'
EVENTS_GLOB = OPDI_DIR / 'events' / 'flight_events_*.parquet'
LIST_GLOB = OPDI_DIR / 'flight_list' / 'flight_list_*.parquet'
FEATURES_PATH = OPDI_DIR / 'opdi_features.parquet'
TRAIN_GLOB = ROOT / 'data' / 'training_*.parquet'
RANKING = ROOT / 'data' / 'ranking.parquet'

AIRPORTS = {'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861), 'EGLL': (51.4700, -0.4543),
            'EHAM': (52.3086, 4.7639), 'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626),
            'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389), 'LSZH': (47.4647, 8.5492),
            'LTFM': (41.2753, 28.7519)}
BOX_LAT, BOX_LON = 0.1, 0.15
MATCH_SLACK_MIN = 5
EVENT_SLACK_SEC = 30

OWN_FEATURES = ['op_matched', 'op_fs_ground', 'op_fs_sec', 'op_own_stand_sec', 'op_first_stand_sec',
                'op_first_apron_sec', 'op_first_twy_sec', 'op_rwy_sec', 'op_deice_sec', 'op_ground_sec']
NEIGHBOUR_FEATURES = ['nb_op_ground_mean_60', 'nb_op_ground_n_60', 'nb_op_ground_mean_pm30',
                      'nb_op_deice_share_60']
OPDI_FEATURES = OWN_FEATURES + NEIGHBOUR_FEATURES
SETTINGS = dict(source='EUROCONTROL OPDI v0.0.2 flight events and flight list (from OpenSky Network ADS-B)',
                box_deg=[BOX_LAT, BOX_LON], match_slack_min=MATCH_SLACK_MIN,
                event_slack_sec=EVENT_SLACK_SEC, features=OPDI_FEATURES)


def _box(lat='latitude', lon='longitude'):
    return ' or '.join(f'({lat} between {la - BOX_LAT} and {la + BOX_LAT} and '
                       f'{lon} between {lo - BOX_LON} and {lo + BOX_LON})' for la, lo in AIRPORTS.values())


def _airport_case(lat='latitude', lon='longitude'):
    return 'case ' + ' '.join(f"when {lat} between {la - BOX_LAT} and {la + BOX_LAT} and "
                              f"{lon} between {lo - BOX_LON} and {lo + BOX_LON} then '{a}'"
                              for a, (la, lo) in AIRPORTS.items()) + ' end'


GROUND_TYPES = ('first_seen', 'take-off', 'entry-runway', 'exit-runway', 'entry-taxiway', 'exit-taxiway',
                'entry-parking_position', 'exit-parking_position', 'entry-apron', 'exit-apron',
                'entry-deicing_pad', 'exit-deicing_pad')


def filter_events(src, dst):
    """Keep ground events near the ten airports from one OPDI flight-events file."""
    con = duckdb.connect()
    info = "nullif(info, '')"
    con.execute(f"""copy (select flight_id, type, event_time::timestamp event_time, latitude lat, longitude lon,
            altitude alt, {_airport_case()} apt, json_extract_string({info}, '$.osm_ref') osm_ref
        from read_parquet('{src}') where type in {GROUND_TYPES} and ({_box()})) to '{dst}' (format parquet)""")
    return con.execute(f"select count(*) from read_parquet('{dst}')").fetchone()[0]


def _norm(expr):
    return f"upper(replace(replace({expr}, ' ', ''), '-', ''))"


def build_features(con=None, path=FEATURES_PATH):
    """One row per 2025 training and 2026 ranking departure with the OPDI inputs."""
    con = con or duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    airports = "('" + "','".join(AIRPORTS) + "')"
    deps = """select MVT_ID_mvt mvt_id, ADEP_mvt adep, MVT_TIME_UTC_mvt t, STAND_mvt stand,
                     nullif(trim(CALLSIGN_flt), '') cs, nullif(trim(FLIGHT_mvt), '') flt
              from read_parquet('{}') where PHASE_mvt = 'DEP'"""
    con.execute(f"""create or replace temp table dep as
        {deps.format(TRAIN_GLOB)} union all {deps.format(RANKING)}""")
    # Older monthly lists (flight list v2.0.0) store the flight ID as UBIGINT, newer ones as
    # BIGINT; events always use BIGINT. Map every ID to the same signed 64-bit value.
    signed = "case when {0}::hugeint >= 9223372036854775808 then {0}::hugeint - 18446744073709551616 else {0}::hugeint end"
    lists = sorted(Path(LIST_GLOB.parent).glob(LIST_GLOB.name))
    con.execute('create or replace temp table fl as ' + ' union all '.join(
        f"""select {signed.format('id')} id, trim(flt_id) cs, adep, first_seen::timestamptz fs,
                   last_seen::timestamptz ls from read_parquet('{f}') where adep in {airports}""" for f in lists))
    events = sorted(Path(EVENTS_GLOB.parent).glob(EVENTS_GLOB.name))
    con.execute('create or replace temp table ev as ' + ' union all '.join(
        f"""select {signed.format('flight_id')} flight_id, type, event_time::timestamptz et, apt, osm_ref
            from read_parquet('{f}')""" for f in events))
    slack = f"interval {MATCH_SLACK_MIN} minute"
    con.execute(f"""create or replace temp table m as select * exclude (rk) from (
        select c.*, row_number() over (
            partition by c.mvt_id order by (c.fs > c.t), abs(epoch(c.t) - epoch(c.fs))) rk
        from (select d.*, f.id oid, f.fs from dep d join fl f on f.adep = d.adep and f.cs = d.cs
                where d.t between f.fs - {slack} and f.ls + {slack}
              union all
              select d.*, f.id oid, f.fs from dep d join fl f on f.adep = d.adep and f.cs = d.flt
                where d.cs is null and d.t between f.fs - {slack} and f.ls + {slack}) c) where rk = 1""")
    lag = f"interval {EVENT_SLACK_SEC} second"
    con.execute(f"""create or replace temp table agg as select m.mvt_id,
        count(*) filter (where e.type = 'first_seen' and e.apt = m.adep) fs_at_apt,
        max(e.et) filter (where e.type = 'exit-parking_position' and {_norm('e.osm_ref')} = {_norm('m.stand')}) own_exit,
        min(e.et) filter (where e.type = 'exit-parking_position') first_stand,
        min(e.et) filter (where e.type = 'exit-apron') first_apron,
        min(e.et) filter (where e.type = 'entry-taxiway') first_twy,
        max(e.et) filter (where e.type = 'entry-runway') last_rwy,
        min(e.et) filter (where e.type = 'entry-deicing_pad') deice_in,
        max(e.et) filter (where e.type = 'exit-deicing_pad') deice_out
        from m join ev e on e.flight_id = m.oid and e.et <= m.t + {lag} and e.apt = m.adep group by 1""")
    sec = lambda col: f"nullif(greatest(epoch(m.t) - epoch({col}), 0), 0)"
    con.execute(f"""create or replace temp table own as select d.mvt_id, d.adep, d.t,
        (m.mvt_id is not null)::int op_matched,
        case when m.mvt_id is null then null else (coalesce(a.fs_at_apt, 0) > 0)::int end op_fs_ground,
        case when a.fs_at_apt > 0 and m.fs <= m.t - interval 30 second then epoch(m.t) - epoch(m.fs) end op_fs_sec,
        {sec('a.own_exit')} op_own_stand_sec, {sec('a.first_stand')} op_first_stand_sec,
        {sec('a.first_apron')} op_first_apron_sec, {sec('a.first_twy')} op_first_twy_sec,
        {sec('a.last_rwy')} op_rwy_sec,
        case when m.mvt_id is null then null
             else coalesce(greatest(epoch(a.deice_out) - epoch(a.deice_in), 0), 0) end op_deice_sec
        from dep d left join m using (mvt_id) left join agg a using (mvt_id)""")
    con.execute("""create or replace temp table own2 as select *,
        greatest(op_fs_sec, op_own_stand_sec, op_first_stand_sec, op_first_apron_sec, op_first_twy_sec) op_ground_sec
        from own""")
    con.execute(f"""copy (select *,
        (sum(op_ground_sec) over w60) / nullif(count(op_ground_sec) over w60, 0) nb_op_ground_mean_60,
        count(op_ground_sec) over w60 nb_op_ground_n_60,
        (coalesce(sum(op_ground_sec) over w30, 0) - coalesce(op_ground_sec, 0))
          / nullif(count(op_ground_sec) over w30 - (op_ground_sec is not null)::int, 0) nb_op_ground_mean_pm30,
        (sum((op_deice_sec > 0)::int) over w60) / nullif(sum(op_matched) over w60, 0) nb_op_deice_share_60
        from own2
        window w60 as (partition by adep order by epoch(t) range between 3600 preceding and 1 preceding),
               w30 as (partition by adep order by epoch(t) range between 1800 preceding and 1800 following))
        to '{path}' (format parquet)""")
    return con.execute(f"select count(*) from '{path}'").fetchone()[0]


def load_features(path=FEATURES_PATH):
    table = pd.read_parquet(path, columns=['mvt_id', *OPDI_FEATURES])
    if table.mvt_id.duplicated().any():
        raise ValueError('OPDI feature table has duplicate movement IDs')
    return table.set_index('mvt_id')


def add_opdi(X, raw, table, features=OPDI_FEATURES):
    """Attach the OPDI inputs to a model matrix whose rows align with `raw`."""
    if not X.index.equals(raw.index):
        raise ValueError('Matrix and raw movement rows must align before adding OPDI inputs')
    overlap = set(X.columns) & set(features)
    if overlap:
        raise ValueError(f'OPDI inputs already present: {sorted(overlap)}')
    missing = ~raw['mvt_id'].isin(table.index)
    if missing.any():
        raise ValueError(f'{int(missing.sum())} departures missing from the OPDI feature table')
    values = table.loc[raw['mvt_id'].to_numpy(), list(features)].to_numpy(float)
    return pd.concat([X, pd.DataFrame(values, index=X.index, columns=list(features))], axis=1)


if __name__ == '__main__':
    n = build_features()
    print(f'Wrote {FEATURES_PATH} ({n:,} departures)')
    sys.exit(0)
