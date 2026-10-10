"""Fold-scoped aircraft rotation inputs from retained ADSB.lol observations.

Identities require a stable callsign (at least three records), a position at the
movement airport within 180 seconds, and exactly one candidate aircraft. Neither
departure taxi-out labels nor departure block times enter this module. Arrival
in-block is reconstructed from the supplied landing time and taxi-in duration.

The shared identity cache is input-only. Feature references are explicitly
restricted to each training fold, or training plus the held month for held rows.
The last training month is the neural stopping month: strict temporal precedence
means its events cannot enter earlier training queries.

The 24-hour count covers observed departures at the ten challenge airports,
not every flight of the aircraft. Own-aircraft arrival inputs expire after 72
hours; the latest-stand identity flag, like the existing stand clock, has no
age cap. Conservative duplicate-identity rejection is shared across folds.
"""
import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/derived/aircraft_rotation_20261010'
HEATMAP = ROOT / 'data/external/adsblol_heatmap'
FEATURE_NAMES = ['ar_aircraft', 'ar_prev_inblock_age_sec', 'ar_prev_same_stand',
                 'ar_prev_taxiin_sec', 'ar_departures_24h',
                 'ar_prev_minus_stand_age_sec', 'ar_stand_same_aircraft',
                 'ar_prev_arrival_delay_sec']
CATEGORICAL = ['ar_aircraft']
SETTINGS = dict(min_callsign_records=3, movement_slack_sec=180,
                max_arrival_age_sec=72 * 3600, identity_conflict_sec=120,
                arrival_context='strictly prior landing and in-block',
                labels='supplied arrival taxi-in only', missing_category='<MISSING>',
                feature_names=FEATURE_NAMES)
FOLDS = {'2025-01': [f'2025-{m:02d}' for m in range(3, 12)],
         '2025-07': ['2025-01', '2025-02', '2025-03', '2025-09',
                     '2025-10', '2025-11', '2025-12']}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def connection():
    con = duckdb.connect()
    con.execute("SET threads=2; SET TimeZone='UTC'; SET memory_limit='4GB'")
    return con


def read_movements(con, paths):
    """Read an explicit allowlist; departure target and block are never selected."""
    con.execute('''CREATE OR REPLACE TABLE movements AS
        SELECT MVT_ID_mvt::BIGINT mvt_id, PHASE_mvt phase, ADEP_mvt airport,
            MVT_TIME_UTC_mvt t,
            upper(replace(coalesce(nullif(trim(CALLSIGN_flt), ''), FLIGHT_mvt, ''), ' ', '')) cs,
            STAND_mvt stand, FLIGHT_ID_mvt IS NULL no_id,
            NULL::TIMESTAMPTZ inblock, NULL::DOUBLE taxiin,
            NULL::TIMESTAMPTZ arrival_schedule
        FROM read_parquet(?) WHERE PHASE_mvt='DEP'
        UNION ALL
        SELECT MVT_ID_mvt::BIGINT, PHASE_mvt, ADES_mvt, MVT_TIME_UTC_mvt,
            upper(replace(coalesce(nullif(trim(CALLSIGN_flt), ''), FLIGHT_mvt, ''), ' ', '')),
            STAND_mvt, FLIGHT_ID_mvt IS NULL,
            CASE WHEN TAXITIME_SEC_mvt>=0
                 THEN MVT_TIME_UTC_mvt + TAXITIME_SEC_mvt * INTERVAL 1 SECOND END,
            CASE WHEN TAXITIME_SEC_mvt>=0 THEN TAXITIME_SEC_mvt::DOUBLE END,
            SCHED_TIME_UTC_mvt
        FROM read_parquet(?) WHERE PHASE_mvt='ARR' ''',
        [[str(p) for p in paths]] * 2)
    n, unique = con.execute('SELECT count(*),count(DISTINCT mvt_id) FROM movements').fetchone()
    if n != unique:
        raise ValueError('Source movement IDs are not unique')


def match_day(con, day, heatmap=HEATMAP):
    """Return independent near-event identities, leaving ambiguity unmatched."""
    stamp = date.fromisoformat(day)
    dates = [str(stamp + timedelta(days=d)) for d in [-1, 0, 1]]
    calls = [heatmap / 'callsigns' / f'{d}.parquet' for d in dates]
    positions = [heatmap / 'positions' / f'{d}.parquet' for d in dates]
    calls, positions = [p for p in calls if p.exists()], [p for p in positions if p.exists()]
    if not (heatmap / 'callsigns' / f'{day}.parquet').exists() or not (
            heatmap / 'positions' / f'{day}.parquet').exists():
        return pd.DataFrame(columns=['mvt_id', 'hex', 'near_ms', 'position_n'])
    hi = str(stamp + timedelta(days=1))
    slack_ms = SETTINGS['movement_slack_sec'] * 1000
    con.execute(f'''CREATE OR REPLACE TEMP TABLE candidates AS
        SELECT DISTINCT m.mvt_id,m.airport,m.t,c.hex
        FROM movements m JOIN read_parquet(?) c
          ON m.cs=upper(replace(trim(c.callsign),' ',''))
          AND epoch_ms(m.t) BETWEEN c.first_ms-{slack_ms} AND c.last_ms+{slack_ms}
        WHERE m.t>=?::TIMESTAMPTZ AND m.t<?::TIMESTAMPTZ
          AND m.cs<>'' AND c.records>={SETTINGS['min_callsign_records']}''',
        [[str(p) for p in calls], day, hi])
    slack = SETTINGS['movement_slack_sec']
    return con.execute(f'''WITH near AS (
        SELECT c.mvt_id,c.hex,min(abs(epoch_ms(c.t)-epoch_ms(p.ts))) near_ms,
            count(*) position_n
        FROM candidates c JOIN read_parquet(?) p
          ON c.airport=p.airport AND c.hex=p.hex
          AND p.ts BETWEEN c.t-INTERVAL {slack} SECOND AND c.t+INTERVAL {slack} SECOND
        WHERE p.ts>=?::TIMESTAMPTZ-INTERVAL {slack} SECOND
          AND p.ts<?::TIMESTAMPTZ+INTERVAL {slack} SECOND
        GROUP BY c.mvt_id,c.hex)
        SELECT * FROM near WHERE mvt_id IN
            (SELECT mvt_id FROM near GROUP BY mvt_id HAVING count(*)=1)''',
        [[str(p) for p in positions], day, hi]).df()


def build_identity(out=DEFAULT, ranking=None, days=None):
    """Build only identities and allowed event fields; no model is fitted."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'identity.parquet').exists():
        raise FileExistsError('Identity cache exists; inspect its manifest before reusing it')
    paths = sorted((ROOT / 'data').glob('training_2025-*.parquet'))
    if ranking is not None:
        paths.append(Path(ranking))
    con = connection()
    read_movements(con, paths)
    available = [r[0] for r in con.execute(
        "SELECT DISTINCT strftime(t,'%Y-%m-%d') FROM movements ORDER BY 1").fetchall()]
    if days is not None:
        missing = set(days) - set(available)
        if missing:
            raise ValueError(f'Requested days not in movement inputs: {sorted(missing)}')
        available = sorted(set(days))
    started = time.monotonic()
    matches, coverage = [], []
    for i, day in enumerate(available):
        frame = match_day(con, day)
        matches.append(frame)
        coverage.append(dict(day=day, identities=len(frame)))
        if (i + 1) % 20 == 0:
            print(f'{i+1}/{len(available)} days, {time.monotonic()-started:.1f}s', flush=True)
    identities = pd.concat(matches, ignore_index=True)
    if identities.mvt_id.duplicated().any():
        raise ValueError('Duplicate movement identities')
    con.register('identified', identities)
    # Distinct movement rows cannot safely claim the same phase of one aircraft
    # within two minutes. Reject both claims, including cross-day neighbours.
    con.execute('''CREATE TABLE identity_raw AS
        SELECT m.*,i.hex,i.near_ms,i.position_n FROM movements m
        LEFT JOIN identified i USING(mvt_id)''')
    con.execute('''CREATE TABLE conflicts AS
        SELECT mvt_id FROM (
            SELECT mvt_id,epoch(t)-lag(epoch(t)) OVER w previous_gap,
                lead(epoch(t)) OVER w-epoch(t) next_gap
            FROM identity_raw WHERE hex IS NOT NULL
            WINDOW w AS (PARTITION BY hex,phase ORDER BY t,mvt_id))
        WHERE previous_gap<=? OR next_gap<=?''',
        [SETTINGS['identity_conflict_sec']] * 2)
    conflict_n = con.execute('SELECT count(*) FROM conflicts').fetchone()[0]
    con.execute('''CREATE TABLE identity_final AS SELECT r.* EXCLUDE(hex,near_ms,position_n),
        CASE WHEN c.mvt_id IS NULL THEN r.hex END hex,
        CASE WHEN c.mvt_id IS NULL THEN r.near_ms END near_ms,
        CASE WHEN c.mvt_id IS NULL THEN r.position_n END position_n,
        strftime(r.t,'%Y-%m') AS month FROM identity_raw r
        LEFT JOIN conflicts c USING(mvt_id)''')
    path = out / 'identity.parquet'
    con.execute('COPY identity_final TO ? (FORMAT PARQUET)', [str(path)])
    summary = con.execute('''SELECT phase,airport,count(*) n,count(hex) identified,
        count(*) FILTER(WHERE no_id) no_id,
        count(hex) FILTER(WHERE no_id) no_id_identified
        FROM identity_final GROUP BY phase,airport ORDER BY phase,airport''').df()
    summary.to_csv(out / 'identity_coverage.csv', index=False)
    source_days = sorted({str(date.fromisoformat(d) + timedelta(days=offset))
                          for d in available for offset in [-1, 0, 1]})
    adsb_sources = [HEATMAP / kind / f'{d}.parquet'
                   for kind in ['callsigns', 'positions'] for d in source_days
                   if (HEATMAP / kind / f'{d}.parquet').exists()]
    write_json(out / 'identity.json', dict(settings=SETTINGS,
        source_sha256=sha256(Path(__file__)), identity_sha256=sha256(path),
        movement_sources=[dict(path=str(p), sha256=sha256(p)) for p in paths],
        adsb_sources=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p))
                      for p in adsb_sources],
        days=coverage, conflicts=conflict_n, elapsed_sec=time.monotonic()-started))
    con.close()


def feature_table(events, query_ids, reference_months, con=None):
    """Eight label-free departure features from explicitly scoped input events."""
    con = con or connection()
    con.register('all_events', events)
    con.register('queries', pd.DataFrame({'mvt_id': np.asarray(query_ids, dtype=np.int64)}))
    con.register('allowed_months', pd.DataFrame({'month': list(reference_months)}))
    con.execute('''CREATE OR REPLACE TEMP TABLE context AS SELECT * FROM all_events
        WHERE month IN (SELECT month FROM allowed_months)''')
    con.execute('''CREATE OR REPLACE TEMP TABLE q AS SELECT e.* FROM queries q
        JOIN all_events e USING(mvt_id) WHERE e.phase='DEP' ''')
    if con.execute('SELECT count(*) FROM q').fetchone()[0] != len(query_ids):
        raise ValueError('Queries must all be unique departure movement IDs')
    if con.execute('SELECT count(*) FROM q WHERE month NOT IN (SELECT month FROM allowed_months)').fetchone()[0]:
        raise ValueError('Query months must be allowed reference months')
    con.execute('''CREATE OR REPLACE TEMP TABLE past_arrivals AS
        SELECT mvt_id aid,hex,airport arrival_airport,stand arrival_stand,
            t arrival_t,inblock arrival_inblock,taxiin arrival_taxiin,
            arrival_schedule previous_arrival_schedule
        FROM context WHERE phase='ARR' AND hex IS NOT NULL AND inblock IS NOT NULL''')
    con.execute('''CREATE OR REPLACE TEMP TABLE stand_arrivals AS
        SELECT mvt_id stand_aid,hex stand_hex,airport,stand,inblock stand_inblock
        FROM context WHERE phase='ARR' AND inblock IS NOT NULL AND stand IS NOT NULL''')
    con.execute('''CREATE OR REPLACE TEMP TABLE departure_counts AS
        SELECT mvt_id,count(*) OVER(PARTITION BY hex ORDER BY epoch(t)
            RANGE BETWEEN 86400 PRECEDING AND 0.000001 PRECEDING) dep_count
        FROM context WHERE phase='DEP' AND hex IS NOT NULL''')
    con.execute('''CREATE OR REPLACE TEMP TABLE joined AS SELECT q.*,a.* EXCLUDE(hex),
        s.stand_aid,s.stand_hex,s.stand_inblock,c.dep_count,
        epoch(q.t)-epoch(a.arrival_inblock) arrival_age,
        strftime(q.t-INTERVAL 24 HOUR,'%Y-%m') count_start_month
        FROM q ASOF LEFT JOIN past_arrivals a ON q.hex=a.hex AND q.t>a.arrival_t
        ASOF LEFT JOIN stand_arrivals s ON q.airport=s.airport AND q.stand=s.stand
            AND q.t>s.stand_inblock
        LEFT JOIN departure_counts c ON q.mvt_id=c.mvt_id''')
    limit = SETTINGS['max_arrival_age_sec']
    valid = f'(airport=arrival_airport AND arrival_age>0 AND arrival_age<={limit})'
    frame = con.execute(f'''SELECT mvt_id,
        CASE WHEN hex IS NULL THEN '<MISSING>' ELSE lower(lpad(to_hex(hex::BIGINT),6,'0')) END ar_aircraft,
        CASE WHEN {valid} THEN arrival_age END ar_prev_inblock_age_sec,
        CASE WHEN {valid} AND stand IS NOT NULL AND arrival_stand IS NOT NULL
             THEN (stand=arrival_stand)::DOUBLE END ar_prev_same_stand,
        CASE WHEN {valid} THEN arrival_taxiin END ar_prev_taxiin_sec,
        CASE WHEN hex IS NOT NULL AND count_start_month IN (SELECT month FROM allowed_months)
             THEN dep_count::DOUBLE END ar_departures_24h,
        CASE WHEN {valid} AND stand_inblock IS NOT NULL
             THEN epoch(stand_inblock)-epoch(arrival_inblock) END ar_prev_minus_stand_age_sec,
        CASE WHEN hex IS NOT NULL AND stand_hex IS NOT NULL
             THEN (hex=stand_hex)::DOUBLE END ar_stand_same_aircraft,
        CASE WHEN {valid} THEN epoch(arrival_inblock)-epoch(previous_arrival_schedule)
             END ar_prev_arrival_delay_sec
        FROM joined''').df()
    frame = frame.set_index('mvt_id').reindex(query_ids)
    if frame.index.duplicated().any() or frame.ar_aircraft.isna().any():
        raise ValueError('Feature rows do not cover each query exactly once')
    for name in FEATURE_NAMES[1:]:
        frame[name] = frame[name].astype(np.float32)
    return frame[FEATURE_NAMES]


def build_folds(out=DEFAULT):
    manifest = json.loads((out / 'identity.json').read_text())
    if sha256(out / 'identity.parquet') != manifest['identity_sha256']:
        raise ValueError('Identity cache changed')
    if manifest['settings'] != SETTINGS:
        raise ValueError('Identity settings changed')
    if sha256(Path(__file__)) != manifest['source_sha256']:
        raise ValueError('Builder source changed after identity construction')
    events = pd.read_parquet(out / 'identity.parquet')
    con = connection()
    reports = {}
    for month, train in FOLDS.items():
        reports[month] = {}
        for part, query_months, references in [('train', train, train),
                                               ('held', [month], train + [month])]:
            path = out / f'{month}_{part}.parquet'
            if path.exists():
                raise FileExistsError(path)
            ids = events.loc[events.phase.eq('DEP') & events.month.isin(query_months), 'mvt_id'].to_numpy()
            started = time.monotonic()
            frame = feature_table(events, ids, references, con)
            frame.reset_index().to_parquet(path, index=False)
            reports[month][part] = dict(rows=len(frame), reference_months=references,
                path=str(path), sha256=sha256(path), elapsed_sec=time.monotonic()-started,
                nonmissing={c: int(frame[c].notna().sum()) for c in FEATURE_NAMES},
                aircraft_identified=int(frame.ar_aircraft.ne('<MISSING>').sum()))
            print(f'{month} {part}: {len(frame):,} rows', flush=True)
    write_json(out / 'folds.json', dict(settings=SETTINGS, folds=reports,
        identity_sha256=manifest['identity_sha256'], source_sha256=sha256(Path(__file__))))
    con.close()


def load_features(path):
    table = pd.read_parquet(path).set_index('mvt_id')
    if table.index.duplicated().any() or list(table.columns) != FEATURE_NAMES:
        raise ValueError('Invalid aircraft rotation feature table')
    return table


def add_features(X, ids, table):
    if len(X) != len(ids) or set(FEATURE_NAMES) & set(X.columns):
        raise ValueError('Matrix length or pre-existing aircraft rotation inputs are invalid')
    values = table.reindex(np.asarray(ids))
    if values.ar_aircraft.isna().any():
        raise ValueError('Feature table misses queried movement IDs')
    values.index = X.index
    values = values.copy()
    values['ar_aircraft'] = values.ar_aircraft.astype('category')
    return pd.concat([X, values], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=['identity', 'folds'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--ranking', type=Path)
    parser.add_argument('--days', nargs='+')
    args = parser.parse_args()
    if args.stage == 'identity':
        build_identity(args.out, args.ranking, args.days)
    else:
        build_folds(args.out)


if __name__ == '__main__':
    main()
