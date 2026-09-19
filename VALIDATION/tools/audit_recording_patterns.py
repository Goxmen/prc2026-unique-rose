"""Reproduce a bounded raw-clock audit; ranking departure labels are never used.

Run from the repo root with .venv/bin/python. One projected parquet scan builds a
small disk-backed DuckDB table, then all comparisons use that table. Candidate
cutoffs (6/12/20/24 hours, clock agreement 60/300 seconds) describe recording
mechanisms, not a validation-score search. The output is exploratory evidence,
not independent validation of rules discovered from the same training labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb


PATTERNS = {
    'all_departures': 'true',
    'missing_flight_id': 'flight_id is null',
    'id_lobt_ge_6h': 'flight_id is not null and g_lobt >= 21600',
    'id_lobt_ge_12h': 'flight_id is not null and g_lobt >= 43200',
    'id_lobt_ge_20h': 'flight_id is not null and g_lobt >= 72000',
    'id_lobt_ge_12h_agree_60s': 'flight_id is not null and g_lobt >= 43200 and abs(g_lobt-g_aobt3)<=60',
    'id_lobt_ge_12h_agree_300s': 'flight_id is not null and g_lobt >= 43200 and abs(g_lobt-g_aobt3)<=300',
    'id_aobt3_ge_12h_lobt_lt_6h': 'flight_id is not null and g_aobt3 >= 43200 and g_lobt < 21600',
    'id_lobt_ge_6h_agree_300s': 'flight_id is not null and g_lobt >= 21600 and abs(g_lobt-g_aobt3)<=300',
    'id_lobt_ge_6h_aobt3_missing': 'flight_id is not null and g_lobt >= 21600 and g_aobt3 is null',
    'id_aobt_ge_2h_agree_30min': 'flight_id is not null and g_aobt3 >= 7200 and abs(g_lobt-g_aobt3)<=1800',
    'id_aobt_ge_6h_agree_30min': 'flight_id is not null and g_aobt3 >= 21600 and abs(g_lobt-g_aobt3)<=1800',
    'id_aobt_ge_12h_agree_30min': 'flight_id is not null and g_aobt3 >= 43200 and abs(g_lobt-g_aobt3)<=1800',
    'rome_noid_ge_4h': "airport='LIRF' and flight_id is null and g_sched >= 14400",
    'rome_noid_ge_14h': "airport='LIRF' and flight_id is null and g_sched >= 50400",
    'rome_noid_ge_24h': "airport='LIRF' and flight_id is null and g_sched >= 86400",
    'nonrome_noid_1to4h': "airport<>'LIRF' and flight_id is null and g_sched between 3600 and 14399",
    'nonrome_noid_ge_4h': "airport<>'LIRF' and flight_id is null and g_sched >= 14400",
    'nonrome_noid_ge_6h': "airport<>'LIRF' and flight_id is null and g_sched >= 21600",
    'nonrome_noid_ge_12h': "airport<>'LIRF' and flight_id is null and g_sched >= 43200",
    'nonrome_noid_ge_24h': "airport<>'LIRF' and flight_id is null and g_sched >= 86400",
    'missing_id_missing_aircraft': 'flight_id is null and aircraft is null',
    'negative_aobt_gap': 'g_aobt3 < 0',
    'negative_lobt_gap': 'g_lobt < 0',
    'negative_sched_gap': 'g_sched < 0',
}


def records(con, sql):
    cur = con.execute(sql)
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def md_table(rows, columns):
    def fmt(value):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                     '| ' + ' | '.join('---' for _ in columns) + ' |',
                     *['| ' + ' | '.join(fmt(r.get(c)) for c in columns) + ' |' for r in rows]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('VALIDATION/reports/recording_artifacts_20260919'))
    parser.add_argument('--scratch', type=Path, default=Path('tmp/recording_artifacts_20260919'))
    parser.add_argument('--reuse-projection', action='store_true', help='Reuse the projected local DuckDB table')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.scratch / 'raw_clock_audit.duckdb'))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{args.scratch / 'spill'}'")
    con.execute('SET TimeZone=\'UTC\'')
    print('Reusing projected table' if args.reuse_projection else 'Building projected departure table', flush=True)
    if not args.reuse_projection:
      con.execute("""CREATE OR REPLACE TABLE departures AS
      WITH raw AS (
        SELECT 'training' AS dataset, MVT_ID_mvt, FLIGHT_ID_mvt, ADEP_mvt, ADES_mvt,
          FLIGHT_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, RUNWAY_mvt,
          MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt, LOBT_flt, AOBT_3_flt,
          EOBT_1_flt, IOBT_flt, TAXITIME_SEC_mvt AS label,
          epoch(MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt) AS block_gap
        FROM read_parquet('data/training_*.parquet') WHERE PHASE_mvt='DEP'
        UNION ALL
        SELECT 'ranking', MVT_ID_mvt, FLIGHT_ID_mvt, ADEP_mvt, ADES_mvt,
          FLIGHT_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, RUNWAY_mvt,
          MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt, LOBT_flt, AOBT_3_flt,
          EOBT_1_flt, IOBT_flt, NULL::INTEGER, NULL::DOUBLE
        FROM read_parquet('data/ranking.parquet') WHERE PHASE_mvt='DEP'
      ) SELECT dataset, MVT_ID_mvt::BIGINT AS id, FLIGHT_ID_mvt::BIGINT AS flight_id,
        ADEP_mvt AS airport, ADES_mvt AS destination, FLIGHT_mvt AS flight,
        STAND_mvt AS stand, AIRCRAFT_TYPE_mvt AS aircraft, RUNWAY_mvt AS runway,
        CASE WHEN regexp_matches(left(FLIGHT_mvt,3),'^[A-Za-z]{3}$')
             THEN left(FLIGHT_mvt,3) ELSE left(FLIGHT_mvt,2) END AS airline,
        MVT_TIME_UTC_mvt AS takeoff, SCHED_TIME_UTC_mvt AS schedule,
        extract(month from MVT_TIME_UTC_mvt)::INTEGER AS month,
        epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) AS g_sched,
        epoch(MVT_TIME_UTC_mvt-LOBT_flt) AS g_lobt,
        epoch(MVT_TIME_UTC_mvt-AOBT_3_flt) AS g_aobt3,
        epoch(MVT_TIME_UTC_mvt-EOBT_1_flt) AS g_eobt1,
        epoch(MVT_TIME_UTC_mvt-IOBT_flt) AS g_iobt,
        date_diff('day', timezone('Europe/Rome', SCHED_TIME_UTC_mvt),
                         timezone('Europe/Rome', MVT_TIME_UTC_mvt)) AS rome_days,
        label, block_gap
      FROM raw""")
    con.execute("""CREATE OR REPLACE VIEW populations AS
      SELECT *, CASE WHEN dataset='training' THEN 'train_2025' ELSE 'ranking_jan_jul' END AS population
      FROM departures
      UNION ALL SELECT *, 'cv_unique_jul_dec' FROM departures WHERE dataset='training' AND month>=7
      UNION ALL SELECT *, 'train_jan' FROM departures WHERE dataset='training' AND month=1
      UNION ALL SELECT *, 'train_jul' FROM departures WHERE dataset='training' AND month=7
      UNION ALL SELECT *, 'ranking_jan' FROM departures WHERE dataset='ranking' AND month=1
      UNION ALL SELECT *, 'ranking_jul' FROM departures WHERE dataset='ranking' AND month=7""")
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  interpretation='Raw-label and observable-feature audit, not model validation. Ranking labels and block times are explicitly NULL.',
                  pattern_definitions=PATTERNS)
    result['populations'] = records(con, """SELECT population,count(*) n,
      count(label) labels, count(*) FILTER(WHERE label<=0 OR label>200000) outside_training_filter,
      max(abs(label-block_gap)) max_label_vs_block_gap_seconds FROM populations GROUP BY population ORDER BY population""")
    branches = []
    for name, where in PATTERNS.items():
        branches.append(f"""SELECT '{name}' pattern, population, count(*) n,
          avg(label) mean_label, median(label) median_label,
          median(label-g_lobt) median_label_minus_lobt,
          quantile_cont(label-g_lobt,0.1) p10_label_minus_lobt,
          quantile_cont(label-g_lobt,0.9) p90_label_minus_lobt,
          median(label-g_aobt3) median_label_minus_aobt3,
          quantile_cont(label-g_aobt3,0.1) p10_label_minus_aobt3,
          quantile_cont(label-g_aobt3,0.9) p90_label_minus_aobt3,
          sqrt(avg(pow(label-g_lobt,2))) lobt_rmse,
          sqrt(avg(pow(label-g_aobt3,2))) aobt3_rmse,
          sqrt(avg(pow(label-g_sched,2))) sched_rmse,
          count(*) FILTER(WHERE abs(label-g_sched)<=60) schedule_matches,
          count(*) FILTER(WHERE abs(label-g_lobt)<=60) lobt_matches,
          count(*) FILTER(WHERE abs(label-g_aobt3)<=60) aobt3_matches,
          count(*) FILTER(WHERE rome_days>=1 AND label-rome_days*86400 BETWEEN 0 AND 7200) day_shift_labels
        FROM populations WHERE {where} GROUP BY population""")
    result['patterns'] = records(con, ' UNION ALL '.join(branches))
    sizes = {r['population']: r['n'] for r in result['populations']}
    for row in result['patterns']:
        row['per_100k'] = 100000 * row['n'] / sizes[row['population']]
    print(json.dumps({'populations': result['populations'], 'rare_patterns': [r for r in result['patterns'] if r['pattern'] in ('id_lobt_ge_12h', 'id_lobt_ge_12h_agree_300s','rome_noid_ge_24h','nonrome_noid_ge_12h') and r['population'] in ('train_2025','cv_unique_jul_dec','ranking_jan_jul')]}, indent=2), flush=True)
    result['noid_airport_delay'] = records(con, """SELECT population, airport,
       CASE WHEN g_sched<3600 THEN 'under_1h' WHEN g_sched<14400 THEN '1_to_4h'
            WHEN g_sched<43200 THEN '4_to_12h' WHEN g_sched<86400 THEN '12_to_24h' ELSE '24h_plus' END gap_bin,
       count(*) n, median(label) median_label, count(label) labels,
       count(*) FILTER(WHERE abs(label-g_sched)<=60) schedule_matches,
       count(*) FILTER(WHERE abs(label-g_sched-3600)<=60) plus_one_hour,
       count(*) FILTER(WHERE abs(label-g_sched+3600)<=60) minus_one_hour,
       count(*) FILTER(WHERE label-rome_days*86400 BETWEEN 0 AND 7200 AND rome_days>=1) day_shift_labels,
       sqrt(avg(pow(label-g_sched,2))) sched_rmse
       FROM populations WHERE flight_id IS NULL AND g_sched IS NOT NULL
        AND population IN ('train_2025','cv_unique_jul_dec','ranking_jan_jul')
       GROUP BY population,airport,gap_bin ORDER BY airport,gap_bin,population""")
    result['clock_offset_signatures'] = records(con, """WITH diffs AS (
        SELECT population,airport,'AOBT3' AS clock,label-g_aobt3 AS clock_error FROM populations
        WHERE flight_id IS NOT NULL AND population IN ('train_2025','cv_unique_jul_dec')
        UNION ALL SELECT population,airport,'LOBT',label-g_lobt FROM populations
        WHERE flight_id IS NOT NULL AND population IN ('train_2025','cv_unique_jul_dec'))
      SELECT population,airport,clock,count(*) FILTER(WHERE abs(clock_error)<=60) zero_hour,
        count(*) FILTER(WHERE abs(clock_error-3600)<=60) plus_hour,
        count(*) FILTER(WHERE abs(clock_error+3600)<=60) minus_hour,
        count(*) FILTER(WHERE abs(clock_error-86400)<=60) plus_day,
        count(*) FILTER(WHERE abs(clock_error+86400)<=60) minus_day,
        count(*) FILTER(WHERE abs(clock_error)>3660) outside_61min
      FROM diffs GROUP BY population,airport,clock ORDER BY population,airport,clock""")
    result['rome_multi_day_airlines'] = records(con, """SELECT population,airline,count(*) n,
      count(label) labels, count(*) FILTER(WHERE abs(label-g_sched)<=60) schedule_matches,
      count(*) FILTER(WHERE label-rome_days*86400 BETWEEN 0 AND 7200 AND rome_days>=1) day_shift_labels,
      median(label-g_sched) median_label_minus_schedule,
      median(label-rome_days*86400) median_label_minus_days,
      sqrt(avg(pow(label-g_sched,2))) schedule_rmse
      FROM populations WHERE airport='LIRF' AND flight_id IS NULL AND g_sched>=50400
        AND population IN ('train_2025','cv_unique_jul_dec','ranking_jan_jul')
      GROUP BY population,airline ORDER BY population,n DESC,airline""")
    result['nonrome_noid_ge4h_labels'] = records(con, """SELECT airport,count(*) n,
      quantile_cont(label,0.99) p99_label,max(label) max_label,
      count(*) FILTER(WHERE label>7200) over_two_hours
      FROM departures WHERE dataset='training' AND airport<>'LIRF'
        AND flight_id IS NULL AND g_sched>=14400
      GROUP BY airport ORDER BY airport""")
    result['short_gap_extreme_noid_labels'] = records(con, """SELECT airport,month,count(*) n,
      min(label) min_label,max(label) max_label FROM departures
      WHERE dataset='training' AND flight_id IS NULL AND label>14400 AND g_sched<7200
      GROUP BY airport,month ORDER BY airport,month""")
    result['rome_rule_ordinary_label_cases'] = records(con, """SELECT airline,month,count(*) n,
      min(label) min_label,max(label) max_label,min(g_sched) min_gap,max(g_sched) max_gap
      FROM departures WHERE dataset='training' AND airport='LIRF' AND flight_id IS NULL
        AND g_sched>=14400 AND label<7200
      GROUP BY airline,month ORDER BY month,airline""")
    # Same-flight clusters use only movement fields; the target is only an audit.
    con.execute("""CREATE OR REPLACE TABLE flight_clusters AS WITH steps AS (
      SELECT *, epoch(takeoff-lag(takeoff) OVER (PARTITION BY dataset,airport,flight ORDER BY takeoff,id)) step
      FROM departures WHERE flight IS NOT NULL), groups AS (
      SELECT *, sum(CASE WHEN step<=180 THEN 0 ELSE 1 END) OVER
        (PARTITION BY dataset,airport,flight ORDER BY takeoff,id ROWS UNBOUNDED PRECEDING) cluster_id
      FROM steps)
      SELECT dataset,airport,flight,cluster_id,count(*) n,count(flight_id) ids,
        count(DISTINCT stand) stands,min(month) AS month,min(takeoff) AS takeoff,
        min(label) min_label,max(label) max_label,
        median(label-g_lobt) median_lobt_error,
        median(label-g_lobt) FILTER(WHERE flight_id IS NOT NULL) id_lobt_error,
        list(id ORDER BY takeoff,id) movement_ids
      FROM groups GROUP BY dataset,airport,flight,cluster_id HAVING count(*)>1""")
    result['duplicate_cluster_types'] = records(con, """SELECT dataset,airport,n,ids,count(*) clusters,
      count(*) FILTER(WHERE month>=7) july_dec_clusters,
      median(max_label-min_label) median_label_spread,
      median(id_lobt_error) median_id_label_minus_lobt,
      count(*) FILTER(WHERE stands>1) different_stand_clusters
      FROM flight_clusters GROUP BY dataset,airport,n,ids ORDER BY dataset,clusters DESC,airport,n,ids""")
    result['duplicate_flight_ids'] = records(con, """WITH repeats AS (
      SELECT dataset,airport,flight_id,count(*) n,count(DISTINCT flight) AS name_count,
      epoch(max(takeoff)-min(takeoff)) spread,max(label)-min(label) label_spread
      FROM departures WHERE flight_id IS NOT NULL GROUP BY dataset,airport,flight_id HAVING count(*)>1)
      SELECT dataset,airport,count(*) clusters,count(*) FILTER(WHERE spread<=180) within_3min,
        count(*) FILTER(WHERE name_count>1) different_names,median(label_spread) median_label_spread
      FROM repeats GROUP BY dataset,airport ORDER BY dataset,clusters DESC,airport""")
    # Local examples remain ignored, including all extreme ranking clock cases.
    local = records(con, """SELECT * EXCLUDE(block_gap)
      REPLACE(takeoff::VARCHAR AS takeoff,schedule::VARCHAR AS schedule) FROM departures WHERE
      (flight_id IS NOT NULL AND (g_lobt>=21600 OR g_aobt3>=43200))
      OR (flight_id IS NULL AND g_sched>=43200)
      ORDER BY dataset,airport,takeoff,id""")
    (args.scratch / 'extreme_raw_rows.json').write_text(json.dumps(local, indent=2, default=str)+'\n')
    (args.scratch / 'duplicate_clusters.json').write_text(json.dumps(records(con,
        'SELECT * REPLACE(takeoff::VARCHAR AS takeoff) FROM flight_clusters ORDER BY dataset,airport,takeoff,flight'), indent=2, default=str)+'\n')
    result['local_details'] = {'extreme_rows': str(args.scratch / 'extreme_raw_rows.json'),
                               'database': str(args.scratch / 'raw_clock_audit.duckdb')}
    scored_file = args.scratch / 'unique-rose_v34.parquet'
    if scored_file.exists():
        con.execute(f"""CREATE OR REPLACE VIEW ranking_scored AS SELECT d.*,p.TAXITIME_SEC_mvt AS v34
          FROM departures d JOIN read_parquet('{scored_file}') p ON d.id=p.MVT_ID_mvt
          WHERE d.dataset='ranking'""")
        result['ranking_anchor_changes'] = []
        result['ranking_nonrome_noid_ge4h_predictions'] = records(con, """SELECT airport,count(*) n,
          max(v34) max_prediction,count(*) FILTER(WHERE v34>3600) over_one_hour,
          count(*) FILTER(WHERE v34>7200) over_two_hours
          FROM ranking_scored WHERE airport<>'LIRF' AND flight_id IS NULL AND g_sched>=14400
          GROUP BY airport ORDER BY airport""")
        for hours in (2,6,12):
            rows = records(con, f"""SELECT count(*) n,{hours} minimum_hours,
              median(abs(v34-g_aobt3)) median_abs_change,
              max(abs(v34-g_aobt3)) max_abs_change,
              sum(pow(v34-g_aobt3,2)) squared_change_sum
              FROM ranking_scored WHERE flight_id IS NOT NULL
              AND g_aobt3>={hours}*3600 AND abs(g_lobt-g_aobt3)<=1800""")
            result['ranking_anchor_changes'].extend(rows)
        ranking_extremes = records(con, """SELECT * EXCLUDE(block_gap,label)
          REPLACE(takeoff::VARCHAR AS takeoff,schedule::VARCHAR AS schedule) FROM ranking_scored
          WHERE flight_id IS NOT NULL AND g_aobt3>=7200 AND abs(g_lobt-g_aobt3)<=1800
          ORDER BY abs(v34-g_aobt3) DESC,id LIMIT 50""")
        (args.scratch / 'ranking_anchor_examples.json').write_text(json.dumps(ranking_extremes,indent=2,default=str)+'\n')
    (args.output / 'raw_clock_audit.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    headline = [r for r in result['patterns'] if r['population'] in ('train_2025','cv_unique_jul_dec','ranking_jan_jul')]
    text = '# Raw recording-pattern audit\n\n' + result['interpretation'] + '\n\n'
    text += 'CV prevalence counts each July–December 2025 departure once, rather than once per fold and seed. The full training-year audit includes labels later used for validation, so discovered rules need explicit exploratory status.\n\n'
    text += md_table(result['populations'], ['population','n','labels','outside_training_filter','max_label_vs_block_gap_seconds']) + '\n\n'
    text += md_table(headline, ['pattern','population','n','per_100k','median_label_minus_lobt','lobt_rmse','aobt3_rmse','schedule_matches','day_shift_labels']) + '\n\n'
    text += 'All clock errors above are single-clock diagnostic RMSE, not experiment validation scores. Full airport, airline, hour/day-offset and duplicate summaries are in `raw_clock_audit.json`. Row examples stay in ignored `tmp/recording_artifacts_20260919/`.\n\n'
    text += 'Reproduce: `.venv/bin/python VALIDATION/tools/audit_recording_patterns.py`. Resource cap: four DuckDB threads, 2 GB memory; no fits, uploads or hidden ranking labels.\n'
    (args.output / 'raw_clock_audit.md').write_text(text)
    con.close()
    print(f'Wrote {args.output}/raw_clock_audit.json and .md', flush=True)


if __name__ == '__main__':
    main()
