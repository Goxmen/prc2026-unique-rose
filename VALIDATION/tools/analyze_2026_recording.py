"""Compare the 2025 and 2026 recording regimes using labels we already hold.

`ranking.parquet` blanks `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` only on
departure rows. Every arrival row in January and July 2026 carries its real
in-block time and taxi-in label, so the 2026 recording regime is directly
observable -- no model, no inference from departures. This reads the parquets
only: no fits, no uploads, no change to any recorded score.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import connect  # noqa: E402
from records import write_json  # noqa: E402

DATA = ROOT / 'data'
OUT = ROOT / 'VALIDATION/reports/recording_regime_2026.json'
PAIRS = [('2025-01', '2026-01'), ('2025-07', '2026-07')]

MOVEMENTS = """
WITH src AS (
    SELECT *, 'training' AS source FROM read_parquet($training)
    UNION ALL BY NAME
    SELECT *, 'ranking' AS source FROM read_parquet($ranking)
)
SELECT
    source,
    PHASE_mvt AS phase,
    strftime(MVT_TIME_UTC_mvt, '%Y-%m') AS ym,
    CASE WHEN PHASE_mvt = 'DEP' THEN ADEP_mvt ELSE ADES_mvt END AS airport,
    TAXITIME_SEC_mvt AS label,
    FLIGHT_ID_mvt IS NULL AS no_flight_id,
    AOBT_3_flt IS NULL AS no_aobt3,
    LOBT_flt IS NULL AS no_lobt,
    STAND_mvt IS NULL AS no_stand,
    AIRCRAFT_TYPE_mvt IS NULL AS no_type,
    AIRCRAFT_OPERATOR_flt IS NULL AS no_operator,
    epoch(BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt) AS block_minus_sched,
    epoch(MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt) AS mvt_minus_sched,
    epoch(MVT_TIME_UTC_mvt - LOBT_flt) AS mvt_minus_lobt
FROM src
"""

METRICS = """
SELECT {keys},
    count(*) AS n,
    100.0 * avg(no_flight_id::INT) AS pct_no_flight_id,
    100.0 * avg(no_aobt3::INT) AS pct_no_aobt3,
    100.0 * avg(no_lobt::INT) AS pct_no_lobt,
    100.0 * avg(no_stand::INT) AS pct_no_stand,
    100.0 * avg(no_type::INT) AS pct_no_type,
    100.0 * avg(no_operator::INT) AS pct_no_operator,
    count(label) AS n_label,
    median(label) AS label_median,
    quantile_cont(label, 0.99) AS label_p99,
    quantile_cont(label, 0.999) AS label_p999,
    max(label) AS label_max,
    100.0 * avg((label > 7200)::INT) AS pct_label_over_2h,
    100.0 * avg((label > 21600)::INT) AS pct_label_over_6h,
    100.0 * avg((label > 43200)::INT) AS pct_label_over_12h,
    -- off-block/in-block logged at the scheduled time: the recording artifact
    100.0 * avg((abs(block_minus_sched) <= 60)::INT) AS pct_sched_recorded,
    100.0 * avg((mvt_minus_sched > 14400 AND no_flight_id)::INT) AS pct_noid_over_4h_late,
    quantile_cont(mvt_minus_sched, 0.999) AS mvt_minus_sched_p999,
    max(mvt_minus_sched) AS mvt_minus_sched_max,
    max(mvt_minus_lobt) AS mvt_minus_lobt_max
FROM mv
GROUP BY {keys}
ORDER BY {keys}
"""


def frame(con, keys):
    df = con.sql(METRICS.format(keys=keys)).df()
    return df.astype(object).where(pd.notna(df), None)


def main():
    con = connect()
    con.execute('SET threads=6')
    con.execute(
        f'CREATE TABLE mv AS {MOVEMENTS}',
        {'training': [str(p) for p in sorted(DATA.glob('training_*.parquet'))],
         'ranking': [str(DATA / 'ranking.parquet')]})
    checks = con.sql("""
        SELECT source, phase, count(*) AS n, count(label) AS n_label,
               count(DISTINCT airport) AS airports,
               count(DISTINCT ym) AS months
        FROM mv GROUP BY 1, 2 ORDER BY 1, 2""").df()
    print('=== coverage check ===')
    print(checks.to_string(index=False))

    monthly = frame(con, 'ym, phase')
    by_airport = frame(con, 'ym, phase, airport')
    report = {'coverage': checks.to_dict('records'),
              'monthly': monthly.to_dict('records'),
              'by_airport': by_airport.to_dict('records')}
    write_json(OUT, report)

    show = ['ym', 'n', 'pct_no_flight_id', 'pct_no_aobt3', 'pct_sched_recorded',
            'pct_label_over_2h', 'pct_label_over_6h', 'pct_label_over_12h',
            'label_median', 'label_p999', 'label_max']
    for phase in ('ARR', 'DEP'):
        part = monthly[monthly.phase == phase]
        print(f'\n=== {phase} by month ===')
        print(part[show].to_string(index=False, float_format=lambda v: f'{v:,.3f}'))

    print('\n=== matched-month arrival comparison (2026 minus 2025) ===')
    arr = monthly[monthly.phase == 'ARR'].set_index('ym')
    for old, new in PAIRS:
        if old in arr.index and new in arr.index:
            print(f'\n{old} → {new}')
            for metric in show[2:]:
                a, b = arr.at[old, metric], arr.at[new, metric]
                if a is not None and b is not None:
                    print(f'  {metric:<22} {a:12,.3f} → {b:12,.3f}   ({b - a:+,.3f})')

    print('\n=== arrival schedule-recorded share by airport ===')
    arr_ap = by_airport[by_airport.phase == 'ARR']
    pivot = arr_ap.pivot_table(index='airport', columns='ym',
                               values='pct_sched_recorded', aggfunc='first')
    cols = [c for pair in PAIRS for c in pair if c in pivot.columns]
    print(pivot[cols].to_string(float_format=lambda v: f'{v:,.3f}'))

    print('\n=== arrival share of labels over 2 h by airport ===')
    pivot2 = arr_ap.pivot_table(index='airport', columns='ym',
                                values='pct_label_over_2h', aggfunc='first')
    print(pivot2[cols].to_string(float_format=lambda v: f'{v:,.4f}'))

    print('\n=== departure no-flight-ID share by airport ===')
    dep_ap = by_airport[by_airport.phase == 'DEP']
    pivot3 = dep_ap.pivot_table(index='airport', columns='ym',
                                values='pct_no_flight_id', aggfunc='first')
    print(pivot3[cols].to_string(float_format=lambda v: f'{v:,.3f}'))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
