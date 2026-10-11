"""Independent SQL source/calendar/score checks for off-block daily alignment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
import numpy as np
import pandas as pd
import daily_offblock_alignment as run
from records import write_json


def audit(source_only=False):
    out = run.OUT
    source_record = run.read(out / 'source_audit.json')
    assert source_record['prepared_rows'] == run.base.artifact(out / 'rows2025.parquet')
    paths = [str(run.base.ROOT / p['path']) for p in source_record['inputs']]
    expected = pd.read_parquet(out / 'daily_source_identity.parquet')
    checks = []
    with duckdb.connect(config={'threads': 1}) as con:
        con.execute("SET TimeZone='UTC'")
        for name, clock in [('takeoff', 'MVT_TIME_UTC_mvt'), ('recorded_offblock', 'BLOCK_TIME_UTC_mvt')]:
            actual = con.execute(f'''with a as (
                select ADEP_mvt airport,strftime({clock},'%Y-%m-%d') "day",count(*) n,
                    sum(greatest(0,epoch(BLOCK_TIME_UTC_mvt-SCHED_TIME_UTC_mvt))) truth
                from read_parquet(?) where PHASE_mvt='DEP' and FLIGHT_RULE_mvt is distinct from 'V'
                group by all)
                select a.*,s.FLT_DEP_IFR_2 n_public,60*s.DLY_ALL_PRE_2 total from a
                left join read_csv_auto(?) s on a.airport=s.APT_ICAO and a.day=s.FLT_DATE::varchar
                order by airport,"day"''', [paths, str(run.base.SOURCE / 'all_pre_departure_delays_2025.csv')]).fetchdf()
            wanted = expected[expected.assignment.eq(name)].sort_values(['airport', 'day']).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[['airport', 'day']], wanted[['airport', 'day']])
            np.testing.assert_allclose(actual[['n', 'truth', 'n_public']].to_numpy(dtype=float, na_value=np.nan),
                wanted[['n', 'truth', 'n_public']].to_numpy(dtype=float, na_value=np.nan), atol=0, rtol=0, equal_nan=True)
            # Independent CSV parsers differ in the last binary digit on a
            # few fractional minute totals (maximum observed <5e-10 seconds).
            np.testing.assert_allclose(actual.total.to_numpy(dtype=float, na_value=np.nan),
                wanted.total.to_numpy(dtype=float, na_value=np.nan), atol=1e-8, rtol=0, equal_nan=True)
            checks.append(dict(assignment=name, rows=len(actual), source_sums_exact=True,
                source_total_tolerance_seconds=1e-8))
        if source_only:
            result = dict(status='pass', source_checks=checks, source=run.base.artifact(__file__),
                source_audit=run.base.artifact(out / 'source_audit.json'))
            write_json(out / 'independent_source_audit.json', result)
            print(json.dumps(result, indent=2))
            return
        run.guard()
        result = run.read(out / 'result.json')
        f = pd.read_parquet(out / 'scored.parquet')
        con.register('f', f)
        con.read_parquet(str(out / 'rows2025.parquet')).create_view('rows')
        # Independently derive assigned calendar dates and the inference scope
        # directly from stored priors and epoch arithmetic, using SQL timestamps.
        calendar = con.execute('''select f.mvt_id,
            strftime(to_timestamp(r.takeoff_epoch-f.parent),'%Y-%m-%d') assigned_day,
            r.fltrule is distinct from 'V' and
            strftime(to_timestamp(r.takeoff_epoch-f.parent),'%Y-%m')=f.month and
            extract(day from to_timestamp(r.takeoff_epoch-f.parent))>1 and
            extract(day from to_timestamp(r.takeoff_epoch-f.parent)) <
                extract(day from last_day(to_timestamp(r.takeoff_epoch-f.parent))) new_scope
            from f join rows r using(mvt_id) order by f.mvt_id''').fetchdf()
        wanted = f.sort_values('mvt_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(calendar, wanted[['mvt_id', 'assigned_day', 'new_scope']])
        np.testing.assert_array_equal(f.loc[~f.new_scope, 'pred'], f.loc[~f.new_scope, 'control'])
        scores = con.execute('''select month,sqrt(avg(pow(parent-y,2))) parent,
            sqrt(avg(pow(old_all_departures-y,2))) old_all_departures,
            sqrt(avg(pow(control-y,2))) control,sqrt(avg(pow(pred-y,2))) pred
            from f group by month order by month''').fetchdf()
        for _, row in scores.iterrows():
            for column in ['parent', 'old_all_departures', 'control', 'pred']:
                np.testing.assert_allclose(row[column], result['scores'][row.month][column], atol=1e-9, rtol=0)
    write_json(out / 'audit.json', dict(status='pass', source_checks=checks,
        assigned_dates_and_scope_sql_rows=len(f), fallback_exact=True, scores_sql_exact=True,
        source=run.base.artifact(__file__), result=run.base.artifact(out / 'result.json'),
        prediction=run.base.artifact(out / 'scored.parquet'),
        scope='Source sums/calendar selection/scoring; unchanged posterior message algorithm has a retained enumeration audit'))
    print('PASS: raw source sums, every assigned date/scope/fallback, independent scores')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    audit(args.source_only)
