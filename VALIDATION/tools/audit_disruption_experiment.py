"""Independent SQL arrival aggregation and direct METAR checks on saved features."""
import bisect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import disruption_run as run
from common import connect, TRAIN_GLOB
from records import sha256, write_json
import weather


def main():
    directory = run.DEFAULT
    record = json.loads((directory / 'result.json').read_text())
    if record['status'] != 'complete':
        raise ValueError('Wait for experiment completion before this audit')
    jobs = {j['id']: j for j in run.overnight_run.schedule() if j['scheme'] == 'v3'}
    samples = []
    for name, info in dict(record['folds'], JAN_RETRO=record['january_holdout']).items():
        frame = pd.read_parquet(directory / info['predictions']['42'])
        take = frame.sample(n=min(20, len(frame)), random_state=103).copy()
        take['donor_months'] = '|'.join([*jobs[info['model_id']]['train'], info['val']])
        samples.append(take)
    sample = pd.concat(samples, ignore_index=True)
    sample['audit_id'] = np.arange(len(sample))
    queries = sample[['audit_id', 'airport', 'mvt_time', 'donor_months']].copy()
    con = connect()
    con.execute('SET threads=2')
    try:
        con.register('queries', queries)
        con.execute(f"""CREATE TEMP TABLE donors AS SELECT ADES_mvt airport,
            strftime(MVT_TIME_UTC_mvt::TIMESTAMP, '%Y-%m') AS month,
            TAXITIME_SEC_mvt::DOUBLE AS taxi,
            MVT_TIME_UTC_mvt::TIMESTAMP + TAXITIME_SEC_mvt * INTERVAL '1 second' AS completed
            FROM read_parquet('{TRAIN_GLOB}') WHERE PHASE_mvt='ARR'
            AND isfinite(TAXITIME_SEC_mvt::DOUBLE) AND TAXITIME_SEC_mvt >= 0""")
        arrival_max_error = {}
        for minutes in (60, 180):
            actual = con.execute(f"""SELECT q.audit_id, count(d.taxi) AS count,
                avg(least(d.taxi,7200)) FILTER (WHERE d.taxi IS NOT NULL) AS mean_sec,
                avg((d.taxi>1800)::DOUBLE) AS over30_share,
                avg((d.taxi>3600)::DOUBLE) AS over60_share
                FROM queries q LEFT JOIN donors d ON q.airport=d.airport
                AND list_contains(string_split(q.donor_months,'|'),d.month)
                AND d.completed < q.mvt_time::TIMESTAMP
                AND d.completed >= q.mvt_time::TIMESTAMP - INTERVAL '{minutes} minutes'
                GROUP BY q.audit_id ORDER BY q.audit_id""").df()
            for suffix in ('count', 'mean_sec', 'over30_share', 'over60_share'):
                col = f'dis_arr_{minutes}m_{suffix}'
                observed, expected = sample[col].to_numpy(), actual[suffix].to_numpy()
                np.testing.assert_allclose(observed, expected, atol=.0005, rtol=0, equal_nan=True)
                arrival_max_error[col] = float(np.nanmax(np.abs(observed - expected)))
    finally:
        con.close()
    metar = pd.read_parquet(weather.METAR, columns=['station', 'valid', 'wxcodes'])
    reports = {}
    for station, rows in metar.groupby('station'):
        values = {}
        for stamp, code in zip(pd.to_datetime(rows.valid, utc=True), rows.wxcodes):
            snow = any('SN' in token or 'SG' in token for token in str(code).replace('+', '').replace('-', '').split())
            values[stamp] = max(values.get(stamp, 0), int(snow))
        times = sorted(values)
        reports[station] = (times, values)
    for _, row in sample.iterrows():
        times, values = reports[row.airport]
        stamps = pd.date_range(end=pd.Timestamp(row.mvt_time).floor('30min'), periods=48, freq='30min')
        snow = []
        for stamp in stamps:
            i = bisect.bisect_right(times, stamp) - 1
            snow.append(values[times[i]] if i >= 0 and stamp - times[i] <= pd.Timedelta(minutes=90) else np.nan)
        for hours, minimum in ((6, 10), (24, 40)):
            window = np.asarray(snow[-hours * 2:])
            count = np.isfinite(window).sum()
            fraction = float(np.nanmean(window)) if count >= minimum else np.nan
            np.testing.assert_allclose(row[f'dis_snow_{hours}h_fraction'], fraction, atol=1e-7, rtol=0, equal_nan=True)
            np.testing.assert_allclose(row[f'dis_snow_{hours}h_coverage'], count / (hours * 2), atol=1e-7, rtol=0)
    result = dict(status='passed', sampled_rows=len(sample), folds=17,
        arrival_sql_checks=len(sample) * 8, snow_direct_checks=len(sample) * 4,
        arrival_max_absolute_error=arrival_max_error,
        source_sha256=sha256(Path(__file__)), run_record_sha256=sha256(directory / 'result.json'))
    path = ROOT / 'VALIDATION/reports/disruption_experiment/independent_audit.json'
    write_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
