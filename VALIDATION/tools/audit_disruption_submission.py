"""Verify the prepared v42 file, correction stack and 320 sampled arrival inputs."""
import argparse
import bisect
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import connect, RANKING, TEMPLATE, TRAIN_GLOB
import opdi_disruption_submit as submit
import plan_window
import record_fixes
from records import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, default=42)
    args = parser.parse_args()
    directory = submit.cv.DEFAULT / f'submission_v{args.version}'
    manifest_path = directory / f'submission_v{args.version}.json'
    manifest = submit.cv.read(manifest_path)
    progress = submit.cv.read(directory / 'progress.json')
    submit.overnight_run.check_artifacts(directory, progress)
    submit.cv.prior.checked_file(directory, progress['training'])
    path = submit.submission(args.version)
    assert sha256(path) == manifest['sha256']
    table, template = pq.read_table(path), pq.read_table(TEMPLATE)
    submit.transfer.validate_submission(table, template)
    for part, info in manifest['model'].items():
        model = lgb.Booster(model_file=str(directory / info['path']), params={'num_threads': 2})
        assert model.feature_name() == info['features']
        new = set(submit.features.FEATURES) & set(model.feature_name())
        assert new == (set(submit.features.FEATURES) if info['part'] == 'ordinary' else set())
        assert model.current_iteration() == (800 if info['part'] == 'ordinary' else 300)
    con = connect()
    con.execute('SET threads=2')
    raw = con.execute(f"SELECT * FROM read_parquet('{RANKING}') WHERE PHASE_mvt='DEP'").df()
    raw = raw.rename(columns={'MVT_ID_mvt': 'mvt_id'}).set_index('mvt_id')
    pieces, stack_error = [], 0.
    for month, info in manifest['predictions'].items():
        frame = pd.read_parquet(directory / info['path']).set_index('mvt_id')
        rows = raw.loc[frame.index].reset_index()
        for name in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt'):
            rows[name] = pd.to_datetime(rows[name], utc=True)
        assert rows.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').eq(month).all()
        assert not manifest['training']['corrections']['day']['handler_stratified']
        for name in submit.corrections.SIGNALS:
            rows[name] = np.nan  # Full-year corrections use the pooled day weight.
        gap = (rows.MVT_TIME_UTC_mvt - rows.SCHED_TIME_UTC_mvt).dt.total_seconds()
        V = pd.DataFrame({submit.cv.rome_hybrid.GAP: gap})
        combined = submit.signals.combine(frame.ordinary.to_numpy(), frame.p.to_numpy(), V, rows,
            late_mean=manifest['training']['parameters']['late_mean'], dayshift_weight=None)
        capped = record_fixes.noid_cap(plan_window.apply(combined['final'], rows), rows)
        predictions, _ = submit.corrections.predict(capped, rows, manifest['training']['corrections'],
            frame.p.to_numpy(), twin_mask=record_fixes.twin_rows(rows))
        np.testing.assert_allclose(frame.pred, predictions['pred_combined'], atol=1e-8, rtol=0)
        np.testing.assert_array_equal(frame.rule_triggered, submit.cv.recipe.rule_rows(rows, 'rome_4h'))
        stack_error = max(stack_error, float(np.abs(frame.pred - predictions['pred_combined']).max()))
        pieces.append(frame)
    predicted = pd.concat(pieces)
    values = predicted.reindex(table['MVT_ID_mvt'].to_numpy()).pred.to_numpy()
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(), np.rint(values).astype('int32'))
    sample = predicted.groupby('month', group_keys=False).sample(n=160, random_state=942).copy()
    sample['airport'] = raw.loc[sample.index, 'ADEP_mvt']
    sample['takeoff'] = raw.loc[sample.index, 'MVT_TIME_UTC_mvt']
    sample['audit_id'] = np.arange(len(sample))
    con.register('queries', sample[['audit_id', 'airport', 'takeoff', 'month']])
    con.execute(f"""CREATE TEMP TABLE donors AS SELECT ADES_mvt airport,
        strftime(MVT_TIME_UTC_mvt::TIMESTAMP, '%Y-%m') AS donor_month,
        TAXITIME_SEC_mvt::DOUBLE taxi,
        MVT_TIME_UTC_mvt::TIMESTAMP + TAXITIME_SEC_mvt * INTERVAL '1 second' completed
        FROM read_parquet(['{TRAIN_GLOB}', '{RANKING}']) WHERE PHASE_mvt='ARR'
        AND isfinite(TAXITIME_SEC_mvt::DOUBLE) AND TAXITIME_SEC_mvt >= 0""")
    max_error = {}
    for minutes in (60, 180):
        expected = con.execute(f"""SELECT q.audit_id, count(d.taxi) AS count,
            avg(least(d.taxi,7200)) FILTER (WHERE d.taxi IS NOT NULL) mean_sec,
            avg((d.taxi>1800)::DOUBLE) over30_share, avg((d.taxi>3600)::DOUBLE) over60_share
            FROM queries q LEFT JOIN donors d ON q.airport=d.airport
            AND (d.donor_month LIKE '2025-%' OR d.donor_month=q.month)
            AND d.completed < q.takeoff::TIMESTAMP
            AND d.completed >= q.takeoff::TIMESTAMP - INTERVAL '{minutes} minutes'
            GROUP BY q.audit_id ORDER BY q.audit_id""").df()
        for suffix in ('count', 'mean_sec', 'over30_share', 'over60_share'):
            name = f'dis_arr_{minutes}m_{suffix}'
            np.testing.assert_allclose(sample[name], expected[suffix], atol=.0005, rtol=0, equal_nan=True)
            max_error[name] = float(np.nanmax(np.abs(sample[name].to_numpy() - expected[suffix].to_numpy())))
    con.close()
    reports = {}
    metar = pd.read_parquet(submit.weather.METAR, columns=['station', 'valid', 'wxcodes'])
    for station, rows in metar.groupby('station'):
        values = {}
        for stamp, code in zip(pd.to_datetime(rows.valid, utc=True), rows.wxcodes):
            snow = any('SN' in token or 'SG' in token for token in str(code).replace('+', '').replace('-', '').split())
            values[stamp] = max(values.get(stamp, 0), int(snow))
        reports[station] = sorted(values), values
    for _, row in sample.iterrows():
        times, values = reports[row.airport]
        stamps = pd.date_range(end=pd.Timestamp(row.takeoff).tz_convert('UTC').floor('30min'), periods=48, freq='30min')
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
    result = dict(status='passed', version=args.version, rows=len(predicted), model_rosters_checked=2,
        correction_stack_max_error=stack_error, sampled_rows=len(sample),
        arrival_sql_checks=len(sample) * 8, snow_direct_checks=len(sample) * 4,
        arrival_max_error=max_error, submission_sha256=sha256(path),
        source_sha256=sha256(Path(__file__)), prepared_manifest_sha256=sha256(manifest_path))
    write_json(directory / 'preupload_audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
