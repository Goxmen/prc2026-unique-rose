"""Audit retained daily production with double-precision tree arithmetic.

The frozen original audit added FLOAT main/airport columns before reaching
the DOUBLE CatBoost column, unlike NumPy's all-double production matrix.
Preserve its source, identity, calibration and every prediction unchanged.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import public_daily_taxi_submit as parent
from records import write_json

OUT, MANIFEST = parent.OUT, parent.MANIFEST
art, read, verify = parent.art, parent.read, parent.verify
aggregate, airport, daily, constraints = parent.aggregate, parent.airport, parent.daily, parent.constraints


def audit():
    parent.guard()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT / 'result.json')
    for artifact in result['artifacts'] + result['component_sources']:
        verify(artifact)
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(OUT / 'predictions.parquet')).create_view('c')
        con.read_parquet(str(OUT / 'candidate.parquet')).create_view('f')
        con.read_parquet(str(aggregate.V76 / 'predictions.parquet')).create_view('p')
        con.read_parquet(str(airport.OUT / 'predictions/old.parquet')).create_view('a')
        con.read_csv(str(daily.base.SOURCE / 'taxi_out_additional_time_2026.csv')).create_view('s')
        n, unique, errors = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.direct_old is distinct from p.candidate or c.tree_old is distinct from p.tree
            or c.airport_new is distinct from a.pred or c.parent76 is distinct from p.submitted
            or c.refined != round_even(greatest(c.refined_float,0),0)
            or c.invalid_share is distinct from (s.TF-s.VALID_FL)/s.TF
            or c.quality_eligible != coalesce((s.TF-s.VALID_FL)/s.TF<0.05,false)
            or c.applied_shift != case when c.quality_eligible then c.unrestricted_shift else 0 end
            or c.submitted != round_even(greatest(c.bounded,0),0)
            or c.submitted is distinct from f.TAXITIME_SEC_mvt then 1 else 0 end)
            from c join p using(mvt_id) join a using(mvt_id) join f on c.mvt_id=f.MVT_ID_mvt
            left join s on c.airport=s.APT_ICAO and substr(c.local_day,1,7)=cast(s.YEAR as varchar)||'-'||lpad(cast(s.MONTH_NUM as varchar),2,'0')''').fetchone()
        assert n == unique == 344841 and errors == 0, (n, unique, errors)
        old_error, tree_error = con.execute('''select
            max(abs(tree_new-(main+airport_new+catboost)/3.0)),
            max(abs(tree_new-(cast(main as double)+cast(airport_new as double)+cast(catboost as double))/3.0)) from c''').fetchone()
        assert tree_error < 1e-8
        con.read_parquet(str(OUT / 'cv.parquet')).create_view('cv')
        cvrows, cvunique = con.execute('select count(*),count(distinct mvt_id) from cv').fetchone()
        assert cvrows == cvunique == 344419
        scores = con.execute('select month,sqrt(avg(pow(pred-y,2))) from cv group by month').fetchall()
        for month, value in scores:
            np.testing.assert_allclose(value, result['cv_months'][month], atol=1e-8, rtol=0)
        np.testing.assert_allclose(np.mean([v for m, v in scores]), result['jan_jul_cv'], atol=1e-8, rtol=0)
    frame = pd.read_parquet(OUT / 'predictions.parquet'); ids = frame.mvt_id.to_numpy()
    expected_tree = frame[['main', 'airport_new', 'catboost']].to_numpy(float).mean(axis=1)
    np.testing.assert_array_equal(expected_tree, frame.tree_new)
    inputs = frame[['mvt_id', 'airport', 'local_day', 'runway', 'eligible', 'month']]
    _, _, shift = daily.apply(inputs, frame.refined.to_numpy(), daily.public(), read(OUT / 'parameters.json'))
    np.testing.assert_array_equal(shift, frame.unrestricted_shift)
    raw = pd.read_parquet(daily.base.OUT / 'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(constraints.exclusion.OUT / 'ranking_matches.parquet')
    np.testing.assert_array_equal(aggregate.constrain(ids, raw,
        frame.refined.to_numpy() + frame.applied_shift.to_numpy(), matches), frame.bounded)
    table, template = pq.read_table(OUT / 'candidate.parquet'), pq.read_table(ROOT / 'data/submitting.parquet')
    assert table.schema == template.schema
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt':
            assert table[name].equals(template[name])
    report = dict(status='pass', rows=n, cv_rows=cvrows, all_sql_controls_quality_serializations_cv=True,
        tree_mean_max_error=float(tree_error), original_float32_intermediate_error=float(old_error),
        independent_numpy_tree_mean_exact=True, all_daily_shifts_and_final_bounds_replayed=True,
        template_ids_schema_exact=True, identity=art(MANIFEST), result=art(OUT / 'result.json'),
        source=art(__file__), original_auditor_source=art(parent.__file__),
        fix='Cast all three components to DOUBLE before adding, matching production arithmetic',
        predictions_calibration_and_identity_unchanged=True, no_fit=True, no_version=True, no_upload=True)
    write_json(OUT / 'audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        audit()
