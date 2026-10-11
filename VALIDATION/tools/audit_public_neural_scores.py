"""Independently check all public-pool CV labels, controls and score arithmetic."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
import numpy as np
import nn_public_pool3 as run
from records import write_json


def audit():
    path = run.OUT / 'score_audit.json'
    assert not path.exists(), 'Preserve completed audit'
    result = run.single.shared.ts.read(run.OUT / 'result.json')
    source = run.cat.r20.features.base.OUT / 'rows2025.parquet'
    predictions = run.OUT / 'scored.parquet'
    parent = run.cat.OUT / 'scored.parquet'
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(predictions)).create_view('candidate')
        con.read_parquet(str(source)).create_view('truth')
        con.read_parquet(str(parent)).create_view('parent')
        count, distinct_ids = con.execute('select count(*),count(distinct mvt_id) from candidate').fetchone()
        assert count == distinct_ids == 344419
        matched, errors = con.execute('''select count(*),sum(case when c.y is distinct from t.y
            or c.month is distinct from t.month or c.day is distinct from t.day
            or c.control is distinct from p.cat_only or c.v75 is distinct from p.cat_only_transfer
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join parent p using(mvt_id)''').fetchone()
        assert matched == count and errors == 0
        scores = con.execute('''select month,sqrt(avg(pow(control-y,2))) control,
            sqrt(avg(pow(pred-y,2))) pred,sqrt(avg(pow(v75-y,2))) v75,
            sqrt(avg(pow(transferred-y,2))) transferred from candidate group by month order by month''').fetchdf()
    assert scores.month.tolist() == ['2025-01', '2025-07']
    for _, row in scores.iterrows():
        for col in ['control', 'pred', 'v75', 'transferred']:
            np.testing.assert_allclose(row[col], result['scores'][row.month][col], rtol=0, atol=1e-8)
    for col in ['control', 'pred', 'v75', 'transferred']:
        np.testing.assert_allclose(scores[col].mean(), result['cv'][col], rtol=0, atol=1e-8)
    for a, b in [('control', 'pred'), ('v75', 'transferred')]:
        record = result['paired'][f'{b}_vs_{a}']
        np.testing.assert_allclose(record['change'], scores[b].mean()-scores[a].mean(), rtol=0, atol=1e-8)
        assert (record['decision'] == 'improvement') == (record['ci95'][1] < 0)
    record = dict(status='pass', rows=count, all_truth_and_control_values_exact=True,
        sql_monthly_and_mean_scores=True, source=run.single.artifact(__file__),
        result=run.single.artifact(run.OUT / 'result.json'), predictions=run.single.artifact(predictions),
        truth=run.single.artifact(source), parent=run.single.artifact(parent),
        scope='Labels, calendar, retained controls and score arithmetic; model reloads are in audit.json')
    write_json(path, record)
    print(record, flush=True)


if __name__ == '__main__':
    audit()
