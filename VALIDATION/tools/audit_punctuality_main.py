"""Independent SQL verification of the fixed ordinary-punctuality CV screen."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
import numpy as np
from threadpoolctl import threadpool_limits

import punctuality_lgb_jan_jul as run
from records import write_json


def audit():
    run.guard()
    dest = run.OUT / 'audit.json'
    assert not dest.exists(), 'Preserve completed audit'
    read = run.parent.parent.ts.read
    result = read(run.OUT / 'result.json')
    assert result['design'] == run.DESIGN
    reloaded = 0
    for month in run.parent.parent.ts.SPLITS:
        path = run.OUT / 'predictions' / f'{month}_s42.json'
        rec = read(path)
        assert rec['identity'] == run.artifact(run.OUT / 'identity.json')
        assert rec['prediction'] == run.artifact(run.ROOT / rec['prediction']['path'])
        assert rec['model'] == run.artifact(run.ROOT / rec['model']['path'])
        assert rec['all_held_reload_exact'] and rec['rounds'] == 10000
        reloaded += rec['held_rows']
    assert reloaded == 344419
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('candidate', run.OUT / 'scored.parquet'),
                           ('control', run.parent.OUT / 'combined_public_cat.parquet'),
                           ('truth', run.parent.features.base.OUT / 'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, unique, wrong = con.execute('''select count(*), count(distinct c.mvt_id),
            sum(case when c.y is distinct from t.y or c.day is distinct from t.day or
            c.month is distinct from t.month or c.control is distinct from p.candidate or
            c.transferred_control is distinct from p.transferred then 1 else 0 end)
            from candidate c join control p using(mvt_id) join truth t using(mvt_id)''').fetchone()
        assert n == unique == 344419 and wrong == 0
        for name in result['cv']:
            metrics = con.execute(f'select month,sqrt(avg(pow({name}-y,2))) as value from candidate group by month').fetchall()
            for month, value in metrics:
                np.testing.assert_allclose(value, result['scores'][month][name], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in metrics]), result['cv'][name], atol=1e-8, rtol=0)
    value = dict(status='pass', rows=n, truth_calendar_controls_exact=True,
        independent_sql_scores=True, retained_all_row_model_reload_count=reloaded,
        source=run.artifact(__file__), identity=run.artifact(run.OUT / 'identity.json'),
        result=run.artifact(run.OUT / 'result.json'), predictions=run.artifact(run.OUT / 'scored.parquet'),
        scope='Independent SQL labels/controls/scores; verify model and prediction hashes and completed all-row fit reload receipts')
    write_json(dest, value)
    print(value, flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        audit()
