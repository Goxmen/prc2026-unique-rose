"""Separately predeclared coefficient-one v79 transport for linear-leaf arms.

The phase/TabM transport registry is untouched. This adapter performs no fitting,
search, extra constraints, production preparation or upload.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import linear_leaf_main_jan_jul as main
from records import write_json

ROOT = main.ROOT
OUT = main.OUT/'v79'
PARENT = ROOT/'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'
OLD = main.v78.OUT
ARMS = main.ARMS
art, read, verify = main.art, main.read, main.verify
DESIGN = dict(parent='exact integer v79', original_control='exact v78',
    formula='round_even(max(0,v79+arm_on_v78-exactv78))', coefficient=1.,
    new_bounds=False, zero_delta_rows='byte-exact v79 values', arms=ARMS,
    primary_transport_comparison='candidate minus matched standardized constant',
    selected_comparisons='each transported arm separately versus actualv79',
    no_fit=True, no_search=True, no_submission=True,
    limitation='reused development and fixed additive transport, not independent confirmation')


def freeze():
    assert read(main.FROZEN)['design'] == main.DESIGN
    assert not (main.OUT/'result.json').exists() and not list((main.OUT/'models').glob('*'))
    parent = read(PARENT/'manifest.json')
    assert parent['state'] == 'scored' and parent['score'] == 219.613
    assert read(OLD/'audit.json')['status'] == 'pass'
    value = dict(design=DESIGN, primary_design=art(main.FROZEN), sources=[art(p) for p in
        [Path(__file__), Path(main.__file__), main.FROZEN, PARENT/'cv.parquet',
         PARENT/'manifest.json', OLD/'scored.parquet', OLD/'result.json', OLD/'audit.json']])
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'design.json').exists():
        assert read(OUT/'design.json') == value
    else:
        write_json(OUT/'design.json', value)
    print('Separate linear v79 transport frozen before either model arm is scored.', flush=True)


def transport(parent, selected78, arm78):
    parent, selected78, arm78 = [np.asarray(x, dtype='float64') for x in [parent, selected78, arm78]]
    assert parent.shape == selected78.shape == arm78.shape
    assert all(np.isfinite(x).all() for x in [parent, selected78, arm78])
    assert np.array_equal(parent, np.rint(parent)) and (parent >= 0).all()
    out = np.rint(np.maximum(0, parent+arm78-selected78))
    np.testing.assert_array_equal(out[arm78 == selected78], parent[arm78 == selected78])
    return out


def score():
    identity = read(OUT/'design.json')
    assert identity['design'] == DESIGN and identity['primary_design'] == art(main.FROZEN)
    for record in identity['sources']:
        verify(record)
    audit = read(main.OUT/'audit.json')
    assert audit['status'] == 'pass' and audit['result'] == art(main.OUT/'result.json')
    result = read(main.OUT/'result.json')
    assert result['identity'] == audit['identity'] == art(main.MANIFEST)
    for record in result['artifacts']+result['model_records']:
        verify(record)
    assert not (OUT/'result.json').exists(), 'Preserve completed secondary comparison'
    parent = pd.read_parquet(PARENT/'cv.parquet')
    old = pd.read_parquet(OLD/'scored.parquet').set_index('mvt_id')
    source = pd.read_parquet(main.OUT/'scored.parquet').set_index('mvt_id')
    assert len(parent) == len(old) == len(source) == 344419
    assert parent.mvt_id.is_unique and old.index.is_unique and source.index.is_unique
    ids = parent.mvt_id; old = old.loc[ids]; source = source.loc[ids]
    for column in ['y', 'month', 'day']:
        np.testing.assert_array_equal(source[column], parent[column])
        np.testing.assert_array_equal(old[column], parent[column])
    np.testing.assert_array_equal(source.selected, old.candidate)
    frame = parent[['mvt_id', 'y', 'month', 'day']].copy(); frame['selected'] = parent.pred
    for arm in ARMS:
        frame[arm] = transport(parent.pred, source.selected, source[arm])
    names = ['selected', *ARMS]
    scores = {month:{name:float(np.sqrt(np.mean((g[name]-g.y)**2))) for name in names}
        for month,g in frame.groupby('month')}
    comparisons = [('matched', 'constant', 'candidate'),
        ('candidate_vs_selected', 'selected', 'candidate'), ('constant_vs_selected', 'selected', 'constant')]
    paired = {name:main.base.parent.ts.paired_bootstrap(frame.day, frame.month,
        (frame[a]-frame.y)**2, (frame[b]-frame.y)**2) for name,a,b in comparisons}
    frame.to_parquet(OUT/'scored.parquet', index=False)
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in [('f',OUT/'scored.parquet'), ('p',PARENT/'cv.parquet'),
                          ('c',main.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        bad_arms = ' or '.join(f'f.{arm} is distinct from round_even(greatest(0,p.pred+c.{arm}-c.selected),0)'
            for arm in ARMS)
        n,unique,bad = con.execute(f'''select count(*),count(distinct f.mvt_id),sum(case when
            f.y is distinct from p.y or f.month is distinct from p.month or f.day is distinct from p.day
            or f.selected is distinct from p.pred or {bad_arms} then 1 else 0 end)
            from f join p using(mvt_id) join c using(mvt_id)''').fetchone()
        assert n == unique == 344419 and bad == 0
        for name in names:
            for month,value in con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from f group by month').fetchall():
                np.testing.assert_allclose(value, scores[month][name], rtol=0, atol=1e-8)
    result = dict(design=DESIGN, rows=n, scores=scores,
        cv={name:float(np.mean([s[name] for s in scores.values()])) for name in names},
        paired=paired, all_row_sql_audit='pass', exact_v78_selected_control=True,
        zero_delta_rows_exact_v79=True, identity=art(OUT/'design.json'),
        sources=[art(main.OUT/p) for p in ['audit.json','result.json','scored.parquet']],
        scored=art(OUT/'scored.parquet'))
    write_json(OUT/'result.json', result)
    write_json(OUT/'audit.json', dict(status='pass', rows=n, all_row_sql=True,
        result=art(OUT/'result.json'), source=art(__file__)))
    print(dict(cv=result['cv'], paired=paired), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'score'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
