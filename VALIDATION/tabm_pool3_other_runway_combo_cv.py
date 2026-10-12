"""Fixed additive combination declared before the extra TabM seed results.

Use original v78 integer deltas and exactly one nonnegative clip on v79.
The primary comparison is three-TabM/runway versus one-TabM/runway.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import tabm_other_runway_combo_cv as first

ROOT, RUNS = first.ROOT, first.RUNS
OUT = RUNS / 'v3_20261012_tabm_pool3_other_runway_combo'
POOL = RUNS / 'v3_20261012_nn_public_tabm_pool3'
STACK = RUNS / 'v3_20261012_nn_public_tabm_pool3_stack'
AUDIT = ROOT / 'VALIDATION/reports/public_tabm_20261012/pool3_score_audit.json'
read, art, verify, write = first.read, first.art, first.verify, first.write
DESIGN = dict(
    parent='exact v79',
    primary='three_tabm_runway versus one_tabm_runway',
    secondary='three_tabm_runway versus v79 and three_tabm alone',
    formula='rint(max(0,v79+(TabM_v78-exactv78)+(other_runway_v78-exactv78)))',
    seeds=[42, 1042, 2042], coefficients=[1, 1], new_bounds=False,
    frozen_before_extra_seed_results=True, no_fit=True, no_search=True,
    no_production_fit=True, no_upload=True,
    caveat='Fixed combination on reused Jan/Jul development; not independent confirmation.',
)


def freeze():
    assert not (OUT / 'identity.json').exists()
    assert not (STACK / 'result.json').exists()
    assert not list(POOL.glob('k*/networks/*.json')), 'Freeze before any extra seed completes'
    first.guard()
    paths = [Path(__file__), Path(first.__file__), first.OUT / 'identity.json',
             first.OUT / 'result.json', first.OUT / 'scored.parquet',
             ROOT / 'VALIDATION/tools/audit_tabm_runway_combination_20261012.py',
             RUNS / 'v3_20261012_tabm_other_runway_combo_submit/independent_audit.json',
             POOL / 'identity.json', ROOT / 'VALIDATION/nn_public_tabm_pool3_stack.py',
             ROOT / 'VALIDATION/tabm_pool3_v79_transport.py',
             ROOT / 'VALIDATION/tools/audit_public_tabm_pool3_scores_20261012.py']
    assert read(paths[6])['status'] == 'pass'
    np.testing.assert_array_equal(
        first.combine([10, 0, 7], [4, 4, 4], [4, 1, 6], [4, 0, 7]), [10, 0, 12])
    write(OUT / 'identity.json', dict(design=DESIGN, source=art(__file__),
          sources=[art(p) for p in paths], pending='four extra seed fits and audited pool scores'))


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN and identity['source'] == art(__file__)
    for record in identity['sources']:
        verify(record)
    first.guard()
    return identity


def score():
    guard()
    assert not (OUT / 'result.json').exists()
    audit = read(AUDIT)
    assert audit['status'] == 'pass'
    for record in audit['inputs']:
        verify(record)
    result = read(STACK / 'result.json')
    assert read(STACK / 'audit.json')['result'] == art(STACK / 'result.json')
    assert result['scored'] == art(STACK / 'scored.parquet')
    parent = pd.read_parquet(first.P79 / 'cv.parquet')
    ids = parent.mvt_id
    assert len(ids) == 344419 and ids.is_unique
    paths = [first.P78 / 'scored.parquet', STACK / 'scored.parquet',
             first.RUNWAY / 'scored.parquet', first.OUT / 'scored.parquet']
    frames = [pd.read_parquet(p).set_index('mvt_id') for p in paths]
    for f in frames:
        assert len(f) == len(ids) and f.index.is_unique and set(f.index) == set(ids)
    old, pool, runway, previous = [f.loc[ids] for f in frames]
    for f in [old, pool, runway, previous]:
        for col in ['y', 'month', 'day']:
            np.testing.assert_array_equal(f[col], parent[col])
    np.testing.assert_array_equal(pool.selected, old.candidate)
    np.testing.assert_array_equal(runway.control, old.candidate)
    f = parent[['mvt_id', 'month', 'day', 'y']].copy()
    f['control'] = parent.pred
    f['one_tabm_runway'] = first.combine(parent.pred, old.candidate, pool.one_tabm, runway.candidate)
    f['three_tabm'] = first.combine(parent.pred, old.candidate, pool.three_tabm, old.candidate)
    f['three_tabm_runway'] = first.combine(parent.pred, old.candidate, pool.three_tabm, runway.candidate)
    np.testing.assert_array_equal(f.one_tabm_runway, previous.combined)
    columns = ['control', 'one_tabm_runway', 'three_tabm', 'three_tabm_runway']
    scores = {m: {c: float(np.sqrt(np.mean((q[c] - q.y) ** 2))) for c in columns}
              for m, q in f.groupby('month')}
    paired = {c: first.bootstrap()(f.day, f.month, (f[c] - f.y) ** 2,
              (f.three_tabm_runway - f.y) ** 2) for c in columns[:-1]}
    f.to_parquet(OUT / 'scored.parquet', index=False)
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        for name, path in [('f', OUT / 'scored.parquet'), ('p', first.P79 / 'cv.parquet'),
                           ('o', paths[0]), ('t', paths[1]), ('r', paths[2]), ('prev', paths[3])]:
            con.read_parquet(str(path)).create_view(name)
        n, u, bad = con.execute('''SELECT count(*), count(distinct f.mvt_id), sum(CASE WHEN
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day OR
            f.control IS DISTINCT FROM p.pred OR f.one_tabm_runway IS DISTINCT FROM prev.combined OR
            f.one_tabm_runway IS DISTINCT FROM round_even(greatest(0,p.pred+t.one_tabm-o.candidate+r.candidate-o.candidate),0) OR
            f.three_tabm IS DISTINCT FROM round_even(greatest(0,p.pred+t.three_tabm-o.candidate),0) OR
            f.three_tabm_runway IS DISTINCT FROM round_even(greatest(0,p.pred+t.three_tabm-o.candidate+r.candidate-o.candidate),0)
            THEN 1 ELSE 0 END) FROM f JOIN p USING(mvt_id) JOIN o USING(mvt_id)
            JOIN t USING(mvt_id) JOIN r USING(mvt_id) JOIN prev USING(mvt_id)''').fetchone()
        assert n == u == 344419 and bad == 0
        for c in columns:
            for m, v in con.execute(f'SELECT month,sqrt(avg(pow({c}-y,2))) FROM f GROUP BY month').fetchall():
                assert abs(v - scores[m][c]) < 1e-8
    value = dict(design=DESIGN, identity=art(OUT / 'identity.json'), scored=art(OUT / 'scored.parquet'),
                 sources=[art(p) for p in [*paths, AUDIT, STACK / 'result.json', STACK / 'audit.json']],
                 scores=scores, cv={c: float(np.mean([q[c] for q in scores.values()])) for c in columns},
                 paired=paired, rows=n, exact_previous_combination=True,
                 all_row_sql_arithmetic_truth_calendar_scores='pass')
    write(OUT / 'result.json', value)
    print(dict(cv=value['cv'], paired=paired), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'score'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
