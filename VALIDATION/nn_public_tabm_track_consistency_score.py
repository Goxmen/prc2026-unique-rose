"""Prefrozen three-arm continuation comparison in the exact serialized v80 stack.

Only target_consistency can pass the protocol. The two controls cannot be
promoted opportunistically. Every arm replaces the same seed42 member and
keeps members1042/2042 and all public corrections unchanged.
"""
import argparse
import ast
import inspect
import textwrap
import types
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nn_public_tabm_track_consistency as candidate
import nn_public_tabm_pool3_stack as previous
import tuning_seasonal as ts

ROOT = candidate.ROOT
OUT = candidate.OUT / 'comparison'
CURRENT = ROOT / 'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
PROTOCOL = ROOT / 'VALIDATION/reports/cohort_learning_20261013/target_track_consistency_protocol.json'
AUDITOR = ROOT / 'VALIDATION/tools/audit_track_consistency_scores_20261013.py'
ARMS = ['supervised_control', 'training_consistency', 'target_consistency']
read, art, verify, write = candidate.read, candidate.art, candidate.verify, candidate.write
DESIGN = dict(formula='R(v80_integer + serialized_new_threeTabM_component - serialized_original_threeTabM_component)',
              rounding='R is nonnegative nearest-even integer serialization',
              arms=ARMS, selected_arm='target_consistency', minimum_cv_gain_seconds=.1,
              comparisons=['v80','supervised_control','training_consistency'],
              gate='Both months improve and paired-day upper<0 against all three references; mean gain>=0.1s versus v80',
              interpretation='Reused transductive Jan/Jul development; not independent confirmation',
              no_control_promotion=True, no_weight_search=True, no_upload=True)


def inherited_score(arm):
    assert arm in ARMS and candidate.ARMS == ARMS

    def completed(month, k, identity):
        if k == 0:
            return candidate.completed(month, arm, identity)
        return candidate.parent.completed(month, k, read(candidate.parent.MANIFEST))

    proxy = types.SimpleNamespace(**vars(candidate))
    proxy.completed, proxy.STOP = completed, candidate.parent.STOP

    def retained_seed42(month, ids):
        path = candidate.parent.paths(month, 0)[1]
        frame = pd.read_parquet(path).set_index('mvt_id')
        assert frame.index.is_unique
        return frame.loc[ids, 'base'].to_numpy(float)

    source = textwrap.dedent(inspect.getsource(previous.score))
    tree = ast.parse(source)

    class OriginalOneMemberControl(ast.NodeTransformer):
        count = 0

        def visit_keyword(self, node):
            if node.arg == 'one_tabm':
                expected = ast.parse('np.mean([new[0], *members[1:]], axis=0)', mode='eval').body
                assert ast.dump(node.value) == ast.dump(expected)
                node.value = ast.parse('np.mean([retained_seed42(month,ids), *members[1:]], axis=0)', mode='eval').body
                self.count += 1
            return self.generic_visit(node)

    edit = OriginalOneMemberControl()
    tree = edit.visit(tree)
    assert edit.count == 1
    restored = ast.unparse(tree).replace('retained_seed42(month, ids)', 'new[0]')
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))
    namespace = dict(previous.score.__globals__, candidate=proxy,
                     OUT=OUT / arm / 'component', DESIGN=DESIGN,
                     retained_seed42=retained_seed42, __file__=__file__)
    exec(compile(ast.fix_missing_locations(tree), '<track-consistency-control-only>', 'exec'), namespace)
    return namespace['score']


def freeze():
    assert not (OUT / 'identity.json').exists()
    assert not any(candidate.OUT.glob('*/models/*.pt')), 'Comparison must predate fitting'
    for arm in ARMS:
        inherited_score(arm)
    parts = [Path(__file__), Path(previous.__file__), Path(ts.__file__), PROTOCOL, AUDITOR,
             candidate.MANIFEST, previous.OUT / 'scored.parquet', previous.OUT / 'result.json',
             previous.OUT / 'audit.json', CURRENT / 'scored.parquet', CURRENT / 'result.json',
             CURRENT / 'identity.json',
             ROOT / 'VALIDATION/reports/public_tabm_20261012/pool3_score_audit.json',
             ROOT / 'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo_submit/independent_audit.json',
             ROOT / 'data/training_2025-01-01_2025-02-01.parquet',
             ROOT / 'data/training_2025-07-01_2025-08-01.parquet']
    write(OUT / 'identity.json', dict(design=DESIGN, protocol=art(PROTOCOL), auditor=art(AUDITOR),
                                     sources=[art(p) for p in parts], before_fitting=True))
    print('Three-arm exact-v80 comparison frozen before fitting', flush=True)


def score():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN and identity['protocol'] == art(PROTOCOL)
    for record in identity['sources']:
        verify(record)
    assert not (OUT / 'result.json').exists()
    fitting = read(candidate.OUT / 'audit.json')
    assert fitting['status'] == 'pass' and fitting['identity'] == art(candidate.MANIFEST)
    assert len(fitting['members']) == 6
    for record in fitting['members']:
        verify(record)
    components = {}
    for arm in ARMS:
        folder = OUT / arm / 'component'
        if not (folder / 'result.json').exists():
            inherited_score(arm)()
        audit = read(folder / 'audit.json')
        assert audit['status'] == 'pass' and audit['result'] == art(folder / 'result.json')
        result = read(folder / 'result.json')
        assert result['scored'] == art(folder / 'scored.parquet')
        components[arm] = pd.read_parquet(folder / 'scored.parquet').set_index('mvt_id')
    original = pd.read_parquet(previous.OUT / 'scored.parquet').set_index('mvt_id')
    current = pd.read_parquet(CURRENT / 'scored.parquet').set_index('mvt_id').loc[original.index]
    assert len(original) == len(current) == 344419 and original.index.is_unique and current.index.is_unique
    frame = current[['month','day','y']].reset_index()
    frame['v80'] = current.three_tabm_runway.to_numpy(float)
    for field in ['month','day','y']:
        np.testing.assert_array_equal(current[field], original[field])
    for arm, component in components.items():
        assert len(component) == 344419 and component.index.is_unique
        component = component.loc[original.index]
        for field in ['month','day','y']:
            np.testing.assert_array_equal(component[field], original[field])
        frame[arm + '_component_delta'] = component.three_tabm.to_numpy(float) - original.three_tabm.to_numpy(float)
        frame[arm] = np.rint(np.maximum(frame.v80 + frame[arm + '_component_delta'], 0))
    scores = {month: {col: float(np.sqrt(np.mean((group[col] - group.y) ** 2)))
                      for col in ['v80', *ARMS]} for month, group in frame.groupby('month')}
    cv = {col: float(np.mean([m[col] for m in scores.values()])) for col in ['v80', *ARMS]}
    np.testing.assert_allclose(cv['v80'], 289.7627782102906, atol=1e-8, rtol=0)
    paired = {ref: ts.paired_bootstrap(frame.day, frame.month,
              (frame[ref] - frame.y) ** 2, (frame.target_consistency - frame.y) ** 2)
              for ref in DESIGN['comparisons']}
    both = {ref: all(m['target_consistency'] < m[ref] for m in scores.values())
            for ref in DESIGN['comparisons']}
    significant = {ref: paired[ref]['ci95'][1] < 0 for ref in DESIGN['comparisons']}
    minimum_gain = cv['v80'] - cv['target_consistency'] >= .1
    gate = dict(both_months=both, intervals_below_zero=significant, minimum_gain=minimum_gain,
                passed=all(both.values()) and all(significant.values()) and minimum_gain)
    frame.to_parquet(OUT / 'scored.parquet', index=False)
    with duckdb.connect(config={'threads':1, 'memory_limit':'1GB'}) as con:
        for name, path in [('f',OUT/'scored.parquet'),('o',previous.OUT/'scored.parquet'),('c',CURRENT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        for arm in ARMS:
            con.read_parquet(str(OUT/arm/'component/scored.parquet')).create_view('d')
            count, unique, bad = con.execute(f'''SELECT count(*),count(distinct f.mvt_id),
                count(*) FILTER(WHERE f.mvt_id IS NULL OR o.mvt_id IS NULL OR d.mvt_id IS NULL OR c.mvt_id IS NULL
                OR f.v80 IS DISTINCT FROM c.three_tabm_runway OR f.y IS DISTINCT FROM o.y
                OR f.y IS DISTINCT FROM c.y OR f.y IS DISTINCT FROM d.y
                OR f.month IS DISTINCT FROM o.month OR f.month IS DISTINCT FROM d.month
                OR f.day IS DISTINCT FROM o.day OR f.day IS DISTINCT FROM d.day
                OR f.{arm}_component_delta IS DISTINCT FROM CAST(d.three_tabm AS DOUBLE)-CAST(o.three_tabm AS DOUBLE)
                OR f.{arm} IS DISTINCT FROM round_even(greatest(0,CAST(c.three_tabm_runway AS DOUBLE)
                    +CAST(d.three_tabm AS DOUBLE)-CAST(o.three_tabm AS DOUBLE)),0))
                FROM f FULL JOIN o USING(mvt_id) FULL JOIN d USING(mvt_id) FULL JOIN c USING(mvt_id)''').fetchone()
            assert (count, unique, bad) == (344419, 344419, 0)
            for month, rmse in con.execute(f'SELECT month,sqrt(avg(pow({arm}-y,2))) FROM f GROUP BY month').fetchall():
                np.testing.assert_allclose(rmse, scores[month][arm], atol=1e-8, rtol=0)
    value = dict(identity=art(OUT/'identity.json'), fit_audit=art(candidate.OUT/'audit.json'),
                 components={arm:art(OUT/arm/'component/result.json') for arm in ARMS},
                 scores=scores, cv=cv, paired=paired, gate=gate,
                 exact_v80_replay=True, all_row_component_transport_SQL_exact=True,
                 scored=art(OUT/'scored.parquet'), independent_audit_pending=True,
                 no_control_promotion=True, no_production_or_upload=True)
    write(OUT/'result.json', value)
    print(dict(cv=cv, paired=paired, gate=gate), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','score'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
