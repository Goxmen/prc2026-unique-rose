"""One fixed post-TabM blend screen: total neural share0.71 to0.85, no fits."""
import argparse
import ast
import inspect
from pathlib import Path
import textwrap

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nn_public_tabm_pool3_stack as retained
import tuning_seasonal as validation

ROOT = retained.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_tabm_neural_share'
CURRENT = ROOT / 'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
art, read, verify, write = retained.art, retained.read, retained.verify, retained.write_json
DESIGN = dict(parent='exactv80', change='Total ordinary neural share0.71 to0.85 in the selected three-TabM component',
    rationale='The retained0.71 share predates the TabM architecture and public model improvements',
    retained='Equal callsign/physical neural mixture, all seeds/models/features/calibration, all source and runway increments',
    transport='One integer new-minus-original TabM-v78 component delta onto exactv80, then nonnegative rounding',
    primary='Both months improve, average CV improves at least0.05seconds and paired-day95% upper<0',
    no_fits=True, no_weight_search=True, failure='Close fixed0.85 screen without trying other weights',
    no_production=True, no_upload=True, caveat='Reused Jan/Jul development; no independent confirmation claim')


def scorer():
    source = textwrap.dedent(inspect.getsource(retained.score))
    tree = ast.parse(source)
    original = ast.parse('.71*.5*(calls+physical)+.29*tree', mode='eval').body
    replacement = ast.parse('(0.85 if name=="more_neural" else 0.71)*.5*(calls+physical)+(0.15 if name=="more_neural" else 0.29)*tree', mode='eval').body

    class AddArm(ast.NodeTransformer):
        arms = weights = 0

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'arms':
                    assert isinstance(node.value, ast.Call) and [k.arg for k in node.value.keywords] == ['selected', 'one_tabm', 'three_tabm']
                    node.value.keywords.append(ast.keyword(arg='more_neural', value=ast.parse('np.mean([new[k] for k in range(3)],axis=0)', mode='eval').body))
                    self.arms += 1
                if node.targets[0].id == 'ordinary':
                    assert ast.dump(node.value) == ast.dump(original)
                    node.value = replacement
                    self.weights += 1
            return self.generic_visit(node)

    edit = AddArm()
    tree = edit.visit(tree)
    assert edit.arms == edit.weights == 1

    class Restore(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'arms':
                    node.value.keywords = [k for k in node.value.keywords if k.arg != 'more_neural']
                if node.targets[0].id == 'ordinary':
                    node.value = original
            return self.generic_visit(node)

    restored = Restore().visit(ast.parse(ast.unparse(tree)))
    assert ast.dump(restored) == ast.dump(ast.parse(source))
    namespace = dict(retained.score.__globals__, OUT=OUT / 'component', DESIGN=DESIGN,
                     ARMS=[*retained.ARMS, 'more_neural'], __file__=__file__)
    exec(compile(ast.fix_missing_locations(tree), '<fixed-neural-share-only>', 'exec'), namespace)
    return namespace['score']


def freeze():
    assert not (OUT / 'identity.json').exists()
    scorer()
    paths = [Path(__file__), Path(retained.__file__), ROOT / 'VALIDATION/tuning_seasonal.py',
             retained.OUT / 'result.json', retained.OUT / 'scored.parquet', retained.OUT / 'audit.json',
             CURRENT / 'result.json', CURRENT / 'scored.parquet']
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'identity.json', dict(design=DESIGN, sources=[art(p) for p in paths],
                                    unchanged_training_and_retained_arms_AST_exact=True, before_scoring=True))


def score():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN
    for a in identity['sources']:
        verify(a)
    assert not (OUT / 'result.json').exists()
    if not (OUT / 'component/result.json').exists():
        scorer()()
    assert read(OUT / 'component/audit.json')['status'] == 'pass'
    old = pd.read_parquet(retained.OUT / 'scored.parquet').set_index('mvt_id')
    new = pd.read_parquet(OUT / 'component/scored.parquet').set_index('mvt_id').loc[old.index]
    parent = pd.read_parquet(CURRENT / 'scored.parquet').set_index('mvt_id').loc[old.index]
    for name in ['month', 'day', 'y', *retained.ARMS]:
        np.testing.assert_array_equal(old[name], new[name])
    for name in ['month', 'day', 'y']:
        np.testing.assert_array_equal(old[name], parent[name])
    f = parent[['month', 'day', 'y']].reset_index()
    f['control'] = parent.three_tabm_runway.to_numpy()
    f['delta'] = new.more_neural.to_numpy() - old.three_tabm.to_numpy()
    f['candidate'] = np.rint(np.maximum(0, f.control + f.delta))
    scores = {m: {c: float(np.sqrt(np.mean((g[c] - g.y) ** 2))) for c in ['control', 'candidate']} for m, g in f.groupby('month')}
    paired = validation.paired_bootstrap(f.day, f.month, (f.control - f.y) ** 2, (f.candidate - f.y) ** 2)
    cv = {c: float(np.mean([r[c] for r in scores.values()])) for c in ['control', 'candidate']}
    np.testing.assert_allclose(cv['control'], 289.7627782102906, atol=1e-8, rtol=0)
    f.to_parquet(OUT / 'scored.parquet', index=False)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('f', OUT / 'scored.parquet'), ('o', retained.OUT / 'scored.parquet'),
                           ('n', OUT / 'component/scored.parquet'), ('c', CURRENT / 'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, u, bad = con.execute('''select count(*),count(distinct f.mvt_id),sum(case when
            f.y is distinct from c.y or f.month is distinct from c.month or f.day is distinct from c.day
            or f.control is distinct from c.three_tabm_runway or f.delta is distinct from n.more_neural-o.three_tabm
            or f.candidate is distinct from round_even(greatest(0,c.three_tabm_runway+n.more_neural-o.three_tabm),0)
            then 1 else 0 end) from f join o using(mvt_id) join n using(mvt_id) join c using(mvt_id)''').fetchone()
        assert n == u == 344419 and bad == 0
        for month, value in con.execute('select month,sqrt(avg(pow(candidate-y,2))) from f group by month').fetchall():
            np.testing.assert_allclose(value, scores[month]['candidate'], atol=1e-8, rtol=0)
    result = dict(identity=art(OUT / 'identity.json'), cv=cv, scores=scores, paired=paired,
        gate=paired['change'] <= -.05 and paired['ci95'][1] < 0 and all(r['candidate'] < r['control'] for r in scores.values()),
        exact_retained_arms_and_v80=True, all_row_SQL_transport_truth_calendar_scores=True,
        scored=art(OUT / 'scored.parquet'), no_production_or_upload=True)
    write(OUT / 'result.json', result)
    print(dict(cv=cv, paired=paired, gate=result['gate']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'score'])
    with threadpool_limits(1):
        globals()[p.parse_args().stage]()
