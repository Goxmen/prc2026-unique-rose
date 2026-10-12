"""Freeze/replay a single deeper TabM member in the exact selected v80 stack."""
import argparse,ast,inspect,textwrap,types
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import nn_public_tabm_depth3 as candidate
import nn_public_tabm_pool3_stack as previous
import tuning_seasonal as ts

ROOT=candidate.ROOT
OUT=candidate.OUT/'comparison'
read,art,verify,write=candidate.read,candidate.art,candidate.verify,candidate.write
DESIGN=dict(candidate.DESIGN,formula='integer v80 plus new-minus-old serialized TabM v78 component, then one nonnegative clip and rounding',
    reference='v80 retains three selected TabM seeds and same-date runway correction',
    guard='Every original one-TabM/v78 control, raw truth, current v80 and component source must remain exact')


def inherited_score():
    def completed(month,k,identity):
        if k==0:return candidate.completed(month,k,identity)
        return candidate.parent.completed(month,k,read(candidate.parent.MANIFEST))
    proxy=types.SimpleNamespace(**vars(candidate));proxy.completed=completed;proxy.STOP=candidate.parent.STOP
    def retained_seed42(month,ids):
        p=candidate.parent.paths(month,0)[1]
        f=pd.read_parquet(p).set_index('mvt_id');assert f.index.is_unique
        return f.loc[ids,'base'].to_numpy(float)
    source=textwrap.dedent(inspect.getsource(previous.score));tree=ast.parse(source)
    class OldControl(ast.NodeTransformer):
        count=0
        def visit_keyword(self,node):
            if node.arg=='one_tabm':
                expected=ast.parse('np.mean([new[0], *members[1:]], axis=0)',mode='eval').body
                assert ast.dump(node.value)==ast.dump(expected)
                node.value=ast.parse('np.mean([retained_seed42(month,ids), *members[1:]], axis=0)',mode='eval').body
                self.count+=1
            return self.generic_visit(node)
    edit=OldControl();tree=edit.visit(tree);assert edit.count==1
    restore=ast.unparse(tree).replace('retained_seed42(month, ids)','new[0]')
    assert ast.dump(ast.parse(restore))==ast.dump(ast.parse(source))
    ns=dict(previous.score.__globals__,candidate=proxy,OUT=OUT/'component',DESIGN=DESIGN,
        retained_seed42=retained_seed42,__file__=__file__)
    exec(compile(ast.fix_missing_locations(tree),'<depth3-stack-control-only>','exec'),ns)
    return ns['score']


def freeze():
    assert not (OUT/'identity.json').exists()
    inherited_score()
    # Selection and exact comparison implementation must predate training.
    assert not any((candidate.OUT/'k0/models').glob('*.pt'))
    parts=[Path(__file__),Path(previous.__file__),candidate.MANIFEST,
        previous.OUT/'scored.parquet',previous.OUT/'result.json',previous.OUT/'audit.json',
        ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo/scored.parquet',
        ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo/result.json']
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'identity.json',dict(design=DESIGN,sources=[art(p) for p in parts],before_fitting=True))
    print('Exact v80 comparison and unmodified correction replay frozen',flush=True)


def score():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for a in identity['sources']:verify(a)
    assert not (OUT/'result.json').exists()
    if not (OUT/'component/result.json').exists():inherited_score()()
    assert read(OUT/'component/audit.json')['status']=='pass'
    original=pd.read_parquet(previous.OUT/'scored.parquet').set_index('mvt_id')
    deeper=pd.read_parquet(OUT/'component/scored.parquet').set_index('mvt_id').loc[original.index]
    current=ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
    frame=pd.read_parquet(current/'scored.parquet').set_index('mvt_id').loc[original.index]
    for col in ['month','day','y']:
        np.testing.assert_array_equal(original[col],deeper[col]);np.testing.assert_array_equal(frame[col],original[col])
    f=frame[['month','day','y']].reset_index();f['control']=frame.three_tabm_runway.to_numpy()
    f['component_delta']=deeper.three_tabm.to_numpy()-original.three_tabm.to_numpy()
    f['candidate']=np.rint(np.maximum(f.control+f.component_delta,0))
    assert len(f)==f.mvt_id.nunique()==344419
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['control','candidate']} for m,g in f.groupby('month')}
    paired=ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.candidate-f.y)**2)
    np.testing.assert_allclose(np.mean([s['control'] for s in scores.values()]),289.7627782102906,atol=1e-8,rtol=0)
    f.to_parquet(OUT/'scored.parquet',index=False)
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('f',OUT/'scored.parquet'),('o',previous.OUT/'scored.parquet'),('d',OUT/'component/scored.parquet'),('c',current/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        assert con.execute('''select count(*) from f full join o using(mvt_id) full join d using(mvt_id) full join c using(mvt_id)
            where f.control is distinct from c.three_tabm_runway or f.y is distinct from o.y
            or f.month is distinct from o.month or f.day is distinct from o.day
            or f.candidate is distinct from round_even(greatest(0,c.three_tabm_runway+d.three_tabm-o.three_tabm),0)''').fetchone()[0]==0
        for month,rmse in con.execute('select month,sqrt(avg(pow(candidate-y,2))) from f group by month').fetchall():
            np.testing.assert_allclose(rmse,scores[month]['candidate'],atol=1e-8,rtol=0)
    value=dict(identity=art(OUT/'identity.json'),scores=scores,paired=paired,
        cv={c:float(np.mean([s[c] for s in scores.values()])) for c in ['control','candidate']},
        gate=paired['ci95'][1]<0 and all(s['candidate']<s['control'] for s in scores.values()),
        exact_v80_replay=True,all_row_independent_SQL_exact=True,scored=art(OUT/'scored.parquet'),
        no_production_or_upload=True)
    write(OUT/'result.json',value);print(value,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):globals()[ap.parse_args().stage]()
