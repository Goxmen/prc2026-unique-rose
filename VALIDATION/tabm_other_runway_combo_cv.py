"""One post-selection additive check, using saved integer predictions only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'VALIDATION/runs'
OUT=RUNS/'v3_20261012_tabm_other_runway_combo'
P79=RUNS/'v3_20261012_target205_candidate/submission_v79'
P78=RUNS/'v3_20261011_joint_public_moments'
TABM=RUNS/'v3_20261012_nn_public_tabm_stack'
RUNWAY=RUNS/'v3_20261012_joint_public_other_runway'
DESIGN=dict(parent='exactv79',formula='rint(max(0,v79+(TabM_v78-exactv78)+(other_runway_v78-exactv78)))',
    coefficients=[1,1],new_bounds=False,no_fit=True,no_search=True,no_upload=True,
    selection='Post-selection combination of two individually positive directions on reused Jan/Jul data; not independent confirmation.',
    comparisons=['combined versus v79','combined versus TabM-only','combined versus runway-only'])


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as handle:
        for b in iter(lambda:handle.read(2**20),b''):h.update(b)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def verify(a):
    path=ROOT/a['path'];assert art(path)==a;return path
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def combine(parent,control,tabm,runway):
    p,c,t,r=[np.asarray(x,float) for x in [parent,control,tabm,runway]]
    assert p.shape==c.shape==t.shape==r.shape
    assert all(np.isfinite(x).all() and np.equal(x,np.rint(x)).all() for x in [p,c,t,r])
    out=np.rint(np.maximum(0,p+(t-c)+(r-c)))
    np.testing.assert_array_equal(out[(t==c)&(r==c)],p[(t==c)&(r==c)])
    return out


def bootstrap():
    # Compile only the unchanged scalar scorer; do not import training modules.
    path=ROOT/'VALIDATION/tuning_seasonal.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='paired_bootstrap')
    namespace={'np':np,'pd':pd}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['paired_bootstrap']


def freeze():
    assert not (OUT/'identity.json').exists() and not (OUT/'result.json').exists()
    paths=[Path(__file__),ROOT/'VALIDATION/tuning_seasonal.py',P79/'manifest.json',P79/'cv.parquet']
    for directory in [P78,TABM,RUNWAY]:
        result=read(directory/'result.json');audit=read(directory/'audit.json')
        assert audit['status']=='pass' and audit['result']==art(directory/'result.json')
        verify(result['scored']);paths.extend(directory/p for p in ['result.json','audit.json','scored.parquet'])
    independent=[ROOT/'VALIDATION/reports/public_tabm_20261012/score_audit.json',
        ROOT/'VALIDATION/reports/all_discrete_peer_20261012/other_runway_block_audit.json']
    for p in independent:assert read(p)['status']=='pass'
    paths+=independent
    np.testing.assert_array_equal(combine([10,0,7],[4,4,4],[4,1,6],[4,0,7]),[10,0,12])
    write(OUT/'identity.json',dict(design=DESIGN,source=art(__file__),sources=[art(p) for p in paths],
        small_arithmetic_check='unchanged rows, negative sum clipping and both fixed deltas pass'))


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN and identity['source']==art(__file__)
    for a in identity['sources']:verify(a)
    return identity


def score():
    guard();assert not (OUT/'result.json').exists()
    parent=pd.read_parquet(P79/'cv.parquet');ids=parent.mvt_id
    assert len(parent)==344419 and ids.is_unique
    frames=[pd.read_parquet(p/'scored.parquet').set_index('mvt_id') for p in [P78,TABM,RUNWAY]]
    for f in frames:assert len(f)==len(parent) and f.index.is_unique and set(f.index)==set(ids)
    old,tabm,runway=[f.loc[ids] for f in frames]
    for f in [old,tabm,runway]:
        for col in ['y','month','day']:np.testing.assert_array_equal(f[col],parent[col])
    np.testing.assert_array_equal(tabm.selected,old.candidate)
    np.testing.assert_array_equal(runway.control,old.candidate)
    f=parent[['mvt_id','month','day','y']].copy();f['control']=parent.pred
    f['tabm']=combine(parent.pred,old.candidate,tabm.replace_tabm,old.candidate)
    f['runway']=combine(parent.pred,old.candidate,old.candidate,runway.candidate)
    f['combined']=combine(parent.pred,old.candidate,tabm.replace_tabm,runway.candidate)
    columns=['control','tabm','runway','combined']
    scores={m:{c:float(np.sqrt(np.mean((q[c]-q.y)**2))) for c in columns} for m,q in f.groupby('month')}
    paired={c:bootstrap()(f.day,f.month,(f[c]-f.y)**2,(f.combined-f.y)**2) for c in columns[:-1]}
    f.to_parquet(OUT/'scored.parquet',index=False)
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        for name,path in [('f',OUT/'scored.parquet'),('p',P79/'cv.parquet'),('o',P78/'scored.parquet'),
                          ('t',TABM/'scored.parquet'),('r',RUNWAY/'scored.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''SELECT count(*),count(distinct f.mvt_id),sum(CASE WHEN
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day OR
            f.control IS DISTINCT FROM p.pred OR
            f.combined IS DISTINCT FROM round_even(greatest(0,p.pred+(t.replace_tabm-o.candidate)+(r.candidate-o.candidate)),0)
            THEN 1 ELSE 0 END) FROM f JOIN p USING(mvt_id) JOIN o USING(mvt_id) JOIN t USING(mvt_id) JOIN r USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for c in columns:
            for m,value in con.execute(f'SELECT month,sqrt(avg(pow({c}-y,2))) FROM f GROUP BY month').fetchall():
                assert abs(value-scores[m][c])<1e-8
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),scored=art(OUT/'scored.parquet'),scores=scores,
        cv={c:float(np.mean([s[c] for s in scores.values()])) for c in columns},paired=paired,
        rows=n,all_row_sql_arithmetic_truth_calendar_scores='pass')
    write(OUT/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
