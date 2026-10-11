"""Score both refinement arms with the same integer boundaries as production.

No fit, parameter change or new weight selection. Preserve the inherited
unrounded comparison as a separate reference instead of changing its result.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_refinements_cv as original
from records import write_json

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_public_refinements_submission_cv'
MANIFEST=OUT/'identity.json'
art,read,verify=original.art,original.read,original.verify
DESIGN=dict(parent='v76',change='public airport trees plus t5 daily and full monthly adjustments',
    method='same fixed recipe; round both v74 boundary and final output in both arms',
    inherited_comparison='retained separately; no historical score overwritten',
    no_fits=True,no_parameter_change=True,no_upload=True,
    limitation='Repeatedly reused Jan/Jul development; serialization parity does not create independent evidence')


def guard():
    original.guard();assert read(original.OUT/'audit.json')['status']=='pass'
    paths=[Path(__file__),Path(original.__file__),Path(original.constraints.__file__),
        Path(original.constraints.exclusion.__file__),original.OUT/'scored.parquet',
        original.OUT/'result.json',original.OUT/'audit.json',original.joint.OUT/'scored.parquet',
        original.airport.OUT/'scored.parquet',original.tail.V74/'cv.parquet',
        original.tail.base.OUT/'rows2025.parquet',original.joint.neural.cat.OUT/'scored.parquet',
        original.constraints.exclusion.OUT/'matches2025.parquet']
    value=dict(design=DESIGN,sources=[art(p) for p in paths])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('*.parquet'));write_json(MANIFEST,value)
    return value


def screen():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed score'
    data=pd.read_parquet(original.OUT/'scored.parquet');ids=data.mvt_id.to_numpy()
    direct=pd.read_parquet(original.joint.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    airport=pd.read_parquet(original.airport.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    prior=pd.read_parquet(original.joint.neural.cat.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    raw=pd.read_parquet(original.tail.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    previous=pd.read_parquet(original.tail.V74/'cv.parquet').set_index('mvt_id').loc[ids]
    for f in [direct,airport,prior,raw,previous]:np.testing.assert_array_equal(f.y,data.y)
    np.testing.assert_array_equal(data.control_v74,previous.pred)
    matches=pd.read_parquet(original.constraints.exclusion.OUT/'matches2025.parquet')
    f=data[['mvt_id','month','day','y']].copy()
    f['control_boundary']=np.rint(np.maximum(data.control_v74,0))
    f['candidate_boundary']=np.rint(np.maximum(data.joint_v74,0))
    f['control_unrounded']=data.control.to_numpy()
    f['candidate_unrounded']=data.combined.to_numpy()
    for name,primary in [('control',direct.combined.to_numpy()),('candidate',airport.candidate.to_numpy())]:
        p=original.constraints.matched_bound(raw,f[name+'_boundary'].to_numpy()+primary-prior.control.to_numpy())
        for month,allowed in original.tail.base.FOLDS.items():
            ix=np.flatnonzero(f.month.eq(month))
            permitted=matches[matches.donor_month.isin(allowed+[month])]
            p[ix],_=original.constraints.exclusion.apply(ids[ix],np.maximum(p[ix],0),permitted)
        f[name+'_before_final_rounding']=p
        f[name]=np.rint(np.maximum(p,0))
    assert len(f)==344419 and f.mvt_id.is_unique and np.isfinite(f.iloc[:,4:]).all().all()
    columns=['control','candidate','control_unrounded','candidate_unrounded']
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in f.groupby('month')}
    paired=original.airport.parent.ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.candidate-f.y)**2)
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,identity=art(MANIFEST),prediction=art(OUT/'scored.parquet'),
        scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in columns},paired=paired,
        same_production_rounding_boundaries=True)
    write_json(OUT/'result.json',result);print(result,flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['prediction'])
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('candidate',OUT/'scored.parquet'),('truth',original.tail.base.OUT/'rows2025.parquet'),
            ('previous',original.OUT/'scored.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.control_unrounded is distinct from p.control or c.candidate_unrounded is distinct from p.combined
            or c.control_boundary != round_even(greatest(p.control_v74,0),0)
            or c.candidate_boundary != round_even(greatest(p.joint_v74,0),0)
            or c.control != round_even(greatest(c.control_before_final_rounding,0),0)
            or c.candidate != round_even(greatest(c.candidate_before_final_rounding,0),0)
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join previous p using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for name in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from candidate group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][name],atol=1e-8,rtol=0)
    report=dict(status='pass',rows=n,independent_sql_truth_dates_boundaries_scores=True,
        previous_unrounded_comparison_exact=True,identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__))
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['guard','screen','audit'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
