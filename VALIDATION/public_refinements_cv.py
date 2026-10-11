"""Combine three completed small gains once, with their original settings.

Public airport trees, t5 ordinary-aggregate weights and full monthly correction.
Reuse all saved models and intermediate arrays. Reconstruct all individual
controls before evaluating the joint recipe; no fit, parameter search or upload.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_heavy_tail_stack as tail
import monthly_strength_stack as monthly
import public_airport_trees as airport
from records import write_json

ROOT=tail.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_public_refinements_cv'
MANIFEST=OUT/'identity.json'
joint,constraints=tail.joint,tail.constraints
art,read,verify=tail.art,tail.read,tail.verify
DESIGN=dict(parent='v76',change='joint public airport/t5/full monthly refinements',
    airport='retained original P04/5000 models with12 public inputs',
    daily='retained t5 replacement and original0.4032872515123992 daily weight',
    monthly='retained full multiplier1 rather than0.5; same fitted parameters and raw cap',
    all_other_parts='v76 components and original matched/recovered plan constraints',
    comparisons=['joint versus v76','joint versus airport only','joint versus aggregate changes'],
    no_fits=True,no_parameter_search=True,no_new_gate=True,
    evaluation='combined after observing individual gains on reused Jan/Jul development',
    no_production=True,no_upload=True)


def guard():
    paths=[Path(__file__),Path(tail.__file__),Path(monthly.__file__),Path(airport.__file__),
        Path(joint.__file__),Path(constraints.__file__),Path(constraints.exclusion.__file__),
        tail.base.OUT/'rows2025.parquet',tail.V74/'cv.parquet',
        joint.neural.cat.OUT/'scored.parquet',constraints.exclusion.OUT/'matches2025.parquet']
    for runner in [tail,monthly,airport,joint]:
        assert read(runner.OUT/'audit.json')['status']=='pass'
        paths.extend([runner.OUT/'scored.parquet',runner.OUT/'result.json',runner.OUT/'audit.json'])
    value=dict(design=DESIGN,sources=[art(p) for p in paths])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('*.parquet'));write_json(MANIFEST,value)
    return value


def screen():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed screen'
    selected=pd.read_parquet(joint.OUT/'scored.parquet');ids=selected.mvt_id.to_numpy()
    raw=pd.read_parquet(tail.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    old74=pd.read_parquet(tail.V74/'cv.parquet').set_index('mvt_id').loc[ids]
    parts=pd.read_parquet(joint.neural.cat.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    a=pd.read_parquet(airport.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    t=pd.read_parquet(tail.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    m=pd.read_parquet(monthly.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    for f in [raw,old74,parts,a,t,m]:np.testing.assert_array_equal(f.y,selected.y)
    matches=pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet')
    def constrain(pred):
        result=constraints.matched_bound(raw,pred)
        for month,allowed in tail.base.FOLDS.items():
            ix=np.flatnonzero(selected.month.eq(month))
            permitted=matches[matches.donor_month.isin(allowed+[month])]
            result[ix],_=constraints.exclusion.apply(ids[ix],np.maximum(result[ix],0),permitted)
        return result
    stages=dict(control_monthly=old74.monthly.to_numpy(),tail_monthly=t.candidate_monthly.to_numpy(),
        full_monthly=m.candidate_monthly.to_numpy(),tail_start=t.candidate_v73.to_numpy())
    stages['joint_monthly']=np.maximum(0,2*stages['tail_monthly']-stages['tail_start'])
    levels={name:constrain(stages[name+'_monthly']) for name in ['control','tail','full','joint']}
    np.testing.assert_array_equal(levels['control'],old74.pred)
    np.testing.assert_array_equal(levels['tail'],t.candidate_v74)
    np.testing.assert_array_equal(levels['full'],m.candidate_v74)
    public=selected.combined.to_numpy();reference=parts.control.to_numpy()
    # Keep the original operation order, including both constraint applications.
    outputs={}
    for name,level,primary in [('control',levels['control'],public),
        ('airport',levels['control'],a.candidate.to_numpy()),
        ('tail',levels['tail'],public),('monthly',levels['full'],public),
        ('aggregate',levels['joint'],public),('combined',levels['joint'],a.candidate.to_numpy())]:
        outputs[name]=constrain(level+primary-reference)
    for name,expected in [('control',selected.combined_transfer),('airport',a.candidate_transfer),
        ('tail',t.candidate),('monthly',m.candidate)]:np.testing.assert_array_equal(outputs[name],expected)
    frame=selected[['mvt_id','month','day','y']].copy()
    for name,value in dict(**stages,**{k+'_v74':v for k,v in levels.items()},**outputs).items():frame[name]=value
    assert len(frame)==344419 and frame.mvt_id.is_unique and np.isfinite(frame[list(outputs)]).all().all()
    scores={month:{name:float(np.sqrt(np.mean((g[name]-g.y)**2))) for name in outputs} for month,g in frame.groupby('month')}
    pairs=[(name,'combined') for name in ['control','airport','aggregate']]
    paired={f'{b}_vs_{a}':airport.parent.ts.paired_bootstrap(frame.day,frame.month,
        (frame[a]-frame.y)**2,(frame[b]-frame.y)**2) for a,b in pairs}
    frame.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,identity=art(MANIFEST),predictions=art(OUT/'scored.parquet'),
        scores=scores,cv={name:float(np.mean([v[name] for v in scores.values()])) for name in outputs},
        paired=paired,all_four_individual_controls_exact=True)
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['predictions'])
    with duckdb.connect(config={'threads':1}) as con:
        sources=[('candidate',OUT/'scored.parquet'),('truth',tail.base.OUT/'rows2025.parquet'),
            ('v76',joint.OUT/'scored.parquet'),('airport',airport.OUT/'scored.parquet'),
            ('tail',tail.OUT/'scored.parquet'),('monthly',monthly.OUT/'scored.parquet')]
        for name,path in sources:con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.month is distinct from r.month or c.day is distinct from r.day
            or c.control is distinct from v.combined_transfer or c.airport is distinct from a.candidate_transfer
            or c.tail is distinct from t.candidate or c.monthly is distinct from m.candidate
            or c.tail_start is distinct from t.candidate_v73 or c.tail_monthly is distinct from t.candidate_monthly
            or c.joint_monthly is distinct from greatest(0,2*t.candidate_monthly-t.candidate_v73)
            then 1 else 0 end) from candidate c join truth r using(mvt_id)
            join v76 v using(mvt_id) join airport a using(mvt_id)
            join tail t using(mvt_id) join monthly m using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for name in result['cv']:
            rows=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from candidate group by month').fetchall()
            for month,value in rows:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in rows]),result['cv'][name],atol=1e-8,rtol=0)
    report=dict(status='pass',rows=n,independent_sql_truth_calendar_four_controls_scores=True,
        joint_monthly_arithmetic_exact=True,all_later_constraints_replayed=True,
        identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__))
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['guard','screen','audit'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
