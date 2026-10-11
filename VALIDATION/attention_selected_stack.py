"""Carry all fixed attention comparisons through the exact selected v78 update.

V78's joint-source prior is v70, independent of the replaced neural member.
Its retained fixed increment therefore stays unchanged in all four arms.
Declared before attention training starts; no new fit or weight selection.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import attention_v77_stack as base
import joint_public_moments_cv as joint

ROOT=base.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_attention_selected_stack'
ARMS=['control','candidate','selected','replace_seed42']
art,read,verify=base.art,base.read,base.verify
DESIGN=dict(parent='exact selected v78',change='fixed single and one-of-three attention member replacement',
    primary='one-of-three replacement versus selected pool',secondary='matched seed42 comparison',
    joint_update='same v70-based joint public increment, unchanged0.5 weight and final constraints',
    no_fit=True,no_weight_scan=True,no_new_gate=True,no_upload=True,
    evidence='reused Jan/Jul development months; no independent confirmation')


def guard():
    assert read(base.OUT/'audit.json')['status']=='pass' and read(joint.OUT/'audit.json')['status']=='pass'
    joint.guard()
    paths=[Path(__file__),Path(base.__file__),Path(joint.__file__),base.OUT/'result.json',base.OUT/'scored.parquet',
           base.OUT/'audit.json',joint.OUT/'scored.parquet',joint.OUT/'result.json',joint.OUT/'audit.json']
    identity=dict(design=DESIGN,sources=[art(p) for p in paths]);OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'identity.json'
    if path.exists():assert read(path)==identity
    else:joint.write(path,identity)


def score():
    guard();assert not (OUT/'result.json').exists()
    frame=pd.read_parquet(base.OUT/'scored.parquet');ids=frame.mvt_id.to_numpy()
    old=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    np.testing.assert_array_equal(frame.selected,old.control)
    daily=base.current.daily;raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    f=frame[['mvt_id','month','day','y']].copy();f['increment']=old.increment.to_numpy()
    for arm in ARMS:
        f[arm+'_before']=frame[arm]
        f[arm]=daily.constrain(raw,ids,f.month,frame[arm].to_numpy()+.5*f.increment.to_numpy(),matches)
    np.testing.assert_array_equal(f.selected,old.candidate)
    f.to_parquet(OUT/'scored.parquet',index=False)
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ARMS} for m,g in f.groupby('month')}
    import tuning_seasonal as ts
    paired={f'{b}_vs_{a}':ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2)
            for a,b in [('control','candidate'),('selected','replace_seed42')]}
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),scored=art(OUT/'scored.parquet'),scores=scores,
        cv={c:float(np.mean([r[c] for r in scores.values()])) for c in ARMS},paired=paired,exact_v78_parent_replayed=True)
    joint.write(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists();result=read(OUT/'result.json');verify(result['scored'])
    daily=base.current.daily
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('f',OUT/'scored.parquet'),('old',joint.OUT/'scored.parquet'),
            ('b',base.OUT/'scored.parquet'),('raw',daily.base.OUT/'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        comparisons=' or '.join(f'f.{arm}_before is distinct from b.{arm}' for arm in ARMS)
        n,u,bad=con.execute(f'''select count(*),count(distinct f.mvt_id),sum(case when
            f.y is distinct from raw.y or f.month is distinct from raw.month or f.day is distinct from raw.day
            or f.selected is distinct from old.candidate or f.increment is distinct from old.increment
            or {comparisons} then 1 else 0 end)
            from f join raw using(mvt_id) join old using(mvt_id) join b using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for arm in ARMS:
            scores=con.execute(f'select month,sqrt(avg(pow({arm}-y,2))) from f group by month').fetchall()
            for month,s in scores:np.testing.assert_allclose(s,result['scores'][month][arm],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([s for _,s in scores]),result['cv'][arm],atol=1e-8,rtol=0)
    f=pd.read_parquet(OUT/'scored.parquet');ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    for arm in ARMS:np.testing.assert_array_equal(daily.constrain(raw,ids,f.month,
        f[arm+'_before'].to_numpy()+.5*f.increment.to_numpy(),matches),f[arm])
    joint.write(OUT/'audit.json',dict(status='pass',identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        rows=n,raw_truth_calendar_parent_increment_sql_scores_and_constraints=True,source=art(__file__)))
    print(dict(audit='pass',rows=n),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['score','audit'])
    with threadpool_limits(1):globals()[p.parse_args().stage]()
