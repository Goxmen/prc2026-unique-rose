"""Fixed attention-member comparison through the exact v77 final stack.

Declared before the two new fits finish. Reuses all other models and recomputes
both serialized constraint boundaries and the selected daily quality correction.
No fitted coefficients, gate changes, seed selection or upload.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_main_refinements_cv as current
import nn_public_attention_jan_jul as candidate
from records import sha256,write_json

ROOT=current.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_attention_v77_stack'
DESIGN=dict(parent='exact selected v77',change='replace physical/public seed42 with learned attention pooling network',
    primary='one-of-three replacement versus selected pool',secondary='matched seed42 versus seed42',
    retained='all other networks, neural/tree shares, trees, Rome and every final correction',
    no_fit=True,no_weight_scan=True,no_new_gate=True,no_upload=True,
    evidence='extensively reused Jan/Jul development; not independent confirmation')


def art(path):
    path=Path(path).resolve();return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def read(path):return current.read(path)


def verify(record):
    path=ROOT/record['path'];assert sha256(path)==record['sha256'];return path


def guard():
    assert read(current.OUT/'audit.json')['status']=='pass'
    assert read(candidate.OUT/'score_audit.json')['status']=='pass'
    current.guard();new_identity=candidate.guard()
    paths=[Path(__file__),Path(current.__file__),Path(candidate.__file__),
        current.OUT/'identity.json',current.OUT/'result.json',current.OUT/'audit.json',
        current.OUT/'scored.parquet',current.joint.neural.MANIFEST,current.joint.neural.OUT/'audit.json',
        candidate.MANIFEST,candidate.OUT/'audit.json',candidate.OUT/'score_audit.json']
    for m in current.daily.base.FOLDS:paths.append(verify(candidate.completed(m,new_identity)['prediction']))
    value=dict(design=DESIGN,sources=[art(p) for p in paths])
    path=OUT/'identity.json';OUT.mkdir(parents=True,exist_ok=True)
    if path.exists():assert read(path)==value
    else:write_json(path,value)


def score():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed score'
    neural=current.joint.neural;replay=neural.cat.r20.physical.replay
    sc,rome,_,_=replay.baseline.baseline();pool_identity=read(neural.single.MANIFEST)
    new_identity=read(candidate.MANIFEST)
    prior=pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    serialized=pd.read_parquet(current.rounded.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(current.OUT/'scored.parquet').set_index('mvt_id')
    quality=pd.read_parquet(current.quality.OUT/'scored.parquet').set_index('mvt_id')
    raw=pd.read_parquet(current.daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    inputs=current.daily.rows().drop(columns='y').set_index('mvt_id')
    source,parameters=current.daily.public(),read(current.daily.OUT/'parameters.json')
    matches=pd.read_parquet(current.daily.constraints.exclusion.OUT/'matches2025.parquet')
    frames=[];sources=[]
    for month in current.daily.base.FOLDS:
        d=sc.d[month];ids=d['ids'];members=[]
        path=replay.combo.sequence.CALLS_PRED;sources.append(art(path))
        calls=replay.values(path,ids,'candidate_nn')
        for k in range(3):
            path=verify(neural.completed(month,k,pool_identity)['prediction']);sources.append(art(path))
            members.append(replay.values(path,ids,'base'))
        path=verify(candidate.completed(month,new_identity)['prediction']);sources.append(art(path))
        new=replay.values(path,ids,'base')
        tree_paths=[verify(current.main.completed(month)['prediction']),
            current.rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet',
            verify(current.cv.completed(month)['prediction'])]
        sources.extend(art(path) for path in tree_paths)
        tree=np.mean([replay.values(path,ids,'pred') for path in tree_paths],axis=0)
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        permitted=matches[matches.donor_month.isin(current.daily.base.FOLDS[month]+[month])]
        values={}
        arms={'control':members[0],'candidate':new,'selected':np.mean(members,axis=0),
            'replace_seed42':np.mean([new,*members[1:]],axis=0)}
        for name,physical in arms.items():
            ordinary=.71*.5*(calls+physical)+.29*tree
            direct=sc.final(month,ordinary,probability)
            start=serialized.loc[ids,'candidate_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy()
            before=current.daily.constraints.matched_bound(raw.loc[ids].reset_index(),start)
            before,_=current.daily.constraints.exclusion.apply(ids,np.maximum(before,0),permitted)
            before=np.rint(np.maximum(before,0))
            updated,_,shift=current.daily.apply(inputs.loc[ids].reset_index(),before,source,parameters[month])
            gated=np.where(quality.loc[ids,'quality_eligible'],updated,before)
            final=current.daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),gated,matches)
            if name=='selected':
                np.testing.assert_array_equal(before,selected.loc[ids,'combined_before'])
                np.testing.assert_array_equal(shift,selected.loc[ids,'combined_shift'])
                np.testing.assert_array_equal(final,selected.loc[ids,'combined'])
            values[name+'_before'],values[name+'_shift'],values[name]=before,shift,final
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**values)))
    f=pd.concat(frames,ignore_index=True);assert len(f)==344419 and f.mvt_id.is_unique
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in arms} for m,g in f.groupby('month')}
    import tuning_seasonal as ts
    paired={f'{b}_vs_{a}':ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2)
        for a,b in [('control','candidate'),('selected','replace_seed42')]}
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),prediction=art(OUT/'scored.parquet'),
        sources=sources,scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in arms},
        paired=paired,all_selected_parent_stages_exact=True)
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['prediction'])
    for source in result['sources']:verify(source)
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('p',OUT/'scored.parquet'),('r',current.daily.base.OUT/'rows2025.parquet'),
            ('c',current.OUT/'scored.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct p.mvt_id),sum(case when
            p.y is distinct from r.y or p.month is distinct from r.month or p.day is distinct from r.day
            or p.selected is distinct from c.combined or p.selected_before is distinct from c.combined_before
            or p.selected_shift is distinct from c.combined_shift then 1 else 0 end)
            from p join r using(mvt_id) join c using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for name in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from p group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in values]),result['cv'][name],atol=1e-8,rtol=0)
    f=pd.read_parquet(OUT/'scored.parquet');ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(current.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    inputs=current.daily.rows().drop(columns='y').set_index('mvt_id')
    quality=pd.read_parquet(current.quality.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    source,parameters=current.daily.public(),read(current.daily.OUT/'parameters.json')
    matches=pd.read_parquet(current.daily.constraints.exclusion.OUT/'matches2025.parquet')
    for arm in result['cv']:
        for month in current.daily.base.FOLDS:
            ix=np.flatnonzero(f.month.eq(month))
            _,_,shift=current.daily.apply(inputs.loc[ids[ix]].reset_index(),f[arm+'_before'].to_numpy()[ix],source,parameters[month])
            np.testing.assert_array_equal(shift,f[arm+'_shift'].to_numpy()[ix])
        value=f[arm+'_before'].to_numpy()+np.where(quality.quality_eligible,f[arm+'_shift'],0.)
        np.testing.assert_array_equal(current.daily.constrain(raw,ids,f.month,value,matches),f[arm])
    write_json(OUT/'audit.json',dict(status='pass',rows=n,identity=art(OUT/'identity.json'),
        result=art(OUT/'result.json'),source=art(__file__),
        all_raw_truth_calendar_parent_stages_and_sql_scores=True,all_daily_shifts_and_constraints_replayed=True))
    print('All-row raw-truth/control/score and final-correction audits passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['score','audit']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
