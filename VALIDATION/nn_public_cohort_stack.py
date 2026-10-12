"""Predeclared exact-v78 replay for the paired-flight context screen.

Compare exact same-desktop seed42 and actual one-of-three pool replacement.
No weight fitting, gate changes, seed selection or new model training.
"""
import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import nn_public_cohort_jan_jul as candidate
import attention_v77_stack as replay_stack
import joint_public_moments_cv as joint
from records import write_json

ROOT=candidate.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_cohort_stack'
ARMS=['parent_single','cohort_single','selected','replace_cohort']
DESIGN=dict(parent='exactv78',primary='replace_cohort versus selected',
    matched='cohort_single versus parent_single on same desktop',
    replacement='one of selected three public/physical members; others1042/2042 unchanged',
    retained='calls15,all3trees,Rome,serialization,public constraints,daily quality and fixed joint increment',
    no_new_weights=True,no_new_gate=True,no_hpo=True,no_upload=True,
    caveat='reused Jan/Jul development; saved same-hardware control and unchanged native2048 updates')
art,verify,read=candidate.art,candidate.verify,candidate.read


def score():
    assert not (OUT/'result.json').exists(),'Preserve completed score'
    identity=read(candidate.MANIFEST)
    assert identity['source']==art(candidate.__file__) and identity['design']==candidate.DESIGN
    for item in identity['files']:verify(item)
    audit=read(candidate.OUT/'audit.json');assert audit['status']=='pass' and audit['identity']==art(candidate.MANIFEST)
    for item in audit['members']:verify(item)
    assert identity['selected_v78']==art(joint.OUT/'result.json')
    verify(identity['selected_v78'])
    current=replay_stack.current;current.guard();joint.guard()
    assert read(current.OUT/'audit.json')['status']==read(joint.OUT/'audit.json')['status']=='pass'
    neural=current.joint.neural;replay=neural.cat.r20.physical.replay
    sc,rome,_,_=replay.baseline.baseline();pool_identity=read(neural.single.MANIFEST)
    prior=pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    serialized=pd.read_parquet(current.rounded.OUT/'scored.parquet').set_index('mvt_id')
    selected77=pd.read_parquet(current.OUT/'scored.parquet').set_index('mvt_id')
    selected78=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id')
    quality=pd.read_parquet(current.quality.OUT/'scored.parquet').set_index('mvt_id')
    raw=pd.read_parquet(current.daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    inputs=current.daily.rows().drop(columns='y').set_index('mvt_id')
    source,parameters=current.daily.public(),read(current.daily.OUT/'parameters.json')
    matches=pd.read_parquet(current.daily.constraints.exclusion.OUT/'matches2025.parquet')
    frames=[];sources=[art(candidate.MANIFEST),art(candidate.OUT/'audit.json'),art(Path(__file__)),
        art(replay_stack.__file__),art(joint.__file__),art(joint.OUT/'result.json'),art(joint.OUT/'scored.parquet')]
    for month in candidate.STOP:
        d=sc.d[month];ids=d['ids'];members=[]
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        for k in range(3):
            path=verify(neural.completed(month,k,pool_identity)['prediction']);sources.append(art(path))
            if k==0:
                assert art(path)==identity['folds'][month]['control_prediction'], 'Selected member0 is not the exact same-desktop seed42 control'
            members.append(replay.values(path,ids,'base'))
        record=candidate.completed(month,identity);assert record['all_held_reloaded_exact']
        path=verify(record['prediction']);sources.append(art(path))
        new=replay.values(path,ids,'base')
        tree_paths=[verify(current.main.completed(month)['prediction']),
            current.rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet',
            verify(current.cv.completed(month)['prediction'])]
        sources.extend(art(path) for path in tree_paths)
        tree=np.mean([replay.values(path,ids,'pred') for path in tree_paths],axis=0)
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        permitted=matches[matches.donor_month.isin(current.daily.base.FOLDS[month]+[month])]
        arms=dict(parent_single=members[0],cohort_single=new,
            selected=np.mean(members,axis=0),replace_cohort=np.mean([new,*members[1:]],axis=0))
        values={}
        for name,physical in arms.items():
            ordinary=.71*.5*(calls+physical)+.29*tree
            direct=sc.final(month,ordinary,probability)
            start=serialized.loc[ids,'candidate_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy()
            before=current.daily.constraints.matched_bound(raw.loc[ids].reset_index(),start)
            before,_=current.daily.constraints.exclusion.apply(ids,np.maximum(before,0),permitted)
            before=np.rint(np.maximum(before,0))
            updated,_,shift=current.daily.apply(inputs.loc[ids].reset_index(),before,source,parameters[month])
            gated=np.where(quality.loc[ids,'quality_eligible'],updated,before)
            v77=current.daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),gated,matches)
            final=current.daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),
                v77+.5*selected78.loc[ids,'increment'].to_numpy(),matches)
            if name=='selected':
                np.testing.assert_array_equal(before,selected77.loc[ids,'combined_before'])
                np.testing.assert_array_equal(shift,selected77.loc[ids,'combined_shift'])
                np.testing.assert_array_equal(v77,selected78.loc[ids,'control'])
                np.testing.assert_array_equal(final,selected78.loc[ids,'candidate'])
            values[name]=final
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**values)))
    f=pd.concat(frames,ignore_index=True);assert len(f)==344419 and f.mvt_id.is_unique
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ARMS} for m,g in f.groupby('month')}
    import tuning_seasonal as ts
    comparisons=[('selected','replace_cohort'),('parent_single','cohort_single')]
    paired={f'{b}_vs_{a}':ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in comparisons}
    OUT.mkdir(parents=True,exist_ok=True);f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,sources=sources,scored=art(OUT/'scored.parquet'),scores=scores,
        cv={c:float(np.mean([v[c] for v in scores.values()])) for c in ARMS},paired=paired,exact_v78_replayed=True)
    write_json(OUT/'result.json',result)
    with duckdb.connect(config={'threads':1}) as con:
        con.read_parquet(str(OUT/'scored.parquet')).create_view('f')
        con.read_parquet(str(current.daily.base.OUT/'rows2025.parquet')).create_view('raw')
        con.read_parquet(str(joint.OUT/'scored.parquet')).create_view('old')
        n,u,bad=con.execute('''select count(*),count(distinct f.mvt_id),sum(case when
            f.y is distinct from raw.y or f.month is distinct from raw.month or f.day is distinct from raw.day
            or f.selected is distinct from old.candidate then1 else0 end)
            from f join raw using(mvt_id) join old using(mvt_id)'''.replace('then1','then 1').replace('else0','else 0')).fetchone()
        assert n==u==344419 and bad==0
        for arm in ARMS:
            rows=con.execute(f'select month,sqrt(avg(pow({arm}-y,2))) from f group by month').fetchall()
            for month,value in rows:np.testing.assert_allclose(value,scores[month][arm],atol=1e-8,rtol=0)
    write_json(OUT/'audit.json',dict(status='pass',rows=n,exact_parent_and_sql_truth_calendar_scores=True,result=art(OUT/'result.json')))
    print(dict(cv=result['cv'],paired=paired),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):score()
