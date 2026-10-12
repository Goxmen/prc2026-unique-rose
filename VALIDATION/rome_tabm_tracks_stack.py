"""Frozen one-third Rome TabM probability replacements through exact v78."""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_tabm_tracks_jan_jul as experiment
import daily_main_refinements_cv as stack
import joint_public_moments_cv as v78
import tuning_seasonal as ts
from records import write_json

ROOT=experiment.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_rome_tabm_tracks_stack'
art,read,verify=experiment.art,experiment.read,experiment.verify
ARMS=['control',*experiment.ARMS]
DESIGN=dict(parent='exactv78',probability_share=1/3,arms=ARMS,
    main='selected184-input main plus fixed selected airport and CatBoost models',
    neural='unchanged calls and3physical PLE pool',
    corrections='unchanged serialization, constraints, daily-quality mask and .5joint increment',
    primary='physical minus control',diagnostic='tabular minus control and physical minus tabular',
    seed=42,no_search=True,no_upload=True)


def score():
    experiment.guard();assert not (OUT/'result.json').exists()
    daily,joint,rounded=stack.daily,stack.joint,stack.rounded
    neural=joint.neural;replay=neural.cat.r20.physical.replay
    sc,rome,_,_=replay.baseline.baseline();identity=read(neural.single.MANIFEST)
    prior=pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    serialized=pd.read_parquet(rounded.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(stack.OUT/'scored.parquet').set_index('mvt_id')
    moment=pd.read_parquet(v78.OUT/'scored.parquet').set_index('mvt_id')
    quality=pd.read_parquet(stack.quality.OUT/'scored.parquet').set_index('mvt_id')
    raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    source,parameters=daily.public(),read(daily.OUT/'parameters.json')
    context=daily.rows().drop(columns='y').set_index('mvt_id')
    matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    frames=[];cells=[];members=[]
    for month in ts.SPLITS:
        d=sc.d[month];ids=d['ids']
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        physical=np.mean([replay.values(verify(neural.completed(month,k,identity)['prediction']),ids,'base') for k in range(3)],axis=0)
        tree_paths=[verify(stack.main.completed(month)['prediction']),
            rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet',verify(stack.cv.completed(month)['prediction'])]
        tree=np.mean([replay.values(p,ids,'pred') for p in tree_paths],axis=0)
        ordinary=.71*.5*(calls+physical)+.29*tree
        old=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        probabilities={'control':old}
        for arm in experiment.ARMS:
            rp=experiment.OUT/f'{month}_{arm}_predictions.json';rec=read(rp)
            assert rec['all_rome_reload_exact']
            assert rec['full_model']==art(experiment.paths(month,arm,'full')[2])
            member=pd.read_parquet(verify(rec['prediction']))
            np.testing.assert_array_equal(member.mvt_id,ids)
            assert ((member.p>=0)&(member.p<=1)).all()
            assert (member.loc[~raw.loc[ids,'airport'].eq('LIRF').to_numpy(),'p']==0).all()
            probabilities[arm]=(2/3)*old+(1/3)*member.p.to_numpy()
            members.extend([art(rp),rec['prediction']])
        permitted=matches[matches.donor_month.isin(daily.base.FOLDS[month]+[month])]
        outputs={}
        for arm,p in probabilities.items():
            direct=sc.final(month,ordinary,p)
            start=serialized.loc[ids,'candidate_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy()
            before=daily.constraints.matched_bound(raw.loc[ids].reset_index(),start)
            before,_=daily.constraints.exclusion.apply(ids,np.maximum(before,0),permitted)
            before=np.rint(np.maximum(before,0))
            updated,cell,shift=daily.apply(context.loc[ids].reset_index(),before,source,parameters[month])
            gated=np.where(quality.loc[ids,'quality_eligible'],updated,before)
            pred77=daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),gated,matches)
            pred78=daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),
                pred77+.5*moment.loc[ids,'increment'].to_numpy(),matches)
            if arm=='control':
                for got,col in [(before,'combined_before'),(shift,'combined_shift'),(pred77,'combined')]:
                    np.testing.assert_array_equal(got,selected.loc[ids,col])
                np.testing.assert_array_equal(pred78,moment.loc[ids,'candidate'])
            outputs.update({arm:pred78,arm+'_v77':pred77,arm+'_before':before,arm+'_shift':shift,arm+'_p':p})
            cells.append(cell.assign(fold=month,arm=arm))
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],
            airport=raw.loc[ids,'airport'].to_numpy(),increment=moment.loc[ids,'increment'].to_numpy(),**outputs)))
    frame=pd.concat(frames,ignore_index=True);assert len(frame)==344419 and frame.mvt_id.is_unique
    nonrome=~frame.airport.eq('LIRF')
    for arm in experiment.ARMS:np.testing.assert_array_equal(frame.loc[nonrome,arm],frame.loc[nonrome,'control'])
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ARMS} for m,g in frame.groupby('month')}
    comparisons=[('control','physical'),('control','tabular'),('tabular','physical')]
    paired={f'{b}_vs_{a}':ts.paired_bootstrap(frame.day,frame.month,(frame[a]-frame.y)**2,(frame[b]-frame.y)**2)
        for a,b in comparisons}
    OUT.mkdir(parents=True,exist_ok=True);frame.to_parquet(OUT/'scored.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    result=dict(design=DESIGN,identity=art(experiment.MANIFEST),source=art(__file__),members=members,
        scores=scores,cv={c:float(np.mean([s[c] for s in scores.values()])) for c in ARMS},paired=paired,
        exact_v77_and_v78_controls=True,unchanged_outside_rome=True,rows=len(frame),
        scored=art(OUT/'scored.parquet'),artifacts=[art(OUT/'scored.parquet'),art(OUT/'cells.parquet')])
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    # Reuse the exact audited daily/constraint checks for each renamed candidate.
    # SQL below independently checks truth, calendar, probability arithmetic and scores.
    experiment.guard();result=read(OUT/'result.json')
    for item in [*result['artifacts'],*result['members']]:verify(item)
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('f',OUT/'scored.parquet'),('r',stack.daily.base.OUT/'rows2025.parquet'),
            ('c',v78.OUT/'scored.parquet'),('v',stack.OUT/'scored.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''SELECT count(*),count(DISTINCT f.mvt_id),sum(CASE WHEN
          f.y IS DISTINCT FROM r.y OR f.month IS DISTINCT FROM r.month OR f.day IS DISTINCT FROM r.day
          OR f.airport IS DISTINCT FROM r.airport OR f.control IS DISTINCT FROM c.candidate
          OR f.increment IS DISTINCT FROM c.increment OR f.control_v77 IS DISTINCT FROM v.combined
          OR f.control_before IS DISTINCT FROM v.combined_before OR f.control_shift IS DISTINCT FROM v.combined_shift
          THEN 1 ELSE 0 END) FROM f JOIN r USING(mvt_id) JOIN c USING(mvt_id) JOIN v USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for arm in ARMS:
            rows=con.execute(f'SELECT month,sqrt(avg(pow({arm}-y,2))) FROM f GROUP BY month').fetchall()
            for month,value in rows:np.testing.assert_allclose(value,result['scores'][month][arm],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in rows]),result['cv'][arm],atol=1e-8,rtol=0)
    f=pd.read_parquet(OUT/'scored.parquet');ids=f.mvt_id.to_numpy();daily=stack.daily
    raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    quality=pd.read_parquet(stack.quality.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    context=daily.rows().drop(columns='y').set_index('mvt_id');source=daily.public()
    parameters=read(daily.OUT/'parameters.json');matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    for arm in ARMS:
        for month in ts.SPLITS:
            ix=np.flatnonzero(f.month.eq(month))
            _,_,shift=daily.apply(context.loc[ids[ix]].reset_index(),f[arm+'_before'].to_numpy()[ix],source,parameters[month])
            np.testing.assert_array_equal(shift,f[arm+'_shift'].to_numpy()[ix])
            if arm!='control':
                p=pd.read_parquet(experiment.OUT/f'{month}_{arm}_predictions.parquet')
                np.testing.assert_array_equal(p.mvt_id,ids[ix])
                np.testing.assert_array_equal(f[arm+'_p'].to_numpy()[ix],(2/3)*f.control_p.to_numpy()[ix]+(1/3)*p.p.to_numpy())
        pred=f[arm+'_before'].to_numpy()+np.where(quality.quality_eligible,f[arm+'_shift'],0.)
        np.testing.assert_array_equal(daily.constrain(raw,ids,f.month,pred,matches),f[arm+'_v77'])
        np.testing.assert_array_equal(daily.constrain(raw,ids,f.month,f[arm+'_v77'].to_numpy()+.5*f.increment.to_numpy(),matches),f[arm])
    write_json(OUT/'audit.json',dict(status='pass',rows=n,result=art(OUT/'result.json'),identity=art(experiment.MANIFEST),
        original_rows_labels_calendar_and_sql_scores=True,all_parent_controls_exact=True,
        probability_arithmetic_and_final_stages_replayed=True,source=art(__file__)))
    print('All344419 rows, probability arithmetic, exact controls and full correction stages audited',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['score','audit'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
