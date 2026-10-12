"""Fixed184+2 peer-clock main-LightGBM comparison through the exactv78 stack."""
import argparse
import gc
from pathlib import Path
import platform
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import flight_id_clock_features as features
import public_daily_taxi_lgb as base
import daily_main_refinements_cv as stack
import joint_public_moments_cv as v78
from records import write_json

ROOT=features.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_flight_id_clock_lgb'
MANIFEST=OUT/'identity.json'
art,read,verify=base.art,base.read,base.verify
DESIGN=dict(parent='exact v78; selected184-input main LightGBM retained control',
    change='two within-month leave-own-flight-ID-out IOBT peer-clock features in main LightGBM only',
    features=features.FEATURES,original_columns=184,columns=186,seed=42,trial='LB1',rounds=10000,threads=4,
    rows_labels='all original ordinary training rows and targets; every held row retained',
    settings='exact retained184-input parameters/categories and same desktop machine',
    primary='full main-tree replacement inside unchanged equal3-tree mean through exactv78',
    secondary='separate fixed coefficient-one delta transport onto exactv79',
    v78_increment='unchanged retained joint-minus-departure-only correction based on originalv70 prior',
    no_feature_search=True,no_weight_search=True,no_stopping_search=True,no_raw_id_feature=True,
    limitations=['reused Jan/Jul development','input-only source lead selected after chronology study',
        'mostly redundant with supplied IOBT; deviations do not prove timestamp errors',
        'no-ID Rome rows cannot benefit directly from the new clock estimate'],
    no_production=True,no_upload=True)


def guard():
    assert read(features.OUT/'audit.json')['status']=='pass'
    fm=read(features.OUT/'manifest.json')
    assert fm['design']==features.DESIGN
    verify(fm['source'])
    for p in [base.OUT/'audit.json',stack.OUT/'audit.json',v78.OUT/'audit.json']:
        assert read(p)['status']=='pass'
    paths=[Path(__file__),ROOT/'VALIDATION/source_clock_phase_lgb.py',Path(features.__file__),Path(base.__file__),Path(stack.__file__),Path(v78.__file__),
        Path(base.parent.__file__),Path(base.main.__file__),Path(stack.daily.__file__),
        Path(stack.daily.constraints.__file__),Path(stack.daily.constraints.exclusion.__file__),
        Path(stack.joint.neural.__file__),ROOT/'VALIDATION/modeling.py',ROOT/'VALIDATION/records.py',
        features.OUT/'manifest.json',features.OUT/'audit.json',features.OUT/'features_training.parquet',
        base.OUT/'identity.json',base.OUT/'audit.json',base.OUT/'result.json',
        stack.OUT/'identity.json',stack.OUT/'audit.json',stack.OUT/'scored.parquet',
        v78.OUT/'identity.json',v78.OUT/'audit.json',v78.OUT/'scored.parquet',v78.OUT/'result.json',
        base.main.features.OUT/'features2025.parquet',base.features.OUT/'features2025.parquet',
        base.main.features.base.OUT/'rows2025.parquet',stack.daily.CACHE,
        stack.daily.SOURCE/'daily_runway.parquet',stack.daily.OUT/'parameters.json',
        stack.daily.constraints.exclusion.OUT/'matches2025.parquet',
        stack.quality.OUT/'scored.parquet',stack.rounded.OUT/'scored.parquet',
        stack.joint.neural.cat.OUT/'scored.parquet']
    for month in base.parent.ts.SPLITS:
        paths.extend([base.parent.input_cache.input_path(month),
            base.OUT/'predictions'/f'{month}_s42.json',base.OUT/'predictions'/f'{month}_s42.parquet',
            base.OUT/'models'/f'{month}_s42.txt',
            verify(stack.cv.completed(month)['prediction']),
            stack.rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet'])
    identity=read(stack.joint.neural.single.MANIFEST)
    for month in base.parent.ts.SPLITS:
        for k in range(3):paths.append(verify(stack.joint.neural.completed(month,k,identity)['prediction']))
    replay=stack.joint.neural.cat.r20.physical.replay
    paths.append(replay.combo.sequence.CALLS_PRED)
    # Preserve the complete existing input/label/model provenance, not merely new hashes.
    for original in [base.OUT/'identity.json',stack.OUT/'identity.json',v78.OUT/'identity.json']:
        for a in read(original).get('sources',[]):verify(a)
    value=dict(design=DESIGN,sources=[art(p) for p in dict.fromkeys(paths)])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('**/*.txt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST,value)
    return value


def append_clocks(frame,ids,phase):
    assert len(frame)==len(ids) and phase.index.is_unique and np.isin(ids,phase.index).all()
    new=pd.concat([frame,phase.loc[ids,features.FEATURES].set_axis(frame.index)],axis=1)
    assert list(new)==list(frame)+features.FEATURES
    pd.testing.assert_frame_equal(new[frame.columns],frame,check_exact=True)
    return new


def inputs(month,tables,phase):
    d=base.inputs(month,*tables)
    control=base.completed(month);assert control is not None
    assert len(control['columns'])==184 and control['rounds']==10000
    assert control['params']['num_threads']==4 and control['params']['seed']==42
    assert control['machine'].lower()==platform.node().lower(),'Keep same-machine comparison'
    for name,ids in [('X',d['train_ids']),('V',d['ids'])]:
        assert list(d[name])==control['columns']
        d[name]=append_clocks(d[name],ids,phase)
        assert len(d[name].columns)==186
    d['control']=control
    return d


def preflight():
    guard();assert not (OUT/'preflight.json').exists(),'Preserve completed preflight'
    tables=base.tables();phase=pd.read_parquet(features.OUT/'features_training.parquet').set_index('mvt_id')
    checks=[]
    for month in base.parent.ts.SPLITS:
        d=inputs(month,tables,phase)
        checks.append(dict(month=month,train_rows=len(d['X']),held_rows=len(d['V']),
            exact_original184=True,exact_original_targets=True,
            columns=list(d['X']),train_months=base.parent.ts.FOLDS[month],
            train_ids_sha256=__import__('hashlib').sha256(np.asarray(d['train_ids']).tobytes()).hexdigest(),
            held_ids_sha256=__import__('hashlib').sha256(np.asarray(d['ids']).tobytes()).hexdigest(),
            new_missing={c:int(d['V'][c].isna().sum()) for c in features.FEATURES}))
        del d;gc.collect()
    write_json(OUT/'preflight.json',dict(status='pass',identity=art(MANIFEST),checks=checks))
    print('Both186-column input views preserve184 columns, categories, memberships and targets',flush=True)


def completed(month):
    pp=OUT/'predictions'/f'{month}_s42.parquet';rp=pp.with_suffix('.json')
    if not rp.exists():
        assert not pp.exists(),'Inspect unrecorded prediction instead of refitting'
        return None
    rec=read(rp);assert rec['identity']==art(MANIFEST)
    for key in ['prediction','model','staged_model']:verify(rec[key])
    assert rec['all_control_and_new_held_reload_exact']
    return rec


def fit():
    guard();assert read(OUT/'preflight.json')['identity']==art(MANIFEST)
    tables=base.tables();phase=pd.read_parquet(features.OUT/'features_training.parquet').set_index('mvt_id')
    for name in ['models','predictions']:(OUT/name).mkdir(exist_ok=True)
    for month in base.parent.ts.SPLITS:
        if completed(month):continue
        d=inputs(month,tables,phase);X,V=d['X'],d['V'];control=d['control']
        old=lgb.Booster(model_file=str(verify(control['model'])))
        prediction=np.maximum(old.predict(V[control['columns']],num_threads=4),0).astype('float32')
        expected=pd.read_parquet(verify(control['prediction']))
        np.testing.assert_array_equal(expected.mvt_id,d['ids']);np.testing.assert_array_equal(prediction,expected.pred)
        del old,prediction,expected
        mp=OUT/'models'/f'{month}_s42.txt';sp=mp.with_suffix('.json')
        pp=OUT/'predictions'/f'{month}_s42.parquet'
        started=time.monotonic();recovery=sp.exists()
        if recovery:
            staged=read(sp);assert staged['identity']==art(MANIFEST) and staged['month']==month
            assert staged['params']==control['params'] and staged['columns']==list(X)
            assert staged['train_rows']==len(X) and staged['held_rows']==len(V)
            model=lgb.Booster(model_file=str(verify(staged['model'])))
        else:
            assert not mp.exists(),'Inspect unrecorded model; do not repeat fitting'
            print(month,len(X),'rows,186 inputs,LB1/10000; original184 reload exact',flush=True)
            dataset=lgb.Dataset(X,d['y'],categorical_feature=control['categoricals'])
            model=lgb.train(control['params'],dataset,num_boost_round=10000,
                callbacks=[lambda env:print(month,'iteration',env.iteration+1,'seconds',round(time.monotonic()-started,1),flush=True)
                           if (env.iteration+1)%1000==0 else None])
            assert model.current_iteration()==10000
            model.save_model(str(mp))
            write_json(sp,dict(identity=art(MANIFEST),month=month,params=control['params'],rounds=10000,
                columns=list(X),categoricals=control['categoricals'],train_rows=len(X),held_rows=len(V),
                model=art(mp),model_saved_before_held_inference=True,machine=platform.node(),
                fit_seconds=time.monotonic()-started))
            del dataset
        prediction=base.parent.modeling.predict(model,V,threads=4).astype('float32')
        assert np.isfinite(prediction).all() and (prediction>=0).all()
        restored=lgb.Booster(model_file=str(mp))
        np.testing.assert_array_equal(prediction,np.maximum(restored.predict(V,num_threads=4),0).astype('float32'))
        pd.DataFrame(dict(mvt_id=d['ids'],pred=prediction)).to_parquet(pp,index=False)
        write_json(pp.with_suffix('.json'),dict(identity=art(MANIFEST),model=art(mp),prediction=art(pp),staged_model=art(sp),
            params=control['params'],rounds=10000,columns=list(X),categoricals=control['categoricals'],
            train_rows=len(X),held_rows=len(V),all_control_and_new_held_reload_exact=True,
            resumed_saved_checkpoint_without_refit=recovery,machine=platform.node(),minutes=(time.monotonic()-started)/60))
        del d,X,V,model,restored;gc.collect()


def score():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed score'
    daily,joint,rounded=stack.daily,stack.joint,stack.rounded
    neural=joint.neural;replay=neural.cat.r20.physical.replay
    sc,rome,_,trees=replay.baseline.baseline();identity=read(neural.single.MANIFEST)
    prior=pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    serialized=pd.read_parquet(rounded.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(stack.OUT/'scored.parquet').set_index('mvt_id')
    moment=pd.read_parquet(v78.OUT/'scored.parquet').set_index('mvt_id')
    quality=pd.read_parquet(stack.quality.OUT/'scored.parquet').set_index('mvt_id')
    raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    source,parameters=daily.public(),read(daily.OUT/'parameters.json')
    context=daily.rows().drop(columns='y').set_index('mvt_id')
    matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet');frames=[];cells=[]
    for month in base.parent.ts.SPLITS:
        d=sc.d[month];ids=d['ids'];rec=completed(month);assert rec is not None
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        physical=np.mean([replay.values(verify(neural.completed(month,k,identity)['prediction']),ids,'base') for k in range(3)],axis=0)
        old_main=replay.values(verify(base.completed(month)['prediction']),ids,'pred')
        new_main=replay.values(verify(rec['prediction']),ids,'pred')
        airport=replay.values(rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        cat=replay.values(verify(stack.cv.completed(month)['prediction']),ids,'pred')
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        permitted=matches[matches.donor_month.isin(daily.base.FOLDS[month]+[month])]
        outputs={}
        for arm,main_prediction in [('control',old_main),('candidate',new_main)]:
            ordinary=.71*.5*(calls+physical)+.29*np.mean([main_prediction,airport,cat],axis=0)
            direct=sc.final(month,ordinary,probability)
            start=serialized.loc[ids,'candidate_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy()
            refined=daily.constraints.matched_bound(raw.loc[ids].reset_index(),start)
            refined,_=daily.constraints.exclusion.apply(ids,np.maximum(refined,0),permitted)
            refined=np.rint(np.maximum(refined,0))
            updated,cell,shift=daily.apply(context.loc[ids].reset_index(),refined,source,parameters[month])
            gated=np.where(quality.loc[ids,'quality_eligible'],updated,refined)
            pred77=daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),gated,matches)
            pred78=daily.constrain(raw.loc[ids].reset_index(),ids,pd.Series(month,index=range(len(ids))),
                pred77+.5*moment.loc[ids,'increment'].to_numpy(),matches)
            if arm=='control':
                for got,col in [(refined,'combined_before'),(shift,'combined_shift'),(pred77,'combined')]:
                    np.testing.assert_array_equal(got,selected.loc[ids,col])
                np.testing.assert_array_equal(pred77,moment.loc[ids,'control'])
                np.testing.assert_array_equal(pred78,moment.loc[ids,'candidate'])
            outputs.update({arm:pred78,arm+'_v77':pred77,arm+'_before':refined,arm+'_shift':shift})
            cells.append(cell.assign(fold=month,arm=arm))
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],
            airport=raw.loc[ids,'airport'].to_numpy(),increment=moment.loc[ids,'increment'].to_numpy(),**outputs)))
    frame=pd.concat(frames,ignore_index=True);assert len(frame)==344419 and frame.mvt_id.is_unique
    columns=['control','candidate','control_v77','candidate_v77']
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in frame.groupby('month')}
    paired={name:base.parent.ts.paired_bootstrap(frame.day,frame.month,(frame[a]-frame.y)**2,(frame[b]-frame.y)**2)
            for name,a,b in [('primary','control','candidate'),('v77','control_v77','candidate_v77')]}
    frame.to_parquet(OUT/'scored.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    result=dict(design=DESIGN,identity=art(MANIFEST),scores=scores,
        cv={c:float(np.mean([s[c] for s in scores.values()])) for c in columns},paired=paired,
        exact_v77_and_v78_controls=True,rows=len(frame),
        artifacts=[art(OUT/'scored.parquet'),art(OUT/'cells.parquet')])
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();result=read(OUT/'result.json')
    for a in result['artifacts']:verify(a)
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in [('p',OUT/'scored.parquet'),('r',stack.daily.base.OUT/'rows2025.parquet'),
            ('c',v78.OUT/'scored.parquet'),('v',stack.OUT/'scored.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''SELECT count(*),count(DISTINCT p.mvt_id),sum(CASE WHEN
          p.y IS DISTINCT FROM r.y OR p.month IS DISTINCT FROM r.month OR p.day IS DISTINCT FROM r.day
          OR p.airport IS DISTINCT FROM r.airport OR p.control IS DISTINCT FROM c.candidate
          OR p.increment IS DISTINCT FROM c.increment OR p.control_v77 IS DISTINCT FROM v.combined
          OR p.control_before IS DISTINCT FROM v.combined_before OR p.control_shift IS DISTINCT FROM v.combined_shift
          THEN 1 ELSE 0 END) FROM p JOIN r USING(mvt_id) JOIN c USING(mvt_id) JOIN v USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for name in result['cv']:
            scored=con.execute(f'SELECT month,sqrt(avg(pow({name}-y,2))) FROM p GROUP BY month').fetchall()
            for month,value in scored:np.testing.assert_allclose(value,result['scores'][month][name],rtol=0,atol=1e-8)
            np.testing.assert_allclose(np.mean([v for _,v in scored]),result['cv'][name],rtol=0,atol=1e-8)
    f=pd.read_parquet(OUT/'scored.parquet');ids=f.mvt_id.to_numpy();daily=stack.daily
    raw=pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    quality=pd.read_parquet(stack.quality.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    context=daily.rows().drop(columns='y').set_index('mvt_id');source=daily.public()
    parameters=read(daily.OUT/'parameters.json');matches=pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    for arm in ['control','candidate']:
        for month in base.parent.ts.SPLITS:
            ix=np.flatnonzero(f.month.eq(month));_,_,shift=daily.apply(context.loc[ids[ix]].reset_index(),
                f[arm+'_before'].to_numpy()[ix],source,parameters[month])
            np.testing.assert_array_equal(shift,f[arm+'_shift'].to_numpy()[ix])
        pred=f[arm+'_before'].to_numpy()+np.where(quality.quality_eligible,f[arm+'_shift'],0.)
        np.testing.assert_array_equal(daily.constrain(raw,ids,f.month,pred,matches),f[arm+'_v77'])
        np.testing.assert_array_equal(daily.constrain(raw,ids,f.month,f[arm+'_v77'].to_numpy()+.5*f.increment.to_numpy(),matches),f[arm])
    write_json(OUT/'audit.json',dict(status='pass',rows=n,identity=art(MANIFEST),result=art(OUT/'result.json'),
        original_labels_and_all_rows_retained=True,all_parent_controls_exact=True,
        independent_sql_scores=True,all_daily_shifts_and_v78_constraints_replayed=True,source=art(__file__)))
    print('All344419 rows, original labels, exactv77/v78 controls and final stages audited',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['guard','preflight','fit','score','audit'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
