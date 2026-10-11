"""Fixed main-LightGBM screen of two inferred public stand/reference inputs."""
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

import public_reference_inverse_features as features
import public_refinements_submission_cv as rounded
import public_stack_cv as joint
from records import write_json

ROOT=joint.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_public_reference_inverse_lgb'
MANIFEST=OUT/'identity.json'
main=joint.main
parent=main.parent
art,read,verify=joint.art,joint.read,joint.verify
DESIGN=dict(parent='v76 with production rounding; selected refinements checked separately',
    change='two public stand/runway reference inverse estimates in main LightGBM',
    added=features.FEATURES, original_columns=174, columns=176,
    seed=42, trial='LB1', rounds=10000, threads=4,
    labels_rows_parameters='original public main-tree inputs, labels and settings retained',
    comparison='same-machine retained174-column main model; every other component fixed',
    primary='production-rounded v76; refined-parent substitution secondary',
    limitation='reused Jan/Jul development; external source cohort differs from competition',
    no_parameter_search=True,no_new_stopping=True,no_production=True,no_upload=True)


def guard():
    assert read(features.OUT/'audit.json')['status']=='pass'
    assert read(main.OUT/'audit.json')['status']=='pass'
    assert read(rounded.OUT/'audit.json')['status']=='pass'
    paths=[Path(__file__),Path(features.__file__),Path(rounded.__file__),Path(joint.__file__),
        Path(main.__file__),Path(parent.__file__),Path(rounded.original.constraints.__file__),
        Path(rounded.original.constraints.exclusion.__file__),
        features.OUT/'manifest.json',features.OUT/'audit.json',features.OUT/'features2025.parquet',
        Path(features.pilot.__file__),features.pilot.OUT/'audit.json',
        main.features.OUT/'features2025.parquet',main.OUT/'result.json',main.OUT/'audit.json',
        rounded.OUT/'scored.parquet',rounded.OUT/'audit.json',joint.OUT/'scored.parquet',
        joint.neural.cat.OUT/'scored.parquet',rounded.original.airport.OUT/'scored.parquet',
        main.features.base.OUT/'rows2025.parquet',rounded.original.constraints.exclusion.OUT/'matches2025.parquet']
    for month in parent.ts.SPLITS:
        paths.extend([parent.input_cache.input_path(month),main.OUT/'predictions'/f'{month}_s42.json',
            main.OUT/'predictions'/f'{month}_s42.parquet',main.OUT/'models'/f'{month}_s42.txt'])
    value=dict(design=DESIGN,sources=[art(p) for p in paths])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists(): assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('**/*.txt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST,value)
    return value


def inputs(month,public,extra,truth,held_only=False):
    d=pd.read_pickle(parent.input_cache.input_path(month));A,B=parent.append_pair(d,month)
    control=read(main.OUT/'predictions'/f'{month}_s42.json')
    assert len(control['columns'])==174 and control['params']['num_threads']==4
    assert set(truth.loc[d['train_ids'],'month'])==set(parent.ts.FOLDS[month])
    assert set(truth.loc[d['ids'],'month'])=={month}
    assert not np.intersect1d(d['train_ids'],d['ids']).size
    np.testing.assert_array_equal(d['y'],truth.loc[d['train_ids'],'y'])
    result=dict(ids=d['ids'],train_ids=d['train_ids'],y=d['y'],control=control)
    for name,frame,ids in [('X',A,d['train_ids']),('V',B,d['ids'])]:
        if held_only and name=='X':continue
        old=pd.concat([frame,public.loc[ids,main.features.FEATURES].set_axis(frame.index)],axis=1)
        assert list(old)==control['columns']
        new=pd.concat([old,extra.loc[ids,features.FEATURES].set_axis(old.index)],axis=1)
        assert list(new)==control['columns']+features.FEATURES and len(new.columns)==176
        pd.testing.assert_frame_equal(new[old.columns],old,check_exact=True)
        result[name]=new
    return result


def tables():
    return (pd.read_parquet(main.features.OUT/'features2025.parquet').set_index('mvt_id'),
        pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id'),
        pd.read_parquet(main.features.base.OUT/'rows2025.parquet').set_index('mvt_id'))


def preflight():
    guard();assert not (OUT/'preflight.json').exists(),'Preserve completed preflight'
    public,extra,truth=tables();checks=[]
    for month in parent.ts.SPLITS:
        d=inputs(month,public,extra,truth)
        checks.append(dict(month=month,train_rows=len(d['X']),held_rows=len(d['V']),
            original174_exact=True,train_months=sorted(set(truth.loc[d['train_ids'],'month'])),
            columns=list(d['X']),new_missing={c:int(d['V'][c].isna().sum()) for c in features.FEATURES}))
        del d;gc.collect()
    write_json(OUT/'preflight.json',dict(status='pass',identity=art(MANIFEST),checks=checks));print(checks,flush=True)


def completed(month):
    p=OUT/'predictions'/f'{month}_s42.parquet';m=OUT/'models'/f'{month}_s42.txt';r=p.with_suffix('.json')
    if not r.exists():
        assert not p.exists() and not m.exists(),'Unrecorded model; inspect before repeating'
        return None
    record=read(r);assert record['identity']==art(MANIFEST)
    verify(record['prediction']);verify(record['model'])
    return record


def fit():
    guard();assert read(OUT/'preflight.json')['identity']==art(MANIFEST)
    public,extra,truth=tables()
    for directory in ['models','predictions']:(OUT/directory).mkdir(exist_ok=True)
    for month in parent.ts.SPLITS:
        if completed(month):continue
        d=inputs(month,public,extra,truth);X,V=d['X'],d['V'];control=d['control']
        # Replay the entire retained control on the reconstructed original174.
        old=lgb.Booster(model_file=str(main.OUT/'models'/f'{month}_s42.txt'))
        old_prediction=np.maximum(old.predict(V[control['columns']],num_threads=4),0).astype('float32')
        expected=pd.read_parquet(main.OUT/'predictions'/f'{month}_s42.parquet')
        np.testing.assert_array_equal(expected.mvt_id,d['ids']);np.testing.assert_array_equal(old_prediction,expected.pred)
        del old,old_prediction,expected
        start=time.monotonic();print(month,len(X),'rows,176 inputs,LB1/10000',flush=True)
        dataset=lgb.Dataset(X,d['y'],categorical_feature=control['categoricals'])
        model=lgb.train(control['params'],dataset,num_boost_round=10000)
        assert model.current_iteration()==10000
        prediction=parent.modeling.predict(model,V,threads=4).astype('float32')
        assert np.isfinite(prediction).all() and (prediction>=0).all()
        mp=OUT/'models'/f'{month}_s42.txt';pp=OUT/'predictions'/f'{month}_s42.parquet'
        model.save_model(str(mp));restored=lgb.Booster(model_file=str(mp))
        np.testing.assert_array_equal(prediction,np.maximum(restored.predict(V,num_threads=4),0).astype('float32'))
        pd.DataFrame(dict(mvt_id=d['ids'],pred=prediction)).to_parquet(pp,index=False)
        write_json(pp.with_suffix('.json'),dict(identity=art(MANIFEST),model=art(mp),prediction=art(pp),
            input=art(parent.input_cache.input_path(month)),columns=list(X),categoricals=control['categoricals'],
            params=control['params'],rounds=10000,train_rows=len(X),held_rows=len(V),
            all_old_and_new_held_reload_exact=True,minutes=(time.monotonic()-start)/60,machine=platform.node()))
        del d,X,V,model,restored,dataset;gc.collect()


def score():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed score'
    neural,cat=joint.neural,joint.neural.cat
    replay=cat.r20.physical.replay;sc,rome,_,trees=replay.baseline.baseline();identity=neural.guard()
    prior=pd.read_parquet(cat.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id')
    refined=pd.read_parquet(rounded.original.airport.OUT/'scored.parquet').set_index('mvt_id')
    serialized=pd.read_parquet(rounded.OUT/'scored.parquet').set_index('mvt_id')
    truth=pd.read_parquet(main.features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    constraints=rounded.original.constraints
    matches=pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet');frames=[]
    for month in parent.ts.SPLITS:
        d=sc.d[month];ids=d['ids'];record=completed(month);assert record is not None
        new=replay.values(verify(record['prediction']),ids,'pred')
        old=replay.values(main.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        physical=np.mean([replay.values(verify(neural.completed(month,k,identity)['prediction']),ids,'base') for k in range(3)],axis=0)
        new_airport=replay.values(rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        permitted=matches[matches.donor_month.isin(parent.ts.FOLDS[month]+[month])]
        outputs={}
        for arm,airport,boundary,expected in [('v76',trees['airport'][42][month],'control',selected.loc[ids,'combined']),
            ('refined',new_airport,'candidate',refined.loc[ids,'candidate'])]:
            for suffix,main_values in [('',old),('_reference',new)]:
                ordinary=.71*.5*(calls+physical)+.29*np.mean([main_values,airport,trees['catboost'][42][month]],axis=0)
                direct=sc.final(month,ordinary,probability)
                if not suffix:np.testing.assert_array_equal(direct,expected)
                start=serialized.loc[ids,boundary+'_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy()
                pred=constraints.matched_bound(truth.loc[ids].reset_index(),start)
                pred,_=constraints.exclusion.apply(ids,np.maximum(pred,0),permitted)
                pred=np.rint(np.maximum(pred,0))
                if not suffix:np.testing.assert_array_equal(pred,serialized.loc[ids,boundary])
                outputs[arm+suffix]=pred
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**outputs)))
    f=pd.concat(frames,ignore_index=True);assert len(f)==344419 and f.mvt_id.is_unique
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in outputs} for m,g in f.groupby('month')}
    paired={arm:parent.ts.paired_bootstrap(f.day,f.month,(f[arm]-f.y)**2,(f[arm+'_reference']-f.y)**2) for arm in ['v76','refined']}
    f.to_parquet(OUT/'scored.parquet',index=False)
    value=dict(design=DESIGN,identity=art(MANIFEST),prediction=art(OUT/'scored.parquet'),scores=scores,
        cv={c:float(np.mean([v[c] for v in scores.values()])) for c in outputs},paired=paired,
        both_serialized_and_unserialized_controls_exact=True)
    write_json(OUT/'result.json',value);print(value,flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['prediction'])
    public,extra,truth=tables();reloaded=[]
    for month in parent.ts.SPLITS:
        d=inputs(month,public,extra,truth,held_only=True);record=completed(month)
        model=lgb.Booster(model_file=str(verify(record['model'])))
        assert model.feature_name()==list(d['V'])==record['columns']
        actual=np.maximum(model.predict(d['V'],num_threads=2),0).astype('float32')
        expected=pd.read_parquet(verify(record['prediction']))
        np.testing.assert_array_equal(expected.mvt_id,d['ids']);np.testing.assert_array_equal(expected.pred,actual)
        reloaded.append(dict(month=month,rows=len(actual),all_reload_exact=True))
        del d,model;gc.collect()
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('c',OUT/'scored.parquet'),('p',rounded.OUT/'scored.parquet'),
            ('r',main.features.base.OUT/'rows2025.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.month is distinct from r.month or c.day is distinct from r.day
            or c.v76 is distinct from p.control or c.refined is distinct from p.candidate
            then 1 else 0 end) from c join p using(mvt_id) join r using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for col in result['cv']:
            scores=con.execute(f'select month,sqrt(avg(pow({col}-y,2))) from c group by month').fetchall()
            for m,v in scores:np.testing.assert_allclose(v,result['scores'][m][col],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in scores]),result['cv'][col],atol=1e-8,rtol=0)
    value=dict(status='pass',rows=n,reloaded=reloaded,independent_sql_truth_calendar_controls_scores=True,
        original174_inputs_exact=True,identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__))
    write_json(OUT/'audit.json',value);print(value,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['guard','preflight','fit','score','audit'])
    a=p.parse_args()
    with threadpool_limits(4 if a.stage in ['fit','audit'] else 1):globals()[a.stage]()
