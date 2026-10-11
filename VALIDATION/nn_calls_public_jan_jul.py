"""Public daily inputs in the original callsign network, fixed single-seed CV.

Preserve all162 original inputs and saved preprocessing. Add the same twelve
public transforms through a zero-initialized head. Reuse the exact original
training function with only its model factory changed. No full-year fits.
"""
import argparse
import gc
import math
import pickle
from pathlib import Path
import tempfile
import time
import types

import duckdb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_calls_public as model_type
import nn_callsign_pool_resume as preprocessing
import chunked_direct_data as adapter
import public_stack_cv as joint
from records import write_json,sha256

calls=model_type.calls
ROOT=calls.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_nn_calls_public'
MANIFEST=OUT/'identity.json'
STOP=calls.STOP_EPOCHS
art,read,verify=joint.art,joint.read,joint.verify
DESIGN=dict(parent='selected public physical pool and main tree, original callsign network',
    change='twelve public daily inputs through a zero-initialized head on the callsign network',
    original_inputs=162,public_inputs=model_type.FEATURES,model=model_type.SETTINGS,
    seed=42,epochs=STOP,training='original shared.train bytecode; model factory only',
    preprocessing='original162-column encoder and bins must match retained callsign state exactly',
    primary='matched original callsign seed42 versus public callsign seed42 in selected ordinary stack',
    selected_reference='original callsign15 pool; different-size reference only',
    no_new_stopping=True,no_blend_search=True,no_production=True,
    evaluation='reused Jan/Jul development; all fits on desktop RTX3080')


def paths(month):
    return OUT/'models'/f'{month}_s42.pt',OUT/'networks'/f'{month}_s42.parquet',OUT/'networks'/f'{month}_s42.json'


def original_record(month):
    calls.validate_completed_month(month)
    record=read(calls.OUT/'networks'/f'{month}_s42.json')
    member=record['networks']['0']
    assert member['seed']==42 and record['stop_epoch']==STOP[month]
    mp=calls.OUT/'models'/f'{month}_s42_k0.pt';pp=calls.prediction_path(month,42,0)
    assert sha256(mp)==member['model_sha256'] and sha256(pp)==member['sha256']
    return record,mp,pp


def training_function():
    original=calls.shared.train
    facade=types.SimpleNamespace(**{**vars(calls.shared.nni),'PleNet':model_type.PublicCallsNet})
    function=types.FunctionType(original.__code__,{**original.__globals__,'nni':facade},
        original.__name__,original.__defaults__,original.__closure__)
    assert function.__code__ is original.__code__ and calls.shared.nni.PleNet is not model_type.PublicCallsNet
    return function


def guard():
    assert read(joint.OUT/'audit.json')['status']=='pass'
    features=joint.main.features
    assert read(features.OUT/'audit.json')['status']=='pass'
    controls={}
    sources=[Path(__file__),Path(model_type.__file__),Path(calls.__file__),Path(calls.shared.__file__),
        Path(calls.shared.nni.__file__),Path(adapter.__file__),Path(preprocessing.__file__),
        Path(model_type.public.__file__),Path(features.__file__),Path(joint.__file__),
        joint.OUT/'scored.parquet',joint.OUT/'audit.json',features.OUT/'features2025.parquet',
        features.OUT/'audit.json',features.base.OUT/'rows2025.parquet']
    for month in STOP:
        rec,mp,pp=original_record(month)
        assert rec['inputs']['selected_baseline_columns']
        feature_sources,builder=calls.validate_feature_sources(month)
        paths_to_add=[mp,pp,calls.OUT/'networks'/f'{month}_s42.json',calls.OUT/'preprocessing'/f'{month}.pkl',
            calls.OUT/'preprocessing'/f'{month}.json',calls.baseline.OUT/'inputs'/f'{month}.pkl',*feature_sources.values()]
        sources.extend(paths_to_add)
        controls[month]=dict(model=art(mp),prediction=art(pp),preprocessing=art(paths_to_add[3]),
            inputs=art(paths_to_add[5]),callsign_features={k:art(v) for k,v in feature_sources.items()},
            feature_builder=builder)
    training_function()
    value=dict(design=DESIGN,sources=[art(p) for p in sources],controls=controls,
        checks=model_type.check('cpu'),training_bytecode_exact=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('**/*.pt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST,value)
    return value


def completed(month):
    mp,pp,rp=paths(month)
    if not rp.exists():
        assert not mp.exists() and not pp.exists(),'Unrecorded fit; inspect before resuming'
        return None
    rec=read(rp);assert rec['identity']==art(MANIFEST) and rec['design']==DESIGN
    for key in ['model','prediction','preprocessing']:verify(rec[key])
    return rec


def transform_audit(frame,encoded):
    indexes=np.unique(np.linspace(0,len(frame)-1,min(48,len(frame)),dtype=int))
    maximum=0.
    for i in indexes:
        for j,(name,scale) in enumerate(zip(model_type.FEATURES,model_type.public.SCALES)):
            value=float(frame.iloc[i][name])
            expected=0. if math.isnan(value) else max(-5.,min(5.,math.asinh(value/scale)/4.))
            error=abs(float(encoded[i,j])-expected);maximum=max(maximum,error)
            assert error<3e-7
    return dict(rows=len(frame),checked_values=len(indexes)*len(model_type.FEATURES),maximum_float32_error=maximum)


def fit():
    identity=guard()
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080'
    cfg=calls.shared.recipe();write_json(OUT/'gpu_preflight.json',model_type.check('cuda'))
    truth=pd.read_parquet(joint.main.features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    public=pd.read_parquet(joint.main.features.OUT/'features2025.parquet').set_index('mvt_id')
    for month in STOP:
        if completed(month) is not None:continue
        start=time.monotonic();contract=identity['controls'][month]
        data=pd.read_pickle(verify(contract['inputs']));names=data['baseline']
        assert len(names)==156 and len(set(names))==156
        for part,key,idkey in [('train','X','train_ids'),('held','V','ids')]:
            table=calls.load_features(verify(contract['callsign_features'][part]))
            data[key]=calls.add_features(data[key][names],data[idkey],table)
            assert list(data[key])==list(names)+calls.FEATURES
        train_ids=np.asarray(data['train_ids']).copy();held_ids=np.asarray(data['ids']).copy()
        assert set(truth.loc[train_ids,'month'])==set(calls.ts.FOLDS[month])
        assert set(truth.loc[held_ids,'month'])=={month} and not np.intersect1d(train_ids,held_ids).size
        np.testing.assert_array_equal(data['y'],truth.loc[train_ids,'y'])
        item=dict(X=data['X'],Vn=data['V'],y=data['y'],tmonth=truth.loc[train_ids,'month'].to_numpy(),
            raw=pd.DataFrame(dict(taxi_sec=truth.loc[held_ids,'y'].to_numpy())),meta=dict(month=month))
        del table
        with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT) as scratch:
            d=adapter.ChunkedDirectData(item,'cuda',scratch)
        state=d.preprocessing();state['design']=calls.DESIGN
        with verify(contract['preprocessing']).open('rb') as f:preprocessing.equal_preprocessing(pickle.load(f),state)
        assert not np.any(d.offset) and not np.any(d.val_offset)
        # The retained single control must reload across the entire held month
        # before a new fit is started; no control is refitted.
        old=calls.shared.nni.PleNet(d.cards,d.bins,d.train[2].shape[1]).cuda()
        old.load_state_dict(torch.load(verify(contract['model']),map_location='cuda',weights_only=False)['state_dict'])
        oldpred=calls.shared.predict(old,d.held,d.sd,d.mu,d.val_offset).astype(np.float32)
        saved=pd.read_parquet(verify(contract['prediction']))
        np.testing.assert_array_equal(saved.mvt_id,held_ids)
        np.testing.assert_array_equal(oldpred,saved.base.to_numpy())
        del old,saved,oldpred,item,data
        gc.collect();torch.cuda.empty_cache()
        feature_audits={}
        for part,ids in [('train',train_ids),('held',held_ids)]:
            frame=public.loc[ids,model_type.FEATURES]
            encoded=model_type.transform(frame);feature_audits[part]=transform_audit(frame,encoded)
            original=getattr(d,part)
            setattr(d,part,(*original[:2],torch.cat([original[2],torch.from_numpy(encoded).cuda()],1)))
        mp,pp,rp=paths(month);mp.parent.mkdir(exist_ok=True);pp.parent.mkdir(exist_ok=True)
        model,pred,curve=training_function()(d,np.arange(len(d.y)),42,STOP[month],(d.held,d.yv,d.val_offset))
        temporary=mp.with_suffix('.pt.tmp')
        torch.save(dict(state_dict=model.state_dict(),config=cfg,design=DESIGN),temporary);temporary.replace(mp)
        restored=model_type.PublicCallsNet(d.cards,d.bins,d.train[2].shape[1]).cuda()
        restored.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
        replay=calls.shared.predict(restored,d.held,d.sd,d.mu,d.val_offset)
        np.testing.assert_array_equal(replay,pred)
        pd.DataFrame(dict(mvt_id=held_ids,base=pred.astype(np.float32))).to_parquet(pp,index=False)
        write_json(rp,dict(identity=art(MANIFEST),design=DESIGN,seed=42,epochs=STOP[month],config=cfg,
            model=art(mp),prediction=art(pp),preprocessing=contract['preprocessing'],
            train_rows=len(train_ids),held_rows=len(held_ids),original_preprocessing_exact=True,
            all_control_predictions_reloaded_exact=True,all_new_predictions_reloaded_exact=True,
            public_transform_audit=feature_audits,curve=curve,minutes=(time.monotonic()-start)/60,
            machine=torch.cuda.get_device_name()))
        print(f'{month}: public callsign fit and all-row old/new reloads complete',flush=True)
        del model,restored,d,frame,encoded,original,train_ids,held_ids
        gc.collect();torch.cuda.empty_cache()


def score():
    identity=guard()
    assert not (OUT/'result.json').exists(),'Retain completed score'
    neural=joint.neural;cat=neural.cat;replay=cat.r20.physical.replay
    selected_identity=neural.guard();sc,rome,_,trees=replay.baseline.baseline()
    truth=pd.read_parquet(joint.main.features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    prior=pd.read_parquet(cat.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id')
    matches=pd.read_parquet(cat.r20.constraints.exclusion.OUT/'matches2025.parquet');frames=[]
    for month in STOP:
        rec=completed(month);assert rec is not None
        d=sc.d[month];ids=d['ids'];raw=truth.loc[ids].reset_index()
        old=pd.read_parquet(verify(identity['controls'][month]['prediction']))
        new=pd.read_parquet(verify(rec['prediction']))
        for frame in [old,new]:np.testing.assert_array_equal(frame.mvt_id,ids)
        bank=[replay.values(verify(neural.completed(month,k,selected_identity)['prediction']),ids,'base') for k in range(3)]
        physical=np.mean(bank,axis=0)
        main=replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        p=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        calls15=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        permitted=matches[matches.donor_month.isin(joint.main.features.base.FOLDS[month]+[month])]
        outputs={}
        for name,call in [('selected',calls15),('control',old.base.to_numpy(float)),('candidate',new.base.to_numpy(float))]:
            final=sc.final(month,.71*.5*(call+physical)+.29*tree,p)
            transferred=cat.r20.constraints.matched_bound(raw,prior.loc[ids,'v74'].to_numpy()+final-prior.loc[ids,'control'].to_numpy())
            transferred,_=cat.r20.constraints.exclusion.apply(ids,np.maximum(transferred,0),permitted)
            outputs[name]=final;outputs[name+'_transfer']=transferred
        np.testing.assert_array_equal(outputs['selected'],selected.loc[ids,'combined'])
        np.testing.assert_array_equal(outputs['selected_transfer'],selected.loc[ids,'combined_transfer'])
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**outputs)))
    f=pd.concat(frames,ignore_index=True);columns=list(outputs)
    assert len(f)==344419 and f.mvt_id.is_unique and np.isfinite(f[columns]).all().all()
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in f.groupby('month')}
    pairs=[('control','candidate'),('control_transfer','candidate_transfer'),('selected_transfer','candidate_transfer')]
    paired={f'{b}_vs_{a}':calls.ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in pairs}
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in columns},
        paired=paired,identity=art(MANIFEST),predictions=art(OUT/'scored.parquet'),selected_replay_exact=True,
        interpretation='Matched single is primary; callsign15 reference has different pool size; no production selected')
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();result=read(OUT/'result.json');verify(result['predictions'])
    assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    for month in STOP:
        record=completed(month)
        assert record['original_preprocessing_exact'] and record['all_control_predictions_reloaded_exact'] and record['all_new_predictions_reloaded_exact']
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('candidate',OUT/'scored.parquet'),('truth',joint.main.features.base.OUT/'rows2025.parquet'),('selected',joint.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.selected is distinct from s.combined or c.selected_transfer is distinct from s.combined_transfer
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join selected s using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for name in result['cv']:
            rows=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) AS value from candidate group by month').fetchall()
            for month,value in rows:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in rows]),result['cv'][name],atol=1e-8,rtol=0)
    report=dict(status='pass',rows=n,independent_sql_truth_calendar_selected_scores=True,
        all_control_and_candidate_held_predictions_reloaded=True,original162_preprocessing_exact=True,
        identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__))
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['guard','fit','score','audit'])
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(2):globals()[args.stage]()
