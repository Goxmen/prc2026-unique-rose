"""Expand the public callsign screen to three matched network seeds.

Reuse the completed seed42 models. Fit only1042/2042 with the original162
preprocessing, public transforms and frozen21/17 epochs. No production/upload.
"""
import argparse
import gc
from pathlib import Path
import pickle
import tempfile
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_calls_public_jan_jul as single
from records import write_json

ROOT = single.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_nn_calls_public_pool3'
MANIFEST = OUT/'identity.json'
SEEDS = [42,1042,2042]
STOP, calls, model_type, joint = single.STOP,single.calls,single.model_type,single.joint
art, read, verify = single.art,single.read,single.verify
DESIGN = dict(single.DESIGN,seed=None,seeds=SEEDS,
    change='three matched seeds of the public callsign network',
    primary='original callsign3 versus public callsign3 in selected v76 stack',
    selected_reference='original callsign15 pool; different-size reference',
    reused_member=42,new_members=[1042,2042],no_production=True)


def original(month,seed):
    k=SEEDS.index(seed)
    record=read(calls.OUT/'networks'/f'{month}_s42.json')
    member=record['networks'][str(k)]
    assert member['seed']==seed and record['stop_epoch']==STOP[month]
    mp=calls.OUT/'models'/f'{month}_s42_k{k}.pt'
    pp=calls.prediction_path(month,42,k)
    assert member['model_sha256']==art(mp)['sha256'] and member['sha256']==art(pp)['sha256']
    return dict(model=art(mp),prediction=art(pp))


def paths(month,seed):
    if seed==42:return single.paths(month)
    return OUT/'models'/f'{month}_s{seed}.pt',OUT/'networks'/f'{month}_s{seed}.parquet',OUT/'networks'/f'{month}_s{seed}.json'


def completed(month,seed):
    if seed==42:return single.completed(month)
    mp,pp,rp=paths(month,seed)
    if not rp.exists():
        assert not mp.exists() and not pp.exists(),'Unrecorded fit; inspect before resuming'
        return None
    record=read(rp)
    assert record['identity']==art(MANIFEST) and record['design']==DESIGN and record['seed']==seed
    for key in ['model','prediction','preprocessing']:verify(record[key])
    return record


def guard():
    parent=single.guard()
    assert read(single.OUT/'audit.json')['status']=='pass'
    controls={month:{str(seed):original(month,seed) for seed in SEEDS} for month in STOP}
    reused={month:art(single.paths(month)[2]) for month in STOP}
    for month in STOP:assert single.completed(month) is not None
    value=dict(design=DESIGN,source=art(__file__),parent_identity=art(single.MANIFEST),
        parent_source=art(single.__file__),parent_audit=art(single.OUT/'audit.json'),
        parent_score=art(single.OUT/'scored.parquet'),controls=controls,reused_members=reused,
        original_preprocessing={month:parent['controls'][month]['preprocessing'] for month in STOP})
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('**/*.pt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST,value)
    return value,parent


def fit():
    identity,parent=guard()
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080'
    cfg=calls.shared.recipe()
    write_json(OUT/'gpu_preflight.json',model_type.check('cuda'))
    truth=pd.read_parquet(joint.main.features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    public=pd.read_parquet(joint.main.features.OUT/'features2025.parquet').set_index('mvt_id')
    for month in STOP:
        pending=[seed for seed in SEEDS[1:] if completed(month,seed) is None]
        if not pending:continue
        contract=parent['controls'][month]
        data=pd.read_pickle(verify(contract['inputs']));names=data['baseline']
        assert len(names)==len(set(names))==156
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
            d=single.adapter.ChunkedDirectData(item,'cuda',scratch)
        state=d.preprocessing();state['design']=calls.DESIGN
        with verify(contract['preprocessing']).open('rb') as f:single.preprocessing.equal_preprocessing(pickle.load(f),state)
        assert not np.any(d.offset) and not np.any(d.val_offset)
        for seed in pending:
            control=identity['controls'][month][str(seed)]
            old=calls.shared.nni.PleNet(d.cards,d.bins,d.train[2].shape[1]).cuda()
            old.load_state_dict(torch.load(verify(control['model']),map_location='cuda',weights_only=False)['state_dict'])
            oldpred=calls.shared.predict(old,d.held,d.sd,d.mu,d.val_offset).astype(np.float32)
            saved=pd.read_parquet(verify(control['prediction']))
            np.testing.assert_array_equal(saved.mvt_id,held_ids)
            np.testing.assert_array_equal(oldpred,saved.base.to_numpy())
            del old,saved,oldpred
        del data,item
        gc.collect();torch.cuda.empty_cache()
        feature_audits={}
        for part,ids in [('train',train_ids),('held',held_ids)]:
            frame=public.loc[ids,model_type.FEATURES]
            encoded=model_type.transform(frame);feature_audits[part]=single.transform_audit(frame,encoded)
            original_inputs=getattr(d,part)
            setattr(d,part,(*original_inputs[:2],torch.cat([original_inputs[2],torch.from_numpy(encoded).cuda()],1)))
        for seed in pending:
            started=time.monotonic();mp,pp,rp=paths(month,seed)
            mp.parent.mkdir(exist_ok=True);pp.parent.mkdir(exist_ok=True)
            model,pred,curve=single.training_function()(d,np.arange(len(d.y)),seed,STOP[month],(d.held,d.yv,d.val_offset))
            temporary=mp.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(),config=cfg,design=DESIGN,seed=seed),temporary);temporary.replace(mp)
            restored=model_type.PublicCallsNet(d.cards,d.bins,d.train[2].shape[1]).cuda()
            restored.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
            np.testing.assert_array_equal(calls.shared.predict(restored,d.held,d.sd,d.mu,d.val_offset),pred)
            pd.DataFrame(dict(mvt_id=held_ids,base=pred.astype(np.float32))).to_parquet(pp,index=False)
            write_json(rp,dict(identity=art(MANIFEST),design=DESIGN,seed=seed,epochs=STOP[month],config=cfg,
                model=art(mp),prediction=art(pp),preprocessing=contract['preprocessing'],
                control=identity['controls'][month][str(seed)],train_rows=len(train_ids),held_rows=len(held_ids),
                original_preprocessing_exact=True,all_control_predictions_reloaded_exact=True,
                all_new_predictions_reloaded_exact=True,public_transform_audit=feature_audits,
                curve=curve,minutes=(time.monotonic()-started)/60,machine=torch.cuda.get_device_name()))
            print(f'{month} seed{seed}: fit and all-row old/new reloads complete',flush=True)
            del model,restored,pred
            gc.collect();torch.cuda.empty_cache()
        del d,frame,encoded,original_inputs,train_ids,held_ids
        gc.collect();torch.cuda.empty_cache()


def score():
    identity,parent=guard()
    assert not (OUT/'result.json').exists(),'Preserve completed score'
    neural=joint.neural;cat=neural.cat;replay=cat.r20.physical.replay
    selected_identity=neural.guard();sc,rome,_,trees=replay.baseline.baseline()
    truth=pd.read_parquet(joint.main.features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    prior=pd.read_parquet(cat.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id')
    matches=pd.read_parquet(cat.r20.constraints.exclusion.OUT/'matches2025.parquet');frames=[]
    for month in STOP:
        d=sc.d[month];ids=d['ids'];raw=truth.loc[ids].reset_index()
        banks={'control':[],'candidate':[]}
        for seed in SEEDS:
            record=completed(month,seed);assert record is not None
            for name,source in [('control',identity['controls'][month][str(seed)]['prediction']),('candidate',record['prediction'])]:
                frame=pd.read_parquet(verify(source));np.testing.assert_array_equal(frame.mvt_id,ids)
                banks[name].append(frame.base.to_numpy(float))
        physical=np.mean([replay.values(verify(neural.completed(month,k,selected_identity)['prediction']),ids,'base') for k in range(3)],axis=0)
        main=replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        values={name:np.mean(bank,axis=0) for name,bank in banks.items()}
        values['selected']=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        permitted=matches[matches.donor_month.isin(calls.ts.FOLDS[month]+[month])]
        outputs={}
        for name,call in values.items():
            final=sc.final(month,.71*.5*(call+physical)+.29*tree,probability)
            transferred=cat.r20.constraints.matched_bound(raw,prior.loc[ids,'v74'].to_numpy()+final-prior.loc[ids,'control'].to_numpy())
            transferred,_=cat.r20.constraints.exclusion.apply(ids,np.maximum(transferred,0),permitted)
            outputs[name]=final;outputs[name+'_transfer']=transferred
        np.testing.assert_array_equal(outputs['selected'],selected.loc[ids,'combined'])
        np.testing.assert_array_equal(outputs['selected_transfer'],selected.loc[ids,'combined_transfer'])
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**outputs)))
    f=pd.concat(frames,ignore_index=True)
    assert len(f)==344419 and f.mvt_id.is_unique and np.isfinite(f[list(outputs)]).all().all()
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in outputs} for m,g in f.groupby('month')}
    pairs=[('control','candidate'),('control_transfer','candidate_transfer'),('selected_transfer','candidate_transfer')]
    paired={f'{b}_vs_{a}':calls.ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in pairs}
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in outputs},
        paired=paired,identity=art(MANIFEST),predictions=art(OUT/'scored.parquet'),selected_replay_exact=True,
        interpretation='Matched three-member comparison; selected callsign15 is a different-size reference; development data reused')
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();result=read(OUT/'result.json');verify(result['predictions'])
    assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    receipts=[]
    for month in STOP:
        for seed in SEEDS:
            record=completed(month,seed)
            assert record['original_preprocessing_exact'] and record['all_control_predictions_reloaded_exact'] and record['all_new_predictions_reloaded_exact']
            receipts.append(art(paths(month,seed)[2]))
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('candidate',OUT/'scored.parquet'),('truth',joint.main.features.base.OUT/'rows2025.parquet'),('selected',joint.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.selected is distinct from s.combined or c.selected_transfer is distinct from s.combined_transfer
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join selected s using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for name in result['cv']:
            rows=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from candidate group by month').fetchall()
            for month,value in rows:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in rows]),result['cv'][name],atol=1e-8,rtol=0)
    report=dict(status='pass',rows=n,all_six_model_control_reload_receipts=receipts,
        independent_sql_truth_calendar_selected_scores=True,identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__))
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['guard','fit','score','audit'])
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(2):globals()[args.stage]()
