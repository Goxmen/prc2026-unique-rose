"""One frozen, same-laptop public/physical TabM versus public-PLE screen.

Original effective2048 batch and24-epoch schedule; four512 microbatches.
Fixed21/17epochs, no held stopping or parameter search. Completed updates are
checkpointed before inference. Portable cached inputs are content verified.
"""
import argparse
import gc
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import tempfile
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_tabm as model_type
import nn_public_scratch_jan_jul as original
from records import sha256,write_json

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm'
MANIFEST=OUT/'portable.json'
STOP={'2025-01':21,'2025-07':17}
ARMS=['tabm','control']
BATCH=2048
MICRO=512
PRED_CHUNK=512
DESIGN=dict(parent='exactv78 selected complete stack',model=model_type.SETTINGS,
    control='same-laptop original public/physical PLE seed42, same microbatch execution',
    seed=42,epochs=STOP,schedule_epochs=24,effective_batch=BATCH,microbatch=MICRO,
    loss_weighting='microbatch row count / actual logical batch count; one optimizer/scheduler update',
    optimizer='AdamW lr.002; fixed original TabM wd.0003, original N09 control wd.00001',
    raw_label_cap=10800.,public_transform='unchanged12, appended after exact original156 encoder',
    sequence='unchanged audited physical scales and original sparse selection',
    inference_chunk=PRED_CHUNK,monitoring='no per-epoch held inference; fixed train epochs; final mean only',
    primary='one-of-three replacement in exactv78 versus selected pool',
    fair_control='same-laptop candidate/control single and one-of-three replacements',
    caveat='architecture family screen includes fixed architecture-specific decay and microbatch dropout trajectory',
    no_hpo=True,no_weight_search=True,no_new_gate=True,no_extra_seed=True,no_upload=True,
    evidence='extensively reused development months, not independent confirmation')

art,verify,read=original.artifact,original.verify,original.shared.ts.read


def prepare():
    old=read(original.MANIFEST)
    assert old['source']==art(original.__file__)
    fingerprints=[ROOT/f'tmp/research_20261010/{host}_input_fingerprints.json' for host in ['desktop','laptop']]
    assert read(fingerprints[0])==read(fingerprints[1])
    prior=read(ROOT/'VALIDATION/runs/v3_20261010_nn_clock_bypass_laptop_v2/portable.json')
    files={Path(__file__),Path(model_type.__file__),original.MANIFEST,*fingerprints}
    fields=['public_features','feature_audit','metadata','metadata_values']
    files.update(verify(old[k]) for k in fields)
    folds={}
    for month in STOP:
        pre=verify(old['folds'][month]['preprocessing']);files.add(pre)
        ref=read(original.paths(month)[2]);files.add(original.paths(month)[2])
        folds[month]=dict(matrix=prior['matrix_records'][month],preprocessing=art(pre),
            rows=ref['train_rows'],held_rows=ref['held_rows'],reference_record=art(original.paths(month)[2]))
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            path=Path(value).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'.venv'):files.add(path)
    files.add(ROOT/'VALIDATION/nn_public_tabm_stack.py')
    cache=old['cache'];files.add(ROOT/cache['path'])
    value=dict(design=DESIGN,source=art(__file__),files=[art(p) for p in sorted(files)],
        folds=folds,cache=cache,**{k:old[k] for k in fields},
        old_public_identity=art(original.MANIFEST),
        selected_v78=art(ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments/result.json'),
        source_fingerprints=[art(p) for p in fingerprints])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value,'Frozen source/design changed'
    else:write_json(MANIFEST,value)
    (OUT/'transfer_files.txt').write_text(''.join(v['path']+'\n' for v in value['files'])+str(MANIFEST.relative_to(ROOT))+'\n')
    print(dict(files=len(files),transfer_bytes=sum(p.stat().st_size for p in files),design=DESIGN),flush=True)


def guard(full=False):
    value=read(MANIFEST);assert value['design']==DESIGN and value['source']==art(__file__)
    for record in value['files']:verify(record)
    for month,fold in value['folds'].items():
        path=ROOT/fold['matrix']['path'];assert path.is_file()
        if full:verify(fold['matrix'])
    manifest=verify(value['cache']);cache=read(manifest)
    assert cache['rows']==value['cache']['rows'] and cache['stored_points']==value['cache']['stored_points']
    assert cache['outputs']==value['cache']['outputs']
    assert cache['settings']['original_sequence']==original.base.seq.SETTINGS
    if full:
        for name,digest in cache['outputs'].items():assert sha256(manifest.parent/name)==digest,name
    return value


def make(arm,cards,bins,n_plain):
    assert arm in ARMS
    return (model_type.PublicTabM if arm=='tabm' else model_type.control.SparseFusion)(cards,bins,n_plain)


def preflight():
    identity=guard(full=True);original.shared.recipe()
    checks=[model_type.check('cpu'),model_type.check('cuda'),model_type.control.check('cpu')]
    results=[]
    for month in STOP:
        with verify(identity['folds'][month]['preprocessing']).open('rb') as f:pre=pickle.load(f)
        # Reserve the actual retained train+held categorical/numeric/target footprint.
        rows=identity['folds'][month]['rows']+identity['folds'][month]['held_rows']
        reserve=rows*(8*len(pre['cards'])+4*(len(pre['ple'])+len(pre['plain'])+12)+4)
        for arm in ARMS:
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            storage=torch.empty(reserve,dtype=torch.uint8,device='cuda')
            torch.manual_seed(42)
            model=make(arm,pre['cards'],pre['bins'],len(pre['plain'])+12).cuda().train()
            tab,track=original.base.prototype.synthetic(pre['cards'],pre['bins'],len(pre['plain'])+12,MICRO,'cuda')
            # Dense worst-case128point windows, retaining valid encoded values.
            track=(track[0],torch.ones_like(track[1]),track[2])
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003 if arm=='tabm' else .00001)
            target=torch.zeros(MICRO,device='cuda');times=[]
            for step in range(4):
                torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
                for _ in range(BATCH//MICRO):model_type.loss(model(*tab,*track),target).div(BATCH//MICRO).backward()
                optimizer.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
            results.append(dict(month=month,arm=arm,reserved_input_bytes=reserve,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),step_seconds=float(np.mean(times[1:])),
                estimated_train_minutes=float(np.mean(times[1:])*math.ceil(identity['folds'][month]['rows']/BATCH)*STOP[month]/60)))
            del model,optimizer,storage,tab,track,target;gc.collect();torch.cuda.empty_cache()
    value=dict(identity=art(MANIFEST),checks=checks,benchmark=results,
        gpu=torch.cuda.get_device_name(),host=platform.node(),torch=str(torch.__version__),
        no_capacity_adjustment=True,design=DESIGN)
    path=OUT/'preflight.json';assert not path.exists();write_json(path,value);print(value,flush=True)


def paths(month,arm):
    stem=f'{month}_{arm}'
    return OUT/'models'/f'{stem}.pt',OUT/'networks'/f'{stem}.parquet',OUT/'networks'/f'{stem}.json'


def completed(month,arm,identity):
    model,pred,path=paths(month,arm)
    if not path.exists():return None
    record=read(path)
    assert record['identity']==art(MANIFEST) and record['design']==DESIGN
    assert record['month']==month and record['arm']==arm and record['seed']==42 and record['epochs']==STOP[month]
    assert record['inputs']==identity['folds'][month]
    for key in ['model','prediction','preprocessing','checkpoint']:verify(record[key])
    return record


def predict(model,d,tracks,rows):
    model.eval();output=[]
    with torch.no_grad():
        for start in range(0,len(rows),PRED_CHUNK):
            batch=model(*(x[start:start+PRED_CHUNK] for x in d.held),*tracks.batch(rows[start:start+PRED_CHUNK]))
            output.append(model_type.mean(batch).cpu().numpy())
    model.train()
    return np.maximum(np.concatenate(output).astype(float)*d.sd+d.mu,0)


def train(model,d,tracks,rows,epochs,arm,month):
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003 if arm=='tabm' else .00001)
    count=len(d.y);schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=24*math.ceil(count/BATCH))
    generator=torch.Generator(device=d.device).manual_seed(42);curve=[]
    for epoch in range(epochs):
        start=time.monotonic();perm=torch.randperm(count,device=d.device,generator=generator);cpu=perm.cpu().numpy();total=0.
        for i in range(0,count,BATCH):
            end=min(i+BATCH,count);opt.zero_grad(set_to_none=True);logical=0.
            for j in range(i,end,MICRO):
                stop=min(j+MICRO,end);ix=perm[j:stop]
                output=model(*(x[ix] for x in d.train),*tracks.batch(rows[cpu[j:stop]]))
                error=model_type.loss(output,d.ty[ix]);(error*((stop-j)/(end-i))).backward()
                logical+=float(error.detach())*(stop-j)
            opt.step();schedule.step();total+=logical
            if epoch==0 and i==19*BATCH:
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                value=dict(month=month,arm=arm,step_seconds=elapsed/20,rows_per_second=20*BATCH/elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),effective_batch=BATCH,microbatch=MICRO)
                write_json(OUT/'runtime'/f'{month}_{arm}.json',value);original.shared.log(str(value))
        curve.append(dict(epoch=epoch+1,train_mse_std=total/count,seconds=time.monotonic()-start))
        original.shared.log(f'{month} {arm} epoch{epoch+1}/{epochs}: train={total/count:.6f}; seconds={curve[-1]["seconds"]:.2f}')
    return curve


def fit():
    identity=guard(full=True);assert read(OUT/'preflight.json')['identity']==art(MANIFEST)
    assert torch.cuda.get_device_name()==read(OUT/'preflight.json')['gpu']
    original.shared.recipe()
    for folder in ['models','networks','runtime','trained']: (OUT/folder).mkdir(exist_ok=True)
    meta,_=original.metadata.load()
    index=pd.read_parquet(original.base.seq.OUT/'full_index.parquet')
    points=np.memmap(original.base.seq.OUT/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    tracks=original.base.tensor_seq.RaggedTracks(index,points,'cuda')
    public=pd.read_parquet(verify(identity['public_features'])).set_index('mvt_id')
    for month in STOP:
        if all(completed(month,arm,identity) is not None for arm in ARMS):continue
        data=pd.read_pickle(verify(identity['folds'][month]['matrix']));names=data['baseline'];assert len(names)==len(set(names))==156
        train_ids,held_ids=np.asarray(data['train_ids']).copy(),np.asarray(data['ids']).copy()
        tm,vm=meta.loc[train_ids],meta.loc[held_ids]
        assert set(tm.month.astype(str))==set(original.shared.ts.FOLDS[month]) and set(vm.month.astype(str))=={month}
        assert not np.intersect1d(train_ids,held_ids).size
        np.testing.assert_array_equal(data['y'],tm.taxi_sec)
        item=dict(X=data['X'][names],Vn=data['V'][names],y=data['y'],raw=vm.reset_index(),
            tmonth=tm.month.astype(str).to_numpy(),meta=dict(month=month))
        del data,tm,vm;gc.collect()
        with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT) as scratch:
            d=original.adapter.ChunkedDirectData(item,'cuda',scratch)
        state=d.preprocessing();state['design']=original.base.DESIGN
        prepath=verify(identity['folds'][month]['preprocessing'])
        with prepath.open('rb') as f:original.base.preprocess_guard.equal_preprocessing(pickle.load(f),state)
        for part,ids in [('train',train_ids),('held',held_ids)]:
            extra=torch.from_numpy(model_type.transform(public.loc[ids])).to(d.device)
            values=getattr(d,part);setattr(d,part,(*values[:2],torch.cat([values[2],extra],dim=1)))
        del item,extra,values;gc.collect()
        tr,va=tracks.align(train_ids),tracks.align(held_ids)
        for arm in ARMS:
            if completed(month,arm,identity) is not None:continue
            torch.manual_seed(42);np.random.seed(42)
            model=make(arm,d.cards,d.bins,d.train[2].shape[1]).cuda()
            mp,pp,rp=paths(month,arm);receipt=OUT/'trained'/f'{month}_{arm}.json'
            if receipt.exists():
                rec=read(receipt)
                assert rec['identity']==art(MANIFEST) and rec['design']==DESIGN and rec['inputs']==identity['folds'][month]
                assert rec['seed']==42 and rec['month']==month and rec['arm']==arm and rec['epochs']==STOP[month]
                checkpoint=torch.load(verify(rec['model']),map_location='cuda',weights_only=False)
                assert checkpoint['identity']==art(MANIFEST) and checkpoint['design']==DESIGN
                model.load_state_dict(checkpoint['state_dict']);curve=checkpoint['curve'];assert len(curve)==STOP[month]
                original.shared.log(f'{month} {arm}: completed updates restored; inference only')
            else:
                assert not mp.exists() and not pp.exists(),'Partial unrecorded artifact; inspect instead of refit'
                torch.cuda.reset_peak_memory_stats();curve=train(model,d,tracks,tr,STOP[month],arm,month)
                tmp=mp.with_suffix('.pt.tmp')
                torch.save(dict(identity=art(MANIFEST),design=DESIGN,state_dict=model.state_dict(),curve=curve),tmp);tmp.replace(mp)
                write_json(receipt,dict(identity=art(MANIFEST),design=DESIGN,month=month,arm=arm,seed=42,epochs=STOP[month],
                    inputs=identity['folds'][month],model=art(mp),preprocessing=art(prepath),updates_completed_before_inference=True))
            prediction=predict(model,d,tracks,va)
            assert np.isfinite(prediction).all() and (prediction>=0).all()
            if pp.exists():
                saved=pd.read_parquet(pp);np.testing.assert_array_equal(saved.mvt_id,held_ids)
                np.testing.assert_array_equal(saved.base,prediction.astype('float32'))
            else:original.shared.nni.save_network(pp,held_ids,prediction)
            # Reload whole held view from saved weights before completion is accepted.
            restored=make(arm,d.cards,d.bins,d.train[2].shape[1]).cuda()
            restored.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
            actual=predict(restored,d,tracks,va);np.testing.assert_array_equal(actual,prediction)
            write_json(rp,dict(identity=art(MANIFEST),design=DESIGN,month=month,arm=arm,seed=42,epochs=STOP[month],
                inputs=identity['folds'][month],model=art(mp),prediction=art(pp),preprocessing=art(prepath),checkpoint=art(receipt),
                train_rows=len(train_ids),held_rows=len(held_ids),curve=curve,all_held_reloaded_exact=True,
                host=platform.node(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
            original.shared.log(f'{month} {arm}: saved and all{len(held_ids)} held predictions reloaded exactly')
            del model,restored,prediction,actual;gc.collect();torch.cuda.empty_cache()
        del d,tr,va;gc.collect();torch.cuda.empty_cache()
    audit=dict(status='pass',identity=art(MANIFEST),members=[art(paths(m,a)[2]) for m in STOP for a in ARMS],
        full_input_hashes_verified=True,original_preprocessing_exact=True,all_held_reload_exact=True)
    write_json(OUT/'audit.json',audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','preflight','fit'])
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(2):globals()[args.stage]()
