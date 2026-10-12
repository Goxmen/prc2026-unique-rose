"""One fixed, input-only paired-flight context screen; no search or upload.

Desktop-only same-hardware saved public/physical controls. Audited fast execution
keeps the original native2048 optimizer updates and saves weights before any
full-held inference. Production scope is deliberately unavailable here.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import tempfile
import time
import types

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_cohort as model_type
import nn_public_scratch_jan_jul as original
import fixed_epoch_sparse_train as fast
from records import write_json

context=model_type.context
ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_cohort'
MANIFEST=OUT/'identity.json'
INPUT_AUDIT=ROOT/'tmp/research_20261012/error_mechanisms/cohort_input_audit.json'
STOP={'2025-01':21,'2025-07':17}
DESIGN=dict(parent='exactv78 public/physical seed42 and one-of-three selected pool replacement',
    model=model_type.SETTINGS,seed=42,epochs=STOP,original_columns=156,public_columns=12,
    context_index='lookup-only appended integer row index, stripped before original network',
    target='same original min(y,10800) standardized target',
    training='audited fast sparse train bytecode, original native2048 AdamW/OneCycle24 updates',
    monitor='first1024 held inputs, eval-only, no held labels or stopping',
    machine='desktop RTX3080; saved same-desktop original public/physical control',
    primary='replace seed42 in selected3; retain1042/2042 and all downstream v78 stack',
    secondary='matched seed42 complete stack',no_hpo=True,no_new_gate=True,no_weight_search=True,
    no_neighbor_labels=True,no_production=True,no_upload=True,reused_development_months=True)
art,verify,read=original.artifact,original.verify,original.shared.ts.read


def paths(month):
    return OUT/'models'/f'{month}_s42.pt',OUT/'networks'/f'{month}_s42.parquet',OUT/'networks'/f'{month}_s42.json'


def parent_completed(month,identity):
    # Original CLI/pool selects its sole public variant at runtime. Validate it
    # explicitly without changing that imported module's global recipe.
    expected={**original.DESIGN,'variant':'public'}
    assert identity['design']==expected
    source=original.completed
    namespace={**source.__globals__,'DESIGN':expected}
    fn=types.FunctionType(source.__code__,namespace,'public_completed',source.__defaults__,source.__closure__)
    return fn(month,identity)


def prepare():
    bank=context.guard('train',full=True)
    assert context.FOLDS==original.shared.ts.FOLDS
    old=read(original.MANIFEST)
    assert old['source']==art(original.__file__)
    for helper in old['helpers']:verify(helper)
    source_index=bank['sources'][0]
    assert source_index['path']==str((original.base.seq.OUT/'full_index.parquet').relative_to(ROOT))
    assert source_index['sha256']==old['cache']['outputs']['full_index.parquet']
    files={Path(__file__),Path(model_type.__file__),Path(context.__file__),Path(fast.__file__),
           original.MANIFEST,ROOT/'VALIDATION/nn_public_cohort_stack.py',
           ROOT/'VALIDATION/cohort_v79_transport.py',ROOT/'VALIDATION/target205_delta_transport.py',
           ROOT/'VALIDATION/tests/test_public_cohort.py',
           ROOT/'VALIDATION/tools/audit_flight_cohort_context_20261012.py',
           ROOT/'VALIDATION/runs/v3_20261012_cohort_delta_transport/design.json'}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            path=Path(value).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'.venv'):files.add(path)
    folds={}
    for month in STOP:
        record=parent_completed(month,old);assert record is not None
        folds[month]={**old['folds'][month],'control_record':art(original.paths(month)[2]),
            'control_prediction':record['prediction'],'rows':record['train_rows'],'held_rows':record['held_rows']}
    identity=dict(design=DESIGN,source=art(__file__),files=[art(p) for p in sorted(files)],
        parent=art(original.MANIFEST),bank=art(context.OUT/'train/manifest.json'),bank_identity=bank,
        folds=folds,cache=old['cache'],public_features=old['public_features'],
        metadata=old['metadata'],metadata_values=old['metadata_values'],
        selected_v78=art(ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments/result.json'))
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==identity,'Frozen cohort identity changed'
    else:write_json(MANIFEST,identity)
    print(dict(prepared=True,bank_rows=bank['rows'],design=DESIGN),flush=True)


def guard(full=False):
    identity=read(MANIFEST)
    assert identity['source']==art(__file__) and identity['design']==DESIGN
    for helper in identity['files']:verify(helper)
    verify(identity['parent']);verify(identity['bank']);verify(identity['selected_v78'])
    assert context.guard('train',full=full)==identity['bank_identity']
    for key in ['public_features','metadata','metadata_values']:verify(identity[key])
    verify(identity['cache'])
    if full:
        for month,fold in identity['folds'].items():
            for key in ['matrix','preprocessing','control_record','control_prediction']:verify(fold[key])
        for name,digest in identity['cache']['outputs'].items():
            assert original.sha256(original.base.seq.OUT/name)==digest
    return identity


def load_bank(month,identity,device):
    folder=context.OUT/'train';bank_record=identity['bank_identity']
    arrays={name:np.load(folder/f'{name}.npy',mmap_mode='r',allow_pickle=False) for name in ['meta','values','clock']}
    neighbor=np.load(folder/f'neighbors_{month}.npy',mmap_mode='r',allow_pickle=False)
    meta,values,clock=(arrays[k] for k in ['meta','values','clock'])
    assert len(meta)==bank_record['rows'] and clock.shape==(len(meta),) and clock.dtype==np.float64
    assert np.all(np.diff(meta[:,0])>0)
    np.testing.assert_array_equal(values[:,8],np.isfinite(clock))
    gap=(meta[:,1]-clock)/1000;valid=np.isfinite(gap)&(gap>=0)&(gap<=10800)
    np.testing.assert_array_equal(values[:,4],valid)
    np.testing.assert_array_equal(values[:,0],np.where(valid,gap,0).astype('float32'))
    months=pd.to_datetime(meta[:,1],unit='ms',utc=True).strftime('%Y-%m').to_numpy()
    # Exhaustive lookup scope/query/airport/time audit in bounded chunks.
    train=np.isin(months,context.FOLDS[month]);held=months==month
    assert (neighbor[~(train|held)]==-1).all()
    for start in range(0,len(meta),8192):
        rows=np.arange(start,min(start+8192,len(meta)));indices=neighbor[rows].astype(np.int64)
        present=indices>=0;safe=np.maximum(indices,0)
        assert (indices>=-1).all() and (indices<len(meta)).all()
        allowed=train[safe]|(held[rows,None]&held[safe])
        assert (~present|allowed).all()
        assert (~present|context.valid_candidates(meta,clock,rows,safe)).all()
    result=model_type.ContextBank(meta,values,neighbor,'cv:'+month).to(device)
    assert result.ranking_view=='cv:'+month
    return result


def training_function(bank):
    engine=types.SimpleNamespace(**vars(fast.original))
    engine.prototype=types.SimpleNamespace(SparseFusion=lambda cards,bins,plain:
        model_type.SparseFusion(cards,bins,plain,bank))
    namespace=dict(fast.train_updates.__globals__);namespace['original']=engine
    fn=types.FunctionType(fast.train_updates.__code__,namespace,'cohort_train_updates',
        fast.train_updates.__defaults__,fast.train_updates.__closure__)
    assert fn.__code__ is fast.train_updates.__code__
    assert fast.original.prototype is original.base.prototype
    return fn


def completed(month,identity):
    model,pred,path=paths(month)
    if not path.exists():return None
    value=read(path)
    assert value['identity']==art(MANIFEST) and value['design']==DESIGN
    assert value['month']==month and value['seed']==42 and value['epochs']==STOP[month]
    assert value['inputs']==identity['folds'][month] and value['bank']==identity['bank']
    for key in ['model','prediction','preprocessing','checkpoint']:verify(value[key])
    assert value['all_held_reloaded_exact']
    return value


def predict(model,data,tracks,rows):
    return original.base.predict(model,data,tracks,rows,chunk=2048)


def audit_guard(identity):
    receipt=read(INPUT_AUDIT)
    assert receipt['status']=='pass' and receipt['rows']==identity['bank_identity']['rows']
    for key,expected in [('manifest',identity['bank']),('builder',art(context.__file__))]:
        assert {k:receipt[key][k] for k in ['path','sha256']}==expected
    verify(receipt['source'])
    return art(INPUT_AUDIT)


def preflight():
    assert not (OUT/'preflight.json').exists(), 'Preserve completed resource/parity probe'
    identity=guard(full=True)
    input_audit=audit_guard(identity)
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080'
    original.shared.recipe()
    sys.path.insert(0,str(ROOT/'VALIDATION/tests'))
    from test_public_cohort import execution_check
    # Default CuDNN is nondeterministic even for original/original repeats.
    # Scope deterministic kernels to exact parity assertions, never to fitting.
    with torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        checks=[model_type.check('cpu'),model_type.check('cuda'),execution_check('cuda')]
    results=[]
    for month in STOP:
        with verify(identity['folds'][month]['preprocessing']).open('rb') as f:pre=pickle.load(f)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        bank=load_bank(month,identity,'cuda')
        rows=identity['folds'][month]['rows']+identity['folds'][month]['held_rows']
        reserve=rows*(8*len(pre['cards'])+4*(len(pre['ple'])+len(pre['plain'])+13)+4)
        storage=torch.empty(reserve,dtype=torch.uint8,device='cuda')
        torch.manual_seed(42)
        model=model_type.SparseFusion(pre['cards'],pre['bins'],len(pre['plain'])+13,bank).cuda().train()
        tab,track=original.base.prototype.synthetic(pre['cards'],pre['bins'],len(pre['plain'])+13,2048,'cuda')
        eligible=torch.nonzero(bank.neighbors[:,0]>=0,as_tuple=True)[0][:2048]
        assert len(eligible)==2048
        tab[2][:,-1]=eligible.to(torch.float32)
        track=(track[0],torch.ones_like(track[1]),track[2])
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.00001);times=[]
        for step in range(5):
            torch.cuda.synchronize();started=time.monotonic();opt.zero_grad(set_to_none=True)
            output=model(*tab,*track);output.square().mean().backward();opt.step()
            torch.cuda.synchronize();times.append(time.monotonic()-started)
        seconds=float(np.mean(times[1:]))
        results.append(dict(month=month,step_seconds=seconds,rows_per_second=2048/seconds,
            reserved_input_bytes=reserve,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            estimated_training_minutes=seconds*math.ceil(identity['folds'][month]['rows']/2048)*STOP[month]/60))
        del bank,storage,model,tab,track,opt,output,eligible;gc.collect();torch.cuda.empty_cache()
    result=dict(status='pass',identity=art(MANIFEST),input_audit=input_audit,checks=checks,
        fast_bytecode_exact=True,benchmark=results,gpu=torch.cuda.get_device_name(),
        no_production_updates=True,production_backend_unchanged=True)
    assert not (OUT/'preflight.json').exists();write_json(OUT/'preflight.json',result);print(result,flush=True)


def fit():
    identity=guard(full=True);input_audit=audit_guard(identity)
    ready=read(OUT/'preflight.json')
    assert ready['status']=='pass' and ready['identity']==art(MANIFEST) and ready['input_audit']==input_audit
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080'
    original.shared.recipe()
    for sub in ['models','networks','trained']:(OUT/sub).mkdir(exist_ok=True)
    metadata,_=original.metadata.load()
    index=pd.read_parquet(original.base.seq.OUT/'full_index.parquet')
    points=np.memmap(original.base.seq.OUT/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    tracks=original.base.tensor_seq.RaggedTracks(index,points,'cuda')
    public=pd.read_parquet(verify(identity['public_features'])).set_index('mvt_id')
    for month in STOP:
        if completed(month,identity) is not None:continue
        data=pd.read_pickle(verify(identity['folds'][month]['matrix']));names=data['baseline']
        assert len(names)==len(set(names))==156
        train_ids,held_ids=np.asarray(data['train_ids']).copy(),np.asarray(data['ids']).copy()
        tm,vm=metadata.loc[train_ids],metadata.loc[held_ids]
        assert set(tm.month.astype(str))==set(context.FOLDS[month]) and set(vm.month.astype(str))=={month}
        assert not np.intersect1d(train_ids,held_ids).size
        np.testing.assert_array_equal(data['y'],tm.taxi_sec)
        item=dict(X=data['X'][names],Vn=data['V'][names],y=data['y'],raw=vm.reset_index(),
            tmonth=tm.month.astype(str).to_numpy(),meta=dict(month=month))
        del data,tm,vm;gc.collect()
        with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT)as scratch:
            d=original.adapter.ChunkedDirectData(item,'cuda',scratch)
        pre=verify(identity['folds'][month]['preprocessing'])
        state=d.preprocessing();state['design']=original.base.DESIGN
        with pre.open('rb')as f:original.base.preprocess_guard.equal_preprocessing(pickle.load(f),state)
        del item;gc.collect()
        bank=load_bank(month,identity,'cuda')
        for part,ids in [('train',train_ids),('held',held_ids)]:
            extra=torch.from_numpy(model_type.parent.transform(public.loc[ids])).to(d.device)
            query=torch.from_numpy(bank.align(ids).astype('float32')[:,None]).to(d.device)
            values=getattr(d,part);setattr(d,part,(*values[:2],torch.cat([values[2],extra,query],1)))
        del extra,query,values;gc.collect()
        tr,va=tracks.align(train_ids),tracks.align(held_ids)
        views=[fast.RankingView('held',0,len(held_ids),tracks,va)]
        mp,pp,rp=paths(month);receipt=OUT/'trained'/f'{month}_s42.json'
        if receipt.exists():
            record=read(receipt)
            assert record['identity']==art(MANIFEST) and record['design']==DESIGN
            assert record['month']==month and record['epochs']==STOP[month] and record['seed']==42
            assert record['inputs']==identity['folds'][month] and record['bank']==identity['bank']
            assert record['preprocessing']==identity['folds'][month]['preprocessing']
            verify(record['preprocessing'])
            checkpoint=torch.load(verify(record['model']),map_location='cuda',weights_only=False)
            assert checkpoint['identity']==art(MANIFEST) and checkpoint['design']==DESIGN
            model=model_type.SparseFusion(d.cards,d.bins,d.train[2].shape[1],bank).to(d.device)
            model.load_state_dict(checkpoint['state_dict']);curve=checkpoint['curve'];assert len(curve)==STOP[month]
            original.shared.log(month+': restoring completed updates for inference only')
        else:
            assert not mp.exists() and not pp.exists(),'Unrecorded partial model; inspect instead of refit'
            model,curve=training_function(bank)(d,tracks,tr,views,42,STOP[month])
            tmp=mp.with_suffix('.pt.tmp');torch.save(dict(identity=art(MANIFEST),design=DESIGN,
                state_dict=model.state_dict(),curve=curve),tmp);tmp.replace(mp)
            write_json(receipt,dict(identity=art(MANIFEST),design=DESIGN,month=month,seed=42,
                epochs=STOP[month],inputs=identity['folds'][month],bank=identity['bank'],
                model=art(mp),preprocessing=art(pre),updates_completed_before_inference=True))
        result=predict(model,d,tracks,va)
        assert np.isfinite(result).all() and (result>=0).all()
        if pp.exists():
            saved=pd.read_parquet(pp);np.testing.assert_array_equal(saved.mvt_id,held_ids)
            np.testing.assert_array_equal(saved.base,result.astype('float32'))
        else:original.shared.nni.save_network(pp,held_ids,result)
        reload=model_type.SparseFusion(d.cards,d.bins,d.train[2].shape[1],bank).to(d.device)
        reload.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
        np.testing.assert_array_equal(predict(reload,d,tracks,va),result)
        write_json(rp,dict(identity=art(MANIFEST),design=DESIGN,month=month,seed=42,epochs=STOP[month],
            inputs=identity['folds'][month],bank=identity['bank'],model=art(mp),prediction=art(pp),
            preprocessing=art(pre),checkpoint=art(receipt),curve=curve,train_rows=len(train_ids),
            held_rows=len(held_ids),host=platform.node(),gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__),all_held_reloaded_exact=True))
        original.shared.log(month+': cohort saved and all held predictions reloaded exactly')
        del d,bank,model,reload,result,tr,va,views;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/'audit.json',dict(status='pass',identity=art(MANIFEST),
        members=[art(paths(m)[2])for m in STOP],input_scope_preprocessing_and_full_reload=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['check','prepare','preflight','fit']);a=p.parse_args()
    torch.set_num_threads(1 if a.stage=='check' else 2)
    with threadpool_limits(1 if a.stage=='check' else 2):
        if a.stage=='check':print(dict(context=context.checks(),model=model_type.check('cpu')))
        else:globals()[a.stage]()
