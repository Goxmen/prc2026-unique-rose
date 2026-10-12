"""Inference only for the original public PLE physical pool on final inputs.

These three saved models are distinct from pre-public physical networks and
the later TabM family. Retain original desktop execution and 2048-row chunks.
Every old prediction must reproduce before the final input view is inferred.
"""
import argparse
import gc
import os
import pickle
import types
from pathlib import Path

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_submit as production
import final_public_tabm_inference as bridge
import final_public_features as public

ROOT = production.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261012_final_public_ple_inference'
MANIFEST = OUT/'identity.json'
old = production.old
art, read, verify, write = production.art, production.read, production.verify, production.write_json
DESIGN = dict(family='original public PLE physical pool, not pre-public physical or TabM',
    seeds=[42, 1042, 2042], epochs=15, views=['old', 'final'],
    preprocessing='saved full-year Encoder.transform/layout/bins/target scale; never refit',
    inputs='same 156 ordinary inputs plus fixed12 public inputs, separate raw-track namespaces',
    bridge='unchanged audited final_public_tabm_inference.data_for_view bytecode; PLE public transform',
    control='all three old-view predictions must replay exactly on original desktop/GPU/Torch',
    prediction='unchanged fixed_epoch_sparse_train.predict_view, original2048-row chunks',
    no_fit=True, no_new_selection=True, no_version=True, no_upload=True)


def data_for_view(manifest, pre, view):
    proxy=types.SimpleNamespace(cv=types.SimpleNamespace(model_type=production.model_type))
    original=bridge.data_for_view
    f=types.FunctionType(original.__code__,dict(original.__globals__,old=old,first=proxy,OUT=OUT),
        original.__name__,original.__defaults__,original.__closure__)
    assert f.__code__ is original.__code__
    return f(manifest,pre,view)


def prepare():
    assert not MANIFEST.exists(), 'Preserve frozen final inference'
    original=production.guard()
    assert read(production.OUT/'audit.json')['status']=='pass'
    assert read(public.RECORDS/'audit.json')['status']=='pass'
    pub=read(public.OUT/'manifest.json')
    assert pub['output']==art(public.OUT/'features.parquet')
    for a in pub['sources']: verify(a)
    physical=original['parent_packet']
    assert 'final' in physical['packet']['views'] and 'final' in physical['tracks']
    members=[]; execution=None
    paths=[Path(__file__),Path(production.__file__),Path(production.model_type.__file__),
        Path(bridge.__file__),Path(public.__file__),Path(old.adapter.__file__),Path(old.fast.__file__),
        production.MANIFEST,production.OUT/'audit.json',production.physical.MANIFEST,
        public.OUT/'manifest.json',public.RECORDS/'audit.json',verify(pub['output']),
        ROOT/'data/submitting.parquet',ROOT/'data/final_submitting.parquet']
    paths += [verify(a) for a in original['helpers']]
    for k in range(3):
        trained=production.trained(k); completed=production.completed(k)
        assert trained is not None and completed is not None
        assert trained['seed']==DESIGN['seeds'][k] and trained['epochs']==15
        observed={name:trained[name] for name in ['host','gpu','torch']}
        if execution is None: execution=observed
        else: assert observed==execution
        member=dict(seed=trained['seed'],identity=art(production.MANIFEST),
            trained=art(production.paths(k)[1]),prediction_record=art(production.paths(k)[2]),
            model=trained['model'],preprocessing=trained['preprocessing'],
            old_prediction=completed['prediction'],design=production.DESIGN)
        members.append(member)
        paths += [verify(member[key]) for key in ['trained','prediction_record','model','preprocessing','old_prediction']]
    assert execution['gpu']=='NVIDIA GeForce RTX 3080'
    paths += [verify(original['ranking_features']),verify(physical['packet']['cached_training_inputs']),
        old.packets.packet_path('final')]
    for view in DESIGN['views']:
        paths += [verify(a) for a in physical['tracks'][view]['files']]
    value=dict(design=DESIGN,execution=execution,members=members,
        original_public=original['ranking_features'],final_public=pub['output'],
        packet=physical['packet'],tracks={v:physical['tracks'][v] for v in DESIGN['views']},
        original_config=physical['config'],source=art(__file__),
        sources=[art(p) for p in sorted(set(paths))])
    OUT.mkdir(parents=True,exist_ok=True);write(MANIFEST,value)
    print(dict(prepared=True,members=3,final_rows=670790,execution=execution,no_fit=True),flush=True)


def guard():
    m=read(MANIFEST)
    assert m['design']==DESIGN and m['source']==art(__file__)
    for a in m['sources']:verify(a)
    assert old.fast.INFERENCE_CHUNK==2048
    assert production.model_type.transform is bridge.first.cv.model_type.transform
    return m


def predict(model,d,tracks,ids):
    view=old.fast.ranking_views(d,{tracks.ranking_view:ids},{tracks.ranking_view:tracks})[0]
    return old.fast.predict_view(model,d,view).astype(np.float32)


def infer():
    m=guard()
    observed=dict(host=os.uname().nodename,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))
    assert observed==m['execution'],'Retain original production hardware and Torch; do not substitute silently'
    production.cv.single.shared.recipe()
    with verify(m['members'][0]['preprocessing']).open('rb') as f:pre=pickle.load(f)
    assert pre['design']==production.DESIGN
    for member in m['members'][1:]:
        with verify(member['preprocessing']).open('rb') as f:other=pickle.load(f)
        production.cv.single.base.preprocess_guard.equal_preprocessing(pre,other)
    (OUT/'predictions').mkdir(exist_ok=True)
    for view in DESIGN['views']:
        record=OUT/f'{view}.json'
        if record.exists():
            receipt=read(record);assert receipt['identity']==art(MANIFEST)
            for a in receipt['predictions']:verify(a)
            continue
        if view=='final':assert read(OUT/'old.json')['all_old_predictions_exact']
        ids,d,tracks,rows=data_for_view(m,pre,view)
        np.testing.assert_array_equal(rows,tracks.align(ids))
        outputs=[]
        for k,member in enumerate(m['members']):
            checkpoint=torch.load(verify(member['model']),map_location='cuda',weights_only=False)
            assert checkpoint['design']==member['design'] and checkpoint['config']==m['original_config']
            assert checkpoint['preprocessing']==member['preprocessing']['path']
            model=production.model_type.SparseFusion(pre['cards'],pre['bins'],len(pre['plain'])+12).cuda()
            model.load_state_dict(checkpoint['state_dict'])
            value=predict(model,d,tracks,ids)
            assert np.isfinite(value).all() and (value>=0).all()
            if view=='old':
                prior=pd.read_parquet(verify(member['old_prediction']))
                np.testing.assert_array_equal(ids,prior.mvt_id);np.testing.assert_array_equal(value,prior.base)
            else:
                restored=production.model_type.SparseFusion(pre['cards'],pre['bins'],len(pre['plain'])+12).cuda()
                restored.load_state_dict(torch.load(verify(member['model']),map_location='cuda',weights_only=False)['state_dict'])
                np.testing.assert_array_equal(value,predict(restored,d,tracks,ids))
                del restored
                path=OUT/'predictions'/f'final_k{k}.parquet'
                f=pd.DataFrame(dict(mvt_id=ids,base=value))
                if path.exists():pd.testing.assert_frame_equal(pd.read_parquet(path),f,check_exact=True)
                else:f.to_parquet(path,index=False)
                outputs.append(art(path))
            del checkpoint,model,value;gc.collect();torch.cuda.empty_cache()
        write(record,dict(identity=art(MANIFEST),view=view,rows=len(ids),execution=observed,
            all_old_predictions_exact=True,all_final_reloads_exact=view=='final',
            predictions=outputs,no_fit=True))
        del ids,d,tracks,rows;gc.collect();torch.cuda.empty_cache()
        print(dict(view=view,verified=True,rows=read(record)['rows']),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','infer'])
    torch.set_num_threads(2)
    with threadpool_limits(2):globals()[ap.parse_args().stage]()
