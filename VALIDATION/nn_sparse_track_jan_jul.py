"""One frozen N09 + sparse-track fusion screen on all Jan/Jul held rows.

Original 156 tabular inputs, ordinary training population, clipped direct target,
AdamW/OneCycle recipe and seed42 are preserved. Historical N09 full-fit stopping
epochs21/17 are frozen. No inner fit, auxiliary target, search or upload exists.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_residual_jan_jul as shared
import nn_sensor_differences_jan_jul as sensor
import nn_aobt_observation_jan_jul as parent_contract
import nn_callsign_pool_resume as preprocess_guard
import nn_sparse_track_prototype as prototype
import sparse_track_sequence as seq
import sparse_track_tensors as tensor_seq
from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = shared.ts.RUNS/'v3_20261010_nn_sparse_track_jan_jul'
STOP = {'2025-01':21,'2025-07':17}
DESIGN = dict(parent='v58', original_tabular_columns=156, appended_tabular_features=[],
    model={key:value for key,value in prototype.SETTINGS.items()
           if key not in ['training','benchmark_batch','benchmark_repeats']},
    sequence={key:value for key,value in seq.SETTINGS.items() if key!='days'},
    sequence_population='all supplied2025 departures',
    target='original direct min(y,10800), standardized with ordinary training population',
    seed=42, member=0, stopping='frozen historical N09 epochs; no new inner fit',
    stop_epochs=STOP, network_share=.71, tree_share=.29,
    arms=['v58','direct_matched','sparse_fusion','half_sparse_half_v58','half_direct_half_v58'],
    absent_branch='sequence residual exactlyzero for fewer than2points; jointly trained tabular branch may change',
    inference='conditional mean only; original Rome and correction stack',
    exploratory_reused_development_months=True)


def cache_identity():
    manifest_path=seq.OUT/'full.json'
    manifest=shared.ts.read(manifest_path)
    builder=ROOT/'VALIDATION/build_sparse_track_full.py'
    if sha256(builder)!=manifest['source_sha256']:
        raise ValueError('Full sequence builder source changed')
    if (manifest['settings']['original_sequence']!=seq.SETTINGS
            or len(manifest['settings']['full_days'])!=365
            or manifest['settings']['full_days'][0]!='2025-01-01'
            or manifest['settings']['full_days'][-1]!='2025-12-31'
            or manifest['settings']['labels_used'] is not False):
        raise ValueError('Full sequence scope/settings changed')
    expected={'full_index.parquet','full_points.bin'}
    if set(manifest['outputs'])!=expected:
        raise ValueError('Unexpected full sequence outputs')
    for name,digest in manifest['outputs'].items():
        if sha256(seq.OUT/name)!=digest:
            raise ValueError('Full sequence output changed: '+name)
    for name,digest in manifest['helpers'].items():
        if sha256(ROOT/name)!=digest:
            raise ValueError('Full sequence helper changed: '+name)
    if sha256(seq.IDENTITY)!=manifest['identity_sha256']:
        raise ValueError('Audited sequence identity changed')
    for geometry_path,digest in [(seq.sf.COORDS,manifest['stand_map_sha256']),
                                (seq.sf.ENDS,manifest['runway_map_sha256'])]:
        if sha256(geometry_path)!=digest: raise ValueError('Sequence geometry map changed')
    for section in ['movement_sources','positions']:
        for name,digest in manifest[section].items():
            if sha256(ROOT/name)!=digest: raise ValueError('Sequence raw source changed: '+name)
    return dict(path=str(manifest_path.relative_to(ROOT)),sha256=sha256(manifest_path),
                outputs=manifest['outputs'],rows=manifest['rows'],stored_points=manifest['stored_points'])


def code_identity(cache):
    modules=[shared,sensor,parent_contract,preprocess_guard,prototype,seq,tensor_seq,
             seq.sf,seq.sg,seq.rotation,shared.nni,shared.nnj,shared.hpo]
    return dict(runner_sha256=sha256(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__)) for m in modules},
        cache=cache,control=parent_contract.control_identity(),design=DESIGN)


def guard(cache):
    value=code_identity(cache)
    path=OUT/'identity.json'
    if path.exists():
        if shared.ts.read(path)!=value:
            raise ValueError('Frozen experiment code, input cache or control changed')
    else:
        write_json(path,value)
        (OUT/'runner_initial.py').write_text(Path(__file__).read_text())


def validate_completed(month,cache):
    path=OUT/'networks'/f'{month}_s42.json'
    if not path.exists(): return None
    record=shared.ts.read(path)
    config=shared.hpo.nn_config(shared.hpo.trial_spec('nn','N09')['overrides'])
    if (record['config']!=config or record['design']!=DESIGN or record['cache']!=cache
            or record['stop_epoch']!=STOP[month] or record['month']!=month):
        raise ValueError('Completed fit recipe/cache differs from frozen experiment')
    sources=[(OUT/'networks'/f'{month}_s42_k0.parquet','prediction_sha256'),
             (OUT/'models'/f'{month}_s42_k0.pt','model_sha256'),
             (OUT/'preprocessing'/f'{month}.pkl','preprocessing_sha256'),
             (shared.baseline.OUT/'inputs'/f'{month}.pkl','input_matrix_sha256')]
    for source,key in sources:
        if sha256(source)!=record[key]: raise ValueError('Completed artifact changed: '+str(source))
    return record


def predict(model,d,tracks,sequence_rows,chunk=2048):
    model.eval()
    output=[]
    with torch.no_grad():
        for i in range(0,len(sequence_rows),chunk):
            tab=tuple(a[i:i+chunk] for a in d.held)
            prediction=model(*tab,*tracks.batch(sequence_rows[i:i+chunk]))
            output.append(prediction[:,0].cpu().numpy())
    model.train()
    return np.maximum(np.concatenate(output).astype(float)*d.sd+d.mu,0)


def train(d,tracks,train_rows,held_rows,stop):
    torch.manual_seed(42)
    np.random.seed(42)
    model=prototype.SparseFusion(d.cards,d.bins,d.train[2].shape[1]).to(d.device)
    opt=torch.optim.AdamW(model.parameters(),lr=shared.nni.NEW['ple']['lr'],
                         weight_decay=shared.nni.NEW['ple']['weight_decay'])
    count,batch=len(d.y),shared.nni.NEW['batch']
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=shared.nni.NEW['ple']['lr'],
        total_steps=shared.nni.NEW['epochs']*math.ceil(count/batch))
    generator=torch.Generator(device=d.device).manual_seed(42)
    curve=[]
    for epoch in range(stop):
        started,total=time.monotonic(),0.
        perm=torch.randperm(count,device=d.device,generator=generator)
        # Same GPU permutation/RNG as N09; copy once per epoch for CPU memmap gathers.
        cpu_perm=perm.cpu().numpy()
        step_started=time.monotonic()
        for i in range(0,count,batch):
            rows=perm[i:i+batch]
            track_batch=tracks.batch(train_rows[cpu_perm[i:i+batch]])
            output=model(*(a[rows] for a in d.train),*track_batch)
            loss=torch.nn.functional.mse_loss(output,d.ty[rows,None].expand_as(output))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
            total+=float(loss)*len(rows)
            if epoch==0 and i==batch*19:
                torch.cuda.synchronize()
                elapsed=time.monotonic()-step_started
                shared.log(f'actual first20 training steps: {1000*elapsed/20:.2f}ms/step; '
                           f'{20*batch/elapsed:.0f}rows/s; allocatedpeak={torch.cuda.max_memory_allocated()/2**30:.3f}GiB')
                write_json(OUT/f'throughput_{stop}epochs.json',dict(steps=20,batch=batch,
                    milliseconds_per_step=1000*elapsed/20,rows_per_second=20*batch/elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        pred=predict(model,d,tracks,held_rows)
        curve.append(dict(epoch=epoch+1,train_mse_std=total/count,
            rmse=shared.nni.rmse(d.yv,pred),seconds=time.monotonic()-started))
        shared.log(f'epoch {epoch+1}/{stop}: train {total/count:.4f}; '
                   f'monitored {curve[-1]["rmse"]:.4f}; {curve[-1]["seconds"]:.1f}s')
    return model,pred,curve


def fit(months,cache):
    cfg=shared.recipe()
    raw=shared.weather_all.raw_lookup()
    index=pd.read_parquet(seq.OUT/'full_index.parquet')
    points=np.memmap(seq.OUT/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    if len(index)!=cache['rows'] or len(points)!=cache['stored_points']:
        raise ValueError('Sequence shape differs from manifest')
    tracks=tensor_seq.RaggedTracks(index,points,'cuda')
    for month in months:
        dest=OUT/'networks'/f'{month}_s42_k0.parquet'
        model_path=OUT/'models'/f'{month}_s42_k0.pt'
        record_path=OUT/'networks'/f'{month}_s42.json'
        source=shared.baseline.OUT/'inputs'/f'{month}.pkl'
        if record_path.exists():
            validate_completed(month,cache)
            shared.log(f'{month}: complete frozen fit reused')
            continue
        if dest.exists() or model_path.exists():
            raise ValueError('Unrecorded partial model/prediction; inspect interrupted save')
        data=pd.read_pickle(source)
        names=data['baseline']
        if len(names)!=156 or len(set(names))!=156:
            raise ValueError('Original156 input contract changed')
        X,V=data['X'].loc[:,names],data['V'].loc[:,names]
        dates=pd.to_datetime(raw.loc[data['train_ids'],'MVT_TIME_UTC_mvt'],utc=True)
        months_train=dates.dt.strftime('%Y-%m').to_numpy()
        assert set(months_train)==set(shared.ts.FOLDS[month])
        assert not np.intersect1d(data['train_ids'],data['ids']).size
        np.testing.assert_array_equal(data['y'],raw.loc[data['train_ids'],'taxi_sec'].to_numpy())
        item=dict(X=X,Vn=V,y=data['y'],raw=raw.loc[data['ids']].reset_index(),
                  tmonth=months_train,meta=dict(month=month))
        shared.log(f'{month}: original156 encode {len(X):,} train/{len(V):,} held')
        d=sensor.DirectData(item,'cuda')
        pre=OUT/'preprocessing'/f'{month}.pkl'
        state=d.preprocessing()
        state['design']=DESIGN
        if pre.exists():
            with pre.open('rb') as handle: preprocess_guard.equal_preprocessing(pickle.load(handle),state)
        else: shared.atomic_pickle(pre,state)
        train_rows,held_rows=tracks.align(data['train_ids']),tracks.align(data['ids'])
        # Geometry/data paths are input-only. The source identity is shared across
        # outer months, while model targets and rows remain the original folds.
        initial=prototype.self_check(d.cards,d.bins,d.train[2].shape[1],'cuda')
        started=time.monotonic()
        model,pred,curve=train(d,tracks,train_rows,held_rows,STOP[month])
        model_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=model_path.with_suffix('.pt.tmp')
        torch.save(dict(state_dict=model.state_dict(),config=cfg,design=DESIGN,
            preprocessing=str(pre.relative_to(ROOT)),cache=cache),temporary)
        temporary.replace(model_path)
        shared.nni.save_network(dest,data['ids'],pred)
        write_json(record_path,dict(config=cfg,design=DESIGN,month=month,stop_epoch=STOP[month],
            initial_check=initial,minutes=(time.monotonic()-started)/60,held_curve=curve,
            train_rows=len(d.y),held_rows=len(d.yv),input_matrix_sha256=sha256(source),
            preprocessing_sha256=sha256(pre),prediction_sha256=sha256(dest),model_sha256=sha256(model_path),
            cache=cache,train_usable=int((tracks.counts[train_rows]>=2).sum()),held_usable=int((tracks.counts[held_rows]>=2).sum())))
        shared.log(f'{month}: frozen fusion saved')
        del d,data,item,X,V,model,train_rows,held_rows
        gc.collect()
        torch.cuda.empty_cache()


def score(cache):
    sc,rome,pool,trees=shared.baseline.baseline()
    errors={name:{} for name in DESIGN['arms']}
    by_seed={name:{} for name in errors}
    rows=[]
    for month in shared.ts.SPLITS:
        prediction_path=OUT/'networks'/f'{month}_s42_k0.parquet'
        if validate_completed(month,cache) is None: raise ValueError('No complete candidate fit')
        frame=pd.read_parquet(prediction_path)
        np.testing.assert_array_equal(frame.mvt_id.to_numpy(),sc.d[month]['ids'])
        direct=shared.direct_member(sc,month,42,0)
        candidate=frame.base.to_numpy(float)
        arms=dict(v58=pool[month],direct_matched=direct,sparse_fusion=candidate,
                  half_sparse_half_v58=.5*(pool[month]+candidate),
                  half_direct_half_v58=.5*(pool[month]+direct))
        month_errors={name:[] for name in arms}
        for seed in shared.hpo.SEEDS:
            replacements={shared.hpo.MEMBER_OF[p]:trees[p][seed][month] for p in trees}
            for name,neural in arms.items():
                squared=sc.se(month,seed,nn=neural,p=rome[month],**replacements)
                month_errors[name].append(squared)
                by_seed[name].setdefault(str(seed),{})[month]=float(np.sqrt(squared.mean()))
        for name in arms: errors[name][month]=np.mean(month_errors[name],axis=0)
        raw=sc.d[month]['held']['raw']
        rows.append(pd.DataFrame(dict(mvt_id=sc.d[month]['ids'],month=month,day=sc.d[month]['day'],
            airport=raw.ADEP_mvt.to_numpy(),y=sc.d[month]['y'],direct_nn=direct,candidate_nn=candidate,
            **{f'se_{name}':errors[name][month] for name in arms})))
    report=dict(design=DESIGN,scores={},comparisons={})
    for name in errors: report['scores'][name]=dict(**sc.cv(errors[name]),by_tree_seed=by_seed[name])
    if abs(report['scores']['v58']['cv']-295.7414415741871)>1e-7:
        raise ValueError('v58 control did not reproduce')
    scored=pd.concat(rows,ignore_index=True)
    for arm,control in [('sparse_fusion','direct_matched'),('sparse_fusion','v58'),
                        ('half_sparse_half_v58','v58'),('half_sparse_half_v58','half_direct_half_v58')]:
        report['comparisons'][f'{arm}_vs_{control}']=shared.ts.paired_bootstrap(scored.day.to_numpy(),
            scored.month.to_numpy(),scored[f'se_{control}'].to_numpy(),scored[f'se_{arm}'].to_numpy())
    scored.to_parquet(OUT/'scored.parquet',index=False)
    write_json(OUT/'score.json',report)
    shared.log(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['fit','score'])
    parser.add_argument('--months',nargs='+',choices=list(STOP),default=list(STOP))
    args=parser.parse_args()
    # A coordinator may temporarily reserve the shared desktop for a bounded
    # preparation job. This marker gates only process scheduling, never data,
    # epochs, labels, model selection or the optimizer recipe.
    hold=OUT/'desktop_fit_hold.json'
    if args.stage=='fit' and hold.exists():
        shared.log('Desktop fit held for coordinator preparation; no cache/encoding/GPU work started')
        while hold.exists(): time.sleep(1)
        shared.log('Desktop coordinator released fit')
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'networks').mkdir(exist_ok=True)
    parent_contract.control_identity()
    cache=cache_identity()
    guard(cache)
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='fit': fit(args.months,cache)
        else: score(cache)
