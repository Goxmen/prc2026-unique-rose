"""Fixed N09 specialist trained on a coherent no-plan view of every ordinary row.

Use original permitted ordinary training rows, direct clipped target, seed42,
and frozen N09 epochs21/17 with its unchanged 24-epoch learning-rate schedule.
The specialist sees 121 retained columns. It replaces the ordinary prediction
only for non-Rome rows whose original actual clock is missing, before applying
the unchanged v58 final stack. Compare full replacement and a fixed half blend
with the parent ordinary prediction within this gate; score every held row.

No inner selection, full-year fit or upload. Save portable encoder/model state.
The compact metadata cache and existing original matrices suffice on either host.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import nn_residual_jan_jul as shared
import nn_sensor_differences_jan_jul as direct
import missing_plan_view as view
import arrival_plan_jan_jul as input_cache
from records import sha256,write_json

ts,hpo=shared.ts,shared.hpo
OUT=ts.RUNS/'v3_20261010_nn_missing_plan_jan_jul'
STOP={'2025-01':21,'2025-07':17}
DESIGN=dict(parent='v58',seed=42,network_index=0,trial='N09',stop_epochs=STOP,
    stop_choice='frozen original N09 epochs; no inner selection',
    target='direct min(taxi_sec,10800)',training_rows='unchanged original permitted ordinary population',
    features=view.SETTINGS,scope='original AOBT3 missing AND airport != LIRF',
    arms=['full_specialist','half_specialist_half_parent'],
    replacement='ordinary prediction before unchanged final stack; parent outside scope',
    parent_ordinary='0.71 original15-network pool +0.29 original3-tree mean',
    tree_seeds=hpo.SEEDS,validation='every Jan/Jul held row; reused development months')


class DirectData(direct.DirectData):
    def preprocessing(self):
        result=super().preprocessing()
        result['design']=DESIGN
        return result


def identity():
    return dict(runner_sha256=sha256(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__))
                 for m in [view,shared,direct,input_cache,shared.nnj,shared.nni,hpo]})


def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,value in [('design',DESIGN),('code_identity',identity())]:
        path=OUT/f'{name}.json'
        if path.exists() and ts.read(path)!=value:
            raise ValueError(f'Existing {name} differs; do not resume changed source')
        write_json(path,value)


def equal_preprocessing(a,b):
    """Torch storage IDs alter pickle bytes; compare actual state on resume."""
    if type(a) is not type(b):raise ValueError('Preprocessing type differs')
    if isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or a.device!=b.device or not torch.equal(a,b):
            raise ValueError('Preprocessing tensor differs')
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,strict=True)
    elif isinstance(a,dict):
        if list(a)!=list(b):raise ValueError('Preprocessing keys/order differ')
        for k in a:equal_preprocessing(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Preprocessing length differs')
        for x,y in zip(a,b):equal_preprocessing(x,y)
    elif isinstance(a,shared.nnj.Encoder):equal_preprocessing(vars(a),vars(b))
    elif isinstance(a,(str,int,float,bool,type(None),np.generic)):
        if isinstance(a,(float,np.floating)) and np.isnan(a) and np.isnan(b):return
        if a!=b:raise ValueError('Preprocessing scalar differs')
    else:raise TypeError(type(a))


def save_preprocessing(path,current):
    if path.exists():
        before=sha256(path)
        with path.open('rb') as f:equal_preprocessing(pickle.load(f),current)
        if sha256(path)!=before:raise ValueError('Saved preprocessing was modified')
    else:shared.atomic_pickle(path,current)


def pred_path(month):
    return OUT/'networks'/f'{month}_s42_k0.parquet'


def validate_completed(month):
    record=OUT/'networks'/f'{month}_s42_k0.json'
    path=pred_path(month)
    if not record.exists():
        if path.exists():raise ValueError('Unrecorded predictions require inspection')
        return False
    info=ts.read(record)
    if info['identity']!=identity() or info['design']!=DESIGN:
        raise ValueError('Completed fit code/design changed')
    for key in ['model','preprocessing','prediction']:
        if sha256(ROOT/info[key]['path'])!=info[key]['sha256']:
            raise ValueError(f'Completed {key} changed')
    if info['source_sha256']!=sha256(input_cache.input_path(month)):
        raise ValueError('Completed fit input cache changed')
    if info['metadata_manifest_sha256']!=sha256(view.DEFAULT/'manifest.json'):
        raise ValueError('Completed fit metadata changed')
    return True


def fit(months):
    if not torch.cuda.is_available():raise RuntimeError('A verified CUDA device is required')
    cfg=shared.recipe()
    metadata,manifest=view.load()
    for month in months:
        if validate_completed(month):
            shared.log(f'{month}: complete and hash checked; skipped')
            continue
        source=input_cache.input_path(month)
        data=pd.read_pickle(source)
        names=data['baseline']
        if len(names)!=156 or not np.intersect1d(data['train_ids'],data['ids']).size==0:
            raise ValueError('Original matrix schema/membership changed')
        tm=metadata.loc[data['train_ids']]
        vm=metadata.loc[data['ids']]
        if set(tm.month.astype(str))!=set(ts.FOLDS[month]) or set(vm.month.astype(str))!={month}:
            raise ValueError('Incorrect training/held months')
        np.testing.assert_array_equal(data['y'],tm.taxi_sec.to_numpy())
        # Original provenance must agree before its columns are removed.
        for X,meta in [(data['X'],tm),(data['V'],vm)]:
            np.testing.assert_array_equal(X.clock_raw_aobt3_missing.to_numpy().astype(bool),
                                          meta.raw_aobt3_missing.to_numpy())
            np.testing.assert_array_equal(X.airport.astype(str).to_numpy(),meta.airport.astype(str).to_numpy())
        X=view.transform(data['X'].loc[:,names],data['train_ids'],metadata)
        V=view.transform(data['V'].loc[:,names],data['ids'],metadata)
        item=dict(X=X,Vn=V,y=data['y'],raw=vm.reset_index(),
                  tmonth=tm.month.astype(str).to_numpy(),meta=dict(month=month))
        shared.log(f'{month}: encode no-plan121 view, {len(X):,} ordinary training rows')
        d=DirectData(item,'cuda')
        pre=OUT/'preprocessing'/f'{month}.pkl'
        save_preprocessing(pre,d.preprocessing())
        provenance=dict(identity=identity(),design=DESIGN,config=cfg,month=month,
            source=str(source.relative_to(ROOT)),source_sha256=sha256(source),
            metadata_manifest_sha256=sha256(view.DEFAULT/'manifest.json'),
            metadata_sha256=manifest['metadata_sha256'],columns=list(X),
            training_rows=len(X),held_rows=len(V),gate_rows=int(view.gate(metadata,data['ids']).sum()),
            original_train_missing_rows=int(tm.raw_aobt3_missing.sum()),
            transformed_train_missing={c:int(X[c].isna().sum()) for c in view.OP_OWN+view.AB_OWN},
            encoding=d.info,host=platform.node(),torch=str(torch.__version__),
            cuda=torch.cuda.get_device_name(),stop_epoch=STOP[month],seed=42,
            preprocessing=dict(path=str(pre.relative_to(ROOT)),sha256=sha256(pre)))
        write_json(OUT/'preprocessing'/f'{month}.json',provenance)
        ids=np.asarray(data['ids']).copy()
        del X,V,item,data,tm,vm
        gc.collect()
        started=time.monotonic()
        model,pred,curve=shared.train(d,np.arange(len(d.y)),42,STOP[month],(d.held,d.yv,d.val_offset))
        model_path=OUT/'models'/f'{month}_s42_k0.pt'
        model_path.parent.mkdir(parents=True,exist_ok=True)
        temp=model_path.with_suffix('.pt.tmp')
        torch.save(dict(state_dict=model.state_dict(),config=cfg,design=DESIGN,
                        preprocessing=str(pre.relative_to(ROOT))),temp)
        temp.replace(model_path)
        dest=pred_path(month)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shared.nni.save_network(dest,ids,pred)
        write_json(dest.with_suffix('.json'),dict(**provenance,
            prediction=dict(path=str(dest.relative_to(ROOT)),sha256=sha256(dest)),
            model=dict(path=str(model_path.relative_to(ROOT)),sha256=sha256(model_path)),
            minutes=(time.monotonic()-started)/60,monitor_curve=curve))
        shared.log(f'{month}: saved no-plan specialist and portable encoder/model')
        del model,d
        gc.collect();torch.cuda.empty_cache()


def score():
    metadata,_=view.load()
    sc,rome,pool,trees=shared.baseline.baseline()
    arms=['v58',*DESIGN['arms']]
    errors={name:{} for name in arms};rows=[];by_seed={name:{} for name in arms}
    for month in ts.SPLITS:
        # Score can run after copying predictions/records only from another host.
        dest=pred_path(month);info=ts.read(dest.with_suffix('.json'))
        if (info['identity']!=identity() or info['design']!=DESIGN
                or info['metadata_manifest_sha256']!=sha256(view.DEFAULT/'manifest.json')
                or info['prediction']['sha256']!=sha256(dest)):
            raise ValueError('Prediction provenance changed')
        f=pd.read_parquet(dest)
        np.testing.assert_array_equal(f.mvt_id,sc.d[month]['ids'])
        selected=view.gate(metadata,f.mvt_id.to_numpy())
        prediction=f.base.to_numpy(float)
        per={name:[] for name in arms}
        for seed in hpo.SEEDS:
            tree=np.mean([trees[p][seed][month] for p in trees],axis=0)
            parent=.71*pool[month]+.29*tree
            bases=dict(v58=parent,
                full_specialist=np.where(selected,prediction,parent),
                half_specialist_half_parent=np.where(selected,.5*prediction+.5*parent,parent))
            for name,base in bases.items():
                final=sc.final(month,base,p=rome[month])
                se=(sc.d[month]['y']-final)**2
                per[name].append(se)
                by_seed[name].setdefault(str(seed),{})[month]=float(np.sqrt(se.mean()))
        for name in arms:errors[name][month]=np.mean(per[name],axis=0)
        for name in DESIGN['arms']:
            np.testing.assert_array_equal(errors[name][month][~selected],errors['v58'][month][~selected])
        rows.append(pd.DataFrame(dict(mvt_id=f.mvt_id,month=month,day=sc.d[month]['day'],
            airport=metadata.loc[f.mvt_id,'airport'].to_numpy(),gate=selected,
            y=sc.d[month]['y'],specialist=prediction,
            **{f'se_{name}':errors[name][month] for name in arms})))
    frame=pd.concat(rows,ignore_index=True)
    report=dict(design=DESIGN,scores={name:dict(**sc.cv(errors[name]),by_tree_seed=by_seed[name]) for name in arms},
                comparisons={},gate_diagnostics=[])
    if abs(report['scores']['v58']['cv']-295.7414415741871)>1e-7:
        raise ValueError('Parent score did not reproduce')
    for name in DESIGN['arms']:
        report['comparisons'][name+'_vs_v58']=ts.paired_bootstrap(frame.day,frame.month,
                                                  frame.se_v58,frame[f'se_{name}'])
    for (month,airport),g in frame[frame.gate].groupby(['month','airport'],observed=True):
        report['gate_diagnostics'].append(dict(month=month,airport=airport,rows=len(g),
            **{name:float(np.sqrt(g[f'se_{name}'].mean())) for name in arms}))
    report['caveat']='Whole-plan missingness is not random. Same reused development months; paired-day intervals are exploratory.'
    frame.to_parquet(OUT/'scored_rows.parquet',index=False)
    write_json(OUT/'result.json',report)
    shared.log(json.dumps({k:v for k,v in report.items() if k!='gate_diagnostics'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['check','fit','score'])
    p.add_argument('--months',nargs='+',choices=ts.SPLITS,default=ts.SPLITS)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.stage=='check':
        cfg=shared.recipe()
        shared.log(f'Frozen N09 {cfg}; metadata/matrix preparation and fits remain separate')
        return
    guard()
    with threadpool_limits(limits=1,user_api='blas'):
        if a.stage=='fit':fit(a.months)
        else:score()


if __name__=='__main__':
    main()
