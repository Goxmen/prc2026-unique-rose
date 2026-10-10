"""v58 direct PLE plus six input-only daily callsign and squawk clocks.

Preserve the original156 columns and clipped direct target. Join numeric input
features by movement ID, including ambiguity/censor indicators and a fold-scoped
previous-own-arrival check. Missing original AOBT does not suppress independent
callsign evidence. No aircraft identity or callsign string enters the model.

Freeze original seed42 N09 stop epochs21 (Jan) and17 (Jul); use the unchanged
24-epoch OneCycle schedule, optimizer and network. Score matched k0 and fixed
half-new/half-v58 neural pools separately. No full-year fit, upload or ledger edit.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
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
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
import nn_residual_jan_jul as shared
import nn_sensor_differences_jan_jul as sensor
import callsign_clocks as clocks
from records import sha256,write_json

hpo,ts,baseline=shared.hpo,shared.ts,shared.baseline
OUT=ts.RUNS/'v3_20261010_nn_callsign_clocks_jan_jul'
FEATURES=['cc_call_start_sec','cc_squawk_start_sec','cc_previous_call_end_sec',
          'cc_before_own_arrival','cc_ambiguous','cc_censored']
if list(clocks.FEATURE_NAMES)!=FEATURES or clocks.rotation.FOLDS!=ts.FOLDS:
    raise ValueError('Callsign feature contract or folds changed')
FEATURE_DIR=clocks.DEFAULT
STOP_EPOCHS={'2025-01':21,'2025-07':17}
DESIGN=dict(parent='v58',control='HPO N09 matched seed42 k0',added_features=FEATURES,
    target='direct min(taxi_sec,10800)',network_share=.71,
    neural_arms=['callsign_clocks','half_callsign_half_v58'],heldout_months=ts.SPLITS,
    stop_choice='frozen historical N09 seed42 epochs; no candidate inner fit',
    frozen_stop_epochs=STOP_EPOCHS,train_input_cache=str(baseline.OUT/'inputs'),
    feature_cache=str(FEATURE_DIR),
    reference='input-only daily callsign/squawk aggregate clocks; prior arrival check uses fold references',
    training_reference='ts.FOLDS[held month] for prior own arrivals',
    held_reference='ts.FOLDS[held month] plus own held month input/arrival context',
    query_missing='numeric NaNs preserved; independent of original AOBT availability',
    feature_settings=clocks.SETTINGS,
    weather_inputs_used=False,groups=[42],networks_per_group=1,
    exploratory_reused_development_months=True)


def control_identity():
    path=hpo.record_path('nn')
    record=ts.read(path)['trials']['N09']
    cfg=hpo.nn_config(hpo.trial_spec('nn','N09')['overrides'])
    if record['config']!=cfg:
        raise ValueError('Historical matched N09 configuration differs')
    for month,epoch in STOP_EPOCHS.items():
        if record[month]['42']['stop_epoch']!=epoch:
            raise ValueError('Historical matched N09 stop epoch differs')
    pred=hpo.pred_path('nn','N09',42)
    return dict(config=cfg,stop_epochs=STOP_EPOCHS,
                record_path=str(path.relative_to(ROOT)),record_sha256=sha256(path),
                predictions_path=str(pred.relative_to(ROOT)),predictions_sha256=sha256(pred),
                control_member='n0',network_share=.71,tree_share=.29)


def load_features(path):
    table=pd.read_parquet(path).set_index('mvt_id')
    if not table.index.is_unique or list(table.columns)!=FEATURES:
        raise ValueError('Invalid Callsign clock feature table')
    if any(table[c].dtype!=np.dtype('float32') for c in FEATURES):
        raise ValueError('Callsign clock inputs must be float32')
    return table


def add_features(X,ids,table):
    if len(X)!=len(ids) or set(FEATURES)&set(X.columns):
        raise ValueError('Invalid row length or pre-existing Callsign clock columns')
    if not table.index.is_unique or not np.isin(ids,table.index).all():
        raise ValueError('Callsign clock feature table misses queried movement IDs')
    values=table.loc[np.asarray(ids),FEATURES].copy().set_axis(X.index)
    result=pd.concat([X,values],axis=1)
    if list(result)!=list(X)+FEATURES or not result.index.equals(X.index):
        raise ValueError('Callsign clock feature join changed input row/column order')
    return result


class DirectData(sensor.DirectData):
    def preprocessing(self):
        result=super().preprocessing()
        result['design']=DESIGN
        return result


def identity():
    parsed=ast.parse(Path(__file__).read_text())
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
              and n.name in {'add_features','DirectData','validate_feature_sources','control_identity','load_features','fit'}]
    digest=hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),
                                 include_attributes=False).encode()).hexdigest()
    return dict(feature_data_ast_sha256=digest,features=FEATURES,control=control_identity(),
        helpers={str(Path(module.__file__).relative_to(ROOT)):sha256(Path(module.__file__))
                 for module in [shared,sensor,clocks,clocks.rotation,shared.nni,shared.nnj,hpo]})


def guard_run():
    path=OUT/'design.json'
    if path.exists() and ts.read(path)!=DESIGN:
        raise ValueError('Existing design changed; use a distinct run directory')
    current=identity()
    path=OUT/'code_identity.json'
    if path.exists():
        if ts.read(path)!=current:
            raise ValueError('Feature/data implementation or training helper sources changed')
    else:
        if list((OUT/'networks').glob('*.parquet')):
            raise ValueError('Existing predictions lack a recorded implementation identity')
        write_json(path,current)
        (OUT/'runner_initial.py').write_text(Path(__file__).read_text())
    if not (OUT/'design.json').exists():
        write_json(OUT/'design.json',DESIGN)


def prediction_path(month,group,k):
    return OUT/'networks'/f'{month}_s{group}_k{k}.parquet'


def validate_feature_sources(month):
    if month not in STOP_EPOCHS:
        raise ValueError('Unexpected held month')
    manifest_path=FEATURE_DIR/'manifest.json'
    manifest=ts.read(manifest_path)
    if manifest['settings']!=clocks.SETTINGS:
        raise ValueError('Callsign clock builder settings changed')
    if manifest['source_sha256']!=sha256(Path(clocks.__file__)):
        raise ValueError('Callsign clock builder source differs from feature manifest')
    rotation_manifest=clocks.rotation.DEFAULT/'folds.json'
    rotation=ts.read(rotation_manifest)
    if manifest['rotation_manifest_sha256']!=sha256(rotation_manifest):
        raise ValueError('Callsign clock parent rotation manifest changed')
    if manifest['identity_sha256']!=sha256(clocks.rotation.DEFAULT/'identity.parquet'):
        raise ValueError('Callsign clock aircraft identity cache changed')
    if manifest['shared_sha256']!=sha256(FEATURE_DIR/'shared.parquet'):
        raise ValueError('Shared callsign clock table changed')
    sources={part:FEATURE_DIR/f'{month}_{part}.parquet' for part in ['train','held']}
    for part,path in sources.items():
        entry=manifest['folds'][month][part]
        expected=ts.FOLDS[month]+([month] if part=='held' else [])
        if entry['reference_months']!=expected or rotation['folds'][month][part]['reference_months']!=expected:
            raise ValueError('Callsign arrival references differ from the frozen fold')
        rotation_path=clocks.rotation.DEFAULT/f'{month}_{part}.parquet'
        if entry['rotation_sha256']!=sha256(rotation_path):
            raise ValueError('Callsign clock prior arrival source changed')
        if entry['sha256']!=sha256(path):
            raise ValueError('Callsign feature table differs from builder manifest')
    return sources,dict(path=str(manifest_path.relative_to(ROOT)),sha256=sha256(manifest_path),
        fold_sources=manifest['folds'][month],
        raw_callsign_provenance='builder manifest retains source hashes; independent builder audit required')


def validate_completed_month(month):
    path=OUT/'preprocessing'/f'{month}.json'
    records=list((OUT/'networks').glob(f'{month}_s*.json'))
    if not path.exists():
        if records:
            raise ValueError('Existing fits lack preprocessing provenance')
        return
    manifest=ts.read(path)
    _,builder=validate_feature_sources(month)
    if manifest['feature_builder']!=builder:
        raise ValueError('Callsign clock builder provenance changed')
    for source in manifest['feature_sources'].values():
        if sha256(ROOT/source['path'])!=source['sha256']:
            raise ValueError('Callsign clock feature cache changed')
    if sha256(ROOT/manifest['source'])!=manifest['source_sha256']:
        raise ValueError('Cached training/held matrices changed')
    if sha256(OUT/'preprocessing'/f'{month}.pkl')!=manifest['preprocessing_sha256']:
        raise ValueError('Saved preprocessing changed')
    known=set()
    for path in records:
        info=ts.read(path)
        if info['inputs']!=manifest or info['design']!=DESIGN:
            raise ValueError('Existing fit provenance changed')
        for k,entry in info['networks'].items():
            pred=prediction_path(month,info['group'],int(k))
            model=OUT/'models'/f'{month}_s{info["group"]}_k{k}.pt'
            if sha256(pred)!=entry['sha256'] or sha256(model)!=entry['model_sha256']:
                raise ValueError('Saved model or prediction changed')
            known.add(pred)
    if set((OUT/'networks').glob(f'{month}_s*_k*.parquet'))-known:
        raise ValueError('Unrecorded prediction file; inspect interrupted fit')


def fit(groups,nets,months):
    cfg=shared.recipe()
    lookup=shared.weather_all.raw_lookup()
    for month in months:
        validate_completed_month(month)
        if all(prediction_path(month,s,k).exists() for s in groups for k in nets):
            shared.log(f'{month}: requested networks already complete; skip')
            continue
        source=baseline.OUT/'inputs'/f'{month}.pkl'
        data=pd.read_pickle(source)
        names=data['baseline']
        if len(names)!=156 or len(set(names))!=156 or set(FEATURES)&set(names):
            raise ValueError('Original v58 input list changed')
        feature_sources,builder=validate_feature_sources(month)
        train_table=load_features(feature_sources['train'])
        held_table=load_features(feature_sources['held'])
        for part,table in [('train',train_table),('held',held_table)]:
            if len(table)!=builder['fold_sources'][part]['rows']:
                raise ValueError('Callsign clock table row count differs from builder manifest')
        data['X']=add_features(data['X'].loc[:,names],data['train_ids'],train_table)
        data['V']=add_features(data['V'].loc[:,names],data['ids'],held_table)
        del train_table,held_table
        dates=lookup.loc[data['train_ids'],'MVT_TIME_UTC_mvt']
        t=pd.to_datetime(dates,utc=True).dt.strftime('%Y-%m').to_numpy()
        assert set(t)==set(ts.FOLDS[month])
        assert not np.intersect1d(data['train_ids'],data['ids']).size
        raw=lookup.loc[data['ids']].reset_index()
        np.testing.assert_array_equal(data['y'],lookup.loc[data['train_ids'],'taxi_sec'].to_numpy())
        item=dict(X=data['X'],Vn=data['V'],y=data['y'],raw=raw,tmonth=t,meta=dict(month=month))
        shared.log(f'{month}: encode {len(data["X"]):,} training / {len(data["V"]):,} held rows')
        d=DirectData(item,'cuda')
        pre=OUT/'preprocessing'/f'{month}.pkl'
        shared.atomic_pickle(pre,d.preprocessing())
        manifest=dict(source=str(source.relative_to(ROOT)),source_sha256=sha256(source),
            selected_baseline_columns=names,added_features=FEATURES,train_rows=len(d.y),held_rows=len(d.yv),
            feature_builder=builder,
            feature_sources={part:dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))
                              for part,path in feature_sources.items()},
            encoding=d.info,preprocessing_sha256=sha256(pre),
            feature_nonmissing={part:{name:int(frame[name].notna().sum()) for name in FEATURES}
                                for part,frame in [('train',data['X']),('held',data['V'])]})
        write_json(OUT/'preprocessing'/f'{month}.json',manifest)
        for group in groups:
            record=OUT/'networks'/f'{month}_s{group}.json'
            if record.exists():
                info=ts.read(record)
                if info['config']!=cfg or info['inputs']!=manifest or info['design']!=DESIGN:
                    raise ValueError('Saved fit does not match requested recipe/inputs')
            else:
                info=dict(config=cfg,design=DESIGN,group=group,month=month,
                          stop_epoch=STOP_EPOCHS[month],inner_curve=[],stop_minutes=0.,
                          epoch_source=control_identity(),networks={},
                          source_sha256=sha256(Path(__file__)),inputs=manifest)
                write_json(record,info)
            for k in nets:
                dest=prediction_path(month,group,k)
                if dest.exists():
                    continue
                started=time.monotonic()
                model,pred,curve=shared.train(d,np.arange(len(d.y)),group+1000*k,info['stop_epoch'],
                    (d.held,d.yv,d.val_offset))
                model_path=OUT/'models'/f'{month}_s{group}_k{k}.pt'
                model_path.parent.mkdir(parents=True,exist_ok=True)
                temp=model_path.with_suffix('.pt.tmp')
                torch.save(dict(state_dict=model.state_dict(),config=cfg,design=DESIGN,
                                preprocessing=str(pre.relative_to(ROOT))),temp)
                temp.replace(model_path)
                shared.nni.save_network(dest,data['ids'],pred)
                info['networks'][str(k)]=dict(seed=group+1000*k,minutes=(time.monotonic()-started)/60,
                    sha256=sha256(dest),model_sha256=sha256(model_path),held_curve=curve)
                write_json(record,info)
                del model
                torch.cuda.empty_cache()
                shared.log(f'{month} group {group} network {k}: saved')
        del d,data,item,raw,t
        gc.collect()
        torch.cuda.empty_cache()


def score(groups,nets):
    sc,rome,pool,trees=baseline.baseline()
    errors={name:{} for name in ['v58','direct_matched','callsign_clocks','half_callsign_half_v58']}
    by_seed={name:{} for name in errors}
    rows=[]
    for month in ts.SPLITS:
        direct,candidate=[],[]
        for group in groups:
            for k in nets:
                frame=pd.read_parquet(prediction_path(month,group,k))
                np.testing.assert_array_equal(frame.mvt_id.to_numpy(),sc.d[month]['ids'])
                candidate.append(frame.base.to_numpy(float))
                direct.append(shared.direct_member(sc,month,group,k))
        dmean,cmean=np.mean(direct,axis=0),np.mean(candidate,axis=0)
        arms=dict(v58=pool[month],direct_matched=dmean,callsign_clocks=cmean,
                  half_callsign_half_v58=.5*pool[month]+.5*cmean)
        month_errors={name:[] for name in arms}
        for seed in hpo.SEEDS:
            replacements={hpo.MEMBER_OF[p]:trees[p][seed][month] for p in trees}
            for name,neural in arms.items():
                squared=sc.se(month,seed,nn=neural,p=rome[month],**replacements)
                month_errors[name].append(squared)
                by_seed[name].setdefault(str(seed),{})[month]=float(np.sqrt(squared.mean()))
        for name in arms:
            errors[name][month]=np.mean(month_errors[name],axis=0)
        raw=sc.d[month]['held']['raw']
        rows.append(pd.DataFrame(dict(mvt_id=sc.d[month]['ids'],month=month,day=sc.d[month]['day'],
            airport=raw.ADEP_mvt.to_numpy(),y=sc.d[month]['y'],direct_nn=dmean,candidate_nn=cmean,
            **{f'se_{name}':errors[name][month] for name in arms})))
    report=dict(design=DESIGN,groups=groups,nets=nets,pool_size=len(groups)*len(nets),
                source_sha256=sha256(Path(__file__)),scores={},comparisons={})
    for name in errors:
        report['scores'][name]=dict(**sc.cv(errors[name]),by_tree_seed=by_seed[name])
    if abs(report['scores']['v58']['cv']-295.7414415741871)>1e-7:
        raise ValueError('v58 control did not reproduce')
    scored=pd.concat(rows,ignore_index=True)
    for name,control in [('callsign_clocks','direct_matched'),('callsign_clocks','v58'),
                         ('half_callsign_half_v58','v58')]:
        report['comparisons'][f'{name}_vs_{control}']=ts.paired_bootstrap(scored.day.to_numpy(),
            scored.month.to_numpy(),scored[f'se_{control}'].to_numpy(),scored[f'se_{name}'].to_numpy())
    tag='g'+'_'.join(map(str,groups))+'_k'+'_'.join(map(str,nets))
    scored.to_parquet(OUT/f'scored_{tag}.parquet',index=False)
    write_json(OUT/f'score_{tag}.json',report)
    shared.log(json.dumps(report,indent=2))


def self_check():
    X=pd.DataFrame({'ordinary':[7.,8.,9.], 'clock_raw_aobt3_missing':[0,0,1]},index=[4,5,6])
    table=pd.DataFrame({'mvt_id':[10,20,30],
        **{name:np.array([100.,np.nan,300.],dtype=np.float32) for name in FEATURES}}).set_index('mvt_id')
    result=add_features(X,np.array([30,10,20]),table)
    pd.testing.assert_frame_equal(result[X.columns],X)
    for name in FEATURES:
        np.testing.assert_equal(result[name].to_numpy(),[300.,100.,np.nan])
    # Original-AOBT absence must not erase independent transponder clocks.
    table.loc[20,FEATURES]=42.
    result=add_features(X,np.array([30,10,20]),table)
    assert (result.iloc[-1][FEATURES]==42.).all()
    try:
        add_features(X,np.array([30,10,999]),table)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete feature ID coverage must fail')
    control=control_identity()
    shared.log(f'six callsign inputs preserve ID alignment, original columns and independent missingness; control epochs {control["stop_epochs"]}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','fit','score'])
    parser.add_argument('--groups',type=int,nargs='+',choices=[42],default=[42])
    parser.add_argument('--nets',type=int,nargs='+',choices=[0],default=[0])
    parser.add_argument('--months',nargs='+',choices=ts.SPLITS,default=ts.SPLITS)
    args=parser.parse_args()
    if len(set(args.groups))!=len(args.groups) or len(set(args.nets))!=len(args.nets):
        raise ValueError('Duplicate ensemble members')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'networks').mkdir(exist_ok=True)
    if args.stage!='check':
        guard_run()
    torch.set_num_threads(2)
    self_check()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='fit': fit(args.groups,args.nets,args.months)
        elif args.stage=='score': score(args.groups,args.nets)


if __name__=='__main__':
    main()
