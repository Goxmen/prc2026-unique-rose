"""Expand the audited callsign PLE candidate from three to fifteen networks.

Reuse group42 k0/k1/k2 exactly; fit only the24 missing month/member pairs. Each
group uses the historical N09 parent epoch for that month/group. No inner fit or
held-label selection occurs. Original helper code, model/prediction files and
preprocessing bytes remain unchanged; existing metadata only gains new entries.
"""
import argparse
import gc
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
import nn_callsign_clocks_jan_jul as runner
import nn_callsign_pool_resume as resume
from records import sha256,write_json

GROUPS=[42,142,242]
NETS=list(range(5))
EPOCHS={'2025-01':{'42':21,'142':13,'242':17},
        '2025-07':{'42':17,'142':16,'242':14}}
DESIGN=dict(parent='v58',candidate='original156 plus six callsign clock inputs',
    groups=GROUPS,nets=NETS,pool_size=15,seed_formula='group+1000*k',
    epochs=EPOCHS,epoch_policy='frozen historical N09 month/group values; no inner fit',
    inherited_members={'42':[0,1,2]},new_fits=24,
    preprocessing='validate regenerated state exactly and retain original saved bytes',
    neural_share=.71,arms=['callsign15','half_callsign15_half_v58'],
    exploratory_reused_development_months=True)
MANIFEST=runner.OUT/'callsign_pool15_manifest.json'


def historical_control():
    path=runner.hpo.record_path('nn')
    trial=runner.ts.read(path)['trials']['N09']
    cfg=runner.hpo.nn_config(runner.hpo.trial_spec('nn','N09')['overrides'])
    if trial['config']!=cfg:
        raise ValueError('Historical N09 config changed')
    for month in runner.ts.SPLITS:
        for group in GROUPS:
            if trial[month][str(group)]['stop_epoch']!=EPOCHS[month][str(group)]:
                raise ValueError('Historical N09 per-group epoch changed')
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),config=cfg,
        predictions={str(p.relative_to(ROOT)):sha256(p) for p in [
            runner.hpo.pred_path('nn','N09',42),runner.hpo.pred_path('nn','N09_k34',42),
            runner.hpo.pred_path('nn','N09',142),runner.hpo.pred_path('nn','N09',242)]})


def guard():
    runner.guard_run()
    control=historical_control()
    current=dict(design=DESIGN,source_sha256=sha256(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__))
                 for m in [runner,resume,runner.shared,runner.sensor]},
        original_code_identity_sha256=sha256(runner.OUT/'code_identity.json'),
        original_pool3_manifest_sha256=sha256(runner.OUT/'callsign_pool_continuation.json'),
        historical_control=control,inherited={})
    for month in runner.ts.SPLITS:
        runner.validate_completed_month(month)
        record=runner.ts.read(runner.OUT/'networks'/f'{month}_s42.json')
        if record['stop_epoch']!=EPOCHS[month]['42']:
            raise ValueError('Inherited group42 stop epoch changed')
        current['inherited'][month]=dict(
            preprocessing_sha256=sha256(runner.OUT/'preprocessing'/f'{month}.pkl'),
            input_manifest_sha256=sha256(runner.OUT/'preprocessing'/f'{month}.json'),
            networks={str(k):record['networks'][str(k)] for k in [0,1,2]})
        for group in GROUPS:
            path=runner.OUT/'networks'/f'{month}_s{group}.json'
            if path.exists() and runner.ts.read(path)['stop_epoch']!=EPOCHS[month][str(group)]:
                raise ValueError('Existing expanded group has the wrong stopping epoch')
    if MANIFEST.exists():
        if runner.ts.read(MANIFEST)!=current:
            raise ValueError('Expanded design, sources, or inherited artifacts changed')
    else:
        write_json(MANIFEST,current)
    return current


def fit():
    cfg=runner.shared.recipe()
    source_digest=sha256(Path(__file__))
    expansion=dict(manifest=str(MANIFEST.relative_to(ROOT)),sha256=sha256(MANIFEST),
                   source_sha256=source_digest,design=DESIGN)
    lookup=runner.shared.weather_all.raw_lookup()
    for month in runner.ts.SPLITS:
        runner.validate_completed_month(month)
        if all(runner.prediction_path(month,g,k).exists() for g in GROUPS for k in NETS):
            runner.shared.log(f'{month}: all15 networks already complete; skip')
            continue
        source=runner.baseline.OUT/'inputs'/f'{month}.pkl'
        data=pd.read_pickle(source)
        names=data['baseline']
        if len(names)!=156 or len(set(names))!=156 or set(runner.FEATURES)&set(names):
            raise ValueError('Original156-column input contract changed')
        feature_sources,builder=runner.validate_feature_sources(month)
        for part,key,ids in [('train','X','train_ids'),('held','V','ids')]:
            table=runner.load_features(feature_sources[part])
            if len(table)!=builder['fold_sources'][part]['rows']:
                raise ValueError('Feature table row count changed')
            data[key]=runner.add_features(data[key].loc[:,names],data[ids],table)
        del table
        dates=lookup.loc[data['train_ids'],'MVT_TIME_UTC_mvt']
        t=pd.to_datetime(dates,utc=True).dt.strftime('%Y-%m').to_numpy()
        if set(t)!=set(runner.ts.FOLDS[month]) or np.intersect1d(data['train_ids'],data['ids']).size:
            raise ValueError('Training/held row scope changed')
        np.testing.assert_array_equal(data['y'],lookup.loc[data['train_ids'],'taxi_sec'].to_numpy())
        raw=lookup.loc[data['ids']].reset_index()
        item=dict(X=data['X'],Vn=data['V'],y=data['y'],raw=raw,tmonth=t,meta=dict(month=month))
        runner.shared.log(f'{month}: encode once for remaining full-pool members')
        d=runner.DirectData(item,'cuda')
        pre=runner.OUT/'preprocessing'/f'{month}.pkl'
        resume.retain_saved_preprocessing(pre,d.preprocessing())
        inputs=runner.ts.read(runner.OUT/'preprocessing'/f'{month}.json')
        if (inputs['source_sha256']!=sha256(source) or inputs['feature_builder']!=builder
                or inputs['selected_baseline_columns']!=names or inputs['encoding']!=d.info
                or inputs['preprocessing_sha256']!=sha256(pre)):
            raise ValueError('Regenerated expanded inputs differ from original fit')
        for group in GROUPS:
            record=runner.OUT/'networks'/f'{month}_s{group}.json'
            epoch=EPOCHS[month][str(group)]
            if record.exists():
                info=runner.ts.read(record)
                if (info['config']!=cfg or info['inputs']!=inputs or info['design']!=runner.DESIGN
                        or info['stop_epoch']!=epoch):
                    raise ValueError('Existing network group differs from requested frozen recipe')
            else:
                info=dict(config=cfg,design=runner.DESIGN,group=group,month=month,
                    stop_epoch=epoch,inner_curve=[],stop_minutes=0.,
                    epoch_source=dict(control=historical_control(),group=group,month=month,epoch=epoch),
                    networks={},source_sha256=source_digest,inputs=inputs,expansion=expansion)
                write_json(record,info)
            for k in NETS:
                dest=runner.prediction_path(month,group,k)
                if dest.exists():
                    continue
                started=time.monotonic()
                model,pred,curve=runner.shared.train(d,np.arange(len(d.y)),group+1000*k,epoch,
                    (d.held,d.yv,d.val_offset))
                model_path=runner.OUT/'models'/f'{month}_s{group}_k{k}.pt'
                temporary=model_path.with_suffix('.pt.tmp')
                torch.save(dict(state_dict=model.state_dict(),config=cfg,design=runner.DESIGN,
                    preprocessing=str(pre.relative_to(ROOT)),expansion=expansion),temporary)
                temporary.replace(model_path)
                runner.shared.nni.save_network(dest,data['ids'],pred)
                info['networks'][str(k)]=dict(seed=group+1000*k,
                    minutes=(time.monotonic()-started)/60,sha256=sha256(dest),
                    model_sha256=sha256(model_path),held_curve=curve,expansion=expansion)
                write_json(record,info)
                del model;torch.cuda.empty_cache()
                runner.shared.log(f'{month} group{group} k{k}: saved, frozen epoch{epoch}')
        del data,d,item,raw,t
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','fit','score'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    historical_control();resume.self_check()
    if args.stage=='check':
        runner.shared.log('All six historical month/group epochs match;24 missing fits; original model recipe unchanged')
        return
    guard()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='fit':fit()
        else:
            runner.score(GROUPS,NETS)
            tag='g42_142_242_k0_1_2_3_4'
            report=runner.ts.read(runner.OUT/f'score_{tag}.json')
            if abs(report['scores']['direct_matched']['cv']-report['scores']['v58']['cv'])>1e-7:
                raise ValueError('Matched historical15 differs from unchanged v58 pool')
            report['pool_expansion']=runner.ts.read(MANIFEST)
            report['half_matched15_control']='identical to v58: matched15 and parent15 are the same pool'
            write_json(runner.OUT/f'score_{tag}.json',report)


if __name__=='__main__':
    main()
