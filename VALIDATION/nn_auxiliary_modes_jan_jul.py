"""Original N09 mean regression plus fixed auxiliary recorder-mode supervision.

Inputs remain the original156. The mean target remains min(y,10800), standardized.
A tiny zero-initialized head predicts11 ordinal indicators of BLOCK-originalAOBT
and one schedule-copy indicator. Auxiliary labels use ORIGINAL clocks and raw
uncapped y, only within the current training rows; neither labels nor logits enter
model inputs or final predictions. Add0.1 times the equal mean of ordinal/copy
BCE to original MSE. Missing-clock task labels are masked.

Initial mean parameters, predictions and dropout RNG exactly match N09. Epoch
selection uses only original mean RMSE on the last allowed training month.
Save held auxiliary probabilities for diagnosis, without a secondary score arm.
Only seed42 k0 screen is supported here; no upload, full-year or ledger mutation.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
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
from records import sha256,write_json

hpo,ts,baseline=shared.hpo,shared.ts,shared.baseline
OUT=ts.RUNS/'v3_20261010_nn_auxiliary_modes_jan_jul'
WITNESSES=[]
FEATURES=[]
AUX_WEIGHT=0.1
THRESHOLDS=np.array([-1800,-600,-300,-120,-60,0,60,120,300,600,1800],dtype=np.float32)
DESIGN=dict(parent='v58',control='HPO N09 matched seed42 k0',
    architecture='original N09 mean plus zero-initialized shared-trunk auxiliary12-logit head',
    auxiliary_parameters=3084,auxiliary_weight=AUX_WEIGHT,
    ordinal_thresholds_sec=THRESHOLDS.tolist(),
    ordinal_target='I((takeoff - original AOBT3 - raw uncapped y)>threshold)',
    copy_target='I(abs(takeoff - original schedule - raw uncapped y)<=60)',
    auxiliary_loss='0.1*(0.5*mean valid ordinal BCE +0.5*mean valid schedule-copy BCE); empty task zero',
    target='direct min(taxi_sec,10800)',added_features=[],network_share=.71,
    neural_arms=['auxiliary_modes','half_auxiliary_half_v58'],
    heldout_months=ts.SPLITS,stop_choice='last training month mean-output RMSE only',
    train_input_cache=str(baseline.OUT/'inputs'),weather_inputs_used=False,
    groups=[42],networks_per_group=1,auxiliary_logits_used_at_inference=False,
    exploratory_reused_development_months=True)


def mode_sources():
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'data').glob('training_2025-*.parquet'))}


def mode_lookup():
    import duckdb
    con=duckdb.connect(config={'threads':2})
    con.execute("SET TimeZone='UTC'")
    frame=con.execute("SELECT MVT_ID_mvt AS mvt_id,epoch(MVT_TIME_UTC_mvt) t, "
        "epoch(AOBT_3_flt) aobt,epoch(SCHED_TIME_UTC_mvt) schedule "
        "FROM read_parquet(?) WHERE PHASE_mvt='DEP'",
        [str(ROOT/'data/training_2025-*.parquet')]).df().set_index('mvt_id')
    con.close()
    if not frame.index.is_unique:
        raise ValueError('Original-clock lookup has duplicate movement IDs')
    return frame


def mode_targets(clocks,y):
    y=np.asarray(y,float)
    if len(clocks)!=len(y):
        raise ValueError('Clock supervision rows differ from ordinary target rows')
    t=clocks.t.to_numpy(float)
    aobt=clocks.aobt.to_numpy(float)
    schedule=clocks.schedule.to_numpy(float)
    actual=np.isfinite(t)&np.isfinite(aobt)&np.isfinite(y)
    scheduled=np.isfinite(t)&np.isfinite(schedule)&np.isfinite(y)
    offset=t-aobt-y
    ordinal=np.where(actual[:,None],offset[:,None]>THRESHOLDS[None,:],False)
    copy=np.where(scheduled,np.abs(t-schedule-y)<=60,False)
    labels=np.column_stack([ordinal,copy]).astype(np.float32)
    mask=np.column_stack([actual,scheduled]).astype(np.float32)
    return labels,mask


def add_features(X):
    return X


class DirectData(sensor.DirectData):
    def __init__(self,item,device):
        super().__init__(item,device)
        target,mask=mode_targets(item['original_clocks'],self.y)
        self.mode_labels=torch.from_numpy(target).to(device)
        self.mode_mask=torch.from_numpy(mask).to(device)
        self.info['auxiliary_valid_actual']=int(mask[:,0].sum())
        self.info['auxiliary_valid_schedule']=int(mask[:,1].sum())
        self.info['auxiliary_schedule_copies']=int((target[:,-1]*mask[:,1]).sum())

    def preprocessing(self):
        result=super().preprocessing()
        result['design']=DESIGN
        return result


class AuxiliaryModeNet(shared.nni.PleNet):
    def __init__(self,cards,bins,n_plain):
        super().__init__(cards,bins,n_plain)
        width=self.mlp[-1].in_features
        self.aux_weight=torch.nn.Parameter(torch.zeros(len(THRESHOLDS)+1,width))
        self.aux_bias=torch.nn.Parameter(torch.zeros(len(THRESHOLDS)+1))

    def forward_parts(self,cat,g,plain):
        z=self.inputs(cat,g,plain)
        for layer in list(self.mlp)[:-1]:
            z=layer(z)
        mean=self.mlp[-1](z)
        logits=torch.nn.functional.linear(z,self.aux_weight,self.aux_bias)
        return mean,logits

    def forward(self,cat,g,plain):
        return self.forward_parts(cat,g,plain)[0]


def auxiliary_loss(logits,labels,mask):
    losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    ordinal=(losses[:,:-1].mean(1)*mask[:,0]).sum()/mask[:,0].sum().clamp_min(1.)
    copy=(losses[:,-1]*mask[:,1]).sum()/mask[:,1].sum().clamp_min(1.)
    return AUX_WEIGHT*.5*(ordinal+copy)


def predict_auxiliary(model,values):
    previous=model.training
    model.eval()
    with torch.no_grad():
        chunks=[torch.sigmoid(model.forward_parts(*(v[i:i+65536] for v in values))[1]).cpu()
                for i in range(0,len(values[0]),65536)]
    model.train(previous)
    return torch.cat(chunks).numpy()


def train(data,rows,seed,stop,monitor):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=AuxiliaryModeNet(data.cards,data.bins,data.train[2].shape[1]).to(data.device)
    opt=torch.optim.AdamW(model.parameters(),lr=shared.nni.NEW['ple']['lr'],
                         weight_decay=shared.nni.NEW['ple']['weight_decay'])
    batch,count=shared.nni.NEW['batch'],len(rows)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=shared.nni.NEW['ple']['lr'],
                  total_steps=shared.nni.NEW['epochs']*math.ceil(count/batch))
    generator=torch.Generator(device=data.device).manual_seed(seed)
    indices=torch.from_numpy(rows).to(data.device)
    curve=[]
    for epoch in range(stop):
        started,total,total_auxiliary=time.monotonic(),0.,0.
        perm=indices[torch.randperm(count,device=data.device,generator=generator)]
        for i in range(0,count,batch):
            b=perm[i:i+batch]
            output,logits=model.forward_parts(*(a[b] for a in data.train))
            mse=torch.nn.functional.mse_loss(output,data.ty[b,None].expand_as(output))
            auxiliary=auxiliary_loss(logits,data.mode_labels[b],data.mode_mask[b])
            loss=mse+auxiliary
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
            total+=float(mse.detach())*len(b)
            total_auxiliary+=float(auxiliary.detach())*len(b)
        pred=shared.predict(model,monitor[0],data.sd,data.mu,monitor[2])
        curve.append(dict(epoch=epoch+1,train_mse_std=total/count,
            auxiliary_loss=total_auxiliary/count,rmse=shared.nni.rmse(monitor[1],pred),
            seconds=time.monotonic()-started))
        shared.log(f'epoch {epoch+1}: train {total/count:.4f}; auxiliary '
                   f'{total_auxiliary/count:.6f}; monitored {curve[-1]["rmse"]:.4f}')
    return model,pred,curve


def identity():
    parsed=ast.parse(Path(__file__).read_text())
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
              and n.name in {'add_features','mode_sources','mode_lookup','mode_targets','DirectData','AuxiliaryModeNet','auxiliary_loss','predict_auxiliary','train','fit'}]
    digest=hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),
                                 include_attributes=False).encode()).hexdigest()
    return dict(feature_data_ast_sha256=digest,witnesses=WITNESSES,features=FEATURES,
        auxiliary_weight=AUX_WEIGHT,thresholds=THRESHOLDS.tolist(),helpers={str(Path(module.__file__).relative_to(ROOT)):sha256(Path(module.__file__))
                 for module in [shared,sensor,shared.nni,shared.nnj,hpo]})


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


def validate_completed_month(month):
    path=OUT/'preprocessing'/f'{month}.json'
    records=list((OUT/'networks').glob(f'{month}_s*.json'))
    if not path.exists():
        if records:
            raise ValueError('Existing fits lack preprocessing provenance')
        return
    manifest=ts.read(path)
    if manifest['original_clock_sources']!=mode_sources():
        raise ValueError('Original training-clock sources changed')
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
            aux=OUT/'auxiliary'/f'{month}_s{info["group"]}_k{k}.parquet'
            if sha256(aux)!=entry['auxiliary_sha256']:
                raise ValueError('Saved auxiliary predictions changed')
            known.add(pred)
    if set((OUT/'networks').glob(f'{month}_s*_k*.parquet'))-known:
        raise ValueError('Unrecorded prediction file; inspect interrupted fit')


def fit(groups,nets,months):
    cfg=shared.recipe()
    lookup=shared.weather_all.raw_lookup()
    original_clocks=mode_lookup()
    clock_sources=mode_sources()
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
        data['X']=add_features(data['X'].loc[:,names])
        data['V']=add_features(data['V'].loc[:,names])
        dates=lookup.loc[data['train_ids'],'MVT_TIME_UTC_mvt']
        t=pd.to_datetime(dates,utc=True).dt.strftime('%Y-%m').to_numpy()
        assert set(t)==set(ts.FOLDS[month])
        assert not np.intersect1d(data['train_ids'],data['ids']).size
        raw=lookup.loc[data['ids']].reset_index()
        np.testing.assert_array_equal(data['y'],lookup.loc[data['train_ids'],'taxi_sec'].to_numpy())
        item=dict(X=data['X'],Vn=data['V'],y=data['y'],raw=raw,tmonth=t,meta=dict(month=month),
                  original_clocks=original_clocks.loc[data['train_ids']])
        shared.log(f'{month}: encode {len(data["X"]):,} training / {len(data["V"]):,} held rows')
        d=DirectData(item,'cuda')
        pre=OUT/'preprocessing'/f'{month}.pkl'
        shared.atomic_pickle(pre,d.preprocessing())
        manifest=dict(source=str(source.relative_to(ROOT)),source_sha256=sha256(source),
            original_clock_sources=clock_sources,selected_baseline_columns=names,added_features=FEATURES,train_rows=len(d.y),held_rows=len(d.yv),
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
                started=time.monotonic()
                model,_,curve=train(d,d.inner_train,group,cfg['epochs'],
                    (d.subset(d.inner_val),d.y[d.inner_val],d.offset[d.inner_val]))
                stop=min(curve,key=lambda entry:entry['rmse'])['epoch']
                info=dict(config=cfg,design=DESIGN,group=group,month=month,stop_epoch=stop,
                          inner_curve=curve,stop_minutes=(time.monotonic()-started)/60,
                          networks={},source_sha256=sha256(Path(__file__)),inputs=manifest)
                write_json(record,info)
                del model
                torch.cuda.empty_cache()
            for k in nets:
                dest=prediction_path(month,group,k)
                if dest.exists():
                    continue
                started=time.monotonic()
                model,pred,curve=train(d,np.arange(len(d.y)),group+1000*k,info['stop_epoch'],
                    (d.held,d.yv,d.val_offset))
                model_path=OUT/'models'/f'{month}_s{group}_k{k}.pt'
                model_path.parent.mkdir(parents=True,exist_ok=True)
                temp=model_path.with_suffix('.pt.tmp')
                torch.save(dict(state_dict=model.state_dict(),config=cfg,design=DESIGN,
                                preprocessing=str(pre.relative_to(ROOT))),temp)
                temp.replace(model_path)
                shared.nni.save_network(dest,data['ids'],pred)
                aux=predict_auxiliary(model,d.held)
                aux_path=OUT/'auxiliary'/f'{month}_s{group}_k{k}.parquet'
                aux_path.parent.mkdir(parents=True,exist_ok=True)
                aux_frame=pd.DataFrame({'mvt_id':data['ids'], 'p_schedule_copy':aux[:,-1],
                    **{f'p_offset_gt_{int(v)}':aux[:,j] for j,v in enumerate(THRESHOLDS)}})
                aux_frame.to_parquet(aux_path,index=False)
                info['networks'][str(k)]=dict(seed=group+1000*k,minutes=(time.monotonic()-started)/60,
                    sha256=sha256(dest),model_sha256=sha256(model_path),auxiliary_sha256=sha256(aux_path),held_curve=curve)
                write_json(record,info)
                del model
                torch.cuda.empty_cache()
                shared.log(f'{month} group {group} network {k}: saved')
        del d,data,item,raw,t
        gc.collect()
        torch.cuda.empty_cache()


def score(groups,nets):
    sc,rome,pool,trees=baseline.baseline()
    errors={name:{} for name in ['v58','direct_matched','auxiliary_modes','half_auxiliary_half_v58']}
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
        arms=dict(v58=pool[month],direct_matched=dmean,auxiliary_modes=cmean,
                  half_auxiliary_half_v58=.5*pool[month]+.5*cmean)
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
    for name,control in [('auxiliary_modes','direct_matched'),('auxiliary_modes','v58'),
                         ('half_auxiliary_half_v58','v58')]:
        report['comparisons'][f'{name}_vs_{control}']=ts.paired_bootstrap(scored.day.to_numpy(),
            scored.month.to_numpy(),scored[f'se_{control}'].to_numpy(),scored[f'se_{name}'].to_numpy())
    tag='g'+'_'.join(map(str,groups))+'_k'+'_'.join(map(str,nets))
    scored.to_parquet(OUT/f'scored_{tag}.parquet',index=False)
    write_json(OUT/f'score_{tag}.json',report)
    shared.log(json.dumps(report,indent=2))


def self_check():
    shared.recipe()
    cards=[4,11]
    bins=[torch.tensor([-2.,-1.,0.,1.,2.])]
    torch.manual_seed(42)
    control=shared.nni.PleNet(cards,bins,2).eval()
    original_rng=torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate=AuxiliaryModeNet(cards,bins,2).eval()
    torch.testing.assert_close(torch.get_rng_state(),original_rng,rtol=0,atol=0)
    for name,value in control.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name],value,rtol=0,atol=0)
    cat=torch.tensor([[0,0],[1,1],[2,5],[3,10]],dtype=torch.long)
    g=torch.tensor([[-1.],[0.],[.4],[1.5]])
    plain=torch.tensor([[0.,1.],[1.,0.],[0.,0.],[1.,1.]])
    with torch.no_grad():
        original=control(cat,g,plain)
        fitted,logits=candidate.forward_parts(cat,g,plain)
        torch.testing.assert_close(original,fitted,rtol=0,atol=0)
        assert not torch.count_nonzero(logits)
        control.train();candidate.train()
        torch.set_rng_state(original_rng)
        original=control(cat,g,plain);after_original=torch.get_rng_state().clone()
        torch.set_rng_state(original_rng)
        fitted=candidate(cat,g,plain)
        torch.testing.assert_close(original,fitted,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),after_original,rtol=0,atol=0)
    clocks=pd.DataFrame({'t':[20000.,20000.,20000.], 'aobt':[5000.,np.nan,19800.],
                         'schedule':[5000.,19000.,np.nan]})
    labels,mask=mode_targets(clocks,np.array([15000.,1000.,100.]))
    np.testing.assert_array_equal(labels[0,:-1],(0>THRESHOLDS).astype(float))
    np.testing.assert_array_equal(mask,[[1,1],[0,1],[1,0]])
    np.testing.assert_array_equal(labels[:,-1],[1,1,0])
    np.testing.assert_array_equal(labels[2,:-1],(100>THRESHOLDS).astype(float))
    logits=torch.zeros(3,12,requires_grad=True)
    loss=auxiliary_loss(logits,torch.from_numpy(labels),torch.from_numpy(mask))
    torch.testing.assert_close(loss,torch.tensor(.1*math.log(2)),rtol=1e-6,atol=1e-7)
    zero=auxiliary_loss(logits,torch.from_numpy(labels),torch.zeros(3,2))
    assert float(zero.detach())==0.
    zero.backward()
    assert torch.count_nonzero(logits.grad)==0
    shared.log('initial N09 mean weights/predictions/dropout RNG identical; raw uncapped mode labels, masks and fixed auxiliary loss pass')


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
    guard_run()
    torch.set_num_threads(2)
    self_check()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='fit': fit(args.groups,args.nets,args.months)
        elif args.stage=='score': score(args.groups,args.nets)


if __name__=='__main__':
    main()
