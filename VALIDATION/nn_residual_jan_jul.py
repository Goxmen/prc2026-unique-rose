"""Bounded actual-AOBT3 residual PLE, a frozen architecture experiment on v58.

The only recipe change is the target parameterization: subtract a bounded *actual*
AOBT3 clock from the clipped label and add it back before clipping predictions.
The offset is clip(mvt_minus_aobt3_sec, 0, 7200) when the original AOBT3 is
available, otherwise zero. No schedule or other-clock fallback is permitted.
Signed residual outputs are unrestricted: stale-hour clocks can be cancelled.

Inputs, ordinary-row membership, target cap, encoder, PLE/backbone and training
schedule match HPO N09 (v58). The last training month chooses the stopping epoch;
held-out labels are read only for monitoring/scoring. Reuse the audited weather
experiment's cached matrices after selecting its original 156 baseline columns.

Staged execution (one GPU process; no uploads):
  fit --groups 42 --nets 0                 # two held-out months, one network each
  score --groups 42 --nets 0               # matched direct control and v58
  fit --groups 42 142 242 --nets 0 1 2 3 4  # resume to the fixed 15-network pool
  score --groups 42 142 242 --nets 0 1 2 3 4

The scored arms are fixed: residual only, and a 50/50 neural mix of the residual
pool and v58's saved 15-network pool; total network share stays 0.71. Small-pool
scores are exploratory. The matched-size direct N09 control separates the target
change from ensemble-size differences. Every Jan/Jul departure is always scored.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline'), str(ROOT/'VALIDATION/tools')]
import hpo_jan_jul as hpo
import nn_improve_jan_jul as nni
import nn_jan_jul as nnj
import open_meteo_all as weather_all
import open_meteo_jan_jul as baseline
import tuning_seasonal as ts
from records import sha256, write_json

OUT = ts.RUNS/'v3_20261010_nn_residual_jan_jul'
OFFSET_LIMIT = 7200.0
DESIGN = dict(parent='v58', control='HPO N09 matched network seeds',
    offset='clip(mvt_minus_aobt3_sec,0,7200) if original AOBT3 is present; else 0',
    offset_required=['mvt_minus_aobt3_sec','clock_aobt3_available','clock_raw_aobt3_missing'],
    target='min(taxi_sec,10800) - offset', reconstruction='max(offset + residual,0)',
    residual_bounds=None, network_share=.71, neural_arms=['residual','half_residual_half_v58'],
    heldout_months=ts.SPLITS, stop_choice='last training month only',
    train_input_cache=str(baseline.OUT/'inputs'), weather_inputs_used=False,
    groups=[42,142,242], networks_per_group=5,
    exploratory_reused_development_months=True)


def log(message):
    print(time.strftime('[%H:%M:%S] ') + str(message), flush=True)


def actual_offset(X):
    """Input-only offset; finite, bounded, and never filled from another clock."""
    gap = X['mvt_minus_aobt3_sec'].to_numpy(float)
    present = ((X['clock_aobt3_available'].to_numpy() == 1)
               & (X['clock_raw_aobt3_missing'].to_numpy() == 0) & np.isfinite(gap))
    return np.where(present, np.clip(gap, 0.0, OFFSET_LIMIT), 0.0)


def atomic_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload=pickle.dumps(obj,protocol=pickle.HIGHEST_PROTOCOL)
    if path.exists():
        if path.read_bytes()!=payload:
            raise ValueError(f'Refusing to overwrite different saved preprocessing: {path}')
        return
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        stream.write(payload)
    temp.replace(path)


def code_identity():
    names={'actual_offset','recipe','ResidualData','predict','train'}
    parsed=ast.parse(Path(__file__).read_text())
    selected=[node for node in parsed.body
              if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    digest=hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),
                                 include_attributes=False).encode()).hexdigest()
    return dict(recipe_ast_sha256=digest,offset_limit=OFFSET_LIMIT,
        helpers={str(Path(module.__file__).relative_to(ROOT)):sha256(Path(module.__file__))
                 for module in [nnj,nni,hpo]})


def guard_run():
    design_path=OUT/'design.json'
    if design_path.exists() and ts.read(design_path)!=DESIGN:
        raise ValueError('Saved experiment design differs; use a separate run directory')
    identity=code_identity()
    path=OUT/'code_identity.json'
    if path.exists():
        previous=ts.read(path)
        if any(previous.get(key)!=value for key,value in identity.items()):
            raise ValueError('Recipe implementation or training helpers changed since this run began')
    else:
        if list((OUT/'networks').glob('*.parquet')):
            raise ValueError('Cannot bless existing predictions without recorded implementation identity')
        (OUT/'runner_initial.py').write_text(Path(__file__).read_text())
        write_json(path,dict(**identity,initial_source_sha256=sha256(Path(__file__))))


def validate_completed_month(month):
    """Do not reuse a completed fit against changed inputs, transforms or predictions."""
    prep=OUT/'preprocessing'/f'{month}.json'
    records=list((OUT/'networks').glob(f'{month}_s*.json'))
    if not prep.exists():
        if records:
            raise ValueError('Fit records exist without their preprocessing provenance')
        return
    manifest=ts.read(prep)
    source=ROOT/manifest['source']
    if sha256(source)!=manifest['source_sha256']:
        raise ValueError(f'Cached training/held inputs changed: {source}')
    prepath=OUT/'preprocessing'/f'{month}.pkl'
    if sha256(prepath)!=manifest['preprocessing_sha256']:
        raise ValueError(f'Saved preprocessing changed: {prepath}')
    for record in records:
        info=ts.read(record)
        if info['inputs']!=manifest or info['design']!=DESIGN:
            raise ValueError(f'Fit provenance changed: {record}')
        for k,entry in info['networks'].items():
            prediction=prediction_path(month,info['group'],int(k))
            model=OUT/'models'/f'{month}_s{info["group"]}_k{k}.pt'
            if sha256(prediction)!=entry['sha256'] or sha256(model)!=entry['model_sha256']:
                raise ValueError(f'Completed model/prediction changed: {record}, network {k}')
    known={prediction_path(month,ts.read(record)['group'],int(k))
           for record in records for k in ts.read(record)['networks']}
    if set((OUT/'networks').glob(f'{month}_s*_k*.parquet'))-known:
        raise ValueError(f'Unrecorded predictions found for {month}; inspect interrupted fit')


def recipe():
    cfg = hpo.nn_config(hpo.trial_spec('nn','N09')['overrides'])
    assert cfg == dict(hidden=[512,512,256], dropout=.1, lr=.002, weight_decay=1e-5,
                      batch=2048, epochs=24, bins=48, d_ple=16, label_clip=10800.)
    hpo.apply_nn(cfg, nnj, nni)
    return cfg


class ResidualData:
    """Identical v58 input encoding, with a residual response and saved encoder."""

    def __init__(self, item, device):
        self.encoder = nnj.Encoder(item['X'])
        enc = self.encoder
        cat, num = enc.transform(item['X'])
        vcat, vnum = enc.transform(item['Vn'])
        g, plain, j = [], [], 0
        for name in enc.nums:
            g.append(j)
            plain.append(j+1)
            j += 2
            if name in enc.flags:
                plain.append(j)
                j += 1
        if j != num.shape[1]:
            raise ValueError('Encoder column layout changed')
        edges = np.quantile(num[:,g],np.linspace(0,1,nni.NEW['bins']+1),axis=0)
        bins, ple = [], []
        for i,col in enumerate(g):
            values = np.unique(edges[:,i].astype(np.float32))
            if len(values) >= 3:
                bins.append(torch.from_numpy(values))
                ple.append(col)
            else:
                plain.append(col)
        self.bins,self.cards,self.ple,self.plain = bins,enc.cards,ple,plain
        self.columns = list(item['X'].columns)
        self.offset = actual_offset(item['X'])
        self.val_offset = actual_offset(item['Vn'])
        self.y = np.asarray(item['y'],float)
        self.yv = item['raw'].taxi_sec.to_numpy(float)
        target = np.minimum(self.y,nnj.RECIPE['label_clip'])-self.offset
        self.mu,self.sd = float(target.mean()),float(target.std())
        if not self.sd > 0:
            raise ValueError('Residual target has zero variance')
        tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.train = (tensor(cat),tensor(num[:,ple]),tensor(num[:,plain]))
        self.held = (tensor(vcat),tensor(vnum[:,ple]),tensor(vnum[:,plain]))
        self.ty = tensor(((target-self.mu)/self.sd).astype(np.float32))
        last = max(ts.FOLDS[item['meta']['month']])
        self.inner_val = np.flatnonzero(item['tmonth']==last)
        self.inner_train = np.flatnonzero(item['tmonth']!=last)
        self.device = device
        self.info = dict(numeric=len(enc.nums),ple_inputs=len(ple),plain_inputs=len(plain),
            categoricals=enc.cats,cards=enc.cards,bins_per_input=[len(b)-1 for b in bins],
            inner_month=last,inner_val_rows=len(self.inner_val),inner_train_rows=len(self.inner_train),
            offset_nonzero=int((self.offset>0).sum()),offset_at_cap=int((self.offset==OFFSET_LIMIT).sum()),
            residual_mu=self.mu,residual_sd=self.sd)

    def subset(self, rows):
        indices = torch.from_numpy(rows).to(self.device)
        return tuple(a[indices] for a in self.train)

    def preprocessing(self):
        return dict(encoder=self.encoder,ple=self.ple,plain=self.plain,columns=self.columns,
                    bins=self.bins,cards=self.cards,mu=self.mu,sd=self.sd,design=DESIGN,info=self.info)


def predict(model, inputs, sd, mu, offset, chunk=65536):
    model.eval()
    with torch.no_grad():
        count=len(inputs[0])
        standardized=torch.cat([model(*(a[i:i+chunk] for a in inputs)).mean(1)
                                for i in range(0,count,chunk)])
    model.train()
    # Reconstruction precedes clipping; signed residuals must never be clipped.
    return np.maximum(standardized.cpu().numpy().astype(float)*sd+mu+offset,0)


def train(data, rows, seed, stop, monitor):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=nni.PleNet(data.cards,data.bins,data.train[2].shape[1]).to(data.device)
    opt=torch.optim.AdamW(model.parameters(),lr=nni.NEW['ple']['lr'],
                         weight_decay=nni.NEW['ple']['weight_decay'])
    batch,count=nni.NEW['batch'],len(rows)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=nni.NEW['ple']['lr'],
                  total_steps=nni.NEW['epochs']*math.ceil(count/batch))
    generator=torch.Generator(device=data.device).manual_seed(seed)
    indices=torch.from_numpy(rows).to(data.device)
    curve=[]
    for epoch in range(stop):
        started,total=time.monotonic(),0.
        perm=indices[torch.randperm(count,device=data.device,generator=generator)]
        for i in range(0,count,batch):
            b=perm[i:i+batch]
            output=model(*(a[b] for a in data.train))
            loss=torch.nn.functional.mse_loss(output,data.ty[b,None].expand_as(output))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
            total+=float(loss)*len(b)
        pred=predict(model,monitor[0],data.sd,data.mu,monitor[2])
        curve.append(dict(epoch=epoch+1,train_mse_std=total/count,
                          rmse=nni.rmse(monitor[1],pred),seconds=time.monotonic()-started))
        log(f'epoch {epoch+1}: train {total/count:.4f}; monitored {curve[-1]["rmse"]:.4f}')
    return model,pred,curve


def prediction_path(month,group,k):
    return OUT/'networks'/f'{month}_s{group}_k{k}.parquet'


def fit(groups,nets,months):
    cfg=recipe()
    lookup=weather_all.raw_lookup()
    for month in months:
        validate_completed_month(month)
        if all(prediction_path(month,s,k).exists() for s in groups for k in nets):
            log(f'{month}: requested networks already complete; skip')
            continue
        source=baseline.OUT/'inputs'/f'{month}.pkl'
        data=pd.read_pickle(source)
        names=data['baseline']
        if len(names)!=156 or len(set(names))!=156:
            raise ValueError('v58 baseline column list changed')
        data['X']=data['X'].loc[:,names]
        data['V']=data['V'].loc[:,names]
        t=pd.to_datetime(lookup.loc[data['train_ids'],'MVT_TIME_UTC_mvt'],utc=True).dt.strftime('%Y-%m').to_numpy()
        assert set(t)==set(ts.FOLDS[month])
        assert not np.intersect1d(data['train_ids'],data['ids']).size
        raw=lookup.loc[data['ids']].reset_index()
        np.testing.assert_array_equal(data['y'],lookup.loc[data['train_ids'],'taxi_sec'].to_numpy())
        item=dict(X=data['X'],Vn=data['V'],y=data['y'],raw=raw,tmonth=t,meta=dict(month=month))
        log(f'{month}: encode {len(data["X"]):,} training / {len(data["V"]):,} held rows')
        d=ResidualData(item,'cuda')
        pre=OUT/'preprocessing'/f'{month}.pkl'
        atomic_pickle(pre,d.preprocessing())
        input_manifest=dict(source=str(source.relative_to(ROOT)),source_sha256=sha256(source),
            selected_columns=names,train_rows=len(d.y),held_rows=len(d.yv),encoding=d.info,
            preprocessing_sha256=sha256(pre))
        write_json(OUT/'preprocessing'/f'{month}.json',input_manifest)
        for group in groups:
            record=OUT/'networks'/f'{month}_s{group}.json'
            if record.exists():
                info=ts.read(record)
                if info['config']!=cfg or info['design']!=DESIGN:
                    raise ValueError('Existing fit does not match frozen recipe')
            else:
                started=time.monotonic()
                model,_,curve=train(d,d.inner_train,group,cfg['epochs'],
                    (d.subset(d.inner_val),d.y[d.inner_val],d.offset[d.inner_val]))
                stop=min(curve,key=lambda entry:entry['rmse'])['epoch']
                info=dict(config=cfg,design=DESIGN,group=group,month=month,stop_epoch=stop,
                          inner_curve=curve,stop_minutes=(time.monotonic()-started)/60,
                          networks={},source_sha256=sha256(Path(__file__)),inputs=input_manifest)
                write_json(record,info)
                del model
                torch.cuda.empty_cache()
            for k in nets:
                path=prediction_path(month,group,k)
                if path.exists():
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
                nni.save_network(path,data['ids'],pred)
                info['networks'][str(k)]=dict(seed=group+1000*k,minutes=(time.monotonic()-started)/60,
                    sha256=sha256(path),model_sha256=sha256(model_path),held_curve=curve)
                write_json(record,info)
                del model
                torch.cuda.empty_cache()
                log(f'{month} group {group} network {k}: saved')
        del d,data,item,raw,t
        gc.collect()
        torch.cuda.empty_cache()


def direct_member(sc,month,group,k):
    trial='N09_k34' if group==42 and k>=3 else 'N09'
    return sc.frame('nn',trial,group)[month][f'n{k}'].to_numpy(float)


def score(groups,nets):
    sc,rome,pool,trees=baseline.baseline()
    errors={name:{} for name in ['v58','direct_matched','residual','half_residual_half_v58']}
    by_seed={name:{} for name in errors}
    rows=[]
    for month in ts.SPLITS:
        direct=[]
        residual=[]
        for group in groups:
            for k in nets:
                frame=pd.read_parquet(prediction_path(month,group,k))
                np.testing.assert_array_equal(frame.mvt_id.to_numpy(),sc.d[month]['ids'])
                residual.append(frame.base.to_numpy(float))
                direct.append(direct_member(sc,month,group,k))
        dmean,rmean=np.mean(direct,axis=0),np.mean(residual,axis=0)
        arms=dict(v58=pool[month],direct_matched=dmean,residual=rmean,
                  half_residual_half_v58=.5*pool[month]+.5*rmean)
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
            airport=raw.ADEP_mvt.to_numpy(),y=sc.d[month]['y'],direct_nn=dmean,residual_nn=rmean,
            **{f'se_{name}':errors[name][month] for name in arms})))
    report=dict(design=DESIGN,groups=groups,nets=nets,pool_size=len(groups)*len(nets),
                source_sha256=sha256(Path(__file__)),scores={},comparisons={})
    for name in errors:
        report['scores'][name]=dict(**sc.cv(errors[name]),by_tree_seed=by_seed[name])
    if abs(report['scores']['v58']['cv']-295.7414415741871)>1e-7:
        raise ValueError('v58 control did not reproduce')
    scored=pd.concat(rows,ignore_index=True)
    for name,control in [('residual','direct_matched'),('residual','v58'),('half_residual_half_v58','v58')]:
        report['comparisons'][f'{name}_vs_{control}']=ts.paired_bootstrap(scored.day.to_numpy(),
            scored.month.to_numpy(),scored[f'se_{control}'].to_numpy(),scored[f'se_{name}'].to_numpy())
    tag='g'+'_'.join(map(str,groups))+'_k'+'_'.join(map(str,nets))
    scored.to_parquet(OUT/f'scored_{tag}.parquet',index=False)
    write_json(OUT/f'score_{tag}.json',report)
    log(json.dumps(report,indent=2))


def self_check():
    """Exercise the mechanism before any expensive fit; no real data needed."""
    frame=pd.DataFrame(dict(mvt_minus_aobt3_sec=[900,-300,90000,np.nan,1800,7200,1800],
        clock_aobt3_available=[1,1,1,1,1,1,0],clock_raw_aobt3_missing=[0,0,0,0,1,0,0]))
    np.testing.assert_array_equal(actual_offset(frame),[900,0,7200,0,0,7200,0])
    assert not any('sched' in name for name in DESIGN['offset_required'])
    # A negative residual cancels a stale 3600-second clock; only final taxi time clips.
    reconstructed=np.maximum(np.array([-3600.,-8000.,500.])+np.array([4500.,7200.,0.]),0.)
    np.testing.assert_array_equal(reconstructed,[900.,0.,500.])
    log('offset, provenance, signed residual reconstruction checks passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','fit','score'])
    parser.add_argument('--groups',type=int,nargs='+',choices=hpo.SEEDS,default=[42])
    parser.add_argument('--nets',type=int,nargs='+',choices=list(range(5)),default=[0])
    parser.add_argument('--months',nargs='+',choices=ts.SPLITS,default=ts.SPLITS)
    args=parser.parse_args()
    if len(set(args.groups))!=len(args.groups) or len(set(args.nets))!=len(args.nets):
        raise ValueError('Duplicate groups or network indices would change ensemble weights')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'networks').mkdir(exist_ok=True)
    guard_run()
    if not (OUT/'design.json').exists():
        write_json(OUT/'design.json',DESIGN)
    torch.set_num_threads(2)
    self_check()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='fit': fit(args.groups,args.nets,args.months)
        elif args.stage=='score': score(args.groups,args.nets)


if __name__=='__main__':
    main()
