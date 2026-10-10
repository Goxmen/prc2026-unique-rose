"""v58 PLE trunk with small airport-specific residual output heads.

Keep the original 156 inputs, direct clipped target, shared PLE/trunk/global output,
and all N09 training settings. Add one linear correction on the final 256-dimensional
hidden representation per airport. Initialize those corrections to zero without
consuming random numbers, exactly reproducing N09's initial global network.
Train with original standardized-target MSE plus 0.1 times mean squared airport
correction. Unknown/missing airport code 0 gets a hard-zero correction.

This is one jointly trained conditional neural model, distinct from the already
tried per-airport LightGBMs or post-hoc per-airport ensemble weights. Fixed score
arms: new neural pool alone and half new/half v58's 15-network pool, total neural
share 0.71. Matched-size N09 is reported separately from full v58.

Run check; fit --groups 42 --nets 0; score --groups 42 --nets 0.
No fits unless the root allocates this machine.
No uploads or ledger mutations. All candidate seeds stay on one machine.
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
OUT=ts.RUNS/'v3_20261010_nn_airport_heads_jan_jul'
WITNESSES=[]
FEATURES=[]
HEAD_PENALTY=0.1
DESIGN=dict(parent='v58',control='HPO N09 matched network seeds',
    architecture='shared N09 trunk/global head plus zero-initialized linear airport residual heads',
    head_parameters_per_airport=257,head_penalty=HEAD_PENALTY,
    penalty_units='mean squared residual-head output in standardized-target units',
    unknown_airport='hard zero residual, global output only',target='direct min(taxi_sec,10800)',
    added_features=[],network_share=.71,neural_arms=['airport_heads','half_airport_half_v58'],
    heldout_months=ts.SPLITS,stop_choice='last training month only',
    train_input_cache=str(baseline.OUT/'inputs'),weather_inputs_used=False,
    groups=[42,142,242],networks_per_group=5,exploratory_reused_development_months=True)


def add_features(X):
    return X


class DirectData(sensor.DirectData):
    def preprocessing(self):
        result=super().preprocessing()
        result['design']=DESIGN
        result['airport_index']=self.encoder.cats.index('airport')
        return result


class AirportHeadNet(shared.nni.PleNet):
    def __init__(self,cards,bins,n_plain,airport_index):
        super().__init__(cards,bins,n_plain)
        self.airport_index=airport_index
        width=self.mlp[-1].in_features
        self.airport_weight=torch.nn.Parameter(torch.zeros(cards[airport_index],width+1))

    def forward_parts(self,cat,g,plain):
        z=self.inputs(cat,g,plain)
        for layer in list(self.mlp)[:-1]:
            z=layer(z)
        global_output=self.mlp[-1](z)
        airport=cat[:,self.airport_index]
        weights=torch.nn.functional.embedding(airport,self.airport_weight,padding_idx=0)
        delta=(weights[:,:-1]*z).sum(1,keepdim=True)+weights[:,-1:]
        delta=delta*(airport!=0)[:,None]
        return global_output+delta,delta

    def forward(self,cat,g,plain):
        return self.forward_parts(cat,g,plain)[0]


def train(data,rows,seed,stop,monitor):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=AirportHeadNet(data.cards,data.bins,data.train[2].shape[1],
                         data.encoder.cats.index('airport')).to(data.device)
    opt=torch.optim.AdamW(model.parameters(),lr=shared.nni.NEW['ple']['lr'],
                         weight_decay=shared.nni.NEW['ple']['weight_decay'])
    batch,count=shared.nni.NEW['batch'],len(rows)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=shared.nni.NEW['ple']['lr'],
                  total_steps=shared.nni.NEW['epochs']*math.ceil(count/batch))
    generator=torch.Generator(device=data.device).manual_seed(seed)
    indices=torch.from_numpy(rows).to(data.device)
    curve=[]
    for epoch in range(stop):
        started,total,total_penalty=time.monotonic(),0.,0.
        perm=indices[torch.randperm(count,device=data.device,generator=generator)]
        for i in range(0,count,batch):
            b=perm[i:i+batch]
            output,delta=model.forward_parts(*(a[b] for a in data.train))
            mse=torch.nn.functional.mse_loss(output,data.ty[b,None].expand_as(output))
            penalty=HEAD_PENALTY*delta.square().mean()
            loss=mse+penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
            total+=float(mse.detach())*len(b)
            total_penalty+=float(penalty.detach())*len(b)
        pred=shared.predict(model,monitor[0],data.sd,data.mu,monitor[2])
        curve.append(dict(epoch=epoch+1,train_mse_std=total/count,
            head_penalty=total_penalty/count,rmse=shared.nni.rmse(monitor[1],pred),
            seconds=time.monotonic()-started))
        shared.log(f'epoch {epoch+1}: train {total/count:.4f}; head penalty '
                   f'{total_penalty/count:.6f}; monitored {curve[-1]["rmse"]:.4f}')
    return model,pred,curve


def identity():
    parsed=ast.parse(Path(__file__).read_text())
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
              and n.name in {'add_features','DirectData','AirportHeadNet','train'}]
    digest=hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),
                                 include_attributes=False).encode()).hexdigest()
    return dict(feature_data_ast_sha256=digest,witnesses=WITNESSES,features=FEATURES,
        head_penalty=HEAD_PENALTY,helpers={str(Path(module.__file__).relative_to(ROOT)):sha256(Path(module.__file__))
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
        data['X']=add_features(data['X'].loc[:,names])
        data['V']=add_features(data['V'].loc[:,names])
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
                                preprocessing=str(pre.relative_to(ROOT)),
                                airport_index=d.encoder.cats.index('airport')),temp)
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
    errors={name:{} for name in ['v58','direct_matched','airport_heads','half_airport_half_v58']}
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
        arms=dict(v58=pool[month],direct_matched=dmean,airport_heads=cmean,
                  half_airport_half_v58=.5*pool[month]+.5*cmean)
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
    for name,control in [('airport_heads','direct_matched'),('airport_heads','v58'),
                         ('half_airport_half_v58','v58')]:
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
    candidate=AirportHeadNet(cards,bins,2,1).eval()
    torch.testing.assert_close(torch.get_rng_state(),original_rng,rtol=0,atol=0)
    for name,value in control.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name],value,rtol=0,atol=0)
    cat=torch.tensor([[0,0],[1,1],[2,5],[3,10]],dtype=torch.long)
    g=torch.tensor([[-1.],[0.],[.4],[1.5]])
    plain=torch.tensor([[0.,1.],[1.,0.],[0.,0.],[1.,1.]])
    with torch.no_grad():
        original=control(cat,g,plain)
        fitted=candidate(cat,g,plain)
        torch.testing.assert_close(original,fitted,rtol=0,atol=0)
        candidate.airport_weight.fill_(1.)
        after,delta=candidate.forward_parts(cat,g,plain)
        torch.testing.assert_close(after[0],original[0],rtol=0,atol=0)
        assert float(delta[0])==0.
        assert torch.count_nonzero(delta[1:])==3
    shared.log('initial N09 parameters/predictions/RNG identical; unknown-airport fallback passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','fit','score'])
    parser.add_argument('--groups',type=int,nargs='+',choices=hpo.SEEDS,default=[42])
    parser.add_argument('--nets',type=int,nargs='+',choices=list(range(5)),default=[0])
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
