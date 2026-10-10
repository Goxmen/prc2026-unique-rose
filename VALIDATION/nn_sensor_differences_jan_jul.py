"""v58 PLE plus ten explicit actual-AOBT3 versus sensor clock disagreements.

Each new input is (takeoff - original AOBT3) - (takeoff - sensor witness),
equivalently sensor-witness time minus original AOBT3. This affine cancellation
is supplied before independent rank/PLE encodings can obscure it. Sensor clocks
come from already cached ADS-B and OPDI inputs, with no new downloads or targets.
Original AOBT3 absent, nonfinite gap, or absent sensor witness means NaN; there is
no schedule fallback. Inputs are not clipped before the existing v58 encoder.

Everything else matches direct-target HPO N09: clipped labels, ordinary training
rows, PLE/backbone, optimizer, last-training-month epoch choice, and final stack.
One fixed ten-feature candidate; fixed neural arms are the new pool alone and a
50/50 mix with v58's 15-network pool. Network share stays .71. Small-pool screens
are compared separately to matched N09 seeds and to the full v58 parent.

Run fit --groups 42 --nets 0 then score --groups 42 --nets 0. Expand only after
review. All requested networks for a candidate should be fitted on one machine.
No upload or submission-ledger mutations are provided.
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
from records import sha256,write_json

hpo,ts,baseline=shared.hpo,shared.ts,shared.baseline
OUT=ts.RUNS/'v3_20261010_nn_sensor_differences_jan_jul'
WITNESSES=['ab_seg_sec','ab_move_sec','ab_still_end_sec','ab_stand_exit_sec',
           'op_fs_sec','op_own_stand_sec','op_first_stand_sec','op_first_apron_sec',
           'op_first_twy_sec','op_ground_sec']
FEATURES=['xclock_'+name for name in WITNESSES]
DESIGN=dict(parent='v58',control='HPO N09 matched network seeds',
    added_features=FEATURES,witnesses=WITNESSES,
    formula='mvt_minus_aobt3_sec - witness (equivalently sensor time - original AOBT3)',
    original_aobt_gate='clock_aobt3_available == 1 and clock_raw_aobt3_missing == 0 and finite gap',
    missing_policy='NaN whenever original AOBT3 or sensor witness unavailable; no fallback',
    target='direct min(taxi_sec,10800)',network_share=.71,
    neural_arms=['sensor_differences','half_sensor_half_v58'],
    heldout_months=ts.SPLITS,stop_choice='last training month only',
    train_input_cache=str(baseline.OUT/'inputs'),weather_inputs_used=False,
    groups=[42,142,242],networks_per_group=5,exploratory_reused_development_months=True)


def add_features(X):
    """Fixed input-only arithmetic; the original v58 columns remain unchanged."""
    result=X.copy()
    gap=X['mvt_minus_aobt3_sec'].to_numpy(float)
    present=((X['clock_aobt3_available'].to_numpy()==1)
             &(X['clock_raw_aobt3_missing'].to_numpy()==0)&np.isfinite(gap))
    for witness,name in zip(WITNESSES,FEATURES):
        value=X[witness].to_numpy(float)
        result[name]=np.where(present&np.isfinite(value),gap-value,np.nan)
    return result


class DirectData(shared.ResidualData):
    """Reuse identical PLE encoding and restore v58's direct response exactly."""

    def __init__(self,item,device):
        self.encoder=shared.nnj.Encoder(item['X'])
        enc=self.encoder
        cat,num=enc.transform(item['X'])
        vcat,vnum=enc.transform(item['Vn'])
        g,plain,j=[],[],0
        for name in enc.nums:
            g.append(j)
            plain.append(j+1)
            j+=2
            if name in enc.flags:
                plain.append(j)
                j+=1
        if j!=num.shape[1]:
            raise ValueError('Encoder column layout changed')
        edges=np.quantile(num[:,g],np.linspace(0,1,shared.nni.NEW['bins']+1),axis=0)
        bins,ple=[],[]
        for i,col in enumerate(g):
            values=np.unique(edges[:,i].astype(np.float32))
            if len(values)>=3:
                bins.append(torch.from_numpy(values))
                ple.append(col)
            else:
                plain.append(col)
        self.bins,self.cards,self.ple,self.plain=bins,enc.cards,ple,plain
        self.columns=list(item['X'].columns)
        self.y=np.asarray(item['y'],float)
        self.yv=item['raw'].taxi_sec.to_numpy(float)
        target=np.minimum(self.y,shared.nnj.RECIPE['label_clip'])
        self.mu,self.sd=float(target.mean()),float(target.std())
        tensor=lambda a:torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.train=(tensor(cat),tensor(num[:,ple]),tensor(num[:,plain]))
        self.held=(tensor(vcat),tensor(vnum[:,ple]),tensor(vnum[:,plain]))
        self.ty=torch.from_numpy(((target-self.mu)/self.sd).astype(np.float32)).to(device)
        self.offset=np.zeros(len(self.y),float)
        self.val_offset=np.zeros(len(self.yv),float)
        last=max(ts.FOLDS[item['meta']['month']])
        self.inner_val=np.flatnonzero(item['tmonth']==last)
        self.inner_train=np.flatnonzero(item['tmonth']!=last)
        self.device=device
        self.info=dict(numeric=len(enc.nums),ple_inputs=len(ple),plain_inputs=len(plain),
            categoricals=enc.cats,cards=enc.cards,bins_per_input=[len(b)-1 for b in bins],
            inner_month=last,inner_val_rows=len(self.inner_val),inner_train_rows=len(self.inner_train),
            target_mode='direct',offset_nonzero=0,offset_at_cap=0,target_mu=self.mu,target_sd=self.sd)

    def preprocessing(self):
        output=super().preprocessing()
        output['design']=DESIGN
        return output


def identity():
    parsed=ast.parse(Path(__file__).read_text())
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
              and n.name in {'add_features','DirectData'}]
    digest=hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),
                                 include_attributes=False).encode()).hexdigest()
    return dict(feature_data_ast_sha256=digest,witnesses=WITNESSES,features=FEATURES,
        helpers={str(Path(module.__file__).relative_to(ROOT)):sha256(Path(module.__file__))
                 for module in [shared,shared.nni,shared.nnj,hpo]})


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
                model,_,curve=shared.train(d,d.inner_train,group,cfg['epochs'],
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
    errors={name:{} for name in ['v58','direct_matched','sensor_differences','half_sensor_half_v58']}
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
        arms=dict(v58=pool[month],direct_matched=dmean,sensor_differences=cmean,
                  half_sensor_half_v58=.5*pool[month]+.5*cmean)
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
    for name,control in [('sensor_differences','direct_matched'),('sensor_differences','v58'),
                         ('half_sensor_half_v58','v58')]:
        report['comparisons'][f'{name}_vs_{control}']=ts.paired_bootstrap(scored.day.to_numpy(),
            scored.month.to_numpy(),scored[f'se_{control}'].to_numpy(),scored[f'se_{name}'].to_numpy())
    tag='g'+'_'.join(map(str,groups))+'_k'+'_'.join(map(str,nets))
    scored.to_parquet(OUT/f'scored_{tag}.parquet',index=False)
    write_json(OUT/f'score_{tag}.json',report)
    shared.log(json.dumps(report,indent=2))


def self_check():
    frame=pd.DataFrame(dict(mvt_minus_aobt3_sec=[1200.,-300.,900.,np.nan,1800.],
        clock_aobt3_available=[1,1,1,1,0],clock_raw_aobt3_missing=[0,0,1,0,0],
        **{name:[900.,100.,200.,300.,400.] for name in WITNESSES}))
    frame.loc[0,'ab_move_sec']=np.nan
    result=add_features(frame)
    pd.testing.assert_frame_equal(result[frame.columns],frame)
    for name in FEATURES:
        expected=np.array([300.,-400.,np.nan,np.nan,np.nan])
        if name=='xclock_ab_move_sec':
            expected[0]=np.nan
        np.testing.assert_equal(result[name].to_numpy(),expected)
    assert len(FEATURES)==len(set(FEATURES))==10
    shared.log('ten feature arithmetic, missing provenance and unchanged input checks passed')


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
