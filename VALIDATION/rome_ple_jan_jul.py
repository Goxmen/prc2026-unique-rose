"""Fixed PLE binary Rome classifier on the original R20 population and68 inputs.

Unlike the ordinary network's auxiliary head, this includes Rome schedule-copy
positives. Encoder/maps/bins fit only inner-training rows during epoch selection,
then are refitted on all allowed outer-training rows. Last training month chooses
the minimum logloss epoch among60; held Jan/Jul labels never choose epochs.
Only probabilities change in the unchanged v58 stack: full or fixed50/50 withR20.
No class weighting, new ADS-B inputs, threshold tuning, calibration or uploads.
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import duckdb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
import nn_jan_jul as nnj
import nn_improve_jan_jul as nni
import rome_adsblol_jan_jul as shared
from records import sha256,write_json

OUT=shared.ts.RUNS/'v3_20261010_rome_ple_jan_jul'
CONFIG=dict(seed=42,hidden=[256,128],dropout=.1,bins=48,d_ple=16,
    lr=.001,weight_decay=1e-4,batch=1024,max_epochs=60,
    optimizer='AdamW',scheduler=None,loss='unweighted BCEWithLogits',
    min_cat_rows=20,quantiles=1024,numeric_clip=5.)
DESIGN=dict(parent='v58',part='Rome recording classifier',family='PLE neural binary classifier',
    config=CONFIG,train_population='original R20 after_drop Rome rows including copy positives',
    features='original R20 68 inputs and11 categorical columns; no ADS-B additions',
    source_cache=str(shared.OUT/'inputs'),target='abs(takeoff-original schedule-raw taxi)<=60',
    inner_stop='minimum logloss among60epochs on last allowed training month; earliest tie',
    preprocessing='inner fitted on inner-train only; full fitted on all allowed training rows',
    arms=['ple','half_ple_half_R20'],network_share=.71,ordinary_networks_unchanged=True,
    thresholds_unchanged=True,probability_calibration=False,heldout_months=shared.ts.SPLITS,
    exploratory_reused_development_months=True)

def log(s):print(f'[{time.strftime("%H:%M:%S")}] {s}',flush=True)

def ensure_helpers():
    assert nni.NEW['d_ple']==CONFIG['d_ple']
    assert nnj.RECIPE['min_cat_rows']==CONFIG['min_cat_rows']
    assert nnj.RECIPE['quantiles']==CONFIG['quantiles']
    assert nnj.RECIPE['clip']==CONFIG['numeric_clip']

def provenance():
    return dict(design=DESIGN,source_sha256=sha256(Path(__file__)),
        helper_sha256={str(Path(mod.__file__).relative_to(ROOT)):sha256(Path(mod.__file__))
            for mod in [shared,shared.baseline,shared.hpo,shared.signals,nnj,nni]},
        raw_date_sources={str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.glob('data/training_2025-*.parquet'))})

def guard_run():
    identity=provenance();path=OUT/'provenance.json'
    if path.exists() and shared.ts.read(path)!=identity:
        raise ValueError('Rome PLE source/design/input provenance changed')
    OUT.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        write_json(path,identity)
        (OUT/'runner_initial.py').write_text(Path(__file__).read_text())
    return identity

class Codec:
    """Label-free reused NNEncoder; PLE edges fitted on this exact training frame."""
    def __init__(self,X):
        self.columns=list(X)
        self.encoder=nnj.Encoder(X)
        cat,num=self.encoder.transform(X)
        gaussian,plain,j=[],[],0
        for name in self.encoder.nums:
            gaussian.append(j);plain.append(j+1);j+=2
            if name in self.encoder.flags:plain.append(j);j+=1
        assert j==num.shape[1]
        edges=np.quantile(num[:,gaussian],np.linspace(0,1,CONFIG['bins']+1),axis=0)
        self.ple,self.bins=[],[]
        for i,col in enumerate(gaussian):
            values=np.unique(edges[:,i].astype(np.float32))
            if len(values)>=3:
                self.ple.append(col);self.bins.append(torch.from_numpy(values))
            else:plain.append(col)
        self.plain=plain;self.cards=self.encoder.cards
        self.info=dict(columns=self.columns,categoricals=self.encoder.cats,cards=self.cards,
            ple_columns=[self.encoder.nums[gaussian.index(c)] for c in self.ple],
            bins_per_column=[len(b)-1 for b in self.bins],plain_count=len(plain),fit_rows=len(X))

    def transform(self,X,device):
        if list(X)!=self.columns:raise ValueError('Encoder feature order changed')
        cat,num=self.encoder.transform(X)
        return tuple(torch.from_numpy(np.ascontiguousarray(v)).to(device)
                     for v in [cat,num[:,self.ple],num[:,self.plain]])

    @classmethod
    def from_state(cls,state):
        """Restore a plain saved dictionary without fitting or __main__ class pickles."""
        result=cls.__new__(cls)
        result.__dict__.update(state)
        return result

class RomePLE(torch.nn.Module):
    def __init__(self,codec):
        super().__init__()
        self.inputs=nni.Inputs(codec.cards,codec.bins,len(codec.plain))
        layers=[];width=sum(self.inputs.chunks)
        for hidden in CONFIG['hidden']:
            layers.extend([torch.nn.Linear(width,hidden),torch.nn.SiLU(),torch.nn.Dropout(CONFIG['dropout'])]);width=hidden
        layers.append(torch.nn.Linear(width,1));self.mlp=torch.nn.Sequential(*layers)

    def forward(self,*values):return self.mlp(self.inputs(*values)).squeeze(1)

def logits(model,values):
    previous=model.training;model.eval()
    with torch.no_grad():
        result=torch.cat([model(*(v[i:i+32768] for v in values)).cpu()
                          for i in range(0,len(values[0]),32768)]).numpy().astype(float)
    model.train(previous)
    return result

def binary_loss(y,z):return float(np.mean(np.logaddexp(0,z)-np.asarray(y,float)*z))

def fit_model(X,y,epochs,monitor=None):
    codec=Codec(X);values=codec.transform(X,'cuda')
    labels=torch.from_numpy(np.asarray(y,np.float32)).to('cuda')
    monitored=codec.transform(monitor[0],'cuda') if monitor is not None else None
    torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed'])
    model=RomePLE(codec).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    generator=torch.Generator(device='cuda').manual_seed(CONFIG['seed'])
    count=len(y);batch=CONFIG['batch'];curve=[];best=None;best_loss=np.inf
    for epoch in range(1,epochs+1):
        started=time.monotonic();total=0.
        perm=torch.randperm(count,device='cuda',generator=generator)
        model.train()
        for start in range(0,count,batch):
            take=perm[start:start+batch]
            pred=model(*(v[take] for v in values))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,labels[take])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(take)
        row=dict(epoch=epoch,training_logloss=total/count,seconds=time.monotonic()-started)
        if monitor is not None:
            z=logits(model,monitored);row['validation_logloss']=binary_loss(monitor[1],z)
            if row['validation_logloss']<best_loss:
                best_loss=row['validation_logloss'];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        curve.append(row)
        if epoch==1 or epoch%5==0 or epoch==epochs:log(json.dumps(row))
    if monitor is not None:model.load_state_dict(best)
    del values,labels,monitored
    return model,codec,curve

def atomic_pickle(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_bytes(pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL));temporary.replace(path)

def save_model(path,model,codec):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    torch.save(dict(state_dict=model.state_dict(),config=CONFIG,codec_info=codec.info),temporary);temporary.replace(path)

def validated_record(month):
    path=OUT/'predictions'/f'{month}_s42.json'
    if not path.exists():return None
    record=shared.ts.read(path)
    if record['provenance']!=provenance():raise ValueError('Saved classifier provenance changed')
    source=shared.OUT/'inputs'/f'{month}.pkl'
    if sha256(source)!=record['input_sha256']:raise ValueError('Original classifier input cache changed')
    for p,digest in record['outputs'].items():
        if sha256(ROOT/p)!=digest:raise ValueError(f'Saved classifier artifact changed: {p}')
    return record

def fit(months):
    con=duckdb.connect(config={'threads':2});con.execute("SET TimeZone='UTC'")
    dates=con.execute("SELECT MVT_ID_mvt mvt_id,strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month FROM read_parquet(?) WHERE PHASE_mvt='DEP'",
        [str(ROOT/'data/training_2025-*.parquet')]).df().set_index('mvt_id').month
    con.close();assert dates.index.is_unique
    for month in months:
        if validated_record(month) is not None:log(f'{month} already complete; skip');continue
        source=shared.OUT/'inputs'/f'{month}.pkl';d=pd.read_pickle(source)
        original=shared.ts.read(shared.ts.RUNS/'v3_20261010_rome_catboost_jan_jul/predictions'/f'{month}_s42.json')
        assert sha256(source)==original['input_sha256']
        names=[c for c in d['A'] if c not in shared.NEW]
        assert names==original['columns'] and len(names)==68
        X,V=d['A'].loc[:,names],d['B'].loc[:,names]
        cats=[c for c in X if isinstance(X[c].dtype,pd.CategoricalDtype)]
        assert cats==original['categoricals'] and len(cats)==11
        y=np.asarray(d['target'],np.float32);assert set(np.unique(y))=={0.,1.}
        assert len(X)==original['training_rows'] and int(y.sum())==original['positives']
        tmonth=dates.loc[d['train_ids']].to_numpy();assert set(tmonth)==set(shared.ts.FOLDS[month])
        assert not np.intersect1d(d['train_ids'],d['ids']).size
        stop_month=max(shared.ts.FOLDS[month]);hold=tmonth==stop_month
        assert hold.any() and (~hold).any() and len(V)==int(np.sum(d['use']))
        log(f'{month}: original R20 population {len(X):,}, positives {int(y.sum()):,}; stop month {stop_month}')
        started=time.monotonic()
        inner,inner_codec,curve=fit_model(X.loc[~hold],y[~hold],CONFIG['max_epochs'],(X.loc[hold],y[hold]))
        stop=min(curve,key=lambda r:r['validation_logloss'])['epoch']
        inner_pre=OUT/'preprocessing'/f'{month}_inner.pkl';inner_model=OUT/'models'/f'{month}_inner.pt'
        atomic_pickle(inner_pre,inner_codec.__dict__);save_model(inner_model,inner,inner_codec)
        del inner,inner_codec;torch.cuda.empty_cache();gc.collect()
        log(f'{month}: full refit for training-selected {stop} epochs')
        model,codec,full_curve=fit_model(X,y,stop)
        pre=OUT/'preprocessing'/f'{month}_full.pkl';model_path=OUT/'models'/f'{month}_full.pt'
        atomic_pickle(pre,codec.__dict__);save_model(model_path,model,codec)
        z=logits(model,codec.transform(V,'cuda'))
        prob=torch.sigmoid(torch.from_numpy(z)).numpy()
        p=np.zeros(len(d['ids']),float);p[d['use']]=prob
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        destination=OUT/'predictions'/f'{month}_s42.parquet';destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.tmp');pd.DataFrame(dict(mvt_id=d['ids'],p=p)).to_parquet(temporary,index=False);temporary.replace(destination)
        outputs={str(p.relative_to(ROOT)):sha256(p) for p in [inner_pre,inner_model,pre,model_path,destination]}
        write_json(destination.with_suffix('.json'),dict(provenance=provenance(),input_sha256=sha256(source),
            original_R20_columns=names,categoricals=cats,training_rows=len(X),positives=int(y.sum()),
            train_ids_sha256=hashlib.sha256(np.asarray(d['train_ids'],np.int64).tobytes()).hexdigest(),
            stop_month=stop_month,stop_epoch=stop,inner_rows=int((~hold).sum()),stop_rows=int(hold.sum()),
            inner_curve=curve,full_curve=full_curve,codec_info=codec.info,outputs=outputs,
            minutes=(time.monotonic()-started)/60))
        del X,V,d,model,codec;gc.collect();torch.cuda.empty_cache()

def score():
    sc,rome,pool,trees=shared.baseline.baseline()
    errors={name:{} for name in ['control',*DESIGN['arms']]};rows=[]
    for month in shared.ts.SPLITS:
        if validated_record(month) is None:raise ValueError(f'{month} classifier missing')
        f=pd.read_parquet(OUT/'predictions'/f'{month}_s42.parquet');np.testing.assert_array_equal(f.mvt_id,sc.d[month]['ids'])
        probabilities=dict(control=rome[month],ple=f.p.to_numpy(float),half_ple_half_R20=.5*f.p.to_numpy(float)+.5*rome[month])
        for arm,p in probabilities.items():
            errors[arm][month]=np.mean([sc.se(month,seed,nn=pool[month],p=p,
                **{shared.hpo.MEMBER_OF[part]:trees[part][seed][month] for part in trees}) for seed in shared.hpo.SEEDS],axis=0)
        rows.append(pd.DataFrame(dict(mvt_id=f.mvt_id,month=month,day=sc.d[month]['day'],**{arm:errors[arm][month] for arm in errors})))
    scored=pd.concat(rows,ignore_index=True);scores={name:sc.cv(v) for name,v in errors.items()}
    assert abs(scores['control']['cv']-295.7414415741871)<1e-7
    comparisons={arm:shared.ts.paired_bootstrap(scored.day,scored.month,scored.control,scored[arm]) for arm in DESIGN['arms']}
    scored.to_parquet(OUT/'scored_errors.parquet',index=False)
    report=dict(design=DESIGN,scores=scores,comparisons=comparisons)
    write_json(OUT/'result.json',report);log(json.dumps(report,indent=2))

def self_check():
    ensure_helpers();torch.manual_seed(42)
    X=pd.DataFrame(dict(cat=pd.Categorical(['A']*32+['B']*32),number=np.linspace(-2,2,64),constant=1.))
    codec=Codec(X);other=X.iloc[:3].copy();other['cat']=pd.Categorical(['NEVER','B',None]);other.loc[other.index[0],'number']=np.nan
    values=codec.transform(other,'cpu');assert values[0][:,0].tolist()==[0,2,0]
    assert all(torch.isfinite(v).all() for v in values)
    restored=Codec.from_state(pickle.loads(pickle.dumps(codec.__dict__)))
    for a,b in zip(values,restored.transform(other,'cpu')):torch.testing.assert_close(a,b,rtol=0,atol=0)
    model=RomePLE(codec).eval();z=model(*values);assert z.shape==(3,)
    y=torch.tensor([0.,1.,0.]);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
    np.testing.assert_allclose(binary_loss(y.numpy(),z.detach().numpy()),float(loss.detach()),rtol=1e-6)
    loss.backward();assert model.mlp[-1].weight.grad is not None
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    clone=RomePLE(codec).eval();clone.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(model(*values),clone(*values),rtol=0,atol=0)
    torch.testing.assert_close(model(*values),clone(*restored.transform(other,'cpu')),rtol=0,atol=0)
    log('codec unseen/missing categories, numeric missingness, serialization, binary logits/loss and model reload pass')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['check','fit','score'])
    parser.add_argument('--months',nargs='+',choices=shared.ts.SPLITS,default=shared.ts.SPLITS);args=parser.parse_args()
    torch.set_num_threads(2);ensure_helpers()
    with threadpool_limits(limits=2):
        if args.stage=='check':self_check()
        else:
            guard_run()
            if args.stage=='fit':fit(args.months)
            else:score()

if __name__=='__main__':main()
