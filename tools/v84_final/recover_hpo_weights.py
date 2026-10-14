#!/usr/bin/env python3
"""One fixed original Oct8 HPO execution, retaining weights; exact-output gates.

The original Data and train functions run unchanged. A factory observation saves
the exact encoder object returned to Data. Training returns the actual model;
its state is saved before comparison. No settings, outputs or epochs are tuned.
Any original float32 prediction mismatch stops the whole sequence permanently.
"""
from __future__ import annotations
import argparse,copy,gc,hashlib,inspect,json,os,pickle,sys,time
from pathlib import Path
from datetime import datetime,timezone

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
import numpy as np
import pandas as pd
import torch
import final_submit as original
import nn_improve_jan_jul as nni
import nn_jan_jul as nnj

OUT=ROOT/'VALIDATION/runs/v3_20261014_v84_final/hpo_weights'
EXPECTED_TRAIN_SHA='a6256c6c9d0ec44747a58d7fd0c6ea0fd66bc422cf1a2c7d332a025af74d30d9'
EXPECTED_NNI_SHA='d6c8b63f328bfa45ef0306778cdd3d79581ce16878ebe5c449a8d3001c4a4f20'
EXPECTED_NNJ_SHA='2f76f8c6d3d8cb218f5d2497bec1a9b8a3ad1210e666247baeb64d0662f7b766'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def art(p):
    p=Path(p);return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}


def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def read(p):return json.loads(Path(p).read_text())


def torch_save(p,d):
    p=Path(p);assert not p.exists();tmp=p.with_suffix(p.suffix+'.tmp');torch.save(d,tmp);tmp.replace(p)


def plan():
    if sha(Path(nni.__file__))!=EXPECTED_NNI_SHA or sha(Path(nnj.__file__))!=EXPECTED_NNJ_SHA:
        raise ValueError('Original neural source differs')
    rec=read(original.V54_PARTS/'nn.json')
    spec=original.hs.searched()['nn'];cfg=original.hs.hpo.nn_config(spec['overrides'])
    if cfg!=rec['config'] or cfg['batch']!=2048:
        raise ValueError('Original HPO configuration differs')
    members=[dict(group=g,k=k,seed=g+1000*k,stop=rec['groups'][str(g)]['stop_epoch'])
             for g in original.ids.GROUPS for k in range(original.hs.hpo.NN_FINAL_NETS)]
    assert len(members)==15 and [members[i]['stop'] for i in [0,5,10]]==[15,21,16]
    return dict(source=art(Path(__file__)),original_source=art(Path(original.__file__)),
        data_function_sha256=hashlib.sha256(inspect.getsource(nni.Data).encode()).hexdigest(),
        train_function_sha256=hashlib.sha256(inspect.getsource(nni.train).encode()).hexdigest(),
        neural_sources=[art(Path(nni.__file__)),art(Path(nnj.__file__))],config=cfg,members=members,
        original_target_predictions=[art(original.WORK/'nn'/f'n{s["group"]}_{s["k"]}.parquet') for s in members],
        expected_inputs=dict(training_sha256=EXPECTED_TRAIN_SHA,
                             final_sha256=read(original.OUT/'inputs_final.json')['sha256']),
        threads=original.THREADS,device='NVIDIA GeForce RTX3080',expected_train_rows=2051327,
        old_rows=344841,final_rows=670790,
        gate='Every original float32 old AND final prediction exactly equal, plus saved-state replay exact; ANY mismatch stops without remaining fits.',
        unchanged_training_function=True,no_hyperparameter_selection=True,no_competition_upload=True)


def build_data(identity):
    # This packet construction is the original final_submit.nn sequence.
    if sha(original.hs.INPUTS)!=EXPECTED_TRAIN_SHA:raise ValueError('Original full-year input changed')
    final_path=original.WORK/'inputs_final.pkl'
    if sha(final_path)!=identity['expected_inputs']['final_sha256']:
        raise ValueError('Original final input changed')
    data=pd.read_pickle(original.hs.INPUTS);final=pd.read_pickle(final_path)
    cols=data['names']+original.ids.COMBO;n_final=len(final['Vn'])
    assert len(cols)==156 and n_final==670790 and len(data['Xn'])==2051327
    Vn=pd.concat(original.widen([final['Vn'][cols].copy(),data['Vn'][cols].copy()]),ignore_index=True)
    original.ts.FOLDS.setdefault(original.ids.FULL,[f'2025-{m:02d}' for m in range(1,13)])
    item=dict(X=data['Xn'][cols],y=data['yn'],Vn=Vn,tmonth=data['tmonth'],meta=dict(month=original.ids.FULL),
              raw=pd.DataFrame({'taxi_sec':np.full(len(Vn),np.nan)}))
    ids=np.concatenate([final['rank_ids'],data['rank_ids']]);original_encoder=nnj.Encoder;captured=[]
    def observe_encoder(*args,**kwargs):
        enc=original_encoder(*args,**kwargs);captured.append(enc);return enc
    nnj.Encoder=observe_encoder
    try:d=nni.Data(item,'cuda')
    finally:nnj.Encoder=original_encoder
    assert len(captured)==1;encoder=captured[0]
    # No encoding recomputation: save the very object used by unchanged Data.
    with (OUT/'encoder.pkl').open('wb') as f:pickle.dump(encoder,f,protocol=pickle.HIGHEST_PROTOCOL)
    torch_save(OUT/'preprocessing.pt',dict(cards=d.cards,bins=d.bins,info=d.info,mu=d.mu,sd=d.sd,
        columns=cols,encoder=art(OUT/'encoder.pkl'),configuration=identity['config'],
        numeric_layout='Original nn_improve_jan_jul.Data; encoder nums/flags plus retained bins and info'))
    np.save(OUT/'ranking_ids.npy',ids)
    torch_save(OUT/'before_training_rng.pt',dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state()))
    save(OUT/'prepared.json',dict(identity=art(OUT/'identity.json'),inputs=[art(original.hs.INPUTS),art(final_path)],
        encoder=art(OUT/'encoder.pkl'),preprocessing=art(OUT/'preprocessing.pt'),ids=art(OUT/'ranking_ids.npy'),
        rows=len(ids),final_rows=n_final,training_rows=len(d.y),device=torch.cuda.get_device_name(),
        unchanged_original_data_constructor=True))
    del data,final,item,Vn,captured,encoder;gc.collect()
    return d,ids,n_final


def run(count):
    identity=plan()
    if OUT.exists():raise ValueError('Existing HPO attempt exists; preserve it; no automatic repeat or resume')
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 3080':
        raise ValueError('Original desktop RTX3080 required')
    OUT.mkdir(parents=True)
    identity['started_at']=datetime.now(timezone.utc).isoformat()
    identity['runtime']=dict(torch=str(torch.__version__),host=os.uname().nodename,
        device=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    save(OUT/'identity.json',identity)
    torch.set_num_threads(original.THREADS)
    original.hs.hpo.apply_nn(identity['config'],nnj,nni)
    d,ids,n_final=build_data(identity)
    allrows=np.arange(len(d.y));results=[]
    for spec in identity['members'][:count]:
        name=f'n{spec["group"]}_{spec["k"]}';folder=OUT/name;folder.mkdir()
        save(folder/'started.json',dict(spec=spec,at=datetime.now(timezone.utc).isoformat(),original_train_called=True))
        start=time.monotonic()
        model,pred,curve=nni.train('ple',d,allrows,spec['seed'],spec['stop'],(d.held,d.yv))
        state=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            cards=d.cards,bins=d.bins,n_plain=d.train[2].shape[1],mu=d.mu,sd=d.sd,
            spec=spec,configuration=identity['config'],preprocessing=art(OUT/'preprocessing.pt'),encoder=art(OUT/'encoder.pkl'),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            actual_returned_model_saved=True,original_train=identity['train_function_sha256'])
        torch_save(folder/'model.pt',state)
        pred=np.asarray(pred,np.float32)
        pd.DataFrame(dict(mvt_id=ids,pred=pred,set=np.r_[np.full(n_final,'final'),np.full(len(ids)-n_final,'old')])).to_parquet(folder/'predictions.parquet',index=False)
        clean_curve=[{k:v for k,v in c.items() if k!='rmse'} for c in curve]
        save(folder/'trained.json',dict(spec=spec,model=art(folder/'model.pt'),prediction=art(folder/'predictions.parquet'),curve=clean_curve,seconds=time.monotonic()-start))
        del model,state;gc.collect();torch.cuda.empty_cache()
        saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        reloaded=nni.PleNet(saved['cards'],saved['bins'],saved['n_plain']).cuda();reloaded.load_state_dict(saved['state_dict'])
        replay=nni.predict(reloaded,d.held,d.sd,d.mu,65536).astype(np.float32)
        reload_exact=bool(np.array_equal(replay,pred))
        teacher_path=original.WORK/'nn'/f'{name}.parquet'
        expected_teacher=next(r for r in identity['original_target_predictions'] if r['path']==str(teacher_path.relative_to(ROOT)))
        if art(teacher_path)!=expected_teacher:raise ValueError('Frozen original member target changed')
        teacher=pd.read_parquet(teacher_path)
        np.testing.assert_array_equal(teacher.mvt_id,ids)
        original_values=teacher.pred.to_numpy(np.float32);delta=pred.astype(float)-original_values.astype(float)
        checks={label:dict(rows=int(mask.sum()),exact=bool(np.array_equal(pred[mask],original_values[mask])),
            changed_rows=int(np.count_nonzero(delta[mask])),rms=float(np.sqrt(np.mean(delta[mask]**2))),max_abs=float(np.max(abs(delta[mask]))))
            for label,mask in [('final',np.arange(len(ids))<n_final),('old',np.arange(len(ids))>=n_final)]}
        exact=reload_exact and all(c['exact'] for c in checks.values())
        rec=dict(status='exact' if exact else 'STOPPED_NONEXACT',spec=spec,checks=checks,reload_exact=reload_exact,
            model=art(folder/'model.pt'),prediction=art(folder/'predictions.parquet'),original_prediction=art(teacher_path))
        save(folder/'comparison.json',rec);results.append(rec)
        save(OUT/'result.json',dict(status='in_progress_exact' if exact else 'STOPPED_NONEXACT',members=results,
            completed_members=len(results),all_selected_fifteen_states_recovered=len(results)==15 and exact,
            no_prediction_substitution=True,no_upload=True))
        print(json.dumps({'member':name,'status':rec['status'],'checks':checks,'reload_exact':reload_exact}),flush=True)
        del reloaded,saved,teacher,replay,pred,delta;gc.collect();torch.cuda.empty_cache()
        if not exact:raise SystemExit('Exact old/final member gate failed; no remaining fits permitted')
    result=read(OUT/'result.json');result['status']='complete_exact15' if count==15 else 'exact_first_member_no_expansion';save(OUT/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['plan','run']);p.add_argument('--members',type=int,choices=[1,15],default=1);a=p.parse_args()
    if a.stage=='plan':print(json.dumps(plan()))
    else:run(a.members)
