"""Expand the successful physical-scale screen to three fixed sparse members.

Reuse completed seed42 unchanged. Add 1042/2042 on the same desktop, with
original21/17 epochs and unchanged optimizer/update order. Compare directly
with v64's same-size sparse3 pool in the same callsign/tree/Rome combination.
"""
import argparse,gc,os,pickle,tempfile,time,types
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import nn_track_physical_scales_jan_jul as single
import nn_sparse_track_pool3 as original
from records import sha256,write_json
ROOT=single.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_nn_track_physical_pool3'
MANIFEST=OUT/'identity.json'
DESIGN=dict(parent='v64',change='replace original sparse3 with physical-scale sparse3',
    seeds=[42,1042,2042],reused_seed=42,new_seeds=[1042,2042],epochs=single.STOP,
    model=single.DESIGN['model'],training='original sparse train AST with seed argument; constructor only replaced',
    neural='equal calls15/physical3',network_share=.71,other_parts='v64 unchanged',
    machine='all physical seeds on desktop RTX3080',no_new_gate_or_weight_search=True,
    evaluation='reused Jan/Jul development months; pool declared after k0 outcome')

def training(runtime):
    parent=original.compile_training(runtime)
    scope=dict(parent.__globals__,prototype=single.physical_scales)
    return types.FunctionType(parent.__code__,scope,parent.__name__,parent.__defaults__,parent.__closure__)

def guard():
    audit=single.shared.ts.read(single.OUT/'audit.json');assert audit['status']=='pass'
    identity=single.shared.ts.read(single.MANIFEST)
    for m in single.STOP:assert single.completed(m,identity) is not None
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080'
    value=dict(design=DESIGN,source=single.artifact(__file__),parent_identity=single.artifact(single.MANIFEST),
        parent_score=single.artifact(single.OUT/'score.json'),parent_audit=single.artifact(single.OUT/'audit.json'),
        helpers=[single.artifact(m.__file__) for m in [single,original,single.physical_scales,single.adapter]],
        seed_specialization=original.check_specialization())
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert single.shared.ts.read(MANIFEST)==value
    else:write_json(MANIFEST,value)
    return identity

def paths(m,k):
    if k==0:return single.paths(m)
    return OUT/'networks'/f'{m}_k{k}.parquet',OUT/'models'/f'{m}_k{k}.pt',OUT/'networks'/f'{m}_k{k}.json'

def completed(m,k,identity):
    if k==0:return single.completed(m,identity)
    p,model,rp=paths(m,k)
    if not rp.exists():
        assert not p.exists() and not model.exists(),'Unrecorded outputs; inspect before retrying'
        return None
    r=single.shared.ts.read(rp)
    assert r['manifest']==single.artifact(MANIFEST) and r['seed']==42+1000*k and r['epochs']==single.STOP[m]
    for key in ['model','prediction','preprocessing']:single.verify(r[key])
    return r

def initial_check(seed,device):
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    torch.manual_seed(seed);old=single.base.prototype.SparseFusion(cards,bins,plain).to(device)
    expected=torch.get_rng_state().clone()
    torch.manual_seed(seed);new=single.physical_scales.SparseFusion(cards,bins,plain).to(device)
    assert torch.equal(expected,torch.get_rng_state())
    for name,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[name]),name
    with torch.no_grad():
        old.fusion[-1].weight.fill_(.01);old.fusion[-1].bias.fill_(.03)
        new.load_state_dict(old.state_dict(),strict=False)
    tab,track=single.base.prototype.synthetic(cards,bins,plain,24,device)
    for mode in [True,False]:
        old.train(mode);new.train(mode)
        torch.manual_seed(seed+17);a=old(*tab,*track);rng=torch.cuda.get_rng_state().clone()
        torch.manual_seed(seed+17);b=new(*tab,*track)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert torch.equal(rng,torch.cuda.get_rng_state())
    return dict(seed=seed,original_weights_rng_and_nonzero_path_exact=True)

def fit(identity):
    single.shared.recipe()
    meta,_=single.metadata.load()
    index=pd.read_parquet(single.base.seq.OUT/'full_index.parquet')
    points=np.memmap(single.base.seq.OUT/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    tracks=single.base.tensor_seq.RaggedTracks(index,points,'cuda')
    for name in ['models','networks','runtime']:(OUT/name).mkdir(exist_ok=True)
    for m in single.STOP:
        pending=[k for k in [1,2] if completed(m,k,identity) is None]
        if not pending:continue
        data=pd.read_pickle(single.verify(identity['folds'][m]['matrix']))
        names=data['baseline'];assert len(names)==156
        tm,vm=meta.loc[data['train_ids']],meta.loc[data['ids']]
        assert set(tm.month.astype(str))==set(single.shared.ts.FOLDS[m])
        assert set(vm.month.astype(str))=={m} and not np.intersect1d(data['train_ids'],data['ids']).size
        np.testing.assert_array_equal(data['y'],tm.taxi_sec)
        tids,ids=np.asarray(data['train_ids']).copy(),np.asarray(data['ids']).copy()
        item=dict(X=data['X'][names],Vn=data['V'][names],y=data['y'],raw=vm.reset_index(),
            tmonth=tm.month.astype(str).to_numpy(),meta=dict(month=m))
        del data,tm,vm;gc.collect()
        with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT) as scratch:
            d=single.adapter.ChunkedDirectData(item,'cuda',scratch)
        state=d.preprocessing();state['design']=single.base.DESIGN
        pre=single.verify(identity['folds'][m]['preprocessing'])
        with pre.open('rb') as h:single.base.preprocess_guard.equal_preprocessing(pickle.load(h),state)
        del item;gc.collect()
        tr,va=tracks.align(tids),tracks.align(ids)
        for k in pending:
            seed=42+1000*k;check=initial_check(seed,'cuda')
            runtime=OUT/'runtime'/f'{m}_k{k}';runtime.mkdir(exist_ok=True)
            started=time.monotonic();model,pred,curve=training(runtime)(d,tracks,tr,va,single.STOP[m],seed)
            path,modelpath,rp=paths(m,k)
            temp=modelpath.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(),design=DESIGN,config=single.shared.recipe(),
                preprocessing=str(pre.relative_to(ROOT))),temp);temp.replace(modelpath)
            single.shared.nni.save_network(path,ids,pred)
            write_json(rp,dict(manifest=single.artifact(MANIFEST),month=m,member=k,seed=seed,
                epochs=single.STOP[m],minutes=(time.monotonic()-started)/60,curve=curve,initial_check=check,
                model=single.artifact(modelpath),prediction=single.artifact(path),preprocessing=single.artifact(pre),
                train_rows=len(tids),held_rows=len(ids),gpu=torch.cuda.get_device_name(),host=os.uname().nodename))
            del model,pred;gc.collect();torch.cuda.empty_cache()
        del d,tr,va;gc.collect();torch.cuda.empty_cache()

def score(identity):
    sc,rome,_,trees=single.shared.baseline.baseline()
    combo=single.combo
    calls=pd.read_parquet(combo.sequence.CALLS_PRED).set_index('mvt_id')
    old=pd.read_parquet(combo.sequence.OUT/'scored_pool3.parquet').set_index('mvt_id')
    reference=pd.read_parquet(combo.OUT/'scored_errors.parquet').set_index('mvt_id')
    rows=[];errors={name:{} for name in ['v64','physical3']}
    for m in single.STOP:
        ids=sc.d[m]['ids'];new=[]
        for k in [0,1,2]:
            assert completed(m,k,identity) is not None
            f=pd.read_parquet(paths(m,k)[0]);np.testing.assert_array_equal(f.mvt_id,ids)
            new.append(f.base.to_numpy(float))
        mean=np.mean(new,axis=0);cn=calls.loc[ids,'candidate_nn'].to_numpy(float)
        p=.5*combo.parent.cat.probabilities(m,ids)+.5*rome[m]
        members={single.shared.hpo.MEMBER_OF[part]:trees[part][42][m] for part in trees}
        main=pd.read_parquet(combo.parent.control.candidate.OUT/'predictions'/f'{m}_s42.parquet')
        np.testing.assert_array_equal(main.mvt_id,ids);members['combo']=main.pred.to_numpy(float)
        for name,sparse in [('v64',old.loc[ids,'candidate_nn'].to_numpy(float)),('physical3',mean)]:
            errors[name][m]=sc.se(m,42,nn=.5*(cn+sparse),p=p,**members)
        np.testing.assert_array_equal(errors['v64'][m],reference.loc[ids,'half_sparse3_combination'])
        rows.append(pd.DataFrame(dict(mvt_id=ids,month=m,day=sc.d[m]['day'],candidate_nn=mean,
            **{name:errors[name][m] for name in errors})))
    f=pd.concat(rows,ignore_index=True);f.to_parquet(OUT/'scored.parquet',index=False)
    scores={name:sc.cv(values) for name,values in errors.items()}
    result=dict(design=DESIGN,manifest=single.artifact(MANIFEST),scores=scores,
        paired=single.shared.ts.paired_bootstrap(f.day,f.month,f.v64,f.physical3),
        scored=single.artifact(OUT/'scored.parquet'),note='Same-size pools, same desktop; reused development months.')
    write_json(OUT/'score.json',result);print(result,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','score']);args=ap.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        identity=guard()
        if args.stage=='fit':fit(identity)
        else:score(identity)
