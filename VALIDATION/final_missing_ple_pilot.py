"""Fixed missing legacy134 PLE seed42 trajectory, checkpoints14/15/17 only.

Stages: freeze, prepare, pilot. No inner stopping fit or seed expansion. A
nonexact original n42_0 old-vector replay closes this pilot for parent review.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'VALIDATION'),str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
import numpy as np
import pandas as pd
import torch
import duckdb
from threadpoolctl import threadpool_limits
import callsign_submit as parent
import chunked_direct_data as adapter
import nn_improve_jan_jul as nni
import nn_jan_jul as nnj

OUT = ROOT/'VALIDATION/runs/v3_20261012_final_missing_ple_pilot'
PLAN = ROOT/'VALIDATION/runs/v3_20261012_final_missing_ple_plan/plan.json'
INPUTS = ROOT/'VALIDATION/runs/v3_20261003_ideas_submit/work/inputs.pkl'
TEACHER = ROOT/'VALIDATION/runs/v3_20261003_ideas_submit/work/parts/nn_v46.parquet'
EPOCHS = [14,15,17]
DESIGN = dict(seed=42,stop=17,checkpoints=EPOCHS,schedule_epochs=24,batch=4096,
    features=134,rows=2051327,device='NVIDIA GeForce RTX 3080',host='GPC',
    train='Unmodified original nni.train; observation wrapper saves state before existing epoch monitor',
    preprocessing='Unchanged original Encoder; previously audited exact row-chunk transform/per-column quantile adapter',
    acceptance='All344841 checkpoint14 float32 values exactly equal retained v50 n42_0; fixed zero tolerance',
    failure='Preserve all three states; no expansion, sweep or silent substitution',
    no_inner_fit=True,no_expansion=True,no_final_inference=True,no_upload=True)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):h.update(block)
    return h.hexdigest()


def art(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))


def save(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def freeze():
    assert not (OUT/'identity.json').exists()
    plan=read(PLAN)
    assert plan['original_stops']=={'48':15,'49':17,'50':{'42':14,'142':17,'242':17}}
    paths={Path(__file__),PLAN,Path(parent.__file__),Path(adapter.__file__),Path(nni.__file__),Path(nnj.__file__),parent.MEMBERSHIP}
    for item in plan['sources']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path'];paths.add(ROOT/item['path'])
    assert sha(INPUTS)==plan['input_source']['sha256'];paths.add(INPUTS)
    assert str(torch.__version__)==plan['torch']
    assert nni.NEW==read(ROOT/'VALIDATION/runs/v3_20261003_nn_improve_submit/submission_v48/submission_v48.json')['identity']['recipe']['new']
    assert nnj.RECIPE==read(ROOT/'VALIDATION/runs/v3_20261003_nn_improve_submit/submission_v48/submission_v48.json')['identity']['recipe']['v47']
    original48=read(ROOT/'VALIDATION/runs/v3_20261003_nn_improve_submit/submission_v48/submission_v48.json')
    training=[r for r in original48['identity']['v46_identity']['data_identity'] if r['file'].startswith('training_2025-')]
    assert len(training)==12
    for entry in training:
        path=ROOT/'data'/entry['file'];assert sha(path)==entry['sha256'];paths.add(path)
    OUT.mkdir(parents=True,exist_ok=True)
    for path in paths:
        if path.suffix=='.py':
            target=OUT/'source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    save(OUT/'identity.json',dict(design=DESIGN,sources=[art(p) for p in sorted(paths)],torch=str(torch.__version__),
        new=nni.NEW,original_recipe=nnj.RECIPE,plan_sha256=sha(PLAN),
        expected_inputs=plan['old_input_summary'],mu=plan['label_mean'],sd=plan['label_sd'],
        original_stop_selection_reused=True,original_models_and_encoder_states_absent=True,
        historical_unknowns=['Historical GPU gradient atomic order is unrecoverable.',
            'Historical CPU torch-thread count and v50 TF32 flag were not recorded; scripts used defaults.',
            'Original encoder/bin contents have no saved state hash; source/data and cards/bin counts/scales are retained.'],
        execution=dict(host=os.uname().nodename,torch_cpu_threads=torch.get_num_threads(),
            matmul_allow_tf32=False,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())))
    print(json.dumps(dict(stage='frozen',sources=len(paths),no_fit=True)),flush=True)


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN and identity['torch']==str(torch.__version__)
    for item in identity['sources']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    assert identity['new']==nni.NEW and identity['original_recipe']==nnj.RECIPE
    return identity


def observed_train(d,stop,save_epochs,observe):
    """Observe a prefix without modifying trainer, RNG, optimizer or schedule."""
    original=nni.predict;count=0
    def prediction(model,data,sd,mu,chunk):
        nonlocal count
        count+=1
        if count in save_epochs:observe('state',count,model,None)
        value=original(model,data,sd,mu,chunk)
        if count in save_epochs:observe('prediction',count,model,value)
        return value
    nni.predict=prediction
    try:return nni.train('ple',d,np.arange(len(d.y)),42,stop,(d.held,d.yv))
    finally:
        nni.predict=original
        assert count==stop,'Interrupted trajectory remains saved; do not restart it automatically'


def parity():
    """Tiny CPU controls: original preprocessing, checkpoint observer and RNG."""
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        frame=pd.DataFrame(dict(category=pd.Categorical(['a']*22+['b']*22),
            varied=np.linspace(-2,3,44),missing=np.where(np.arange(44)%9==0,np.nan,np.arange(44))))
        item=dict(X=frame,y=np.linspace(100,300,44),Vn=frame.iloc[:9],raw=pd.DataFrame({'taxi_sec':np.arange(9)+100.}),
            tmonth=np.array(['2025-11']*35+['2025-12']*9),meta={'month':'2025'})
        nni.ts.FOLDS.setdefault('2025',[f'2025-{m:02d}' for m in range(1,13)])
        d=nni.Data(item,'cpu');enc=nnj.Encoder(frame)
        with tempfile.TemporaryDirectory(dir=OUT,prefix='parity_') as tmp:
            cat,num=adapter.encode(enc,frame,tmp,'train',chunk_rows=7)
            bins,ple,plain=adapter.layout(enc,num)
            for a,b in zip(d.train,(torch.tensor(np.array(cat)),torch.tensor(np.array(num[:,ple])),torch.tensor(np.array(num[:,plain])))):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            for a,b in zip(d.bins,bins):torch.testing.assert_close(a,b,atol=0,rtol=0)
            cat._mmap.close();num._mmap.close()
        model,pred,curve=nni.train('ple',d,np.arange(len(d.y)),42,3,(d.held,d.yv))
        state={k:v.detach().clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();np_state=np.random.get_state()
        captures={}
        def observe(kind,epoch,network,value):
            if kind=='state':captures[epoch]={k:v.detach().clone() for k,v in network.state_dict().items()}
        observed,got,other=observed_train(d,3,[1,2,3],observe)
        np.testing.assert_array_equal(pred,got);torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
        other_np=np.random.get_state();assert np_state[0]==other_np[0] and np_state[2:]==other_np[2:]
        np.testing.assert_array_equal(np_state[1],other_np[1])
        for key,value in state.items():
            torch.testing.assert_close(value,observed.state_dict()[key],atol=0,rtol=0)
            torch.testing.assert_close(value,captures[3][key],atol=0,rtol=0)
        assert [x['train_mse_std'] for x in curve]==[x['train_mse_std'] for x in other]
        return dict(original_and_chunked_encoder_tensors_bins_exact=True,
            original_and_observed_training_outputs_weights_losses_RNG_exact=True,synthetic_epochs=3)
    finally:torch.set_num_threads(old_threads)


def prepare():
    identity=guard();assert not (OUT/'prepared.json').exists()
    assert not (OUT/'encoding').exists(),'Preserve interrupted encoding; inspect before any resume'
    controls=parity();data=pd.read_pickle(INPUTS)
    ids=parent.training_ids(data)
    names=data['names'];assert len(names)==134
    X=data['Xn'][names];V=data['Vn'][names]
    assert len(X)==DESIGN['rows'] and len(V)==344841
    check=pd.DataFrame(dict(mvt_id=ids,y=data['yn']))
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.register('members',check)
        n,bad=con.execute('''WITH raw AS (SELECT MVT_ID_mvt mvt_id,TAXITIME_SEC_mvt y,
            ADEP_mvt airport,FLIGHT_ID_mvt plan_id,epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) gap
            FROM read_parquet(?) WHERE PHASE_mvt='DEP'), eligible AS
            (SELECT * FROM raw WHERE y BETWEEN 1 AND 200000
             AND NOT coalesce(airport='LIRF' AND plan_id IS NULL AND gap>=14400,false)
             AND NOT coalesce(airport='LIRF' AND abs(gap-y)<=60,false))
            SELECT count(*),sum(CASE WHEN e.mvt_id IS NULL OR m.mvt_id IS NULL OR e.y IS DISTINCT FROM m.y THEN 1 ELSE 0 END)
            FROM eligible e FULL JOIN members m USING(mvt_id)''',[str(ROOT/'data/training_2025-*.parquet')]).fetchone()
        assert n==DESIGN['rows'] and bad==0,(n,bad)
    del check
    template=pd.read_parquet(ROOT/'data/submitting.parquet',columns=['MVT_ID_mvt']).MVT_ID_mvt
    assert set(data['rank_ids'])==set(template) and len(np.unique(data['rank_ids']))==344841
    encoder=nnj.Encoder(X);directory=OUT/'encoding'
    cat,num=adapter.encode(encoder,X,directory,'train')
    vcat,vnum=adapter.encode(encoder,V,directory,'old')
    bins,ple,plain=adapter.layout(encoder,num)
    actual=dict(numeric=len(encoder.nums),ple_inputs=len(ple),plain_inputs=len(plain),categoricals=encoder.cats,
        cards=encoder.cards,bins_per_input=[len(b)-1 for b in bins],inner_month='2025-12',
        inner_val_rows=int(np.sum(data['tmonth']=='2025-12')),inner_train_rows=int(np.sum(data['tmonth']!='2025-12')))
    assert actual==identity['expected_inputs']
    clipped=np.minimum(data['yn'],nnj.RECIPE['label_clip']);mu=float(clipped.mean());sd=float(clipped.std())
    assert mu==identity['mu'] and sd==identity['sd']
    target=((clipped-mu)/sd).astype(np.float32)
    np.save(directory/'target.npy',target);np.save(directory/'raw_target.npy',data['yn'])
    np.save(directory/'train_ids.npy',ids);np.save(directory/'old_ids.npy',data['rank_ids'])
    for array in [cat,num,vcat,vnum]:array._mmap.close()
    pre=dict(encoder=encoder,bins=bins,ple=ple,plain=plain,cards=encoder.cards,mu=mu,sd=sd,columns=names,info=actual)
    with (OUT/'preprocessing.pkl').open('wb') as stream:pickle.dump(pre,stream,protocol=pickle.HIGHEST_PROTOCOL)
    artifacts=[art(p) for p in sorted(directory.glob('*.npy'))]
    save(OUT/'prepared.json',dict(identity=art(OUT/'identity.json'),preprocessing=art(OUT/'preprocessing.pkl'),
        arrays=artifacts,rows=len(X),old_rows=len(V),columns=names,inputs=actual,controls=controls,
        original_order_labels_airport_stand_runway_month_exact=True,original_cards_bins_counts_scales_exact=True,
        independent_SQL_raw_membership_labels_and_exclusions_exact=True,
        original_encoder_bin_state_hash_unavailable=True,fit_started=False))
    print(json.dumps(dict(stage='preflight_pass',rows=len(X),old_rows=len(V),controls=controls)),flush=True)


def pilot():
    identity=guard();prepared=read(OUT/'prepared.json')
    assert prepared['identity']==art(OUT/'identity.json') and not (OUT/'fit_started.json').exists()
    assert os.uname().nodename==DESIGN['host'] and torch.cuda.get_device_name()==DESIGN['device']
    assert torch.get_num_threads()==identity['execution']['torch_cpu_threads']
    assert torch.backends.cudnn.allow_tf32==identity['execution']['cudnn_allow_tf32']
    assert torch.are_deterministic_algorithms_enabled()==identity['execution']['deterministic_algorithms']
    torch.backends.cuda.matmul.allow_tf32=False
    assert prepared['preprocessing']==art(OUT/'preprocessing.pkl')
    for item in prepared['arrays']:assert sha(ROOT/item['path'])==item['sha256']
    with (OUT/'preprocessing.pkl').open('rb') as stream:pre=pickle.load(stream)
    directory=OUT/'encoding';arrays={name:np.load(directory/f'{name}.npy',mmap_mode='r') for name in ['train_cat','train_num','old_cat','old_num']}
    parts=lambda name:(adapter.tensor_columns(arrays[name+'_cat'],None,'cuda'),
        adapter.tensor_columns(arrays[name+'_num'],pre['ple'],'cuda'),adapter.tensor_columns(arrays[name+'_num'],pre['plain'],'cuda'))
    d=SimpleNamespace(train=parts('train'),held=parts('old'),cards=pre['cards'],bins=pre['bins'],device='cuda',
        y=np.load(directory/'raw_target.npy'),yv=np.full(344841,np.nan),
        ty=torch.from_numpy(np.load(directory/'target.npy')).cuda(),mu=pre['mu'],sd=pre['sd'])
    for array in arrays.values():array._mmap.close()
    ids=np.load(directory/'old_ids.npy');checkpoints={};predictions={};started=time.monotonic()
    save(OUT/'fit_started.json',dict(identity=art(OUT/'identity.json'),preflight=art(OUT/'prepared.json'),
        seed=42,stop=17,checkpoints=EPOCHS,inner_fit=False,automatic_restart_forbidden=True))
    def observe(kind,epoch,model,value):
        if kind=='state':
            path=OUT/f'epoch{epoch}.pt';assert not path.exists()
            temporary=path.with_suffix('.pt.tmp');assert not temporary.exists()
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                cards=pre['cards'],bins=pre['bins'],n_plain=len(pre['plain']),mu=pre['mu'],sd=pre['sd'],
                seed=42,epoch=epoch,schedule_epochs=24,preprocessing=prepared['preprocessing'],
                original_source=art(Path(nni.__file__)),saved_before_monitor=True,resume_training_supported=False),temporary)
            temporary.replace(path)
            checkpoints[str(epoch)]=art(path)
            save(OUT/'checkpoints.json',dict(identity=art(OUT/'identity.json'),states=checkpoints))
        else:
            path=OUT/f'old_epoch{epoch}.parquet';pd.DataFrame(dict(mvt_id=ids,pred=value)).to_parquet(path,index=False)
            predictions[str(epoch)]=art(path)
    model,_,curve=observed_train(d,17,EPOCHS,observe)
    save(OUT/'trained.json',dict(identity=art(OUT/'identity.json'),checkpoints=checkpoints,predictions=predictions,
        seed=42,epochs=17,seconds=time.monotonic()-started,
        curve=[dict(epoch=c['epoch'],train_mse_std=c['train_mse_std'],seconds=c['seconds']) for c in curve],
        saved_before_diagnostics=True,held_labels_used=False))
    del model;gc.collect();torch.cuda.empty_cache()
    reloads={}
    for epoch in EPOCHS:
        c=torch.load(OUT/f'epoch{epoch}.pt',map_location='cuda',weights_only=False)
        model=nni.PleNet(pre['cards'],pre['bins'],len(pre['plain'])).cuda();model.load_state_dict(c['state_dict'])
        got=nni.predict(model,d.held,d.sd,d.mu,65536)
        expected=pd.read_parquet(OUT/f'old_epoch{epoch}.parquet')
        np.testing.assert_array_equal(expected.mvt_id,ids);np.testing.assert_array_equal(expected.pred,got)
        reloads[str(epoch)]=dict(all_old_float_predictions_exact=True,rows=len(got));del model,c
        torch.cuda.empty_cache()
    teacher=pd.read_parquet(TEACHER,columns=['mvt_id','n42_0']).set_index('mvt_id').loc[ids,'n42_0'].to_numpy()
    pred=pd.read_parquet(OUT/'old_epoch14.parquet').pred.to_numpy().astype(np.float32)
    delta=pred.astype(float)-teacher.astype(float);exact=bool(np.array_equal(pred,teacher))
    result=dict(state='exact_old_pilot_no_expansion' if exact else 'closed_nonexact_pilot',identity=art(OUT/'identity.json'),
        trained=art(OUT/'trained.json'),reloads=reloads,original_n42_0_exact=exact,rows=len(pred),
        changed_float32_rows=int(np.count_nonzero(delta)),rms_difference=float(np.sqrt(np.mean(delta**2))),
        maximum_absolute_difference=float(np.max(abs(delta))),
        changed_single_member_rounded_rows=int(np.count_nonzero(np.rint(pred.astype(float))-np.rint(teacher.astype(float)))),
        no_expansion=True,no_final_inference=True,no_upload=True,acceptance_tolerance_fixed=0,
        caveat='A single member fidelity check is not an ensemble accuracy or official-score claim.')
    save(OUT/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','prepare','pilot']);a=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):globals()[a.stage]()
