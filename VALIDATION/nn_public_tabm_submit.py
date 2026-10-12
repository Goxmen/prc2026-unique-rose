"""One frozen full-year seed-42 TabM member; no submission builder or upload.

The other two selected public/physical members remain byte-for-byte unchanged.
Reuse the audited full-year input/track caches and original 15-epoch stop.
The exact CV update function is reused in an isolated output namespace.
"""
import argparse,gc,math,os,pickle,sys,tempfile,time,types
from pathlib import Path
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import nn_public_tabm_jan_jul as cv
import nn_public_submit as parent
from records import write_json
ROOT=cv.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_submit'
MANIFEST=OUT/'identity.json'
EPOCHS=15
SEED=42
TRAIN_ROWS=2051327
RANK_ROWS=344841
VIEW='old'
CHOICE='replace_public_k0_with_tabm42'
art,verify,read=cv.art,cv.verify,cv.read
old=parent.old
DESIGN=dict(choice=CHOICE,model=cv.model_type.SETTINGS,seed=SEED,epochs=EPOCHS,
    stop_source='unchanged N09 full-year group42 and selected public production stop',
    schedule_epochs=24,effective_batch=cv.BATCH,microbatch=cv.MICRO,
    target='min(raw taxi-out,10800), unchanged population mean/std',
    population=TRAIN_ROWS,ranking_view=VIEW,ranking_rows=RANK_ROWS,
    columns=156,public_columns=12,preprocessing='exact original full-year encoder; append frozen public transform',
    track='unchanged separately bound train and old ranking stores',
    execution='identical nn_public_tabm_jan_jul.train bytecode, only output directory replaced',
    optimizer='AdamW lr .002, weight decay .0003; individual-head MSE',
    inference='fixed 512-row chunks, head mean then original seconds scaling/nonnegative clip',
    retained='original public/physical seeds1042 and2042, saved predictions unchanged',
    export='one TabM member plus equal mean of [TabM42,original1042,original2042]',
    machine='laptop RTX5060; same host/GPU as passing TabM CV family screen',
    no_early_stopping=True,no_extra_seed=True,no_weight_search=True,no_whole_pool_substitution=True,
    no_candidate_stack_or_submission_build=True,no_upload=True)


def trainer():
    source=cv.train
    fn=types.FunctionType(source.__code__,dict(source.__globals__,OUT=OUT),
        'full_year_tabm_train',source.__defaults__,source.__closure__)
    assert fn.__code__ is source.__code__ and cv.OUT!=OUT
    return fn


def check():
    cv.original.shared.recipe();result=cv.model_type.check('cpu')
    trainer()
    assert parent.EPOCHS==EPOCHS and parent.SEEDS==[42,1042,2042]
    return dict(model=result,training_bytecode_exact=True,original_globals_unchanged=True)


def prepare():
    assert not list(OUT.glob('**/*.pt')) and not list(OUT.glob('**/*.parquet'))
    selection=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_stack'
    transport=ROOT/'VALIDATION/runs/v3_20261012_delta_transport/tabm'
    result=read(selection/'result.json');secondary=read(transport/'result.json')
    audit=read(ROOT/'VALIDATION/reports/public_tabm_20261012/score_audit.json')
    assert result['paired']['replace_tabm_vs_selected']['decision']=='improvement'
    assert result['paired']['replace_tabm_vs_replace_control']['decision']=='improvement'
    assert secondary['paired']['replace_tabm']['decision']=='improvement' and audit['status']=='pass'
    assert read(selection/'audit.json')['result']==art(selection/'result.json')
    for record in audit['inputs']:verify(record)
    cv_identity=read(cv.MANIFEST)
    assert cv_identity['source']==art(cv.__file__) and cv_identity['design']==cv.DESIGN
    for record in cv_identity['files']:verify(record)
    cv_audit=read(cv.OUT/'audit.json')
    assert cv_audit['identity']==art(cv.MANIFEST) and cv_audit['status']=='pass'
    for record in cv_audit['members']:verify(record)
    prior=read(parent.MANIFEST);packet=prior['parent_packet'];inputs=packet['packet']
    assert prior['source']==art(parent.__file__) and prior['design']==parent.DESIGN
    for record in prior['helpers']:verify(record)
    parent_audit=read(parent.OUT/'audit.json')
    assert parent_audit['status']=='pass' and parent_audit['identity']==art(parent.MANIFEST)
    assert prior['design']['training_rows']==TRAIN_ROWS and prior['design']['epochs']==EPOCHS
    history_path=old.parent.PARENT_NN;history=read(history_path)
    assert history['groups']['42']['stop_epoch']==EPOCHS
    assert history['config']==packet['config']==cv.original.shared.recipe()
    assert len(inputs['columns'])==len(set(inputs['columns']))==156
    members={}
    for k,seed in enumerate([42,1042,2042]):
        trained=read(parent.paths(k)[1]);record=read(parent.paths(k)[2])
        assert trained['identity']==art(parent.MANIFEST) and trained['seed']==seed and trained['epochs']==EPOCHS
        assert record['trained']==art(parent.paths(k)[1]) and record['rows']==RANK_ROWS
        assert record['ranking_view']==VIEW and record['all_ranking_reload_exact']
        assert parent_audit['checks'][k]['member']==k and parent_audit['checks'][k]['record']==art(parent.paths(k)[2])
        verify(record['prediction'])
        members[str(k)]=dict(seed=seed,trained=art(parent.paths(k)[1]),record=art(parent.paths(k)[2]),prediction=record['prediction'])
    resources=dict(matrix=inputs['cached_training_inputs'],training_ids=inputs['training_ids'],
        reference_preprocessing=packet['reference_preprocessing'],
        training_features=prior['training_features'],ranking_features=prior['ranking_features'],
        template=art(ROOT/'data/submitting.parquet'))
    files={Path(__file__),Path(cv.__file__),Path(cv.model_type.__file__),parent.MANIFEST,history_path,
        selection/'result.json',selection/'audit.json',transport/'result.json',cv.MANIFEST,cv.OUT/'audit.json',
        ROOT/'VALIDATION/reports/public_tabm_20261012/score_audit.json',
        ROOT/'VALIDATION/tests/test_public_tabm_submit.py',parent.OUT/'audit.json'}
    for key in ['training_feature_audit','ranking_feature_audit']:files.add(verify(prior[key]))
    for member in members.values():
        files.update(ROOT/member[k]['path'] for k in ['trained','record','prediction'])
    # Frozen local Python dependencies; numerical libraries retain their recorded versions.
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'.venv'):files.add(path)
    tracks={name:packet['tracks'][name] for name in ['train',VIEW]}
    for name,t in tracks.items():
        identity=t['identity'];manifest=identity if name=='train' else identity['manifest']
        files.add(ROOT/manifest['path'])
        if name!= 'train':files.add(ROOT/identity['design']['path'])
    value=dict(design=DESIGN,source=art(__file__),checks=check(),resources=resources,columns=inputs['columns'],
        ranking_packet=dict(columns=inputs['columns'],cached_training_inputs=inputs['cached_training_inputs'],views={VIEW:inputs['views'][VIEW]}),
        tracks=tracks,parent_identity=art(parent.MANIFEST),parent_members=members,
        historical_epochs=art(history_path),config=history['config'],
        cv_identity=art(cv.MANIFEST),selected_v78=art(selection/'result.json'),selected_v79=art(transport/'result.json'),
        files=[art(p) for p in sorted(files)],training_months=old.parent.features.TRAIN_MONTHS,
        source_validation='audited unchanged parent packet and exact CV evidence; full needed resource bytes verified on allocated laptop before preflight/fit')
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value,'Frozen production design/source changed'
    else:write_json(MANIFEST,value)
    paths={r['path'] for r in value['files']}|{r['path'] for r in resources.values()}
    paths.update(r['path'] for t in tracks.values() for r in t['files'])
    (OUT/'transfer_files.txt').write_text('\n'.join(sorted(paths|{str(MANIFEST.relative_to(ROOT))}))+'\n')
    print(dict(frozen=True,identity=art(MANIFEST),files=len(paths)),flush=True)


def guard(full=True):
    value=read(MANIFEST)
    assert value['design']==DESIGN and value['source']==art(__file__)
    for record in value['files']:verify(record)
    for record in value['resources'].values():
        if full:verify(record)
        else:assert (ROOT/record['path']).is_file()
    for name,t in value['tracks'].items():
        ident=t['identity'];manifest=read(verify(ident if name=='train' else ident['manifest']))
        assert manifest['outputs']==ident['outputs'] and manifest['rows']==ident['rows']
        assert manifest['stored_points']==ident['stored_points']
        assert manifest['settings']['original_sequence']==cv.original.base.seq.SETTINGS
        if name!= 'train':
            assert ident['ranking_view']==manifest['settings']['ranking_view']==VIEW
            verify(ident['design'])
        for record in t['files']:
            assert manifest['outputs'][Path(record['path']).name]==record['sha256']
            if full:verify(record)
    assert read(verify(value['historical_epochs']))['groups']['42']['stop_epoch']==EPOCHS
    assert cv.original.shared.recipe()==value['config']
    return value


def preflight():
    identity=guard();assert not (OUT/'preflight.json').exists()
    assert '5060' in torch.cuda.get_device_name()
    with verify(identity['resources']['reference_preprocessing']).open('rb') as f:pre=pickle.load(f)
    checks=[cv.model_type.check('cuda')]
    rows=TRAIN_ROWS+RANK_ROWS
    reserve=rows*(8*len(pre['cards'])+4*(len(pre['ple'])+len(pre['plain'])+12))+4*TRAIN_ROWS
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();storage=torch.empty(reserve,dtype=torch.uint8,device='cuda')
    torch.manual_seed(SEED);model=cv.model_type.PublicTabM(pre['cards'],pre['bins'],len(pre['plain'])+12).cuda().train()
    tab,track=cv.original.base.prototype.synthetic(pre['cards'],pre['bins'],len(pre['plain'])+12,cv.MICRO,'cuda')
    track=(track[0],torch.ones_like(track[1]),track[2]);target=torch.zeros(cv.MICRO,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003);times=[]
    for step in range(4):
        torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
        for _ in range(cv.BATCH//cv.MICRO):cv.model_type.loss(model(*tab,*track),target).div(cv.BATCH//cv.MICRO).backward()
        opt.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
    result=dict(identity=art(MANIFEST),checks=checks,gpu=torch.cuda.get_device_name(),host=os.uname().nodename,
        torch=str(torch.__version__),reserved_input_bytes=reserve,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        logical_step_seconds=float(np.mean(times[1:])),estimated_fit_minutes=float(np.mean(times[1:])*math.ceil(TRAIN_ROWS/cv.BATCH)*EPOCHS/60),
        capacity_adjustment=False,training_only_dense128point_probe=True)
    write_json(OUT/'preflight.json',result);print(result,flush=True)


def trained(identity):
    path=OUT/'trained/k0.json';model=OUT/'models/k0.pt'
    if not path.exists():
        assert not model.exists(),'Unrecorded weights; inspect instead of refitting'
        return None
    value=read(path)
    assert value['identity']==art(MANIFEST) and value['design']==DESIGN
    assert value['seed']==SEED and value['epochs']==EPOCHS and value['training_rows']==TRAIN_ROWS
    assert value['resources']==identity['resources'] and len(value['curve'])==EPOCHS
    assert [x['epoch'] for x in value['curve']]==list(range(1,EPOCHS+1))
    assert value['updates_completed_before_inference']
    assert value['model']==art(model) and value['preprocessing']==art(OUT/'preprocessing/full_year.pkl')
    for field in ['model','preprocessing']:verify(value[field])
    return value


def completed(identity):
    record=trained(identity);path=OUT/'networks/k0.json'
    if not path.exists():return None
    assert record is not None
    value=read(path)
    assert value['identity']==art(MANIFEST) and value['trained']==art(OUT/'trained/k0.json')
    assert value['ranking_view']==VIEW and value['rows']==RANK_ROWS and value['all_ranking_reload_exact']
    assert value['prediction']==art(OUT/'networks/old_k0.parquet')
    verify(value['prediction']);return value


def fit():
    identity=guard();preflight=read(OUT/'preflight.json')
    assert preflight['identity']==art(MANIFEST) and preflight['gpu']==torch.cuda.get_device_name()
    assert preflight['host']==os.uname().nodename
    if completed(identity) is not None:return
    for folder in ['models','networks','trained','preprocessing','runtime']:(OUT/folder).mkdir(exist_ok=True)
    resource=identity['resources'];data=pd.read_pickle(verify(resource['matrix']))
    membership=pd.read_parquet(verify(resource['training_ids']))
    np.testing.assert_array_equal(membership.original_index,data['Xn'].index)
    rank=old.packets.load_packet(VIEW,identity['ranking_packet'],data=data)
    tids=membership.mvt_id.to_numpy();ids=np.asarray(rank['rank_ids']).copy();names=identity['columns']
    assert len(tids)==len(data['yn'])==TRAIN_ROWS and len(ids)==RANK_ROWS
    assert pd.Index(tids).is_unique and pd.Index(ids).is_unique and not np.intersect1d(tids,ids).size
    assert set(data['tmonth'])==set(identity['training_months'])
    template_ids=pd.read_parquet(verify(resource['template']),columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    np.testing.assert_array_equal(ids,template_ids)
    for member in identity['parent_members'].values():
        prior_ids=pd.read_parquet(verify(member['prediction']),columns=['mvt_id']).mvt_id.to_numpy()
        np.testing.assert_array_equal(ids,prior_ids)
    old.parent.cv.ts.FOLDS.setdefault(old.parent.hs.ids.FULL,identity['training_months'])
    item=dict(X=data['Xn'][names],Vn=rank['Vn'],y=data['yn'],tmonth=data['tmonth'],
        meta=dict(month=old.parent.hs.ids.FULL),raw=pd.DataFrame({'taxi_sec':np.full(len(ids),np.nan)}))
    del data,membership,rank,template_ids,prior_ids;gc.collect()
    with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT) as scratch:
        d=old.adapter.ChunkedDirectData(item,'cuda',scratch)
    assert not np.any(d.offset) and not np.any(d.val_offset)
    state=d.preprocessing();state['design']=old.DESIGN
    with verify(resource['reference_preprocessing']).open('rb') as f:cv.original.base.preprocess_guard.equal_preprocessing(pickle.load(f),state)
    state['design']=DESIGN;pre=OUT/'preprocessing/full_year.pkl'
    if pre.exists():cv.original.base.preprocess_guard.retain_saved_preprocessing(pre,state)
    else:cv.original.shared.atomic_pickle(pre,state)
    for part,flight_ids,key in [('train',tids,'training_features'),('held',ids,'ranking_features')]:
        public=pd.read_parquet(verify(resource[key])).set_index('mvt_id');assert public.index.is_unique
        extra=torch.from_numpy(cv.model_type.transform(public.loc[flight_ids])).to(d.device)
        before=getattr(d,part);setattr(d,part,(*before[:2],torch.cat([before[2],extra],dim=1)))
        del public,extra,before
    del item,state;gc.collect()
    stores={name:old.load_tracks(ROOT/t['directory'],t['identity'],'cuda',None if name=='train' else VIEW) for name,t in identity['tracks'].items()}
    tr,va=stores['train'].align(tids),stores[VIEW].align(ids)
    torch.manual_seed(SEED);np.random.seed(SEED)
    model=cv.model_type.PublicTabM(d.cards,d.bins,d.train[2].shape[1]).cuda();record=trained(identity);mp=OUT/'models/k0.pt'
    if record is None:
        curve=trainer()(model,d,stores['train'],tr,EPOCHS,'tabm','full_year')
        tmp=mp.with_suffix('.pt.tmp');torch.save(dict(state_dict=model.state_dict(),identity=art(MANIFEST),design=DESIGN,curve=curve),tmp);tmp.replace(mp)
        write_json(OUT/'trained/k0.json',dict(identity=art(MANIFEST),design=DESIGN,seed=SEED,epochs=EPOCHS,
            training_rows=TRAIN_ROWS,resources=resource,model=art(mp),preprocessing=art(pre),curve=curve,
            gpu=torch.cuda.get_device_name(),host=os.uname().nodename,torch=str(torch.__version__),updates_completed_before_inference=True))
    else:
        assert record['preprocessing']==art(pre)
        checkpoint=torch.load(verify(record['model']),map_location='cuda',weights_only=False)
        assert checkpoint['identity']==art(MANIFEST) and checkpoint['design']==DESIGN
        model.load_state_dict(checkpoint['state_dict'])
    prediction=cv.predict(model,d,stores[VIEW],va).astype(np.float32)
    restored=cv.model_type.PublicTabM(d.cards,d.bins,d.train[2].shape[1]).cuda()
    restored.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
    reloaded=cv.predict(restored,d,stores[VIEW],va).astype(np.float32)
    np.testing.assert_array_equal(prediction,reloaded)
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    path=OUT/'networks/old_k0.parquet'
    if path.exists():
        saved=pd.read_parquet(path);np.testing.assert_array_equal(saved.mvt_id,ids);np.testing.assert_array_equal(saved.base,prediction)
    else:pd.DataFrame(dict(mvt_id=ids,base=prediction)).to_parquet(path,index=False)
    write_json(OUT/'networks/k0.json',dict(identity=art(MANIFEST),trained=art(OUT/'trained/k0.json'),prediction=art(path),
        ranking_view=VIEW,rows=len(ids),all_ranking_reload_exact=True))
    print('Single full-year TabM saved and every old-ranking row reloaded exactly',flush=True)


def export():
    identity=guard(full=False);record=completed(identity);assert record is not None
    assert not (OUT/'export.json').exists(),'Preserve completed export'
    frames=[pd.read_parquet(verify(identity['parent_members'][str(k)]['prediction'])) for k in range(3)]
    new=pd.read_parquet(verify(record['prediction']));ids=frames[0].mvt_id
    for f in [*frames[1:],new]:np.testing.assert_array_equal(f.mvt_id,ids)
    assert len(ids)==RANK_ROWS and ids.is_unique
    before=np.mean([f.base.to_numpy(dtype=float) for f in frames],axis=0)
    after=np.mean([new.base.to_numpy(dtype=float),*(f.base.to_numpy(dtype=float) for f in frames[1:])],axis=0)
    result=pd.DataFrame(dict(mvt_id=ids,parent_k0=frames[0].base,tabm_k0=new.base,retained_k1=frames[1].base,
        retained_k2=frames[2].base,parent_pool=before,replacement_pool=after))
    assert np.isfinite(result.drop(columns='mvt_id').to_numpy()).all()
    result.to_parquet(OUT/'replacement_pool.parquet',index=False)
    write_json(OUT/'export.json',dict(identity=art(MANIFEST),member=art(OUT/'networks/k0.json'),
        retained_members={k:identity['parent_members'][k] for k in ['1','2']},pool=art(OUT/'replacement_pool.parquet'),
        rows=len(ids),view=VIEW,formula='float64 equal mean of saved float32 member predictions',
        scope='Standalone network replacement only; no final stack, candidate file or upload'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['check','prepare','preflight','fit','export'])
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(2):
        value=globals()[args.stage]()
        if value is not None:print(value,flush=True)
