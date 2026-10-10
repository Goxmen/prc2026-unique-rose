"""Portable full-year physical-scale sparse networks, with frozen15 epochs.

Desktop prepare verifies the existing production sources/caches and the passing
single-member experiment. The laptop verifies a minimal packet of those same
bytes and fits all requested production seeds on one GPU. No upload occurs here.
Models are persisted before inference; completed fits are never repeated.
"""
import argparse,gc,os,pickle,tempfile,time,types
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import nn_sparse_track_submit as old
import nn_track_physical_scales_jan_jul as cv
from records import sha256,write_json
ROOT=cv.ROOT;OUT=ROOT/'VALIDATION/runs/v3_20261011_nn_track_physical_submit'
MANIFEST=OUT/'portable.json';SEEDS=[42,1042,2042];EPOCHS=15
DESIGN=dict(parent='v64 sparse networks',change=cv.DESIGN['model'],original_columns=156,
    seeds=SEEDS,epochs=EPOCHS,epochs_source='original N09 full-year group42',
    training='fixed_epoch_sparse_train.train_updates bytecode; model constructor replaced only',
    ranking_views=['old','final'],inputs='existing verified original156/full-year population and track caches',
    validation='passing audited k0 screen; pool3 CV must pass before selecting production pool3',
    machine='all production members on same laptop GPU',no_new_stopping_or_weight_search=True)

def art(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=sha256(p))
def verify(r):
    p=ROOT/r['path'];assert sha256(p)==r['sha256'],str(p);return p

def prepare():
    assert cv.shared.ts.read(cv.OUT/'audit.json')['status']=='pass'
    score=cv.shared.ts.read(cv.OUT/'score.json')
    assert score['comparisons']['physical_scales_single_mix_vs_selected_sparse3_mix']['decision']=='improvement'
    packet,prepared,identity=old.guard()
    tracks={}
    for name,folder in [('train',cv.base.seq.OUT),('old',old.ranking_cache.OUT/'old'),('final',old.ranking_cache.OUT/'final')]:
        tracks[name]=dict(directory=str(folder.relative_to(ROOT)),
            files=[art(folder/f) for f in ['full_index.parquet','full_points.bin']],
            identity=identity['training_cache'] if name=='train' else identity['ranking_caches'][name])
    sources=dict(identity['helpers'])
    for mod in [cv,cv.physical_scales,old,old.fast,old.adapter]:sources[str(Path(mod.__file__).relative_to(ROOT))]=sha256(mod.__file__)
    value=dict(design=DESIGN,source=art(__file__),helpers=sources,config=identity['config'],
        single_score=art(cv.OUT/'score.json'),single_audit=art(cv.OUT/'audit.json'),
        original_production_identity=art(old.IDENTITY),packet=packet,training_rows=prepared['training_rows'],
        reference_preprocessing=art(old.OUT/'preprocessing/full_year.pkl'),tracks=tracks,
        packet_files=[packet['cached_training_inputs'],packet['training_ids'],art(old.packets.packet_path('final'))])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert cv.shared.ts.read(MANIFEST)==value
    else:write_json(MANIFEST,value)
    print('Verified portable physical-scale production packet',flush=True)

def guard():
    m=cv.shared.ts.read(MANIFEST)
    assert m['design']==DESIGN and m['source']==art(__file__)
    for path,digest in m['helpers'].items():assert sha256(ROOT/path)==digest,path
    for r in [m['single_score'],m['single_audit'],m['original_production_identity'],m['reference_preprocessing'],*m['packet_files']]:verify(r)
    for track in m['tracks'].values():
        for r in track['files']:verify(r)
    assert cv.shared.recipe()==m['config']
    return m

def paths(k):return OUT/'models'/f'k{k}.pt',OUT/'trained'/f'k{k}.json',OUT/'networks'/f'k{k}.json'

def trained(k,m):
    model,rp,_=paths(k)
    if not rp.exists():
        assert not model.exists(),'Unrecorded model; inspect before retrying';return None
    r=cv.shared.ts.read(rp)
    assert r['manifest']==art(MANIFEST) and r['seed']==SEEDS[k] and r['epochs']==EPOCHS
    verify(r['model']);verify(r['preprocessing'])
    return r

def completed(k,m):
    r=trained(k,m);_,_,rp=paths(k)
    if not rp.exists():return None
    record=cv.shared.ts.read(rp);assert r is not None and record['trained']==art(paths(k)[1])
    for a in record['predictions'].values():verify(a)
    return record

def trainer():
    # Preserve the original module and its global state. The new namespace
    # changes exactly the model factory seen by the existing training bytecode.
    proxy=types.SimpleNamespace(**dict(vars(old.fast.original),prototype=cv.physical_scales))
    namespace=dict(old.fast.train_updates.__globals__,original=proxy)
    fn=types.FunctionType(old.fast.train_updates.__code__,namespace,'physical_train_updates',
        old.fast.train_updates.__defaults__,old.fast.train_updates.__closure__)
    assert old.fast.original.prototype is cv.base.prototype
    return fn

def fit(members):
    m=guard();device=torch.cuda.get_device_name()
    assert '5060' in device,'Production seeds allocated to the laptop GPU'
    for k in range(3):
        r=trained(k,m)
        if r is not None:assert r['gpu']==device,'Do not distribute production seeds across GPUs'
    pending=[k for k in members if completed(k,m) is None]
    if not pending:return
    for name in ['models','trained','networks','preprocessing']:(OUT/name).mkdir(exist_ok=True)
    write_json(OUT/'initial_checks.json',dict(checks=[cv.physical_scales.check('cpu'),cv.physical_scales.check('cuda')],
        training_bytecode_exact=True,gpu=device))
    packet=m['packet'];data=pd.read_pickle(verify(packet['cached_training_inputs']))
    saved_ids=pd.read_parquet(verify(packet['training_ids']))
    np.testing.assert_array_equal(saved_ids.original_index,data['Xn'].index)
    train_ids=saved_ids.mvt_id.to_numpy();assert len(train_ids)==len(data['yn'])==m['training_rows']
    names=packet['columns'];assert len(names)==156
    held=[];ids={}
    for view in ['old','final']:
        p=old.packets.load_packet(view,packet,data=data)
        held.append(p['Vn']);ids[view]=np.asarray(p['rank_ids']).copy()
    V=pd.concat(old.parent.fs.widen(held),ignore_index=True);X=data['Xn'].loc[:,names]
    assert list(V)==list(X)==names
    old.parent.cv.ts.FOLDS.setdefault(old.parent.hs.ids.FULL,old.parent.features.TRAIN_MONTHS)
    item=dict(X=X,Vn=V,y=data['yn'],tmonth=data['tmonth'],meta=dict(month=old.parent.hs.ids.FULL),
        raw=pd.DataFrame({'taxi_sec':np.full(len(V),np.nan)}))
    del data,saved_ids,held,p;gc.collect()
    with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT) as scratch:
        d=old.adapter.ChunkedDirectData(item,'cuda',scratch)
    state=d.preprocessing();state['design']=old.DESIGN
    with verify(m['reference_preprocessing']).open('rb') as h:
        cv.base.preprocess_guard.equal_preprocessing(pickle.load(h),state)
    state['design']=DESIGN;pre=OUT/'preprocessing/full_year.pkl'
    if pre.exists():cv.base.preprocess_guard.retain_saved_preprocessing(pre,state)
    else:cv.shared.atomic_pickle(pre,state)
    del X,V,item,state;gc.collect()
    stores={name:old.load_tracks(ROOT/t['directory'],t['identity'],'cuda',None if name=='train' else name)
        for name,t in m['tracks'].items()}
    train=stores.pop('train');rows=train.align(train_ids);views=old.fast.ranking_views(d,ids,stores)
    for k in pending:
        start=time.monotonic();mp,rp,npth=paths(k);r=trained(k,m)
        if r is None:
            model,curve=trainer()(d,train,rows,views,SEEDS[k],EPOCHS)
            tmp=mp.with_suffix('.pt.tmp');torch.save(dict(state_dict=model.state_dict(),design=DESIGN,
                config=m['config'],preprocessing=str(pre.relative_to(ROOT))),tmp);tmp.replace(mp)
            write_json(rp,dict(manifest=art(MANIFEST),seed=SEEDS[k],epochs=EPOCHS,model=art(mp),
                preprocessing=art(pre),curve=curve,minutes=(time.monotonic()-start)/60,
                gpu=device,host=os.uname().nodename,torch=str(torch.__version__)))
        else:
            model=cv.physical_scales.SparseFusion(d.cards,d.bins,d.train[2].shape[1]).cuda()
            model.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
        predictions,seconds=old.fast.infer_views(model,d,views)
        reloaded=cv.physical_scales.SparseFusion(d.cards,d.bins,d.train[2].shape[1]).cuda()
        reloaded.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
        outputs={}
        for view in views:
            pred=predictions[view.name]
            np.testing.assert_array_equal(old.fast.predict_view(reloaded,d,view,limit=old.fast.INFERENCE_CHUNK),pred[:old.fast.INFERENCE_CHUNK])
            path=OUT/'networks'/f'{view.name}_k{k}.parquet'
            pd.DataFrame(dict(mvt_id=ids[view.name],base=pred.astype(np.float32))).to_parquet(path,index=False)
            outputs[view.name]=art(path)
        write_json(npth,dict(trained=art(rp),predictions=outputs,inference_seconds=seconds,
            reload_rows_per_view=old.fast.INFERENCE_CHUNK,gpu=device))
        del model,reloaded,predictions;gc.collect();torch.cuda.empty_cache()
        print(f'Production physical-scale k{k} complete',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit']);ap.add_argument('--members',type=int,nargs='+',default=[0],choices=[0,1,2]);a=ap.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        if a.stage=='prepare':prepare()
        else:fit(a.members)
