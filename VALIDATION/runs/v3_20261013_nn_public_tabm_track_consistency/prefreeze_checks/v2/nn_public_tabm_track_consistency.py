"""Fixed two-epoch source versus target track-view consistency screen.

Six serial laptop continuations. Cached outer-held inputs contain no labels;
held metadata is projected to IDs/month and the legacy adapter receives NaNs.
No source calibration, pseudo target selection, stopping or parameter scan.
"""
import argparse,copy,gc,json,math,os,pickle,platform,tempfile,time
from pathlib import Path
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import nn_public_tabm_pool3 as parent
import track_consistency_view as view
from records import write_json

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'
MANIFEST=OUT/'identity.json'
PROTOCOL=ROOT/'VALIDATION/reports/cohort_learning_20261013/target_track_consistency_protocol.json'
ARMS=['supervised_control','training_consistency','target_consistency']
STOP=parent.STOP
original=parent.original
art,read,verify=parent.art,parent.read,parent.verify
DESIGN=dict(protocol=art(PROTOCOL),parent='exactv80; replace seed42 only',
    arms=ARMS,epochs=2,lr=3e-5,weight_decay=.0003,batch=2048,microbatch=512,
    consistency_weight=1.,consistency_batch=512,seed=42,sampling_seed=20261013,
    view=view.SETTINGS,held_labels='absent from cache and projected metadata; adapter all-NaN placeholder',
    teacher='retained original seed42, frozen eval, unclipped standardized32-head mean',
    no_hpo=True,no_stopping=True,no_extra_seed=True,no_upload=True)


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);write_json(path,value)


def paths(month,arm):
    assert month in STOP and arm in ARMS
    stem=f'{month}_s42'
    return OUT/arm/'models'/f'{stem}.pt',OUT/arm/'networks'/f'{stem}.parquet',OUT/arm/'networks'/f'{stem}.json'


def digest_ids(ids):
    import hashlib
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()


def isolated_item(data,metadata,month):
    # This cache's exact six-field schema has training targets only. Do not use
    # raw metadata loader or include a held label field in the input packet.
    assert set(data)=={'X','V','y','train_ids','ids','baseline'}
    names=data['baseline'];assert len(names)==len(set(names))==156
    assert not set(names)&{'taxi_sec','TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt'}
    train_ids=np.asarray(data['train_ids'],dtype=np.int64)
    held_ids=np.asarray(data['ids'],dtype=np.int64)
    assert len(np.unique(train_ids))==len(train_ids) and len(np.unique(held_ids))==len(held_ids)
    assert not np.intersect1d(train_ids,held_ids).size
    tm=metadata.loc[train_ids,'month'].astype(str).to_numpy()
    hm=metadata.loc[held_ids,'month'].astype(str).to_numpy()
    assert set(tm)==set(original.original.shared.ts.FOLDS[month]) and set(hm)=={month}
    assert len(data['y'])==len(train_ids) and np.isfinite(data['y']).all()
    raw=pd.DataFrame(dict(mvt_id=held_ids,taxi_sec=np.full(len(held_ids),np.nan)))
    item=dict(X=data['X'][names],Vn=data['V'][names],y=np.asarray(data['y']),raw=raw,
        tmonth=tm,meta=dict(month=month))
    scope=dict(train_rows=len(train_ids),held_rows=len(held_ids),
        train_ids_sha256=digest_ids(train_ids),held_ids_sha256=digest_ids(held_ids),
        train_months=sorted(set(tm)),held_months=sorted(set(hm)),
        held_labels_absent_from_cache=True,held_metadata_columns=['mvt_id','month'],
        adapter_targets_all_nan=bool(np.isnan(raw.taxi_sec).all()))
    return item,train_ids,held_ids,scope


def poison_check():
    cols=[f'feature{i}' for i in range(156)]
    months=original.original.shared.ts.FOLDS['2025-01']
    n=len(months);train=np.arange(n);held=np.arange(n,n+2)
    data=dict(X=pd.DataFrame(np.arange(n*156).reshape(n,156),columns=cols),
        V=pd.DataFrame(np.arange(2*156).reshape(2,156),columns=cols),
        y=np.arange(n,dtype=float),train_ids=train,ids=held,baseline=cols)
    m=pd.DataFrame(dict(mvt_id=np.r_[train,held],month=months+['2025-01']*2,
        taxi_sec=np.arange(n+2,dtype=float))).set_index('mvt_id')
    a=isolated_item(data,m,'2025-01');m.loc[held,'taxi_sec']=[1e30,-1e30]
    b=isolated_item(data,m,'2025-01');m=m.drop(columns='taxi_sec');c=isolated_item(data,m,'2025-01')
    for other in [b,c]:
        assert a[3]==other[3]
        for key in ['X','Vn','raw']:pd.testing.assert_frame_equal(a[0][key],other[0][key])
        for key in ['y','tmonth']:np.testing.assert_array_equal(a[0][key],other[0][key])
    return dict(held_label_poison_and_removal_invariant=True,synthetic_only=True)


def make(pre,device):
    return original.make('tabm',pre['cards'],pre['bins'],len(pre['plain'])+12).to(device)


def zero_parity(device):
    # Exact zero-weight branch versus original member loss/AdamW first update.
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,17
    torch.manual_seed(42);a=original.make('tabm',cards,bins,plain).to(device)
    b=copy.deepcopy(a)
    tab,track=original.original.base.prototype.synthetic(cards,bins,plain,12,device)
    y=torch.linspace(-1,1,12,device=device)
    states=[]
    for model,branch in [(a,False),(b,True)]:
        torch.manual_seed(90210);model.train();opt=optimizer(model);opt.zero_grad(set_to_none=True)
        before=model(*tab,*track);error=original.model_type.loss(before,y);error.backward()
        weight=0.
        if branch:view.backward_consistency(model,tab,track,torch.zeros_like(y),weight)
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()}
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device!='cpu' else None
        opt.step();states.append((before.detach(),gradients,cpu,gpu,{n:p.detach().clone() for n,p in model.named_parameters()}))
    for x,z in [(states[0][0],states[1][0]),(states[0][2],states[1][2])]:assert torch.equal(x,z)
    if device!='cpu':assert torch.equal(states[0][3],states[1][3])
    for index in [1,4]:
        for key in states[0][index]:assert torch.equal(states[0][index][key],states[1][index][key]),key
    return dict(device=device,zero_weight_output_gradient_rng_update_exact=True)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.0003,betas=(.9,.999),eps=1e-8,amsgrad=False)


def cpu_check():
    with torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        value=dict(source=art(__file__),view=art(view.__file__),protocol=art(PROTOCOL),
            poison=poison_check(),view_check=view.check('cpu'),zero_parity=zero_parity('cpu'))
    path=OUT/'cpu_checks.json';assert not path.exists();write(path,value);print(value,flush=True)


def freeze():
    assert not MANIFEST.exists()
    cpu=read(OUT/'cpu_checks.json')
    assert cpu['source']==art(__file__) and cpu['view']==art(view.__file__) and cpu['protocol']==art(PROTOCOL)
    review_path=OUT/'source_review.json';review=read(review_path)
    assert review['status']=='pass' and review['source']==art(__file__) and review['view']==art(view.__file__)
    assert review['protocol']==art(PROTOCOL)
    score=ROOT/'VALIDATION/nn_public_tabm_track_consistency_score.py'
    assert review['score']==art(score)
    prior=parent.guard();assert read(parent.OUT/'audit.json')['status']=='pass'
    files=[Path(__file__),Path(view.__file__),score,PROTOCOL,OUT/'cpu_checks.json',review_path,
        parent.MANIFEST,Path(parent.__file__),ROOT/'VALIDATION/reports/cohort_learning_20261013/target_track_consistency_design_review.json',
        ROOT/'VALIDATION/reports/cohort_learning_20261013/track_consistency_source_feasibility.json',verify(review['auditor'])]
    controls={}
    for month in STOP:
        controls[month]=[]
        for k in range(3):
            r=parent.completed(month,k,prior);assert r is not None
            controls[month].append(art(parent.paths(month,k)[2]));files += [parent.paths(month,k)[2]]
            files += [verify(r[key]) for key in ['model','prediction','preprocessing','checkpoint']]
    records={a['path']:a for a in [*prior['files'],*[art(p) for p in files]]}
    value=dict(prior,source=art(__file__),design=DESIGN,files=list(records.values()),
        parent_identity=art(parent.MANIFEST),source_review=art(review_path),controls=controls,
        protocol=art(PROTOCOL),auditor=review['auditor'])
    write(MANIFEST,value)
    (OUT/'transfer_files.txt').write_text('\n'.join(sorted(set(records)|{str(MANIFEST.relative_to(ROOT))}))+'\n')
    print(dict(identity=art(MANIFEST),new_fits=6,files=len(records)),flush=True)


def guard(full=False):
    m=read(MANIFEST);assert m['design']==DESIGN and m['source']==art(__file__) and m['protocol']==art(PROTOCOL)
    assert m['parent_identity']==art(parent.MANIFEST)
    for a in m['files']:verify(a)
    if full:
        for fold in m['folds'].values():verify(fold['matrix'])
        cache=read(verify(m['cache']))
        assert cache['outputs']==m['cache']['outputs']
        for name,digest in cache['outputs'].items():verify(dict(path=str(Path(m['cache']['path']).parent/name),sha256=digest))
    return m


def load_data(month,m,device):
    data=pd.read_pickle(verify(m['folds'][month]['matrix']))
    meta=pd.read_parquet(verify(m['metadata_values']),columns=['mvt_id','month']).set_index('mvt_id')
    item,train_ids,held_ids,scope=isolated_item(data,meta,month)
    del data,meta;gc.collect()
    with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT) as scratch:
        d=original.original.adapter.ChunkedDirectData(item,device,scratch)
    assert np.isnan(d.yv).all()
    saved=verify(m['folds'][month]['preprocessing'])
    with saved.open('rb') as f:pre=pickle.load(f)
    state=d.preprocessing();state['design']=original.original.base.DESIGN
    original.original.base.preprocess_guard.equal_preprocessing(pre,state)
    public=pd.read_parquet(verify(m['public_features'])).set_index('mvt_id')
    for part,ids in [('train',train_ids),('held',held_ids)]:
        extra=torch.from_numpy(original.model_type.transform(public.loc[ids])).to(d.device)
        current=getattr(d,part);setattr(d,part,(*current[:2],torch.cat([current[2],extra],dim=1)))
    del item,public,extra,current;gc.collect()
    return d,train_ids,held_ids,scope,pre


def load_tracks(m,device):
    folder=ROOT/Path(m['cache']['path']).parent
    index=pd.read_parquet(folder/'full_index.parquet')
    points=np.memmap(folder/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    return original.original.base.tensor_seq.RaggedTracks(index,points,device)


def teacher_paths(month):
    folder=OUT/'teacher'/month
    return folder/'train_mean.npy',folder/'held_mean.npy',folder/'receipt.json'


def teacher_completed(month,m):
    tp,hp,rp=teacher_paths(month)
    if not rp.exists():return None
    r=read(rp)
    assert r['identity']==art(MANIFEST) and r['month']==month and r['model']==art(parent.paths(month,0)[0])
    assert r['inputs']==m['folds'][month] and r['all_held_original_exact']
    assert r['train_mean']==art(tp) and r['held_mean']==art(hp)
    for k in ['train_mean','held_mean','model','control_prediction','preprocessing']:verify(r[k])
    return r


def forward_means(model,part,tracks,rows):
    model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(rows),512):
            batch=model(*(x[start:start+512] for x in part),*tracks.batch(rows[start:start+512]))
            values.append(original.model_type.mean(batch).cpu().numpy())
    return np.concatenate(values).astype(np.float32,copy=False)


def cache_teacher(month,m,d,tracks,train_ids,held_ids,scope,pre):
    existing=teacher_completed(month,m)
    if existing:return existing
    tp,hp,rp=teacher_paths(month);assert not tp.exists() and not hp.exists(),'Partial teacher cache: inspect, do not overwrite'
    model=make(pre,d.device)
    checkpoint=torch.load(verify(read(parent.paths(month,0)[2])['model']),map_location=d.device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict']);model.eval().requires_grad_(False)
    tr,va=tracks.align(train_ids),tracks.align(held_ids)
    # Held replay is done first and again at the first fixed512-row chunk.
    held=forward_means(model,d.held,tracks,va)
    repeat=forward_means(model,tuple(x[:512] for x in d.held),tracks,va[:512]);np.testing.assert_array_equal(held[:512],repeat)
    seconds=np.maximum(held.astype(float)*d.sd+d.mu,0).astype(np.float32)
    reference=pd.read_parquet(parent.paths(month,0)[1]);np.testing.assert_array_equal(reference.mvt_id,held_ids)
    np.testing.assert_array_equal(reference.base,seconds)
    train=forward_means(model,d.train,tracks,tr)
    assert np.isfinite(train).all() and np.isfinite(held).all()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    tp.parent.mkdir(parents=True,exist_ok=True);np.save(tp,train);np.save(hp,held)
    eligible={part:int((tracks.counts[rows]>=3).sum()) for part,rows in [('train',tr),('held',va)]}
    assert min(eligible.values())>0
    receipt=dict(identity=art(MANIFEST),month=month,inputs=m['folds'][month],scope=scope,
        model=art(parent.paths(month,0)[0]),control_prediction=art(parent.paths(month,0)[1]),
        preprocessing=m['folds'][month]['preprocessing'],train_mean=art(tp),held_mean=art(hp),
        mu=d.mu,sd=d.sd,eligible=eligible,all_held_original_exact=True,repeated_first_chunk_exact=True,
        teacher_frozen=True,full_inputs=True,unclipped_standardized_mean=True)
    write(rp,receipt);del model,checkpoint,held,train,seconds,reference;gc.collect();torch.cuda.empty_cache()
    return receipt


def preflight():
    m=guard(full=True);path=OUT/'preflight.json';assert not path.exists()
    execution=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))
    assert execution==m['execution']
    original.original.shared.recipe();backend=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
    with torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        checks=[view.check('cuda'),zero_parity('cuda')]
    assert backend==(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
    tracks=load_tracks(m,'cuda');records=[];capacity=[]
    for month in STOP:
        d,train_ids,held_ids,scope,pre=load_data(month,m,'cuda')
        rec=cache_teacher(month,m,d,tracks,train_ids,held_ids,scope,pre);records.append(art(teacher_paths(month)[2]))
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=make(pre,'cuda').train();opt=optimizer(model)
        tab,track=original.original.base.prototype.synthetic(pre['cards'],pre['bins'],len(pre['plain'])+12,512,'cuda')
        track=(track[0],torch.ones_like(track[1]),track[2]);target=torch.zeros(512,device='cuda')
        timing=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
            for _ in range(4):original.model_type.loss(model(*tab,*track),target).div(4).backward()
            view.consistency_loss(model,tab,track,target).backward();opt.step();torch.cuda.synchronize();timing.append(time.monotonic()-start)
        peak=torch.cuda.max_memory_allocated();assert peak<7*(1<<30),'Reject frozen capacity; do not resize'
        capacity.append(dict(month=month,actual_train_held_tensors_and_track_geometry_reserved=True,
            peak_allocated_bytes=peak,logical_update_seconds=float(np.mean(timing[1:])),
            estimated_single_arm_minutes=float(np.mean(timing[1:])*2*math.ceil(len(train_ids)/2048)/60),
            input_scope=scope,eligible=rec['eligible']))
        del d,model,opt,tab,track,target;gc.collect();torch.cuda.empty_cache()
    write(path,dict(status='pass',identity=art(MANIFEST),execution=execution,checks=checks,
        cpu_checks=art(OUT/'cpu_checks.json'),teachers=records,capacity=capacity,
        deterministic_cudnn_check_only=True,original_backend_flags=list(backend),restored=True,
        no_candidate_fit=True));print(dict(status='pass',capacity=capacity),flush=True)


def completed(month,arm,identity):
    mp,pp,rp=paths(month,arm)
    if not rp.exists():return None
    r=read(rp)
    assert r['identity']==art(MANIFEST) and r['design']==DESIGN and r['month']==month and r['arm']==arm
    assert r['seed']==42 and r['epochs']==2 and r['inputs']==identity['folds'][month]
    assert r['all_held_reloaded_exact'] and r['teacher']==art(teacher_paths(month)[2])
    assert r['model']==art(mp) and r['prediction']==art(pp)
    for k in ['model','prediction','preprocessing','checkpoint','teacher']:verify(r[k])
    trained=read(verify(r['checkpoint']))
    assert trained['identity']==art(MANIFEST) and trained['month']==month and trained['arm']==arm
    assert trained['model']==r['model'] and trained['teacher']==r['teacher']
    for key,path in [('preflight',OUT/'preflight.json'),('prefit_audit',OUT/'prefit_audit.json'),('allocation',OUT/'allocation.json')]:
        assert trained[key]==art(path);verify(trained[key])
    return r


def train(model,d,tracks,rows,teacher,eligible,arm):
    opt=optimizer(model);count=len(d.y)
    generator=torch.Generator(device=d.device).manual_seed(42)
    sampler=None if arm=='supervised_control' else view.CycleSampler(eligible)
    part=d.train if arm=='training_consistency' else d.held
    pool_rows=rows['train'] if arm=='training_consistency' else rows['held']
    curve=[];updates=0
    for epoch in range(2):
        start=time.monotonic();perm=torch.randperm(count,device=d.device,generator=generator);cpu=perm.cpu().numpy();total=0.;ctotal=0.;steps=0
        for i in range(0,count,2048):
            end=min(i+2048,count);opt.zero_grad(set_to_none=True);logical=0.
            for j in range(i,end,512):
                stop=min(j+512,end);ix=perm[j:stop]
                output=model(*(x[ix] for x in d.train),*tracks.batch(rows['train'][cpu[j:stop]]))
                error=original.model_type.loss(output,d.ty[ix]);(error*((stop-j)/(end-i))).backward()
                logical+=float(error.detach())*(stop-j)
            if sampler is not None:
                selected=sampler.take(512);ix=torch.from_numpy(selected).to(d.device)
                target=torch.from_numpy(np.asarray(teacher[selected])).to(d.device)
                ctotal+=view.backward_consistency(model,tuple(x[ix] for x in part),tracks.batch(pool_rows[selected]),target)
            assert math.isfinite(logical)
            opt.step();total+=logical;steps+=1;updates+=1
        assert all(torch.isfinite(p).all() for p in model.parameters())
        curve.append(dict(epoch=epoch+1,train_mse_std=total/count,consistency_mse_std=None if sampler is None else ctotal/steps,
            updates=steps,consistency_rows=0 if sampler is None else 512*steps,seconds=time.monotonic()-start))
        print(dict(arm=arm,**curve[-1]),flush=True)
    assert updates==2*math.ceil(count/2048)
    return curve,dict(optimizer=opt.state_dict(),permutation_generator=generator.get_state(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),updates=updates,
        sampler=None if sampler is None else dict(rng=sampler.rng.bit_generator.state,order=sampler.order,cursor=sampler.cursor,cycles=sampler.cycles,draws=sampler.draws))


def fit():
    m=guard(full=True);pref=read(OUT/'preflight.json');audit=read(OUT/'prefit_audit.json');allocation=read(OUT/'allocation.json')
    assert pref['status']==audit['status']=='pass' and pref['identity']==audit['identity']==art(MANIFEST)
    assert audit['preflight']==art(OUT/'preflight.json') and audit['teachers']==pref['teachers']
    for a in pref['teachers']:verify(a)
    assert allocation['authority']=='/root' and allocation['action']=='fit_fixed_track_consistency'
    assert allocation['identity']==art(MANIFEST) and allocation['prefit_audit']==art(OUT/'prefit_audit.json')
    comparison=read(OUT/'comparison/identity.json')
    assert comparison['before_fitting'] and comparison['protocol']==art(PROTOCOL)
    assert art(MANIFEST) in comparison['sources']
    for a in comparison['sources']:verify(a)
    assert dict(host=platform.node(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))==m['execution']
    original.original.shared.recipe();tracks=load_tracks(m,'cuda')
    for month in STOP:
        if all(completed(month,a,m) is not None for a in ARMS):continue
        d,train_ids,held_ids,scope,pre=load_data(month,m,'cuda')
        rec=teacher_completed(month,m);assert rec is not None and rec['scope']==scope
        rows=dict(train=tracks.align(train_ids),held=tracks.align(held_ids))
        means={part:np.load(teacher_paths(month)[i],mmap_mode='r',allow_pickle=False) for i,part in enumerate(['train','held'])}
        for arm in ARMS:
            if completed(month,arm,m) is not None:continue
            mp,pp,rp=paths(month,arm);checkpoint_receipt=OUT/arm/'trained'/f'{month}_s42.json';started=OUT/arm/'started'/f'{month}.json'
            torch.manual_seed(42);np.random.seed(42);model=make(pre,'cuda').train()
            if checkpoint_receipt.exists():
                r=read(checkpoint_receipt)
                assert r['identity']==art(MANIFEST) and r['arm']==arm and r['month']==month and r['teacher']==art(teacher_paths(month)[2])
                for key,path in [('preflight',OUT/'preflight.json'),('prefit_audit',OUT/'prefit_audit.json'),('allocation',OUT/'allocation.json')]:
                    assert r[key]==art(path);verify(r[key])
                saved=torch.load(verify(r['model']),map_location='cuda',weights_only=False)
                assert saved['identity']==art(MANIFEST) and saved['design']==DESIGN and saved['arm']==arm and saved['month']==month
                model.load_state_dict(saved['state_dict']);curve=saved['curve'];assert len(curve)==2
            else:
                assert not mp.exists() and not pp.exists() and not started.exists(),'Partial unrecorded fit: preserve and inspect; never refit automatically'
                write(started,dict(identity=art(MANIFEST),arm=arm,month=month,allocation=art(OUT/'allocation.json')))
                control=torch.load(parent.paths(month,0)[0],map_location='cuda',weights_only=False)
                model.load_state_dict(control['state_dict']);del control
                pool='train' if arm=='training_consistency' else 'held';eligible=np.flatnonzero(tracks.counts[rows[pool]]>=3)
                curve,terminal=train(model,d,tracks,rows,means[pool],eligible,arm)
                mp.parent.mkdir(parents=True,exist_ok=True);tmp=mp.with_suffix('.pt.tmp')
                torch.save(dict(identity=art(MANIFEST),design=DESIGN,month=month,arm=arm,
                    state_dict=model.state_dict(),curve=curve,terminal_training_state=terminal),tmp);tmp.replace(mp)
                write(checkpoint_receipt,dict(identity=art(MANIFEST),arm=arm,month=month,model=art(mp),
                    teacher=art(teacher_paths(month)[2]),preflight=art(OUT/'preflight.json'),prefit_audit=art(OUT/'prefit_audit.json'),
                    allocation=art(OUT/'allocation.json'),updates_completed_before_inference=True))
                del terminal
            mean=forward_means(model,d.held,tracks,rows['held']);prediction=np.maximum(mean.astype(float)*d.sd+d.mu,0)
            assert np.isfinite(prediction).all()
            if pp.exists():
                old=pd.read_parquet(pp);np.testing.assert_array_equal(old.mvt_id,held_ids);np.testing.assert_array_equal(old.base,prediction.astype(np.float32))
            else:
                pp.parent.mkdir(parents=True,exist_ok=True)
                original.original.shared.nni.save_network(pp,held_ids,prediction)
            # Constructor/reset RNG after training cannot affect a later arm:
            # each arm initializes its complete stream from the same fixed seed.
            restored=make(pre,'cuda');restored.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
            actual=forward_means(restored,d.held,tracks,rows['held']);np.testing.assert_array_equal(actual,mean)
            write(rp,dict(identity=art(MANIFEST),design=DESIGN,month=month,arm=arm,seed=42,epochs=2,
                inputs=m['folds'][month],model=art(mp),prediction=art(pp),preprocessing=m['folds'][month]['preprocessing'],
                checkpoint=art(checkpoint_receipt),teacher=art(teacher_paths(month)[2]),
                train_rows=len(train_ids),held_rows=len(held_ids),scope=scope,curve=curve,
                all_held_reloaded_exact=True,host=platform.node(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
            del model,restored,mean,actual,prediction;gc.collect();torch.cuda.empty_cache()
        del d,means;gc.collect();torch.cuda.empty_cache()
    write(OUT/'audit.json',dict(status='pass',identity=art(MANIFEST),members=[art(paths(m,a)[2]) for m in STOP for a in ARMS],
        preflight=art(OUT/'preflight.json'),prefit_audit=art(OUT/'prefit_audit.json'),allocation=art(OUT/'allocation.json'),
        all_held_labels_nan=True,all_held_reload_exact=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['cpu_check','freeze','preflight','fit'])
    torch.set_num_threads(2)
    with threadpool_limits(2):globals()[ap.parse_args().stage]()
