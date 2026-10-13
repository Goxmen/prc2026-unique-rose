"""Fixed TabICLv2 April mechanism screen; no optimizer or configuration search.

CLI: freeze, preflight, infer, score. The established retrieval input/audit chain
is reused. Numeric stages require a separate isolated environment receipt,
root allocation, and an independent audit before inference/scoring. Context
pickles contain fitted preprocessing only; original learned weights stay in the
hash-pinned checkpoint and volatile KV caches may be rebuilt without refitting.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'VALIDATION/reports/tabicl_frontier_20261013'
PLAN = REPORT/'fixed_protocol.json'
OUT = ROOT/'VALIDATION/runs/v3_20261013_tabicl_frontier'
OLD = ROOT/'VALIDATION/runs/v3_20261013_train_label_retrieval'
AUDITOR = ROOT/'VALIDATION/tools/audit_tabicl_frontier_20261013.py'
ARMS = ['labels_off', 'labels_on']


def read(p): return json.loads(Path(p).read_text())


def art(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return dict(path=str(p.relative_to(ROOT)), sha256=h.hexdigest(), bytes=p.stat().st_size)


def verify(a):
    p = (ROOT/a['path']).resolve(); p.relative_to(ROOT)
    b = art(p)
    assert all(b[k] == v for k, v in a.items()), p
    return p


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(x, f, indent=2, allow_nan=False); f.write('\n')


def freeze():
    review = read(OUT/'source_review.json')
    assert review['status'] == 'pass' and review['source'] == art(__file__)
    assert review['auditor'] == art(AUDITOR) and review['protocol'] == art(PLAN)
    plan = read(PLAN)
    for a in plan['sources'] + [plan['inputs'][s] for s in
                              ['inherited_identity', 'prepared', 'independent_input_audit']]: verify(a)
    audit = read(OLD/'prefit_audit.json'); prepared = read(OLD/'prepared.json')
    assert audit['status'] == 'pass' and audit['identity'] == art(OLD/'identity.json')
    assert audit['prepared'] == art(OLD/'prepared.json')
    for a in prepared['artifacts']: verify(a)
    environment = read(OUT/'environment.json')
    assert environment['status'] == 'pass' and environment['package_revision'] == plan['package']['revision']
    checkpoint = environment['checkpoint']; verify(checkpoint)
    expected = plan['checkpoint']['file']
    assert checkpoint['sha256'] == expected['lfs']['sha256'] and checkpoint['bytes'] == expected['size']
    assert environment['isolated_prefix'] != str(ROOT/'.venv')
    for a in environment['artifacts']: verify(a)
    sources = [art(__file__), art(AUDITOR), art(PLAN), art(OUT/'source_review.json'),
               art(OUT/'environment.json'), art(OLD/'identity.json'), art(OLD/'prepared.json'),
               art(OLD/'prefit_audit.json'), art(OLD/'preflight.json')]
    if 'audit_recovery' in audit: sources.append(audit['audit_recovery'])
    write(OUT/'identity.json', dict(source=art(__file__), auditor=art(AUDITOR), protocol=art(PLAN),
        environment=art(OUT/'environment.json'), sources=sources,
        inputs=prepared['artifacts'], checkpoint=checkpoint,
        no_target_results_used_to_choose_settings=True))


def start(stage):
    identity = read(OUT/'identity.json')
    for a in identity['sources']: verify(a)
    assert identity['source'] == art(__file__) and identity['auditor'] == art(AUDITOR)
    allocation = read(OUT/'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(OUT/'identity.json')
    assert allocation['state'] == 'released_for_fixed_tabicl_frontier' and stage in allocation['stages']
    assert not (OUT/'closed.json').exists()
    write(OUT/'attempts'/f'{stage}.json', dict(identity=art(OUT/'identity.json'), allocation=art(OUT/'allocation.json')))
    return identity


def libraries():
    global np, pd, torch, TabICLRegressor
    import numpy as np
    import pandas as pd
    import torch
    from tabicl import TabICLRegressor
    e = read(OUT/'environment.json')
    assert str(Path(sys.prefix).resolve()) == e['isolated_prefix']
    for name, version in e['packages'].items(): assert importlib.metadata.version(name) == version
    assert str(Path(__import__('tabicl').__file__).resolve()).startswith(e['isolated_prefix']+'/')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.backends.cuda.enable_cudnn_sdp(False)
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()


def flags():
    return dict(torch=torch.__version__, device=torch.cuda.get_device_name(0),
        float32_precision=torch.get_float32_matmul_precision(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(), math_sdp=torch.backends.cuda.math_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(), threads=torch.get_num_threads())


def resolved_config(model):
    out = {}
    for key in ['COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG']:
        cfg = dict(getattr(model.inference_config_, key).items())
        assert cfg['use_amp'] is False and cfg['use_fa3'] is False
        out[key] = {k:(str(v) if isinstance(v, (torch.device, torch.dtype)) else v) for k,v in cfg.items()}
    return out


def budget(started, seconds):
    torch.cuda.synchronize()
    gpu = torch.cuda.max_memory_allocated()/(1 << 20)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    assert gpu <= 9216 and rss <= 8192, ('resource_cap', gpu, rss)
    assert time.monotonic()-started <= seconds, 'fixed time budget exhausted'
    return dict(seconds=time.monotonic()-started, GPU_allocated_MiB=gpu, process_RSS_MiB=rss)


def project(f, plan):
    x = f[plan['inputs']['features']].copy()
    for col in plan['inputs']['categorical']:
        x[col] = x[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    for col in plan['inputs']['numerical']: x[col] = x[col].astype(np.float64)
    return x


def inputs(identity):
    allowed = {a['path']: a for a in identity['inputs']}
    paths = [OLD/'inputs'/s for s in ['train_features.parquet', 'query_features.parquet', 'train_y.npy']]
    for p in paths: verify(allowed[str(p.relative_to(ROOT))])
    plan = read(PLAN); cols = ['mvt_id']+plan['inputs']['features']
    tr, qu = [pd.read_parquet(p, columns=cols) for p in paths[:2]]
    y = np.load(paths[2], allow_pickle=False)
    assert len(tr) == len(y) and np.isfinite(y).all()
    assert tr.mvt_id.is_unique and qu.mvt_id.is_unique
    assert set(tr.airport) == set(qu.airport)
    return plan, tr, qu, y


def estimator():
    p = read(PLAN)['model']
    keys = ['n_estimators','norm_methods','feat_shuffle_method','outlier_threshold','random_state',
            'batch_size','kv_cache','allow_auto_download','checkpoint_version','device','use_amp',
            'use_fa3','offload_mode','disk_offload_dir','n_jobs','verbose']
    return TabICLRegressor(**{k:p[k] for k in keys}, model_path=verify(read(OUT/'identity.json')['checkpoint']))


def fit_context(airport, tr, y, plan):
    ix = tr.airport.eq(airport).to_numpy(); f = tr.loc[ix]
    assert 0 < len(f) <= 4096
    model = estimator().fit(project(f, plan), y[ix])
    assert model.model_config_['num_quantiles'] == 999 and model.model_config_['max_classes'] == 0
    folder = OUT/'contexts'/airport; folder.mkdir(parents=True, exist_ok=False)
    # Preserve every fitted preprocessing object. Learned checkpoint remains a separate immutable artifact.
    state = {k:v for k,v in model.__dict__.items() if k not in ['model_', 'model_kv_cache_']}
    with (folder/'context.pkl').open('xb') as fp: pickle.dump(state, fp, protocol=5)
    write(folder/'fit.json', dict(identity=art(OUT/'identity.json'), airport=airport,
        context=art(folder/'context.pkl'), train_ids=f.mvt_id.astype(int).tolist(), train_rows=len(f),
        checkpoint=read(OUT/'identity.json')['checkpoint'], model_config=model.model_config_,
        native_target_mean=float(model.y_scaler_.mean_[0]), native_target_scale=float(model.y_scaler_.scale_[0]),
        resolved_inference_config=resolved_config(model), flags=flags(), no_gradient_fit=True))
    return model


def load_context(airport):
    folder = OUT/'contexts'/airport; receipt = read(folder/'fit.json')
    assert receipt['identity'] == art(OUT/'identity.json')
    with verify(receipt['context']).open('rb') as fp: state = pickle.load(fp)
    model = estimator(); model.__dict__.update(state)
    model._load_model(); model.model_.to(model.device_)
    model.model_kv_cache_ = None; model._build_kv_cache()
    return model


def prediction(model, x):
    out = model.predict(x, output_type=['mean', 'raw_quantiles'])
    assert out['raw_quantiles'].shape == (len(x), 999)
    assert np.isfinite(out['mean']).all() and np.isfinite(out['raw_quantiles']).all()
    return out


def preflight():
    identity = start('preflight'); libraries(); begun = time.monotonic()
    plan, tr, qu, y = inputs(identity)
    airport = sorted(tr.airport.unique(), key=lambda a:(-int(tr.airport.eq(a).sum()), a))[0]
    query = qu.loc[qu.airport.eq(airport)]
    order = sorted(range(len(query)), key=lambda j:hashlib.sha256(f'tabicl_v2_preflight|{int(query.iloc[j].mvt_id)}'.encode()).digest())
    probe = query.iloc[order[:32]]; x = project(probe, plan)
    poisoned = probe.copy(); poisoned['TAXITIME_SEC_mvt'] = np.arange(len(probe))*1e20
    pd.testing.assert_frame_equal(project(poisoned, plan), x)
    model = fit_context(airport, tr, y, plan); budget(begun, 600)
    base = prediction(model, x); outputs = {'base':base}
    outputs['repeat'] = prediction(model, x); budget(begun, 600)
    for name, size in [('single', 1), ('chunk7', 7)]:
        chunks = [prediction(model, x.iloc[j:j+size]) for j in range(0,len(x),size)]
        outputs[name] = {k:np.concatenate([c[k] for c in chunks]) for k in base}; budget(begun,600)
    reverse = prediction(model, x.iloc[::-1]); outputs['reverse'] = {k:v[::-1] for k,v in reverse.items()}
    cache = model.model_kv_cache_; model.model_kv_cache_ = None
    try: outputs['uncached'] = prediction(model, x)
    finally: model.model_kv_cache_ = cache
    folder=OUT/'preflight'; folder.mkdir()
    np.savez(folder/'parity.npz', mvt_id=probe.mvt_id.to_numpy(),
        **{name+'_'+k:v for name,o in outputs.items() for k,v in o.items()})
    metrics = {}; scale = float(model.y_scaler_.scale_[0]); center=float(model.y_scaler_.mean_[0])
    for name,out in outputs.items():
        d = out['mean'].astype(float)-base['mean'].astype(float)
        rmse = float(np.sqrt(np.mean(d*d))); maximum = float(np.max(np.abs(d)))
        assert rmse <= .01 and maximum <= .1, (name, rmse, maximum)
        np.testing.assert_allclose((out['raw_quantiles'].astype(float)-center)/scale,
                                   (base['raw_quantiles'].astype(float)-center)/scale, atol=5e-5, rtol=5e-6)
        metrics[name] = dict(RMS=rmse, maximum=maximum)
    torch.cuda.synchronize(); tick=time.monotonic()
    capacity = prediction(model, project(query.iloc[order[:256]], plan))
    torch.cuda.synchronize(); duration=time.monotonic()-tick
    np.savez(folder/'capacity.npz',mvt_id=query.iloc[order[:256]].mvt_id.to_numpy(),**capacity)
    write(OUT/'preflight.json', dict(status='pass', identity=art(OUT/'identity.json'),
        context=art(OUT/'contexts'/airport/'fit.json'), parity=art(folder/'parity.npz'),
        airport=airport, native_target_scale=scale, native_target_mean=center,
        comparisons=metrics, capacity=art(folder/'capacity.npz'), capacity_rows=len(capacity['mean']),
        capacity_seconds=duration, rough_query_seconds=duration*len(qu)/len(capacity['mean']),
        runtime=flags(), resources=budget(begun,600), poisoned_target_projection_unchanged=True,
        scope='Numerical/resource proof only; no April labels or outcome-based model choice'))


def infer():
    identity = start('infer'); libraries(); begun=time.monotonic()
    audit=read(OUT/'preflight_audit.json')
    assert audit['status']=='pass' and audit['identity']==art(OUT/'identity.json')
    assert audit['preflight']==art(OUT/'preflight.json')
    plan,tr,qu,y=inputs(identity); saved=[]
    for airport in sorted(tr.airport.unique()):
        context=OUT/'contexts'/airport/'fit.json'
        model=load_context(airport) if context.exists() else fit_context(airport,tr,y,plan)
        group=qu.loc[qu.airport.eq(airport)]; folder=OUT/'predictions'/airport;folder.mkdir(parents=True)
        for j in range(0,len(group),256):
            f=group.iloc[j:j+256]; raw=model.predict(project(f,plan), output_type='mean').astype(float)
            assert raw.shape==(len(f),) and np.isfinite(raw).all()
            p=folder/f'{j:08d}.parquet';assert not p.exists()
            pd.DataFrame(dict(mvt_id=f.mvt_id.to_numpy(),raw=raw,serialized=np.maximum(0,np.rint(raw)))).to_parquet(p,index=False)
            saved.append(art(p));budget(begun,1800)
        del model;gc.collect();torch.cuda.empty_cache()
    f=pd.concat([pd.read_parquet(verify(a)) for a in saved],ignore_index=True).set_index('mvt_id')
    assert f.index.is_unique and len(f)==len(qu)
    f=f.loc[qu.mvt_id].reset_index();p=OUT/'predictions.parquet';f.to_parquet(p,index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),prediction=art(p),parts=saved,
        contexts=[art(p) for p in sorted((OUT/'contexts').glob('*/fit.json'))],
        preflight_audit=art(OUT/'preflight_audit.json'),rows=len(f),runtime=flags(),
        resources=budget(begun,1800),no_April_target_read=True))


def score():
    start('score');import numpy as np;import pandas as pd
    audit=read(OUT/'inference_audit.json');assert audit['status']=='pass'
    assert audit['identity']==art(OUT/'identity.json') and audit['inference']==art(OUT/'inference.json')
    oldaudit=read(OLD/'result_audit.json');assert oldaudit['status']=='pass'
    assert oldaudit['identity']==art(OLD/'identity.json') and oldaudit['result']==art(OLD/'result.json')
    old=read(OLD/'result.json'); refs={Path(a['path']).name:a for a in old['artifacts']}
    f=pd.read_parquet(verify(refs['scored.parquet']))
    pred=pd.read_parquet(verify(read(OUT/'inference.json')['prediction']))
    np.testing.assert_array_equal(f.mvt_id,pred.mvt_id);f['tabicl']=pred.serialized.to_numpy()
    dest=OUT/'scored.parquet';f.to_parquet(dest,index=False)
    eligible=f.loc[f.finite_target];arms=ARMS+['tabicl']
    sq=np.column_stack([(eligible[s].to_numpy()-eligible.y.to_numpy())**2 for s in arms])
    days,inverse=np.unique(eligible.day,return_inverse=True);n=np.bincount(inverse)
    sums=np.column_stack([np.bincount(inverse,weights=sq[:,i]) for i in range(3)])
    draws=np.random.Generator(np.random.PCG64(20261013)).integers(0,len(days),(2000,len(days)))
    boot=np.sqrt(sums[draws].sum(1)/n[draws].sum(1)[:,None]);delta=boot[:,2,None]-boot[:,:2]
    np.save(OUT/'bootstrap.npy',delta);rmse=np.sqrt(sq.mean(0));ci=np.quantile(delta,[.025,.975],axis=0)
    comparisons={a:dict(delta=float(rmse[2]-rmse[i]),CI95=ci[:,i].tolist()) for i,a in enumerate(ARMS)}
    gate=bool(all(v['delta']<=-1 and v['CI95'][1]<0 for v in comparisons.values()))
    write(OUT/'result.json',dict(identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),
        inference_audit=art(OUT/'inference_audit.json'),control_result=art(OLD/'result.json'),
        control_audit=art(OLD/'result_audit.json'),scored=art(dest),bootstrap=art(OUT/'bootstrap.npy'),
        rows=len(f),scored_rows=len(eligible),RMSE=dict(zip(arms,rmse.tolist())),comparisons=comparisons,
        gate=gate,scope='April mechanism evidence only; genuine Jan/Jul current-parent gate still required'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','preflight','infer','score'])
    args=parser.parse_args()
    try:globals()[args.stage]()
    except Exception as exc:
        if args.stage!='freeze' and (OUT/'attempts'/f'{args.stage}.json').exists() and not (OUT/'closed.json').exists():
            write(OUT/'closed.json',dict(stage=args.stage,status='failed_preserved_no_retry',error=repr(exc)))
        raise
