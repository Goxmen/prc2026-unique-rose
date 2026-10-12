"""Retained v60/v61 weather models on separately audited final inputs.

No model is fitted. Missing encoder state is reconstructed once from the exact
retained 2025 matrix, compared with all checkpoint bins/cards/scales, and saved.
Every old component and both original integer vectors must replay exactly.
"""
import argparse
import gc
import os
from pathlib import Path
import pickle
import tempfile

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits

import open_meteo_submit as original
import final_weather_features as weather
import callsign_submit as parent
import chunked_direct_data as adapter
import nn_improve_jan_jul as nni
import nn_jan_jul as nnj
import modeling
import catboost_jan_jul as cbj
from records import write_json

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_weather_inference'
MANIFEST=OUT/'identity.json'
read,art,verify=weather.read,weather.art,weather.verify
GROUPS=[42,142,242]
PARTS=['rome','lgb','airport','catboost','nn']
DESIGN=dict(versions=[60,61],views=['old','final'],models=28,neural_members=15,
    inputs='Retained original156 plus audited11 ERA5 inputs; exact saved model column order',
    preprocessing='Reconstruct unchanged Encoder using exact original2025 matrix only; saved checkpoint bins/cards/mu/sd exact',
    inference='Retained original PLE65536-row chunks on desktopRTX3080; full old component equality and final model reloads',
    composition='Original v60 Rome-only and v61 allfive recipes, original weights/rules/integer conversion',
    provenance='v60 final retains October8 parent base-neural reconstruction provenance',
    no_model_fit=True,no_new_selection=True,no_upload=True)


def model_path(name):return original.WORK/'models'/name


def freeze():
    assert not MANIFEST.exists()
    w=weather.guard();assert read(weather.OUT/'audit.json')['status']=='pass'
    prepared=read(original.WORK/'prepared.json')
    assert art(original.INPUTS)['sha256']==prepared['inputs_sha256']
    assert art(original.WORK/'rome.pkl')['sha256']==prepared['rome_sha256']
    paths=[Path(__file__),Path(original.__file__),Path(weather.__file__),Path(parent.__file__),
        Path(parent.fs.__file__),Path(adapter.__file__),Path(nni.__file__),Path(nnj.__file__),
        Path(original.hs.__file__),Path(original.nis.__file__),Path(original.rs.signals.__file__),Path(modeling.__file__),
        weather.MANIFEST,weather.OUT/'audit.json',weather.OUT/'features.json',
        verify(read(weather.OUT/'features.json')['output']),original.INPUTS,original.WORK/'rome.pkl',
        original.WORK/'prepared.json',original.OUT/'full_year_audit.json',original.OUT/'design.json']
    paths += [verify(r) for r in w['original']+w['additional']]
    correction=original.nis.v43s.v42.parent
    helpers=[original.wx,original.rs,original.rs.rome_hybrid,original.hs.hpo,cbj,
        correction,correction.corrections,correction.plan_window,correction.record_fixes]
    paths.extend(Path(module.__file__) for module in helpers)
    paths.append(verify(read(weather.OUT/'audit.json')['source']))
    models={};parts={};versions={}
    for path in sorted((original.WORK/'models').glob('*')):
        if path.suffix in ['.txt','.pt','.cbm']:models[path.name]=art(path);paths.append(path)
    assert len(models)==28 and sum(p.endswith('.pt') for p in models)==15
    for part in PARTS:
        p=original.WORK/'parts'/f'{part}.parquet';receipt=p.with_suffix('.json')
        assert art(p)['sha256']==read(receipt)['sha256']
        parts[part]=art(p);paths.extend([p,receipt])
    for g in GROUPS:
        paths.append(original.WORK/'parts'/f'nn_group_{g}.json')
        for k in range(5):paths.append(original.WORK/'parts'/f'nn_{g}_{k}.parquet')
    for v in [60,61]:
        receipt=original.OUT/f'submission_v{v}/submission_v{v}.json';record=read(receipt)
        p=ROOT/record['file'];assert art(p)['sha256']==record['sha256']==record['remote_sha256']
        versions[str(v)]=art(p);paths.extend([receipt,p])
    for view in DESIGN['views']:
        paths += [parent.fs.WORK/f'inputs_{view}.pkl',parent.fs.OUT/f'inputs_{view}.json',
            ROOT/'data'/('submitting.parquet' if view=='old' else 'final_submitting.parquet')]
    for month in ['2026-01','2026-07']:paths.append(original.V58/f'ranking_{month}.parquet')
    for part in ['nn','trees','rome']:
        paths.extend([parent.fs.WORK/f'parts/{part}.parquet',parent.fs.WORK/f'parts/{part}.json'])
    cfg=read(original.WORK/'parts/nn.json')['config']
    assert cfg==original.hs.hpo.nn_config(original.hs.searched()['nn']['overrides'])
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(MANIFEST,dict(design=DESIGN,source=art(__file__),models=models,old_parts=parts,
        original_versions=versions,config=cfg,columns=read(original.WORK/'parts/nn.json')['columns'],
        sources=[art(p) for p in sorted(set(paths))],torch=str(torch.__version__),
        execution=dict(host='GPC',gpu='NVIDIA GeForce RTX 3080'),
        historical_torch_version_not_recorded=True))
    print(dict(frozen=True,models=len(models),no_model_fit=True),flush=True)


def guard():
    m=read(MANIFEST);assert m['design']==DESIGN and m['source']==art(__file__)
    for a in m['sources']:verify(a)
    original.hs.hpo.apply_nn(m['config'],nnj,nni)
    assert nni.NEW['bins']==48 and str(torch.__version__)==m['torch']
    return m


def ranking(view):
    d=parent.ranking_data(view);names=d['names']+original.COMBO
    assert len(names)==156
    if view=='old':
        raw=pd.concat([x['raw'] for x in d['months'].values()],ignore_index=True)
        np.testing.assert_array_equal(raw.mvt_id,d['rank_ids'])
        extra=original.wx.weather_join(raw,original.wx.weather_table())
    else:
        extra=pd.read_parquet(verify(read(weather.OUT/'features.json')['output'])).set_index('mvt_id')
        assert extra.index.is_unique
        extra=extra.loc[d['rank_ids'],original.wx.WEATHER]
    d['Vn']=pd.concat([d['Vn'][names],extra.set_axis(d['Vn'].index)],axis=1)
    assert list(d['Vn'])==read(MANIFEST)['columns'] and np.isfinite(extra.to_numpy()).all()
    return d


def checkpoint_equal(checkpoint,pre):
    assert checkpoint['cards']==pre['cards'] and checkpoint['mu']==pre['mu'] and checkpoint['sd']==pre['sd']
    assert len(checkpoint['bins'])==len(pre['bins'])
    for a,b in zip(checkpoint['bins'],pre['bins']):torch.testing.assert_close(a.cpu(),b.cpu(),atol=0,rtol=0)


def prepare():
    m=guard();receipt=OUT/'prepared.json'
    if receipt.exists():
        r=read(receipt);assert r['identity']==art(MANIFEST);verify(r['preprocessing']);return
    assert not (OUT/'preprocessing.pkl').exists()
    d=pd.read_pickle(original.INPUTS);cols=m['columns']
    assert list(d['Xn'])==cols and list(d['Vn'])==cols
    old=ranking('old');np.testing.assert_array_equal(old['rank_ids'],d['rank_ids'])
    pd.testing.assert_frame_equal(old['Vn'],d['Vn'],check_exact=True)
    assert old['parameters']==d['parameters'] and old['fitted']==d['fitted']
    del old;gc.collect()
    encoder=nnj.Encoder(d['Xn'][cols])
    with tempfile.TemporaryDirectory(prefix='weather_encoder_',dir=OUT) as tmp:
        cat,num=adapter.encode(encoder,d['Xn'][cols],tmp,'train')
        bins,ple,plain=adapter.layout(encoder,num)
        cat._mmap.close();num._mmap.close()
    clipped=np.minimum(d['yn'],nnj.RECIPE['label_clip'])
    pre=dict(encoder=encoder,bins=bins,ple=ple,plain=plain,cards=encoder.cards,
        mu=float(clipped.mean()),sd=float(clipped.std()),columns=cols,config=m['config'])
    for g in GROUPS:
        for k in range(5):
            checkpoint=torch.load(verify(m['models'][f'nn_{g}_{k}.pt']),map_location='cpu',weights_only=False)
            checkpoint_equal(checkpoint,pre)
    path=OUT/'preprocessing.pkl'
    with path.open('wb') as f:pickle.dump(pre,f,protocol=pickle.HIGHEST_PROTOCOL)
    write_json(receipt,dict(identity=art(MANIFEST),preprocessing=art(path),training_rows=len(d['Xn']),
        original167_input_values_exact=True,old_rows=len(d['Vn']),checkpoint_bins_cards_scales_exact=15,
        preprocessing_reconstructed_from_training_only=True,no_model_fit=True))
    print(dict(prepared=True,training_rows=len(d['Xn']),old_rows=len(d['Vn']),checkpoints_exact=15),flush=True)


def trees_predict(m,d,part):
    V=d['Vn'];ids=d['rank_ids']
    if part=='catboost':
        from catboost import CatBoostRegressor,Pool
        cols=d['names']+original.wx.WEATHER;cats=original.rs.signals._categorical(cols)
        model=CatBoostRegressor().load_model(str(verify(m['models']['catboost.cbm'])))
        assert model.feature_names_==cols
        p=np.maximum(model.predict(Pool(cbj.frame(V[cols],cats),cat_features=cats),thread_count=4),0)
    elif part=='rome':
        model=lgb.Booster(model_file=str(verify(m['models']['rome.txt'])));parts=[];offset=0
        assert model.current_iteration()==1700
        for item in d['months'].values():
            n=len(item['raw']);use=item['raw'].ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(item['V'][original.rs.rome_hybrid.GAP].to_numpy(float))
            B=original.rs.signals.classifier_matrix(item['V'].loc[use],item['raw'].loc[use],original.rs.KIND,original.ts.CLASSIFIER)
            extra=V.iloc[offset:offset+n].loc[use,original.wx.WEATHER]
            B=pd.concat([B,extra.set_axis(B.index)],axis=1)
            assert list(B)==model.feature_name()
            pred=np.zeros(n);pred[use]=model.predict(B,num_threads=4);parts.append(pred);offset+=n
        assert offset==len(ids);p=np.concatenate(parts)
        return pd.DataFrame(dict(mvt_id=ids,pred=p))
    elif part=='lgb':
        model=lgb.Booster(model_file=str(verify(m['models']['lgb.txt'])))
        assert model.feature_name()==list(V) and model.current_iteration()==10000
        p=modeling.predict(model,V,threads=4)
    else:
        assert part=='airport';p=np.full(len(ids),np.nan);airport=V.airport.astype(str).to_numpy()
        for ap in sorted(set(airport)):
            model=lgb.Booster(model_file=str(verify(m['models'][f'airport_{ap}.txt'])))
            assert model.feature_name()==list(V) and model.current_iteration()==5000
            p[airport==ap]=modeling.predict(model,V.loc[airport==ap],threads=4)
    return pd.DataFrame(dict(mvt_id=ids,pred=np.asarray(p,np.float32)))


def neural_predict(m,d,view):
    assert dict(host=os.uname().nodename,gpu=torch.cuda.get_device_name())==m['execution']
    record=read(OUT/'prepared.json');assert record['identity']==art(MANIFEST)
    with verify(record['preprocessing']).open('rb') as f:pre=pickle.load(f)
    with tempfile.TemporaryDirectory(prefix='weather_infer_',dir=OUT) as tmp:
        cat,num=adapter.encode(pre['encoder'],d['Vn'],tmp,'view')
        data=(adapter.tensor_columns(cat,None,'cuda'),adapter.tensor_columns(num,pre['ple'],'cuda'),
            adapter.tensor_columns(num,pre['plain'],'cuda'))
        torch.cuda.synchronize();cat._mmap.close();num._mmap.close()
    result=pd.DataFrame(dict(mvt_id=d['rank_ids']))
    for g in GROUPS:
        for k in range(5):
            path=verify(m['models'][f'nn_{g}_{k}.pt'])
            checkpoint=torch.load(path,map_location='cuda',weights_only=False);checkpoint_equal(checkpoint,pre)
            model=nni.PleNet(pre['cards'],pre['bins'],len(pre['plain'])).cuda()
            model.load_state_dict(checkpoint['state_dict'])
            value=nni.predict(model,data,pre['sd'],pre['mu'],65536).astype(np.float32)
            if view=='final':
                restored=nni.PleNet(pre['cards'],pre['bins'],len(pre['plain'])).cuda()
                restored.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['state_dict'])
                np.testing.assert_array_equal(value,nni.predict(restored,data,pre['sd'],pre['mu'],65536).astype(np.float32))
                del restored
            result[f'n{g}_{k}']=value
            del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    del data;torch.cuda.empty_cache()
    return result


def infer(part):
    m=guard();assert part in PARTS
    for view in DESIGN['views']:
        directory=OUT/view;directory.mkdir(exist_ok=True);receipt=directory/f'{part}.json'
        if receipt.exists():
            r=read(receipt);assert r['identity']==art(MANIFEST);verify(r['prediction']);continue
        if view=='final':assert read(OUT/f'old/{part}.json')['all_original_values_exact']
        d=ranking(view)
        if part=='nn':f=neural_predict(m,d,view)
        else:
            f=trees_predict(m,d,part)
            if view=='final':pd.testing.assert_frame_equal(f,trees_predict(m,d,part),check_exact=True)
        assert f.mvt_id.is_unique and np.isfinite(f.drop(columns='mvt_id').to_numpy()).all()
        if view=='old':pd.testing.assert_frame_equal(f,pd.read_parquet(verify(m['old_parts'][part])),check_exact=True)
        p=directory/f'{part}.parquet'
        if p.exists():pd.testing.assert_frame_equal(f,pd.read_parquet(p),check_exact=True)
        else:f.to_parquet(p,index=False)
        write_json(receipt,dict(identity=art(MANIFEST),prediction=art(p),view=view,part=part,rows=len(f),
            all_original_values_exact=True,all_final_reloads_exact=view=='final',no_model_fit=True))
        del d,f;gc.collect();torch.cuda.empty_cache()
        print(dict(view=view,part=part,pass_all_rows=True),flush=True)


def package():
    m=guard()
    for view in DESIGN['views']:
        d=parent.ranking_data(view);ids=d['rank_ids'];parts={}
        for part in PARTS:
            record=read(OUT/view/f'{part}.json');assert record['identity']==art(MANIFEST)
            f=pd.read_parquet(verify(record['prediction']));np.testing.assert_array_equal(f.mvt_id,ids)
            parts[part]=f.drop(columns='mvt_id').to_numpy(float).mean(axis=1)
        if view=='final':neural,tree,_,_=parent.parent_parts(view,d)
        template=pq.read_table(ROOT/'data'/('submitting.parquet' if view=='old' else 'final_submitting.parquet'))
        for v in DESIGN['versions']:
            receipt=OUT/f'{view}_v{v}.json'
            if receipt.exists():
                r=read(receipt);assert r['identity']==art(MANIFEST);verify(r['prediction']);continue
            if view=='final':assert read(OUT/f'old_v{v}.json')['all_original_predictions_exact']
            frames=[];offset=0
            for month,item in d['months'].items():
                n=len(item['raw']);sl=slice(offset,offset+n)
                if v==60:
                    if view=='old':
                        reference=pd.read_parquet(original.V58/f'ranking_{month}.parquet')
                        np.testing.assert_array_equal(reference.mvt_id,ids[sl]);base=reference.blend_base.to_numpy(float)
                    else:base=.71*neural[sl]+(1-.71)*tree[sl]
                else:base=.71*parts['nn'][sl]+.29*np.mean([parts[p][sl] for p in original.hs.GBM],axis=0)
                f,_=original.nis.final(base,parts['rome'][sl],item['V'],item['raw'],d['parameters'],d['fitted'])
                f['month']=month;frames.append(f[['mvt_id','month','pred']]);offset+=n
            frame=pd.concat(frames,ignore_index=True);np.testing.assert_array_equal(frame.mvt_id,ids)
            serialized=original.hs.fill_template(template,ids,frame.pred.to_numpy())
            frame=frame.set_index('mvt_id').loc[template['MVT_ID_mvt'].to_numpy()].reset_index()
            frame['submitted']=serialized['TAXITIME_SEC_mvt'].to_numpy()
            if view=='old':assert serialized.equals(pq.read_table(verify(m['original_versions'][str(v)])))
            p=OUT/f'{view}_v{v}.parquet';frame.to_parquet(p,index=False)
            write_json(receipt,dict(identity=art(MANIFEST),prediction=art(p),version=v,view=view,rows=len(frame),
                all_original_predictions_exact=True,all_original_component_values_exact=True,
                final_template_order_exact=True,parent_neural_provenance=DESIGN['provenance'],no_model_fit=True))
            print(dict(view=view,version=v,rows=len(frame),exact_old_gate=True),flush=True)
        del d,parts;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','prepare','infer','package'])
    p.add_argument('--part',choices=PARTS);a=p.parse_args();torch.set_num_threads(4)
    with threadpool_limits(4):
        if a.stage=='infer':infer(a.part)
        else:globals()[a.stage]()
