"""Replay saved public main/airport trees and Rome classifiers on final inputs.

No fitting or new selection. Each component must reproduce all old-view
predictions exactly before its final view is computed. Final outputs are
independently reloaded from saved models and checked on every row.
"""
import argparse
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_delay_lgb_submit as main
import public_airport_submit as airport
import rome_public_delay_submit as rome
import final_public_features as public
from records import write_json

ROOT=main.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_public_tree_inference'
MANIFEST=OUT/'identity.json'
read,art,verify=main.read,main.artifact,main.verify
DESIGN=dict(views=['old','final'],parts=['main','airport','rome_cat'],
    preprocessing='Unchanged saved156-input packets; cached callsign features where selected; audited twelve public inputs',
    controls='Every original-view saved prediction must replay exactly before final inference',
    no_training=True,no_new_selection=True,no_version=True,no_upload=True)


def prepare():
    assert not MANIFEST.exists()
    mr=main.guard(); ar=airport.guard(); rome.guard()
    original=main.parent.validate_prepared()
    rp=rome.parent.validate_prepared()
    assert read(public.RECORDS/'audit.json')['status']=='pass'
    models={'main':read(main.OUT/'model.json'),'airport':{},'rome_cat':{}}
    for ap in ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']:
        models['airport'][ap]=airport.completed(ap)
        assert models['airport'][ap] is not None
    for seed in rome.parent.SEEDS:
        models['rome_cat'][str(seed)]=read(rome.OUT/f'predictions/cat_s{seed}.json')
    paths=[Path(__file__),Path(main.__file__),Path(airport.__file__),Path(rome.__file__),Path(public.__file__),
        main.OUT/'identity.json',main.OUT/'model.json',airport.MANIFEST,rome.OUT/'identity.json',rome.OUT/'prepared.json',
        main.parent.OUT/'prepared.json',rome.parent.OUT/'prepared.json',
        public.OUT/'features.parquet',public.OUT/'inputs.parquet',public.OUT/'manifest.json',public.RECORDS/'audit.json',
        ROOT/'data/submitting.parquet',ROOT/'data/final_submitting.parquet',
        verify(mr['ranking_features']),main.cv.features.base.OUT/'ranking_inputs.parquet']
    for view in DESIGN['views']:
        _,provenance=main.old.ranking_inputs(view,original)
        paths.append(verify(provenance['manifest']))
        paths.append(verify(rp['views'][view]['packet']))
        _,recipe=main.parent.features.load_features(view)
        paths.append(main.parent.features.DEFAULT/f'{view}.parquet')
    for rec in [models['main'],*models['airport'].values(),*models['rome_cat'].values()]:
        paths.append(verify(rec['model']))
    old_predictions={}
    for name,folder in [('main',main.OUT),('airport',airport.OUT)]:
        old_predictions[name]=read(folder/'predictions/old.json')['prediction']
        paths.append(verify(old_predictions[name]))
    for name,rec in models['rome_cat'].items():paths.append(verify(rec['prediction']))
    paths.extend(verify(read(public.OUT/'manifest.json')[k]) for k in ['output','raw'])
    for record in read(public.OUT/'manifest.json')['sources']:paths.append(verify(record))
    value=dict(design=DESIGN,main_recipe=mr,airport_recipe=ar,original=original,rome_original=rp,
        rome_public=read(rome.OUT/'prepared.json'),models=models,old_predictions=old_predictions,
        sources=[art(p) for p in sorted(set(paths))])
    OUT.mkdir(parents=True,exist_ok=True);write_json(MANIFEST,value)
    print(dict(prepared=True,models=14,final_rows=670790),flush=True)


def public_values(view,ids,index):
    path=main.public_rome.OUT/'ranking_features.parquet' if view=='old' else public.OUT/'features.parquet'
    f=pd.read_parquet(path).set_index('mvt_id')
    assert f.index.is_unique
    return f.loc[ids,main.cv.features.FEATURES].set_axis(index)


def ids_check(view,ids):
    name='submitting.parquet' if view=='old' else 'final_submitting.parquet'
    expected=pd.read_parquet(ROOT/'data'/name,columns=['MVT_ID_mvt']).MVT_ID_mvt
    # Final cached packets and the submission template have different month order.
    np.testing.assert_array_equal(ids if view=='old' else np.sort(ids),
        expected if view=='old' else np.sort(expected))
    assert len(ids)==len(set(ids))==(344841 if view=='old' else 670790)


def tree_frame(m,view,part):
    packet,_=main.old.ranking_inputs(view,m['original'])
    ids=np.asarray(packet['rank_ids']);ids_check(view,ids)
    base=packet['Vn'][m['original']['original_columns']]
    if part=='main':
        clocks,_=main.parent.features.load_features(view)
        V=main.cv.parent.clocks.add_features(base,ids,clocks)
    else:V=base
    V=pd.concat([V,public_values(view,ids,V.index)],axis=1)
    recipe=m[f'{part}_recipe']
    assert list(V)==recipe['columns']
    pd.testing.assert_frame_equal(V[m['original']['original_columns']],base,check_exact=True)
    rawpath=main.cv.features.base.OUT/'ranking_inputs.parquet' if view=='old' else public.OUT/'inputs.parquet'
    raw=pd.read_parquet(rawpath,columns=['mvt_id','airport']).set_index('mvt_id')
    aps=raw.loc[ids,'airport'].to_numpy()
    assert sorted(set(aps))==sorted(m['models']['airport'])
    return ids,V,aps


def tree_predict(m,V,aps,part,threads):
    records={'all':m['models']['main']} if part=='main' else m['models']['airport']
    values=np.full(len(V),np.nan,dtype=np.float32)
    for key,rec in records.items():
        use=np.ones(len(V),bool) if key=='all' else aps==key
        model=lgb.Booster(model_file=str(verify(rec['model'])))
        assert model.feature_name()==list(V) and model.current_iteration()==rec['rounds']
        values[use]=np.maximum(model.predict(V.loc[use],num_threads=threads),0).astype(np.float32)
        del model;gc.collect()
    assert np.isfinite(values).all() and (values>=0).all()
    return values


def infer(part):
    m=read(MANIFEST);assert m['design']==DESIGN
    for a in m['sources']:verify(a)
    out=OUT/part;out.mkdir(exist_ok=True)
    for view in DESIGN['views']:
        receipt=out/f'{view}.json'
        if receipt.exists():
            previous=read(receipt);assert previous['identity']==art(MANIFEST)
            for a in previous.get('predictions',[]):verify(a)
            continue
        if view=='final':assert read(out/'old.json')['old_replay_exact']
        outputs=[]
        if part in ['main','airport']:
            ids,V,aps=tree_frame(m,view,part)
            pred=tree_predict(m,V,aps,part,4)
            if view=='old':
                prior=pd.read_parquet(verify(m['old_predictions'][part]))
                np.testing.assert_array_equal(ids,prior.mvt_id)
                np.testing.assert_array_equal(pred,prior.pred)
            else:
                np.testing.assert_array_equal(pred,tree_predict(m,V,aps,part,2))
                path=out/'final.parquet';pd.DataFrame(dict(mvt_id=ids,pred=pred)).to_parquet(path,index=False)
                outputs.append(art(path))
            del V,aps,pred;gc.collect()
        else:
            from catboost import CatBoostClassifier,Pool
            packet=pd.read_pickle(verify(m['rome_original']['views'][view]['packet']))
            ids=np.asarray(packet['ids']);ids_check(view,ids)
            V=pd.concat([packet['X'],public_values(view,ids[packet['use']],packet['X'].index)],axis=1)
            assert list(V)==m['rome_public']['columns']
            if view=='old':
                prior_packet=pd.read_pickle(verify(m['rome_public']['old']))
                pd.testing.assert_frame_equal(V,prior_packet['X'],check_exact=True)
                np.testing.assert_array_equal(packet['use'],prior_packet['use'])
                del prior_packet
            _,cats,_=rome.parent.parent_spec()
            X=Pool(rome.parent.cbj.frame(V,cats),cat_features=cats)
            for seed,rec in m['models']['rome_cat'].items():
                model=CatBoostClassifier();model.load_model(str(verify(rec['model'])))
                pred=np.zeros(len(ids));pred[packet['use']]=model.predict_proba(X,thread_count=2)[:,1]
                assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
                if view=='old':
                    prior=pd.read_parquet(verify(rec['prediction']))
                    np.testing.assert_array_equal(ids,prior.mvt_id);np.testing.assert_array_equal(pred,prior.p)
                else:
                    restored=CatBoostClassifier();restored.load_model(str(verify(rec['model'])))
                    np.testing.assert_array_equal(pred[packet['use']],restored.predict_proba(X,thread_count=1)[:,1])
                    path=out/f'final_s{seed}.parquet';pd.DataFrame(dict(mvt_id=ids,p=pred)).to_parquet(path,index=False)
                    outputs.append(art(path));del restored
                del model,pred
            del X,V,packet;gc.collect()
        write_json(receipt,dict(identity=art(MANIFEST),part=part,view=view,rows=len(ids),
            old_replay_exact=True,final_reload_exact=view=='final',predictions=outputs,no_fit=True))
        print(dict(part=part,view=view,rows=len(ids),audit='pass'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','main','airport','rome_cat'])
    with threadpool_limits(4):
        stage=p.parse_args().stage
        if stage=='prepare':prepare()
        else:infer(stage)
