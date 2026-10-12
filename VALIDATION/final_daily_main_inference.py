"""Inference-only final view of the selected 184-input daily main tree.

The 174-input public parent already has a final view; it is not a substitute
for this saved selected model. Replay every old feature and prediction before
using separately jittered four-month inputs. Retain missing public cells.
"""
import argparse
import gc
import types
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_main_submit as selected
import public_daily_taxi_features as daily
import final_public_tree_inference as trees
from records import write_json

ROOT=selected.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_daily_main_inference'
FEATURES=OUT/'features'
MANIFEST=OUT/'identity.json'
read,art,verify=selected.read,selected.art,selected.verify
DESIGN=dict(model='Unchanged selected184-input LB1/10000 seed42',
    inputs='Original156 +6 callsign +12 public delay +10 daily taxi',
    final_rows=670790,old_rows=344841,
    no_fit=True,no_new_selection=True,no_substitution=True,no_upload=True)


def prepare_features():
    assert not (FEATURES/'manifest.json').exists()
    assert read(daily.OUT/'audit.json')['status']=='pass'
    assert read(daily.source.REPORT/'public_daily_taxi_source_audit.json')['status']=='pass'
    source=pd.read_parquet(daily.source.SOURCE/'daily_runway.parquet')
    old=daily.build(daily.load(ROOT/'data/ranking.parquet'),source)
    pd.testing.assert_frame_equal(old,pd.read_parquet(daily.OUT/'featuresranking.parquet'),check_exact=True)
    del old
    raw=daily.load(ROOT/'data/final_ranking.parquet')
    expected=pd.read_parquet(ROOT/'data/final_submitting.parquet',columns=['MVT_ID_mvt']).MVT_ID_mvt
    assert len(raw)==raw.mvt_id.nunique()==len(expected)==DESIGN['final_rows']
    np.testing.assert_array_equal(np.sort(raw.mvt_id),np.sort(expected))
    raw=raw.set_index('mvt_id').loc[expected].reset_index()
    result=daily.build(raw,source)
    assert sorted(raw.takeoff.dt.strftime('%Y-%m').unique())==['2026-01','2026-02','2026-06','2026-07']
    FEATURES.mkdir(parents=True,exist_ok=True)
    raw.to_parquet(FEATURES/'inputsfinal.parquet',index=False)
    result.to_parquet(FEATURES/'featuresfinal.parquet',index=False)
    record=dict(raw=art(ROOT/'data/final_ranking.parquet'),inputs=art(FEATURES/'inputsfinal.parquet'),
        features=art(FEATURES/'featuresfinal.parquet'),rows=len(raw))
    write_json(FEATURES/'manifest.json',dict(source=art(daily.__file__),public=art(daily.source.SOURCE/'daily_runway.parquet'),
        public_manifest=art(daily.source.SOURCE/'manifest.json'),public_audit=art(daily.source.REPORT/'public_daily_taxi_source_audit.json'),
        records=dict(final=record),columns=daily.FEATURES,labels_read=False,old_all_features_exact=True,
        builder=art(__file__),template=art(ROOT/'data/final_submitting.parquet')))
    print(dict(stage='features',rows=len(raw),old_replay_exact=True),flush=True)


def audit_features():
    # Invoke the retained independent SQL equations with only the output path
    # rebound. Bytecode, feature arithmetic and missing-value checks unchanged.
    scope=dict(daily.audit.__globals__,OUT=FEATURES)
    audit=types.FunctionType(daily.audit.__code__,scope,daily.audit.__name__)
    audit()
    m=read(FEATURES/'manifest.json');assert m['builder']==art(__file__)
    with duckdb.connect(config={'threads':1}) as con:
        n,u,bad=con.execute('''SELECT count(*),count(distinct f.mvt_id),sum(
            (f.airport IS DISTINCT FROM r.ADEP_mvt)::INT+
            (f.takeoff IS DISTINCT FROM r.MVT_TIME_UTC_mvt)::INT+
            (f.runway IS DISTINCT FROM r.RUNWAY_mvt)::INT)
            FROM read_parquet(?) f JOIN read_parquet(?) r ON f.mvt_id=r.MVT_ID_mvt
            WHERE r.PHASE_mvt='DEP' ''',[str(FEATURES/'inputsfinal.parquet'),str(ROOT/'data/final_ranking.parquet')]).fetchone()
    assert n==u==DESIGN['final_rows'] and bad==0
    receipt=read(FEATURES/'audit.json')
    receipt.update(raw_input_rows_independently_checked=n,all_old_features_exact=m['old_all_features_exact'],
        builder=art(__file__),retained_audit_bytecode_unchanged=True)
    write_json(FEATURES/'audit.json',receipt)


def prepare():
    assert not MANIFEST.exists()
    assert read(FEATURES/'audit.json')['status']=='pass'
    recipe=selected.validate();model=read(selected.OUT/'model.json')
    assert model['identity']==art(selected.MANIFEST)
    verify(model['model'])
    parent=read(trees.MANIFEST)
    assert parent['design']==trees.DESIGN
    paths=[Path(__file__),Path(selected.__file__),Path(daily.__file__),Path(trees.__file__),
        selected.MANIFEST,selected.OUT/'model.json',selected.OUT/'predictions/old.json',
        selected.OUT/'predictions/old.parquet',FEATURES/'manifest.json',FEATURES/'audit.json',
        FEATURES/'featuresfinal.parquet',FEATURES/'inputsfinal.parquet',trees.MANIFEST]
    paths.extend(verify(a) for a in parent['sources'])
    paths.extend(verify(a) for a in recipe['inputs'].values())
    paths.extend([verify(model['model']),daily.OUT/'manifest.json',daily.OUT/'audit.json'])
    write_json(MANIFEST,dict(design=DESIGN,recipe=recipe,model=model,parent_tree_identity=art(trees.MANIFEST),
        sources=[art(p) for p in sorted(set(paths))]))
    print(dict(stage='prepared',no_fit=True),flush=True)


def infer():
    m=read(MANIFEST);assert m['design']==DESIGN
    for a in m['sources']:verify(a)
    parent=read(verify(m['parent_tree_identity']))
    for view in ['old','final']:
        receipt=OUT/f'{view}.json'
        if receipt.exists():
            prior=read(receipt);assert prior['identity']==art(MANIFEST)
            verify(prior['prediction']);continue
        if view=='final':assert read(OUT/'old.json')['all_original_predictions_exact']
        ids,frame,_=trees.tree_frame(parent,view,'main')
        path=daily.OUT/'featuresranking.parquet' if view=='old' else FEATURES/'featuresfinal.parquet'
        extra=pd.read_parquet(path).set_index('mvt_id')
        assert extra.index.is_unique
        frame=pd.concat([frame,extra.loc[ids,daily.FEATURES].set_axis(frame.index)],axis=1)
        assert list(frame)==m['recipe']['columns'] and len(frame.columns)==184
        model=lgb.Booster(model_file=str(verify(m['model']['model'])))
        assert model.feature_name()==list(frame) and model.current_iteration()==10000
        prediction=np.maximum(model.predict(frame,num_threads=4),0).astype('float32')
        assert np.isfinite(prediction).all()
        restored=lgb.Booster(model_file=str(verify(m['model']['model'])))
        np.testing.assert_array_equal(prediction,np.maximum(restored.predict(frame,num_threads=2),0).astype('float32'))
        if view=='old':
            old=pd.read_parquet(selected.OUT/'predictions/old.parquet')
            np.testing.assert_array_equal(ids,old.mvt_id);np.testing.assert_array_equal(prediction,old.pred)
        target=OUT/f'{view}.parquet'
        pd.DataFrame(dict(mvt_id=ids,pred=prediction)).to_parquet(target,index=False)
        write_json(receipt,dict(identity=art(MANIFEST),model=m['model']['model'],prediction=art(target),
            rows=len(ids),view=view,all_original_predictions_exact=True,all_model_reload_predictions_exact=True,
            final_join_by_id_required=True,no_fit=True))
        del frame,extra,prediction,model,restored;gc.collect()
        print(dict(stage=view,rows=len(ids),pass_all_rows=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare_features','audit_features','prepare','infer'])
    with threadpool_limits(4):globals()[p.parse_args().stage]()
