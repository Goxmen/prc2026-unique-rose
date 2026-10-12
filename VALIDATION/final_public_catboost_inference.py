"""Inference-only final view for the selected 174-input ordinary CatBoost."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
from threadpoolctl import threadpool_limits
import final_public_tree_inference as tree
import catboost_full_inputs_submit as original

ROOT=tree.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_public_catboost_inference'
read,art,verify,write=tree.read,tree.art,tree.verify,tree.write_json


def run():
    recipe=original.guard();model_record=read(original.OUT/'model.json')
    old_record=read(original.OUT/'predictions/old.json')
    assert model_record['identity']==old_record['identity']==art(original.MANIFEST)
    assert model_record['model']==old_record['model']
    prepared=read(tree.MANIFEST)
    assert recipe['columns']==prepared['main_recipe']['columns']
    for a in prepared['sources']:verify(a)
    paths=[Path(__file__),Path(tree.__file__),Path(original.__file__),tree.MANIFEST,
        original.MANIFEST,original.OUT/'model.json',original.OUT/'predictions/old.json',
        verify(model_record['model']),verify(old_record['prediction'])]
    identity=dict(columns=recipe['columns'],categoricals=recipe['categoricals'],
        model=model_record['model'],sources=[art(p) for p in paths],
        shared_inputs=art(tree.MANIFEST),no_fit=True,no_new_selection=True,no_upload=True)
    OUT.mkdir(parents=True,exist_ok=True);manifest=OUT/'identity.json'
    if manifest.exists():assert read(manifest)==identity
    else:write(manifest,identity)
    for view in ['old','final']:
        receipt=OUT/f'{view}.json'
        if receipt.exists():
            r=read(receipt);assert r['identity']==art(manifest)
            for a in r.get('predictions',[]):verify(a)
            continue
        if view=='final':assert read(OUT/'old.json')['old_replay_exact']
        ids,V,_=tree.tree_frame(prepared,view,'main')
        converted=original.cv.cb.frame(V,recipe['categoricals'])
        base=original.cv.cb.frame(V[recipe['original_columns']],original.cv.parent.ideas.signals._categorical(recipe['original_columns']))
        pd.testing.assert_frame_equal(converted[recipe['original_columns']],base,check_exact=True)
        X=Pool(converted,cat_features=recipe['categoricals'])
        model=CatBoostRegressor();model.load_model(str(verify(model_record['model'])))
        assert model.feature_names_==recipe['columns'] and model.tree_count_==10000
        pred=np.maximum(model.predict(X,thread_count=4),0).astype(np.float32)
        assert np.isfinite(pred).all()
        outputs=[]
        if view=='old':
            prior=pd.read_parquet(verify(old_record['prediction']))
            np.testing.assert_array_equal(ids,prior.mvt_id);np.testing.assert_array_equal(pred,prior.pred)
        else:
            restored=CatBoostRegressor();restored.load_model(str(verify(model_record['model'])))
            np.testing.assert_array_equal(pred,np.maximum(restored.predict(X,thread_count=1),0).astype(np.float32))
            path=OUT/'final.parquet';pd.DataFrame(dict(mvt_id=ids,pred=pred)).to_parquet(path,index=False)
            outputs=[art(path)];del restored
        write(receipt,dict(identity=art(manifest),view=view,rows=len(ids),old_replay_exact=True,
            final_reload_exact=view=='final',predictions=outputs,original134_inputs_exact=True,no_fit=True))
        print(dict(view=view,rows=len(ids),status='pass'),flush=True)
        del V,converted,base,X,model,pred;gc.collect()


if __name__=='__main__':
    with threadpool_limits(4):run()
