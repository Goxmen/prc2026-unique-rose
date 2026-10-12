"""Test recovered v38 boosters on cached old inputs before any final inference.

The two archived model hashes match their original submission manifest. Current
feature caches may differ historically: the old integer vector must match in
every row. A failed replay is recorded and stops without fitting or substitution.
"""
import gc,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import artifact_submit as recipe
from records import sha256,write_json

ROOT=recipe.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_legacy_v38_replay'


def art(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha256(p))


def run():
    assert not (OUT/'old.json').exists()
    manifest=json.loads(recipe.MANIFEST.read_text());models={};paths=[]
    for key,a in manifest['model'].items():
        p=recipe.RUN/a['path'];assert sha256(p)==a['sha256'];paths.append(p)
        models[key]=lgb.Booster(model_file=str(p))
    cache=ROOT/'data/derived/scored_submission_cache_20261011/manifest.json'
    reference=json.loads(cache.read_text())['38'];oldpath=ROOT/reference['file'];assert sha256(oldpath)==reference['sha256']
    expected=pd.read_parquet(oldpath).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    OUT.mkdir(parents=True,exist_ok=True)
    paths += [Path(__file__),Path(recipe.__file__),Path(recipe.signals.__file__),Path(recipe.control.__file__),
        Path(recipe.corrections.__file__),recipe.MANIFEST,cache,oldpath]
    identity=dict(sources=[art(p) for p in paths],rule_parameters=manifest['calibration'],
        gate='All original344841 integer predictions exact; otherwise no final inference',no_fit=True,no_upload=True)
    write_json(OUT/'identity.json',identity)
    for view in ['old','final']:
        source=ROOT/f'VALIDATION/runs/v3_20261008_final_submit/work/inputs_{view}.pkl'
        if view=='final':assert json.loads((OUT/'old.json').read_text())['all_original_predictions_exact']
        data=pd.read_pickle(source);frames=[]
        for month,item in data['months'].items():
            V,raw=item['V'],item['raw']
            assert set(models['ordinary'].feature_name()).issubset(V) and set(models['classifier'].feature_name()).issubset(V)
            values=recipe.signals.predict(models['ordinary'],models['classifier'],V,raw,recipe.KIND,
                threads=4,extra=recipe.control.ORDINARY,classifier_extra=recipe.control.CLASSIFIER,
                late_mean=manifest['calibration']['parameters']['late_mean'],dayshift_weight=None)
            f,_=recipe.apply_corrections(values['final'],raw,V,manifest['calibration']['corrections'],values['p'])
            frames.append(f[['mvt_id','pred_combined']])
        frame=pd.concat(frames,ignore_index=True).set_index('mvt_id').loc[data['rank_ids']].reset_index()
        frame['submitted']=np.rint(np.maximum(frame.pred_combined,0))
        assert frame.mvt_id.is_unique and np.isfinite(frame.submitted).all()
        frame.to_parquet(OUT/f'{view}.parquet',index=False)
        result=dict(identity=art(OUT/'identity.json'),inputs=art(source),prediction=art(OUT/f'{view}.parquet'),rows=len(frame))
        if view=='old':
            assert len(frame)==len(expected)==344841
            delta=frame.submitted.to_numpy()-expected.loc[frame.mvt_id].to_numpy()
            result.update(all_original_predictions_exact=bool((delta==0).all()),changed_rows=int((delta!=0).sum()),
                rms_difference=float(np.sqrt(np.mean(delta*delta))),max_abs_difference=float(abs(delta).max()))
        else:result['old_control_exact']=True
        write_json(OUT/f'{view}.json',result);print({k:v for k,v in result.items() if k not in ['identity','inputs','prediction']},flush=True)
        if view=='old' and not result['all_original_predictions_exact']:return
        del data,frames,frame;gc.collect()


if __name__=='__main__':
    with threadpool_limits(4):run()
