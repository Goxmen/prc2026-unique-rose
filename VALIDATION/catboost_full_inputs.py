"""Give the ordinary K07 CatBoost all174 retained main-model inputs.

Its historical134 inputs omitted22 combined inputs,6 callsign clocks and12
public daily statistics. Keep its original rows, labels, seed42 and10000 rounds.
Score only this component replacement inside v76. No production or upload.
"""
import argparse
import gc
from pathlib import Path
import platform
import time

from catboost import CatBoostRegressor, Pool
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import catboost_jan_jul as cb
import public_stack_cv as joint
from records import write_json

ROOT=joint.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_catboost_full_inputs'
MANIFEST=OUT/'identity.json'
parent,features=joint.main.parent,joint.main.features
art,read,verify=joint.art,joint.read,joint.verify
DESIGN=dict(parent='v76',change='all174 main-model inputs in the ordinary CatBoost',
    original_inputs=134,combined_inputs=parent.ideas.COMBO,callsign_inputs=parent.FEATURES,
    public_inputs=features.FEATURES,seed=42,trial='K07',rounds=10000,
    original_training_rows=True,original_uncapped_labels=True,no_new_stopping=True,
    no_parameter_search=True,comparison='same selected stack; only ordinary CatBoost changes',
    caveat='Saved K07 control was fitted on laptop; desktop GPU and thread-count variation can affect small differences',
    no_production=True,no_upload=True)


def params():
    winner=parent.hpo.winner('catboost')
    assert winner['trial']=='K07' and winner['checkpoint']=='c10000'
    spec=parent.hpo.trial_spec('catboost','K07')
    assert spec['iterations']==10000
    value=dict(parent.hpo.CB_BASE,**spec['overrides'])
    assert not value.pop('combo_inputs',False)
    return dict(loss_function='RMSE',iterations=10000,random_seed=42,task_type='GPU',devices='0',
        thread_count=4,gpu_ram_part=.9,verbose=1000,allow_writing_files=False,**value)


def guard():
    assert read(joint.OUT/'audit.json')['status']=='pass'
    assert read(features.OUT/'audit.json')['status']=='pass'
    parent.checked_manifest()
    paths=[Path(__file__),Path(cb.__file__),Path(joint.__file__),Path(parent.__file__),
        Path(features.__file__),Path(parent.hpo.__file__),Path(parent.ideas.__file__),
        Path(parent.ideas.signals.__file__),Path(parent.ideas.ef.__file__),
        joint.OUT/'scored.parquet',joint.OUT/'audit.json',features.OUT/'features2025.parquet',
        features.OUT/'audit.json',features.base.OUT/'rows2025.parquet',
        parent.hpo.pred_path('catboost','K07',42),parent.hpo.record_path('catboost')]
    for month in parent.ts.SPLITS:
        paths.extend([parent.input_cache.input_path(month),parent.feature_path(month,'train'),parent.feature_path(month,'held')])
    value=dict(design=DESIGN,params=params(),sources=[art(p) for p in paths])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('**/*.cbm')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST,value)
    return value


def paths(month):
    return OUT/'models'/f'{month}_s42.cbm',OUT/'predictions'/f'{month}_s42.parquet',OUT/'predictions'/f'{month}_s42.json'


def completed(month):
    mp,pp,rp=paths(month)
    if not rp.exists():
        assert not mp.exists() and not pp.exists(),'Unrecorded fit; inspect before resuming'
        return None
    record=read(rp)
    assert record['identity']==art(MANIFEST) and record['params']==params()
    verify(record['model']);verify(record['prediction'])
    return record


def inputs(month,table,truth,held_only=False):
    data=pd.read_pickle(parent.input_cache.input_path(month));names=data['baseline']
    assert len(names)==len(set(names))==156
    original=[c for c in names if c not in parent.ideas.COMBO]
    assert len(original)==134 and len(parent.ideas.COMBO)==22
    train_ids,ids=np.asarray(data['train_ids']),np.asarray(data['ids'])
    assert not np.intersect1d(train_ids,ids).size
    assert set(truth.loc[train_ids,'month'])==set(parent.ts.FOLDS[month])
    assert set(truth.loc[ids,'month'])=={month}
    np.testing.assert_array_equal(data['y'],truth.loc[train_ids,'y'])
    result=dict(y=data['y'],train_ids=train_ids,ids=ids,original_columns=original)
    for key,part,row_ids in [('X','train',train_ids),('V','held',ids)]:
        if held_only and key=='X':continue
        baseline=data[key].loc[:,names]
        clocks=parent.clocks.load_features(parent.feature_path(month,part))
        extra=parent.clocks.add_features(baseline,row_ids,clocks)
        value=pd.concat([extra,table.loc[row_ids,features.FEATURES].set_axis(extra.index)],axis=1)
        assert list(value)==list(names)+list(parent.FEATURES)+features.FEATURES and len(value.columns)==174
        pd.testing.assert_frame_equal(value[original],data[key][original],check_exact=True)
        cats=parent.ideas.signals._categorical(value.columns)+[c for c in parent.ideas.ef.CATEGORICAL if c in value]
        assert len(cats)==len(set(cats))
        converted=cb.frame(value,cats)
        # This checks the actual CatBoost categorical representation too.
        old=cb.frame(data[key][original],parent.ideas.signals._categorical(original))
        pd.testing.assert_frame_equal(converted[original],old,check_exact=True)
        for name in converted.columns:
            if name not in cats:assert pd.api.types.is_numeric_dtype(converted[name]),name
        result[key]=converted;result['categoricals']=cats
    return result


def fit():
    identity=guard()
    table=pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id')
    truth=pd.read_parquet(features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    (OUT/'models').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    for month in parent.ts.SPLITS:
        if completed(month) is not None:continue
        data=inputs(month,table,truth);cats=data['categoricals']
        X,V=data['X'],data['V'];mp,pp,rp=paths(month)
        start=time.monotonic();parent.log(f'{month}: {len(X):,} rows,134+40 inputs,K07/10000 seed42')
        training=Pool(X,data['y'],cat_features=cats);held=Pool(V,cat_features=cats)
        model=CatBoostRegressor(**identity['params']);model.fit(training)
        assert model.tree_count_==10000 and model.feature_names_==list(X)
        pred=np.maximum(model.predict(held,thread_count=4),0)
        assert np.isfinite(pred).all()
        model.save_model(str(mp))
        restored=CatBoostRegressor();restored.load_model(str(mp))
        np.testing.assert_array_equal(np.maximum(restored.predict(held,thread_count=4),0),pred)
        pd.DataFrame(dict(mvt_id=data['ids'],pred=pred.astype(np.float32))).to_parquet(pp,index=False)
        write_json(rp,dict(identity=art(MANIFEST),design=DESIGN,model=art(mp),prediction=art(pp),
            input=art(parent.input_cache.input_path(month)),params=identity['params'],
            original_columns=data['original_columns'],columns=list(X),categoricals=cats,
            train_rows=len(X),held_rows=len(V),all_original134_values_and_categories_exact=True,
            all_saved_predictions_reloaded_exact=True,minutes=(time.monotonic()-start)/60,
            machine=platform.node(),actual_params=model.get_all_params()))
        parent.log(f'{month}: CatBoost fit and all-row reload complete')
        del data,X,V,training,held,model,restored,pred
        gc.collect()


def score():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed score'
    neural=joint.neural;cat=neural.cat;replay=cat.r20.physical.replay
    identity=neural.guard();sc,rome,_,trees=replay.baseline.baseline()
    prior=pd.read_parquet(cat.OUT/'scored.parquet').set_index('mvt_id')
    selected=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id')
    truth=pd.read_parquet(features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    matches=pd.read_parquet(cat.r20.constraints.exclusion.OUT/'matches2025.parquet');frames=[]
    for month in parent.ts.SPLITS:
        record=completed(month);assert record is not None
        d=sc.d[month];ids=d['ids']
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        physical=np.mean([replay.values(verify(neural.completed(month,k,identity)['prediction']),ids,'base') for k in range(3)],axis=0)
        main=replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        new=replay.values(verify(record['prediction']),ids,'pred')
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        permitted=matches[matches.donor_month.isin(parent.ts.FOLDS[month]+[month])]
        outputs={}
        for name,ordinary_cat in [('control',trees['catboost'][42][month]),('candidate',new)]:
            ordinary=.71*.5*(calls+physical)+.29*np.mean([main,trees['airport'][42][month],ordinary_cat],axis=0)
            final=sc.final(month,ordinary,probability)
            transferred=cat.r20.constraints.matched_bound(truth.loc[ids].reset_index(),
                prior.loc[ids,'v74'].to_numpy()+final-prior.loc[ids,'control'].to_numpy())
            transferred,_=cat.r20.constraints.exclusion.apply(ids,np.maximum(transferred,0),permitted)
            outputs[name]=final;outputs[name+'_transfer']=transferred
        np.testing.assert_array_equal(outputs['control'],selected.loc[ids,'combined'])
        np.testing.assert_array_equal(outputs['control_transfer'],selected.loc[ids,'combined_transfer'])
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**outputs)))
    frame=pd.concat(frames,ignore_index=True)
    assert len(frame)==344419 and frame.mvt_id.is_unique and np.isfinite(frame[list(outputs)]).all().all()
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in outputs} for m,g in frame.groupby('month')}
    paired={f'candidate{s}_vs_control{s}':parent.ts.paired_bootstrap(frame.day,frame.month,
        (frame['control'+s]-frame.y)**2,(frame['candidate'+s]-frame.y)**2) for s in ['', '_transfer']}
    frame.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,identity=art(MANIFEST),predictions=art(OUT/'scored.parquet'),
        scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in outputs},
        paired=paired,selected_control_exact=True)
    write_json(OUT/'result.json',result);print(dict(cv=result['cv'],paired=paired),flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['predictions'])
    table=pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id')
    truth=pd.read_parquet(features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    checks=[]
    for month in parent.ts.SPLITS:
        record=completed(month);assert record is not None
        data=inputs(month,table,truth,held_only=True)
        model=CatBoostRegressor();model.load_model(str(verify(record['model'])))
        assert model.feature_names_==list(data['V'])==record['columns']
        assert data['categoricals']==record['categoricals'] and model.tree_count_==10000
        pred=np.maximum(model.predict(Pool(data['V'],cat_features=data['categoricals']),thread_count=2),0).astype(np.float32)
        saved=pd.read_parquet(verify(record['prediction']))
        np.testing.assert_array_equal(saved.mvt_id,data['ids']);np.testing.assert_array_equal(saved.pred,pred)
        checks.append(dict(month=month,rows=len(saved),all_predictions_exact=True,all_original134_values_exact=True))
        del data,model,pred,saved;gc.collect()
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('candidate',OUT/'scored.parquet'),('truth',features.base.OUT/'rows2025.parquet'),('selected',joint.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.control is distinct from s.combined or c.control_transfer is distinct from s.combined_transfer
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join selected s using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for name in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from candidate group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][name],atol=1e-8,rtol=0)
    report=dict(status='pass',identity=art(MANIFEST),result=art(OUT/'result.json'),model_reloads=checks,
        independent_sql_truth_calendar_selected_scores=True,all_rows=n,source=art(__file__))
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['guard','fit','score','audit'])
    args=parser.parse_args()
    with threadpool_limits(4):globals()[args.stage]()
