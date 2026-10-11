"""Fit the passing public airport component once on the full ordinary year.

Retain ten reusable models and ranking predictions for a combined candidate.
Use exactly the 168-column CV recipe; no new selection, version or upload.
"""
import argparse
import gc
from pathlib import Path
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_airport_trees as cv
import public_refinements_cv as combined
import public_delay_lgb_submit as main
from records import write_json

ROOT = cv.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_airport_submit'
MANIFEST = OUT / 'identity.json'
parent, old = main.parent, main.old
art, read, verify = cv.art, cv.read, cv.verify


def guard():
    cv.guard(); combined.guard()
    result = read(cv.OUT / 'result.json')
    assert read(cv.OUT / 'audit.json')['status'] == 'pass'
    assert result['paired']['candidate_transfer_vs_control_transfer']['decision'] == 'improvement'
    assert all(s['candidate_transfer'] < s['control_transfer'] for s in result['scores'].values())
    result_joint = read(combined.OUT / 'result.json')
    assert read(combined.OUT / 'audit.json')['status'] == 'pass'
    assert result_joint['paired']['combined_vs_control']['decision'] == 'improvement'
    records = [cv.completed(month, airport) for month in cv.parent.ts.SPLITS
        for airport in ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']]
    assert all(r is not None for r in records)
    first = records[0]
    for record in records:
        for key in ['params', 'columns', 'categoricals', 'rounds']:
            assert record[key] == first[key]
    assert first['params'] == cv.params() and len(first['columns']) == 168
    original = read(parent.OUT / 'prepared.json')
    assert list(original['original_columns']) + cv.features.FEATURES == first['columns']
    value = dict(source=art(__file__), helpers=[art(m.__file__) for m in [cv,combined,main,old,parent]],
        params=first['params'], columns=first['columns'], categoricals=first['categoricals'],
        rounds=5000, seed=42, train_rows=2051327, view='old',
        cv=art(cv.OUT/'result.json'), cv_audit=art(cv.OUT/'audit.json'),
        joint_cv=art(combined.OUT/'result.json'), joint_audit=art(combined.OUT/'audit.json'),
        original_prepared=art(parent.OUT/'prepared.json'),
        original_inputs=art(parent.hs.INPUTS), training_ids=original['training_ids'],
        training_features=art(cv.features.OUT/'features2025.parquet'),
        ranking_features=art(main.public_rome.OUT/'ranking_features.parquet'),
        ranking_feature_audit=art(main.public_rome.OUT/'ranking_feature_audit.json'),
        truth=art(cv.features.base.OUT/'rows2025.parquet'),
        no_new_selection=True, no_version=True, no_upload=True)
    assert read(verify(value['ranking_feature_audit']))['status'] == 'pass'
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists(): assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('**/*.txt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST, value)
    return value


def paths(airport):
    return OUT/'models'/f'{airport}.txt', OUT/'members'/f'{airport}.json'


def completed(airport):
    mp, rp = paths(airport)
    if not rp.exists():
        assert not mp.exists(), 'Unrecorded model; inspect before resuming'
        return None
    record = read(rp)
    assert record['identity'] == art(MANIFEST) and record['params'] == cv.params()
    verify(record['model'])
    return record


def ranking(recipe, original):
    packet, provenance = old.ranking_inputs('old', original)
    ids = np.asarray(packet['rank_ids'])
    assert len(ids) == 344841 and pd.Index(ids).is_unique and packet['view'] == 'old'
    base = packet['Vn'][original['original_columns']]
    public = pd.read_parquet(verify(recipe['ranking_features'])).set_index('mvt_id')
    value = pd.concat([base, public.loc[ids, cv.features.FEATURES].set_axis(base.index)], axis=1)
    assert list(value) == recipe['columns']
    pd.testing.assert_frame_equal(value[original['original_columns']], base, check_exact=True)
    raw = pd.read_parquet(cv.features.base.OUT/'ranking_inputs.parquet', columns=['mvt_id','airport']).set_index('mvt_id')
    airports = raw.loc[ids,'airport'].to_numpy()
    assert len(set(airports)) == 10
    return value, ids, airports, provenance


def fit():
    recipe = guard(); original = parent.validate_prepared()
    airports_expected = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
    for directory in ['models','members','predictions']: (OUT/directory).mkdir(exist_ok=True)
    if any(completed(airport) is None for airport in airports_expected):
        data = pd.read_pickle(verify(recipe['original_inputs']))
        membership = pd.read_parquet(verify(recipe['training_ids']))
        np.testing.assert_array_equal(membership.original_index, data['Xn'].index)
        ids = membership.mvt_id.to_numpy(np.int64)
        assert len(ids) == original['training_rows'] == recipe['train_rows'] == len(data['yn'])
        truth = pd.read_parquet(verify(recipe['truth'])).set_index('mvt_id').loc[ids]
        assert set(truth.month) == {f'2025-{m:02}' for m in range(1,13)}
        y = np.asarray(data['yn'], float)
        np.testing.assert_array_equal(y, truth.y)
        airports = truth.airport.to_numpy()
        assert sorted(set(airports)) == airports_expected
        base = data['Xn'][original['original_columns']]
        public = pd.read_parquet(verify(recipe['training_features'])).set_index('mvt_id')
        X = pd.concat([base,public.loc[ids,cv.features.FEATURES].set_axis(base.index)],axis=1)
        pd.testing.assert_frame_equal(X[original['original_columns']],base,check_exact=True)
        cats = cv.parent.ideas.signals._categorical(X.columns)+[c for c in cv.parent.ideas.ef.CATEGORICAL if c in X]
        assert list(X) == recipe['columns'] and cats == recipe['categoricals']
        del data,membership,truth,base,public
        gc.collect()
        for airport in airports_expected:
            if completed(airport) is not None: continue
            mask = airports == airport; mp,rp = paths(airport)
            start = time.monotonic()
            print(f'Full year {airport}: {mask.sum():,} ordinary rows,168 inputs,P04/5000',flush=True)
            dataset = lgb.Dataset(X.loc[mask],y[mask],categorical_feature=cats)
            model = lgb.train(recipe['params'],dataset,num_boost_round=5000)
            assert model.current_iteration() == 5000 and model.feature_name() == recipe['columns']
            temporary = mp.with_suffix('.txt.tmp'); model.save_model(str(temporary)); temporary.replace(mp)
            write_json(rp,dict(identity=art(MANIFEST),model=art(mp),airport=airport,
                params=recipe['params'],rounds=5000,seed=42,columns=recipe['columns'],categoricals=cats,
                train_rows=int(mask.sum()),training_ids=recipe['training_ids'],all_training_truth_exact=True,
                minutes=(time.monotonic()-start)/60,host=platform.node(),lightgbm=lgb.__version__))
            del dataset,model; gc.collect()
        del X,y,airports,ids; gc.collect()
    V,ids,airports,provenance = ranking(recipe,original)
    values = np.full(len(ids),np.nan,dtype=np.float32); checks=[]
    for airport in airports_expected:
        record = completed(airport); assert record is not None
        model = lgb.Booster(model_file=str(verify(record['model'])))
        assert model.feature_name() == recipe['columns'] and model.current_iteration() == 5000
        mask = airports == airport
        pred = cv.parent.modeling.predict(model,V.loc[mask],threads=4).astype(np.float32)
        restored = lgb.Booster(model_file=str(verify(record['model'])))
        np.testing.assert_array_equal(pred,np.maximum(restored.predict(V.loc[mask],num_threads=2),0).astype(np.float32))
        values[mask]=pred
        checks.append(dict(airport=airport,rows=int(mask.sum()),member=art(paths(airport)[1]),all_reload_exact=True))
        del model,restored; gc.collect()
    assert np.isfinite(values).all() and (values>=0).all()
    frame = pd.DataFrame(dict(mvt_id=ids,pred=values)); pp=OUT/'predictions/old.parquet'
    if pp.exists(): pd.testing.assert_frame_equal(pd.read_parquet(pp),frame,check_exact=True)
    else: frame.to_parquet(pp,index=False)
    record=dict(identity=art(MANIFEST),prediction=art(pp),rows=len(ids),checks=checks,
        ranking_packet=provenance,all_ranking_reload_exact=True)
    rp=pp.with_suffix('.json')
    if rp.exists(): assert read(rp)==record
    else: write_json(rp,record)
    print('Ten full-year models and all344841 ranking reloads complete',flush=True)


def audit():
    recipe=guard(); assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    original=parent.validate_prepared(); V,ids,airports,_=ranking(recipe,original)
    record=read(OUT/'predictions/old.json'); assert record['identity']==art(MANIFEST)
    saved=pd.read_parquet(verify(record['prediction']))
    np.testing.assert_array_equal(saved.mvt_id,ids)
    checks=[]
    for airport in sorted(set(airports)):
        member=completed(airport); assert member is not None
        mask=airports==airport
        model=lgb.Booster(model_file=str(verify(member['model'])))
        assert model.feature_name()==recipe['columns'] and model.current_iteration()==5000
        values=np.maximum(model.predict(V.loc[mask],num_threads=2),0).astype(np.float32)
        np.testing.assert_array_equal(values,saved.pred.to_numpy()[mask])
        checks.append(dict(airport=airport,rows=int(mask.sum()),all_predictions_exact=True,member=art(paths(airport)[1])))
        del model; gc.collect()
    assert sum(c['rows'] for c in checks)==344841
    report=dict(status='pass',identity=art(MANIFEST),predictions=art(OUT/'predictions/old.json'),
        all344841_predictions_reloaded=True,original156_ranking_columns_exact=True,checks=checks,
        source=art(__file__),no_version=True,no_upload=True)
    write_json(OUT/'audit.json',report); print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['guard','fit','audit'])
    args=parser.parse_args()
    with threadpool_limits(4): globals()[args.stage]()
