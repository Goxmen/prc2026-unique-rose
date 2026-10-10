"""Legacy raw/capped/clock recipes on current Jan/Jul folds, for a new blend.

These are new fold fits, not repeated full-year submissions or old ledger
backfills. Histories use allowed training labels only and freeze at the training
end. Every held-out row is predicted. One CPU fit at a time; resume is hashed.
"""
import gc,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import lightgbm as lgb,numpy as np,pandas as pd
import seasonal,tuning_seasonal as ts,modeling,overnight_models as om
from common import connect,TRAIN_GLOB,build_features,CATEGORICAL
from movement_data import load_phase
from clock_features import clock_features
from features_time import fpl_flags
from registered_rules import apply_rule
from records import sha256,write_json
OUT=ROOT/'VALIDATION/runs/v3_20261011_legacy_raw_jan_jul'
VERSIONS=[1,2,3,4,5,6,7,8,9,17,18,19,20,21]

def frozen_backend():
    meta=json.loads((ROOT/'submissions/unique-rose_v5.meta.json').read_text())
    for name,digest in meta['model_source'].items():
        if name=='VALIDATION/modeling.py':continue
        assert sha256(ROOT/name)==digest,name
    historical=meta['model_source']['VALIDATION/modeling.py']
    # The exact September 15 source blob is not retained. This earlier,
    # archived original27 implementation is explicit; do not claim its hash
    # equals the historical submission's source hash. Model metadata below
    # independently verifies feature order and training parameters.
    archived=ROOT/'VALIDATION/archive/runs/v3_20260909T161759260709Z/source/VALIDATION/modeling.py'
    wanted='5eda81a506acb91e1baba28e369fac379bf7bfbbed31680deaf7474da8ed920d'
    assert sha256(archived)==wanted
    path=OUT/'source/modeling.py';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_bytes(archived.read_bytes())
    assert path.exists() and sha256(path)==wanted
    spec=importlib.util.spec_from_file_location('legacy_raw_frozen',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.PARAMS==meta['params']
    meta['reconstruction']=dict(historical_source_sha256=historical,archived_source_sha256=wanted,
        exact_historical_source_retained=False,
        basis='Original27 feature implementation; recorded feature order and parameters matched')
    return module,meta

def paths(month,version):
    prefix=f'{month}_v{version}'
    return OUT/'predictions'/f'{prefix}.parquet',OUT/'models'/f'{prefix}.txt',OUT/'records'/f'{prefix}.json'

def done(month,version,identity):
    pred,model,record=paths(month,version)
    if not record.exists():
        assert not pred.exists() and not model.exists(),'Unrecorded output; inspect interrupted save'
        return False
    r=json.loads(record.read_text());assert r['identity']==identity
    assert sha256(pred)==r['prediction_sha256']
    if version!=1:assert sha256(model)==r['model_sha256']
    return True

def save(month,version,ids,pred,identity,booster=None,**info):
    path,model,record=paths(month,version)
    assert len(ids)==len(pred) and np.isfinite(pred).all()
    pd.DataFrame(dict(mvt_id=ids,month=month,pred=np.maximum(pred,0))).to_parquet(path,index=False)
    if booster is not None:booster.save_model(str(model))
    write_json(record,dict(identity=identity,version=version,month=month,rows=len(ids),
        prediction_sha256=sha256(path),model_sha256=sha256(model) if booster is not None else None,**info))
    print(f'{month} v{version} complete',flush=True)

def main():
    for sub in ['predictions','models','records']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    backend,reference=frozen_backend()
    identity=dict(source_sha256=sha256(__file__),backend_sha256=sha256(OUT/'source/modeling.py'),
        helpers={str(Path(m.__file__).relative_to(ROOT)):sha256(m.__file__) for m in [modeling,om,ts]},
        folds=ts.FOLDS,freezes={m:ts.fold_freeze(m) for m in ts.SPLITS},versions=VERSIONS,seed=42,threads=6,
        reconstruction=reference['reconstruction'])
    ip=OUT/'identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity
    write_json(ip,identity)
    con=connect();dep=load_phase(con,TRAIN_GLOB,'DEP');arr=load_phase(con,TRAIN_GLOB,'ARR');con.close()
    for month in ts.SPLITS:
        pending=[v for v in VERSIONS if not done(month,v,identity)]
        if not pending:continue
        td,ta,held=seasonal.split_frames(dep,arr,dict(train=ts.FOLDS[month],val=[month]));vd,va=held[month]
        assert not set(td.mvt_id)&set(vd.mvt_id)
        ids=vd.mvt_id.to_numpy()
        cfg=dict(reference['config'],max_taxi_sec=200000,fpl_flag=False,threads=6)
        print(f'{month}: preparing original27 on {len(td)} rows',flush=True)
        X,y,cats=backend.training_matrix(td,ta,cfg)
        V=backend.validation_matrix(td,ta,vd,va,ts.fold_freeze(month),cats,cfg)
        raw=td.loc[X.index]
        assert np.array_equal(y,raw.taxi_sec.to_numpy()) and list(X)==om.ORIGINAL_FEATURES
        X45=pd.concat([X,clock_features(raw)],axis=1)
        V45=pd.concat([V,clock_features(vd)],axis=1)
        for version in pending:
            if done(month,version,identity):continue
            started=time.monotonic()
            info=dict(training_months=ts.FOLDS[month],history_frozen_at=ts.fold_freeze(month),host=os.uname().nodename)
            if version==1:
                clean=td[td.taxi_sec.between(1,7200)]
                means=clean.groupby('ADEP_mvt').taxi_sec.mean()
                pred=vd.ADEP_mvt.map(means).fillna(clean.taxi_sec.mean()).to_numpy(float)
                save(month,version,ids,pred,identity,training_rows=len(clean),**info);continue
            if version==2:
                keep=td.taxi_sec.between(1,7200);A=build_features(td.loc[keep]);B=build_features(vd)
                for c in CATEGORICAL:
                    A[c]=A[c].astype('category');B[c]=pd.Categorical(B[c],categories=A[c].cat.categories)
                target=td.loc[keep,'taxi_sec'].to_numpy(float)
                params=dict(objective='regression',metric='rmse',learning_rate=.05,num_leaves=255,
                    min_data_in_leaf=200,feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,seed=42,verbosity=-1,num_threads=6)
                rounds=3000;booster=lgb.train(params,lgb.Dataset(A,target,categorical_feature=CATEGORICAL),num_boost_round=rounds)
                pred=booster.predict(B,num_threads=6)
            elif version in range(3,10):
                meta=json.loads((ROOT/f'submissions/unique-rose_v{version}.meta.json').read_text())
                c=meta.get('config',dict(max_taxi_sec=meta.get('max_taxi_sec'),
                    rounds=meta.get('rounds'),fpl_flag=meta.get('fpl_flag'),lag_train_days=0,early_stopping_rounds=0))
                keep=y<=c['max_taxi_sec'];A=X.loc[keep].copy();B=V.copy();target=y[keep]
                if c['fpl_flag']:
                    A=pd.concat([A,fpl_flags(raw.loc[keep])],axis=1);B=pd.concat([B,fpl_flags(vd)],axis=1)
                A,cat=backend.fit_categories(A);B=backend.apply_categories(B,cat)
                assert list(A)==meta.get('features',om.ORIGINAL_FEATURES)
                for key,value in backend.PARAMS.items():assert meta['params'][key]==value
                rounds=c['rounds'];config=dict(c,threads=6)
                booster=backend.fit_booster(A,target,42,rounds,config,f'{month} v{version}')
                pred=np.maximum(booster.predict(B,num_threads=6),0)
            else:
                recipe={17:'no_month',18:'no_history',19:'clock_features',20:'small_trees',21:'rule_routing'}[version]
                booster,_=om.fit_model(X45,y,raw,recipe,42,rounds=800,threads=6)
                pred=om.predict_model(booster,V45,recipe,threads=6);pred,_=apply_rule(pred,vd,'rome_4h')
                target=y;rounds=800
            save(month,version,ids,pred,identity,booster,training_rows=len(target),rounds=rounds,
                 seconds=time.monotonic()-started,**info)
            del booster;gc.collect()
        del X,V,X45,V45,td,ta,vd,va,held;gc.collect()
    write_json(OUT/'complete.json',dict(identity=identity,versions=VERSIONS))

if __name__=='__main__':main()
