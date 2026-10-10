"""Frozen R20 + prior-own-aircraft arrival schedule-copy indicator and age.

Preparation reuses audited fold-scoped arrival features and checks every original
classifier value/row. Fit is standalone for compact laptop transfer: only this
script, two compressed packets and their manifests are required. Two fixed1700
round seed42 classifiers; unchanged v58 scoring with full and50/50 probabilities.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'VALIDATION/runs/v3_20261010_rome_own_arrival_jan_jul'
ROT=ROOT/'data/derived/aircraft_rotation_20261010'
SOURCE=ROOT/'VALIDATION/runs/v3_20261010_rome_adsblol_jan_jul'
MONTHS=['2025-01','2025-07']
FOLDS={'2025-01':[f'2025-{m:02d}' for m in range(3,12)],
       '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
FEATURES=['rome_own_arrival_copy','rome_own_arrival_age_sec']
PARAMS=dict(objective='binary',metric='binary_logloss',learning_rate=.02,num_leaves=127,
    min_data_in_leaf=50,feature_fraction=.6,bagging_fraction=.8,bagging_freq=1,verbosity=-1,
    seed=42,bagging_seed=43,feature_fraction_seed=44,num_threads=8)
DESIGN=dict(parent='v58',part='R20 Rome classifier',features=FEATURES,rounds=1700,params=PARAMS,
    rows='unchanged original R20 after_drop, including both target classes',
    original_features=68,added_features=2,target='unchanged abs(takeoff-schedule-taxi)<=60',
    own_copy='abs(ar_prev_arrival_delay_sec)<=60; NaN when unavailable',
    age='audited ar_prev_inblock_age_sec, no extra fill or clipping',
    reference_scope='train arrivals only for training queries; train+own held month arrivals for held queries',
    score_arms=['own_arrival','half_own_arrival_half_R20'],network_share=.71,
    exploratory_reused_development_months=True,no_threshold_or_parameter_search=True)

def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def log(value):print(f'[{time.strftime("%H:%M:%S")}] {value}',flush=True)

def add_features(X,ids,table):
    if len(X)!=len(ids) or not table.index.is_unique or set(FEATURES)&set(X):
        raise ValueError('Invalid input shape, duplicate identity, or feature overlap')
    if not pd.Index(ids).isin(table.index).all():raise ValueError('Rotation table misses classifier queries')
    source=table.loc[np.asarray(ids)]
    delay=source.ar_prev_arrival_delay_sec.to_numpy(float)
    flags=np.where(np.isfinite(delay),(np.abs(delay)<=60).astype(float),np.nan).astype(np.float32)
    age=source.ar_prev_inblock_age_sec.to_numpy(np.float32)
    result=X.copy()
    result[FEATURES[0]]=flags;result[FEATURES[1]]=age
    pd.testing.assert_frame_equal(result.loc[:,X.columns],X,check_exact=True)
    assert result.index.equals(X.index) and list(result)==list(X)+FEATURES
    return result

def prepare():
    sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
    import hpo_jan_jul as hpo
    import rome_hybrid as rh
    spec=hpo.trial_spec('rome','R20')
    assert dict(rh.CLASSIFIER_PARAMS,**spec['overrides'],seed=42,bagging_seed=43,
                feature_fraction_seed=44,num_threads=8)==PARAMS
    assert hpo.winner('rome')['trial']=='R20' and int(hpo.winner('rome')['checkpoint'][1:])==1700
    manifest=read(ROT/'folds.json');assert manifest['source_sha256']==sha(ROOT/'VALIDATION/aircraft_rotation.py')
    for month in MONTHS:
        dest=OUT/'inputs'/f'{month}.pkl.gz';meta_path=dest.with_suffix('.json')
        if dest.exists():
            meta=read(meta_path)
            assert meta['design']==DESIGN and meta['runner_sha256']==sha(Path(__file__)) and meta['packet_sha256']==sha(dest)
            log(f'{month}: existing guarded packet; skip');continue
        source=SOURCE/'inputs'/f'{month}.pkl';d=pd.read_pickle(source)
        original=read(ROOT/'VALIDATION/runs/v3_20261010_rome_catboost_jan_jul/predictions'/f'{month}_s42.json')
        assert sha(source)==original['input_sha256']
        names=original['columns'];cats=original['categoricals'];assert len(names)==68 and len(cats)==11
        assert len(d['A'])==original['training_rows'] and int(np.sum(d['target']))==original['positives']
        packet={name:d[name] for name in ['target','use','ids','train_ids']};rot_sources={}
        for part,key,ids in [('train','A',d['train_ids']),('held','B',d['ids'][d['use']])]:
            path=ROT/f'{month}_{part}.parquet';entry=manifest['folds'][month][part]
            assert sha(path)==entry['sha256']
            expected=FOLDS[month]+([month] if part=='held' else [])
            assert entry['reference_months']==expected
            table=pd.read_parquet(path,columns=['mvt_id','ar_prev_arrival_delay_sec','ar_prev_inblock_age_sec']).set_index('mvt_id')
            packet[key]=add_features(d[key].loc[:,names],ids,table)
            pd.testing.assert_frame_equal(packet[key].loc[:,names],d[key].loc[:,names],check_exact=True)
            rot_sources[part]=dict(path=str(path.relative_to(ROOT)),sha256=entry['sha256'],reference_months=expected)
        for key in ['target','use','ids','train_ids']:np.testing.assert_array_equal(packet[key],d[key])
        packet['categoricals']=cats;packet['columns']=names+FEATURES
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix('.tmp')
        with gzip.open(temp,'wb',compresslevel=3) as stream:pickle.dump(packet,stream,protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(dest)
        write(meta_path,dict(design=DESIGN,runner_sha256=sha(Path(__file__)),packet_sha256=sha(dest),
            original_cache_sha256=sha(source),rotation_manifest_sha256=sha(ROT/'folds.json'),rotation_sources=rot_sources,
            month=month,columns=packet['columns'],categoricals=cats,train_rows=len(packet['A']),held_rome_rows=len(packet['B']),
            total_held_rows=len(packet['ids']),positives=int(np.sum(packet['target'])),bytes=dest.stat().st_size,
            exact_original_columns_verified=names,exact_original_membership_verified=True,
            nonmissing={key:{f:int(packet[key][f].notna().sum()) for f in FEATURES} for key in ['A','B']}))
        log(f'{month}: compressed {dest.stat().st_size/1e6:.2f}MB; original68 columns/rows/targets identical')
        del packet,d,table;gc.collect()

def load_packet(month):
    source=OUT/'inputs'/f'{month}.pkl.gz';meta=read(source.with_suffix('.json'))
    if meta['design']!=DESIGN or meta['runner_sha256']!=sha(Path(__file__)) or sha(source)!=meta['packet_sha256']:
        raise ValueError('Prepared input/runner/recipe changed')
    with gzip.open(source,'rb') as stream:d=pickle.load(stream)
    assert list(d['A'])==list(d['B'])==meta['columns'] and d['categoricals']==meta['categoricals']
    assert len(d['A'])==meta['train_rows'] and len(d['B'])==meta['held_rome_rows'] and len(d['ids'])==meta['total_held_rows']
    assert int(np.sum(d['target']))==meta['positives']
    assert len(np.unique(d['train_ids']))==len(d['train_ids']) and len(np.unique(d['ids']))==len(d['ids'])
    assert not np.intersect1d(d['train_ids'],d['ids']).size
    return d,meta

def fit():
    import lightgbm as lgb
    for month in MONTHS:
        d,meta=load_packet(month)
        pred_path=OUT/'predictions'/f'{month}_s42.parquet';record_path=pred_path.with_suffix('.json')
        if record_path.exists():
            record=read(record_path)
            assert record['input_manifest']==meta and record['runner_sha256']==sha(Path(__file__))
            assert sha(pred_path)==record['prediction_sha256'] and sha(ROOT/record['model_path'])==record['model_sha256']
            log(f'{month}: guarded completed fit; skip');continue
        if pred_path.exists():raise ValueError('Unrecorded predictions exist; inspect interrupted fit')
        started=time.monotonic();log(f'{month}: R20+2 fit {len(d["A"]):,} rows,1700 rounds')
        dataset=lgb.Dataset(d['A'],d['target'],categorical_feature=d['categoricals'])
        model=lgb.train(PARAMS,dataset,num_boost_round=1700)
        p=np.zeros(len(d['ids']),float);p[d['use']]=model.predict(d['B'],num_threads=8)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        model_path=OUT/'models'/f'{month}_s42.txt';model_path.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(model_path))
        pred_path.parent.mkdir(parents=True,exist_ok=True);temp=pred_path.with_suffix('.tmp')
        pd.DataFrame(dict(mvt_id=d['ids'],p=p)).to_parquet(temp,index=False);temp.replace(pred_path)
        write(record_path,dict(design=DESIGN,input_manifest=meta,runner_sha256=sha(Path(__file__)),params=PARAMS,
            model_path=str(model_path.relative_to(ROOT)),model_sha256=sha(model_path),prediction_sha256=sha(pred_path),
            hostname=socket.gethostname(),lightgbm_version=lgb.__version__,minutes=(time.monotonic()-started)/60))
        log(f'{month}: saved probabilities/model');del d,model,dataset;gc.collect()

def score():
    sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
    import open_meteo_jan_jul as base
    import hpo_jan_jul as hpo
    import tuning_seasonal as ts
    sc,rome,pool,trees=base.baseline();errors={name:{} for name in ['control',*DESIGN['score_arms']]};rows=[]
    for month in MONTHS:
        path=OUT/'predictions'/f'{month}_s42.parquet';record=read(path.with_suffix('.json'))
        assert record['design']==DESIGN and record['runner_sha256']==sha(Path(__file__)) and sha(path)==record['prediction_sha256']
        f=pd.read_parquet(path);np.testing.assert_array_equal(f.mvt_id,sc.d[month]['ids'])
        probabilities=dict(control=rome[month],own_arrival=f.p.to_numpy(float),half_own_arrival_half_R20=.5*f.p.to_numpy(float)+.5*rome[month])
        for arm,p in probabilities.items():
            errors[arm][month]=np.mean([sc.se(month,seed,nn=pool[month],p=p,
                **{hpo.MEMBER_OF[part]:trees[part][seed][month] for part in trees}) for seed in hpo.SEEDS],axis=0)
        rows.append(pd.DataFrame(dict(mvt_id=f.mvt_id,month=month,day=sc.d[month]['day'],**{arm:errors[arm][month] for arm in errors})))
    scored=pd.concat(rows,ignore_index=True);scores={name:sc.cv(values) for name,values in errors.items()}
    assert abs(scores['control']['cv']-295.7414415741871)<1e-7
    comparisons={arm:ts.paired_bootstrap(scored.day,scored.month,scored.control,scored[arm]) for arm in DESIGN['score_arms']}
    scored.to_parquet(OUT/'scored_errors.parquet',index=False)
    report=dict(design=DESIGN,scores=scores,comparisons=comparisons,machine_caveat='Candidate fit on recorded host; historicalR20 control fit on desktop.')
    write(OUT/'result.json',report);log(json.dumps(report,indent=2))

def check():
    X=pd.DataFrame(dict(category=pd.Categorical(['A','B','A','C','D']),value=[1.,2.,3.,4.,5.]),index=[8,5,4,2,1])
    table=pd.DataFrame(dict(mvt_id=[10,20,30,40,50],ar_prev_arrival_delay_sec=[-61,-60,60,61,np.nan],
        ar_prev_inblock_age_sec=[3600,7200,300,100,np.nan])).set_index('mvt_id')
    ids=np.array([30,20,10,50,40]);result=add_features(X,ids,table)
    np.testing.assert_allclose(result[FEATURES[0]],[1,1,0,np.nan,0],equal_nan=True)
    np.testing.assert_allclose(result[FEATURES[1]],[300,7200,3600,np.nan,100],equal_nan=True)
    pd.testing.assert_frame_equal(result[X.columns],X,check_exact=True)
    assert all(result[c].dtype==np.dtype('float32') for c in FEATURES)
    log('60s boundary, missingness, arbitrary ID order, original values/categories/index andfloat32 checks pass')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['check','prepare','fit','score']);args=parser.parse_args()
    globals()[args.stage]()

if __name__=='__main__':main()
