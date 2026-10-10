"""Fixed stand/month arrival-recorder inputs on original R20, matched laptop fits.

Two inputs only: airline-adjusted arrival-copy residual shrunk with K50, and
eligible arrival support. Preserve original68 inputs/rows/labels/1700 rounds.
The selected ordinary stack retains its fixed half-Cat3 Rome component.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd

import rome_own_arrival_jan_jul as original

ROOT = original.ROOT
sys.path.insert(0, str(ROOT/'VALIDATION/tools'))
import research_stand_arrival_copy_20261010 as features

OUT = ROOT/'VALIDATION/runs/v3_20261010_rome_stand_arrival_jan_jul'
FEATURES = ['handler_stand_arr_copy_residual', 'handler_stand_arr_n']
DESIGN = dict(parent='R20 original68', features=FEATURES, feature_design=features.DESIGN,
    support='known-airline, known-stand arrivals with supplied valid arrival clocks',
    context='training queries: allowed training arrivals; held queries: training plus own held month arrivals',
    original_columns=68, rounds=1700, params=original.PARAMS, seed=42,
    membership='exact original R20 rows/labels/categories',
    arms=['historical', 'matched', 'candidate', 'half'],
    v58='original15 NN and three original tree seeds; replace R20 only',
    selected='fixed half sparse3/half calls15, calls main tree seed42; replace only R20 half of half-Cat3/R20',
    half='half new R20 probability plus half same-laptop original68 R20 probability',
    tuning=False, full_year_fit=False, upload=False,
    exploratory_reused_development_months=True)


def artifact(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=original.sha(path))


def verify(record):
    path=ROOT/record['path']
    if original.sha(path)!=record['sha256']:
        raise ValueError('Artifact changed: '+str(path))
    return path


def identity():
    return dict(design=DESIGN, source=artifact(Path(__file__)),
        helpers=[artifact(Path(m.__file__)) for m in [original,features]],
        airline_parser=artifact(ROOT/'VALIDATION/rome_signals.py'))


def add_features(X, ids, table):
    if X.shape[1]!=68 or len(X)!=len(ids) or not table.index.is_unique:
        raise ValueError('Original68 shape or new feature identity invalid')
    if set(FEATURES)&set(X) or not pd.Index(ids).isin(table.index).all():
        raise ValueError('Feature overlap or missing query ID')
    result=X.copy()
    for name in FEATURES:
        result[name]=table.loc[ids,name].to_numpy(np.float32)
    pd.testing.assert_frame_equal(result[X.columns],X,check_exact=True)
    if list(result)!=list(X)+FEATURES or not result.index.equals(X.index):
        raise ValueError('Original column order/index changed')
    return result


def prepare_scoring_reference():
    """Export only necessary prediction columns; original evidence stays local."""
    sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
    import sparse_track_combo_score as combo
    selected=original.read(combo.OUT/'result.json')
    if original.sha(Path(combo.__file__))!=selected['source_sha256']:
        raise ValueError('Selected scorer source changed')
    for name,digest in selected['inputs'].items():
        if original.sha(ROOT/name)!=digest:
            raise ValueError('Selected combination evidence changed: '+name)
    calls=pd.read_parquet(combo.sequence.CALLS_PRED,columns=['mvt_id','candidate_nn']).set_index('mvt_id')
    sparse=pd.read_parquet(combo.sequence.OUT/'scored_pool3.parquet',columns=['mvt_id','candidate_nn']).set_index('mvt_id')
    errors=pd.read_parquet(combo.OUT/'scored_errors.parquet',columns=['mvt_id','month','half_sparse3_combination'])
    if not calls.index.is_unique or not sparse.index.is_unique or not errors.mvt_id.is_unique:
        raise ValueError('Duplicate selected prediction IDs')
    records={};inputs=[]
    for month in original.MONTHS:
        reference=errors.loc[errors.month.eq(month)]
        ids=reference.mvt_id.to_numpy()
        sparse_members=[]
        for k in combo.sequence.NETS:
            member=pd.read_parquet(combo.sequence.paths(month,k)[0],columns=['mvt_id','base'])
            np.testing.assert_array_equal(member.mvt_id,ids)
            sparse_members.append(member.base.to_numpy(float))
        sparse_mean=np.mean(sparse_members,axis=0)
        np.testing.assert_array_equal(sparse_mean,sparse.loc[ids,'candidate_nn'].to_numpy(float))
        neural=.5*(calls.loc[ids,'candidate_nn'].to_numpy(float)+sparse_mean)
        cat=combo.parent.cat.probabilities(month,ids)
        for seed in combo.parent.cat.SEEDS:
            p=(combo.parent.cat.parent_paths(month)[0] if seed==42 else
                combo.parent.cat.OUT/'predictions'/f'{month}_s{seed}.parquet')
            inputs.extend([artifact(p),artifact(p.with_suffix('.json'))])
        main_path=combo.parent.control.candidate.OUT/'predictions'/f'{month}_s42.parquet'
        main_record=original.read(main_path.with_suffix('.json'))
        if original.sha(main_path)!=main_record['prediction_sha256']:
            raise ValueError('Selected main prediction changed')
        main=pd.read_parquet(main_path);np.testing.assert_array_equal(main.mvt_id,ids)
        path=OUT/'inputs'/f'{month}_score.parquet'
        path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(dict(mvt_id=ids,nn=neural,cat=cat,main=main.pred.to_numpy(float),
            expected_se=reference.half_sparse3_combination.to_numpy(float))).to_parquet(path,index=False)
        records[month]=dict(artifact=artifact(path),rows=len(ids))
    return dict(selected_result=artifact(combo.OUT/'result.json'),
        selected_source=artifact(ROOT/'VALIDATION/sparse_track_combo_score.py'),
        selected_errors=artifact(combo.OUT/'scored_errors.parquet'),
        selected_input_hashes=selected['inputs'],cat_inputs=inputs,views=records,
        expected_cv=selected['scores']['half_sparse3_combination']['cv'],
        verification_scope='Desktop preparation verified complete frozen selected inputs; portable scoring verifies compact packet and exact selected all-row SSE replay')


def prepare():
    import duckdb
    started=time.monotonic()
    current=identity()
    manifest_path=OUT/'prepared.json'
    if manifest_path.exists():
        load_manifest()
        original.log('Guarded prepared stand features already exist');return
    upstream=original.read(features.ROT/'identity.json')
    if original.sha(features.ROT/'identity.parquet')!=upstream['identity_sha256']:
        raise ValueError('Rotation input identity changed')
    if original.sha(ROOT/'VALIDATION/aircraft_rotation.py')!=upstream['source_sha256']:
        raise ValueError('Rotation source changed')
    for record in upstream['movement_sources']:
        if original.sha(Path(record['path']))!=record['sha256']:
            raise ValueError('Original movement source changed')
    con=duckdb.connect(config={'threads':1,'memory_limit':'1GB'})
    con.execute("SET TimeZone='UTC'")
    flight=con.execute("SELECT MVT_ID_mvt mvt_id,FLIGHT_mvt FROM read_parquet(?) WHERE PHASE_mvt='ARR' AND ADES_mvt='LIRF'",
        [str(ROOT/'data/training_2025-*.parquet')]).df()
    arrival=con.execute("SELECT mvt_id,month,stand,abs(epoch(inblock)-epoch(arrival_schedule))<=60 AS copy FROM read_parquet(?) WHERE phase='ARR' AND airport='LIRF' AND inblock IS NOT NULL AND arrival_schedule IS NOT NULL",
        [str(features.ROT/'identity.parquet')]).df()
    query=con.execute("SELECT mvt_id,month,stand FROM read_parquet(?) WHERE phase='DEP' AND airport='LIRF'",
        [str(features.ROT/'identity.parquet')]).df().set_index('mvt_id')
    con.close()
    if not flight.mvt_id.is_unique or not arrival.mvt_id.is_unique or not query.index.is_unique:
        raise ValueError('Duplicate movement IDs')
    flight['airline']=features.airline_parser()(flight.FLIGHT_mvt).to_numpy()
    arrival=arrival.merge(flight[['mvt_id','airline']],on='mvt_id',how='left',validate='one_to_one')
    arrival['copy']=arrival['copy'].astype(float)
    del flight
    result=dict(identity=current, rotation_manifest=artifact(features.ROT/'identity.json'),
        rotation_identity=artifact(features.ROT/'identity.parquet'),
        movement_sources=upstream['movement_sources'], full_current_source_hashes_verified=True, folds={})
    for month in original.MONTHS:
        packet,meta=original.load_packet(month)
        names=meta['exact_original_columns_verified']
        if len(names)!=68 or meta['categoricals']!=packet['categoricals']:
            raise ValueError('Original feature/category contract changed')
        entries={}
        for part,key,ids in [('train','A',packet['train_ids']),('held','B',packet['ids'][packet['use']])]:
            references=features.FOLDS[month]+([month] if part=='held' else [])
            q=query.loc[ids]
            expected=set(features.FOLDS[month]) if part=='train' else {month}
            if set(q.month)!=expected:
                raise ValueError('Query fold membership changed')
            delta,count,coverage=features.features(arrival.loc[arrival.month.isin(references)],q)
            table=pd.DataFrame(dict(mvt_id=ids,**{FEATURES[0]:delta.astype(np.float32),FEATURES[1]:count.astype(np.float32)}))
            original_X=packet[key].loc[:,names]
            appended=add_features(original_X,ids,table.set_index('mvt_id'))
            pd.testing.assert_frame_equal(appended[names],original_X,check_exact=True)
            path=OUT/'inputs'/f'{month}_{part}.parquet'
            path.parent.mkdir(parents=True,exist_ok=True);table.to_parquet(path,index=False)
            entries[part]=dict(artifact=artifact(path),reference_months=references,rows=len(table),
                coverage=coverage,nonmissing={name:int(table[name].notna().sum()) for name in FEATURES})
        result['folds'][month]=dict(packet=artifact(original.OUT/'inputs'/f'{month}.pkl.gz'),
            packet_manifest=artifact(original.OUT/'inputs'/f'{month}.pkl.json'),original_meta=meta,
            columns=names,categoricals=packet['categoricals'],features=entries)
        del packet
        gc.collect()
    result['scoring']=prepare_scoring_reference()
    result['seconds']=time.monotonic()-started
    original.write(manifest_path,result)
    original.log(f'Prepared all four compact feature tables in {result["seconds"]:.2f}s')


def load_manifest():
    manifest=original.read(OUT/'prepared.json')
    if manifest['identity']!=identity():
        raise ValueError('Source/helper/design changed')
    for month,fold in manifest['folds'].items():
        verify(fold['packet']);verify(fold['packet_manifest'])
        for part,entry in fold['features'].items():
            verify(entry['artifact'])
            expected=features.FOLDS[month]+([month] if part=='held' else [])
            if entry['reference_months']!=expected:
                raise ValueError('Arrival scope changed')
    for entry in manifest['scoring']['views'].values():verify(entry['artifact'])
    return manifest


def matrices(month,manifest):
    packet,meta=original.load_packet(month)
    fold=manifest['folds'][month]
    if meta!=fold['original_meta']:
        raise ValueError('Original packet contract changed')
    names=fold['columns']
    frames={}
    for part,key,ids in [('train','A',packet['train_ids']),('held','B',packet['ids'][packet['use']])]:
        table=pd.read_parquet(verify(fold['features'][part]['artifact'])).set_index('mvt_id')
        np.testing.assert_array_equal(table.index,ids)
        frames[key]=add_features(packet[key][names],ids,table)
    return packet,frames,names


def fit():
    import lightgbm as lgb
    manifest=load_manifest()
    binding=artifact(OUT/'prepared.json')
    for month in original.MONTHS:
        packet,frames,names=matrices(month,manifest)
        for arm in ['matched','candidate']:
            path=OUT/'predictions'/f'{month}_{arm}.parquet'
            model_path=OUT/'models'/f'{month}_{arm}.txt'
            record_path=path.with_suffix('.json')
            if record_path.exists():
                record=original.read(record_path)
                if record['identity']!=identity() or record['prepared']!=binding:
                    raise ValueError('Completed fit identity changed')
                verify(record['model']);verify(record['prediction'])
                original.log(f'{month} {arm}: guarded fit already complete');continue
            if path.exists() or model_path.exists():
                raise ValueError('Unrecorded artifacts; inspect rather than repeating updates')
            columns=names+(FEATURES if arm=='candidate' else [])
            A,B=frames['A'][columns],frames['B'][columns]
            started=time.monotonic()
            original.log(f'{month} {arm}: {len(A):,} rows/{len(columns)} inputs/1700 rounds')
            dataset=lgb.Dataset(A,packet['target'],categorical_feature=packet['categoricals'])
            model=lgb.train(original.PARAMS,dataset,num_boost_round=1700)
            p=np.zeros(len(packet['ids']),float)
            p[packet['use']]=model.predict(B,num_threads=8)
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise ValueError('Invalid probabilities')
            model_path.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(model_path))
            path.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(dict(mvt_id=packet['ids'],p=p)).to_parquet(path,index=False)
            original.write(record_path,dict(identity=identity(),prepared=binding,month=month,arm=arm,
                columns=columns,categoricals=packet['categoricals'],train_rows=len(A),
                positives=int(np.sum(packet['target'])),params=original.PARAMS,rounds=1700,
                model=artifact(model_path),prediction=artifact(path),hostname=socket.gethostname(),
                lightgbm_version=lgb.__version__,seconds=time.monotonic()-started,
                verification_scope='portable source/design/packet/feature hashes; desktop prepare verified full original sources'))
            del model,dataset,A,B
            gc.collect()
        del packet,frames
        gc.collect()


def score():
    sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
    import open_meteo_jan_jul as baseline
    import hpo_jan_jul as hpo
    import tuning_seasonal as ts
    manifest=load_manifest()
    sc,rome,pool,trees=baseline.baseline()
    errors={f'{stack}_{arm}':{} for stack in ['v58','selected'] for arm in DESIGN['arms']}
    rows=[];prediction_inputs=[]
    for month in original.MONTHS:
        ids=sc.d[month]['ids']; probabilities={}
        for arm in ['matched','candidate']:
            path=OUT/'predictions'/f'{month}_{arm}.parquet'
            record=original.read(path.with_suffix('.json'))
            if record['identity']!=identity() or record['prepared']!=artifact(OUT/'prepared.json'):
                raise ValueError('Completed fit provenance changed')
            f=pd.read_parquet(verify(record['prediction']))
            np.testing.assert_array_equal(f.mvt_id,ids)
            probabilities[arm]=f.p.to_numpy(float);prediction_inputs.append(artifact(path))
        probabilities['historical']=rome[month]
        probabilities['half']=.5*(probabilities['candidate']+probabilities['matched'])
        for arm,p in probabilities.items():
            errors['v58_'+arm][month]=np.mean([sc.se(month,seed,nn=pool[month],p=p,
                **{hpo.MEMBER_OF[part]:trees[part][seed][month] for part in trees})
                for seed in hpo.SEEDS],axis=0)
        context=pd.read_parquet(verify(manifest['scoring']['views'][month]['artifact']))
        np.testing.assert_array_equal(context.mvt_id,ids)
        neural,cat=context.nn.to_numpy(float),context.cat.to_numpy(float)
        members={hpo.MEMBER_OF[part]:trees[part][42][month] for part in trees}
        members[hpo.MEMBER_OF['lgb']]=context.main.to_numpy(float)
        for arm,p in probabilities.items():
            errors['selected_'+arm][month]=sc.se(month,42,nn=neural,p=.5*cat+.5*p,**members)
        np.testing.assert_array_equal(errors['selected_historical'][month],context.expected_se.to_numpy(float))
        rows.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=sc.d[month]['day'],
            **{name:value[month] for name,value in errors.items()})))
    frame=pd.concat(rows,ignore_index=True);scores={name:sc.cv(value) for name,value in errors.items()}
    if abs(scores['v58_historical']['cv']-295.7414415741871)>1e-7:
        raise ValueError('Historical v58 does not reproduce')
    if scores['selected_historical']['cv']!=manifest['scoring']['expected_cv']:
        raise ValueError('Historical selected CV does not reproduce')
    comparisons={}
    for stack in ['v58','selected']:
        for arm,control in [('matched','historical'),('candidate','matched'),('half','matched'),('candidate','historical'),('half','historical')]:
            comparisons[f'{stack}_{arm}_vs_{control}']=ts.paired_bootstrap(
                frame.day,frame.month,frame[f'{stack}_{control}'],frame[f'{stack}_{arm}'])
    frame.to_parquet(OUT/'scored_errors.parquet',index=False)
    result=dict(identity=identity(),prepared=artifact(OUT/'prepared.json'),scores=scores,comparisons=comparisons,
        selected_reference=manifest['scoring'],prediction_inputs=prediction_inputs,
        interpretation='Exploratory reused Jan/Jul data; matched local controls isolate classifier feature change. Historical replay retains original host differences.')
    original.write(OUT/'result.json',result);original.log(json.dumps(result,indent=2))


def check():
    X=pd.DataFrame({f'x{i}':np.arange(5,dtype=float) for i in range(67)},index=[7,3,8,9,1])
    X['cat']=pd.Categorical(['b','a',None,'b','c'],categories=['c','a','b','unused'])
    table=pd.DataFrame(dict(mvt_id=[10,20,30,40,50],**{FEATURES[0]:[.1,0,np.nan,-.1,.2],FEATURES[1]:[1,0,np.nan,50,100]})).set_index('mvt_id')
    ids=np.array([30,50,20,40,10]);result=add_features(X,ids,table)
    pd.testing.assert_frame_equal(result[X.columns],X,check_exact=True)
    np.testing.assert_allclose(result[FEATURES[0]],[np.nan,.2,0,-.1,.1],equal_nan=True)
    assert all(result[name].dtype==np.dtype('float32') for name in FEATURES)
    arrivals=pd.DataFrame(dict(month=['2025-03']*4,stand=['A','A','B','B'],airline=['AAA','BBB','AAA','BBB'],copy=[1.,0.,0.,1.]))
    q=pd.DataFrame(dict(month=['2025-03']*3,stand=['A','unseen',None]))
    a=features.features(arrivals,q)[:2]
    poison=q.assign(target=1e9,taxi_sec=1e9,BLOCK_TIME_UTC_mvt='2099-01-01')
    b=features.features(arrivals,poison)[:2]
    for left,right in zip(a,b):np.testing.assert_array_equal(left,right)
    np.testing.assert_allclose(a[1],[2,0,np.nan],equal_nan=True)
    original.log('Exact68 categories/order/IDs, float32, missing/zero support and query-label isolation pass')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','prepare','fit','score'])
    globals()[parser.parse_args().stage]()
