"""Reconstruct absent B0/R3 models once, with exact old-vector acceptance.

These two original full-year models are absent; this does not refit retained
models. R3 source is exact. B0 uses its already audited equivalent recovery-off
implementation. Failed replay is retained as an approximation, never promoted.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','6')

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from common import connect, TRAIN_GLOB, CATEGORICAL
from movement_data import load_phase
from records import sha256, write_json, runtime_identity
from registered_rules import apply_rule

OUT=ROOT/'VALIDATION/runs/v3_20261012_final_missing_basic_reconstruct'
REFERENCE=ROOT/'VALIDATION/reports/leaderboard_selected_20260916.json'
SOURCE=ROOT/'VALIDATION/runs/v3_20261011_legacy_basic_jan_jul/source/VALIDATION/modeling.py'
EQUIVALENCE=ROOT/'VALIDATION/reports/stage1_off_equivalence.json'
MAPPINGS={'baseline':[(12,'off'),(13,'rome_6h'),(15,'rome_4h')],
          'recovery':[(14,'rome_6h'),(16,'rome_4h')]}
DESIGN=dict(purpose='Recover absent original B0/R3 full-year models, one fixed reconstruction each',
    rounds=800,seed=42,threads=6,train_rows=2084659,features=27,freeze='2025-12-31',
    source='Exact R3 implementation, with previously audited B0 recovery-off equivalence',
    original_machine='Intel i7-13650HX laptop; current reconstruction on AMD desktop is recorded',
    success='All original integer version vectors exact before any final inference',
    failure='Preserve reconstructed model and mismatch, stop; no seed/thread/library search or silent substitution',
    scope='No CV refits, weights changed, version assigned, upload or original artifact overwritten')


def read(path):return json.loads(Path(path).read_text())
def art(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha256(path))


def freeze():
    assert not (OUT/'identity.json').exists()
    batch=read(REFERENCE);original=read(ROOT/'VALIDATION/runs/v3_20261011_legacy_basic_jan_jul/identity.json')
    assert sha256(SOURCE)==batch['models']['recovery']['model_source']['VALIDATION/modeling.py']
    for rel,h in original['sources'].items():
        assert sha256(SOURCE if rel=='VALIDATION/modeling.py' else ROOT/rel)==h,rel
    assert runtime_identity()['versions']==batch['models']['recovery']['runtime']['versions']
    eq=read(EQUIVALENCE)
    assert eq['current_modeling_sha256']==sha256(SOURCE)
    assert eq['frozen_modeling_sha256']==batch['models']['baseline']['model_source']['VALIDATION/modeling.py']
    paths=[Path(__file__),REFERENCE,SOURCE,EQUIVALENCE,
           ROOT/'VALIDATION/runs/v3_20261011_legacy_basic_jan_jul/identity.json',
           ROOT/'data/derived/scored_submission_cache_20261011/manifest.json',
           ROOT/'VALIDATION/runs/overnight_20260917/progress.json',
           ROOT/'VALIDATION/runs/overnight_20260917/models/refit_2025__control.train_ids.parquet',
           ROOT/'VALIDATION/runs/overnight_20260917/models/refit_2025.categories.json',
           ROOT/'pipeline/registered_rules.py',ROOT/'pipeline/prediction_rules.py',
           ROOT/'VALIDATION/records.py']
    paths += [ROOT/rel for rel in original['sources'] if rel!='VALIDATION/modeling.py']
    for item in batch['models']['recovery']['data_identity']:
        p=ROOT/'data'/item['file'];assert sha256(p)==item['sha256'];paths.append(p)
    paths += [ROOT/'data/final_ranking.parquet',ROOT/'data/final_submitting.parquet',ROOT/'data/submitting.parquet']
    cache=read(ROOT/'data/derived/scored_submission_cache_20261011/manifest.json')
    for mappings in MAPPINGS.values():
        for v,_ in mappings:
            p=ROOT/cache[str(v)]['file'];assert sha256(p)==cache[str(v)]['sha256'];paths.append(p)
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'identity.json',dict(design=DESIGN,sources=[art(p) for p in sorted(set(paths))],
        runtime=runtime_identity(),mappings=MAPPINGS,original_models={k:v['model_sha256'] for k,v in batch['models'].items()}))
    print(json.dumps(dict(stage='frozen',sources=len(set(paths)),fits_started=0)))


def guard():
    identity=read(OUT/'identity.json')
    assert identity['design']==DESIGN
    for rec in identity['sources']:assert sha256(ROOT/rec['path'])==rec['sha256'],rec['path']
    return identity


def preflight(X,y,dep,prepared,family,reference):
    keep=dep.taxi_sec.between(1,200000)
    original=dep.loc[keep]
    working=prepared[0][0] if prepared else dep
    raw=working.loc[keep]
    assert X.index.equals(raw.index) and list(X)==reference['features']
    assert len(X)==DESIGN['train_rows'] and len(X.columns)==27
    np.testing.assert_array_equal(y,original.taxi_sec.to_numpy(float))
    np.testing.assert_array_equal(raw.mvt_id,original.mvt_id)
    oldids_path=ROOT/'VALIDATION/runs/overnight_20260917/models/refit_2025__control.train_ids.parquet'
    oldids=pd.read_parquet(oldids_path)
    np.testing.assert_array_equal(oldids.mvt_id.to_numpy(),original.mvt_id.to_numpy())
    # Independent raw-file membership/label aggregation and rowwise basic
    # feature arithmetic. The nine temporal-history columns retain the exact
    # separately audited source; they are not claimed as a new independent test.
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        frozen=con.execute("SELECT MVT_ID_mvt mvt_id,TAXITIME_SEC_mvt taxi_sec FROM read_parquet(?) WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt BETWEEN 1 AND 200000",[TRAIN_GLOB]).df().set_index('mvt_id')
        assert frozen.index.is_unique and set(frozen.index)==set(original.mvt_id)
        np.testing.assert_array_equal(frozen.loc[original.mvt_id,'taxi_sec'],y)
        basics=X.iloc[:,:18].copy();basics['mvt_id']=raw.mvt_id.to_numpy()
        con.register('features',basics);con.register('raw',raw)
        expected={'airport':'r.ADEP_mvt','dest':'r.ADES_mvt','aircraft_type':'r.AIRCRAFT_TYPE_mvt',
            'runway':"r.ADEP_mvt||':'||r.RUNWAY_mvt",'stand':"r.ADEP_mvt||':'||r.STAND_mvt",
            'flight_rule':'r.FLIGHT_RULE_mvt','market_segment':'r.MARKET_SEGMENT_flt',
            'flight_type':'r.FLIGHT_TYPE_flt','wake_cat':'r.WK_TBL_CAT_flt','operator':'r.AIRCRAFT_OPERATOR_flt',
            'hour':'hour(r.MVT_TIME_UTC_mvt)','dow':'isodow(r.MVT_TIME_UTC_mvt)-1','month':'month(r.MVT_TIME_UTC_mvt)'}
        for name,col in [('sched','SCHED_TIME_UTC_mvt'),('lobt','LOBT_flt'),('iobt','IOBT_flt'),('eobt1','EOBT_1_flt'),('aobt3','AOBT_3_flt')]:
            expected[f'mvt_minus_{name}_sec']=f'epoch(r.MVT_TIME_UTC_mvt-r.{col})'
        checks=' OR '.join(f'f."{key}" IS DISTINCT FROM {expr}' for key,expr in expected.items())
        n,u,bad=con.execute(f'SELECT count(*),count(distinct f.mvt_id),sum(CASE WHEN {checks} THEN 1 ELSE 0 END) FROM features f JOIN raw r USING(mvt_id)').fetchone()
        assert n==u==len(X) and bad==0,(n,u,bad)
    return dict(family=family,rows=len(X),features=list(X),original_order_and_labels_exact=True,
        independent_raw_SQL_membership_and_basic18_features_exact=True,history_columns_source_hash_verified=True,
        history_independent_original_equivalence_report=art(EQUIVALENCE),
        training_ID_sha256=sha256(oldids_path),finite_labels=True,
        recovery_source_exact=(family=='recovery'),baseline_source_equivalence=(family=='baseline'))


def run(family):
    identity=guard();directory=OUT/family;directory.mkdir(exist_ok=True)
    assert not (directory/'result.json').exists(),'Preserve completed family'
    assert not (directory/'model.txt').exists(),'A saved fit must be resumed for inference, never repeated'
    spec=importlib.util.spec_from_file_location('recovered_original_stage1',SOURCE)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    batch=read(REFERENCE);reference=batch['models'][family];config=dict(reference['config'])
    assert config['threads']==6 and reference['seed']==42 and reference['rounds']==800
    started=time.monotonic()
    con=connect();con.execute('SET threads=2')
    dep,arr=load_phase(con,TRAIN_GLOB,'DEP'),load_phase(con,TRAIN_GLOB,'ARR');con.close()
    prepared=[]
    X,y,cats=backend.training_matrix(dep,arr,config,prepared_out=prepared)
    check=preflight(X,y,dep,prepared,family,reference)
    write_json(directory/'preflight.json',dict(check,identity=art(OUT/'identity.json'),runtime=runtime_identity()))
    write_json(directory/'categories.json',{k:v.tolist() for k,v in cats.items()})
    print(json.dumps(dict(stage='preflight_pass',family=family,rows=len(X),threads=6)),flush=True)
    history={}
    booster=backend.fit_booster(X,y,42,800,config,'missing '+family,training_history=history)
    booster.save_model(str(directory/'model.txt'))
    write_json(directory/'training_loss.json',history)
    write_json(directory/'trained.json',dict(identity=art(OUT/'identity.json'),model=art(directory/'model.txt'),
        categories=art(directory/'categories.json'),preflight=art(directory/'preflight.json'),
        same_original_model_hash=sha256(directory/'model.txt')==reference['model_sha256'],runtime=runtime_identity(),
        minutes=(time.monotonic()-started)/60,source=art(SOURCE)))
    del X,y;gc.collect()
    replay=lgb.Booster(model_file=str(directory/'model.txt'))
    cache=read(ROOT/'data/derived/scored_submission_cache_20261011/manifest.json')
    reports={}
    for view in ['old','final']:
        source=ROOT/('data/ranking.parquet' if view=='old' else 'data/final_ranking.parquet')
        template=ROOT/('data/submitting.parquet' if view=='old' else 'data/final_submitting.parquet')
        ids=pq.read_table(template,columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy()
        con=connect();con.execute('SET threads=2')
        rank,rank_arr=load_phase(con,str(source),'DEP'),load_phase(con,str(source),'ARR');con.close()
        assert rank.mvt_id.is_unique and set(rank.mvt_id)==set(ids)
        parts={v:[] for v,_ in MAPPINGS[family]}
        for month in sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m').unique()):
            raw=backend.month_rows(rank,month)
            V=backend.validation_matrix(dep,arr,raw,backend.month_rows(rank_arr,month),DESIGN['freeze'],cats,config,
                **({'prepared_training':prepared[0]} if prepared else {}))
            pred=backend.predict(booster,V,threads=6);reload=backend.predict(replay,V,threads=6)
            np.testing.assert_array_equal(pred,reload)
            for version,rule in MAPPINGS[family]:
                corrected,_=apply_rule(pred,raw,rule)
                parts[version].append(pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(),month=month,
                    pred_raw=pred,pred=corrected,submitted=np.rint(np.maximum(corrected,0)))))
        reports[view]={}
        for version,pieces in parts.items():
            frame=pd.concat(pieces,ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
            assert np.isfinite(frame[['pred_raw','pred','submitted']]).all().all()
            path=directory/f'{view}_v{version}.parquet';frame.to_parquet(path,index=False)
            report=dict(prediction=art(path),rows=len(frame),months=frame.month.value_counts().sort_index().to_dict(),
                all_model_reload_predictions_exact=True,template_order_exact=True)
            if view=='old':
                target=pd.read_parquet(ROOT/cache[str(version)]['file']).set_index('MVT_ID_mvt').loc[ids,'TAXITIME_SEC_mvt'].to_numpy()
                difference=frame.submitted.to_numpy()-target
                report.update(exact_original_integer_replay=bool(np.all(difference==0)),
                    changed_rows=int(np.count_nonzero(difference)),rms_difference=float(np.sqrt(np.mean(difference**2))),
                    maximum_absolute_difference=float(np.max(abs(difference))))
            reports[view][str(version)]=report
        write_json(directory/f'{view}.json',reports[view])
        del rank,rank_arr,parts;gc.collect()
        if view=='old' and not all(x['exact_original_integer_replay'] for x in reports[view].values()):
            write_json(directory/'result.json',dict(state='closed_approximation_old_replay_failed',family=family,
                identity=art(OUT/'identity.json'),views=reports,no_final_inference=True,no_upload=True,
                no_thread_seed_library_sweep=True))
            print(json.dumps(dict(family=family,state='closed_approximation_old_replay_failed',old=reports[view])),flush=True)
            return
    write_json(directory/'result.json',dict(state='exact_old_replay_final_inference_complete',family=family,
        identity=art(OUT/'identity.json'),views=reports,original_model_hash_exact=sha256(directory/'model.txt')==reference['model_sha256'],
        no_upload=True,independent_final_audit_pending=True))
    print(json.dumps(dict(family=family,state='exact_old_replay_final_inference_complete',minutes=(time.monotonic()-started)/60)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run']);parser.add_argument('--family',choices=list(MAPPINGS))
    args=parser.parse_args()
    with threadpool_limits(6):
        if args.stage=='freeze':freeze()
        else:
            assert args.family
            run(args.family)
