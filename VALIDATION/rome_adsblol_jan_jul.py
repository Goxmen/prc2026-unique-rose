"""Frozen R20 + 22 existing ADSB.lol inputs, scored inside v58; no uploads.

Only the Rome recording classifier changes. Existing OPDI inputs are retained;
ADSB.lol's 14 original and eight stand inputs are added without duplication.
Classifier rows, labels, categories, R20 settings, 1,700 rounds and seed 42 match
the existing HPO classifier. No checkpoint selection. All held-out departures
are scored with saved v58 ordinary parts, corrections and network share 0.71.

Run prepare, then fit when the desktop's sole heavy-job slot is available, then
score. The prepare stage does no fitting and caches small Rome-only matrices.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline')]
import adsblol
import hpo_jan_jul as hpo
import open_meteo_jan_jul as baseline
import rome_hybrid
import rome_signals as signals
import tuning_seasonal as ts
from records import sha256, write_json

OUT = ts.RUNS/'v3_20261010_rome_adsblol_jan_jul'
ARM = 'adsblol'
NEW = adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES
DESIGN = dict(parent='v58',part='Rome recording classifier',new_features=NEW,
    retained_classifier_extras=list(ts.CLASSIFIER),seed=42,rounds=1700,trial='R20',
    network_share=.71,heldout_months=ts.SPLITS,training_membership='unchanged after_drop',
    checkpoint_selection=False,cov_day_added=False,exploratory_reused_development_months=True)


def log(value):
    print(time.strftime('[%H:%M:%S] ') + str(value),flush=True)


def ensure_design():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'design.json'
    if path.exists() and ts.read(path)!=DESIGN:
        raise ValueError('Existing design differs from frozen screen')
    write_json(path,DESIGN)


def prepare():
    if all((OUT/'inputs'/f'{month}.pkl').exists() for month in ts.SPLITS):
        log('All prepared matrices exist; skip')
        return
    assert len(NEW)==22 and not set(NEW).intersection(ts.CLASSIFIER)
    # Preserve the original load_phase departure order and verify membership below.
    con=duckdb.connect(config={'threads':2})
    con.execute("SET TimeZone='UTC'")
    raw_all=con.execute("SELECT MVT_ID_mvt AS mvt_id,TAXITIME_SEC_mvt AS taxi_sec,"
        "ADEP_mvt,MVT_TIME_UTC_mvt,SCHED_TIME_UTC_mvt FROM read_parquet(?) WHERE PHASE_mvt='DEP'",
        [str(ROOT/'data/training_*.parquet')]).df()
    con.close()
    assert raw_all.mvt_id.notna().all() and not raw_all.mvt_id.duplicated().any()
    table=adsblol.load_features().join(adsblol.load_features(adsblol.STAND_PATH),how='left')
    (OUT/'inputs').mkdir(exist_ok=True)
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [adsblol.FEATURES_PATH,adsblol.STAND_PATH]}
    for month in ts.SPLITS:
        path=OUT/'inputs'/f'{month}.pkl'
        if path.exists():
            continue
        src=ts.split_dir(hpo.SPLITS_FROM,month)
        train,held=pd.read_pickle(src/'after_drop.pkl'),pd.read_pickle(src/'heldout.pkl')
        lirf=train['airport']=='LIRF'
        X,y=train['X'].loc[lirf],np.asarray(train['y'],float)[lirf]
        raw=raw_all.loc[X.index]
        assert raw.ADEP_mvt.eq('LIRF').all()
        np.testing.assert_array_equal(raw.taxi_sec.to_numpy(),y)
        gap=(raw.MVT_TIME_UTC_mvt-raw.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
        np.testing.assert_allclose(gap,X[rome_hybrid.GAP].to_numpy(),equal_nan=True)
        assert set(raw.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m')).issubset(ts.FOLDS[month])
        target=rome_hybrid.substituted(X,y,raw).astype(float)
        A=signals.classifier_matrix(X,raw,ts.KIND,ts.CLASSIFIER)
        A=adsblol.add_adsblol(A,raw,table,NEW)
        vr,V=held['raw'],held['V']
        use=vr.ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(V[rome_hybrid.GAP].to_numpy(float))
        B=signals.classifier_matrix(V.loc[use],vr.loc[use],ts.KIND,ts.CLASSIFIER)
        B=adsblol.add_adsblol(B,vr.loc[use],table,NEW)
        assert list(A)==list(B)
        assert not np.intersect1d(raw.mvt_id.to_numpy(),vr.mvt_id.to_numpy()).size
        temp=path.with_suffix('.tmp')
        pd.to_pickle(dict(A=A,B=B,target=target,use=use,ids=vr.mvt_id.to_numpy(),train_ids=raw.mvt_id.to_numpy()),temp)
        temp.replace(path)
        write_json(path.with_suffix('.json'),dict(design=DESIGN,columns=list(A),rows=len(A),
            held_rows=len(vr),held_rome_rows=int(use.sum()),positives=int(target.sum()),
            feature_sources=sources,prepared_sha256=sha256(path),
            training_ground_share=float(A.ab_seg_sec.notna().mean()),
            held_ground_share=float(B.ab_seg_sec.notna().mean()),
            source_sha256=sha256(Path(__file__))))
        log(f'{month}: prepared {len(A):,} train / {len(B):,} held Rome rows')
        del train,held,X,A,B,V,vr
        gc.collect()


def fit(threads):
    spec=hpo.trial_spec('rome','R20')
    assert int(hpo.winner('rome')['checkpoint'][1:])==DESIGN['rounds']
    params=dict(rome_hybrid.CLASSIFIER_PARAMS,**spec['overrides'],seed=42,
                bagging_seed=43,feature_fraction_seed=44,num_threads=threads)
    (OUT/'predictions').mkdir(exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    for month in ts.SPLITS:
        path=OUT/'predictions'/f'{month}_s42.parquet'
        if path.exists():
            log(f'{month}: saved probability already exists; skip')
            continue
        d=pd.read_pickle(OUT/'inputs'/f'{month}.pkl')
        started=time.monotonic()
        model=signals._train(params,d['A'],d['target'],DESIGN['rounds'],f'Rome ADSB {month}')
        pred=np.zeros(len(d['ids']))
        pred[d['use']]=model.predict(d['B'],num_threads=threads)
        assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
        mp=OUT/'models'/f'{month}_s42.txt'
        model.save_model(str(mp))
        temp=path.with_suffix('.tmp')
        pd.DataFrame(dict(mvt_id=d['ids'],p=pred)).to_parquet(temp,index=False)
        temp.replace(path)
        write_json(path.with_suffix('.json'),dict(design=DESIGN,params=params,
            minutes=(time.monotonic()-started)/60,model_sha256=sha256(mp),
            input_sha256=sha256(OUT/'inputs'/f'{month}.pkl'),prediction_sha256=sha256(path),
            source_sha256=sha256(Path(__file__))))
        del d,model
        gc.collect()


def score():
    sc,oldp,nn,trees=baseline.baseline()
    errors={arm:{} for arm in ['control',ARM]}
    rows=[]
    for month in ts.SPLITS:
        candidate=pd.read_parquet(OUT/'predictions'/f'{month}_s42.parquet')
        np.testing.assert_array_equal(candidate.mvt_id.to_numpy(),sc.d[month]['ids'])
        newp=candidate.p.to_numpy(float)
        for arm,p in [('control',oldp[month]),(ARM,newp)]:
            errors[arm][month]=np.mean([sc.se(month,seed,nn=nn[month],p=p,
                **{hpo.MEMBER_OF[part]:trees[part][seed][month] for part in trees}) for seed in hpo.SEEDS],axis=0)
        d=sc.d[month]
        base=.71*nn[month]+.29*np.mean([trees[part][42][month] for part in trees],axis=0)
        control=sc.final(month,base,p=oldp[month])
        new=sc.final(month,base,p=newp)
        rows.append(pd.DataFrame(dict(mvt_id=d['ids'],month=month,day=d['day'],
            airport=d['held']['raw'].ADEP_mvt.to_numpy(),y=d['y'],p_control=oldp[month],p_candidate=newp,
            pred_control=control,pred_candidate=new,se_control=errors['control'][month],se_candidate=errors[ARM][month])))
    f=pd.concat(rows,ignore_index=True)
    control=sc.cv(errors['control'])
    if abs(control['cv']-295.7414415741871)>1e-7:
        raise ValueError('v58 control did not reproduce')
    change=ts.paired_bootstrap(f.day.to_numpy(),f.month.to_numpy(),f.se_control.to_numpy(),f.se_candidate.to_numpy())
    result=dict(design=DESIGN,control=control,candidate=sc.cv(errors[ARM]),change=change,
        largest_changes=f.assign(abs_change=(f.pred_candidate-f.pred_control).abs()).nlargest(25,'abs_change').to_dict('records'))
    f.to_parquet(OUT/'scored_rows.parquet',index=False)
    write_json(OUT/'result.json',result)
    log(json.dumps({k:v for k,v in result.items() if k!='largest_changes'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','fit','score'])
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.threads<1:
        parser.error('--threads must be positive')
    ensure_design()
    with threadpool_limits(2):
        if args.stage=='prepare': prepare()
        elif args.stage=='fit': fit(args.threads)
        else: score()
