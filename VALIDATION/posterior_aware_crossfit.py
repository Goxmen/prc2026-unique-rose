"""One fixed six-prior CPU experiment with honest frozen peer messages.

Stages are separately resumable. Input preparation is label-blind except for
copying raw labels into separate month files. Calibration and each model fit
read only their declared training files; message generation reads no labels.
Outer held labels are first read by score(), after all tilt coefficients exist.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

import posterior_aware_learning as model
from records import sha256,write_json,runtime_identity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'VALIDATION/runs/v3_20261013_posterior_aware_crossfit'
PLAN = ROOT/'VALIDATION/reports/cohort_learning_20261013/posterior_aware_protocol.json'
PARENT = model.gate.PARENT
CURRENT = ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
V80 = ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_candidate/submission_v80/manifest.json'
BANK = ROOT/'data/derived/flight_cohort_context_20261012/train'
INPUTS = OUT/'inputs'
MONTHS = ['2025-01','2025-07']
HALVES = {
    '2025-01': [['2025-03','2025-05','2025-07','2025-09','2025-11'],
                ['2025-04','2025-06','2025-08','2025-10']],
    '2025-07': [['2025-01','2025-03','2025-10','2025-12'],
                ['2025-02','2025-09','2025-11']]}
BASE_SETTINGS = dict(objective='regression',learning_rate=.05,num_leaves=255,
    min_data_in_leaf=200,feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,
    seed=42,num_threads=8,verbosity=-1)
ROUNDS = 800
CAT = ['airport','dest','aircraft_type','runway','stand','flight_rule',
       'market_segment','flight_type','wake_cat','operator','airline']
FEATURES = CAT[:-1]+['hour','dow','month']+[
    'mvt_minus_'+c+'_sec' for c in ['sched','lobt','iobt','eobt1','aobt3']]+[
    'physical_'+str(i) for i in range(17)]+['missing_flight_id','airline']
META_COLUMNS = ['mvt_id','airport','month','day','takeoff_epoch','gap','has_id','nm_gap',
                'fltrule','runway','local_day','eligible','quality']


def read(path): return json.loads(Path(path).read_text())
def art(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))
def verify(record):
    path=ROOT/record['path']
    assert sha256(path)==record['sha256'],path
    return path


def receipt(path,arm=None,outer=None):
    value=read(path)
    assert value['identity']==art(OUT/'identity.json'),path
    if arm is not None: assert value['arm']==arm,path
    if outer is not None: assert value['outer']==outer,path
    for key in ['artifacts','optimizers','dependencies']:
        for record in value.get(key,[]): verify(record)
    return value


def arms():
    result=[]
    for outer,(a,b) in HALVES.items():
        allowed=sorted(a+b)
        assert allowed==model.gate.original.prior.base.FOLDS[outer]
        assert not set(a)&set(b) and outer not in allowed
        for name,train,held in [('A',b,a),('B',a,b),('outer',allowed,[outer])]:
            result.append(dict(key=outer+'_'+name,outer=outer,part=name,train=sorted(train),held=sorted(held)))
    return result


def source_paths():
    names=['posterior_aware_crossfit','posterior_aware_learning','posterior_aware_sensitivity',
           'joint_public_moments','daily_delay_posterior','daily_delay_reconcile',
           'daily_offblock_alignment','taxi_count_sum_jan_jul','public_daily_taxi',
           'monthly_taxi_reconcile','records','tuning_seasonal']
    return [ROOT/'VALIDATION'/f'{name}.py' for name in names]+[PLAN,ROOT/'pipeline/common.py']


def prepare():
    """Bounded desktop-only projection into a small immutable portable cache."""
    assert not (OUT/'prepared.json').exists(),'Preserve prepared inputs'
    assert not INPUTS.exists(),'Partial input preparation requires inspection'
    started=time.monotonic()
    import sys
    sys.path.insert(0,str(ROOT))
    from pipeline import common
    import monthly_taxi_reconcile as monthly
    model.gate.historical_inputs()
    paths=sorted((ROOT/'data').glob('training_2025-*.parquet'))
    assert len(paths)==12
    columns=(['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','TAXITIME_SEC_mvt']+
        [s.strip() for s in common.RAW_COLS.replace('\n','').split(',')])
    frames=[]
    for path in paths:
        frames.append(pd.read_parquet(path,columns=columns,
            filters=[('PHASE_mvt','==','DEP'),('ADEP_mvt','==','LIRF')]))
    raw=pd.concat(frames,ignore_index=True)
    assert raw.MVT_ID_mvt.is_unique
    # Flat donor arrays contain only observed physical inputs, never y.
    bank=read(BANK/'manifest.json')
    bank_artifacts={Path(x['path']).name:x for x in bank['outputs']}
    for name in ['meta.npy','values.npy']:
        verify(bank_artifacts[name])
    meta=np.load(BANK/'meta.npy',mmap_mode='r')
    values=np.load(BANK/'values.npy',mmap_mode='r')
    ix=pd.Index(meta[:,0]).get_indexer(raw.MVT_ID_mvt.to_numpy())
    assert (ix>=0).all() and len(np.unique(ix))==len(ix)
    physical=np.asarray(values[ix],np.float32).copy()
    assert physical.shape==(len(raw),15) and np.isfinite(physical).all()
    gap=(raw.MVT_TIME_UTC_mvt-raw.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    assert np.isfinite(gap).all()
    # Exact fixed transform from daily_cohort_bank.transform, no torch import.
    physical[:,:4]=np.arcsinh(physical[:,:4]/3600)
    physical[:,9:11]=np.clip(physical[:,9:11]/2000,-10,10)
    physical=np.column_stack([physical,np.clip(np.arcsinh(gap/7200),-10,10),np.ones(len(gap))]).astype(np.float32)
    x=common.build_features(raw)
    for k in range(17): x['physical_'+str(k)]=physical[:,k]
    x['missing_flight_id']=raw.FLIGHT_ID_mvt.isna().astype(np.float32)
    x['airline']=raw.FLIGHT_mvt.str[:3]
    assert list(x)==FEATURES and len(FEATURES)==37
    x.insert(0,'mvt_id',raw.MVT_ID_mvt.to_numpy())
    f=pd.DataFrame(dict(mvt_id=raw.MVT_ID_mvt.to_numpy(),airport='LIRF',
        month=raw.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m'),day=raw.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d'),
        takeoff_epoch=(raw.MVT_TIME_UTC_mvt-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds(),
        gap=gap,has_id=raw.FLIGHT_ID_mvt.notna(),
        nm_gap=(raw.MVT_TIME_UTC_mvt-raw.LOBT_flt).dt.total_seconds(),fltrule=raw.FLIGHT_RULE_mvt,
        runway=raw.RUNWAY_mvt,local_day=raw.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome').dt.strftime('%Y-%m-%d'),
        eligible=raw.STAND_mvt.notna()&raw.RUNWAY_mvt.notna()))
    quality=monthly.public(2025).set_index(['airport','local_month']).invalid
    keys=pd.MultiIndex.from_arrays([f.airport,f.local_day.str[:7]])
    f['quality']=quality.reindex(keys).to_numpy()<.05
    assert list(f)==META_COLUMNS
    # Cross-month 120-second same-aircraft aliases would invalidate complete-
    # month honesty. Fail rather than silently purging a source cohort.
    aliases=pd.DataFrame(dict(hex=meta[ix,5],time=meta[ix,1],month=f.month,mvt_id=f.mvt_id))
    aliases=aliases[aliases.hex.ge(0)].sort_values(['hex','time','mvt_id'])
    prev=aliases.shift()
    collision=aliases.hex.eq(prev.hex)&aliases.time.sub(prev.time).abs().le(120000)&aliases.month.ne(prev.month)
    assert not collision.any(), 'Cross-month same-aircraft alias; fixed protocol cannot proceed'
    INPUTS.mkdir(parents=True)
    x.to_parquet(INPUTS/'features.parquet',index=False)
    f.to_parquet(INPUTS/'rows.parquet',index=False)
    for month,g in f.groupby('month',sort=True):
        pd.DataFrame(dict(mvt_id=g.mvt_id,y=raw.TAXITIME_SEC_mvt.iloc[g.index].to_numpy(float))).to_parquet(
            INPUTS/f'labels_{month}.parquet',index=False)
    for name in ['taxi_source.parquet','delay_source.parquet']:
        pd.read_parquet(PARENT/name).to_parquet(INPUTS/name,index=False)
    params=read(PARENT/'parameters.json')
    for outer in MONTHS:
        assert params[outer]['taxi']['allowed']==model.gate.original.prior.base.FOLDS[outer]
    write_json(INPUTS/'outer_parameters.json',params)
    pd.read_parquet(PARENT/'inputs.parquet').to_parquet(INPUTS/'old_inputs.parquet',index=False)
    pd.read_parquet(PARENT/'predictions.parquet').to_parquet(INPUTS/'old_predictions.parquet',index=False)
    # Provenance is retained on the desktop; remote stages consume only compact
    # projections and never require downloading the entire raw/bank cache.
    paths += [BANK/'manifest.json',BANK/'meta.npy',BANK/'values.npy',
        ROOT/'VALIDATION/daily_cohort_bank.py',PARENT/'identity.json',PARENT/'parameters.json',
        PARENT/'input_audit.json',PARENT/'inference.json',PARENT/'inputs.parquet',PARENT/'predictions.parquet',
        PARENT/'taxi_source.parquet',PARENT/'delay_source.parquet',
        model.gate.original.prior.base.SOURCE/'taxi_out_additional_time_2025.csv']
    outputs=[art(p) for p in sorted(INPUTS.glob('*'))]
    write_json(OUT/'prepared.json',dict(raw_sources=[art(p) for p in paths],outputs=outputs,
        source=art(__file__),feature_order=FEATURES,rows=len(f),months=f.groupby('month').size().to_dict(),
        raw_peers_include_unscored_rows=True,cross_month_aircraft_aliases=0,
        elapsed_seconds=time.monotonic()-started,peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        portable_bytes=sum((ROOT/a['path']).stat().st_size for a in outputs)))
    print({k:read(OUT/'prepared.json')[k] for k in ['rows','months','elapsed_seconds','peak_MiB','portable_bytes']},flush=True)


def freeze():
    assert not (OUT/'identity.json').exists()
    prepared=read(OUT/'prepared.json')
    for a in prepared['outputs']: verify(a)
    plan=read(PLAN)
    for m in MONTHS:
        assert HALVES[m]==[plan['six_priors'][m]['inner_A'],plan['six_priors'][m]['inner_B']]
    assert plan['learning']['loss'].find('0.001')>=0
    snapshot=OUT/'preparation_source.py'
    assert sha256(snapshot)==prepared['source']['sha256']
    import ast,inspect
    saved=next(n for n in ast.parse(snapshot.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    current=ast.parse(inspect.getsource(prepare)).body[0]
    assert ast.dump(saved)==ast.dump(current),'Input preparation changed since projection'
    official=read(V80)
    assert official['version']==80 and official['jan_jul_cv']==289.7627782102906
    verify(official['cv_record']);verify(official['audit'])
    assert official['cv_record']==art(CURRENT/'result.json')
    assert read(ROOT/official['audit']['path'])['status']=='pass'
    result=read(CURRENT/'result.json')
    verify(result['scored']);verify(result['identity'])
    assert result['scored']==art(CURRENT/'scored.parquet')
    score_paths=[V80,CURRENT/'scored.parquet',CURRENT/'result.json',CURRENT/'identity.json',
                 ROOT/official['audit']['path']]
    record=dict(plan=art(PLAN),sources=[art(p) for p in source_paths()],inputs=prepared['outputs'],
        prepared=art(OUT/'prepared.json'),preparation_source=art(snapshot),score_inputs=[art(p) for p in score_paths],
        arms=arms(),features=FEATURES,
        base_settings=BASE_SETTINGS,rounds=ROUNDS,checks=model.checks())
    write_json(OUT/'identity.json',record)
    print(dict(identity=art(OUT/'identity.json'),arms=len(record['arms'])),flush=True)


def guard():
    frozen=read(OUT/'identity.json')
    assert frozen['arms']==arms() and frozen['features']==FEATURES
    assert frozen['base_settings']==BASE_SETTINGS and frozen['rounds']==ROUNDS
    for a in frozen['sources']+frozen['inputs']+frozen['score_inputs']+[
        frozen['prepared'],frozen['preparation_source']]: verify(a)
    return frozen


def labels(months,ids=None):
    frames=[pd.read_parquet(INPUTS/f'labels_{m}.parquet') for m in months]
    f=pd.concat(frames,ignore_index=True)
    assert f.mvt_id.is_unique
    if ids is not None:
        f=f.set_index('mvt_id').loc[np.asarray(ids)].reset_index()
    return f


def calibrate():
    guard()
    import daily_offblock_alignment as aligned
    import taxi_count_sum_jan_jul as taxi
    rows=pd.read_parquet(INPUTS/'rows.parquet')
    st=pd.read_parquet(INPUTS/'taxi_source.parquet').assign(airport='LIRF')
    sd=pd.read_parquet(INPUTS/'delay_source.parquet').assign(airport='LIRF')
    for arm in arms():
        dest=OUT/'arms'/arm['key'];dest.mkdir(parents=True,exist_ok=True)
        path=dest/'calibration.json'
        if path.exists():
            receipt(path,arm=arm)
            continue
        if arm['part']=='outer':
            par=read(INPUTS/'outer_parameters.json')[arm['outer']]
            assert par['taxi']['allowed']==arm['train']
            count=dict(retained_original_outer=True,allowed=arm['train'],
                       cells=par['taxi']['cells'],covariance_cells=par['taxi']['covariance_cells'])
        else:
            tr=rows[rows.month.isin(arm['train'])].copy()
            tr=tr.merge(labels(arm['train']),on='mvt_id',validate='one_to_one')
            assert not set(tr.month)&set(arm['held']+[arm['outer']])
            assert np.isfinite(tr.y).all()
            support=aligned.posterior.fit(tr,sd)['LIRF']
            canonical=tr[~tr.fltrule.eq('V')].copy()
            canonical['day']=aligned.dates(canonical.takeoff_epoch,canonical.y)
            keep=np.zeros(len(canonical),bool)
            for month in arm['train']: keep|=aligned.interior(canonical.day,month)
            canonical=canonical.loc[keep].copy()
            delay=aligned.posterior.fit(canonical,sd)['LIRF']
            tp,cells=taxi.calibrate(tr,st,arm['train'])
            assert tp['cells']>=100 and tp['covariance_cells']>=100
            par=dict(taxi=tp,delay=delay,support=support)
            cells.to_parquet(dest/'calibration_cells.parquet',index=False)
            tr[['mvt_id','month']].to_parquet(dest/'calibration_ids.parquet',index=False)
            canonical[['mvt_id','month','day']].to_parquet(dest/'offblock_calibration_ids.parquet',index=False)
            count=dict(retained_original_outer=False,allowed=arm['train'],training_rows=len(tr),
                canonical_rows=len(canonical),cells=tp['cells'],covariance_cells=tp['covariance_cells'])
        write_json(dest/'parameters.json',par)
        artifacts=[art(p) for p in sorted(dest.glob('*')) if p.suffix in ['.parquet','.json']]
        write_json(path,dict(identity=art(OUT/'identity.json'),arm=arm,counts=count,artifacts=artifacts))
        print(dict(arm=arm['key'],**count),flush=True)


def input_rows(meta,pred,held):
    """Original input-only assignment with complete predicted local-day cohorts."""
    import daily_offblock_alignment as aligned
    import public_daily_taxi as taxi
    assert set(meta.month)==set(held)
    assigned=aligned.dates(meta.takeoff_epoch,pred)
    scope=np.zeros(len(meta),bool)
    for m in held: scope|=meta.month.eq(m).to_numpy()&aligned.interior(assigned,m)
    complete=taxi.complete_days(meta,held)
    return pd.DataFrame(dict(mvt_id=meta.mvt_id,month=meta.month,gap=meta.gap,prior=pred,
        eligible=meta.eligible,quality=meta.quality&complete,runway=meta.runway,local_day=meta.local_day,
        assigned_day=assigned,fltrule=meta.fltrule,delay_scope=scope))


def preflight():
    guard();assert not (OUT/'preflight.json').exists()
    for arm in arms(): receipt(OUT/'arms'/arm['key']/'calibration.json',arm=arm)
    x=pd.read_parquet(INPUTS/'features.parquet');meta=pd.read_parquet(INPUTS/'rows.parquet')
    np.testing.assert_array_equal(x.mvt_id,meta.mvt_id)
    assert list(x)==['mvt_id']+FEATURES and not {'y','true_day','BLOCK_TIME_UTC_mvt'}&set(x)
    st=pd.read_parquet(INPUTS/'taxi_source.parquet');sd=pd.read_parquet(INPUTS/'delay_source.parquet')
    rows=pd.read_parquet(INPUTS/'old_inputs.parquet');saved=pd.read_parquet(INPUTS/'old_predictions.parquet')
    par=read(INPUTS/'outer_parameters.json');reports=[]
    for month in MONTHS:
        r=rows[rows.month.eq(month)].reset_index(drop=True)
        packet=model.messages(r,st,sd,par[month])
        pred=model.predictions(r,packet,np.zeros(4))
        old=saved.set_index('mvt_id').loc[r.mvt_id]
        np.testing.assert_array_equal(pred.touched,old.touched)
        for col in ['dep_only','joint','increment']:
            np.testing.assert_allclose(pred[col],old[col],atol=1e-8,rtol=0)
        poisoned=model.messages(r.assign(y=1e12,true_day='1900-01-01'),st,sd,par[month])
        for k in packet: np.testing.assert_array_equal(packet[k],poisoned[k])
        report=model.diagnostics(packet)
        assert report['rows']>0 and report['incremental_derivative_rank']==4
        report.update(month=month,old_output_replay=True,poisoned_labels_invariant=True)
        reports.append(report)
    write_json(OUT/'preflight.json',dict(status='pass',identity=art(OUT/'identity.json'),
        source=art(__file__),synthetic=model.checks(),historical_incremental_diagnostics=reports,
        calibrations=[art(OUT/'arms'/a['key']/'calibration.json') for a in arms()],
        raw_peer_counts=meta.groupby('month').size().to_dict(),historical_peer_counts=rows.groupby('month').size().to_dict(),
        no_model_or_tilt_fit=True,runtime=runtime_identity()))
    print(dict(status='pass',reports=reports),flush=True)


def fit_priors():
    guard()
    assert read(OUT/'preflight.json')['status']=='pass'
    assert read(OUT/'preflight.json')['identity']==art(OUT/'identity.json')
    # Independent review/SQL/calibration audit is a hard pre-fit gate.
    audit=read(OUT/'prefit_audit.json')
    assert audit['status']=='pass' and audit['identity']==art(OUT/'identity.json')
    assert audit['preflight']==art(OUT/'preflight.json')
    calibrations=[art(OUT/'arms'/a['key']/'calibration.json') for a in arms()]
    assert audit['calibrations']==calibrations
    for arm in arms(): receipt(OUT/'arms'/arm['key']/'calibration.json',arm=arm)
    import lightgbm as lgb
    x=pd.read_parquet(INPUTS/'features.parquet');meta=pd.read_parquet(INPUTS/'rows.parquet')
    np.testing.assert_array_equal(x.mvt_id,meta.mvt_id)
    for arm in arms():
        dest=OUT/'arms'/arm['key'];path=dest/'prior.json'
        if path.exists():
            receipt(path,arm=arm)
            continue
        assert not (dest/'model.txt').exists(),'Saved model without receipt needs inference-only recovery; never refit'
        started=time.monotonic()
        tr=np.flatnonzero(meta.month.isin(arm['train']))
        test=np.flatnonzero(meta.month.isin(arm['held']))
        y=labels(arm['train'],meta.mvt_id.iloc[tr]).y.to_numpy(float)
        keep=np.isfinite(y)&(y>=1)&(y<=200000)
        tr=tr[keep];y=y[keep]
        train=x.iloc[tr][FEATURES].copy();held=x.iloc[test][FEATURES].copy()
        categories={}
        for col in CAT:
            categories[col]=sorted(train[col].dropna().astype(str).unique().tolist())
            train[col]=pd.Categorical(train[col],categories=categories[col])
            held[col]=pd.Categorical(held[col],categories=categories[col])
        write_json(dest/'categories.json',categories)
        meta.iloc[tr][['mvt_id','month']].to_parquet(dest/'prior_training_ids.parquet',index=False)
        assert not set(meta.month.iloc[tr])&set(arm['held']+[arm['outer']])
        dataset=lgb.Dataset(train,label=y,categorical_feature=CAT,free_raw_data=True)
        with threadpool_limits(8):
            booster=lgb.train(BASE_SETTINGS,dataset,num_boost_round=ROUNDS)
            # Save immediately; a subsequent failure must not cause a refit.
            booster.save_model(str(dest/'model.txt'))
            p=np.maximum(booster.predict(held,num_threads=8),0.)
            reload=np.maximum(lgb.Booster(model_file=str(dest/'model.txt')).predict(held,num_threads=8),0.)
        np.testing.assert_array_equal(p,reload)
        assert np.isfinite(p).all()
        pred=meta.iloc[test][['mvt_id','month']].copy();pred['prior']=p
        pred.to_parquet(dest/'prior.parquet',index=False)
        write_json(path,dict(identity=art(OUT/'identity.json'),arm=arm,training_rows=len(train),
            predicted_raw_peers=len(held),all_row_reload_exact=True,seconds=time.monotonic()-started,
            dependencies=[art(OUT/'prefit_audit.json'),art(OUT/'preflight.json'),art(dest/'calibration.json')],
            runtime=runtime_identity(),artifacts=[art(dest/name) for name in ['model.txt','categories.json',
                'prior_training_ids.parquet','prior.parquet']]))
        print(dict(arm=arm['key'],training_rows=len(train),raw_peers=len(held),seconds=time.monotonic()-started),flush=True)


def packetize():
    guard()
    # No label() call or label-bearing frame is allowed in this stage.
    meta=pd.read_parquet(INPUTS/'rows.parquet').set_index('mvt_id')
    st=pd.read_parquet(INPUTS/'taxi_source.parquet');sd=pd.read_parquet(INPUTS/'delay_source.parquet')
    for arm in arms():
        dest=OUT/'arms'/arm['key'];path=dest/'packet.json'
        if path.exists():
            receipt(path,arm=arm)
            continue
        receipt(dest/'prior.json',arm=arm)
        receipt(dest/'calibration.json',arm=arm)
        pred=pd.read_parquet(dest/'prior.parquet')
        r=meta.loc[pred.mvt_id].reset_index()
        np.testing.assert_array_equal(r.month,pred.month)
        inp=input_rows(r,pred.prior.to_numpy(float),arm['held'])
        par=read(dest/'parameters.json')
        packet=model.messages(inp,st,sd,par)
        diagnostics=model.diagnostics(packet)
        # Numerical-rank check is feasibility only; no threshold tuning.
        assert diagnostics['rows']>0 and diagnostics['incremental_derivative_rank']==4
        zero=model.predictions(inp,packet,np.zeros(4))
        old,logs=model.gate.original.apply(inp,st,sd,par['taxi'],par['delay'],par['support'])
        np.testing.assert_array_equal(zero.touched,old.touched)
        for c in ['dep_only','joint','increment']:
            np.testing.assert_allclose(zero[c],old[c],atol=1e-8,rtol=0)
        inp.to_parquet(dest/'posterior_inputs.parquet',index=False)
        zero.to_parquet(dest/'zero.parquet',index=False)
        np.savez_compressed(dest/'packet.npz',**packet)
        write_json(path,dict(identity=art(OUT/'identity.json'),arm=arm,diagnostics=diagnostics,
            complete_raw_peers=len(inp),targets=len(packet['mvt_id']),original_apply_zero_replay=True,
            dependencies=[art(dest/'prior.json'),art(dest/'calibration.json')],
            artifacts=[art(dest/name) for name in ['posterior_inputs.parquet','zero.parquet','packet.npz']]))
        print(dict(arm=arm['key'],raw_peers=len(inp),targets=len(packet['mvt_id'])),flush=True)


def learn():
    guard()
    for outer in MONTHS:
        dest=OUT/outer;dest.mkdir(exist_ok=True)
        if (dest/'learned.json').exists():
            receipt(dest/'learned.json',outer=outer)
            continue
        assert not list(dest.glob('optimizer_*.json')),'Preserve partial optimizer receipts; no retry'
        packets=[]
        dependencies=[]
        for arm in [a for a in arms() if a['outer']==outer and a['part']!='outer']:
            folder=OUT/'arms'/arm['key'];receipt(folder/'packet.json',arm=arm)
            dependencies.append(art(folder/'packet.json'))
            with np.load(folder/'packet.npz',allow_pickle=False) as f:
                packets.append({k:f[k] for k in f.files})
        packet=model.concatenate(packets)
        allowed=model.gate.original.prior.base.FOLDS[outer]
        y=labels(allowed,packet['mvt_id']).y.to_numpy(float)
        # Both inner banks are immutable before their own targets enter loss.
        records=[]
        for kind in ['prior','posterior']:
            record=model.optimize(packet,y,kind)
            write_json(dest/f'optimizer_{kind}.json',record)
            assert record['success'],record
            records.append(record)
        write_json(dest/'learned.json',dict(identity=art(OUT/'identity.json'),outer=outer,allowed=allowed,
            targets=len(y),parameters={r['objective']:r['theta'] for r in records},
            raw_target_no_clipping=True,outer_labels_not_read=True,
            dependencies=dependencies,
            optimizers=[art(dest/f'optimizer_{k}.json') for k in ['prior','posterior']]))
        print(dict(outer=outer,targets=len(y),optimizers=records),flush=True)


def infer():
    guard()
    for outer in MONTHS:
        dest=OUT/outer
        if (dest/'inference.json').exists():
            receipt(dest/'inference.json',outer=outer)
            continue
        learned=receipt(dest/'learned.json',outer=outer)
        folder=OUT/'arms'/f'{outer}_outer'
        arm=next(a for a in arms() if a['key']==f'{outer}_outer')
        receipt(folder/'packet.json',arm=arm)
        rows=pd.read_parquet(folder/'posterior_inputs.parquet')
        with np.load(folder/'packet.npz',allow_pickle=False) as f:packet={k:f[k] for k in f.files}
        result=model.predictions(rows,packet,np.zeros(4)).rename(columns={'increment':'zero_increment'})
        for kind in ['prior','posterior']:
            result[kind+'_increment']=model.predictions(rows,packet,learned['parameters'][kind]).increment
        result.to_parquet(dest/'predictions.parquet',index=False)
        write_json(dest/'inference.json',dict(identity=art(OUT/'identity.json'),outer=outer,
            dependencies=[art(dest/'learned.json'),art(folder/'packet.json')],
            artifacts=[art(dest/'predictions.parquet')],outer_labels_not_read=True))
        print(dict(outer=outer,raw_rows=len(rows),targets=len(packet['mvt_id'])),flush=True)


def score():
    guard();assert not (OUT/'result.json').exists()
    import tuning_seasonal as ts
    # Candidate coefficients and held outputs must already be immutable.
    predictions=[]
    for m in MONTHS:
        receipt(OUT/m/'inference.json',outer=m)
        predictions.append(pd.read_parquet(OUT/m/'predictions.parquet'))
    pred=pd.concat(predictions,ignore_index=True).set_index('mvt_id')
    assert pred.index.is_unique
    current=pd.read_parquet(CURRENT/'scored.parquet')
    f=current[['mvt_id','month','day','y']].copy()
    f['control']=current.three_tabm_runway.to_numpy(float)
    assert len(f)==f.mvt_id.nunique()==344419
    old=pd.read_parquet(INPUTS/'old_predictions.parquet').set_index('mvt_id')
    # All scored Rome rows must be among the new full raw peer predictions.
    assert set(old.index).issubset(pred.index)
    at=f.mvt_id.isin(pred.index)
    ix=np.flatnonzero(at);new=pred.loc[f.mvt_id.iloc[ix]]
    for c in ['zero','prior','posterior']:
        delta=np.zeros(len(f))
        inc=new.zero_increment.to_numpy(float) if c=='zero' else new[c+'_increment'].to_numpy(float)
        delta[ix]=.5*(inc-new.zero_increment.to_numpy(float))
        f[c+'_delta']=delta
        f[c]=np.rint(np.maximum(f.control.to_numpy()+delta,0.))
        # Family diagnostics use identical clipping and nearest-even rounding.
        previous=old.increment.reindex(new.index).fillna(0.).to_numpy(float)
        family=np.zeros(len(f));family[ix]=.5*(inc-previous)
        f[c+'_family']=np.rint(np.maximum(f.control.to_numpy()+family,0.))
    np.testing.assert_array_equal(f.zero,f.control)
    for m,g in f.groupby('month'):
        truth=labels([m],g.loc[g.mvt_id.isin(pred.index),'mvt_id'])
        np.testing.assert_array_equal(truth.y,g.loc[g.mvt_id.isin(pred.index),'y'])
    columns=['control','zero','prior','posterior','zero_family','prior_family','posterior_family']
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in f.groupby('month')}
    paired={c:ts.paired_bootstrap(f.day,f.month,(f[c]-f.y)**2,(f.posterior-f.y)**2) for c in ['control','prior']}
    cv={c:float(np.mean([s[c] for s in scores.values()])) for c in columns}
    np.testing.assert_allclose(cv['control'],289.7627782102906,atol=1e-8,rtol=0)
    gate=all(p['ci95'][1]<0 for p in paired.values()) and all(
        s['posterior']<min(s['control'],s['prior']) for s in scores.values())
    f.to_parquet(OUT/'scored.parquet',index=False)
    write_json(OUT/'result.json',dict(identity=art(OUT/'identity.json'),scores=scores,cv=cv,paired=paired,
        gate=gate,scored=art(OUT/'scored.parquet'),current=art(CURRENT/'scored.parquet'),
        exact_zero_replay=True,raw_peer_prediction_rows=len(pred),scored_rows=len(f),
        interpretation='Fixed additive serialized-v80 transport; reused development; family substitution reported separately',
        no_production_or_upload=True))
    print(dict(cv=cv,paired=paired,gate=gate),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','freeze','calibrate','preflight','fit_priors','packetize','learn','infer','score'])
    args=parser.parse_args()
    with threadpool_limits(1): globals()[args.stage]()
