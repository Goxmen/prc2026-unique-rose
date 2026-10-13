"""Independent raw inputs and nested calibration audit; never fits a prior/tilt."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import ast
import hashlib
import json
import time
import numpy as np
import pandas as pd
import duckdb
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_posterior_aware_crossfit'
INPUT=OUT/'inputs'
IDENTITY_SHA='acab0bee4edd1ae7c5ba84493239e133b2bf029164a05a891efb8ed9427132c4'
PREFLIGHT_SHA='c1e76bc2730ff77890b5042e7c68eb5e434cec3b30b77c91e5db35ea90f27d77'
FOLDS={'2025-01':[f'2025-{m:02d}' for m in range(3,12)],
       '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}


def read(p):return json.loads(p.read_text())


def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())


def verify(a):
    path=ROOT/a['path'];assert art(path)==a,path
    return path


def close(a,b,atol=1e-8,rtol=1e-10):np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)


def complete(day,allowed):
    start=datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Rome'))
    end=start+timedelta(days=1)-timedelta(microseconds=1)
    return start.astimezone(timezone.utc).strftime('%Y-%m') in allowed and end.astimezone(timezone.utc).strftime('%Y-%m') in allowed


def delay_parameters(rows,source,actual):
    f=rows.assign(contribution=np.maximum(rows.gap-rows.y,0.))
    cells=f.groupby('day',as_index=False).agg(n=('mvt_id','size'),truth=('contribution','sum'))
    cells=cells.merge(source,on='day',validate='one_to_one')
    cells['eps']=cells.total-cells.truth;cells['dn']=cells.n_public-cells.n
    cells=cells[np.isfinite(cells.eps)&np.isfinite(cells.dn)]
    same=cells[cells.dn.eq(0)]
    used=same if len(same)>=10 else cells
    bias=float(used.eps.median())
    sigma=max(60.,float(1.4826*np.median(abs(used.eps-np.median(used.eps)))))
    kappa=max(600.,float(np.median(cells.total/cells.n)))
    bad=cells[cells.dn.ne(0)]
    if len(bad)>=10:
        kappa=max(kappa,float(np.sqrt(np.median(np.maximum((bad.eps-bias)**2-sigma**2,0)/bad.dn**2))))
    held=rows[(rows.gap>3600)&(rows.y<rows.gap-600)&rows.y.between(30,10800)]
    tau=float(held.y.mean()) if len(held)>50 else 1200.
    quantile=np.quantile(held.y,np.linspace(.025,.975,15))
    expected=dict(bias=bias,sigma=sigma,kappa=kappa,tau=tau,days=len(cells),same_count_days=len(same),
        held_rows=len(held),eps_abs_median=float(abs(cells.eps).median()),spread=(quantile-quantile.mean()).tolist())
    assert set(actual)==set(expected)
    for key,value in expected.items():close(actual[key],value)
    return dict(days=len(cells),same_count_days=len(same),support_rows=len(held))


def taxi_parameters(rows,source,allowed,actual,saved):
    x=rows[rows.eligible].copy()
    x['valid']=x.y.between(0,7200);x['contribution']=np.where(x.valid,x.y,0.)
    cells=x.groupby(['airport','local_day','runway'],as_index=False).agg(n=('y','size'),valid=('valid','sum'),total=('contribution','sum'))
    cells=cells.merge(source.assign(airport='LIRF'),on=['airport','local_day','runway'],validate='one_to_one')
    good=(cells.valid>=20)&(cells.source_n>=20)&(cells.source_n/cells.valid).between(.5,1.2)
    good&=np.array([complete(day,allowed) for day in cells.local_day])
    cells=cells[good].reset_index(drop=True)
    assert len(cells)>=100 and len(cells)==actual['cells']==len(saved)
    pd.testing.assert_frame_equal(cells,saved[cells.columns],check_dtype=False,check_exact=True)
    design=np.column_stack([np.ones(len(cells)),cells.n.to_numpy()/100.])
    target=np.column_stack([cells.valid-cells.source_n,cells.total-cells.source_n*cells.source_mean])
    scale=np.maximum(1.4826*np.median(abs(target-np.median(target,axis=0)),axis=0),[1.,60.])
    close(actual['huber_scale'],scale)
    residual=target-design@np.asarray(actual['coefficients'])
    close(saved[['omitted_count','omitted_sum']],target)
    close(saved[['residual_count','residual_sum']],residual)
    # Huber coefficient optimality checked directly, without refitting.
    gradient=design.T@np.clip(-residual/scale,-1.,1.)/len(cells)
    assert abs(gradient).max()<1e-4,('Huber normalized gradient',gradient)
    normalized=residual/np.sqrt(cells.n.to_numpy())[:,None]
    center=np.median(normalized,axis=0)
    mad=np.maximum(1.4826*np.median(abs(normalized-center),axis=0),[1e-6,1e-3])
    keep=np.max(abs(normalized-center)/mad,axis=1)<=3
    np.testing.assert_array_equal(keep,saved.covariance_selected)
    assert keep.sum()>=100 and keep.sum()==actual['covariance_cells']
    close(mad,actual['normalized_mad'])
    z=normalized[keep]-normalized[keep].mean(axis=0)
    covariance=z.T@z/(len(z)-1)
    close(covariance,actual['covariance_per_row'])
    assert np.linalg.eigvalsh(covariance).min()>0 and actual['allowed']==allowed
    return dict(source_cells=len(cells),covariance_cells=int(keep.sum()),huber_mean_gradient_max=float(abs(gradient).max()))


def input_audit(prepared,rows,features,labels):
    rawpaths=[str(ROOT/a['path']) for a in prepared['raw_sources'] if Path(a['path']).name.startswith('training_2025-')]
    assert len(rawpaths)==12
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.read_parquet(rawpaths).create_view('r')
        sql='''select MVT_ID_mvt mvt_id,ADEP_mvt airport,ADES_mvt dest,AIRCRAFT_TYPE_mvt aircraft_type,
          ADEP_mvt||':'||RUNWAY_mvt runway,ADEP_mvt||':'||STAND_mvt stand,FLIGHT_RULE_mvt flight_rule,
          MARKET_SEGMENT_flt market_segment,FLIGHT_TYPE_flt flight_type,WK_TBL_CAT_flt wake_cat,
          AIRCRAFT_OPERATOR_flt "operator",extract(hour from MVT_TIME_UTC_mvt) "hour",
          (extract(dow from MVT_TIME_UTC_mvt)+6)%7 dow,extract(month from MVT_TIME_UTC_mvt) "month",
          epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) mvt_minus_sched_sec,
          epoch(MVT_TIME_UTC_mvt-LOBT_flt) mvt_minus_lobt_sec,epoch(MVT_TIME_UTC_mvt-IOBT_flt) mvt_minus_iobt_sec,
          epoch(MVT_TIME_UTC_mvt-EOBT_1_flt) mvt_minus_eobt1_sec,epoch(MVT_TIME_UTC_mvt-AOBT_3_flt) mvt_minus_aobt3_sec,
          FLIGHT_ID_mvt is null missing_flight_id,substring(FLIGHT_mvt,1,3) airline,
          strftime(MVT_TIME_UTC_mvt,'%Y-%m') utc_month,strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') day,
          epoch(MVT_TIME_UTC_mvt) takeoff_epoch,FLIGHT_ID_mvt is not null has_id,RUNWAY_mvt raw_runway,
          strftime(timezone('Europe/Rome',MVT_TIME_UTC_mvt),'%Y-%m-%d') local_day,
          STAND_mvt is not null and RUNWAY_mvt is not null eligible,TAXITIME_SEC_mvt y
          from r where PHASE_mvt='DEP' and ADEP_mvt='LIRF' '''
        raw=con.execute(sql).fetchdf()
    assert len(raw)==raw.mvt_id.nunique()==160704
    assert len(rows)==len(features)==len(labels)==len(raw)
    raw=raw.set_index('mvt_id').loc[rows.mvt_id].reset_index()
    np.testing.assert_array_equal(features.mvt_id,rows.mvt_id)
    np.testing.assert_array_equal(labels.mvt_id,rows.mvt_id)
    np.testing.assert_array_equal(labels.y,raw.y)
    for column in [c for c in features if c!='mvt_id' and not c.startswith('physical_')]:
        pd.testing.assert_series_equal(features[column].reset_index(drop=True),raw[column],check_dtype=False,check_names=False,check_exact=True)
    mapping={'airport':'airport','month':'utc_month','day':'day','takeoff_epoch':'takeoff_epoch','gap':'mvt_minus_sched_sec',
             'has_id':'has_id','nm_gap':'mvt_minus_lobt_sec','fltrule':'flight_rule','runway':'raw_runway','local_day':'local_day','eligible':'eligible'}
    for column,source in mapping.items():
        pd.testing.assert_series_equal(rows[column],raw[source],check_dtype=False,check_names=False,check_exact=True)
    quality_path=next(ROOT/a['path'] for a in prepared['raw_sources'] if Path(a['path']).name=='taxi_out_additional_time_2025.csv')
    quality=pd.read_csv(quality_path);quality=quality[quality.APT_ICAO.eq('LIRF')]
    quality['key']=quality.YEAR.astype(str)+'-'+quality.MONTH_NUM.astype(str).str.zfill(2)
    invalid=((quality.TF-quality.VALID_FL)/quality.TF).set_axis(quality.key)
    expected=invalid.reindex(rows.local_day.str[:7]).to_numpy()<.05
    np.testing.assert_array_equal(rows.quality,expected)
    bank=ROOT/'data/derived/flight_cohort_context_20261012/train'
    meta=np.load(bank/'meta.npy',mmap_mode='r');values=np.load(bank/'values.npy',mmap_mode='r')
    ix=pd.Index(meta[:,0]).get_indexer(rows.mvt_id)
    assert (ix>=0).all() and len(np.unique(ix))==len(ix)
    original=np.array(values[ix],dtype=np.float32)
    original[:,:4]=np.arcsinh(original[:,:4]/3600)
    original[:,9:11]=np.clip(original[:,9:11]/2000,-10,10)
    physical=np.column_stack([original,np.clip(np.arcsinh(raw.mvt_minus_sched_sec.to_numpy()/7200),-10,10),np.ones(len(rows))]).astype(np.float32)
    np.testing.assert_array_equal(features[[f'physical_{i}' for i in range(17)]],physical)
    aliases=pd.DataFrame(dict(hex=meta[ix,5],clock=meta[ix,1],month=rows.month))
    aliases=aliases[aliases.hex>=0].sort_values(['hex','clock'])
    before=aliases.shift()
    assert not (aliases.hex.eq(before.hex)&(aliases.clock-before.clock<=120000)&aliases.month.ne(before.month)).any()
    return dict(raw_rows=len(raw),feature_values=len(raw)*37,physical_values=len(raw)*17,
                raw_ID_calendar_label_projection_and_quality_exact=True,cross_month_aliases=0)


def main():
    started=time.monotonic();target=OUT/'prefit_audit.json';assert not target.exists()
    identity=read(OUT/'identity.json');prepared=read(OUT/'prepared.json');preflight=read(OUT/'preflight.json')
    assert art(OUT/'identity.json')['sha256']==IDENTITY_SHA and art(OUT/'preflight.json')['sha256']==PREFLIGHT_SHA
    for a in identity['sources']+identity['inputs']+identity['score_inputs']+[identity['prepared'],identity['preparation_source']]:verify(a)
    for a in prepared['raw_sources']:verify(a)
    assert preflight['status']=='pass' and preflight['identity']==art(OUT/'identity.json')
    assert preflight['source']==art(ROOT/'VALIDATION/posterior_aware_crossfit.py')
    assert preflight['no_model_or_tilt_fit']
    rows=pd.read_parquet(INPUT/'rows.parquet');features=pd.read_parquet(INPUT/'features.parquet')
    label_frames=[pd.read_parquet(INPUT/f'labels_{m}.parquet') for m in sorted(rows.month.unique())]
    labels=pd.concat(label_frames,ignore_index=True).set_index('mvt_id').loc[rows.mvt_id].reset_index()
    assert rows.mvt_id.is_unique and labels.mvt_id.is_unique and not {'y','true_day','BLOCK_TIME_UTC_mvt'}&set(features)
    assert list(features)==['mvt_id']+identity['features'] and len(identity['features'])==37
    with threadpool_limits(limits=1):
        inputs=input_audit(prepared,rows,features,labels)
        source_taxi=pd.read_parquet(INPUT/'taxi_source.parquet')
        source_delay=pd.read_parquet(INPUT/'delay_source.parquet')
        full=rows.merge(labels,on='mvt_id',validate='one_to_one')
        original=read(INPUT/'outer_parameters.json')
        summaries=[];calibrations=[]
        for arm in identity['arms']:
            folder=OUT/'arms'/arm['key'];calibration=read(folder/'calibration.json')
            assert calibration['identity']==art(OUT/'identity.json') and calibration['arm']==arm
            for a in calibration['artifacts']:verify(a)
            calibrations.append(art(folder/'calibration.json'))
            assert set(arm['train']).isdisjoint(arm['held']) and arm['outer'] not in arm['train']
            assert set(arm['train']).issubset(FOLDS[arm['outer']])
            params=read(folder/'parameters.json')
            if arm['part']=='outer':
                assert arm['train']==FOLDS[arm['outer']] and params==original[arm['outer']]
                summaries.append(dict(arm=arm['key'],retained_outer_parameters_exact=True));continue
            assert sorted(arm['train']+arm['held'])==FOLDS[arm['outer']]
            training=full[full.month.isin(arm['train'])].reset_index(drop=True)
            ids=pd.read_parquet(folder/'calibration_ids.parquet')
            pd.testing.assert_frame_equal(ids.reset_index(drop=True),training[['mvt_id','month']],check_dtype=False)
            support=delay_parameters(training,source_delay,params['support'])
            canonical=training[~training.fltrule.eq('V')].copy()
            day=pd.to_datetime(canonical.takeoff_epoch-canonical.y,unit='s',utc=True)
            keep=day.dt.strftime('%Y-%m').isin(arm['train']) & day.dt.day.gt(1) & day.dt.day.lt(day.dt.days_in_month)
            canonical['day']=day.dt.strftime('%Y-%m-%d');canonical=canonical[keep].reset_index(drop=True)
            saved=pd.read_parquet(folder/'offblock_calibration_ids.parquet')
            pd.testing.assert_frame_equal(saved.reset_index(drop=True),canonical[['mvt_id','month','day']],check_dtype=False)
            delay=delay_parameters(canonical,source_delay,params['delay'])
            taxi=taxi_parameters(training,source_taxi,arm['train'],params['taxi'],pd.read_parquet(folder/'calibration_cells.parquet'))
            summaries.append(dict(arm=arm['key'],train_months=arm['train'],inner_held_months=arm['held'],
                rows=len(training),offblock_rows=len(canonical),support=support,delay=delay,taxi=taxi,
                no_outer_or_inner_held_label_in_calibration=True))
    assert calibrations==preflight['calibrations']
    for item in preflight['historical_incremental_diagnostics']:
        assert item['incremental_derivative_rank']==4 and item['old_output_replay'] and item['poisoned_labels_invariant']
        assert item['finite_difference_max_abs']<2e-5 and item['no_predictive_gain_claim']
        assert np.isfinite(item['singular_values']).all() and min(item['singular_values'])>0
        assert len(item['cancellation_abs_difference_over_sum'])==5
    assert not list((OUT/'arms').glob('*/model.txt')) and not list(OUT.glob('2025-*/learned.json'))
    report=dict(status='pass',source=art(Path(__file__)),identity=art(OUT/'identity.json'),
        preflight=art(OUT/'preflight.json'),calibrations=calibrations,inputs=inputs,calibration_checks=summaries,
        seconds=time.monotonic()-started,prior_fits=0,tilt_fits=0,source_parameter_refits=0,
        calibration_label_scope='Only each declared complement used in independent parameter calculations; full labels checked solely for raw projection fidelity.',
        likelihood_evaluations=0,incremental_preflight_receipt_verified=True,
        limitations=['Original incremental finite-difference/replay arrays were not exported; immutable producer receipts and previously reviewed equations verified, no posterior inference repeated.',
                    'Source sensitivity and valid crossfit construction do not establish predictive gain.',
                    'Root must separately allocate six sequential CPU priors; this receipt permits the fixed fit guard only.'])
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(audit=art(target),inputs=inputs,calibrations=summaries,seconds=report['seconds'])))


if __name__=='__main__':main()
