"""Fixed Rome fractional-OTP source-noise falsifier; no posterior or CV.

Only March/September/October/November2025 labels are projected. January/July
priors are a separate, strictly label-free uncertainty reference. Other source
calendars remain separate and overlaps are descriptive, never a joint noise fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

import daily_delay_posterior as retained

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'VALIDATION/runs/v3_20261012_punctuality_fraction_noise'
COMMON=['2025-03','2025-09','2025-10','2025-11']
HELD=['2025-01','2025-07']
RUNS=ROOT/'VALIDATION/runs'
PUN=ROOT/'data/external/eurocontrol_performance_20261011/pun_apt_data.json'
PUBLIC_INPUTS=RUNS/'v3_20261011_public_delay_features/inputs2025.parquet'
LABELS=RUNS/'v3_20261011_daily_offblock_alignment/rows2025.parquet'
TAXI_ROWS=ROOT/'tmp/research_20261011/daily_taxi_source/training_rows.parquet'
TAXI_CAL=RUNS/'v3_20261011_taxi_count_sum/calibration.parquet'
JOINT=RUNS/'v3_20261011_joint_public_moments'
DELAY=ROOT/'data/external/eurocontrol_performance_20261011/all_pre_departure_delays_2025.csv'
ARRIVAL=ROOT/'data/derived/aircraft_rotation_20261010/identity.parquet'
ARRIVAL_MANIFEST=ARRIVAL.with_suffix('.json')
DESIGN=dict(airport='LIRF',source_airport='Rome Fiumicino',label_months=COMMON,
    otp_calendar='unchanged UTC takeoff date from public-delay input packet; source date meaning remains uncertain',
    otp_population='all supplied Rome departures except explicit VFR; unknown flight rules included; no source-count matching',
    statistic='mean(1[gap-y<=900]); early departures count as on-time',
    calibration='one scalar median(public_fraction-supplied_fraction) on the other three complete months',
    noise='training MAD*1.4826 and leave-month-out residual RMSE/MAE/MAD; no floor or likelihood fitting',
    stability='report all four excluded-month biases/noises and errors; no subset/calendar/threshold selection',
    uncertainty_reference='original Jan/Jul strict input-only joint-public packet and frozen outer-fold priors; no held labels',
    uncertainty='same normal200s ordinary priors and retained exact long-row supports; independent Bernoulli variance plus existing15s ordinary common-shift linearization reported separately',
    taxi_diagnostic='original local-takeoff/runway source error on complete common-month cells; explicit membership-overlap table only, no joint covariance estimate',
    delay_diagnostic='original recorded off-block UTC source error from common-month reference rows; boundary attrition explicit; only descriptive overlap',
    arrival_witness='same-day public arrival OTP minus supplied in-block/schedule OTP, UTC landing date; all supplied valid arrivals because retained cache has no flight rules; no identity gate; descriptive paired-day covariance/correlation only, no regression',
    no_joint_covariance_fit=True,no_denominator_recovery=True,no_binomial_source_precision=True,
    no_posterior=True,no_cv=True,no_taxi_model_fit=True,no_parameter_search=True,no_production=True,
    resources=dict(threads=1,duckdb_memory='256MB',external_timeout_seconds=60))


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)==record;return path

def mad(values):
    a=np.asarray(values,float);return float(1.4826*np.median(abs(a-np.median(a))))

def crossfit(days):
    """Pure scalar source calibration; query month never enters its bias/noise."""
    assert set(days.month)==set(COMMON) and days.day.is_unique
    f=days.copy();f['error']=f.public_otp-f.own_otp
    records=[];f['lomo_error']=np.nan
    for month in COMMON:
        refs=f.loc[f.month.ne(month),'error'];ix=f.month.eq(month)
        assert len(refs)>0 and ix.any()
        bias=float(refs.median());f.loc[ix,'lomo_error']=f.loc[ix,'error']-bias
        e=f.loc[ix,'lomo_error'].to_numpy(float)
        records.append(dict(query_month=month,reference_months=[m for m in COMMON if m!=month],
            reference_days=len(refs),query_days=int(ix.sum()),bias=bias,reference_mad=mad(refs),
            query_mad=mad(e),query_rmse=float(np.sqrt(np.mean(e*e))),query_mae=float(np.mean(abs(e))),
            query_mean=float(e.mean())))
    assert np.isfinite(f.lomo_error).all()
    return f,records


def source_fraction():
    raw=read(PUN);ix={n:i for i,n in enumerate(raw[0])};records=[]
    for row in raw[2:]:
        day=row[ix['Date']]
        if row[ix['Airport']]==DESIGN['source_airport'] and day[:7] in COMMON+HELD:
            records.append(dict(day=day,public_otp=float(row[ix['Departure Punctuality %']]),
                public_arrival_otp=float(row[ix['Arrival Punctuality %']])))
    f=pd.DataFrame(records)
    assert f.day.is_unique and f[['public_otp','public_arrival_otp']].ge(0).all().all()
    assert f[['public_otp','public_arrival_otp']].le(1).all().all()
    return f


def freeze():
    assert not (OUT/'identity.json').exists()
    assert set(COMMON)==set.intersection(*(set(v) for v in retained.base.FOLDS.values()))
    punctual=read(RUNS/'v3_20261011_public_punctuality/manifest.json')
    assert punctual['inputs']==art(PUBLIC_INPUTS) and punctual['source']==art(PUN)
    source=read(RUNS/'v3_20261011_daily_offblock_alignment/source_audit.json')
    assert source['status']=='pass' and source['prepared_rows']==art(LABELS) and source['public']==art(DELAY)
    joint=read(JOINT/'identity.json');assert joint['labels_absent_from_inference_packet']
    for record in joint['inputs']:verify(record)
    for record in joint['helpers']:verify(record)
    assert art(retained.__file__) in joint['helpers'] and art(retained.base.__file__) in joint['helpers']
    assert not {'y','true_day','BLOCK_TIME_UTC_mvt'}&set(pd.read_parquet(JOINT/'inputs.parquet').columns)
    taxi=read(RUNS/'v3_20261011_taxi_count_sum/result.json')
    assert read(RUNS/'v3_20261011_taxi_count_sum/audit.json')['result']==art(RUNS/'v3_20261011_taxi_count_sum/result.json')
    assert art(TAXI_CAL) in taxi['artifacts']
    arrival=read(ARRIVAL_MANIFEST)
    assert arrival['identity_sha256']==art(ARRIVAL)['sha256']
    assert arrival['source_sha256']==art(ROOT/'VALIDATION/aircraft_rotation.py')['sha256']
    paths=[Path(__file__),Path(retained.__file__),Path(retained.base.__file__),PUN,PUBLIC_INPUTS,LABELS,TAXI_ROWS,TAXI_CAL,DELAY,
        JOINT/'identity.json',JOINT/'inputs.parquet',JOINT/'parameters.json',
        RUNS/'v3_20261011_public_punctuality/manifest.json',RUNS/'v3_20261011_daily_offblock_alignment/source_audit.json',
        RUNS/'v3_20261011_taxi_count_sum/result.json',RUNS/'v3_20261011_taxi_count_sum/audit.json',
        ARRIVAL,ARRIVAL_MANIFEST,ROOT/'VALIDATION/aircraft_rotation.py']
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'identity.json').write_text(json.dumps(dict(design=DESIGN,source=art(__file__),files=[art(p) for p in paths]),indent=2)+'\n')


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN and identity['source']==art(__file__)
    for record in identity['files']:verify(record)
    return identity


def projected(con):
    # This is the only departure-label projection. Its WHERE scope is fixed.
    train=con.execute('''SELECT mvt_id,day,month,gap,y,fltrule,takeoff_epoch,true_day
        FROM read_parquet(?) WHERE airport='LIRF' AND month IN (SELECT unnest(?))''',[str(LABELS),COMMON]).df()
    assert set(train.month)==set(COMMON) and train.mvt_id.is_unique
    inputs=con.execute('''SELECT mvt_id,day,gap,clock,fltrule FROM read_parquet(?)
        WHERE airport='LIRF' AND substr(day,1,7) IN (SELECT unnest(?))''',[str(PUBLIC_INPUTS),COMMON+HELD]).df()
    assert inputs.mvt_id.is_unique
    raw=inputs.set_index('mvt_id').loc[train.mvt_id]
    for col in ['day','gap']:
        np.testing.assert_array_equal(train[col],raw[col])
    assert train.fltrule.fillna('<NULL>').tolist()==raw.fltrule.fillna('<NULL>').tolist()
    assert np.isfinite(train[['gap','y','takeoff_epoch']]).all().all()
    return train.loc[~train.fltrule.eq('V')].copy(),inputs


def prior_otp(gap,mean,months,parameters):
    """Label-free support integration; equality at 900 seconds is on time."""
    gap=np.asarray(gap,float);mean=np.asarray(mean,float)
    z=(mean-gap+900)/200.;prob=ndtr(z);load=np.exp(-z*z/2)/(np.sqrt(2*np.pi)*200)
    for i in np.flatnonzero(gap>3600):
        values,weights,_=retained.support(gap[i],mean[i],False,np.nan,'LIRF',parameters[months[i]]['support'])
        prob[i]=float(weights@(values>=gap[i]-900));load[i]=0.
    assert np.isfinite(prob).all() and np.all((prob>=-1e-14)&(prob<=1+1e-14))
    return np.clip(prob,0,1),load  # Only the declared floating-point roundoff range.


def uncertainty(inputs):
    names=['mvt_id','month','gap','prior','fltrule']
    f=pd.read_parquet(JOINT/'inputs.parquet',columns=names)
    assert not {'y','true_day','BLOCK_TIME_UTC_mvt'}&set(f.columns)
    assert set(f.month)==set(HELD) and f.mvt_id.is_unique
    q=inputs.set_index('mvt_id').loc[f.mvt_id]
    np.testing.assert_array_equal(f.gap,q.gap);np.testing.assert_array_equal(f.month,q.day.str[:7])
    assert f.fltrule.fillna('<NULL>').tolist()==q.fltrule.fillna('<NULL>').tolist()
    f['day']=q.day.to_numpy();f=f.loc[~f.fltrule.eq('V')].reset_index(drop=True)
    pars=read(JOINT/'parameters.json')
    prob,load=prior_otp(f.gap,f.prior,f.month.to_numpy(),pars)
    f['probability']=prob;f['bernoulli_variance']=prob*(1-prob);f['ordinary_loading']=load
    g=f.groupby(['month','day'],as_index=False).agg(n=('mvt_id','size'),mean=('probability','mean'),
        variance_sum=('bernoulli_variance','sum'),loading_sum=('ordinary_loading','sum'))
    g['independent_variance']=g.variance_sum/g.n**2
    g['common_variance']=15.**2*(g.loading_sum/g.n)**2
    g['prior_sd']=np.sqrt(g.independent_variance+g.common_variance)
    g['independent_max_sd']=.5/np.sqrt(g.n)
    return g


def arrival_witness(con,days):
    # No departure timestamps/labels or aircraft identifiers enter this query.
    rows=con.execute('''SELECT mvt_id,strftime(t,'%Y-%m-%d') AS "day",
        epoch(inblock)-epoch(arrival_schedule) delay
        FROM read_parquet(?) WHERE phase='ARR' AND airport='LIRF'
        AND strftime(t,'%Y-%m') IN (SELECT unnest(?))''',[str(ARRIVAL),COMMON]).df()
    assert rows.mvt_id.is_unique and set(rows.day.str[:7])==set(COMMON)
    rows['known']=np.isfinite(rows.delay)
    rows['ontime']=np.where(rows.known,rows.delay<=900,np.nan)
    daily=rows.groupby('day',as_index=False).agg(arrivals=('mvt_id','size'),
        valid_arrivals=('known','sum'),own_arrival_otp=('ontime','mean'))
    result=days.merge(daily,on='day',how='left',validate='one_to_one')
    result['arrival_fraction_error']=result.public_arrival_otp-result.own_arrival_otp
    return result,paired_statistics(result)


def paired_statistics(days):
    """Descriptive day-level covariance only, never fit a correction."""
    groups=[('ALL',days)]+list(days.groupby('month'))
    result=[]
    for month,group in groups:
        q=group[['arrival_fraction_error','error','lomo_error']].replace([np.inf,-np.inf],np.nan).dropna()
        record=dict(month=month,days=len(q),statistics={})
        for target in ['error','lomo_error']:
            x,y=q.arrival_fraction_error.to_numpy(float),q[target].to_numpy(float)
            valid=len(q)>1 and np.std(x)>0 and np.std(y)>0
            record['statistics'][target]=dict(covariance=float(np.cov(x,y,ddof=1)[0,1]) if len(q)>1 else None,
                correlation=float(np.corrcoef(x,y)[0,1]) if valid else None)
        result.append(record)
    return result


def overlap_diagnostics(con,train,days):
    """Keep original source calendars; these are descriptive, not noise fitting."""
    taxi=con.execute('''SELECT mvt_id,runway,stand,local_takeoff local_day
        FROM read_parquet(?) WHERE airport='LIRF'
        AND strftime(takeoff AT TIME ZONE 'UTC','%Y-%m') IN (SELECT unnest(?))''',[str(TAXI_ROWS),COMMON]).df()
    assert taxi.mvt_id.is_unique
    linked=train[['mvt_id','day']].merge(taxi,on='mvt_id',validate='one_to_one')
    linked=linked.loc[linked.runway.notna()&linked.stand.notna()]
    edges=linked.groupby(['day','local_day','runway'],as_index=False).size().rename(columns={'size':'overlap_n'})
    # A first local day includes previous UTC-month labels; do not project it.
    cal=con.execute('''SELECT local_day,runway,n,valid,total,source_n,source_mean,omitted_count,omitted_sum
        FROM read_parquet(?) WHERE airport='LIRF' AND fold='2025-01'
        AND substr(local_day,1,7) IN (SELECT unnest(?)) AND substr(local_day,9,2)<>'01' ''',[str(TAXI_CAL),COMMON]).df()
    assert not cal.duplicated(['local_day','runway']).any()
    cal['source_count_error_per_row']=-cal.omitted_count/cal.n
    cal['source_sum_error_per_row']=-cal.omitted_sum/cal.n
    edges=edges.merge(cal,on=['local_day','runway'],how='left',validate='many_to_one')
    edges=edges.merge(days[['day','lomo_error']],on='day',validate='many_to_one')
    source=pd.read_csv(DELAY,usecols=['APT_ICAO','FLT_DATE','FLT_DEP_IFR_2','DLY_ALL_PRE_2'])
    source=source.loc[source.APT_ICAO.eq('LIRF')].rename(columns={'FLT_DATE':'true_day','FLT_DEP_IFR_2':'source_n'})
    source['source_total']=60*source.DLY_ALL_PRE_2
    tr=train.assign(delay=np.maximum(train.gap-train.y,0))
    off=tr.groupby('true_day',as_index=False).agg(n=('mvt_id','size'),own_total=('delay','sum'))
    off=off.loc[off.true_day.str[:7].isin(COMMON)]
    off=off.merge(source[['true_day','source_n','source_total']],on='true_day',how='left',validate='one_to_one')
    off['source_error_per_row']=(off.source_total-off.own_total)/off.n
    off['boundary_day']=pd.to_datetime(off.true_day).dt.is_month_start|pd.to_datetime(off.true_day).dt.is_month_end
    dep_edges=tr.groupby(['day','true_day'],as_index=False).size().rename(columns={'size':'overlap_n'})
    dep_edges=dep_edges.merge(off,on='true_day',how='left',validate='many_to_one').merge(days[['day','lomo_error']],on='day',validate='many_to_one')
    return edges,dep_edges


def pilot():
    guard();assert not (OUT/'result.json').exists();start=time.monotonic()
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        train,inputs=projected(con)
        assert not set(train.month)&set(HELD)
        tr=train.assign(ontime=(train.gap-train.y<=900).astype(float))
        days=tr.groupby(['month','day'],as_index=False).agg(n=('mvt_id','size'),own_otp=('ontime','mean'))
        days=days.merge(source_fraction(),on='day',how='left',validate='one_to_one')
        assert days.public_otp.notna().all()
        days,folds=crossfit(days)
        prior=uncertainty(inputs)
        taxi,delay=overlap_diagnostics(con,train,days)
        arrival,arrival_statistics=arrival_witness(con,days)
    summary=[]
    for record in folds:
        for month,g in prior.groupby('month'):
            summary.append(dict(noise_query_month=record['query_month'],prior_reference_month=month,
                noise_rmse=record['query_rmse'],noise_mad=record['query_mad'],
                prior_sd_median=float(g.prior_sd.median()),prior_sd_p90=float(g.prior_sd.quantile(.9)),
                independent_max_sd_median=float(g.independent_max_sd.median()),
                rmse_to_median_prior_sd=record['query_rmse']/float(g.prior_sd.median())))
    for name,frame in [('days',days),('prior_uncertainty',prior),('taxi_overlaps',taxi),('delay_overlaps',delay),('arrival_witness',arrival)]:frame.to_parquet(OUT/(name+'.parquet'),index=False)
    result=dict(identity=art(OUT/'identity.json'),source=art(__file__),label_rows=len(train),label_months=sorted(train.month.unique()),
        calibration_folds=folds,noise_vs_prior=summary,
        training_fraction_noise=dict(lomo_rmse=float(np.sqrt(np.mean(days.lomo_error**2))),lomo_mad=mad(days.lomo_error)),
        arrival_witness=dict(statistics=arrival_statistics,arrivals=int(arrival.arrivals.sum()),
            valid_arrivals=int(arrival.valid_arrivals.sum()),calendar='UTC landing date',
            population='All supplied Rome arrivals with known in-block and schedule; flight-rule filtering unavailable in retained cache; no aircraft-identity filtering.',
            interpretation='Descriptive common-day source-bias witness only; mismatched arrival/departure populations and source calendars remain unresolved. No regression fitted.'),
        calendars=dict(otp='UTC takeoff',taxi='Europe/Rome local takeoff/runway',delay='recorded off-block UTC from permitted common-month departure labels'),
        overlap_scope='Descriptive memberships and raw source errors only; no joint covariance estimated. Off-block cells may lack flights taking off outside common months; first/last dates flagged, no denominator inference.',
        artifacts=[art(OUT/(n+'.parquet')) for n in ['days','prior_uncertainty','taxi_overlaps','delay_overlaps','arrival_witness']],
        seconds=time.monotonic()-start,no_held_labels=True,no_posterior=True,no_cv=True,no_count_selected=True,
        interpretation='Fixed source-noise falsifier. Prior uncertainty reference dates differ from calibration months; no gain or scoring conclusion follows directly.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','pilot'])
    with threadpool_limits(1):globals()[p.parse_args().stage]()
