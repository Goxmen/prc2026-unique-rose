"""Step 4: conditional arrival-label and recording audit, using local data only.

No prediction models, departure labels from ranking, or validation weights.
Arrival weather is aligned to ADES (landing airport), not ADEP (origin).
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import weather

OUT=ROOT/'VALIDATION/reports/train_ranking_step4'
CACHE=ROOT/'tmp/train_ranking_step4'
AIRPORTS=['EHAM','EDDF','LSZH']
THRESHOLDS=[1800,3600,7200]
SEED=20260921


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def preceding_counts(query,events,seconds=1800):
    """Count [t-window,t), excluding simultaneous movements including self."""
    q=pd.to_datetime(query).to_numpy('datetime64[ns]').astype('int64')
    e=np.sort(pd.to_datetime(events).to_numpy('datetime64[ns]').astype('int64'))
    return np.searchsorted(e,q,side='left')-np.searchsorted(e,q-seconds*10**9,side='left')


def attach_weather(arr,metar):
    # Existing physical METAR transformations; no changes to model features.
    reports=weather._with_window(weather.report_table(metar))
    codes=metar.drop_duplicates(['station','valid','metar']).copy()
    codes['valid']=weather._utc_naive(codes.valid)
    tokens=weather._codes(codes.wxcodes)
    codes['strict_snow']=weather._any(tokens,lambda t:'SN' in t or 'SG' in t).to_numpy()
    codes['ice_or_small_hail']=weather._any(tokens,lambda t:'PL' in t or 'GS' in t).to_numpy()
    # report_table uses stable station/time sorting; enforce exact row alignment.
    codes=codes.sort_values(['station','valid'],kind='stable').reset_index(drop=True)
    np.testing.assert_array_equal(codes[['station','valid']],reports[['station','valid']])
    reports['strict_snow']=codes.strict_snow
    reports['ice_or_small_hail']=codes.ice_or_small_hail
    pieces=[]
    for _,r in reports.groupby('station',sort=False):
        r=r.copy(); window=r.set_index('valid').rolling('24h',closed='both')
        r['snow_prev24h']=window.strict_snow.max().to_numpy()
        r['reports_prev24h']=window.strict_snow.count().to_numpy()
        r['weather_code']=codes.loc[r.index,'wxcodes'].fillna('').to_numpy()
        pieces.append(r)
    reports=pd.concat(pieces,ignore_index=True)
    query=arr[['mvt_id','airport','landing']].rename(columns={'airport':'station'})
    query['landing']=weather._utc_naive(query.landing)
    joined=pd.merge_asof(query.sort_values('landing'),reports.sort_values('valid'),
        left_on='landing',right_on='valid',by='station',direction='backward')
    age=(joined.landing-joined.valid).dt.total_seconds()/60
    known=joined.valid.notna()&age.between(0,90)
    columns=weather.WEATHER_FEATURES[:-len(weather.WINDOW_FEATURES)]+weather.WINDOW_FEATURES
    joined['wx_age_min']=age
    extras=['strict_snow','ice_or_small_hail','snow_prev24h','reports_prev24h']
    joined.loc[~known,columns+extras]=np.nan
    # Fewer than 12 reports in a day is inadequate coverage for a negative
    # recent-snow classification. Positive reports remain observed evidence.
    joined.loc[joined.reports_prev24h.lt(12)&joined.snow_prev24h.eq(0),'snow_prev24h']=np.nan
    joined=joined.set_index('mvt_id').loc[arr.mvt_id]
    assert not (joined.valid>joined.landing).any()
    return joined[columns+extras+['valid','weather_code']].reset_index(drop=True)


def make_features(con):
    arr=con.execute("""SELECT MVT_ID_mvt AS mvt_id, ADES_mvt AS airport,
        ADEP_mvt AS origin, FLIGHT_mvt AS flight, FLIGHT_ID_mvt AS flight_id,
        AIRCRAFT_TYPE_mvt AS aircraft, WK_TBL_CAT_flt AS wake,
        AIRCRAFT_OPERATOR_flt AS operator, STAND_mvt AS stand, RUNWAY_mvt AS runway,
        CAST(MVT_TIME_UTC_mvt AS TIMESTAMP) AS landing,
        CAST(BLOCK_TIME_UTC_mvt AS TIMESTAMP) AS block,
        CAST(SCHED_TIME_UTC_mvt AS TIMESTAMP) AS schedule,
        CAST(ARVT_1_flt AS TIMESTAMP) AS arvt1, CAST(ARVT_3_flt AS TIMESTAMP) AS arvt3,
        TAXITIME_SEC_mvt AS taxi_sec, AOBT_3_flt IS NULL AS no_aobt3
        FROM raw WHERE PHASE_mvt='ARR' AND ADES_mvt IN ('EHAM','EDDF','LSZH')""").df()
    assert arr.mvt_id.is_unique and arr.mvt_id.notna().all()
    assert arr[['landing','block','taxi_sec']].notna().all().all()
    arr['date']=arr.landing.dt.floor('D'); arr['month']=arr.landing.dt.strftime('%Y-%m')
    arr['no_flight_id']=arr.flight_id.isna()
    flight=arr.flight.astype('string')
    arr['airline']=flight.str[:3].where(flight.str[:3].str.isalpha(),flight.str[:2])
    arr['stand_area']=arr.stand.astype('string').str.extract(r'^([A-Za-z]+)',expand=False).fillna('numeric_or_missing')
    arr['hour6']=arr.landing.dt.hour//6
    metar=pd.read_parquet(weather.METAR)
    metar=metar[metar.station.isin(AIRPORTS)].copy()
    wx=attach_weather(arr,metar)
    arr=pd.concat([arr,wx],axis=1)
    dep=con.execute("""SELECT ADEP_mvt AS airport, CAST(MVT_TIME_UTC_mvt AS TIMESTAMP) AS takeoff,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month
        FROM raw WHERE PHASE_mvt='DEP' AND ADEP_mvt IN ('EHAM','EDDF','LSZH')""").df()
    arr['arrivals_prev30']=np.nan;arr['departures_prev30']=np.nan
    for (airport,month),g in arr.groupby(['airport','month']):
        arr.loc[g.index,'arrivals_prev30']=preceding_counts(g.landing,g.landing)
        events=dep[dep.airport.eq(airport)&dep.month.eq(month)]
        arr.loc[g.index,'departures_prev30']=preceding_counts(g.landing,events.takeoff)
    boundary=arr.landing.dt.day.eq(1)&arr.landing.dt.hour.eq(0)&arr.landing.dt.minute.lt(30)
    arr.loc[boundary,['arrivals_prev30','departures_prev30']]=np.nan
    arr['traffic_prev30']=arr.arrivals_prev30+arr.departures_prev30
    arr['clock_difference_sec']=arr.taxi_sec-(arr.block-arr.landing).dt.total_seconds()
    for name in ['schedule','arvt1','arvt3']:
        arr['block_minus_'+name]=(arr.block-arr[name]).dt.total_seconds()
    arr['landing_minus_schedule']=(arr.landing-arr.schedule).dt.total_seconds()
    arr['landing_minus_arvt3']=(arr.landing-arr.arvt3).dt.total_seconds()
    arr['block_second_zero']=arr.block.dt.second.eq(0)
    arr['block_minute_zero']=arr.block.dt.minute.eq(0)&arr.block.dt.second.eq(0)
    arr['label_minute_multiple']=arr.taxi_sec.mod(60).eq(0)
    arr['block_batch3']=arr.groupby(['airport','block']).mvt_id.transform('size').ge(3)
    arr['same_stand_block_duplicate']=arr.groupby(['airport','stand','block']).mvt_id.transform('size').gt(1).fillna(False)
    return arr


def summarize(g):
    y=g.taxi_sec.to_numpy(float)
    out=dict(n=len(g),median=float(np.median(y)),p99=float(np.quantile(y,.99)),
        p999=float(np.quantile(y,.999)),maximum=float(y.max()),mean=float(y.mean()),
        zero_or_negative=int((y<=0).sum()))
    for t in THRESHOLDS:
        tail=g[g.taxi_sec>t];daily=tail.groupby('date').size().sort_values(ascending=False)
        out[f'over_{t}_n']=len(tail);out[f'over_{t}_pct']=100*len(tail)/len(g)
        out[f'over_{t}_days']=len(daily)
        out[f'over_{t}_top3day_pct']=float(100*daily.iloc[:3].sum()/len(tail)) if len(tail) else None
    return out


def day_weights(n,reps,block,rng):
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    counts=np.zeros((reps,n))
    np.add.at(counts,(np.repeat(np.arange(reps),n),indices.ravel()),1)
    return counts


def cell_data(source,target,columns,min_source=20,min_target=5):
    s=source.copy();t=target.copy()
    for c in columns:
        s[c]=s[c].astype('string').fillna('<missing>');t[c]=t[c].astype('string').fillna('<missing>')
    sc=s.groupby(columns,dropna=False).size();tc=t.groupby(columns,dropna=False).size()
    sizes=pd.concat([sc.rename('n_source'),tc.rename('n_target')],axis=1).fillna(0)
    common=sizes[(sizes.n_source>=min_source)&(sizes.n_target>=min_target)].copy()
    common['cell']=np.arange(len(common))
    s=s.join(common.cell,on=columns);t=t.join(common.cell,on=columns)
    return s,t,common.reset_index()


def standardize(source,target,columns,threshold,min_source=20,min_target=5):
    s,t,cells=cell_data(source,target,columns,min_source,min_target)
    sk=s[s.cell.notna()];tk=t[t.cell.notna()]
    row=dict(source_n=len(s),target_n=len(t),source_supported_n=len(sk),target_supported_n=len(tk),
        target_coverage_pct=100*len(tk)/len(t),source_coverage_pct=100*len(sk)/len(s),
        cells=len(cells),threshold=threshold,min_source=min_source,min_target=min_target)
    for name,g in [('source_all',s),('target_all',t),('source_supported',sk),('target_supported',tk)]:
        row[name+'_tail_pct']=float(100*g.taxi_sec.gt(threshold).mean()) if len(g) else None
    if not len(tk):
        row.update(expected_target_tail_pct=None,residual_pp=None,composition_pp=None)
        return row,[],cells
    src=sk.groupby('cell').taxi_sec.agg(n='size',tails=lambda x:(x>threshold).sum())
    target_n=tk.groupby('cell').size()
    expected=float((src.tails/src.n*target_n).sum()/len(tk))
    row.update(expected_target_tail_pct=100*expected,
               residual_pp=row['target_supported_tail_pct']-100*expected,
               composition_pp=100*expected-row['source_supported_tail_pct'])
    cells=cells.set_index('cell')
    cells['source_tail_n']=src.tails;cells['source_tail_pct']=100*src.tails/src.n
    cells['target_tail_n']=tk.groupby('cell').taxi_sec.apply(lambda x:(x>threshold).sum())
    cells['target_tail_pct']=100*cells.target_tail_n/cells.n_target
    cells['target_expected_tail_n']=cells.n_target*cells.source_tail_pct/100
    matrices=[]
    for g in [s,t]:
        dates=pd.date_range(g.date.min(),g.date.max(),freq='D')
        kept=g[g.cell.notna()]
        n=np.zeros((len(dates),len(cells)));y=n.copy()
        day=dates.get_indexer(kept.date);cell=kept.cell.to_numpy(int)
        np.add.at(n,(day,cell),1);np.add.at(y,(day,cell),kept.taxi_sec.gt(threshold).to_numpy(int))
        matrices.append((n,y))
    cis=[]
    rng=np.random.default_rng(SEED)
    for block in [1,3]:
        estimates=[]
        for n,y in matrices:
            w=day_weights(len(n),1000,block,rng)
            estimates.append((w@n,w@y))
        sn,sy=estimates[0];tn,ty=estimates[1]
        valid=((sn>0)|(tn==0)).all(axis=1)&(tn.sum(axis=1)>0)
        expected_b=np.sum(np.divide(sy,sn,out=np.zeros_like(sy),where=sn>0)*tn,axis=1)
        delta=np.divide(ty.sum(axis=1)-expected_b,tn.sum(axis=1),out=np.full(len(tn),np.nan),where=tn.sum(axis=1)>0)*100
        interval=np.quantile(delta[valid],[.025,.975]).tolist() if valid.sum()>=800 else None
        cis.append(dict(block_days=block,valid_replicates=int(valid.sum()),replicates=1000,
                        residual_ci95_pp=interval,
                        note='Support cells fixed; interval withheld if fewer than 80% of day-resampled replicates retain all needed source cells.'))
    return row,cis,cells.reset_index()


def bins(source,target):
    source=source.copy();target=target.copy()
    cuts=np.unique(source.traffic_prev30.dropna().quantile([1/3,2/3]).to_numpy())
    top_origins=set(k for k,v in sorted(source.origin.value_counts().items(),key=lambda item:(-item[1],str(item[0])))[:10])
    for f in [source,target]:
        f['recent_snow']=f.snow_prev24h.map({0.:'no',1.:'yes'}).fillna('missing')
        f['temperature']=pd.cut(f.wx_temp_c,[-np.inf,0,3,np.inf],labels=['freezing','0_to_3','over_3']).astype('string').fillna('missing')
        f['visibility']=np.where(f.wx_visibility_m.isna(),'missing',np.where(f.wx_visibility_m<3000,'under_3km','at_least_3km'))
        f['wind']=np.where(f.wx_gust_kt.isna(),'missing',np.where(f.wx_gust_kt>=20,'20kt_plus','under_20kt'))
        # cut returns float codes if ANY row is missing, otherwise integer
        # codes. Normalize before string conversion so '0.0' cannot differ
        # from '0' solely because the other population lacks a missing row.
        f['traffic']=pd.cut(f.traffic_prev30,[-np.inf,*cuts,np.inf],labels=False).astype('Int64').astype('string').fillna('missing')
        f['route_group']=f.origin.where(f.origin.isin(top_origins),'other')
        f['all_flights']='all'
    return source,target,dict(traffic_cutoffs=cuts.tolist(),top_origins=sorted(top_origins))


def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    inputs=[*sorted((ROOT/'data').glob('training_*.parquet')),ROOT/'data/ranking.parquet',weather.METAR]
    identity={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    build_hash=hashlib.sha256(''.join(inspect.getsource(f) for f in [make_features,attach_weather,preceding_counts]).encode()).hexdigest()
    signature=dict(inputs=identity,feature_builder=build_hash,weather_source=sha(Path(weather.__file__)),timezone='UTC')
    con=duckdb.connect();con.execute("SET TimeZone='UTC'");con.execute('SET threads=6')
    paths=[str(p) for p in inputs[:-1]]
    con.read_parquet(paths).create_view('raw')
    print('Monthly label summaries',flush=True)
    monthly=con.execute("""SELECT strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,ADES_mvt AS airport,
        count(*) AS n,count(TAXITIME_SEC_mvt) AS n_label,
        median(TAXITIME_SEC_mvt) AS median,quantile_cont(TAXITIME_SEC_mvt,.99) AS p99,
        quantile_cont(TAXITIME_SEC_mvt,.999) AS p999,max(TAXITIME_SEC_mvt) AS maximum,
        count(*) FILTER(WHERE TAXITIME_SEC_mvt>1800) AS over_1800_n,
        count(*) FILTER(WHERE TAXITIME_SEC_mvt>3600) AS over_3600_n,
        count(*) FILTER(WHERE TAXITIME_SEC_mvt>7200) AS over_7200_n
        FROM raw WHERE PHASE_mvt='ARR' GROUP BY 1,2 ORDER BY 1,2""").df()
    monthly['missing_label_n']=monthly.n-monthly.n_label
    assert monthly.loc[monthly.month.str.startswith('2026'),'missing_label_n'].eq(0).all()
    for t in THRESHOLDS:monthly[f'over_{t}_pct']=100*monthly[f'over_{t}_n']/monthly.n_label
    monthly.to_csv(OUT/'monthly_all_airports.csv',index=False)
    manifest_path=CACHE/'features_manifest.json';cache_path=CACHE/'arrival_features.parquet'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        assert manifest['signature']==signature,'feature cache changed; investigate before rebuilding'
        assert sha(cache_path)==manifest['sha256']
        arr=pd.read_parquet(cache_path)
        print('Reuse local arrival feature cache',len(arr),flush=True)
    else:
        print('Build local weather/traffic/recording features',flush=True)
        arr=make_features(con)
        arr.to_parquet(cache_path,index=False,compression='zstd')
        manifest=dict(signature=signature,sha256=sha(cache_path),rows=len(arr))
        write(manifest_path,manifest)
    assert arr.mvt_id.is_unique
    audit=dict(design_sha256=sha(OUT/'design.json'),source_sha256=sha(Path(__file__)),
               features=manifest,models_fitted=0,uploads=0,downloads=0,
               max_label_arithmetic_difference=float(arr.clock_difference_sec.abs().max()),
               label_arithmetic_mismatch_n=int(arr.clock_difference_sec.abs().gt(1).sum()))
    by_group=[];daily=[];patterns=[];profiles=[]
    for (airport,month),g in arr.groupby(['airport','month']):
        profiles.append(dict(airport=airport,month=month,**summarize(g),
            weather_missing_n=int(g.wx_temp_c.isna().sum()),traffic_missing_n=int(g.traffic_prev30.isna().sum())))
        conditions={'all':np.ones(len(g),bool),'strict_snow_now':g.strict_snow.eq(1),
            'snow_prev24h':g.snow_prev24h.eq(1),'no_snow_prev24h':g.snow_prev24h.eq(0),
            'snow24h_unknown':g.snow_prev24h.isna(),'missing_flight_id':g.no_flight_id,
            'with_flight_id':~g.no_flight_id,'freezing':g.wx_temp_c.le(0),
            'schedule_match60':g.block_minus_schedule.abs().le(60),
            'not_schedule_match60':g.block_minus_schedule.abs().gt(60)}
        for name,mask in conditions.items():
            if mask.sum():by_group.append(dict(airport=airport,month=month,group=name,**summarize(g.loc[mask])))
        for date,part in g.groupby('date'):
            daily.append(dict(airport=airport,month=month,date=str(date.date()),**summarize(part),
                snow_now_pct=100*float(part.strict_snow.eq(1).mean()),
                snow24h_pct=100*float(part.snow_prev24h.eq(1).mean()),
                temperature_min=float(part.wx_temp_c.min()),
                traffic_mean=float(part.traffic_prev30.mean())))
        flags={'missing_flight_id':g.no_flight_id,'missing_aobt3':g.no_aobt3,
            'block_second_zero':g.block_second_zero,'block_hour_exact':g.block_minute_zero,
            'label_minute_multiple':g.label_minute_multiple,'block_batch3':g.block_batch3,
            'same_stand_block_duplicate':g.same_stand_block_duplicate,
            'nonpositive':g.taxi_sec.le(0),'over_one_day':g.taxi_sec.ge(86400),
            'landing_matches_arvt3_60':g.landing_minus_arvt3.abs().le(60)}
        for clock in ['schedule','arvt1','arvt3']:
            for tol in [0,60]:flags[f'block_matches_{clock}_{tol}']=g['block_minus_'+clock].abs().le(tol)
        for population,subset in [('all',np.ones(len(g),bool)),('over30min',g.taxi_sec.gt(1800)),('over60min',g.taxi_sec.gt(3600))]:
            for name,mask in flags.items():
                n=int(np.asarray(subset).sum());count=int((mask&subset).sum())
                patterns.append(dict(airport=airport,month=month,population=population,pattern=name,n=n,count=count,pct=100*count/n if n else None))
    pd.DataFrame(profiles).to_csv(OUT/'profiles.csv',index=False)
    pd.DataFrame(by_group).to_csv(OUT/'weather_and_id_groups.csv',index=False)
    days=pd.DataFrame(daily);days.to_csv(OUT/'daily.csv',index=False)
    pd.DataFrame(patterns).to_csv(OUT/'recording_patterns.csv',index=False)
    for t in [1800,3600]:
        tail=arr[arr.airport.eq('EHAM')&arr.month.eq('2026-01')&arr.taxi_sec.gt(t)]
        tail.sort_values('taxi_sec',ascending=False).to_csv(CACHE/f'EHAM_2026-01_over{t}.csv',index=False)
    groups=[]
    for (airport,month),g in arr[arr.month.isin(['2025-01','2026-01','2025-07','2026-07'])].groupby(['airport','month']):
        for variable in ['runway','stand_area','stand','airline','origin','aircraft']:
            for category,part in g.groupby(variable,dropna=False):
                groups.append(dict(airport=airport,month=month,variable=variable,category=str(category),n=len(part),tail30=int(part.taxi_sec.gt(1800).sum()),tail60=int(part.taxi_sec.gt(3600).sum())))
    pd.DataFrame(groups).to_csv(OUT/'categorical_groups.csv',index=False)
    # Independent source/target resampling by day, with fixed common support.
    base=['recent_snow','temperature','visibility','wind']
    stages={'unadjusted':['all_flights'],'snow_only':['recent_snow'],'weather':base,'weather_traffic':base+['traffic'],
        'weather_traffic_runway':base+['traffic','runway'],
        'plus_stand_area':base+['traffic','runway','stand_area'],
        'plus_time':base+['traffic','runway','stand_area','hour6'],
        'plus_route_fleet':base+['traffic','runway','stand_area','hour6','route_group','wake']}
    comparisons=[('EHAM','2025-01','2026-01'),('EHAM','2025-07','2026-07'),
                 ('EDDF','2025-01','2026-01'),('LSZH','2025-01','2026-01'),('EHAM','all_2025','2026-01')]
    adjusted=[];intervals=[];cell_tables=[];bin_specs={}
    with threadpool_limits(limits=1):
        for airport,source_month,target_month in comparisons:
            print('Conditional comparison',airport,source_month,target_month,flush=True)
            source=arr[arr.airport.eq(airport)&(arr.month.str.startswith('2025') if source_month=='all_2025' else arr.month.eq(source_month))]
            target=arr[arr.airport.eq(airport)&arr.month.eq(target_month)]
            source,target,spec=bins(source,target);case=f'{airport}_{source_month}_{target_month}';bin_specs[case]=spec
            for stage,columns in stages.items():
                for threshold in THRESHOLDS:
                    result,ci,cells=standardize(source,target,columns,threshold)
                    keys=dict(case=case,airport=airport,source=source_month,target=target_month,stage=stage,threshold=threshold)
                    adjusted.append(dict(**{k:v for k,v in keys.items() if k!='threshold'},**result))
                    intervals.extend(dict(**keys,**c) for c in ci)
                    if threshold==1800:
                        cells['case']=case;cells['stage']=stage;cell_tables.append(cells)
    pd.DataFrame(adjusted).to_csv(OUT/'conditional_tail_rates.csv',index=False)
    write(OUT/'conditional_intervals.json',intervals)
    pd.concat(cell_tables,ignore_index=True).to_csv(OUT/'supported_cells.csv',index=False)
    audit['bin_specs']=bin_specs
    # The departure bridge uses previously hashed feature caches, never labels.
    delivery=json.loads((ROOT/'VALIDATION/reports/train_ranking_step2/delivery.json').read_text())
    bridge=[]
    for month in ['2025-01','2026-01','2025-07','2026-07']:
        p=ROOT/f'tmp/train_ranking_step1/{month}.parquet'
        assert sha(p)==delivery['source_cache_sha256'][month]
        dep=pd.read_parquet(p,columns=['date','airport','no_flight_id','takeoff_minus_sched_min'])
        dep=dep[dep.airport.eq('EHAM')]
        for date,g in dep.groupby('date'):
            bridge.append(dict(airport='EHAM',date=str(date.date()),dep_n=len(g),
                dep_no_id_n=int(g.no_flight_id.eq(1).sum()),dep_gap4h_n=int(g.takeoff_minus_sched_min.gt(240).sum())))
    bridge=pd.DataFrame(bridge).merge(days[days.airport.eq('EHAM')],on=['airport','date'],validate='one_to_one')
    bridge['dep_no_id_pct']=100*bridge.dep_no_id_n/bridge.dep_n
    bridge['dep_gap4h_pct']=100*bridge.dep_gap4h_n/bridge.dep_n
    bridge.to_csv(OUT/'departure_arrival_days.csv',index=False)
    # Only own outputs: later follow-ups have separate manifests and must not
    # invalidate this one when their independently generated files change.
    own_outputs=['monthly_all_airports.csv','profiles.csv','weather_and_id_groups.csv',
        'daily.csv','recording_patterns.csv','categorical_groups.csv',
        'conditional_tail_rates.csv','conditional_intervals.json','supported_cells.csv',
        'departure_arrival_days.csv']
    audit['outputs']={name:sha(OUT/name) for name in own_outputs}
    write(OUT/'audit.json',audit)
    print('Complete: no model fits, uploads, downloads or weight changes.',flush=True)


if __name__=='__main__':main()
