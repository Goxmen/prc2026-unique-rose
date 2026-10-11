"""Public-only stand/runway reference estimates with matched two-month context."""
import argparse
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent/'tools'))
import probe_public_reference_inverse as pilot
from records import write_json

source = pilot.source
ROOT = source.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_public_reference_inverse_features'
FEATURES = ['pub_ref_inverse_stand_sec', 'pub_ref_inverse_contrast_sec']
art, read, verify = source.art, source.read, source.verify
DESIGN = dict(change='two public reference-time inverse features, no departure labels',
    context='pairs of months six apart within the same year; Jan/Jul pilot reused without refitting',
    metadata='stand/runway present, UTC and local month in pair; first local day excluded from fitting',
    eligibility='20 source and metadata rows per day; count ratio0.5..1.2; at least20 days,2 held and3 other days per month',
    fit='fixed pilot ridge(10/300)^2, stand fractions and separate month intercepts, all eligible public source days',
    contrast='estimated stand reference minus mean published runway/day reference within its local month',
    missing='NaN for missing stand/runway, unsupported groups or UTC/local pair boundary',
    features=FEATURES, no_labels=True, no_new_parameter_search=True,
    limitation='source population differs; coefficients underidentified; monthly stand-specific drift unmodeled')


def metadata(view):
    if view == '2025':
        f = pd.read_parquet(source.CACHE, columns=['mvt_id','airport','takeoff','runway','stand'])
    else:
        f = pd.read_parquet(ROOT/'data/ranking.parquet',
            columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt','STAND_mvt'],
            filters=[('PHASE_mvt','==','DEP')]).rename(columns=dict(MVT_ID_mvt='mvt_id',
                ADEP_mvt='airport',MVT_TIME_UTC_mvt='takeoff',RUNWAY_mvt='runway',STAND_mvt='stand'))
        f = f.drop(columns='PHASE_mvt')
    assert f.mvt_id.is_unique
    f['utc_month'] = pd.to_datetime(f.takeoff,utc=True).dt.strftime('%Y-%m')
    f['day'] = ''
    for airport,zone in source.ZONES.items():
        mask = f.airport.eq(airport)
        f.loc[mask,'day'] = pd.to_datetime(f.loc[mask,'takeoff'],utc=True).dt.tz_convert(zone).dt.strftime('%Y-%m-%d')
    assert f.day.str.len().eq(10).all()
    f['month'] = f.day.str[:7]
    return f


def eligible_metadata(f, pair):
    periods = [f'2025-{m:02}' for m in [pair,pair+6]]
    return f[f.utc_month.isin(periods) & f.month.isin(periods) & f.stand.notna() & f.runway.notna()]


def fit_pair(f, published):
    counts = f.groupby(['airport','runway','day','stand'],observed=True,as_index=False).size()
    totals = counts.groupby(['airport','runway','day'],observed=True,as_index=False)['size'].sum().rename(columns={'size':'metadata_n'})
    cells = totals.merge(published[['airport','runway','day','n','reference']],
        on=['airport','runway','day'],how='inner',validate='one_to_one')
    cells = cells[cells.metadata_n.ge(20) & cells.n.ge(20) &
        (cells.n/cells.metadata_n).between(.5,1.2) & cells.day.str[-2:].ne('01')].copy()
    observations,coefficients = [],[]
    for (airport,runway), c in cells.groupby(['airport','runway'],observed=True):
        c = c.sort_values('day').reset_index(drop=True)
        c['month'] = c.day.str[:7]
        c['held'] = c.day.str[-2:].astype(int)%4 == 0
        per = c.groupby('month').agg(held=('held','sum'),n=('held','size'))
        if len(c)<20 or (per.held<2).any() or ((per.n-per.held)<3).any():
            continue
        q = counts[counts.airport.eq(airport) & counts.runway.eq(runway)]
        A = q.pivot(index='day',columns='stand',values='size').fillna(0).reindex(c.day).fillna(0)
        A = A.loc[:,A.sum()>0]
        X = A.to_numpy(float)/c.metadata_n.to_numpy()[:,None]
        np.testing.assert_allclose(X.sum(axis=1),1.,atol=1e-12,rtol=0)
        full,baseline,beta,intercepts = pilot.solve(X,c.reference.to_numpy(),c.month.to_numpy(),np.ones(len(c),bool))
        c['full_fit'] = full
        c['source_level'] = baseline
        observations.append(c)
        for month, level in intercepts.items():
            source_level = float(c.loc[c.month.eq(month),'reference'].mean())
            for stand, value in zip(A.columns,beta):
                coefficients.append(dict(airport=airport,runway=runway,month=month,stand=stand,
                    reference=float(level+value),source_level=source_level))
    return pd.concat(observations,ignore_index=True),pd.DataFrame(coefficients)


def build(f, estimates):
    assert set(f).issubset({'mvt_id','airport','takeoff','runway','stand','day','month','utc_month'})
    assert not estimates.duplicated(['airport','runway','month','stand']).any()
    joined = f.merge(estimates,on=['airport','runway','month','stand'],how='left',validate='many_to_one',sort=False)
    # A local-month rollover outside the paired UTC input window has no source fit.
    pair_utc = (joined.utc_month.str[-2:].astype(int)-1)%6
    pair_local = (joined.month.str[-2:].astype(int)-1)%6
    same = pair_utc.eq(pair_local) & joined.utc_month.str[:4].eq(joined.month.str[:4])
    joined.loc[~same,['reference','source_level']] = np.nan
    result = joined[['mvt_id']].copy()
    result[FEATURES[0]] = joined.reference.astype('float32')
    result[FEATURES[1]] = (joined.reference-joined.source_level).astype('float32')
    np.testing.assert_array_equal(result.mvt_id,f.mvt_id)
    assert not np.isinf(result[FEATURES]).any().any()
    return result


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(),'Preserve completed feature build'
    assert read(pilot.OUT/'audit.json')['status']=='pass'
    prior = read(pilot.OUT/'result.json')
    for record in prior['artifacts']: verify(record)
    public = pd.read_parquet(source.SOURCE/'daily_runway.parquet')
    public['reference'] = 60*public.ref_min/public.n
    prior_obs = pd.read_parquet(pilot.OUT/'observations.parquet')
    prior_est = pd.read_parquet(pilot.OUT/'reference_estimates.parquet')
    records = {}
    for view,expected in [('2025',2085047),('2026',344841)]:
        f = metadata(view)
        assert len(f)==expected
        obs = prior_obs[prior_obs.view.eq(view)].copy()
        levels = obs.groupby(['airport','runway','month'],as_index=False).reference.mean().rename(columns={'reference':'source_level'})
        est = prior_est[prior_est.view.eq(view)].drop(columns='view').merge(levels,
            on=['airport','runway','month'],how='left',validate='many_to_one')
        obs = obs.merge(levels,on=['airport','runway','month'],how='left',validate='many_to_one')
        if view=='2025':
            for pair in range(2,7):
                o,e = fit_pair(eligible_metadata(f,pair),public)
                obs = pd.concat([obs,o],ignore_index=True)
                est = pd.concat([est,e],ignore_index=True)
        name = '2025' if view=='2025' else 'ranking'
        values = build(f,est)
        f.to_parquet(OUT/f'inputs{name}.parquet',index=False)
        values.to_parquet(OUT/f'features{name}.parquet',index=False)
        est.to_parquet(OUT/f'estimates{name}.parquet',index=False)
        obs.to_parquet(OUT/f'observations{name}.parquet',index=False)
        records[name] = dict(rows=len(f),sources=[art(OUT/f'{prefix}{name}.parquet')
            for prefix in ['inputs','features','estimates','observations']],
            missing={c:int(values[c].isna().sum()) for c in FEATURES},
            supported_month_runways=len(est[['airport','runway','month']].drop_duplicates()))
    value = dict(design=DESIGN,columns=FEATURES,source=art(__file__),pilot_source=art(pilot.__file__),
        pilot_result=art(pilot.OUT/'result.json'),pilot_audit=art(pilot.OUT/'audit.json'),
        public=art(source.SOURCE/'daily_runway.parquet'),
        raw=[art(source.CACHE),art(ROOT/'data/ranking.parquet')],records=records,
        pilot_coefficients_reused=True,departure_labels_read=False)
    write_json(OUT/'manifest.json',value)
    print(value,flush=True)


def audit():
    assert not (OUT/'audit.json').exists(),'Preserve completed feature audit'
    manifest = read(OUT/'manifest.json')
    assert manifest['source']==art(__file__)
    for key in ['pilot_source','pilot_result','pilot_audit','public']:verify(manifest[key])
    for a in manifest['raw']:verify(a)
    checks = []
    for name,record in manifest['records'].items():
        for a in record['sources']:verify(a)
        with duckdb.connect(config={'threads':1}) as con:
            for alias,prefix in [('i','inputs'),('f','features'),('e','estimates'),('o','observations')]:
                con.read_parquet(str(OUT/f'{prefix}{name}.parquet')).create_view(alias)
            con.read_parquet(str(source.SOURCE/'daily_runway.parquet')).create_view('s')
            if name=='2025':
                con.read_parquet(str(source.CACHE)).create_view('raw')
            else:
                con.execute(f"create view raw as select MVT_ID_mvt mvt_id,ADEP_mvt airport,MVT_TIME_UTC_mvt takeoff,RUNWAY_mvt runway,STAND_mvt stand from read_parquet('{ROOT/'data/ranking.parquet'}') where PHASE_mvt='DEP'")
            zone = 'case i.airport '+' '.join(f"when '{a}' then '{z}'" for a,z in source.ZONES.items())+' end'
            n,u,wrong = con.execute(f'''select count(*),count(distinct i.mvt_id),sum(case when
                i.airport is distinct from raw.airport or i.stand is distinct from raw.stand
                or i.runway is distinct from raw.runway or i.takeoff is distinct from raw.takeoff
                or i.utc_month is distinct from strftime(i.takeoff AT TIME ZONE 'UTC','%Y-%m')
                or i.day is distinct from strftime(i.takeoff AT TIME ZONE ({zone}),'%Y-%m-%d')
                or i.month is distinct from substr(i.day,1,7) then 1 else 0 end)
                from i join raw using(mvt_id)''').fetchone()
            assert n==u==record['rows'] and wrong==0
            nfeat,errors = con.execute(f'''select count(*),sum(case when
                f.{FEATURES[0]} is distinct from cast(e.reference as float)
                or f.{FEATURES[1]} is distinct from cast(e.reference-e.source_level as float)
                then 1 else 0 end) from i join f using(mvt_id) left join e
                on i.airport=e.airport and i.runway=e.runway and i.stand=e.stand and i.month=e.month
                and (cast(right(i.utc_month,2) as int)-1)%6=(cast(right(i.month,2) as int)-1)%6
                and left(i.utc_month,4)=left(i.month,4)''').fetchone()
            assert nfeat==n and errors==0
            ncells,bad_source = con.execute('''select count(*),sum(case when o.n!=s.n
                or abs(o.reference-60*s.ref_min/s.n)>1e-8 then 1 else 0 end)
                from o join s using(airport,runway,day)''').fetchone()
            assert ncells==con.execute('select count(*) from o').fetchone()[0] and bad_source==0
            # Independently aggregate raw metadata on each fitted public-source day.
            con.execute('''create temp table counts as select airport,runway,day,month,stand,count(*) n
                from i where stand is not null and runway is not null
                and (cast(right(utc_month,2) as int)-1)%6=(cast(right(month,2) as int)-1)%6
                and left(utc_month,4)=left(month,4) group by all''')
            replay = con.execute('''with x as (select c.airport,c.runway,c.day,sum(c.n) metadata_n,
                sum(c.n*e.reference)/sum(c.n) full_fit from counts c join o using(airport,runway,day)
                join e on c.airport=e.airport and c.runway=e.runway and c.month=e.month and c.stand=e.stand group by all)
                select count(*),sum(case when x.metadata_n!=o.metadata_n or abs(x.full_fit-o.full_fit)>1e-8
                then 1 else 0 end),max(abs(x.full_fit-o.full_fit)) from x join o using(airport,runway,day)''').fetchone()
            assert replay[0]==ncells and replay[1]==0,replay
            means = con.execute('''with a as (select airport,runway,month,avg(reference) source_level from o group by all)
                select count(*),sum(case when abs(e.source_level-a.source_level)>1e-8 then 1 else 0 end)
                from e join a using(airport,runway,month)''').fetchone()
            assert means[0]>0 and means[1]==0
        inputs = pd.read_parquet(OUT/f'inputs{name}.parquet').sample(2000,random_state=215)
        estimates = pd.read_parquet(OUT/f'estimates{name}.parquet')
        actual = build(inputs,estimates).set_index('mvt_id').sort_index()
        expected = pd.read_parquet(OUT/f'features{name}.parquet').set_index('mvt_id').loc[inputs.mvt_id].sort_index()
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        checks.append(dict(name=name,rows=n,sql_feature_entries=2*n,public_cells=ncells,
            full_estimate_reconstruction_max_error=replay[2],permuted_rows=2000))
    value = dict(status='pass',source=art(__file__),manifest=art(OUT/'manifest.json'),checks=checks,
        metadata_calendar_every_feature_public_targets_coefficient_reconstruction_sql=True,
        no_departure_labels_read=True,not_taxi_cv=True)
    write_json(OUT/'audit.json',value)
    print(value,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','audit'])
    a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
