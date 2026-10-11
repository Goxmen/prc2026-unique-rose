"""Public daily taxi means and runway shares, joined without departure labels."""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_daily_taxi as source
from records import write_json

ROOT=source.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_public_daily_taxi_features'
FEATURES=[f'pub_taxi_{level}_{metric}' for level in ['runway','airport']
          for metric in ['mean','reference','additional','count']]+['pub_taxi_runway_share','pub_taxi_runway_minus_airport']
art,read,verify=source.art,source.read,source.verify


def load(path, cached=False):
    if cached:
        f=pd.read_parquet(path,columns=['mvt_id','airport','takeoff','runway'])
    else:
        f=pd.read_parquet(path,columns=['MVT_ID_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt'],
            filters=[('PHASE_mvt','==','DEP')]).rename(columns=dict(MVT_ID_mvt='mvt_id',ADEP_mvt='airport',
            MVT_TIME_UTC_mvt='takeoff',RUNWAY_mvt='runway'))
    f['local_day']=''
    for airport,zone in source.ZONES.items():
        m=f.airport.eq(airport)
        f.loc[m,'local_day']=f.loc[m,'takeoff'].dt.tz_convert(zone).dt.strftime('%Y-%m-%d')
    assert f.mvt_id.is_unique and f.local_day.str.len().eq(10).all()
    return f


def build(inputs, public):
    assert set(inputs).issubset({'mvt_id','airport','takeoff','runway','local_day'})
    values=['n','total_min','ref_min','add_min']
    g=public.groupby(['airport','day'],as_index=False)[values].sum()
    f=inputs[['mvt_id','airport','local_day','runway']].merge(
        public[['airport','day','runway']+values].rename(columns={'day':'local_day'}),
        on=source.KEY,how='left',validate='many_to_one')
    f=f.merge(g.rename(columns={'day':'local_day'}),on=['airport','local_day'],
        how='left',validate='many_to_one',suffixes=('_runway','_airport'))
    out=f[['mvt_id']].copy()
    for level in ['runway','airport']:
        for name,column in [('mean','total_min'),('reference','ref_min'),('additional','add_min')]:
            out[f'pub_taxi_{level}_{name}']=60*f[f'{column}_{level}']/f[f'n_{level}']
        out[f'pub_taxi_{level}_count']=f[f'n_{level}']
    out['pub_taxi_runway_share']=f.n_runway/f.n_airport
    out['pub_taxi_runway_minus_airport']=out.pub_taxi_runway_mean-out.pub_taxi_airport_mean
    out[FEATURES]=out[FEATURES].astype('float32')
    assert not np.isinf(out[FEATURES]).any().any()
    np.testing.assert_array_equal(out.mvt_id,inputs.mvt_id)
    return out[['mvt_id']+FEATURES]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(), 'Preserve completed inputs'
    assert read(source.REPORT/'public_daily_taxi_source_audit.json')['status']=='pass'
    public=pd.read_parquet(source.SOURCE/'daily_runway.parquet')
    records={}
    for name,path,cached,expected in [('2025',source.CACHE,True,2085047),('ranking',ROOT/'data/ranking.parquet',False,344841)]:
        inputs=load(path,cached);features=build(inputs,public)
        assert len(features)==expected
        inputs.to_parquet(OUT/f'inputs{name}.parquet',index=False)
        features.to_parquet(OUT/f'features{name}.parquet',index=False)
        records[name]=dict(raw=art(path),inputs=art(OUT/f'inputs{name}.parquet'),
            features=art(OUT/f'features{name}.parquet'),rows=expected,
            missing={c:int(features[c].isna().sum()) for c in FEATURES})
    write_json(OUT/'manifest.json',dict(source=art(__file__),public=art(source.SOURCE/'daily_runway.parquet'),
        public_manifest=art(source.SOURCE/'manifest.json'),public_audit=art(source.REPORT/'public_daily_taxi_source_audit.json'),
        records=records,columns=FEATURES,labels_read=False,
        design='Ten fixed raw public runway/airport daily inputs; retain missing values; no calibration or target features'))
    print(records,flush=True)


def audit():
    assert not (OUT/'audit.json').exists(), 'Preserve completed audit'
    manifest=read(OUT/'manifest.json'); assert manifest['source']==art(__file__)
    for a in ['public','public_manifest','public_audit']:verify(manifest[a])
    checks=[]
    for name,record in manifest['records'].items():
        for a in ['raw','inputs','features']:verify(record[a])
        with duckdb.connect(config={'threads':1}) as con:
            con.read_parquet(str(OUT/f'inputs{name}.parquet')).create_view('r')
            con.read_parquet(str(OUT/f'features{name}.parquet')).create_view('f')
            con.read_parquet(str(source.SOURCE/'daily_runway.parquet')).create_view('s')
            zone='case r.airport '+' '.join(f"when '{a}' then '{z}'" for a,z in source.ZONES.items())+' end'
            n,u,wrong=con.execute(f'''select count(*),count(distinct mvt_id),sum(case when
                r.local_day != strftime(r.takeoff AT TIME ZONE ({zone}),'%Y-%m-%d') then1 else0 end)
                from r'''.replace('then1','then 1').replace('else0','else 0')).fetchone()
            assert n==u==record['rows'] and wrong==0
            expressions=[]
            for level,alias in [('runway','s'),('airport','a')]:
                for metric,column in [('mean','total_min'),('reference','ref_min'),('additional','add_min')]:
                    expressions.append((f'pub_taxi_{level}_{metric}',f'60*{alias}.{column}/{alias}.n'))
                expressions.append((f'pub_taxi_{level}_count',f'{alias}.n'))
            expressions.extend([('pub_taxi_runway_share','s.n/a.n'),
                ('pub_taxi_runway_minus_airport','60*s.total_min/s.n-60*a.total_min/a.n')])
            failures=' + '.join(f'(case when f.{col} is distinct from cast(({expr}) as float) then 1 else 0 end)'
                                for col,expr in expressions)
            count,errors=con.execute(f'''with a as (select airport,day,sum(n) n,sum(total_min) total_min,
                sum(ref_min) ref_min,sum(add_min) add_min from s group by airport,day)
                select count(*),sum({failures}) from r join f using(mvt_id)
                left join s on r.airport=s.airport and r.local_day=s.day and r.runway=s.runway
                left join a on r.airport=a.airport and r.local_day=a.day''').fetchone()
            assert count==n and errors==0,(name,count,errors)
        # Reconstruction is invariant to input order and never accepts labels.
        rows=pd.read_parquet(OUT/f'inputs{name}.parquet').sample(n=2000,random_state=215)
        public=pd.read_parquet(source.SOURCE/'daily_runway.parquet')
        actual=build(rows,public).sort_values('mvt_id').reset_index(drop=True)
        expected=pd.read_parquet(OUT/f'features{name}.parquet').set_index('mvt_id').loc[rows.mvt_id].reset_index().sort_values('mvt_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        checks.append(dict(name=name,rows=n,sql_feature_entries=n*10,permuted_rows=2000))
    report=dict(status='pass',independent_sql_local_calendar_and_every_feature=True,
        missing_source_values_preserved=True,no_departure_labels_read=True,checks=checks,
        manifest=art(OUT/'manifest.json'),source=art(__file__))
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','audit'])
    a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
