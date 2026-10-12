"""Independent source/calendar/SQL audit of retained and new ERA5 final inputs."""
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs,urlparse

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
from records import sha256,write_json

OUT=ROOT/'VALIDATION/runs/v3_20261012_final_weather_features'
VARIABLES=['temperature_2m','precipitation','rain','snowfall','wind_speed_10m']
COLS=['om_'+c for c in VARIABLES]+[f'om_{v}_{h}h' for v in ['precipitation','snowfall'] for h in [6,24,48]]
UNITS=dict(time='iso8601',temperature_2m='°C',precipitation='mm',rain='mm',snowfall='cm',wind_speed_10m='m/s')


def read(path):return json.loads(Path(path).read_text())


def verify(a):
    path=ROOT/a['path'];assert sha256(path)==a['sha256'],str(path)
    return path


def artifact(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def main():
    features=read(OUT/'features.json');manifest_path=verify(features['identity']);manifest=read(manifest_path)
    for key in ['source','weather_source','input','template','raw','output','hourly']:verify(features[key])
    frozen=read(verify(manifest['identity']))
    verify(manifest['original_manifest']);verify(manifest['source']);verify(manifest['weather_source'])
    assert manifest['original']==frozen['original_records']
    assert len(manifest['original'])==30 and len(manifest['additional'])==20
    assert features['columns']==COLS and features['all_old_weather_exact']
    old_coordinates={};all_sources=[];source_checks=[];additional=set()
    for i,record in enumerate(manifest['original']+manifest['additional']):
        path=verify(record);obj=read(path);query=parse_qs(urlparse(record['url']).query)
        assert urlparse(record['url']).netloc=='archive-api.open-meteo.com'
        assert urlparse(record['url']).path=='/v1/archive'
        assert query['hourly']==[','.join(VARIABLES)] and query['models']==['era5']
        assert query['timezone']==['GMT'] and query['wind_speed_unit']==['ms']
        assert obj['utc_offset_seconds']==0 and obj['hourly_units']==UNITS==record['units']
        ap,period=path.stem.split('_',1)
        coords=(query['latitude'][0],query['longitude'][0],obj['latitude'],obj['longitude'])
        if i<30:
            assert period in ['2025','2026-01','2026-07']
            assert 'conditions_20261008/weather' in str(path)
            if ap in old_coordinates:assert old_coordinates[ap]==coords
            else:old_coordinates[ap]=coords
        else:
            assert period in ['2026-02','2026-06'] and 'conditions_final_20261012/weather' in str(path)
            assert old_coordinates[ap]==coords
            expected_dates={'2026-02':('2026-01-29','2026-02-28'),'2026-06':('2026-05-29','2026-06-30')}
            assert (query['start_date'][0],query['end_date'][0])==expected_dates[period]
            additional.add((ap,period))
        h=pd.DataFrame(obj['hourly']);assert list(h)==['time',*VARIABLES]
        h['time']=pd.to_datetime(h.time,utc=True)
        expected=pd.date_range(query['start_date'][0],pd.Timestamp(query['end_date'][0])+pd.Timedelta(hours=23),freq='h',tz='UTC')
        np.testing.assert_array_equal(h.time,expected)
        assert np.isfinite(h[VARIABLES].to_numpy(float)).all()
        assert (h[['precipitation','rain','snowfall','wind_speed_10m']]>=0).all().all()
        all_sources.append(h.assign(airport=ap,file_key=i))
        source_checks.append(dict(path=record['path'],rows=len(h),first=str(h.time.iloc[0]),last=str(h.time.iloc[-1])))
    assert len(old_coordinates)==10 and additional=={(a,m) for a in old_coordinates for m in ['2026-02','2026-06']}
    h=pd.concat(all_sources,ignore_index=True)
    dup=h.loc[h.duplicated(['airport','time'],keep=False)]
    assert dup.groupby(['airport','time'])[VARIABLES].nunique().le(1).all().all()
    rows=pd.read_parquet(verify(features['raw']));f=pd.read_parquet(verify(features['output']))
    template=pd.read_parquet(verify(features['template']),columns=['MVT_ID_mvt']).MVT_ID_mvt
    assert len(rows)==len(f)==rows.MVT_ID_mvt.nunique()==f.mvt_id.nunique()==len(template)==670790
    np.testing.assert_array_equal(rows.MVT_ID_mvt,template);np.testing.assert_array_equal(f.mvt_id,template)
    assert set(rows.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m'))=={'2026-01','2026-02','2026-06','2026-07'}
    direct=[f'{v} AS om_{v}' for v in VARIABLES]
    rolled=[]
    for v in ['precipitation','snowfall']:
        for n in [6,24,48]:
            window=f'PARTITION BY file_key ORDER BY time ROWS BETWEEN {n-1} PRECEDING AND CURRENT ROW'
            rolled.append(f'CASE WHEN count(*) OVER ({window})={n} THEN sum({v}) OVER ({window}) END AS om_{v}_{n}h')
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        con.register('source',h);con.register('flight',rows)
        n,u,bad=con.execute('''SELECT count(*),count(distinct f.MVT_ID_mvt),sum(
            (f.ADEP_mvt IS DISTINCT FROM r.ADEP_mvt)::INT+
            (f.MVT_TIME_UTC_mvt IS DISTINCT FROM r.MVT_TIME_UTC_mvt)::INT)
            FROM flight f JOIN read_parquet(?) r USING (MVT_ID_mvt)
            WHERE r.PHASE_mvt='DEP' ''',[str(verify(features['input']))]).fetchone()
        assert n==u==670790 and bad==0
        query='''WITH calc AS (SELECT airport,time,file_key,'''+','.join(direct+rolled)+''' FROM source),
          chosen AS (SELECT * FROM calc QUALIFY row_number() OVER
            (PARTITION BY airport,time ORDER BY om_precipitation_48h ASC NULLS LAST,file_key)=1)
          SELECT f.MVT_ID_mvt AS mvt_id,'''+','.join('c.'+x for x in COLS)+'''
          FROM flight f LEFT JOIN chosen c ON f.ADEP_mvt=c.airport
            AND date_trunc('hour',f.MVT_TIME_UTC_mvt)=c.time'''
        expected=con.execute(query).df().set_index('mvt_id').loc[template]
    a=f[COLS].to_numpy(np.float32);b=expected[COLS].to_numpy(np.float32)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_array_equal(a[:,:5],b[:,:5])
    # Independent window summation order may differ in the last floating-point bit.
    bound=4*np.finfo(np.float32).eps*np.maximum(1,np.abs(b[:,5:]))
    error=np.abs(a[:,5:].astype(float)-b[:,5:].astype(float));assert np.all(error<=bound)
    checks=dict(status='pass',features_identity=artifact(OUT/'features.json'),source=artifact(Path(__file__)),
        retained_original_manifest=manifest['original_manifest'],original_files=30,new_files=20,
        source_hourly_rows=len(h),source_checks=source_checks,overlap_raw_values_exact=True,
        rows=n,columns=COLS,sql_feature_values=n*len(COLS),direct_values_float32_exact=True,
        rolling_float32_exact=bool(np.array_equal(a[:,5:],b[:,5:])),rolling_max_abs=float(error.max()),
        rolling_tolerance='4*float32 epsilon*max(1,abs(reference))',raw_source_and_utc_floor_exact=True,
        final_template_order_exact=True,no_labels=True,no_fit=True)
    write_json(OUT/'audit.json',checks)
    print(json.dumps({k:v for k,v in checks.items() if k!='source_checks'}),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
