"""Extend retained ERA5 inputs to February/June in a separate cache namespace.

Only missing airport-month requests are fetched. The original 30 sources and
manifest stay immutable. No model fitting or predictions are performed here.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import external_conditions as wx
from records import sha256, write_json

ROOT=wx.ROOT
CACHE=ROOT/'data/external/conditions_final_20261012'
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_weather_features'
MANIFEST=CACHE/'manifest.json'
PERIODS=[('2026-02','2026-01-29','2026-02-28'),('2026-06','2026-05-29','2026-06-30')]
DESIGN=dict(variables=wx.VARIABLES,columns=wx.WEATHER,periods=PERIODS,
    source='Unchanged Open-Meteo ERA5 archive API, GMT, wind m/s, same airport coordinates',
    features='Unchanged per-file hourly_features and backward UTC weather_join; original overlap policy',
    old_cache='Read-only original30 source files; never replace or redownload',
    no_labels=True,no_fit=True,no_prediction=True)


def art(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),bytes=path.stat().st_size)


def verify(a):
    p=ROOT/a['path'];assert sha256(p)==a['sha256'],str(p)
    return p


def read(p):return json.loads(Path(p).read_text())


def original_sources():
    manifest=wx.CACHE/'weather_manifest.json'
    records=read(manifest)
    assert len(records)==30
    for r in records:verify(r)
    return manifest,records


def download():
    if MANIFEST.exists():guard();return
    original,records=original_sources()
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    frozen=OUT/'download_identity.json'
    identity=dict(design=DESIGN,source=art(__file__),weather_source=art(wx.__file__),
        original_manifest=art(original),original_records=records)
    if frozen.exists():assert read(frozen)==json.loads(json.dumps(identity))
    else:write_json(frozen,identity)
    additional=[]
    for ap,(lat,lon) in wx.AIRPORTS.items():
        for month,start,end in PERIODS:
            assert not (wx.CACHE/'weather'/f'{ap}_{month}.json').exists()
            params=dict(latitude=lat,longitude=lon,start_date=start,end_date=end,
                hourly=','.join(wx.VARIABLES),models='era5',timezone='GMT',wind_speed_unit='ms')
            url='https://archive-api.open-meteo.com/v1/archive?'+urlencode(params)
            path=CACHE/'weather'/f'{ap}_{month}.json';receipt=path.with_suffix('.receipt.json')
            if receipt.exists():
                r=read(receipt);assert r['url']==url;verify(r)
            else:
                obj=json.loads(wx.fetch(url,path))
                hourly=obj['hourly'];times=pd.to_datetime(hourly['time'],utc=True)
                expected=pd.date_range(start,pd.Timestamp(end)+pd.Timedelta(hours=23),freq='h',tz='UTC')
                np.testing.assert_array_equal(times,expected)
                assert obj['utc_offset_seconds']==0 and obj['hourly_units']==records[0]['units']
                assert set(hourly)=={'time',*wx.VARIABLES}
                assert np.isfinite(pd.DataFrame(hourly)[wx.VARIABLES].to_numpy(float)).all()
                r=dict(**art(path),url=url,airport=ap,month=month,start=start,end=end,
                    rows=len(times),units=obj['hourly_units'],grid_latitude=obj['latitude'],
                    grid_longitude=obj['longitude'])
                write_json(receipt,r)
            additional.append(r)
            print(dict(airport=ap,month=month,rows=r['rows'],sha256=r['sha256']),flush=True)
    assert art(original)==identity['original_manifest']
    for r in records:verify(r)
    write_json(MANIFEST,dict(identity=art(frozen),design=DESIGN,original_manifest=art(original),
        original=records,additional=additional,source=art(__file__),weather_source=art(wx.__file__)))


def guard():
    m=read(MANIFEST)
    assert m['design']==json.loads(json.dumps(DESIGN)) and m['source']==art(__file__)
    assert m['weather_source']==art(wx.__file__)
    verify(m['identity']);verify(m['original_manifest'])
    assert len(m['original'])==30 and len(m['additional'])==20
    for r in m['original']+m['additional']:verify(r)
    return m


def table(m):
    tables=[];overlap_count=0
    for ap in wx.AIRPORTS:
        sources=[r for r in m['original']+m['additional'] if Path(r['path']).name.startswith(ap+'_')]
        raw=[];features=[]
        for r in sources:
            h=pd.DataFrame(read(verify(r))['hourly']);h['time']=pd.to_datetime(h.time,utc=True)
            raw.append(h);features.append(wx.hourly_features(h))
        values=pd.concat(raw)
        repeated=values.loc[values.time.duplicated(keep=False)]
        # Revised observations must not silently change retained historical inputs.
        different=repeated.groupby('time',sort=False)[wx.VARIABLES].nunique(dropna=False)
        assert different.le(1).all().all(),f'Changed ERA5 overlap: {ap}'
        overlap_count+=len(different)
        f=pd.concat(features).sort_values('om_precipitation_48h',na_position='last')
        tables.append(f.drop_duplicates('time').assign(airport=ap))
    return pd.concat(tables).sort_values('time'),overlap_count


def raw(path):
    f=pd.read_parquet(path,columns=['MVT_ID_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','PHASE_mvt'])
    f=f.loc[f.PHASE_mvt.eq('DEP')].drop(columns='PHASE_mvt').reset_index(drop=True)
    assert f.MVT_ID_mvt.is_unique and f.MVT_TIME_UTC_mvt.notna().all()
    return f


def prepare():
    m=guard();dest=OUT/'features.json';assert not dest.exists()
    joined,overlaps=table(m)
    old=raw(ROOT/'data/ranking.parquet')
    a=wx.weather_join(old,wx.weather_table());b=wx.weather_join(old,joined)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    final=raw(ROOT/'data/final_ranking.parquet')
    template=pd.read_parquet(ROOT/'data/final_submitting.parquet',columns=['MVT_ID_mvt']).MVT_ID_mvt
    assert len(final)==len(template)==670790
    np.testing.assert_array_equal(np.sort(final.MVT_ID_mvt),np.sort(template))
    final=final.set_index('MVT_ID_mvt').loc[template].reset_index()
    values=wx.weather_join(final,joined)
    assert list(values)==wx.WEATHER and not values.isna().any().any()
    assert np.isfinite(values.to_numpy()).all()
    values.insert(0,'mvt_id',final.MVT_ID_mvt.to_numpy())
    final.to_parquet(OUT/'raw_final.parquet',index=False)
    values.to_parquet(OUT/'features_final.parquet',index=False)
    joined.to_parquet(OUT/'hourly.parquet',index=False)
    write_json(dest,dict(identity=art(MANIFEST),source=art(__file__),weather_source=art(wx.__file__),
        input=art(ROOT/'data/final_ranking.parquet'),template=art(ROOT/'data/final_submitting.parquet'),
        raw=art(OUT/'raw_final.parquet'),output=art(OUT/'features_final.parquet'),hourly=art(OUT/'hourly.parquet'),
        rows=len(final),columns=wx.WEATHER,old_rows=len(old),all_old_weather_exact=True,
        all_overlapping_raw_values_exact=True,overlap_airport_hours=overlaps,no_labels=True,no_fit=True))
    print(dict(prepared=True,rows=len(final),old_weather_exact=True,overlap_airport_hours=overlaps),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['download','prepare'])
    with threadpool_limits(1):globals()[p.parse_args().stage]()
