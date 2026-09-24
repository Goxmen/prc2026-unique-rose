"""Exploratory step-4 snow-duration check; preserves the original coarse audit."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent))
import investigate_arrival_shift as base


def snowfall_grid(metar):
    reports=metar[metar.station.eq('EHAM')].copy()
    reports['valid']=base.weather._utc_naive(reports.valid)
    # METAR ties use deterministic text order; classify any simultaneous report
    # containing snow as snow rather than choosing one arbitrarily.
    tokens=base.weather._codes(reports.wxcodes)
    reports['snow']=base.weather._any(tokens,lambda x:'SN' in x or 'SG' in x).astype(float)
    reports=reports.groupby('valid',as_index=False).snow.max().sort_values('valid')
    grid=pd.DataFrame({'sample':pd.date_range(reports.valid.min().floor('30min'),reports.valid.max().ceil('30min'),freq='30min')})
    grid=pd.merge_asof(grid,reports,left_on='sample',right_on='valid',direction='backward')
    known=(grid['sample']-grid.valid).le(pd.Timedelta(minutes=90))&grid.valid.notna()
    grid.loc[~known,'snow']=np.nan
    grid['snow_fraction24h']=grid.snow.rolling(48,min_periods=40).mean()
    grid['known_samples24h']=grid.snow.rolling(48,min_periods=1).count()
    assert not (grid.valid>grid['sample']).any()
    return grid


def main():
    audit=json.loads((base.OUT/'audit.json').read_text())
    feature_path=base.CACHE/'arrival_features.parquet'
    assert base.sha(feature_path)==audit['features']['sha256']
    arr=pd.read_parquet(feature_path);arr=arr[arr.airport.eq('EHAM')].copy()
    grid=snowfall_grid(pd.read_parquet(base.weather.METAR))
    arr['landing']=base.weather._utc_naive(arr.landing)
    arr=pd.merge_asof(arr.sort_values('landing'),grid[['sample','snow_fraction24h','known_samples24h']],left_on='landing',right_on='sample',direction='backward')
    arr['snow_duration']=pd.cut(arr.snow_fraction24h,[-np.inf,0,.1,.25,.5,1],
        labels=['none','0_to_10pct','10_to_25pct','25_to_50pct','over_50pct']).astype('string').fillna('missing')
    arr['snow_now']=arr.strict_snow.map({0.:'no',1.:'yes'}).fillna('missing')
    arr.to_parquet(base.CACHE/'EHAM_snow_duration.parquet',index=False,compression='zstd')
    grid.to_parquet(base.CACHE/'EHAM_snow_grid.parquet',index=False,compression='zstd')
    groups=[]
    for (month,category),g in arr.groupby(['month','snow_duration']):
        groups.append(dict(month=month,snow_duration=category,**base.summarize(g)))
    pd.DataFrame(groups).to_csv(base.OUT/'snow_duration_groups.csv',index=False)
    stages={'snow_duration':['snow_duration'],
        'plus_current_snow':['snow_duration','snow_now'],
        'plus_weather':['snow_duration','snow_now','temperature','visibility','wind'],
        'plus_traffic_runway':['snow_duration','snow_now','temperature','visibility','wind','traffic','runway']}
    results=[];intervals=[];cells_out=[]
    with threadpool_limits(limits=1):
        for ref in ['2025-01','all_2025']:
            source=arr[arr.month.eq(ref) if ref!='all_2025' else arr.month.str.startswith('2025')]
            target=arr[arr.month.eq('2026-01')]
            source,target,_=base.bins(source,target)
            print('Duration comparison',ref,flush=True)
            for stage,columns in stages.items():
                for threshold in [1800,3600,7200]:
                    row,ci,cells=base.standardize(source,target,columns,threshold)
                    results.append(dict(reference=ref,stage=stage,**row))
                    intervals.extend(dict(reference=ref,stage=stage,threshold=threshold,**x) for x in ci)
                    if threshold==1800:
                        cells['reference']=ref;cells['stage']=stage;cells_out.append(cells)
    pd.DataFrame(results).to_csv(base.OUT/'snow_duration_conditional.csv',index=False)
    pd.concat(cells_out,ignore_index=True).to_csv(base.OUT/'snow_duration_cells.csv',index=False)
    base.write(base.OUT/'snow_duration_intervals.json',intervals)
    meta=dict(source_sha256=base.sha(Path(__file__)),base_source_sha256=base.sha(Path(base.__file__)),
        design_sha256=base.sha(base.OUT/'severity_followup_design.json'),
        inputs={str(feature_path.relative_to(base.ROOT)):base.sha(feature_path),str(base.weather.METAR.relative_to(base.ROOT)):base.sha(base.weather.METAR)},
        outputs={p.name:base.sha(p) for p in base.OUT.glob('snow_duration*') if p.suffix in ['.csv','.json'] and p.name!='snow_duration_audit.json'},
        cache={p.name:base.sha(p) for p in base.CACHE.glob('EHAM_snow*.parquet')})
    base.write(base.OUT/'snow_duration_audit.json',meta)
    print('Snow-duration follow-up complete',flush=True)


if __name__=='__main__':main()
