"""Apply the existing monthly 5% source-quality gate to daily taxi correction.

Declared after the daily correction's winter failure diagnosis. No new gate
threshold or airport selection; all trained parameters and predictions reused.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_daily_taxi as original
from records import write_json

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_public_daily_taxi_quality'
monthly=original.constraints.monthly
art,read,verify=original.art,original.read,original.verify
DESIGN=dict(parent='production-rounded v76 and selected refined parent',
    change='existing monthly invalid-share<0.05 gate also limits daily runway taxi corrections',
    threshold_provenance='unchanged monthly_taxi_reconcile.DESIGN maximum_invalid_share',
    parameters='all original daily fit coefficients/noise and shifts unchanged',
    calendar='airport local takeoff month; unknown source quality leaves parent unchanged',
    interpretation='follow-up after winter source-population failure diagnosis; reused Jan/Jul development',
    no_new_threshold_search=True,no_airport_selection=True,no_new_fit=True,no_production=True,no_upload=True)


def guard():
    assert monthly.DESIGN['maximum_invalid_share']==.05
    assert read(original.OUT/'audit.json')['status']=='pass'
    paths=[Path(__file__),Path(original.__file__),Path(monthly.__file__),original.OUT/'identity.json',
        original.OUT/'scored.parquet',original.OUT/'result.json',original.OUT/'audit.json',
        original.OUT/'parameters.json',original.OUT/'diagnosis/result.json',
        original.base.SOURCE/'taxi_out_additional_time_2025.csv']
    value=dict(design=DESIGN,sources=[art(p) for p in paths])
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'identity.json'
    if path.exists():assert read(path)==value
    else:write_json(path,value)


def screen():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed screen'
    f=pd.read_parquet(original.OUT/'scored.parquet')
    pub=monthly.public(2025)
    quality=f[['airport','local_day']].assign(local_month=f.local_day.str[:7]).merge(
        pub[['airport','local_month','invalid']],on=['airport','local_month'],how='left',validate='many_to_one')
    f['invalid_share']=quality.invalid.to_numpy()
    f['quality_eligible']=f.invalid_share.lt(monthly.DESIGN['maximum_invalid_share'])
    for name in ['v76','refined']:
        f[name+'_quality']=np.where(f.quality_eligible,f[name+'_daily'],f[name])
    names=['v76','v76_daily','v76_quality','refined','refined_daily','refined_quality']
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in names} for m,g in f.groupby('month')}
    comparisons=[(a,b) for name in ['v76','refined'] for a,b in [(name,name+'_quality'),(name+'_daily',name+'_quality')]]
    paired={f'{b}_vs_{a}':original.parent.original.airport.parent.ts.paired_bootstrap(
        f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in comparisons}
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in names},
        paired=paired,identity=art(OUT/'identity.json'),prediction=art(OUT/'scored.parquet'),
        source_quality_excluded=int((~f.quality_eligible).sum()),
        reverted_changed_rows=int((~f.quality_eligible & f.refined_daily.ne(f.refined)).sum()))
    write_json(OUT/'result.json',result);print(result,flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['prediction'])
    with duckdb.connect(config={'threads':1}) as con:
        con.read_parquet(str(OUT/'scored.parquet')).create_view('c')
        con.read_parquet(str(original.OUT/'scored.parquet')).create_view('p')
        con.read_csv(str(original.base.SOURCE/'taxi_out_additional_time_2025.csv')).create_view('s')
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from p.y or c.day is distinct from p.day or c.month is distinct from p.month
            or c.v76 is distinct from p.v76 or c.refined is distinct from p.refined
            or c.v76_daily is distinct from p.v76_daily or c.refined_daily is distinct from p.refined_daily
            or c.invalid_share is distinct from ((s.TF-s.VALID_FL)/s.TF)
            or c.quality_eligible != coalesce(((s.TF-s.VALID_FL)/s.TF)<0.05,false)
            or c.v76_quality != (case when c.quality_eligible then p.v76_daily else p.v76 end)
            or c.refined_quality != (case when c.quality_eligible then p.refined_daily else p.refined end)
            then 1 else 0 end) from c join p using(mvt_id) left join s
            on c.airport=s.APT_ICAO and substr(c.local_day,1,7)=cast(s.YEAR as varchar)||'-'||lpad(cast(s.MONTH_NUM as varchar),2,'0')''').fetchone()
        assert n==u==344419 and errors==0,(n,u,errors)
        for column in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({column}-y,2))) from c group by month').fetchall()
            for m,v in values:np.testing.assert_allclose(v,result['scores'][m][column],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][column],atol=1e-8,rtol=0)
    value=dict(status='pass',rows=n,independent_sql_source_quality_gate_truth_calendar_controls_scores=True,
        no_new_fits=True,identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),source=art(__file__))
    write_json(OUT/'audit.json',value);print(value,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['screen','audit'])
    a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
