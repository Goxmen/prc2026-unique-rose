"""Describe Rome's two public measurements and joint-inference complexity.

This is source calibration, not taxi prediction. True labels define source
errors on permitted training months; they never produce a deployment rule.
The taxi source uses local takeoff days and a truncated stand/runway cohort;
the departure-delay source uses UTC off-block days and the IFR cohort. Their
same-named days are not identical flight sets. No new data downloads or fits
of a taxi model, and no CV gain or candidate is claimed.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_public_source_probe'
FOLDS={'2025-01':[f'2025-{m:02d}' for m in range(3,12)],
       '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
DESIGN=dict(scope='LIRF 2025 only, permitted outer training months for all covariance estimates',
    taxonomy='source discrepancies, not prediction residuals; same calendar labels have different source cohorts',
    calibration='fixed Huber regressions of three source discrepancies on intercept and input cohort counts',
    scale='training MAD with count1/sum60/delay60 floors; jointly retain residuals within3MAD',
    questions=['noise dependence of two sources','number of long-flight states per public source day',
               'calendar/cohort mismatch preventing naive same-cell joint inference'],
    no_taxi_fit=True,no_candidate=True,no_upload=True,no_gain_claim=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def art(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    import duckdb
    assert not (OUT/'identity.json').exists(),'Preserve prepared probe'
    raw=ROOT/'VALIDATION/runs/v3_20261011_daily_offblock_alignment/rows2025.parquet'
    taxi=ROOT/'tmp/research_20261011/daily_taxi_source/training_rows.parquet'
    pubtaxi=ROOT/'data/external/eurocontrol_daily_taxi_20261011/daily_runway.parquet'
    pubdelay=ROOT/'data/external/eurocontrol_performance_20261011/all_pre_departure_delays_2025.csv'
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in [('r',raw),('t',taxi),('p',pubtaxi)]:con.read_parquet(str(path)).create_view(name)
        n,u,wrong=con.execute('''select count(*),count(distinct r.mvt_id),sum(case when r.y!=t.y
            or r.true_day!=t.utc_block or r.day!=t.utc_takeoff or r.airport!=t.airport then 1 else 0 end)
            from r join t using(mvt_id) where r.airport='LIRF' ''').fetchone()
        assert n==u and wrong==0
        rows=con.execute('''select r.mvt_id,r.month,r.gap,r.has_id,r.fltrule,r.y,r.true_day,
            t.local_takeoff local_day,t.runway,t.stand is not null and t.runway is not null eligible
            from r join t using(mvt_id) where r.airport='LIRF' ''').df()
        a=con.execute('''select local_takeoff as "day",count(*) taxi_cohort,
            count(*) filter(where y between 0 and 7200) taxi_n,
            sum(case when y between 0 and 7200 then y else 0 end) taxi_sum
            from t where airport='LIRF' and stand is not null and runway is not null group by 1''').df()
        b=con.execute('''select true_day as "day",count(*) ifr_cohort,sum(greatest(gap-y,0)) delay_sum,
            count(*) filter(where gap>3600) long_flights,
            count(*) filter(where gap>3600 and gap<50400) mixture_gap_flights
            from r where airport='LIRF' and fltrule is distinct from 'V' group by 1''').df()
        c=con.execute("select day,sum(n) source_taxi_n,60*sum(total_min) source_taxi_sum from p where airport='LIRF' group by day").df()
    d=pd.read_csv(pubdelay)
    d=d[d.APT_ICAO.eq('LIRF')].rename(columns={'FLT_DATE':'day','FLT_DEP_IFR_2':'source_ifr_n'})
    d['source_delay_sum']=60*d.DLY_ALL_PRE_2
    cells=a.merge(b,on='day',validate='one_to_one').merge(c,on='day',validate='one_to_one').merge(
        d[['day','source_ifr_n','source_delay_sum']],on='day',validate='one_to_one')
    for key in ['taxi_n','taxi_sum','delay_sum']:
        cells[key+'_error']=cells[key]-cells['source_'+key]
    # Independent pandas reductions check every exported SQL sum and count.
    rr=rows[rows.eligible].copy();rr['valid']=rr.y.between(0,7200);rr['total']=np.where(rr.valid,rr.y,0)
    aa=rr.groupby('local_day').agg(taxi_cohort=('mvt_id','size'),taxi_n=('valid','sum'),taxi_sum=('total','sum'))
    rr=rows[~rows.fltrule.eq('V')].copy();rr['total']=np.maximum(rr.gap-rr.y,0)
    bb=rr.groupby('true_day').agg(ifr_cohort=('mvt_id','size'),delay_sum=('total','sum'))
    for frame in [aa,bb]:
        for col in frame:
            np.testing.assert_array_equal(cells[col],frame.loc[cells.day,col])
    OUT.mkdir(parents=True,exist_ok=True)
    cells.sort_values('day').to_parquet(OUT/'cells.parquet',index=False)
    # Compact training membership includes a label-derived off-block date.
    rows[['mvt_id','month','gap','fltrule','true_day','local_day','runway','eligible']].to_parquet(OUT/'membership.parquet',index=False)
    identity=dict(design=DESIGN,source=art(__file__),created_at=datetime.now(timezone.utc).isoformat(),
        sources=[art(p) for p in [raw,taxi,pubtaxi,pubdelay]],
        artifacts=[art(OUT/p) for p in ['cells.parquet','membership.parquet']],
        row_count=n,source_cells=len(cells),independent_sql_pandas_aggregates_exact=True,
        membership_true_day='training observed off-block date is label-derived; never available as a test input')
    write(OUT/'identity.json',identity)
    print(dict(rows=n,cells=len(cells),portable_bytes=sum((OUT/p).stat().st_size for p in ['cells.parquet','membership.parquet'])),flush=True)


def analyze():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN and identity['source']==art(__file__)
    assert not (OUT/'result.json').exists(),'Preserve completed diagnostic'
    for a in identity['artifacts']:assert sha(ROOT/a['path'])==a['sha256']
    cells=pd.read_parquet(OUT/'cells.parquet');members=pd.read_parquet(OUT/'membership.parquet')
    date=pd.DatetimeIndex(cells.day);results={}
    for fold,allowed in FOLDS.items():
        c=cells[date.strftime('%Y-%m').isin(allowed)&(date.day>1)&(date.day<date.days_in_month)].copy()
        assert set(c.day.str[:7]).issubset(allowed) and fold not in allowed
        X=np.column_stack([np.ones(len(c)),c.taxi_cohort/100.,c.ifr_cohort/100.])
        Y=c[['taxi_n_error','taxi_sum_error','delay_sum_error']].to_numpy(float)
        scale=np.maximum(1.4826*np.median(abs(Y-np.median(Y,axis=0)),axis=0),[1.,60.,60.])
        coefficients=[]
        for k in range(3):
            fit=least_squares(lambda b:(X@b-Y[:,k])/scale[k],np.r_[np.median(Y[:,k]),0.,0.],loss='huber',f_scale=1)
            assert fit.success;coefficients.append(fit.x)
        residual=Y-X@np.array(coefficients).T
        center=np.median(residual,axis=0)
        mad=np.maximum(1.4826*np.median(abs(residual-center),axis=0),[1.,60.,60.])
        use=(abs(residual-center)/mad<=3).all(1);assert use.sum()>=100
        covariance=np.cov(residual[use].T);sd=np.sqrt(np.diag(covariance))
        correlation=covariance/np.outer(sd,sd)
        conditional=covariance[:2,:2]-np.outer(covariance[:2,2],covariance[:2,2])/covariance[2,2]
        assert np.linalg.eigvalsh(covariance).min()>0 and np.linalg.eigvalsh(conditional).min()>0
        results[fold]=dict(allowed=allowed,days=len(c),robust_days=int(use.sum()),
            error_order=['taxi_count','taxi_sum','departure_delay_sum'],coefficients=np.array(coefficients).T.tolist(),
            covariance=covariance.tolist(),correlation=correlation.tolist(),residual_sd=sd.tolist(),
            conditional_taxi_sd_ratio=np.sqrt(np.diag(conditional)/np.diag(covariance)[:2]).tolist(),
            interpretation='Training source-error covariance; conditioning on unknown true error is a mathematical noise diagnostic, not deployable correction or CV.')
    same=members.true_day.eq(members.local_day)
    cross=members[~same]
    complexity={col:dict(quantiles=cells[col].quantile([0,.5,.9,.95,1]).tolist(),days_over12=int(cells[col].gt(12).sum()))
        for col in ['long_flights','mixture_gap_flights']}
    result=dict(identity=art(OUT/'identity.json'),host=socket.gethostname(),folds=results,
        calendars=dict(rows=len(members),different_dates=int((~same).sum()),
            affected_offblock_dates=int(cross.true_day.nunique()),affected_local_dates=int(cross.local_day.nunique()),
            note='UTC off-block and local takeoff groupings; dates cannot be equated in a joint posterior'),
        complexity=complexity,no_prediction_or_candidate=True)
    write(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','analyze']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
