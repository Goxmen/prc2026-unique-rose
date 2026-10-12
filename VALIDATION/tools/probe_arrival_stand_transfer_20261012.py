"""Training-only falsification of arrival-supervised stand adaptation.

Unlike prior landing-clock transfer, this uses supplied taxi-in durations at
the same stand. Only March/September/October/November departure labels are read.
No Jan/Jul departure labels, model errors or competition predictions are read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_arrival_stand_transfer'
MONTHS=['2025-03','2025-09','2025-10','2025-11']
DESIGN=dict(months=MONTHS,group=['airport','stand'],minimum_arrival_and_departure_rows=20,
    arrivals='provided taxi-in1..7200 seconds; stand and actual UTC landing month',
    departures='training-only taxi-out1..10800 seconds; stand and actual UTC takeoff month',
    centered='stand median minus same-airport same-month median, separately per phase',
    transfer='held-month centered stand median minus mean of the other3 centered stand medians',
    calibration_contrasts='each training month minus the other2 training months; held departure month never enters training references',
    pilot_model='one slope through origin; weighted least squares fit on the other3 months',
    weights='minimum(departure_cell_count,100)',
    gate='positive slope and lower weighted squared error than zero change in every held training month',
    novelty='arrival-duration supervision for stand drift, not the failed landing-clock discrepancy transfer',
    availability='arrival targets are supplied in ranking; future-arrival context is a batch-competition input',
    limitation='feasibility only; grouped medians are not actual model residuals or evidence of CV gain',
    no_jan_jul_departure_labels=True,no_taxi_model_fit=True,no_cv=True,no_upload=True,no_search=True)


def art(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze():
    assert not (OUT/'identity.json').exists()
    sources=[Path(__file__),*sorted(ROOT.glob('data/training_2025-*.parquet'))]
    write(OUT/'identity.json',dict(design=DESIGN,sources=[art(p) for p in sources]))


def pilot():
    identity=json.loads((OUT/'identity.json').read_text());assert identity['design']==DESIGN
    assert not (OUT/'result.json').exists()
    for record in identity['sources']:assert art(ROOT/record['path'])==record
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        # SQL selects the allowed calendar before exposing any departure target.
        con.execute('''CREATE TEMP TABLE r AS SELECT MVT_ID_mvt::BIGINT AS mvt_id,
            PHASE_mvt AS phase, CASE WHEN PHASE_mvt='DEP' THEN ADEP_mvt ELSE ADES_mvt END AS airport,
            STAND_mvt AS stand, strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,
            TAXITIME_SEC_mvt::DOUBLE AS y FROM read_parquet(?)
            WHERE strftime(MVT_TIME_UTC_mvt,'%Y-%m') IN (SELECT unnest(?))
              AND PHASE_mvt IN ('ARR','DEP') AND STAND_mvt IS NOT NULL
              AND TAXITIME_SEC_mvt BETWEEN1 AND CASE WHEN PHASE_mvt='DEP' THEN10800 ELSE7200 END'''
            .replace('BETWEEN1','BETWEEN 1').replace('THEN10800','THEN 10800').replace('ELSE7200','ELSE 7200'),
            [str(ROOT/'data/training_2025-*.parquet'),MONTHS])
        assert {v[0] for v in con.execute('SELECT DISTINCT month FROM r').fetchall()}==set(MONTHS)
        n,u=con.execute('SELECT count(*),count(distinct mvt_id) FROM r').fetchone();assert n==u
        f=con.execute('''WITH a AS (SELECT phase,airport,month,count(*) AS airport_n,
                median(y) AS airport_median FROM r GROUP BY ALL),
            s AS (SELECT phase,airport,stand,month,count(*) AS n,median(y) AS stand_median
                FROM r GROUP BY ALL HAVING count(*)>=20)
            SELECT s.*,a.airport_n,a.airport_median,s.stand_median-a.airport_median AS centered
            FROM s JOIN a USING(phase,airport,month) ORDER BY phase,airport,stand,month''').df()
    f.to_parquet(OUT/'cells.parquet',index=False)
    wide=f.pivot(index=['airport','stand','month'],columns='phase',values=['centered','n']).dropna()
    wide.columns=['_'.join(v) for v in wide.columns];wide=wide.reset_index()
    keys=wide.groupby(['airport','stand']).month.nunique()
    keys=keys[keys==4].reset_index()[['airport','stand']]
    wide=wide.merge(keys,on=['airport','stand'],validate='many_to_one')
    wide['weight']=np.minimum(wide.n_DEP,100)
    records=[];predictions=[]
    for month in MONTHS:
        train=wide.loc[wide.month.ne(month)].copy();test=wide.loc[wide.month.eq(month)].copy()
        assert len(test)>0
        for phase in ['ARR','DEP']:
            column='centered_'+phase
            totals=train.groupby(['airport','stand'])[column].transform('sum')
            train['delta_'+phase]=train[column]-(totals-train[column])/2
            reference=train.groupby(['airport','stand'])[column].mean()
            reference=reference.loc[pd.MultiIndex.from_frame(test[['airport','stand']])].to_numpy()
            test['delta_'+phase]=test[column]-reference
        denominator=float(np.sum(train.weight*train.delta_ARR**2))
        slope=float(np.sum(train.weight*train.delta_ARR*train.delta_DEP)/denominator) if denominator else 0.
        baseline=float(np.average(test.delta_DEP**2,weights=test.weight))
        error=float(np.average((test.delta_DEP-slope*test.delta_ARR)**2,weights=test.weight))
        records.append(dict(month=month,cells=len(test),slope=slope,baseline_mse=baseline,candidate_mse=error,
            mse_delta=error-baseline,baseline_rmse=baseline**.5,candidate_rmse=error**.5))
        test['prediction']=slope*test.delta_ARR
        predictions.append(test)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'transfer.parquet',index=False)
    passed=all(r['slope']>0 and r['mse_delta']<0 for r in records)
    result=dict(identity=art(OUT/'identity.json'),sources_rows=n,eligible_phase_cells=len(f),common_stands=len(keys),
        records=records,passes_fixed_transfer_gate=passed,
        artifacts=[art(OUT/'cells.parquet'),art(OUT/'transfer.parquet')],
        interpretation='Training-month stand-drift feasibility, not comparison with the selected model or a CV result.')
    write(OUT/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','pilot'])
    globals()[p.parse_args().stage]()
