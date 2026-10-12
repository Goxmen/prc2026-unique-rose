"""Independent bounded raw/causal arithmetic audit; no taxi labels or blocks."""
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
from records import sha256,write_json

OUT=ROOT/'tmp/research_20261012/signal_scout/arrival_landing_clock'


def run():
    started=time.monotonic();report=json.loads((OUT/'result.json').read_text());design=report['design']
    assert sha256(Path(__file__).with_name('research_arrival_landing_clock_20261012.py'))==report['source_sha256']
    for name,digest in report['outputs'].items():assert sha256(OUT/name)==digest
    for records in report['raw_source_inventory'].values():
        for record in records:assert sha256(ROOT/record['path'])==record['sha256']
    paths=[p for p in sorted((ROOT/'data').glob('training_2025-*.parquet'))
           if any(p.name.startswith(f'training_2025-{month:02d}-') for month in [3,9,10,11])]
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        con.register('dates',pd.DataFrame({'d':design['pilot_dates']}))
        raw=con.execute('''SELECT MVT_ID_mvt AS mvt_id,ADES_mvt airport,FLIGHT_mvt flight,
            MVT_TIME_UTC_mvt landing,ARVT_3_flt arrival_plan FROM read_parquet(?)
            WHERE PHASE_mvt='ARR' AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN(SELECT d FROM dates)''',
            [[str(p) for p in paths]]).df()
    raw['t']=pd.to_datetime(raw.landing,utc=True).map(lambda value:value.timestamp())
    raw['delta']=(pd.to_datetime(raw.landing,utc=True)-pd.to_datetime(raw.arrival_plan,utc=True)).dt.total_seconds()
    raw['z']=np.clip(raw.delta,-600,600)
    raw['date']=pd.to_datetime(raw.landing,utc=True).dt.strftime('%Y-%m-%d')
    def airline(value):
        if not isinstance(value,str) or len(value)<3:return None
        return value[:3] if value[:3].isalpha() else value[:2]
    raw['airline']=raw.flight.map(airline)
    raw=raw[raw.arrival_plan.notna()].set_index('mvt_id')
    q=pd.read_parquet(OUT/'pilot_rows.parquet').set_index('mvt_id');r=raw.loc[q.index]
    assert q.index.is_unique and len(q)==report['pilot']['complete_queries']
    for col in ['t','delta','z']:np.testing.assert_array_equal(q[col],r[col])
    assert q.airport.equals(r.airport) and q.date.equals(r.date)
    assert (pd.to_datetime(r.landing,utc=True).dt.hour>=2).all()
    picked=q.groupby(['airport','date'],sort=True).head(1).index.union(q[q.air_n.eq(0)].head(20).index)
    maximum=0.
    for row in q.loc[picked].itertuples():
        other=raw[raw.airport.eq(row.airport)&raw.date.ne(row.date)]
        prior=other.z.mean();same=other[other.airline.eq(row.airline)]
        expected=(same.z.sum()+20*prior)/(len(same)+20)
        within=raw[raw.airport.eq(row.airport)&raw.date.eq(row.date)&raw.t.ge(row.t-7200)&raw.t.lt(row.t)]
        air=within[within.airline.eq(row.airline)]
        values={'static':expected,'apt_n':len(within),'apt_sum':within.z.sum(),'air_n':len(air),'air_sum':air.z.sum(),
            'recent_airport':(within.z.sum()+20*expected)/(len(within)+20),
            'recent_airline':(air.z.sum()+10*expected)/(len(air)+10)}
        for name,value in values.items():
            err=abs(float(value)-float(getattr(row,name)));maximum=max(maximum,err)
            assert err<1e-8,(row.Index,name,value,getattr(row,name))
    for key,part in [('all',q),*[('airport:'+str(k),g) for k,g in q.groupby('airport')],
                     *[('half:'+str(k),g) for k,g in q.groupby('half')]]:
        metrics=report['pilot']['all'] if key=='all' else report['pilot']['by_airport' if key.startswith('airport:') else 'by_half'][key.split(':',1)[1]]
        for col in ['static','recent_airport','recent_airline']:
            error=part[col].to_numpy()-part.z.to_numpy()
            np.testing.assert_allclose(np.abs(error).mean(),metrics[col]['mae'],rtol=0,atol=1e-9)
            np.testing.assert_allclose(np.sqrt(np.mean(error**2)),metrics[col]['rmse'],rtol=0,atol=1e-9)
    write_json(OUT/'audit.json',dict(status='pass',all_query_rows=len(q),scalar_causal_queries=len(picked),
        scalar_values=len(picked)*7,max_scalar_error=maximum,all_reported_metrics_recomputed=True,
        all_raw_sources_and_output_hashes_verified=True,no_departure_labels_blocks_or_taxi_labels=True,
        seconds=time.monotonic()-started,source_sha256=sha256(Path(__file__)),result_sha256=sha256(OUT/'result.json')))
    print(json.dumps(json.loads((OUT/'audit.json').read_text()),indent=2),flush=True)


if __name__=='__main__':run()
