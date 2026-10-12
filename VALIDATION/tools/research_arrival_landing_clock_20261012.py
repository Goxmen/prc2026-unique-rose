"""One-core source-only feasibility of current airport-versus-NM landing clocks.

No departure target, departure block, arrival block or taxi label is read.
The pilot predicts another observable source-clock discrepancy, not taxi-out.
"""
import ast
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
DATES=[f'2025-{m:02d}-{d:02d}' for m in [3,9,10,11] for d in [5,15,25]]
DESIGN=dict(signal='ARR MVT_TIME_UTC_mvt minus original ARVT_3_flt; seconds',
    source_columns=['MVT_ID_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','ARVT_3_flt','FLIGHT_mvt'],
    descriptive='all2025 and old/final2026 airport/month; raw availability,median,IQR,>60s,>600s',
    pilot_dates=DATES,pilot_target='observable arrival landing-clock residual clipped[-600,600]; no value-based row exclusion',
    static='leave-whole-date-out airport mean, then movement-airline mean with K20 airport prior',
    current='strictly earlier sameUTCdate arrivals in prior2h; airport K20 or movement-airline K10 shrinkage to static',
    scope='finite original ARVT3 and landing; queries after02:00UTC ensure full same-day2h support',
    comparison='MAE/RMSE against static on exact same rows; whole,eachairport,firstsix/lastsixdates',
    interpretation='current source-clock predictability is necessary feasibility evidence, not a taxi-out gain',
    no_departure_labels=True,no_movement_blocks=True,no_taxi_labels=True,
    no_threshold_selection=True,no_model_fit=True,no_download=True)


def airline_parser():
    path=ROOT/'VALIDATION/rome_signals.py'
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='airline_code')
    scope={'pd':pd};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['airline_code']


def metrics(frame):
    result=dict(rows=len(frame),airport_context=int(frame.apt_n.gt(0).sum()),airline_context=int(frame.air_n.gt(0).sum()),
        airport_median_support=float(frame.apt_n.median()),airline_median_support=float(frame.air_n.median()))
    for name in ['static','recent_airport','recent_airline']:
        error=frame[name].to_numpy()-frame.z.to_numpy()
        result[name]=dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))))
    return result


def pilot(events):
    e=events.dropna(subset=['delta','t']).copy().sort_values(['date','t','mvt_id']).reset_index(drop=True)
    e['z']=np.clip(e.delta,-600,600);e['airline']=airline_parser()(e.flight).to_numpy()
    e['static']=np.nan
    for date in DATES:
        donors=e[e.date.ne(date)];query=e[e.date.eq(date)]
        airport=donors.groupby('airport',observed=True).z.mean()
        by=donors.groupby(['airport','airline'],observed=True).z.agg(['sum','size'])
        keys=pd.MultiIndex.from_frame(query[['airport','airline']]);hits=by.reindex(keys)
        prior=query.airport.map(airport).to_numpy(float)
        e.loc[query.index,'static']=(hits['sum'].fillna(0).to_numpy()+20*prior)/(hits['size'].fillna(0).to_numpy()+20)
    for keys,name,k in [(['date','airport'],'apt',20),(['date','airport','airline'],'air',10)]:
        e[name+'_n']=0.;e[name+'_sum']=0.
        for _,g in e.groupby(keys,dropna=False,observed=True,sort=False):
            times=g.t.to_numpy();a=np.searchsorted(times,times-7200,side='left')
            b=np.searchsorted(times,times,side='left')
            total=np.r_[0.,np.cumsum(g.z.to_numpy())]
            e.loc[g.index,name+'_n']=b-a;e.loc[g.index,name+'_sum']=total[b]-total[a]
    e['recent_airport']=(e.apt_sum+20*e.static)/(e.apt_n+20)
    e['recent_airline']=(e.air_sum+10*e.static)/(e.air_n+10)
    # UTC epoch arithmetic, independent of machine-local timezone.
    complete=np.mod(e.t,86400)>=7200
    q=e[complete&e.static.notna()].copy()
    q['half']=np.where(q.date.isin(DATES[:6]),'first_six_dates','last_six_dates')
    return q,dict(all=metrics(q),by_airport={str(k):metrics(g) for k,g in q.groupby('airport')},
        by_half={str(k):metrics(g) for k,g in q.groupby('half')},
        raw_rows=len(events),known_clock_rows=len(e),complete_queries=len(q),
        no_comparison_prior_rows=int(e.static.isna().sum()),month_start_source_boundary='same-date complete2h only')


def run():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'result.json').exists()
    path=OUT/'design.json'
    if path.exists():assert json.loads(path.read_text())==DESIGN
    else:write_json(path,DESIGN)
    started=time.monotonic();training=sorted((ROOT/'data').glob('training_2025-*.parquet'))
    assert len(training)==12
    sources=dict(training=training,old=[ROOT/'data/ranking.parquet'],final=[ROOT/'data/final_ranking.parquet'])
    summaries=[]
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'");con.execute("SET memory_limit='1GB'")
        for view,files in sources.items():
            result=con.execute('''WITH e AS (SELECT ADES_mvt airport,
                strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS "month",
                epoch(MVT_TIME_UTC_mvt)-epoch(ARVT_3_flt) delta
                FROM read_parquet(?) WHERE PHASE_mvt='ARR')
                SELECT airport,"month",count(*) AS n,count(delta) AS known,
                median(delta) AS median,quantile_cont(delta,.25) AS q25,quantile_cont(delta,.75) AS q75,
                count(*) FILTER(WHERE abs(delta)>60) AS over60,
                count(*) FILTER(WHERE abs(delta)>600) AS over600
                FROM e GROUP BY airport,"month" ORDER BY airport,"month"''',[[str(p) for p in files]]).df()
            result['view']=view;summaries.append(result)
        pilot_files=[p for p in training if any(p.name.startswith(f'training_2025-{m:02d}-') for m in [3,9,10,11])]
        assert len(pilot_files)==4
        con.register('pilot_dates',pd.DataFrame({'date':DATES}))
        events=con.execute('''SELECT MVT_ID_mvt::BIGINT AS mvt_id,ADES_mvt airport,
            FLIGHT_mvt flight,strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS date,
            epoch(MVT_TIME_UTC_mvt) AS t,epoch(MVT_TIME_UTC_mvt)-epoch(ARVT_3_flt) AS delta
            FROM read_parquet(?) WHERE PHASE_mvt='ARR'
              AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN(SELECT date FROM pilot_dates)
            ORDER BY MVT_ID_mvt''',[[str(p) for p in pilot_files]]).df()
    assert events.mvt_id.is_unique
    rows,report=pilot(events)
    pd.concat(summaries,ignore_index=True).to_parquet(OUT/'source_summary.parquet',index=False)
    rows.to_parquet(OUT/'pilot_rows.parquet',index=False)
    write_json(OUT/'result.json',dict(design=DESIGN,pilot=report,elapsed_seconds=time.monotonic()-started,
        source_sha256=sha256(Path(__file__)),airline_parser_source_sha256=sha256(ROOT/'VALIDATION/rome_signals.py'),
        raw_source_inventory=json.loads((ROOT/'data/derived/source_clock_phase_20261012/manifest.json').read_text())['raw_sources'],
        source_binding='same retained raw inputs as completed all-row phase audit; no new source files',
        outputs={p:sha256(OUT/p) for p in ['source_summary.parquet','pilot_rows.parquet','design.json']}))
    print(json.dumps(dict(pilot=report,elapsed_seconds=time.monotonic()-started),indent=2),flush=True)


if __name__=='__main__':run()
