"""Training-only falsifier of same-airport landing-source context transfer.

Twelve dates inside both original outer training folds; no Jan/Jul departure
labels, model matrices, track reads, parameter search or competition model fit.
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

SOURCE=ROOT/'tmp/research_20261012/signal_scout/arrival_landing_clock'
OUT=ROOT/'tmp/research_20261012/signal_scout/arrival_departure_transfer'
MEMBERSHIP=ROOT/'VALIDATION/runs/v3_20261010_callsign_submit/training_ids.parquet'
DATES=[f'2025-{m:02d}-{d:02d}' for m in [3,9,10,11] for d in [5,15,25]]
DESIGN=dict(dates=DATES,allowed_months=['2025-03','2025-09','2025-10','2025-11'],
    population='exact retained original ordinary-training IDs; raw AOBT3/BLOCK present; takeoff after02:00UTC',
    target='departure BLOCK_TIME_UTC_mvt minus genuine original AOBT_3_flt; raw plus fixedclip[-3600,3600]',
    predictor='causal prior2h airport arrival landing-minus-ARVT3 clip[-600,600] mean shrunkK20 to other-date airport/airlineK20; subtract that static mean',
    static_target='airport/airline K20 mean of clipped departure clock error; fitted excluding entire query date',
    conditional_association='demean x and target within airport/airline/month; descriptive only',
    linear_falsifier='one common through-origin OLS coefficient on other-date residuals relative to their fitted static baseline; no held-date targets in any fitted quantity',
    fixed_scopes=['all','eachairport','firstsixdates','lastsixdates'],
    limitations=['training-only diagnostic, not taxi-out CV','does not establish incremental signal beyond existing full model',
        'known-AOBT scope cannot test no-plan departure target transfer'],
    no_held_Jan_Jul_departure_labels=True,no_parameter_or_feature_search=True,no_competition_model_fit=True)


def airline_parser():
    path=ROOT/'VALIDATION/rome_signals.py'
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='airline_code')
    scope={'pd':pd};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['airline_code']


def static_mean(donors,queries,value):
    apt=donors.groupby('airport',observed=True)[value].mean()
    groups=donors.groupby(['airport','airline'],observed=True)[value].agg(['sum','size'])
    hit=groups.reindex(pd.MultiIndex.from_frame(queries[['airport','airline']]))
    return (hit['sum'].fillna(0).to_numpy()+20*queries.airport.map(apt).to_numpy())/(hit['size'].fillna(0).to_numpy()+20)


def summary(f):
    def correlation(a,b):
        return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    centered=f[['x','target']]-f.groupby(['airport','airline','month'],dropna=False,observed=True)[['x','target']].transform('mean')
    result=dict(rows=len(f),context_rows=int(f.support.gt(0).sum()),median_support=float(f.support.median()),
        raw_correlation=correlation(f.x,f.target),within_airport_airline_month_correlation=correlation(centered.x,centered.target),
        x_sd=float(f.x.std()),x_abs_p95=float(f.x.abs().quantile(.95)))
    for y in ['target','raw_target']:
        result[y]={}
        for name in ['baseline','candidate']:
            err=f[name].to_numpy()-f[y].to_numpy()
            result[y][name]=dict(rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.abs(err).mean()))
    return result


def run():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'result.json').exists()
    path=OUT/'design.json'
    if path.exists():assert json.loads(path.read_text())==DESIGN
    else:write_json(path,DESIGN)
    source=json.loads((SOURCE/'result.json').read_text());audit=json.loads((SOURCE/'audit.json').read_text())
    assert audit['status']=='pass' and audit['result_sha256']==sha256(SOURCE/'result.json')
    prepared=json.loads((MEMBERSHIP.parent/'prepared.json').read_text())
    assert prepared['training_ids']['sha256']==sha256(MEMBERSHIP)
    files=[p for p in sorted((ROOT/'data').glob('training_2025-*.parquet'))
        if any(p.name.startswith('training_'+m+'-') for m in DESIGN['allowed_months'])]
    assert len(files)==4
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.register('dates',pd.DataFrame({'d':DATES}))
        dep=con.execute('''SELECT MVT_ID_mvt::BIGINT AS mvt_id,ADEP_mvt airport,FLIGHT_mvt flight,
            strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS date,strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS "month",
            epoch(MVT_TIME_UTC_mvt) AS t,epoch(BLOCK_TIME_UTC_mvt)-epoch(AOBT_3_flt) raw_target
            FROM read_parquet(?) WHERE PHASE_mvt='DEP'
            AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN(SELECT d FROM dates)''',[[str(p) for p in files]]).df()
        arr=con.execute('''SELECT MVT_ID_mvt::BIGINT AS mvt_id,ADES_mvt airport,FLIGHT_mvt flight,
            strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS date,epoch(MVT_TIME_UTC_mvt) AS t,
            epoch(MVT_TIME_UTC_mvt)-epoch(ARVT_3_flt) delta FROM read_parquet(?) WHERE PHASE_mvt='ARR'
            AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN(SELECT d FROM dates)''',[[str(p) for p in files]]).df()
    assert dep.mvt_id.is_unique and arr.mvt_id.is_unique and set(dep.date)==set(DATES)
    dep.to_parquet(OUT/'shared_training_departure_clocks.parquet',index=False)
    arr.to_parquet(OUT/'shared_training_arrival_clocks.parquet',index=False)
    original=len(dep);ids=pd.read_parquet(MEMBERSHIP,columns=['mvt_id']).mvt_id
    ordinary=dep.mvt_id.isin(ids);known=dep.raw_target.notna();complete=dep.t.mod(86400).ge(7200)
    scope=dict(all_twelve_date_departures=original,original_ordinary_ids=int(ordinary.sum()),
        ordinary_known_clock=int((ordinary&known).sum()),eligible=int((ordinary&known&complete).sum()))
    q=dep[ordinary&known&complete].copy().reset_index(drop=True)
    arr=arr[arr.delta.notna()].copy();parse=airline_parser()
    arr['airline']=parse(arr.flight).to_numpy();q['airline']=parse(q.flight).to_numpy()
    arr['z']=arr.delta.clip(-600,600);q['target']=q.raw_target.clip(-3600,3600)
    q['arrival_static']=np.nan;q['support']=0.;q['arrival_sum']=0.
    for date in DATES:
        ix=q.index[q.date.eq(date)];q.loc[ix,'arrival_static']=static_mean(arr[arr.date.ne(date)],q.loc[ix],'z')
    for (date,airport),g in q.groupby(['date','airport'],sort=False):
        donors=arr[arr.date.eq(date)&arr.airport.eq(airport)].sort_values(['t','mvt_id'])
        times=donors.t.to_numpy();a=np.searchsorted(times,g.t.to_numpy()-7200,side='left')
        b=np.searchsorted(times,g.t.to_numpy(),side='left');prefix=np.r_[0.,np.cumsum(donors.z.to_numpy())]
        q.loc[g.index,'support']=b-a;q.loc[g.index,'arrival_sum']=prefix[b]-prefix[a]
    q['x']=(q.arrival_sum+20*q.arrival_static)/(q.support+20)-q.arrival_static
    q['baseline']=np.nan;q['candidate']=np.nan;coefficients=[]
    for date in DATES:
        train=q[q.date.ne(date)];held=q[q.date.eq(date)]
        train_base=static_mean(train,train,'target');held_base=static_mean(train,held,'target')
        x=train.x.to_numpy();residual=train.target.to_numpy()-train_base
        denominator=float(x@x);beta=float(x@residual)/denominator if denominator>0 else 0.
        q.loc[held.index,'baseline']=held_base;q.loc[held.index,'candidate']=held_base+beta*held.x.to_numpy()
        coefficients.append(dict(held_date=date,training_rows=len(train),held_rows=len(held),beta=beta))
    assert np.isfinite(q[['x','baseline','candidate','target']].to_numpy()).all()
    q['half']=np.where(q.date.isin(DATES[:6]),'first_six_dates','last_six_dates')
    q.to_parquet(OUT/'diagnostic_rows.parquet',index=False)
    report=dict(design=DESIGN,scope=scope,all=summary(q),
        by_airport={str(k):summary(g) for k,g in q.groupby('airport')},
        by_half={str(k):summary(g) for k,g in q.groupby('half')},coefficients=coefficients,
        source_sha256=sha256(Path(__file__)),upstream_result_sha256=sha256(SOURCE/'result.json'),
        upstream_audit_sha256=sha256(SOURCE/'audit.json'),membership_sha256=sha256(MEMBERSHIP),
        raw_sources=source['raw_source_inventory'],
        source_scope='only four shared training months/12 dates; original source hashes from preceding audit',
        outputs={p:sha256(OUT/p) for p in ['shared_training_departure_clocks.parquet','shared_training_arrival_clocks.parquet','diagnostic_rows.parquet','design.json']},
        elapsed_seconds=time.monotonic()-started)
    write_json(OUT/'result.json',report)
    print(json.dumps(dict(scope=scope,all=report['all'],by_half=report['by_half'],coefficients=coefficients,
        elapsed_seconds=report['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':run()
