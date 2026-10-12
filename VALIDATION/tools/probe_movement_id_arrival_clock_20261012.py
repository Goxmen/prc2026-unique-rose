"""Frozen coarse ID-order falsifier using supplied arrival clocks as anchors.

This differs from inferring exact timestamps from adjacent-ID monotonicity.
It asks whether coarse cross-phase ordering separates Rome's two recorded
off-block hypotheses. No January/July departure labels or model fit are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_movement_id_arrival_clock'
CACHE=ROOT/'data/derived/aircraft_rotation_20261010/identity.parquet'
LABELS=ROOT/'VALIDATION/runs/v3_20261011_daily_offblock_alignment/rows2025.parquet'
MONTHS=['2025-03','2025-09','2025-10','2025-11']
DESIGN=dict(airport='LIRF',months=MONTHS,identifier='MVT_ID_mvt, not NM FLIGHT_ID_mvt',
    source='retained supplied-arrival cache; no aircraft identity gate or external fetch',
    reference='same airport and UTC movement month; arrivals with known landing and in-block',
    peer_count='nearest5 strictly smaller and nearest5 strictly larger arrival movement IDs; require all10; no extrapolation',
    clock='linear interpolation between median ID and median in-block of each5-anchor side',
    secondary='identical interpolation using landing instead of in-block, descriptive only',
    query='all permitted-month Rome departures; numeric IDs never become a direct predictor',
    labels='only fixed four shared-training months, joined after input clocks exist',
    primary_population='no flight ID and original takeoff-minus-schedule gap strictly>3600 and<=14400seconds',
    copy_label='abs(original gap minus recorded taxi)<=60seconds, unchanged historical classifier target',
    copy_score='(takeoff-minus-estimated-ID-clock minus1050)/(gap-minus1050); larger means nearer the schedule-copy hypothesis',
    diagnostics='coverage, source spans, recorded off-block absolute error, AUC by all four months and pooled; no learned calibration',
    retain_if=dict(primary_coverage_min=.8,pooled_auc_min=.65,each_month_auc_min=.55,each_month_each_class_min=10),
    interpretation='Engineering falsifier only, not a necessary mathematical condition or demonstrated incremental prediction gain',
    no_held_departure_labels=True,no_fit=True,no_cv=True,no_parameter_or_calendar_search=True,
    resources=dict(threads=1,duckdb_memory='256MB',external_timeout_seconds=60))


def art(path):
    path=Path(path);return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):return json.loads(Path(path).read_text())


def freeze():
    assert not (OUT/'identity.json').exists()
    arrival=read(CACHE.with_suffix('.json'))
    assert arrival['identity_sha256']==art(CACHE)['sha256']
    assert arrival['source_sha256']==art(ROOT/'VALIDATION/aircraft_rotation.py')['sha256']
    audit=ROOT/'VALIDATION/runs/v3_20261011_daily_offblock_alignment/source_audit.json'
    assert read(audit)['status']=='pass' and read(audit)['prepared_rows']==art(LABELS)
    paths=[Path(__file__),CACHE,CACHE.with_suffix('.json'),LABELS,audit,
        ROOT/'VALIDATION/reports/research_network_20261010/id_clock_order.json',
        ROOT/'VALIDATION/tests/test_movement_id_arrival_clock.py']
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'identity.json').write_text(json.dumps(dict(design=DESIGN,sources=[art(p) for p in paths]),indent=2)+'\n')


def clocks(query_ids,arrival_ids,landing,inblock):
    query_ids,arrival_ids=np.asarray(query_ids),np.asarray(arrival_ids)
    landing,inblock=np.asarray(landing,float),np.asarray(inblock,float)
    assert np.all(np.diff(arrival_ids)>0) and not np.intersect1d(query_ids,arrival_ids).size
    assert np.isfinite(landing).all() and np.isfinite(inblock).all()
    at=np.searchsorted(arrival_ids,query_ids)
    valid=(at>=5)&(at+5<=len(arrival_ids));q=np.flatnonzero(valid)
    result=dict(available=valid,landing=np.full(len(query_ids),np.nan),inblock=np.full(len(query_ids),np.nan),
                anchor_time_spread=np.full(len(query_ids),np.nan),id_span=np.full(len(query_ids),np.nan))
    if not len(q):return result
    lo=at[q,None]+np.arange(-5,0);hi=at[q,None]+np.arange(5)
    assert np.all(arrival_ids[lo]<query_ids[q,None]) and np.all(arrival_ids[hi]>query_ids[q,None])
    left_id=np.median(arrival_ids[lo],axis=1);right_id=np.median(arrival_ids[hi],axis=1)
    weight=(query_ids[q]-left_id)/(right_id-left_id)
    assert np.all((weight>0)&(weight<1))
    for key,source in [('landing',landing),('inblock',inblock)]:
        left=np.median(source[lo],axis=1);right=np.median(source[hi],axis=1)
        result[key][q]=left+weight*(right-left)
    joined=np.concatenate([inblock[lo],inblock[hi]],axis=1)
    result['anchor_time_spread'][q]=np.ptp(joined,axis=1)
    result['id_span'][q]=right_id-left_id
    return result


def auc(label,score):
    label=np.asarray(label,bool);score=np.asarray(score,float)
    assert np.isfinite(score).all()
    n1=int(label.sum());n0=int((~label).sum())
    return None if not n1 or not n0 else float((rankdata(score)[label].sum()-n1*(n1+1)/2)/(n0*n1))


def metrics(frame):
    frame=frame.copy();frame['copy']=(frame.gap-frame.y).abs().le(60)
    frame['primary']=~frame.has_id&frame.gap.gt(3600)&frame.gap.le(14400)
    groups=[('ALL',frame)]+list(frame.groupby('month'));records=[]
    for month,group in groups:
        for scope,part in [('all',group),('primary',group.loc[group.primary])]:
            q=part.loc[part.available];n=len(part)
            rec=dict(month=month,scope=scope,rows=n,covered=len(q),coverage=len(q)/max(n,1),
                positives=int(q['copy'].sum()),negatives=int((~q['copy']).sum()),clocks={})
            for key in ['inblock','landing']:
                error=q[key]-(q.takeoff_epoch-q.y)
                with np.errstate(divide='ignore',invalid='ignore'):
                    score=(q.takeoff_epoch-q[key]-1050)/(q.gap-1050)
                use=np.isfinite(score)
                rec['clocks'][key]=dict(absolute_error_quantiles=error.abs().quantile([.5,.9,.99]).tolist() if len(q) else [None]*3,
                    copy_auc=auc(q.loc[use,'copy'],score[use]) if len(q) else None)
            rec['anchor_time_spread_quantiles']=q.anchor_time_spread.quantile([.5,.9,.99]).tolist() if len(q) else [None]*3
            records.append(rec)
    primary=[r for r in records if r['scope']=='primary'];combined=next(r for r in primary if r['month']=='ALL')
    individual=[r for r in primary if r['month']!='ALL'];gate=DESIGN['retain_if']
    enough=len(individual)==4 and all(min(r['positives'],r['negatives'])>=gate['each_month_each_class_min'] for r in individual)
    keep=enough and combined['coverage']>=gate['primary_coverage_min'] and combined['clocks']['inblock']['copy_auc']>=gate['pooled_auc_min']
    keep=keep and all(r['clocks']['inblock']['copy_auc']>=gate['each_month_auc_min'] for r in individual)
    return records,dict(retain_for_future_model_design=bool(keep),enough_class_support=enough,
        interpretation=DESIGN['interpretation'],no_secondary_clock_selection=True)


def pilot():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for rec in identity['sources']:assert art(ROOT/rec['path'])==rec
    assert not (OUT/'result.json').exists();start=time.monotonic()
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        # Input-only query: no departure block/taxi label is present in this projection.
        rows=con.execute('''SELECT mvt_id,phase,strftime(t,'%Y-%m') AS "month",
            epoch(t) landing,epoch(inblock) inblock FROM read_parquet(?)
            WHERE airport='LIRF' AND strftime(t,'%Y-%m') IN (SELECT unnest(?))''',[str(CACHE),MONTHS]).df()
        assert rows.mvt_id.is_unique;frames=[]
        for month,group in rows.groupby('month'):
            arrivals=group.loc[group.phase.eq('ARR')].dropna(subset=['landing','inblock']).sort_values('mvt_id')
            query=group.loc[group.phase.eq('DEP')].copy()
            values=clocks(query.mvt_id,arrivals.mvt_id,arrivals.landing,arrivals.inblock)
            frame=query[['mvt_id','month']].copy()
            for key,value in values.items():frame[key]=value
            frames.append(frame)
        inputs=pd.concat(frames,ignore_index=True)
        inputs.to_parquet(OUT/'input_clocks.parquet',index=False)
        truth=con.execute('''SELECT mvt_id,month,gap,has_id,y,takeoff_epoch FROM read_parquet(?)
            WHERE airport='LIRF' AND month IN (SELECT unnest(?))''',[str(LABELS),MONTHS]).df()
    assert truth.mvt_id.is_unique and set(truth.month)==set(MONTHS)
    frame=truth.merge(inputs,on=['mvt_id','month'],validate='one_to_one',how='left')
    assert len(frame)==len(inputs)==len(truth) and frame.available.notna().all()
    records,decision=metrics(frame)
    frame.to_parquet(OUT/'diagnostic_rows.parquet',index=False)
    result=dict(identity=art(OUT/'identity.json'),records=records,decision=decision,seconds=time.monotonic()-start,
        rows=len(frame),sources=[art(OUT/'input_clocks.parquet'),art(OUT/'diagnostic_rows.parquet')],
        no_held_labels=True,no_fit=True,no_cv=True)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(rows=len(frame),seconds=result['seconds'],decision=decision))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','pilot'])
    globals()[parser.parse_args().stage]()
