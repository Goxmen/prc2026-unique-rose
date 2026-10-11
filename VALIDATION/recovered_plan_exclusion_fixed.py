"""Audit/test the failed-NM-join interval using uniquely matched arrival plans.

Lead: gracious-tornado's recovered-orphan rule. Original implementation with
stricter ambiguity, aircraft, plan-route and fold-context checks. Airport gate
LIRF/LTFM is fixed from that public lead; this is reused development evidence.
No departure label or block timestamp enters the recovery query or application.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import daily_delay_reconcile as base
from records import write_json

OUT=base.ROOT/'VALIDATION/runs/v3_20261011_recovered_exclusion_fixed'
CACHE=base.ROOT/'VALIDATION/runs/v3_20261011_recovered_exclusion'
PARENT=base.ROOT/'VALIDATION/runs/v3_20261011_expanded_affine/submission_v73'
DESIGN=dict(parent='v73',airports=['LIRF','LTFM'],window_sec=3606,landing_gap_sec=[600,72000],
    matching='exact flight and movement route; compatible aircraft; consistent plan route; exactly one candidate',
    rule='if prediction inside excluded interval, project to nearest nonnegative edge',
    target_access='scoring and diagnostic only; recovery input allowlist',
    caveat='Nonconvex support does not itself guarantee squared-error improvement; published gate saw reused2025 data')

def recover(paths):
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("set TimeZone='UTC'")
        con.execute('''create temp table public_movements as select MVT_ID_mvt id,FLIGHT_ID_mvt plan_id,
          FLIGHT_mvt flight,ADEP_mvt adep,ADES_mvt ades,PHASE_mvt phase,AIRCRAFT_TYPE_mvt aircraft,
          MVT_TIME_UTC_mvt movement,LOBT_flt lobt,ADEP_flt plan_adep,ADES_flt plan_ades
          from read_parquet(?) where ADEP_mvt in ('LIRF','LTFM')''',[list(map(str,paths))])
        rows=con.execute('''with pairs as (
          select d.id mvt_id,d.adep airport,d.movement takeoff,a.id donor_mvt_id,a.plan_id donor_plan_id,
            a.movement landing,a.lobt recovered_lobt,
            count(*) over(partition by d.id) candidates,
            epoch(d.movement-a.lobt) recovered_gap,
            strftime(a.movement,'%Y-%m') donor_month,
            (d.aircraft is null or a.aircraft is null or d.aircraft=a.aircraft) aircraft_ok,
            (a.plan_adep=d.adep and a.plan_ades=d.ades) route_ok
          from public_movements d join public_movements a
            on d.flight=a.flight and d.adep=a.adep and d.ades=a.ades
            and a.movement>d.movement+interval 10 minute
            and a.movement<d.movement+interval 20 hour
          where d.phase='DEP' and d.plan_id is null and d.flight is not null and d.flight<>''
            and a.phase='ARR' and a.plan_id is not null and a.lobt is not null)
          select * from pairs order by mvt_id,donor_mvt_id''').fetchdf()
    rows['accepted']=(rows.candidates==1)&rows.aircraft_ok.fillna(False)&rows.route_ok.fillna(False)
    assert rows.loc[rows.accepted,'mvt_id'].is_unique
    return rows

def apply(ids,pred,matched):
    q=np.asarray(pred,float).copy();m=matched.loc[matched.accepted].set_index('mvt_id').reindex(ids)
    gap=m.recovered_gap.to_numpy(float);lo=gap-3606;hi=gap+3606
    inside=np.isfinite(gap)&(q>lo)&(q<hi)
    use_lower=(lo[inside]>=0)&(q[inside]-lo[inside]<=hi[inside]-q[inside])
    q[inside]=np.where(use_lower,lo[inside],hi[inside])
    assert ((q[inside]<=lo[inside])|(q[inside]>=hi[inside])).all()
    assert np.isfinite(q).all() and (q>=0).all()
    return q,pd.DataFrame(dict(mvt_id=ids,accepted=np.isfinite(gap),inside=inside,low=lo,high=hi,
        donor_mvt_id=m.donor_mvt_id.to_numpy(),pred=np.asarray(pred),candidate=q))

def screen():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'result.json').exists()
    paths=sorted((base.ROOT/'data').glob('training_2025-*.parquet'))
    write_json(OUT/'design.json',dict(design=DESIGN,source=base.artifact(__file__),inputs=[base.artifact(p) for p in paths]))
    matches=pd.read_parquet(CACHE/'matches2025.parquet');matches.to_parquet(OUT/'matches2025.parquet',index=False)
    write_json(OUT/'cache_identity.json',dict(training=base.artifact(CACHE/'matches2025.parquet'),ranking=base.artifact(CACHE/'ranking_matches.parquet'),original_source=base.artifact(base.ROOT/'VALIDATION/recovered_plan_exclusion.py'),correction='A negative lower boundary is not feasible; move to upper boundary instead of zero'))
    rows=pd.read_parquet(base.OUT/'rows2025.parquet').set_index('mvt_id')
    accepted=matches.loc[matches.accepted].copy();accepted['y']=rows.loc[accepted.mvt_id,'y'].to_numpy()
    accepted['truth_outside']=(accepted.y<=accepted.recovered_gap-3606)|(accepted.y>=accepted.recovered_gap+3606)
    accepted['month']=rows.loc[accepted.mvt_id,'month'].to_numpy()
    diagnostics={a:dict(rows=len(q),truth_outside=int(q.truth_outside.sum())) for a,q in accepted.groupby('airport')}
    f=pd.read_parquet(PARENT/'cv.parquet');q=f.pred.to_numpy(float).copy();logs=[];training=[]
    for month,allowed in base.FOLDS.items():
        context=allowed+[month];ix=np.flatnonzero(f.month.eq(month))
        permitted=matches.loc[matches.donor_month.isin(context)]
        q[ix],audit=apply(f.mvt_id.to_numpy()[ix],q[ix],permitted);logs.append(audit.assign(month=month))
        support=accepted.loc[accepted.month.isin(allowed)&accepted.donor_month.isin(context)]
        training.append(dict(fold=month,rows=len(support),outside=int(support.truth_outside.sum())))
    frame=f[['mvt_id','month','day','airport','y']].copy();frame['parent']=f.pred;frame['pred']=q
    frame.to_parquet(OUT/'predictions.parquet',index=False);pd.concat(logs).to_parquet(OUT/'audit.parquet',index=False)
    import tuning_seasonal as ts
    scores={m:dict(parent=float(np.sqrt(np.mean((g.parent-g.y)**2))),candidate=float(np.sqrt(np.mean((g.pred-g.y)**2))),
        changed=int((g.pred!=g.parent).sum())) for m,g in frame.groupby('month')}
    paired=ts.paired_bootstrap(frame.day,frame.month,(frame.parent-frame.y)**2,(frame.pred-frame.y)**2)
    test=pd.read_parquet(CACHE/'ranking_matches.parquet');test.to_parquet(OUT/'ranking_matches.parquet',index=False)
    p=pd.read_parquet(base.ROOT/'submissions/unique-rose_v73.parquet')
    test_pred,test_audit=apply(p.MVT_ID_mvt.to_numpy(),p.TAXITIME_SEC_mvt.to_numpy(float),test)
    test_audit.to_parquet(OUT/'ranking_audit.parquet',index=False)
    result=dict(design=DESIGN,months=scores,paired=paired,diagnostics=diagnostics,training=training,
        candidate_pairs=len(matches),accepted=int(matches.accepted.sum()),ranking_pairs=len(test),ranking_accepted=int(test.accepted.sum()),
        ranking_changed=int(test_audit.inside.sum()),cv=float(np.mean([s['candidate'] for s in scores.values()])),source=base.artifact(__file__))
    write_json(OUT/'result.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen']);a=p.parse_args();globals()[a.stage]()
