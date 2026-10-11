"""All-airport recovered-plan screen, gates learned only in allowed training months.

A high-precision support gate requires at least50 unique matches and no interval
violations in training. Reuse strict route/aircraft/ambiguity checks from v74.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import duckdb,numpy as np,pandas as pd
import daily_delay_reconcile as base
import recovered_plan_exclusion_fixed as parent
from records import write_json
OUT=base.ROOT/'VALIDATION/runs/v3_20261011_recovered_all_airports'
PARENT=base.ROOT/'VALIDATION/runs/v3_20261011_public_constraints/submission_v74'
DESIGN=dict(parent='v74',gate='at least50 accepted training matches, every label outside recovered window',
    matching=parent.DESIGN['matching'],landing_gap_sec=[600,72000],window_sec=3606,
    airports='all airports; gate fitted on allowed training months',held_labels='final development scoring only')

def recover(paths):
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("set TimeZone='UTC'")
        con.execute('''create temp table public_movements as select MVT_ID_mvt id,FLIGHT_ID_mvt plan_id,
          FLIGHT_mvt flight,ADEP_mvt adep,ADES_mvt ades,PHASE_mvt phase,AIRCRAFT_TYPE_mvt aircraft,
          MVT_TIME_UTC_mvt movement,LOBT_flt lobt,ADEP_flt plan_adep,ADES_flt plan_ades
          from read_parquet(?)''',[list(map(str,paths))])
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

def support(matches,rows,allowed,context):
    p=matches.loc[matches.accepted & matches.donor_month.isin(context)].copy()
    p['month']=rows.loc[p.mvt_id,'month'].to_numpy()
    p=p.loc[p.month.isin(allowed)]
    y=rows.loc[p.mvt_id,'y'].to_numpy(float)
    p['outside']=(y<=p.recovered_gap-3606)|(y>=p.recovered_gap+3606)
    table=p.groupby('airport').outside.agg(['size','sum']).reset_index()
    table['allowed']=(table['size']>=50)&(table['size']==table['sum'])
    return table

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'result.json').exists()
    paths=sorted((base.ROOT/'data').glob('training_2025-*.parquet'))
    write_json(OUT/'design.json',dict(design=DESIGN,source=base.artifact(__file__),
        helpers=[base.artifact(parent.__file__)],inputs=[base.artifact(p) for p in paths],
        parent=base.artifact(PARENT/'manifest.json')))
    matches=recover(paths);matches.to_parquet(OUT/'matches2025.parquet',index=False)
    rows=pd.read_parquet(base.OUT/'rows2025.parquet').set_index('mvt_id')
    allmonths=sorted(rows.month.unique());tables={};logs=[]
    f=pd.read_parquet(PARENT/'cv.parquet');q=f.pred.to_numpy(float).copy()
    for month,allowed in base.FOLDS.items():
        context=allowed+[month];table=support(matches,rows,allowed,context);tables[month]=table.to_dict('records')
        keep=table.loc[table.allowed,'airport'];permitted=matches.loc[matches.airport.isin(keep)&matches.donor_month.isin(context)]
        ix=np.flatnonzero(f.month.eq(month));q[ix],audit=parent.apply(f.mvt_id.to_numpy()[ix],q[ix],permitted)
        logs.append(audit.assign(month=month));print(month,table.to_string(index=False),flush=True)
    frame=f[['mvt_id','month','day','airport','y']].copy();frame['parent']=f.pred;frame['pred']=q
    frame.to_parquet(OUT/'predictions.parquet',index=False);pd.concat(logs).to_parquet(OUT/'audit.parquet',index=False)
    import tuning_seasonal as ts
    scores={m:dict(parent=float(np.sqrt(np.mean((g.parent-g.y)**2))),candidate=float(np.sqrt(np.mean((g.pred-g.y)**2))),changed=int((g.pred!=g.parent).sum())) for m,g in frame.groupby('month')}
    paired=ts.paired_bootstrap(frame.day,frame.month,(frame.parent-frame.y)**2,(frame.pred-frame.y)**2)
    table=support(matches,rows,allmonths,allmonths);tables['production']=table.to_dict('records')
    test=recover([base.ROOT/'data/ranking.parquet']);test.to_parquet(OUT/'ranking_matches.parquet',index=False)
    keep=table.loc[table.allowed,'airport'];test=test.loc[test.airport.isin(keep)]
    p=pd.read_parquet(base.ROOT/'submissions/unique-rose_v74.parquet')
    test_pred,test_audit=parent.apply(p.MVT_ID_mvt.to_numpy(),p.TAXITIME_SEC_mvt.to_numpy(float),test)
    test_audit.to_parquet(OUT/'ranking_audit.parquet',index=False)
    result=dict(design=DESIGN,support=tables,months=scores,paired=paired,
        cv=float(np.mean([s['candidate'] for s in scores.values()])),ranking_changed=int(test_audit.inside.sum()),
        source=base.artifact(__file__))
    write_json(OUT/'result.json',result);print(result,flush=True)
if __name__=='__main__':main()
