"""Saved-output audit of labels above the inherited three-hour neural cap."""
import os,sys,json
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,pandas as pd,duckdb
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import rome_signals as signals,opdi_submit
OUT=ROOT/'tmp/research_20261010/error_mechanisms'
f=pd.read_parquet(OUT/'diagnostic_rows.parquet');out={'held':{},'training':{}}
for month,g in f.groupby('month'):
 g=g.reset_index(drop=True)
 meta=json.load(open(ROOT/'VALIDATION/runs/v3_20261002_adsblol_jan_jul/splits'/month/'meta.json'))
 def final(base):
  x=signals.combine(base,g.p.to_numpy(),g,g,late_mean=meta['parameters']['late_mean'],dayshift_weight=None)['final']
  return opdi_submit.apply_corrections(x,g,g,meta['corrections'],g.p.to_numpy())[0].pred.to_numpy()
 baseline=final(g.base.to_numpy());np.testing.assert_allclose(baseline,g.pred,atol=1e-8)
 sensitivity=final(g.base.to_numpy()+1)-baseline
 tail=g.y.gt(10800).to_numpy();active=sensitivity>1e-6
 masks={'all_tail':tail,'tail_ordinary_active':tail&active,
   'tail_ordinary_active_excluding_two_LFPG_extremes':tail&active&~g.mvt_id.isin([182378789,182377662]).to_numpy()}
 results={}
 for name,mask in masks.items():
  sub=g.loc[mask];sse=float(sub.se.sum());total=float(g.se.sum());n=len(g)
  results[name]=dict(rows=int(mask.sum()),sse=sse,sse_share=sse/total,
   oracle_global_rmse_gain=float(np.sqrt(total/n)-np.sqrt((total-sse)/n)),
   airport_counts=sub.groupby('airport',observed=True).size().to_dict(),
   actual_aobt_rows=int(sub.AOBT_3_flt.notna().sum()))
 out['held'][month]=results
 tailrows=g.loc[tail,['mvt_id','month','airport','y','pred','base','se','group','raw_flight_id_missing','p','gap','day']].copy()
 tailrows['ordinary_derivative']=sensitivity[tail]
 tailrows.to_csv(OUT/f'nn_tail_cap_{month}_rows.csv',index=False)
con=duckdb.connect(config={'threads':2});con.execute("SET TimeZone='UTC'")
r=con.execute('''SELECT MVT_ID_mvt::BIGINT AS mvt_id,ADEP_mvt AS airport,TAXITIME_SEC_mvt::DOUBLE AS y,
 FLIGHT_ID_mvt IS NULL AS noid,AOBT_3_flt IS NOT NULL AS actual_aobt,
 epoch(MVT_TIME_UTC_mvt)-epoch(SCHED_TIME_UTC_mvt) AS gap,
 strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',
 [str(ROOT/'data/training_2025-*.parquet')]).df()
folds={'2025-01':[f'2025-{i:02d}' for i in range(3,12)],'2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
for held,months in folds.items():
 q=r[r.month.isin(months)&r.y.between(1,200000)].copy()
 rule=q.airport.eq('LIRF')&q.noid&q.gap.ge(14400)
 copy=q.airport.eq('LIRF')&(q.gap-q.y).abs().le(60)
 ordinary=q.loc[~rule&~copy]
 meta=json.load(open(ROOT/'VALIDATION/runs/v3_20261002_adsblol_jan_jul/splits'/held/'meta.json'))
 assert int(r.month.isin(months).sum())==meta['training_rows'];assert int(rule.sum())==meta['excluded_rule_rows'];assert len(ordinary)==meta['regressor_rows']
 tail=ordinary.loc[ordinary.y>10800]
 out['training'][held]=dict(ordinary_rows=len(ordinary),tail_rows=len(tail),tail_fraction=len(tail)/len(ordinary),
   tail_above21600=int((tail.y>21600).sum()),tail_actual_aobt_rows=int(tail.actual_aobt.sum()),
   tail_by_airport=tail.groupby('airport',observed=True).size().to_dict(),
   tail_by_airport_and_noid={str(k):int(v) for k,v in tail.groupby(['airport','noid'],observed=True).size().items()},
   reference_months=months)
 tail.to_csv(OUT/f'nn_tail_cap_{held}_training.csv',index=False)
record=json.load(open(ROOT/'VALIDATION/runs/v3_20261005_hpo_jan_jul/nn.json'))
out['existing_hpo']={name:record['scores'][name] for name in ['N00','N09','N14','N15']}
out['existing_hpo']['note']='N14 increases clip10800 to21600 against batch4096 N00, not current batch2048 N09; three networks seed42. No evidence of full uncapped trial.'
(OUT/'nn_tail_cap_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
