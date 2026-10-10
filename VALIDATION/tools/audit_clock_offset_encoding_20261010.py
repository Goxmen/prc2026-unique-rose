"""Separate Pandas median recomputation and sample arithmetic for encoder artifacts."""
import hashlib,json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'data/derived/clock_offset_encoding_20261010'
OUT=ROOT/'tmp/research_20261010/error_mechanisms/clock_encoding_audit.json'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
m=json.loads((DIR/'folds.json').read_text())
assert sha(ROOT/'VALIDATION/clock_offset_encoding.py')==m['source_sha256']
for s in m['movement_sources']:assert sha(s['path'])==s['sha256']
con=duckdb.connect(config={'threads':2});con.execute("SET TimeZone='UTC'")
raw=con.execute('''SELECT MVT_ID_mvt::BIGINT AS mvt_id,ADEP_mvt AS airport,
 nullif(STAND_mvt,'') AS stand,nullif(AIRCRAFT_OPERATOR_flt,'') AS op,nullif(RUNWAY_mvt,'') AS rwy,
 epoch(MVT_TIME_UTC_mvt) AS t,epoch(BLOCK_TIME_UTC_mvt) AS b,epoch(AOBT_3_flt) AS a,
 epoch(SCHED_TIME_UTC_mvt) AS s,TAXITIME_SEC_mvt::DOUBLE AS y,
 strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS mo FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',
 [[s['path'] for s in m['movement_sources']]]).df().set_index('mvt_id')
assert raw.index.is_unique and ((raw.t-raw.b)==raw.y).all()
raw['offset']=raw.b-raw.a
clean=raw.a.notna()&raw.b.notna()&raw.s.notna()&raw.y.ge(0)&raw.offset.abs().lt(3600)&(raw.b-raw.s).abs().gt(60)
features=['te_dlt_stand','te_dlt_op','te_dlt_rwy','imp_minus_dlt_stand']
groups={'te_dlt_stand':'stand','te_dlt_op':'op','te_dlt_rwy':'rwy'}
expected_folds={'2025-01':[f'2025-{i:02d}' for i in range(3,12)],'2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
lookup_n=0;sample_n=0;view_reports=[];fold_reports=[]
for held,parts in m['folds'].items():
 for part,info in parts.items():
  assert info['reference_months']==expected_folds[held]
  table=pd.read_parquet(info['path']).set_index('mvt_id');assert sha(info['path'])==info['sha256']
  expected_months=expected_folds[held] if part=='train' else [held]
  expected_ids=raw.index[raw.mo.isin(expected_months)].to_numpy()
  np.testing.assert_array_equal(np.sort(table.index),np.sort(expected_ids))
  assert table.index.is_unique and list(table)==features
  assert all(d==np.dtype('float32') for d in table.dtypes)
  missing=raw.loc[table.index,'a'].isna().to_numpy()
  assert table.loc[missing].isna().all().all()
  fold_reports.append(dict(held=held,part=part,rows=len(table),missing_clock_rows=int(missing.sum())))
  for view,lk in info['lookup_tables'].items():
   refs=expected_folds[held] if part=='held' else [x for x in expected_folds[held] if x!=view]
   assert lk['reference_months']==refs
   if part=='train':assert info['per_query_month_reference_months'][view]==refs
   assert sha(lk['path'])==lk['sha256']
   saved=pd.read_parquet(lk['path'])
   # A separate implementation recomputes every lookup, not just its hash.
   r=raw.loc[raw.mo.isin(refs)&clean]
   prior=r.groupby('airport').offset.agg(['median','size'])
   expected=prior.reset_index().rename(columns={'median':'offset_sec','size':'n'})
   expected['feature']='airport';expected['key']=None
   maps={}
   pieces=[expected[['feature','airport','key','offset_sec','n']]]
   for f,col in groups.items():
    stats=r.groupby(['airport',col],dropna=True).offset.agg(['median','size'])
    stats=stats.loc[stats['size']>=30]
    maps[f]=stats['median']
    frame=stats.reset_index().rename(columns={col:'key','median':'offset_sec','size':'n'})
    frame['feature']=f;pieces.append(frame[['feature','airport','key','offset_sec','n']])
   expected=pd.concat(pieces,ignore_index=True)
   joined=expected.merge(saved,on=['feature','airport','key'],how='outer',suffixes=('_expected','_saved'),indicator=True)
   assert len(joined)==len(saved)==len(expected) and joined['_merge'].eq('both').all()
   np.testing.assert_array_equal(joined.offset_sec_expected,joined.offset_sec_saved)
   np.testing.assert_array_equal(joined.n_expected,joined.n_saved)
   lookup_n+=len(saved)
   query_month=held if part=='held' else view
   q=raw.loc[raw.mo.eq(query_month)]
   sample=q.sample(n=min(12,len(q)),random_state=42)
   missing_q=q.loc[q.a.isna()].head(2)
   sample=pd.concat([sample,missing_q]).loc[lambda x:~x.index.duplicated()]
   for mid,row in sample.iterrows():
    expected_values=[]
    for f,col in groups.items():
     if pd.isna(row.a):v=np.nan
     else:v=maps[f].get((row.airport,row[col]),prior['median'].get(row.airport,np.nan))
     expected_values.append(np.float32(v))
    expected_values.append(np.float32(row.t-row.a-expected_values[0]))
    np.testing.assert_allclose(table.loc[mid,features].to_numpy(),expected_values,rtol=0,atol=0,equal_nan=True)
    sample_n+=4
   view_reports.append(dict(held=held,part=part,view=view,reference_months=refs,
                            clean_rows=len(r),lookup_rows=len(saved),sample_rows=len(sample)))
result=dict(status='passed',source_rows=len(raw),exact_label_relationships=len(raw),lookup_rows_recomputed=lookup_n,
    sample_feature_values=sample_n,all_table_missing_clock_masks_checked=True,folds=fold_reports,views=view_reports,
    manifest_sha256=sha(DIR/'folds.json'),builder_sha256=m['source_sha256'])
OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['folds','views']},indent=2))
