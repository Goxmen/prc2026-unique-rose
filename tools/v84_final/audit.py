"""Independent selected-v84 final data/recipe audit; no model fits or inference.

Model availability is a separate required gate. This audit binds existing
model-inference audits and checks the new final composition independently.
"""
from pathlib import Path
from datetime import datetime,timezone
import gc,hashlib,json
import duckdb,numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'VALIDATION/runs/v3_20261014_v84_final'
OLD=ROOT/'VALIDATION/runs/v3_20261014_exact_v80_recovery/prediction_fidelity_recheck'
OLD_SHA='ab5749c68560aa02bae68aadf9e13b962a87f0c5227b2f2303303a41cff16763'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def art(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p):return json.loads(Path(p).read_text())
def verify(a):
 p=ROOT/a['path'];assert sha(p)==a['sha256'],str(p);return p
def same_art(a,p):return a['path']==str(Path(p).relative_to(ROOT)) and a['sha256']==sha(p)
def align(p,ids,idc='mvt_id',cols=None):
 f=pd.read_parquet(p,columns=cols);assert len(f)==len(ids) and f[idc].is_unique and set(f[idc])==set(ids)
 return f.set_index(idc).loc[ids]

def check_v3():
 root=RUN/'v3';ident=read(root/'identity.json');old=read(root/'old.json');final=read(root/'final.json')
 for r in [old,final]:assert r['status']=='pass' and same_art(r['identity'],root/'identity.json')
 assert old['old_selected_replay_exact'] and final['all_rows_fresh_inference'] and final['no_old_id_splicing'] and final['independent_model_reload_float_exact']
 verify(ident['model']);assert ident['model']['sha256']=='da634438f7027d06019f5254e34280a0727c251d56ed1dc63ce3fdf6ef6cca80'
 original=ROOT/'VALIDATION/runs/v3_20261014_exact_v80_recovery/tree_audit/v3_original_replay'
 expected=pd.read_parquet(original/'submissions/unique-rose_v3.parquet');actual=pd.read_parquet(verify(old['prediction']))
 np.testing.assert_array_equal(actual.mvt_id,expected.MVT_ID_mvt);np.testing.assert_array_equal(actual.submitted,expected.TAXITIME_SEC_mvt)
 packet=pd.read_pickle(verify(final['inputs']));X=packet['X'];ids=packet['ids'];assert len(X)==670790 and list(X)==ident['features']
 rawcols=['MVT_ID_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','ADEP_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','FLIGHT_RULE_mvt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','WK_TBL_CAT_flt','AIRCRAFT_OPERATOR_flt','SCHED_TIME_UTC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt','AOBT_3_flt']
 raw=pd.read_parquet(ROOT/'data/final_ranking.parquet',columns=rawcols);raw=raw.loc[raw.PHASE_mvt=='DEP'];assert raw.MVT_ID_mvt.is_unique
 raw=raw.set_index('MVT_ID_mvt').loc[ids].reset_index(drop=True);assert len(raw)==670790
 categories={'airport':'ADEP_mvt','dest':'ADES_mvt','aircraft_type':'AIRCRAFT_TYPE_mvt','flight_rule':'FLIGHT_RULE_mvt','market_segment':'MARKET_SEGMENT_flt','flight_type':'FLIGHT_TYPE_flt','wake_cat':'WK_TBL_CAT_flt','operator':'AIRCRAFT_OPERATOR_flt'}
 values={c:raw[k] for c,k in categories.items()}
 values.update(runway=raw.ADEP_mvt+':'+raw.RUNWAY_mvt,stand=raw.ADEP_mvt+':'+raw.STAND_mvt)
 for c,value in values.items():
  cats=ident['categories'][c];assert isinstance(X[c].dtype,pd.CategoricalDtype) and X[c].cat.categories.tolist()==cats
  expected_cat=pd.Categorical(value.where(value.isin(cats)),categories=cats)
  np.testing.assert_array_equal(X[c].cat.codes.to_numpy(),expected_cat.codes)
 time=raw.MVT_TIME_UTC_mvt
 for c,value in [('hour',time.dt.hour),('dow',time.dt.dayofweek),('month',time.dt.month)]:np.testing.assert_array_equal(X[c].to_numpy(),value.to_numpy())
 for c,k in [('sched','SCHED_TIME_UTC_mvt'),('lobt','LOBT_flt'),('iobt','IOBT_flt'),('eobt1','EOBT_1_flt'),('aobt3','AOBT_3_flt')]:np.testing.assert_array_equal(X[f'mvt_minus_{c}_sec'].to_numpy(),(time-raw[k]).dt.total_seconds().to_numpy())
 assert {str(k):int(v) for k,v in X.isna().sum().items()}==final['feature_missing_counts']
 assert {str(k):int(v) for k,v in time.dt.strftime('%Y-%m').value_counts().items()}==final['month_counts']
 f=pd.read_parquet(verify(final['prediction']));assert len(f)==670790 and f.mvt_id.is_unique and set(f.mvt_id)==set(ids)
 assert np.isfinite(f.raw).all() and (f.raw>=0).all();np.testing.assert_array_equal(np.rint(f.raw).astype('int32'),f.submitted)
 report={'old_selected_integer_rows_exact':344841,'final_rows':670790,'all18_direct_source_features_exact':True,'saved_training_categories_and_missing_values_exact':True,'all_missing_counts_and_month_counts_exact':True,'original_temporal_feature_source_bound_no_temporal_rebuild_repeated':True,'old_receipt':art(root/'old.json'),'final_receipt':art(root/'final.json'),'model':ident['model']}
 del packet,X,raw,f,actual,expected;gc.collect();return report

def main():
 output=RUN/'independent_audit.json';assert not output.exists(),'Preserve existing audit'
 complete=read(RUN/'assembly_complete.json');assert complete['status']=='pass' and complete['rows']==670790
 identity_path=verify(complete['identity']);identity=read(identity_path);producer_audit_path=verify(complete['audit']);producer=read(producer_audit_path)
 assert producer['status']=='pass' and producer['view']=='final' and producer['rows']==670790
 assert same_art(producer['identity'],identity_path)
 result_path=verify(producer['result']);result=read(result_path);assert result['candidate']==producer['candidate']==complete['candidate']
 assert result['view']=='final' and result['rows']==670790 and same_art(result['identity'],identity_path)
 for a in result['outputs']:verify(a)
 candidate_path=verify(result['candidate']);old_path=ROOT/'submissions/unique-rose_v84.parquet';assert sha(old_path)==OLD_SHA==sha(OLD/'old/candidate.parquet')
 old_identity=read(OLD/'assembly_identity.json');fixed=['geometry71','geometry73','versions','weights','runtime','unchanged_posteriors']
 for field in fixed:assert identity[field]==old_identity[field],field
 assert same_art(identity['old_selected_v84']['identity'],OLD/'assembly_identity.json')
 assert same_art(identity['old_selected_v84']['candidate'],OLD/'old/candidate.parquet')
 verify(identity['source']);verify(identity['original_assembler']);assert identity['original_assembler'] in old_identity['sources']
 leaves=read(verify(identity['final_leaf_catalog']));old_leaves=read(OLD/'leaf_catalog.json')
 assert set(leaves['components'])==set(old_leaves['components'])==set(map(str,identity['geometry71']['versions']))
 for v,rec in leaves['components'].items():assert rec['selected_old_component']==old_leaves['components'][v]
 template_path=ROOT/'data/final_submitting.parquet';template=pq.read_table(template_path);candidate=pq.read_table(candidate_path)
 assert candidate.num_rows==template.num_rows==670790 and candidate.schema.equals(template.schema,check_metadata=True)
 assert candidate['MVT_ID_mvt'].equals(template['MVT_ID_mvt']);ids=template['MVT_ID_mvt'].to_numpy()
 assert len(np.unique(ids))==len(ids) and np.isfinite(ids).all()
 values=candidate['TAXITIME_SEC_mvt'].to_numpy();assert candidate['TAXITIME_SEC_mvt'].null_count==0 and pa.types.is_integer(candidate.schema.field('TAXITIME_SEC_mvt').type)
 assert np.isfinite(values).all() and (values>=0).all() and (values==np.rint(values)).all()
 basis=align(RUN/'final/basis48.parquet',ids);leaf_checks=[]
 for v,rec in leaves['components'].items():
  p=verify(rec['final']);idc=rec.get('id_column','mvt_id');val=rec.get('value_column','submitted')
  f=align(p,ids,idc,[idc,val]);np.testing.assert_array_equal(f[val].to_numpy(),basis['v'+v].to_numpy())
  leaf_checks.append({'version':int(v),'artifact':rec['final'],'all670790_values_exact':True});del f
 del basis;gc.collect()
 matrix=align(RUN/'final/basis51.parquet',ids);stages=align(RUN/'final/stages.parquet',ids)
 assert list(matrix)==['v'+str(v) for v in identity['versions']]
 # Explicit SQL summation independently verifies the final affine input and serialization.
 expression='+'.join(f'("v{v}"*CAST({repr(float(w))} AS DOUBLE))' for v,w in zip(identity['versions'],identity['weights']))
 with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
  frame=matrix.reset_index();frame['pos']=np.arange(len(frame));con.register('basis',frame)
  total=con.execute(f'SELECT {expression} AS pred FROM basis ORDER BY pos').fetchnumpy()['pred']
  error=float(np.max(np.abs(total-stages.raw79.to_numpy())));assert error<=1e-8
  np.testing.assert_array_equal(np.rint(np.maximum(stages.raw79,0)).astype('int32'),stages.v79)
  np.testing.assert_array_equal(stages.tabm_v78-stages.v78,stages.tabm_delta)
  np.testing.assert_array_equal(stages.robust_v78-stages.v78,stages.robust_delta)
  independent_v80=np.rint(np.maximum(stages.v79.to_numpy(dtype=float)+stages.tabm_delta+stages.robust_delta,0)).astype('int32')
  np.testing.assert_array_equal(independent_v80,stages.v80);np.testing.assert_array_equal(independent_v80,values)
 metadata=pd.read_parquet(ROOT/'data/final_ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','MVT_TIME_UTC_mvt'])
 metadata=metadata.loc[metadata.PHASE_mvt=='DEP'];assert metadata.MVT_ID_mvt.is_unique and set(metadata.MVT_ID_mvt)==set(ids)
 months=metadata.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').value_counts().to_dict();assert months=={'2026-07':192122,'2026-06':181791,'2026-01':152719,'2026-02':144158}
 del matrix,stages,frame,total,metadata;gc.collect()
 v3=check_v3()
 report={'status':'pass','checked_at_utc':datetime.now(timezone.utc).isoformat(),'source':art(__file__),'candidate':art(candidate_path),
  'assembly_identity':art(identity_path),'producer_audit':art(producer_audit_path),'old_leaderboard_candidate':art(old_path),
  'producer_result':art(result_path),'source_review':art(RUN/'source_review.json'),'rows':670790,
  'checks':{'template_schema_and_metadata_exact':True,'all_template_IDs_in_order_exact':True,'four_month_population_exact':months,
    'finite_nonnegative_integer_predictions':True,'minimum_sec':int(values.min()),'maximum_sec':int(values.max()),
    'all48_declared_final_components_exact':True,'selected_old_recipe_and_coefficients_unchanged':fixed,
    'independent_SQL_affine_max_abs_error':error,'all_final_rounding_and_additive_deltas_exact':True,
    'nonlinear_stage_checks_bound_to_existing_producer_audit':True,'v3':v3},'leaf_checks':leaf_checks,
  'model_weights_availability':'Separate mandatory gate; this prediction audit does not establish all neural states available.',
  'no_fits':True,'no_inference_repeated':True,'no_upload':True}
 output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'pass','receipt':art(output),'candidate':report['candidate']}))

if __name__=='__main__':main()
