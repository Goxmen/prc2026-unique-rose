"""Reload all production CatBoosts, replay v75, independently check constraints."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'VALIDATION'))
import numpy as np,pandas as pd,pyarrow.parquet as pq
from catboost import CatBoostClassifier,Pool
import rome_public_delay_submit as run
from records import write_json,sha256

def main():
    dest=run.OUT/'submission_v75';manifest_path=dest/'manifest.json';m=run.read(manifest_path)
    assert m['arm']=='cat' and m['state']=='prepared'
    for p,digest in m['sources'].items():assert sha256(ROOT/p)==digest,p
    prepared=run.read(run.OUT/'prepared.json');d=pd.read_pickle(run.verify(prepared['old']))
    _,cats,_=run.parent.parent_spec();X=run.parent.cbj.frame(d['X'],cats);pred=[];reports=[]
    for seed in run.parent.SEEDS:
        path=run.OUT/'predictions'/f'cat_s{seed}.parquet';record=run.read(path.with_suffix('.json'))
        model=CatBoostClassifier();model.load_model(str(run.verify(record['model'])))
        actual=model.predict_proba(Pool(X,cat_features=cats),thread_count=1)[:,1]
        saved=pd.read_parquet(run.verify(record['prediction']));np.testing.assert_array_equal(saved.mvt_id,d['ids'])
        np.testing.assert_array_equal(actual,saved.p.to_numpy()[d['use']]);assert (saved.p.to_numpy()[~d['use']]==0).all()
        pred.append(saved.p.to_numpy(float));reports.append(dict(seed=seed,rows=len(actual),max_error=0.))
    physical=run.physical;data=physical.parent.ranking_data('old');ids=data['rank_ids']
    v70=run.read(physical.OUT/'submission_v70/manifest.json');ref=pd.read_parquet(run.verify(v70['parts']))
    oldcat=pd.read_parquet(run.verify(run.read(run.parent.OUT/'old.json')['prediction']))
    np.testing.assert_array_equal(oldcat.mvt_id,ids)
    prob=.5*(np.mean(pred,axis=0)+np.clip(2*ref.p.to_numpy()-oldcat.p.to_numpy(),0,1))
    new,_=physical.parent.final_frame(data,ref.nn_base.to_numpy(),ref.gbm_base.to_numpy(),prob)
    out=pd.read_parquet(dest/'predictions.parquet');np.testing.assert_array_equal(out.mvt_id,ids)
    np.testing.assert_array_equal(out.candidate,new.pred);np.testing.assert_array_equal(out.p,prob)
    best=pd.read_parquet(ROOT/'submissions/unique-rose_v74.parquet');np.testing.assert_array_equal(best.MVT_ID_mvt,ids)
    q=best.TAXITIME_SEC_mvt.to_numpy(float)+new.pred.to_numpy()-ref.pred.to_numpy()
    raw=pd.read_parquet(run.features.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids]
    known=raw.has_id.to_numpy(bool)&raw.nm_gap.notna().to_numpy();gap=raw.nm_gap.to_numpy(float)
    q[known]=np.maximum(0,np.minimum(np.maximum(q[known],gap[known]-3606),gap[known]+3606));q=np.maximum(q,0)
    matches=pd.read_parquet(run.r20.constraints.exclusion.OUT/'ranking_matches.parquet');pos={v:i for i,v in enumerate(ids)}
    for row in matches.loc[matches.accepted].itertuples():
        i=pos[row.mvt_id];lo=row.recovered_gap-3606;hi=row.recovered_gap+3606
        if lo<q[i]<hi:
            possible=[x for x in [lo,hi] if x>=0];q[i]=min(possible,key=lambda x:abs(q[i]-x))
    np.testing.assert_array_equal(q,out.pred)
    expected=np.rint(q).astype('int32');file=ROOT/m['file'];table=pq.read_table(file);template=pq.read_table(ROOT/'data/submitting.parquet')
    assert table.schema==template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),expected)
    unchanged=raw.airport.ne('LIRF').to_numpy();np.testing.assert_array_equal(expected[unchanged],best.TAXITIME_SEC_mvt.to_numpy()[unchanged])
    cv=pd.read_parquet(dest/'cv.parquet');cv_score=float(np.mean([np.sqrt(np.mean((f.pred-f.y)**2)) for _,f in cv.groupby('month')]))
    assert abs(cv_score-m['jan_jul_cv'])<1e-10
    result=dict(status='pass',models=reports,full_prediction_replay_rows=len(ids),independent_constraints_exact=True,
        unchanged_non_rome_rows=int(unchanged.sum()),cv=cv_score,sha256=sha256(file),source=run.artifact(__file__))
    write_json(dest/'audit.json',result)
    paths=[Path(__file__),dest/'audit.json',run.OUT/'ranking_feature_audit.json',run.cat.OUT/'audit.json',run.features.OUT/'audit.json']
    for p in paths:m['sources'][str(p.relative_to(ROOT))]=sha256(p)
    m['audit']['independent_review']=result;write_json(manifest_path,m);print(result,flush=True)
if __name__=='__main__':main()
