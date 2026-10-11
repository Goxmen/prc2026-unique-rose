"""Compose independently screened public monthly levels and plan constraints.

Preparation only. The raw-parent predictions and every intermediate stage are
retained. CV always uses the original Jan/Jul fold training-month restrictions.
"""
import argparse,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import monthly_taxi_reconcile as monthly
import recovered_plan_exclusion_fixed as exclusion
import daily_delay_reconcile as base
from records import sha256,write_json
from submit import fill_template

OUT=base.ROOT/'VALIDATION/runs/v3_20261011_public_constraints'
PARENT=monthly.PARENT

def matched_bound(rows,pred):
    x=np.asarray(pred,float).copy();gap=rows.nm_gap.to_numpy(float)
    eligible=rows.has_id.to_numpy(bool)&np.isfinite(gap)
    x[eligible]=np.maximum(0,np.clip(x[eligible],gap[eligible]-3606,gap[eligible]+3606))
    return x

def summarize(frame,column):
    return {m:float(np.sqrt(np.mean((q[column]-q.y)**2))) for m,q in frame.groupby('month')}

def prepare(version):
    dest=OUT/f'submission_v{version}';output=base.ROOT/f'submissions/unique-rose_v{version}.parquet'
    assert not dest.exists() and not output.exists(),'Preserve prepared candidates'
    import tuning_seasonal as ts
    mr=ts.read(monthly.OUT/'result.json');er=ts.read(exclusion.OUT/'result.json')
    assert mr['paired']['decision']=='improvement' and er['paired']['decision']=='improvement'
    assert mr['source']==base.artifact(monthly.__file__) and er['source']==base.artifact(exclusion.__file__)
    train=pd.read_parquet(monthly.OUT/'rows2025.parquet')
    ordinary=pd.read_parquet(base.OUT/'rows2025.parquet')
    cv=pd.read_parquet(PARENT/'cv.parquet');ids=cv.mvt_id.to_numpy()
    mrows=train.set_index('mvt_id').loc[ids].reset_index();brows=ordinary.set_index('mvt_id').loc[ids].reset_index()
    np.testing.assert_array_equal(cv.y,brows.y);np.testing.assert_array_equal(cv.y,mrows.y)
    matches=pd.read_parquet(exclusion.OUT/'matches2025.parquet')
    frame=cv[['mvt_id','month','day','airport','y']].copy();frame['parent']=cv.pred
    after_monthly=cv.pred.to_numpy(float).copy();parameters={}
    for month,allowed in base.FOLDS.items():
        ix=np.flatnonzero(cv.month.eq(month));parameters[month]=monthly.fit(train[train.utc_month.isin(allowed)],monthly.public(2025))
        after_monthly[ix],_=monthly.apply(mrows.iloc[ix].reset_index(drop=True),after_monthly[ix],monthly.public(2025),parameters[month])
    expected=pd.read_parquet(monthly.OUT/'predictions.parquet').set_index('mvt_id').loc[ids,'pred'].to_numpy()
    np.testing.assert_array_equal(after_monthly,expected)
    frame['monthly']=after_monthly;frame['matched_bound']=matched_bound(brows,after_monthly)
    final=frame.matched_bound.to_numpy().copy()
    for month,allowed in base.FOLDS.items():
        ix=np.flatnonzero(cv.month.eq(month));permitted=matches[matches.donor_month.isin(allowed+[month])]
        final[ix],_=exclusion.apply(ids[ix],final[ix],permitted)
    frame['pred']=final;frame['submitted']=np.rint(np.maximum(final,0))
    stages={name:summarize(frame,name) for name in ['parent','monthly','matched_bound','pred','submitted']}
    paired=ts.paired_bootstrap(frame.day,frame.month,(frame.parent-frame.y)**2,(frame.pred-frame.y)**2)
    assert paired['decision']=='improvement'
    # Full-year composition parameters see full2025 labels, never ranking labels.
    full_params=monthly.fit(train,monthly.public(2025))
    rank=monthly.load([base.ROOT/'data/ranking.parquet'],False)
    brank=pd.read_parquet(base.OUT/'ranking_inputs.parquet')
    template_path=base.ROOT/'data/submitting.parquet';template=pq.read_table(template_path);test_ids=template['MVT_ID_mvt'].to_numpy()
    rank=rank.set_index('mvt_id').loc[test_ids].reset_index();brank=brank.set_index('mvt_id').loc[test_ids].reset_index()
    parent_file=base.ROOT/'submissions/unique-rose_v73.parquet';parent=pq.read_table(parent_file)
    assert parent['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    raw=parent['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    month_pred,cells=monthly.apply(rank,raw,monthly.public(2026),full_params)
    bound_pred=matched_bound(brank,month_pred)
    test_matches=pd.read_parquet(exclusion.OUT/'ranking_matches.parquet')
    pred,audit=exclusion.apply(test_ids,bound_pred,test_matches)
    assert 'y' not in rank and 'y' not in brank
    table=fill_template(template,test_ids,pred)
    assert table.schema==template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    expected=np.rint(np.maximum(pred,0)).astype(np.int32)
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),expected)
    dest.mkdir(parents=True);frame.to_parquet(dest/'cv.parquet',index=False);cells.to_parquet(dest/'monthly_cells.parquet',index=False)
    audit.to_parquet(dest/'exclusion_audit.parquet',index=False)
    pd.DataFrame(dict(mvt_id=test_ids,parent=raw,monthly=month_pred,matched_bound=bound_pred,pred=pred,submitted=expected)).to_parquet(dest/'predictions.parquet',index=False)
    write_json(dest/'parameters.json',dict(folds=parameters,full_year=full_params))
    cvscore=float(np.mean(list(stages['pred'].values())))
    write_json(dest/'cv.json',dict(cv=cvscore,stages=stages,paired=paired,prediction=base.artifact(dest/'cv.parquet'),
        caveat='Reused Jan/Jul development; exact edge has empirical source-matching support, not a mathematical guarantee for nonconvex label support.'))
    inputs=[__file__,monthly.__file__,exclusion.__file__,base.__file__,PARENT/'manifest.json',PARENT/'cv.parquet',
        monthly.OUT/'result.json',monthly.OUT/'rows2025.parquet',monthly.OUT/'predictions.parquet',exclusion.OUT/'result.json',
        exclusion.OUT/'cache_identity.json',exclusion.OUT/'matches2025.parquet',exclusion.OUT/'ranking_matches.parquet',
        base.OUT/'rows2025.parquet',base.OUT/'ranking_inputs.parquet',base.ROOT/'data/ranking.parquet',template_path,parent_file,
        base.SOURCE/'manifest.json',base.SOURCE/'taxi_out_additional_time_2025.csv',base.SOURCE/'taxi_out_additional_time_2026.csv',
        dest/'parameters.json',dest/'cv.json',dest/'cv.parquet',dest/'monthly_cells.parquet',dest/'exclusion_audit.parquet',dest/'predictions.parquet']
    from pathlib import Path
    sources={str(Path(p).relative_to(base.ROOT)):sha256(p) for p in inputs}
    pq.write_table(table,output);assert pq.read_table(output).equals(table)
    write_json(dest/'manifest.json',dict(version=version,state='prepared',file=str(output.relative_to(base.ROOT)),sha256=sha256(output),
        parent='v73',change='public monthly-level and recovered-plan correction stage',rows=len(test_ids),jan_jul_cv=cvscore,
        cv_record=base.artifact(dest/'cv.json'),sources=sources,authorization='October11 user authorized autonomous uploads',
        audit=dict(template_ids_and_schema_exact=True,readback_exact=True,monthly_cv_reproduced_exact=True,no_targets_in_inference=True),
        changed_rows=int((expected!=raw).sum()),matched_bound_changes=int((month_pred!=bound_pred).sum()),
        recovered_exclusion_changes=int((pred!=bound_pred).sum()),limitations=monthly.DESIGN['limitation']))
    print(dict(version=version,cv=cvscore,stages=stages,paired=paired,changed_rows=int((expected!=raw).sum()),
        matched_bound_changes=int((month_pred!=bound_pred).sum()),recovered_exclusion_changes=int((pred!=bound_pred).sum())),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);a=p.parse_args();prepare(a.version)
