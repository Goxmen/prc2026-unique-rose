"""Prepare an audited public-delay reconciliation candidate, without tree/NN fits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import daily_delay_posterior as posterior
from records import sha256, write_json
from submit import fill_template

base=posterior.base
OUT=base.ROOT/'VALIDATION/runs/v3_20261011_daily_delay_submit'

def build(version,parent):
    dest=OUT/f'submission_v{version}';output=base.ROOT/f'submissions/unique-rose_v{version}.parquet'
    assert not dest.exists() and not output.exists(),'Never overwrite a candidate'
    assert parent in [70,71]
    root=posterior.OUT if parent==71 else posterior.OUT/'v70'
    cp=root/('screen.parquet' if parent==71 else 'predictions.parquet')
    cr=root/('screen.json' if parent==71 else 'result.json')
    cv=pd.read_parquet(cp);cv['submitted']=np.rint(np.maximum(cv.pred,0))
    months={m:dict(parent=float(np.sqrt(np.mean((q.parent-q.y)**2))),candidate=float(np.sqrt(np.mean((q.pred-q.y)**2))),
        submitted=float(np.sqrt(np.mean((q.submitted-q.y)**2)))) for m,q in cv.groupby('month')}
    # Exploratory but meaningful evidence: both monthly point estimates improve.
    # The reused-development bootstrap interval is recorded, not suppressed.
    assert all(v['candidate']<v['parent'] for v in months.values())
    train=pd.read_parquet(base.OUT/'rows2025.parquet');source=base.public(2025)
    params=posterior.fit(train,source)
    ranking=pd.read_parquet(base.OUT/'ranking_inputs.parquet')
    template=pq.read_table(base.ROOT/'data/submitting.parquet');ids=template['MVT_ID_mvt'].to_numpy()
    rows=ranking.set_index('mvt_id').loc[ids].reset_index()
    parent_path=base.ROOT/f'submissions/unique-rose_v{parent}.parquet';old=pq.read_table(parent_path)
    assert old.schema==template.schema and old['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    original=old['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    pred,days=posterior.apply(rows,original,base.public(2026),params)
    # Inference frame intentionally has no target or block-time column.
    assert 'y' not in rows and 'BLOCK_TIME_UTC_mvt' not in rows
    table=fill_template(template,ids,pred)
    expected=np.rint(np.maximum(pred,0)).astype(np.int32)
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),expected)
    assert table.schema==template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    assert table['TAXITIME_SEC_mvt'].null_count==0 and np.isfinite(pred).all()
    dest.mkdir(parents=True);pq.write_table(table,output);assert pq.read_table(output).equals(table)
    cv.to_parquet(dest/'cv.parquet',index=False);days.to_parquet(dest/'days.parquet',index=False)
    pd.DataFrame(dict(mvt_id=ids,parent=original,pred=pred,submitted=expected)).to_parquet(dest/'predictions.parquet',index=False)
    write_json(dest/'parameters.json',dict(design=posterior.DESIGN,parameters=params))
    inputs=[Path(__file__),Path(posterior.__file__),Path(base.__file__),cp,cr,parent_path,base.PARENT/'manifest.json',
        base.OUT/'rows2025.parquet',base.OUT/'ranking_inputs.parquet',base.OUT/'identity.json',base.OUT/'sql_audit.json',
        base.ROOT/'data/ranking.parquet',base.ROOT/'data/submitting.parquet',
        base.SOURCE/'manifest.json',base.SOURCE/'all_pre_departure_delays_2025.csv',base.SOURCE/'all_pre_departure_delays_2026.csv',
        posterior.OUT/'design.json',posterior.OUT/'screen.json',posterior.OUT/'paired.json',
        posterior.OUT/'numerical_audit.json',base.ROOT/'VALIDATION/tools/audit_daily_delay.py',
        dest/'parameters.json',dest/'predictions.parquet',dest/'days.parquet']
    for path in sorted((base.ROOT/'data').glob('training_2025-*.parquet')):inputs.append(path)
    sources={str(p.relative_to(base.ROOT)):sha256(p) for p in inputs}
    cvscore=float(np.mean([v['candidate'] for v in months.values()]));cv_int=float(np.mean([v['submitted'] for v in months.values()]))
    write_json(dest/'cv.json',dict(cv=cvscore,serialized_cv=cv_int,months=months,prediction=base.artifact(dest/'cv.parquet'),
        sources=sources,selection='Exploratory public aggregate signal; two reused development months'))
    for p in [dest/'cv.json',dest/'cv.parquet']:sources[str(p.relative_to(base.ROOT))]=sha256(p)
    write_json(dest/'manifest.json',dict(version=version,state='prepared',file=str(output.relative_to(base.ROOT)),sha256=sha256(output),
        rows=len(ids),parent=f'v{parent}',change='public daily pre-departure delay posterior',jan_jul_cv=cvscore,
        cv_record=base.artifact(dest/'cv.json'),sources=sources,authorization='October11 user authorized autonomous uploads',
        audit=dict(template_ids_and_schema_exact=True,readback_exact=True,no_targets_in_inference=True,independent_daily_totals_sql=True),
        external_source='EUROCONTROL all_pre_departure_delays_2025/2026.csv; attribution and original unmodified local files',
        limitations='Jan/Jul point estimates improve; uncertainty interval includes no gain. Leaderboard transfer unestablished. Prior scales fixed; public population mismatch remains.'))
    print(dict(version=version,parent=parent,cv=cvscore,serialized_cv=cv_int,months=months,
        changed_rows=int(np.count_nonzero(expected!=original)),rms_change=float(np.sqrt(np.mean((expected-original)**2)))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,required=True);parser.add_argument('--parent',type=int,choices=[70,71],required=True)
    a=parser.parse_args();build(a.version,a.parent)
