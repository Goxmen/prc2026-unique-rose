"""Independent packaging/provenance check; no model/CV replay or remote calls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
from submission_slot_policy import check_slot
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_feedback_candidates'


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):digest.update(b)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest())
def verify(record):
    p=ROOT/record['path'];assert art(p)==record,record['path'];return p


def main():
    start=time.monotonic();target=OUT/'independent_packaging_review.json';assert not target.exists()
    manifests=[read(OUT/f'submission_v{v}/manifest.json') for v in [81,82]]
    sources={}
    for m in manifests:
        for path,digest in m['sources'].items():
            assert sources.get(path,digest)==digest;sources[path]=digest
    for p,h in sources.items():verify(dict(path=p,sha256=h))
    template=pq.read_table(ROOT/'data/submitting.parquet');ids=template['MVT_ID_mvt'].to_numpy()
    assert template.num_rows==344841 and len(set(ids))==344841
    history=read(OUT/'actual_submission_fingerprints.json');snapshot=read(verify(history['snapshot']))
    assert snapshot['snapshotSource']=='live' and history['actual_remote_files_only'] is True
    records=history['records'];assert len(records)==len(snapshot['history'])==70
    assert {r['file'] for r in records}=={r['file'] for r in snapshot['history']}
    def canonical(path):
        t=pq.read_table(path,columns=['MVT_ID_mvt','TAXITIME_SEC_mvt']);idx=pd.Index(t['MVT_ID_mvt'].to_numpy())
        assert idx.is_unique;at=idx.get_indexer(ids);assert (at>=0).all()
        pred=t['TAXITIME_SEC_mvt'].to_numpy()[at].astype('<f8');assert np.isfinite(pred).all()
        return hashlib.sha256(pred.tobytes()).hexdigest()
    seen=set()
    for record in records:
        p=verify(record['local']);raw=p.read_bytes();stat=read(verify(record['remote_stat']))
        assert stat['name']==record['file'] and stat['status']=='success' and stat['size']==len(raw)
        metadata={str(k).lower():v for k,v in stat.get('metadata',{}).items()}
        remote_sha=metadata.get('x-amz-meta-sha256',metadata.get('sha256'));etag=stat['etag'].strip('"')
        assert (remote_sha==record['local']['sha256']) or ('-' not in etag and hashlib.md5(raw).hexdigest()==etag)
        digest=canonical(p);assert digest==record['canonical_prediction_sha256'];seen.add(digest)
    source_cv=read(verify(manifests[0]['cv_record']));cv_audit=read(verify(manifests[0]['cv_audit']))
    assert cv_audit['status']=='pass' and cv_audit['metrics']['score']==manifests[0]['cv_record']
    assert cv_audit['identity']==source_cv['identity'] and cv_audit['result']==source_cv['result']
    then=datetime.fromisoformat(snapshot['updatedAt']).astimezone(timezone.utc)
    used=[r['file'] for r in snapshot['history'] if r.get('uploadedAt') and datetime.fromisoformat(r['uploadedAt'].replace('Z','+00:00')).astimezone(timezone.utc).date()==then.date()]
    assert len(used)+2<=4
    results=[]
    for m,v in zip(manifests,[81,82]):
        path=OUT/f'submission_v{v}/manifest.json';assert m['version']==v and m['state']=='prepared' and m['rows']==344841
        assert m['file']==f'submissions/unique-rose_v{v}.parquet' and Path(m['file']).name not in {r['file'] for r in records}
        assert m['authorization'] and m['maximum_exploratory_uploads']==2 and m['reserved_daily_slot'] is True
        assert m['no_official_gain_claimed'] is True and m['upload_not_performed'] is True
        assert m['template']==art(ROOT/'data/submitting.parquet')
        output=verify(dict(path=m['file'],sha256=m['sha256']));assert m['sha256']==m['source_candidate']['sha256']
        assert pq.read_table(output).equals(pq.read_table(verify(m['source_candidate'])))
        table=pq.read_table(output);assert table.schema==template.schema and table.num_rows==344841
        for key in template.column_names:
            if key!='TAXITIME_SEC_mvt':assert table[key].equals(template[key])
        pred=table['TAXITIME_SEC_mvt'];assert pred.null_count==0
        y=pred.to_numpy();assert np.isfinite(y).all() and (y>=0).all() and np.array_equal(y,np.rint(y))
        digest=canonical(output);assert digest==m['duplicate_check']['canonical_prediction_sha256'] and digest not in seen;seen.add(digest)
        assert m['duplicate_check']['actual_submission_receipt']==art(OUT/'actual_submission_fingerprints.json')
        assert m['duplicate_check']['compared_actual_files']==[r['file'] for r in records]
        audit=read(verify(m['audit']));assert audit['status']=='pass'
        result=read(verify(audit['result']));assert audit['identity']==result['identity']
        if v==81:
            assert audit['views']['old']['candidate']==result['views']['old']['candidate']==m['source_candidate']
            assert audit['views']['old']['prediction']==result['views']['old']['prediction']==m['source_prediction']
            assert result['cv']==m['jan_jul_cv']
        else:
            assert audit['view']==result['view']=='old'
            assert audit['candidate']==result['candidate']==m['source_candidate']
            assert audit['prediction']==result['prediction']==m['source_prediction']
        key=m['cv_metric_key'];assert key==('direct_reserve' if v==81 else 'submitted')
        assert m['jan_jul_cv']==source_cv['cv'][key]
        assert m['cv_months']=={month:values[key] for month,values in source_cv['months'].items()}
        assert abs(m['jan_jul_cv']-np.mean(list(m['cv_months'].values())))<1e-9
        assert abs(m['cv_change_vs_v80']-(m['jan_jul_cv']-source_cv['cv']['v80']))<1e-9 and m['cv_change_vs_v80']>0
        check_slot(m,used,then.date());used.append(Path(m['file']).name)
        packaging=read(verify(m['packaging_audit']));assert packaging['status']=='pass' and packaging['file']==art(output)
        results.append(dict(version=v,manifest=art(path),prediction=art(output),rows=344841,cv=m['jan_jul_cv'],
                            cv_change_vs_v80=m['cv_change_vs_v80'],canonical_digest=digest))
    receipt=dict(status='pass',source=art(Path(__file__)),manifests=results,source_files_verified=len(sources),
        historical_predictions_independently_rehashed=70,no_duplicate_actual_or_sibling=True,
        snapshot=history['snapshot'],snapshot_time=snapshot['updatedAt'],quota_snapshot_only=True,
        fresh_upload_time_checks_required=True,final_daily_slot_preserved_at_snapshot=True,
        purpose='Two limited portable-parent official comparisons; both regress reused Jan/Jul CV, no improvement claimed',
        no_models_or_CV_replayed=True,no_remote_reads=True,no_upload=True,seconds=time.monotonic()-start)
    target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status='pass',receipt=art(target),seconds=receipt['seconds'],versions=[81,82])))


if __name__=='__main__':main()
