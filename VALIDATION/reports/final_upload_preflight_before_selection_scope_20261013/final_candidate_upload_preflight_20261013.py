"""Read-only validation of an explicitly prepared final submission.

No upload command is present and the manifest is never modified. A passing
receipt is a time-stamped observation: quota and filename must be rechecked at
the actual upload, which must separately use an atomic create-only operation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import dashboard_data
import submission_slot_policy

TEMPLATE=ROOT/'data/final_submitting.parquet'
TEMPLATE_SHA='b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924'
EXPECTED_FILE='submissions/unique-rose_final.parquet'
TARGET='TAXITIME_SEC_mvt'
ID='MVT_ID_mvt'


def read(path):return json.loads(Path(path).read_text())
def artifact(path):
    path=Path(path).resolve();assert path.is_relative_to(ROOT)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest())


def verify(record):
    path=(ROOT/record['path']).resolve();assert path.is_relative_to(ROOT)
    assert artifact(path)=={k:record[k] for k in ['path','sha256']},record['path']
    return path


def pinned(manifest,record):
    assert manifest['sources'].get(record['path'])==record['sha256'],record['path']
    return verify(record)


def check_tables(template,candidate,source,expected_rows=670790):
    assert template.num_rows==candidate.num_rows==source.num_rows==expected_rows
    assert template.schema==candidate.schema==source.schema
    assert candidate[ID].null_count==0 and len(set(candidate[ID].to_pylist()))==expected_rows
    for name in template.column_names:
        if name!=TARGET:assert candidate[name].equals(template[name]),name
        assert candidate[name].equals(source[name]),('audited source differs',name)
    values=candidate[TARGET]
    assert pa.types.is_integer(values.type) and values.null_count==0
    values=values.to_numpy()
    assert np.isfinite(values).all() and (values>=0).all() and np.array_equal(values,np.rint(values))
    return dict(rows=expected_rows,minimum=int(values.min()),maximum=int(values.max()),
                all_template_columns_preserved=True,all_audited_source_columns_exact=True)


def validate_manifest(manifest_path):
    manifest_path=Path(manifest_path).resolve();assert manifest_path.is_relative_to(ROOT/'VALIDATION/runs')
    manifest=read(manifest_path)
    assert manifest['state']=='prepared' and manifest['submission_kind']=='final'
    assert manifest['file']==EXPECTED_FILE and manifest['rows']==670790
    assert isinstance(manifest['authorization'],str) and manifest['authorization'].strip()
    assert isinstance(manifest['jan_jul_cv'],(int,float)) and not isinstance(manifest['jan_jul_cv'],bool)
    assert np.isfinite(manifest['jan_jul_cv']) and manifest['jan_jul_cv']>=0
    assert isinstance(manifest['sources'],dict) and manifest['sources']
    for path,digest in manifest['sources'].items():verify(dict(path=path,sha256=digest))
    helpers=[Path(__file__),Path(dashboard_data.__file__),Path(submission_slot_policy.__file__),ROOT/'VALIDATION/records.py']
    if submission_slot_policy.POLICY.exists():helpers.append(submission_slot_policy.POLICY)
    for helper in helpers:pinned(manifest,artifact(helper))
    assert manifest['template']==dict(path='data/final_submitting.parquet',sha256=TEMPLATE_SHA)
    pinned(manifest,manifest['template'])
    source_path=pinned(manifest,manifest['final_source_prediction'])
    source_result=read(pinned(manifest,manifest['final_source_result']))
    source_audit=read(pinned(manifest,manifest['final_source_audit']))
    assert source_audit['status']=='pass' and source_audit['result']==manifest['final_source_result']
    assert source_audit['identity']==source_result['identity'];pinned(manifest,source_result['identity'])
    if 'views' in source_result:
        assert set(source_result['views'])==set(source_audit['views'])=={'old','cv','final'}
        result,checked=source_result['views']['final'],source_audit['views']['final']
        assert result['rows']==checked['rows']==670790
        assert result['candidate']==checked['candidate']==manifest['final_source_prediction']
        assert result['prediction']==checked['prediction']
        source_kind='audited_nested_final_candidate'
    else:
        # Transport result has no view key; its audited view is still explicit.
        assert source_result.get('view','final')==source_audit['view']=='final'
        assert source_result['rows']==source_audit['rows']==670790
        assert source_result['candidate']==source_audit['candidate']==manifest['final_source_prediction']
        source_kind='audited_final_candidate'
    output=verify(dict(path=manifest['file'],sha256=manifest['sha256']))
    checks=check_tables(pq.read_table(TEMPLATE),pq.read_table(output),pq.read_table(source_path))
    return manifest,dict(manifest=artifact(manifest_path),file=artifact(output),template=manifest['template'],
        final_source_prediction=manifest['final_source_prediction'],final_source_result=manifest['final_source_result'],
        final_source_audit=manifest['final_source_audit'],source_kind=source_kind,table_checks=checks)


def check_live_slots(manifest,uploads,today):
    assert isinstance(uploads,list)
    names=[item['file'] for item in uploads]
    assert len(names)==len(set(names)), 'Duplicate bucket metadata'
    assert Path(manifest['file']).name not in names, 'Remote filename occupied; never overwrite'
    used=[]
    for item in uploads:
        assert item['file'].endswith('.parquet') and item.get('uploadedAt'), 'Incomplete live bucket metadata'
        stamp=datetime.fromisoformat(item['uploadedAt'].replace('Z','+00:00'))
        assert stamp.tzinfo is not None, 'Upload timestamp lacks timezone'
        if stamp.astimezone(timezone.utc).date()==today:used.append(item['file'])
    assert len(used)<5, 'Five-upload UTC-day limit reached'
    policy=submission_slot_policy.check_slot(manifest,used,today)
    return dict(uploads_today=used,daily_limit=5,filename_unoccupied=True,slot_policy=policy)


def run(manifest_path,receipt=None):
    if receipt is not None:
        receipt=Path(receipt).resolve()
        assert receipt.is_relative_to(ROOT/'VALIDATION/runs') and not receipt.exists()
    manifest,local=validate_manifest(manifest_path)
    # This function performs only reads (mc ls/cat) and fails on live access errors.
    uploads=dashboard_data.fetch_bucket(strict=True)
    checked_at=datetime.now(timezone.utc)
    live=check_live_slots(manifest,uploads,checked_at.date())
    value=dict(status='pass_read_only_preflight',at=checked_at.isoformat(),source=artifact(Path(__file__)),
        **local,live=live,authorization=manifest['authorization'],no_upload_performed=True,
        no_manifest_modified=True,requires_fresh_atomic_create_only_upload=True)
    if receipt is not None:
        with receipt.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(value,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();run(args.manifest,args.receipt)
