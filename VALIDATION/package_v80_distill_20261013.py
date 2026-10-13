"""Package the audited student old-view candidate as v83; never upload.

Requires separate root preparation allocation, completed production/CV audits,
a fresh live snapshot and all actual-submission fingerprints. Uses the existing
audited_candidate_upload.py manifest contract without changing the uploader.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

import package_portable_feedback_20261013 as prior
import portable_v80_distill_20261013 as student
import submission_slot_policy as slots

ROOT=student.ROOT
OUT=student.OUT/'submission_v83'
VERSION=83
HISTORY=prior.OUT/'actual_submission_fingerprints.json'
SOURCE_REVIEW=OUT/'source_review.json'
art,read,verify=student.art,student.read,student.verify
write=student.write


def fingerprints(snapshot,ids):
    """Reuse authenticated historical hashes; add only completed v81/v82 uploads."""
    history=read(HISTORY);paths={HISTORY};records={}
    for rec in history['records']:
        verify(rec['local']);verify(rec['remote_stat'])
        records[rec['file']]=rec
        paths.update([verify(rec['local']),verify(rec['remote_stat'])])
    for version in [81,82]:
        path=prior.OUT/f'submission_v{version}/manifest.json';m=read(path)
        assert m['state']=='scored' and m['version']==version and m['rows']==344841
        source=ROOT/m['file'];assert art(source)['sha256']==m['sha256']==m['remote_sha256']
        official_path=path.parent/'official_result.json';official=read(official_path)
        assert official==m['official_result'] and official['status']=='Succeeded' and official['used_pairs']==344841
        assert official['file']==source.name
        assert m['duplicate_check']['actual_submission_receipt']==art(HISTORY)
        records[source.name]=dict(file=source.name,local=art(source),
            canonical_prediction_sha256=prior.canonical(source,ids),
            actual_upload_manifest=art(path),actual_result=art(official_path))
        paths.update([path,source,official_path])
    live={rec['file']:rec for rec in snapshot['history']}
    assert len(live)==len(snapshot['history']) and set(live)==set(records),'Unaccounted actual upload; inspect, never omit from duplicate check'
    for name,rec in records.items():
        assert verify(rec['local']).stat().st_size==live[name]['sizeBytes']
    return records,paths


def package(snapshot_path):
    snapshot_path=Path(snapshot_path).resolve()
    assert snapshot_path.is_relative_to(ROOT)
    assert not (OUT/'manifest.json').exists() and not (OUT/'started.json').exists()
    source_review=read(SOURCE_REVIEW)
    assert source_review['status']=='pass' and source_review['source']==art(__file__)
    allocation=read(OUT/'allocation.json')
    assert allocation['authority']=='/root' and allocation['source']==art(__file__)
    assert allocation['identity']==art(student.MANIFEST) and allocation['version']==VERSION
    assert allocation['state']=='released_for_distillation_old_packaging'
    student.guard()
    result_path=student.OUT/'production/result.json';audit_path=student.OUT/'production/audit.json'
    result,audit=read(result_path),read(audit_path)
    score_path=student.OUT/'cv/score.json';cv_audit_path=student.OUT/'cv/audit.json'
    score,cv_audit=read(score_path),read(cv_audit_path)
    assert audit['status']==cv_audit['status']=='pass' and score['gate'] and cv_audit['gate']
    assert result['identity']==audit['identity']==score['identity']==cv_audit['identity']==art(student.MANIFEST)
    assert audit['result']==art(result_path) and cv_audit['score']==art(score_path)
    assert result['score']==art(score_path) and result['cv_audit']==art(cv_audit_path)
    inference=student.inference_guard()
    assert score['inference']==art(student.OUT/'cv/inference.json')
    for view in ['old','final']:
        for key in ['candidate','prediction']:
            assert audit['views'][view][key]==result['views'][view][key]
            verify(result['views'][view][key])
    snapshot=read(snapshot_path);now=datetime.now(timezone.utc)
    assert snapshot['snapshotSource']=='live'
    age=(now-datetime.fromisoformat(snapshot['updatedAt'])).total_seconds()
    assert 0<=age<3600,'A fresh supplied live snapshot is required'
    used=[r['file'] for r in snapshot['history'] if r.get('uploadedAt') and
        datetime.fromisoformat(r['uploadedAt'].replace('Z','+00:00')).astimezone(timezone.utc).date()==now.date()]
    policy=read(slots.POLICY)
    assert policy['utc_date']==now.date().isoformat() and policy['reserved_slots']>=1
    assert len(used)<policy['daily_limit']-policy['reserved_slots'],'Do not consume the reserved last slot'
    filename=f'unique-rose_v{VERSION}.parquet'
    assert filename not in {r['file'] for r in snapshot['history']}
    output=ROOT/'submissions'/filename;assert not output.exists()
    template_path=ROOT/'data/submitting.parquet';template=pq.read_table(template_path)
    assert len(template)==344841
    ids=template['MVT_ID_mvt'].to_numpy();records,paths=fingerprints(snapshot,ids)
    source=verify(result['views']['old']['candidate']);candidate=pq.read_table(source)
    assert candidate.schema==template.schema and len(candidate)==344841
    for column in template.column_names:
        if column!='TAXITIME_SEC_mvt':assert candidate[column].equals(template[column])
    values=candidate['TAXITIME_SEC_mvt'].to_numpy()
    assert candidate['TAXITIME_SEC_mvt'].null_count==0 and np.isfinite(values).all() and (values>=0).all()
    np.testing.assert_array_equal(values,np.rint(values))
    digest=prior.canonical(source,ids)
    assert digest not in {r['canonical_prediction_sha256'] for r in records.values()},'Actual submitted prediction duplicated'
    paths.update([Path(__file__),Path(prior.__file__),Path(slots.__file__),slots.POLICY,SOURCE_REVIEW,
        OUT/'allocation.json',snapshot_path,student.MANIFEST,result_path,audit_path,score_path,cv_audit_path,
        template_path,source,ROOT/'VALIDATION/audited_candidate_upload.py',ROOT/'VALIDATION/scored_blend_submit.py',
        ROOT/'VALIDATION/records.py'])
    paths.update(verify(rec) for rec in read(student.MANIFEST)['sources'])
    paths.update(verify(rec) for rec in [result['model'],score['scored'],score['inference'],
        score['input_audit'],inference['prediction'],*inference['coefficients']])
    paths.update(verify(result['views'][view][key]) for view in ['old','final'] for key in ['candidate','prediction'])
    manifest=dict(version=VERSION,state='prepared',name='fixed_v80_teacher_distillation',file=str(output.relative_to(ROOT)),
        sha256=art(source)['sha256'],rows=344841,jan_jul_cv=score['cv']['submitted'],
        cv_months={m:r['submitted'] for m,r in score['months'].items()},
        cv_change_vs_v82=score['paired']['v82']['change'],cv_change_vs_v80=score['paired']['v80']['change'],
        paired=score['paired'],cv_record=art(score_path),cv_audit=art(cv_audit_path),component=art(result_path),audit=art(audit_path),
        source_candidate=result['views']['old']['candidate'],source_prediction=result['views']['old']['prediction'],
        sources={str(p.relative_to(ROOT)):art(p)['sha256'] for p in sorted(paths)},template=art(template_path),
        authorization='User permits autonomous uploads; this action packages only under root allocation. Upload requires separate root action and preserves the final reserved daily slot.',
        basis_scores={'unique-rose_v80.parquet':218.8525},prepared_at=now.isoformat(),reserved_daily_slot=True,
        snapshot=art(snapshot_path),duplicate_check=dict(canonical_prediction_sha256=digest,
            historical_fingerprints=art(HISTORY),compared_actual_files=sorted(records)),
        interpretation='Blocked same-month student beatsv82 on reusedCV; comparison withv80 is inconclusive and January worse. Originalranking output is in-sample teacher distillation, never fit to rankingtruth. New finalmonths remain unvalidated.',
        final_capability=dict(state='audited',candidate=result['views']['final']['candidate'],audit=art(audit_path)),
        no_official_gain_claimed=True,upload_not_performed=True)
    slot_receipt=slots.check_slot(manifest,used,now.date());assert slot_receipt and slot_receipt['reserved_slots']>=1
    write(OUT/'started.json',dict(source=art(__file__),allocation=art(OUT/'allocation.json'),snapshot=art(snapshot_path)))
    # Exclusive destination creation, then byte-for-byte copy of the audited candidate.
    with source.open('rb') as src,output.open('xb') as dst:shutil.copyfileobj(src,dst)
    assert art(output)['sha256']==manifest['sha256']
    write(OUT/'packaging_audit.json',dict(status='pass',source=art(__file__),file=art(output),
        source_candidate=manifest['source_candidate'],parent_audit=art(audit_path),cv_audit=art(cv_audit_path),
        all_template_columns_exact=True,finite_integer_nonnegative=True,duplicate_check=manifest['duplicate_check'],
        slot_policy=slot_receipt,uploads_today=used,no_upload=True))
    manifest['packaging_audit']=art(OUT/'packaging_audit.json')
    manifest['sources'][manifest['packaging_audit']['path']]=manifest['packaging_audit']['sha256']
    write(OUT/'manifest.json',manifest)
    print(json.dumps(dict(manifest=art(OUT/'manifest.json'),file=art(output),no_upload=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',required=True,type=Path)
    package(parser.parse_args().snapshot)
