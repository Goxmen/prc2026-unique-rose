"""Byte-copy the selected, already audited v83 final artifact; never upload.

Root supplies a resolved-selection receipt. This builder neither selects a
candidate nor changes predictions, fits models, reads CV labels, or contacts
the bucket. The separately reviewed final validator/uploader remain unchanged.
Usage: python package_final_v83_20261013.py --selection <root-selection.json>
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import pyarrow.parquet as pq

import final_candidate_upload_preflight_20261013 as preflight
import audited_final_upload_20261013 as uploader

ROOT=preflight.ROOT
STUDENT=ROOT/'VALIDATION/runs/v3_20261013_v80_teacher_distillation'
PUBLIC=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
TILT=PUBLIC/'final/tilt_recovery'
OUT=STUDENT/'final_submission'
CV=289.60760827359303
FINAL_SHA='f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f'
FIXED={
    'identity.json':'94b63f64ca5021036422f70304664d7128beecdd418aa051b3257227cf638d41',
    'cv/score.json':'8ec5d7f37cf1e776d93982e3b9ddce1ab0fe805b2b93e99610419309ba71543c',
    'cv/audit.json':'794f351288f86cf8a2741c6393e673b9e5e5a277e8ea2be591f4e47612323713',
    'production/result.json':'34b7e9b08bb2c005312a5776eb12e34cdff57ebf510e7f45cffaed802783d1d2',
    'production/audit.json':'3eef97502ec645fbb8f5b2ed7995c1d49d3e8b8cc92411b0ae54dd5c91454c4c',
    'production/model.json':'21031645415924c157af701784ea4ea4c5a7944cad601679df373cccdcc2065e',
}
art,read,verify=preflight.artifact,preflight.read,preflight.verify


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def package(selection):
    selection=Path(selection).resolve();assert selection.is_relative_to(ROOT/'VALIDATION/runs')
    output=ROOT/preflight.EXPECTED_FILE;manifest_path=OUT/'manifest.json'
    assert not output.exists() and not manifest_path.exists() and not (OUT/'prepare_started.json').exists()
    review=read(OUT/'source_review.json')
    assert review['status']=='pass' and review['source']==art(Path(__file__))
    sources={}
    def pin(record):
        record={key:record[key] for key in ['path','sha256']};verify(record)
        assert record['path'] not in sources or sources[record['path']]==record['sha256']
        sources[record['path']]=record['sha256'];return record
    for name,digest in FIXED.items():pin(dict(path=str((STUDENT/name).relative_to(ROOT)),sha256=digest))
    identity=read(STUDENT/'identity.json');assert len(identity['sources'])==286
    for record in identity['sources']:pin(record)
    score,cv_audit,production,audit=[read(STUDENT/name) for name in ['cv/score.json','cv/audit.json','production/result.json','production/audit.json']]
    identity_record=art(STUDENT/'identity.json')
    assert score['identity']==cv_audit['identity']==production['identity']==audit['identity']==identity_record
    assert cv_audit['status']==audit['status']=='pass' and score['gate'] is cv_audit['gate'] is True
    assert cv_audit['score']==production['score']==art(STUDENT/'cv/score.json')
    assert production['cv_audit']==audit['cv_audit']==art(STUDENT/'cv/audit.json')
    assert audit['result']==art(STUDENT/'production/result.json') and audit['model']==production['model']==art(STUDENT/'production/model.json')
    assert production['solves']==1 and production['no_taxi_labels_read'] and production['exact_v80_claim'] is False
    assert set(production['views'])==set(audit['views'])=={'old','cv','final'}
    assert abs(preflight.cv_audit_value(audit)-CV)<1e-12 and abs(score['cv']['submitted']-CV)<1e-12
    for key in ['inference','input_audit','scored','allocation']:pin(score[key])
    for record in [production['allocation'],production['model'],identity_record,identity['review'],identity['fixture'],identity['metadata']]:pin(record)
    for view in ['old','final']:
        for key in ['candidate','prediction']:
            assert production['views'][view][key]==audit['views'][view][key];pin(production['views'][view][key])
    source_candidate=production['views']['final']['candidate']
    assert source_candidate['sha256']==FINAL_SHA and production['views']['final']['rows']==audit['views']['final']['rows']==670790
    assert production['views']['cv']==dict(rows=344419,prediction=score['scored'])

    # Retain the selected anchor's explicit numerical-domain extension and its
    # independent event audit; it is inherited by this complete student recipe.
    handoff=read(TILT/'handoff.json');assert handoff['status']=='complete_independently_audited'
    for key in ['result','audit','candidate','numerical_recovery','supplement_result','event_audit']:pin(handoff[key])
    pin(art(TILT/'handoff.json'))
    parent,parent_audit=read(PUBLIC/'final/result.json'),read(PUBLIC/'final/audit.json')
    supplement,event=read(verify(handoff['supplement_result'])),read(verify(handoff['event_audit']))
    recovery=read(verify(handoff['numerical_recovery']))
    assert parent_audit['status']==event['status']=='pass' and supplement['status']=='complete'
    assert handoff['result']==parent_audit['result']==supplement['result']==event['final_result']==art(PUBLIC/'final/result.json')
    assert handoff['audit']==art(PUBLIC/'final/audit.json') and handoff['candidate']==parent['candidate']
    assert handoff['numerical_recovery']==parent_audit['numerical_supplement']==supplement['identity']==event['recovery_identity']
    assert handoff['supplement_result']==parent_audit['numerical_supplement_result']==event['recovery_result']
    assert handoff['event_audit']==parent_audit['independent_event_audit']
    assert parent['identity']==parent_audit['identity']==recovery['original_identity']==art(PUBLIC/'identity.json')
    assert identity['specs']['final']['anchor']['artifact']==parent['prediction']==parent_audit['prediction']
    assert event['no_mean_tolerance_change'] and event['original_success_branch_unmodified'] and event['no_completed_group_repeated']
    for record in recovery['sources']:pin(record)
    for record in [recovery['source'],recovery['auditor'],parent_audit['source'],parent['prediction'],parent['identity']]:pin(record)

    helpers=[Path(__file__),OUT/'source_review.json',Path(preflight.__file__),Path(uploader.__file__),
             Path(uploader.transport.__file__),Path(preflight.dashboard_data.__file__),
             Path(preflight.submission_slot_policy.__file__),ROOT/'VALIDATION/records.py',preflight.submission_slot_policy.POLICY]
    for helper in helpers:
        record=art(helper)
        if helper not in [Path(__file__),OUT/'source_review.json']:assert record in review['helpers']
        pin(record)
    template=pin(art(preflight.TEMPLATE));assert template['sha256']==preflight.TEMPLATE_SHA
    selected=pin(art(selection));decision=read(selection)
    for record in decision['evidence']:pin(record)
    manifest=dict(state='prepared',submission_kind='final',file=preflight.EXPECTED_FILE,sha256=FINAL_SHA,rows=670790,
        template=template,final_source_prediction=source_candidate,final_source_result=art(STUDENT/'production/result.json'),
        final_source_audit=art(STUDENT/'production/audit.json'),jan_jul_cv=CV,jan_jul_cv_scope='complete_portable_recipe',
        jan_jul_cv_source=art(STUDENT/'production/audit.json'),reserved_slot_selection=selected,sources=sources,
        authorization='User authorized autonomous uploads; final delivery reserved for the strongest validated candidate. Root supplied the resolved selection receipt.',
        recipe='Audited fixed v80 teacher distillation anchored on portable v82; same retained production coefficient vector.',
        model=production['model'],cv_score=art(STUDENT/'cv/score.json'),cv_audit=art(STUDENT/'cv/audit.json'),
        inherited_v82_numerical_recovery=dict(handoff=art(TILT/'handoff.json'),identity=handoff['numerical_recovery'],
            result=handoff['supplement_result'],event_audit=handoff['event_audit'],original_success_branch_unchanged=True,
            no_mean_tolerance_change=True),
        qualifications=dict(exact_v80_replay=False,unseen_month_transfer_unvalidated=True,reused_development=True,
                            final_score_unavailable=True,historical_cv_caveat=identity['historical_cv_caveat']),
        source_identity_entries=286,no_new_fit_or_inference=True,no_prediction_changes=True,no_upload_performed=True)
    preflight.check_final_selection(manifest,datetime.now(timezone.utc).date())
    source_table=pq.read_table(verify(source_candidate));template_table=pq.read_table(preflight.TEMPLATE)
    checks=preflight.check_tables(template_table,source_table,source_table)
    write(OUT/'prepare_started.json',dict(source=art(Path(__file__)),review=art(OUT/'source_review.json'),
        selection=selected,source_candidate=source_candidate,at=datetime.now(timezone.utc).isoformat()))
    output.parent.mkdir(parents=True,exist_ok=True)
    with verify(source_candidate).open('rb') as src,output.open('xb') as dst:shutil.copyfileobj(src,dst,8<<20)
    assert art(output)['sha256']==FINAL_SHA
    write(manifest_path,manifest)
    checked,local=preflight.validate_manifest(manifest_path);assert checked==manifest
    preflight.check_final_selection(manifest,datetime.now(timezone.utc).date())
    write(OUT/'package_audit.json',dict(status='pass',source=art(Path(__file__)),manifest=art(manifest_path),
        file=art(output),source_candidate=source_candidate,table_checks=checks,local_preflight=local,
        byte_exact_copy=True,no_prediction_changes=True,no_live_bucket_access=True,no_upload=True))
    print(json.dumps(dict(status='prepared',manifest=art(manifest_path),package_audit=art(OUT/'package_audit.json'),
                         rows=670790,jan_jul_cv=CV,byte_exact_copy=True,no_upload=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selection',type=Path,required=True)
    package(parser.parse_args().selection)
