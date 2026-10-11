"""One audited fifth-slot follow-up after actual v75--v78 leaderboard results.

The existing bounded score diagnosis, full candidate builder, separate SQL
auditor and conditional uploader retain their checks. No new fit or optimizer
runs here. A terminal/error receipt prevents automatic retries after any
ambiguous upload outcome; an existing candidate is never overwritten.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

from threadpoolctl import threadpool_limits

import joint_scored_candidate as candidate
import audited_candidate_upload as uploader
from records import sha256,write_json

ROOT=candidate.ROOT


def read(path):return json.loads(Path(path).read_text())


def active_desktop_jobs(directory):
    """Match a live child command, avoiding stale queue files and reused PIDs."""
    active=[]
    for path in directory.glob('*queue.json'):
        q=read(path)
        if q.get('state')!='running' or not q.get('child_pid') or q['child_pid']==os.getpid():continue
        try:
            expected=q['jobs'][q['job']]['commands'][q['stage']]
            actual=Path(f'/proc/{q["child_pid"]}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        except (FileNotFoundError,KeyError,IndexError):continue
        if actual==[str(x).encode() for x in expected]:active.append(dict(pid=q['child_pid'],queue=str(path)))
    return active


def best_score(public):
    ours=[r for r in public if r['teamName']==uploader.transport.live.TEAM
          and r.get('score') is not None and r.get('usedPairs')==344841]
    assert ours,'No accepted official result'
    return min(ours,key=lambda r:r['score'])


class Services:
    def now(self):return datetime.now(timezone.utc)
    def sleep(self,seconds):time.sleep(seconds)
    def public(self):return uploader.transport.live.fetch_leaderboard()
    def uploads(self):return uploader.transport.live.fetch_bucket(strict=True)
    def build(self,version):candidate.build(version)
    def audit(self,version):
        path=ROOT/'VALIDATION/tools/audit_joint_scored_candidate.py'
        spec=importlib.util.spec_from_file_location('joint_followup_auditor',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.audit(version)
    def upload(self,path):uploader.run(path,'upload')
    def result(self,path):uploader.run(path,'result')


def execute(spec,status,services):
    """Fail-stop orchestration; all mutation remains in the retained audited tools."""
    def permitted():
        return not (ROOT/spec['cancel_file']).exists() and read(ROOT/spec['objective'])['state']=='active'
    if not permitted():status('cancelled');return
    if services.now()>=datetime.fromisoformat(spec['expires_utc']):status('expired');return
    result=candidate.read(candidate.diagnosis.OUT/'result.json')
    prepared=candidate.diagnosis.OUT/'prepared.json'
    assert result['prepared']==candidate.art(prepared)
    assert result['design']==candidate.diagnosis.DESIGN and result['independent_mse_identity']
    best=best_score(services.public());upper=result['conservative_rmse_interval'][1]
    if not candidate.useful(upper,best['score']):
        status('skipped_no_useful_gain_or_goal_met',current_best=best,conservative_upper=upper);return
    today=services.now().date()
    uploads=services.uploads()
    used=[r['file'] for r in uploads if r.get('uploadedAt') and
          datetime.fromisoformat(r['uploadedAt'].replace('Z','+00:00')).astimezone(timezone.utc).date()==today]
    if len(used)>=5:status('skipped_quota',uploads_today=used);return
    version=spec['version'];directory=candidate.OUT/f'submission_v{version}'
    output=ROOT/f'submissions/unique-rose_v{version}.parquet'
    assert not directory.exists() and not output.exists(),'Existing candidate requires review; never overwrite'
    assert output.name not in {r['file'] for r in uploads},'Remote name already exists'
    if not permitted():status('cancelled');return
    status('building',diagnosis=candidate.art(candidate.diagnosis.OUT/'result.json'),current_best=best,conservative_upper=upper)
    services.build(version)
    status('auditing');services.audit(version)
    manifest=directory/'manifest.json';value=read(manifest)
    assert value['state']=='prepared' and value['audit']['independent_review']['status']=='pass'
    assert not value['audit']['independent_final_audit_pending']
    assert value['authorization']==spec['authorization']
    assert value['rows']==344841 and isinstance(value['jan_jul_cv'],(int,float))
    assert sha256(ROOT/value['file'])==value['sha256']
    for path,digest in value['sources'].items():assert sha256(ROOT/path)==digest,path
    if not permitted():status('cancelled_prepared',manifest=candidate.art(manifest));return
    if services.now()>=datetime.fromisoformat(spec['expires_utc']):status('expired_prepared',manifest=candidate.art(manifest));return
    best=best_score(services.public())
    if not candidate.useful(value['conservative_rmse_interval'][1],best['score']):
        status('skipped_prepared_no_longer_useful',manifest=candidate.art(manifest),current_best=best);return
    status('uploading',reviewed_manifest=candidate.art(manifest))
    # No retry of upload, even if the transport reports an ambiguous error.
    services.upload(manifest)
    value=read(manifest);assert value['state']=='uploaded' and value['remote_sha256']==value['sha256']
    status('uploaded',manifest=str(manifest.relative_to(ROOT)),sha256=value['sha256'])
    deadline=min(datetime.fromisoformat(spec['expires_utc']).timestamp(),services.now().timestamp()+3600)
    while services.now().timestamp()<deadline:
        services.sleep(30)
        try:services.result(manifest)
        except Exception as exc:status('awaiting_result',last_result_error=str(exc));continue
        value=read(manifest)
        if value['state'] in ['scored','rejected']:
            status(value['state'],score=value.get('score'),official_result=value.get('official_result'),
                   goal_evidence_ready=value['state']=='scored' and value['score']<=215.)
            return
    status('result_pending_timeout')


def run(path):
    path=Path(path).resolve();spec=read(path)
    assert path.is_relative_to(ROOT/'VALIDATION/runs') and spec['version']==79
    assert spec['authorization']=='User authorized autonomous leaderboard uploads'
    assert spec['selection']==candidate.SELECTION
    for source,digest in spec['sources'].items():assert sha256(ROOT/source)==digest,source
    receipt=ROOT/spec['receipt'];assert not receipt.exists(),'Inspect existing execution; never restart automatically'
    lock=receipt.with_suffix('.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    state=dict(spec=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path)),pid=os.getpid(),version=spec['version'])
    def status(value,**extra):
        state.update(state=value,updated_at=datetime.now(timezone.utc).isoformat(),**extra);write_json(receipt,state)
    try:
        status('starting');services=Services()
        while active:=active_desktop_jobs(path.parent):
            if (ROOT/spec['cancel_file']).exists() or read(ROOT/spec['objective'])['state']!='active':status('cancelled');return
            if services.now()>=datetime.fromisoformat(spec['expires_utc']):status('expired');return
            status('waiting_desktop_resource',verified_live_jobs=active);services.sleep(10)
        execute(spec,status,services)
    except Exception as exc:
        status('error',error=str(exc),traceback=traceback.format_exc());raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('spec');args=parser.parse_args()
    with threadpool_limits(1):run(args.spec)
