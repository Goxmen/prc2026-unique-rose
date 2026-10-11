"""Isolated coordinator failure checks; no live API, real candidate or upload."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'VALIDATION'))
import joint_scored_followup as run
from records import sha256,write_json


def check():
    results={}
    with tempfile.TemporaryDirectory(prefix='joint_resource_fixture_') as temporary:
        directory=Path(temporary);argv=[sys.executable,'-c','import time; time.sleep(30)']
        process=subprocess.Popen(argv)
        try:
            q=dict(state='running',child_pid=process.pid,job=0,stage=0,jobs=[dict(commands=[argv])])
            write_json(directory/'test_queue.json',q)
            found=run.active_desktop_jobs(directory);assert len(found)==1 and found[0]['pid']==process.pid
            q['jobs'][0]['commands']=[['different-command']];write_json(directory/'test_queue.json',q)
            assert not run.active_desktop_jobs(directory)
            q['child_pid']=__import__('os').getpid();write_json(directory/'test_queue.json',q)
            assert not run.active_desktop_jobs(directory)
        finally:process.terminate();process.wait(timeout=5)
        results['resource_gate']=dict(pass_check=True,live_command_verified=True,stale_pid_command_and_self_ignored=True)
    for case in ['success','goal_met','small_gain','quota','audit_failure','ambiguous_upload',
                 'cancelled','cancel_after_audit','better_score_before_upload','expired','occupied','changed_source']:
        with tempfile.TemporaryDirectory(prefix='joint_followup_fixture_') as temporary:
            root=Path(temporary);out=root/'VALIDATION/runs/followup';out.mkdir(parents=True)
            cand=root/'VALIDATION/runs/candidate';geo=root/'VALIDATION/runs/geometry';geo.mkdir()
            (root/'submissions').mkdir();write_json(out/'objective.json',dict(state='active'))
            (root/'source.py').write_text('# synthetic fixture source identity\n')
            def art(path):return dict(path=str(Path(path).relative_to(root)),sha256=sha256(path))
            run.ROOT=root;run.candidate.OUT=cand;run.candidate.art=art
            run.candidate.diagnosis=types.SimpleNamespace(OUT=geo,DESIGN=dict(synthetic=True))
            write_json(geo/'prepared.json',dict(synthetic_only=True))
            upper=221.6 if case=='small_gain' else 214.8
            write_json(geo/'result.json',dict(prepared=art(geo/'prepared.json'),design=dict(synthetic=True),
                independent_mse_identity=True,conservative_rmse_interval=[214.1,upper]))
            spec=dict(version=79,authorization='User authorized autonomous leaderboard uploads',selection=run.candidate.SELECTION,
                sources={'source.py':sha256(root/'source.py')},objective='VALIDATION/runs/followup/objective.json',
                cancel_file='VALIDATION/runs/followup/cancel',expires_utc='2026-10-14T09:30:00+00:00',
                receipt='VALIDATION/runs/followup/execution.json')
            specpath=out/'spec.json';write_json(specpath,spec)
            if case=='cancelled':(root/spec['cancel_file']).touch()
            if case=='occupied':(root/'submissions/unique-rose_v79.parquet').write_bytes(b'pre-existing synthetic fixture')
            if case=='changed_source':(root/'source.py').write_text('# changed synthetic fixture\n')
            calls=[];clock=[datetime(2026,10,12,1,tzinfo=timezone.utc)];public_calls=[0]
            class FakeServices:
                def now(self):
                    return datetime(2026,10,15,tzinfo=timezone.utc) if case=='expired' else clock[0]
                def sleep(self,seconds):clock[0]+=timedelta(seconds=seconds)
                def public(self):
                    calls.append('public');public_calls[0]+=1
                    score=214.9 if case=='goal_met' or (case=='better_score_before_upload' and public_calls[0]>1) else 221.6881
                    return [dict(teamName=run.uploader.transport.live.TEAM,score=score,usedPairs=344841,filename='synthetic-fixture.parquet')]
                def uploads(self):
                    calls.append('uploads')
                    return [dict(file=f'synthetic-{k}.parquet',uploadedAt=clock[0].isoformat()) for k in range(5 if case=='quota' else 4)]
                def build(self,version):
                    calls.append('build');destination=cand/f'submission_v{version}';destination.mkdir(parents=True)
                    file=root/f'submissions/unique-rose_v{version}.parquet';file.write_bytes(b'synthetic candidate fixture; not actual parquet')
                    write_json(destination/'manifest.json',dict(state='prepared',version=version,file=str(file.relative_to(root)),
                        sha256=sha256(file),sources={'source.py':sha256(root/'source.py')},audit=dict(independent_final_audit_pending=True),
                        rows=344841,jan_jul_cv=300.,authorization=spec['authorization'],conservative_rmse_interval=[214.1,upper]))
                def audit(self,version):
                    calls.append('audit')
                    if case=='audit_failure':raise AssertionError('synthetic audit failure')
                    path=cand/f'submission_v{version}/manifest.json';value=run.read(path)
                    value['audit']=dict(independent_final_audit_pending=False,independent_review=dict(status='pass'));write_json(path,value)
                    if case=='cancel_after_audit':(root/spec['cancel_file']).touch()
                def upload(self,path):
                    calls.append('upload')
                    if case=='ambiguous_upload':raise RuntimeError('synthetic ambiguous transport outcome')
                    value=run.read(path);value.update(state='uploaded',remote_sha256=value['sha256']);write_json(path,value)
                def result(self,path):
                    calls.append('result');value=run.read(path)
                    value.update(state='scored',score=214.7,official_result=dict(status='Succeeded',used_pairs=344841,score=214.7));write_json(path,value)
            run.Services=FakeServices
            failures=['audit_failure','ambiguous_upload','occupied','changed_source']
            try:run.run(specpath)
            except (AssertionError,RuntimeError):assert case in failures,case
            else:assert case not in failures,case
            receipt=out/'execution.json';state=run.read(receipt)['state'] if receipt.exists() else 'no_receipt'
            if case=='success':
                assert state=='scored' and run.read(receipt)['goal_evidence_ready']
                assert [c for c in calls if c in ['build','audit','upload','result']]==['build','audit','upload','result']
            elif case in ['goal_met','small_gain']:assert state=='skipped_no_useful_gain_or_goal_met' and 'build' not in calls
            elif case=='quota':assert state=='skipped_quota' and 'build' not in calls
            elif case=='cancelled':assert state=='cancelled' and not calls
            elif case=='expired':assert state=='expired' and not calls
            elif case=='cancel_after_audit':assert state=='cancelled_prepared' and 'upload' not in calls
            elif case=='better_score_before_upload':assert state=='skipped_prepared_no_longer_useful' and 'upload' not in calls
            elif case=='changed_source':assert not calls and state=='no_receipt'
            elif case=='occupied':assert 'build' not in calls and (root/'submissions/unique-rose_v79.parquet').read_bytes()==b'pre-existing synthetic fixture'
            elif case=='audit_failure':assert 'upload' not in calls and state=='error'
            elif case=='ambiguous_upload':assert calls.count('upload')==1 and state=='error'
            before=list(calls)
            try:run.run(specpath)
            except AssertionError:pass
            else:raise AssertionError('Automatic restart was permitted')
            assert calls==before
            results[case]=dict(pass_check=True,terminal_state=state,no_automatic_retry=True)
    return dict(status='pass',synthetic_only=True,no_live_api_or_upload=True,no_real_candidate_created=True,
        checks=results,sources={str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__),REPO/'VALIDATION/joint_scored_followup.py']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=check()
    if args.report:write_json(args.report,result)
    print(json.dumps(result,indent=2))
