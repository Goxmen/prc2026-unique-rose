"""Offline final-uploader control tests; all bucket/transport calls are mocked.

The separately tested full parquet validator is stubbed here to isolate upload
state transitions. Real source pinning, exclusive attempt creation, selection
validation, UTC quota/reservation policy and remote hash checks remain active.
Temporary fixtures live under runs and are removed after execution. No network,
real submission file, real upload manifest, or shared policy is modified.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import audited_final_upload_20261013 as uploader

pf=uploader.preflight
OUT=ROOT/'VALIDATION/reports/final_uploader_offline_preflight_20261013.json'


def art(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not OUT.exists()
    cases=[]
    with tempfile.TemporaryDirectory(prefix='final_upload_offline_',dir=ROOT/'VALIDATION/runs') as tmp:
        folder=Path(tmp);today=datetime.now(timezone.utc).date()
        policy=folder/'policy.json'
        write(policy,dict(utc_date=today.isoformat(),daily_limit=5,reserved_slots=1))

        def scenario(name,used=0,occupied=False,selection=True,resolved=True,
                     local_error=False,bucket_error=False,put_mode='ok',race=False):
            case=folder/name;case.mkdir();payload=b'offline synthetic final bytes\n'
            output=case/'unique-rose_final.parquet';output.write_bytes(payload)
            relative=str(output.relative_to(ROOT));digest=hashlib.sha256(payload).hexdigest()
            evidence=case/'evidence.json';write(evidence,dict(status='synthetic_pass'))
            decision=case/'selection.json'
            write(decision,dict(status='selected_best_available',utc_date=today.isoformat(),
                file=relative,sha256=digest,compared_candidates=['synthetic_A','synthetic_B'],
                selection_reason='Synthetic control test',evidence=[art(evidence)],
                remaining_promising_experiments_resolved=resolved))
            sources={record['path']:record['sha256'] for record in
                     [art(Path(uploader.__file__)),art(Path(uploader.transport.__file__)),art(decision)]}
            manifest=dict(state='prepared',submission_kind='final',file=relative,sha256=digest,
                          sources=sources,authorization='Offline synthetic fixture; never upload')
            if selection:manifest['reserved_slot_selection']=art(decision)
            path=case/'manifest.json';write(path,manifest)
            calls=[];remote={};attempt=case/'final_upload_attempt.json'
            uploads=[dict(file=f'fake_v{i}.parquet',uploadedAt=today.isoformat()+'T00:00:00+00:00') for i in range(used)]
            if occupied:uploads.append(dict(file=output.name,uploadedAt=today.isoformat()+'T00:00:00+00:00'))

            def validate(given):
                assert given==path;calls.append('validate')
                if local_error:raise AssertionError('synthetic local validation failure')
                if race:write(attempt,dict(synthetic_competing_attempt=True))
                return pf.read(path),dict(file=art(output),synthetic_local_validation=True)
            def bucket(*,strict):
                assert strict;calls.append('bucket')
                if bucket_error:raise RuntimeError('synthetic live read failure')
                return uploads
            def mc(*args,binary=False):
                calls.append(args[0]);key=uploader.transport.live.BUCKET+output.name
                if args[0]=='cp':
                    assert not binary
                    assert args==('cp','--disable-multipart','--custom-header','If-None-Match:*',
                                  '--attr',f'sha256={digest}',output,key)
                    assert attempt.exists() and key not in remote
                    if put_mode=='fail_before':raise RuntimeError('synthetic PUT rejected')
                    remote[key]=payload if put_mode!='wrong_hash' else b'synthetic wrong object'
                    if put_mode=='uncertain_after':raise RuntimeError('synthetic response lost after PUT')
                    return 'synthetic atomic PUT complete'
                assert args==('cat',key) and binary
                if key not in remote:raise FileNotFoundError('synthetic remote absent')
                return remote[key]
            return dict(path=path,case=case,calls=calls,remote=remote,attempt=attempt,
                        expected_file=relative,validate=validate,bucket=bucket,mc=mc)

        def contexts(item):
            from contextlib import ExitStack
            stack=ExitStack()
            stack.enter_context(patch.object(pf,'EXPECTED_FILE',item['expected_file']))
            stack.enter_context(patch.object(pf,'validate_manifest',item['validate']))
            stack.enter_context(patch.object(pf.dashboard_data,'fetch_bucket',item['bucket']))
            stack.enter_context(patch.object(pf.submission_slot_policy,'POLICY',policy))
            stack.enter_context(patch.object(uploader.transport,'mc',item['mc']))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return stack

        def expect_failure(action):
            try:action()
            except (AssertionError,RuntimeError,FileNotFoundError,FileExistsError,KeyError) as error:
                return type(error).__name__
            raise AssertionError('Expected rejection')

        for name,kwargs in [
            ('local_validation_failure',dict(local_error=True)),
            ('live_read_failure',dict(bucket_error=True)),
            ('occupied_filename',dict(occupied=True)),
            ('five_used_slots',dict(used=5)),
            ('fifth_slot_missing_selection',dict(used=4,selection=False)),
            ('selection_required_before_fifth_slot',dict(selection=False)),
            ('unresolved_selection',dict(resolved=False)),
            ('concurrent_attempt_exclusive_create',dict(race=True)),
        ]:
            item=scenario(name,**kwargs)
            with contexts(item):error=expect_failure(lambda:uploader.upload(item['path']))
            assert item['calls'].count('cp')==0 and item['calls'].count('cat')==0
            assert item['attempt'].exists()==bool(kwargs.get('race'))
            if kwargs.get('local_error'):assert 'bucket' not in item['calls']
            cases.append(dict(name=name,status='pass',rejection=error,put_calls=0))

        for name,used in [('successful_delivery',0),('selected_fifth_slot_delivery',4)]:
            item=scenario(name,used=used)
            with contexts(item):
                uploader.upload(item['path'])
                assert pf.read(item['path'])['state']=='delivered_final'
                receipt=pf.read(item['case']/'final_delivery.json')
                assert receipt['score_acceptance_unverified'] is True and receipt['no_reupload'] is True
                if used==4:assert 'selection' in pf.read(item['attempt'])['live_slots']['slot_policy']
                expect_failure(lambda:uploader.upload(item['path']))
                uploader.verify_existing(item['path'])
            assert item['calls'].count('cp')==1 and item['calls'].count('cat')==2
            cases.append(dict(name=name,status='pass',put_calls=1,repeat_upload_rejected=True,
                              read_only_reconciliation=True,score_acceptance_unverified=True))

        for mode in ['fail_before','uncertain_after','wrong_hash']:
            item=scenario(mode,put_mode=mode)
            with contexts(item):
                expect_failure(lambda:uploader.upload(item['path']))
                assert item['attempt'].exists() and pf.read(item['path'])['state']=='prepared'
                assert not (item['case']/'final_delivery.json').exists()
                expect_failure(lambda:uploader.upload(item['path']))
                if mode=='uncertain_after':
                    uploader.verify_existing(item['path'])
                    assert pf.read(item['path'])['state']=='delivered_final'
                else:expect_failure(lambda:uploader.verify_existing(item['path']))
            assert item['calls'].count('cp')==1
            cases.append(dict(name=mode,status='pass',put_calls=1,no_put_retry=True,
                              reconciliation_succeeds_only_for_exact_object=mode=='uncertain_after'))

        item=scenario('changed_prepared_manifest',put_mode='uncertain_after')
        with contexts(item):
            expect_failure(lambda:uploader.upload(item['path']))
            value=pf.read(item['path']);value['unexpected_change']=True;write(item['path'],value)
            prior_reads=item['calls'].count('cat')
            expect_failure(lambda:uploader.verify_existing(item['path']))
            assert item['calls'].count('cat')==prior_reads and item['calls'].count('cp')==1
        cases.append(dict(name='changed_prepared_manifest',status='pass',reconciliation_rejected_before_network=True))

    report=dict(status='pass',source=art(Path(__file__)),uploader=art(Path(uploader.__file__)),
                validator=art(Path(pf.__file__)),transport=art(Path(uploader.transport.__file__)),
                slot_policy=art(Path(pf.submission_slot_policy.__file__)),cases=cases,
                full_parquet_validation='Separately tested validator; stubbed here to isolate uploader transitions',
                network_calls_mocked=True,no_network=True,no_upload=True,no_real_manifest_or_policy_modified=True,
                temporary_fixtures_removed=True)
    with OUT.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=len(cases),receipt=art(OUT))))


if __name__=='__main__':main()
