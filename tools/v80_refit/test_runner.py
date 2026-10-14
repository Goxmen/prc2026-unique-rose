"""Offline failure-path tests. No competition/GitHub calls or real model fits."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).parent))
import upload
import host
import run as runner

class UploadContracts(unittest.TestCase):
    def test_prefix_stat_requires_exact_filename(self):
        first=dict(name=upload.NAME,status='success',etag='a')
        second=dict(name=upload.NAME+'_result.json',status='success',etag='b')
        data='\n'.join(map(json.dumps,[first,second])).encode()
        self.assertEqual(upload.parse_stat(data,upload.NAME),first)
        with self.assertRaises(ValueError):
            upload.parse_stat(json.dumps(second).encode(),upload.NAME)

    def test_conditional_overwrite_has_exact_target(self):
        args=upload.put_arguments(Path('/tmp/candidate.parquet'),'abc123','fed')
        self.assertIn('If-Match:"abc123"',args)
        self.assertIn('--disable-multipart',args)
        self.assertEqual(args[-1],upload.BUCKET+upload.NAME)
        with self.assertRaises(ValueError):
            upload.put_arguments(Path('x'),'abc\nOther:header','d')

    def test_stale_success_is_not_new_acceptance(self):
        result=dict(status='Succeeded',file=upload.NAME,bucket='prc-2026-unique-rose',used_pairs=670790,
                    inputs={'test_set_file':'final_submitting.parquet'})
        attempt={'started_at':'2026-10-13T20:00:00+00:00'}
        self.assertFalse(upload.fresh_acceptance({'lastModified':'2026-10-13T12:35:00+02:00'},result,attempt))
        self.assertTrue(upload.fresh_acceptance({'lastModified':'2026-10-13T20:00:01Z'},result,attempt))
        for change in [{'used_pairs':344841},{'file':'unique-rose_v80.parquet'}, {'status':'Failed'}]:
            self.assertFalse(upload.fresh_acceptance({'lastModified':'2026-10-13T20:00:01Z'},dict(result,**change),attempt))

    def test_quota_utc_reset_and_last_slot(self):
        current=datetime(2026,10,14,0,10,tzinfo=timezone.utc)
        rows=[dict(uploadedAt='2026-10-14T01:59:00+02:00')]*5
        self.assertEqual(upload.check_quota(current,rows)['observed_uploaded_objects'],0)
        today=[dict(uploadedAt='2026-10-14T02:01:00+02:00')]
        self.assertEqual(upload.check_quota(current,today*4)['observed_uploaded_objects'],4)
        with self.assertRaises(ValueError):
            upload.check_quota(current,today*5)

    def test_uncertain_put_is_never_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'delivery').mkdir()
            candidate=dict(path='candidate.parquet',sha256='new',bytes=5)
            (root/'delivery/attempt.json').write_text(json.dumps(dict(candidate=candidate,started_at='2026-10-13T20:00:00Z')))
            with patch.object(upload,'validate',return_value={'candidate':candidate}), \
                 patch.object(upload,'verify',return_value=root/'candidate.parquet'), \
                 patch.object(upload,'fetch',return_value={'sha256':'old'}), patch.object(upload,'mc') as mc:
                with self.assertRaisesRegex(RuntimeError,'NOT be retried'):
                    upload.deliver(root,wait_seconds=0)
                mc.assert_not_called()

    def test_different_candidate_cannot_reuse_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'delivery').mkdir()
            (root/'delivery/attempt.json').write_text(json.dumps(dict(candidate={'sha256':'other'})))
            with patch.object(upload,'validate',return_value={'candidate':{'sha256':'new'}}), \
                 patch.object(upload,'verify',return_value=root/'candidate.parquet'), patch.object(upload,'mc') as mc:
                with self.assertRaisesRegex(ValueError,'different candidate'):
                    upload.deliver(root,wait_seconds=0)
                mc.assert_not_called()

    def test_deadline_blocks_before_remote_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(upload,'validate',return_value={'candidate':{'sha256':'new'}}), \
                 patch.object(upload,'verify',return_value=root/'candidate.parquet'), \
                 patch.object(upload,'now',return_value=upload.DEADLINE), patch.object(upload,'mc') as mc:
                with self.assertRaisesRegex(ValueError,'deadline'):
                    upload.deliver(root,wait_seconds=0)
                mc.assert_not_called()

class RunnerContracts(unittest.TestCase):
    def test_completed_exports_resume_without_remote_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(runner.completed_components(root))
            rec = dict(path='saved', sha256='good')
            for family in ('trees', 'neural'):
                runner.save(root/family/'result.json', dict(status='pass', identity=rec,
                    predictions={'old':rec, 'final':rec}, models=[rec], artifacts=[rec]))
            with patch.object(runner, 'verify') as verify, \
                 patch.object(runner, 'remote_python', side_effect=AssertionError('no remote access')):
                self.assertTrue(runner.completed_components(root))
                self.assertEqual(verify.call_count, 10)
            with patch.object(runner, 'verify', side_effect=ValueError('corrupt saved output')):
                with self.assertRaisesRegex(ValueError, 'corrupt'):
                    runner.completed_components(root)

    def test_pid_reuse_is_not_our_worker(self):
        with patch.object(host,'process_start',return_value='222'):
            self.assertFalse(host.alive({'pid':123,'process_start':'111'}))
            self.assertTrue(host.alive({'pid':123,'process_start':'222'}))
            self.assertFalse(host.alive({}))

    def test_resume_ignores_readiness_but_not_recipe_changes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'source_paths',return_value=[]):
            root=Path(temp)
            schedule=dict(required_files=[],threads=4,stages={'assemble':{'inputs_ready':{'trees':False,'neural':False}}})
            before=runner.freeze(root,schedule)
            schedule['stages']['assemble']['inputs_ready']={'trees':True,'neural':True}
            self.assertEqual(runner.freeze(root,schedule),before)
            schedule['threads']=8
            with self.assertRaisesRegex(ValueError,'identity changed'):
                runner.freeze(root,schedule)

    def test_failure_stops_other_owned_worker(self):
        calls=[]
        def status(command,out,stage,threads,remote=False):
            calls.append((command,stage))
            return {'status':'failed','alive':False} if stage=='trees' else {'status':'running','alive':True}
        with patch.object(runner,'host_call',side_effect=status):
            with self.assertRaisesRegex(RuntimeError,'trees stopped'):
                runner.wait_workers([('trees',False),('neural',True)],runner.DEFAULT,4)
        self.assertIn(('stop','neural'),calls)

if __name__=='__main__':
    unittest.main()
