"""Remote results must match their completed job before a dependent audit proceeds."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remote_component_return as module


class RemoteReturnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'local'; self.root.mkdir()
        self.remote = Path(self.temp.name) / 'remote'; self.remote.mkdir()
        self.patcher = patch.object(module, 'ROOT', self.root)
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        module.write_json(self.root / 'run/identity.json', {'recipe': 'fixed'})
        identity = dict(path='run/identity.json', sha256=module.sha256(self.root / 'run/identity.json'))
        (self.remote / 'run').mkdir()
        (self.remote / 'run/model.txt').write_bytes(b'Frozen model bytes')
        (self.remote / 'run/pred.parquet').write_bytes(b'Frozen prediction bytes')
        module.write_json(self.remote / 'run/input_audit.json', dict(status='pass'))
        def artifact(name):
            return dict(path=name, sha256=module.sha256(self.remote / name))
        model = artifact('run/model.txt')
        module.write_json(self.remote / 'run/model.json', dict(identity=identity, model=model,
            host='GLAPTOP', rounds=10000, input_audit=artifact('run/input_audit.json')))
        module.write_json(self.remote / 'run/pred.json', dict(identity=identity, model=model,
            prediction=artifact('run/pred.parquet'), all_ranking_reload_exact=True, rows=344841))
        self.queue = dict(pid=71, state='complete', jobs=[dict(
            completed_result_sha256=module.sha256(self.remote / 'run/pred.json'))])
        self.plan = dict(source_sha256=module.sha256(module.__file__), host='glaptop-wsl',
            identity=identity, remote_queue='run/queue.json', remote_pid=71, remote_start='101',
            receipt='run/collection.json', record='run/pred.json', model_record='run/model.json',
            files=['run/model.txt', 'run/model.json', 'run/pred.parquet', 'run/pred.json', 'run/input_audit.json'])
        self.path = self.root / 'run/plan.json'; module.write_json(self.path, self.plan)

    def transport(self, argv, **kwargs):
        if argv[0] == 'ssh':
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(dict(queue=self.queue, process_start=None)))
        self.assertEqual(argv[0], 'rsync')
        for name in self.plan['files']:
            shutil.copyfile(self.remote / name, self.root / name)
        return subprocess.CompletedProcess(argv, 0)

    def test_completed_job_is_verified_and_wait_accepts_it(self):
        with patch.object(module.subprocess, 'run', side_effect=self.transport):
            module.collect(self.path)
        self.assertEqual(module.read(self.root / self.plan['receipt'])['state'], 'complete')
        module.wait(self.path)

    def test_corrupted_transfer_never_releases_waiter(self):
        def corrupt(argv, **kwargs):
            result = self.transport(argv, **kwargs)
            if argv[0] == 'rsync':
                (self.root / 'run/model.txt').write_bytes(b'corrupted')
            return result
        with patch.object(module.subprocess, 'run', side_effect=corrupt):
            with self.assertRaises(AssertionError):
                module.collect(self.path)
        self.assertEqual(module.read(self.root / self.plan['receipt'])['state'], 'error')
        with self.assertRaises(AssertionError):
            module.wait(self.path)

    def test_missing_live_remote_handle_is_not_completion(self):
        self.queue['state'] = 'running'
        with patch.object(module.subprocess, 'run', side_effect=self.transport):
            with self.assertRaisesRegex(AssertionError, 'ended without completion'):
                module.collect(self.path)
        self.assertFalse((self.root / 'run/model.txt').exists())

    def test_returned_artifacts_are_checked_again_before_audit(self):
        with patch.object(module.subprocess, 'run', side_effect=self.transport):
            module.collect(self.path)
        (self.root / 'run/pred.parquet').write_bytes(b'later corruption')
        with self.assertRaises(AssertionError):
            module.wait(self.path)


if __name__ == '__main__':
    unittest.main()
