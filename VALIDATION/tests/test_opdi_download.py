"""Download failures cannot silently become missing data or completed files."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import download_opdi_cache as fetch


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for key, value in [('ROOT', self.root), ('CACHE', self.root / 'data'), ('LOGS', self.root / 'logs')]:
            self.enterContext(patch.object(fetch, key, value))
        fetch.LOGS.mkdir()

    def test_success_is_validated_before_atomic_rename(self):
        def serve(command, **kwargs):
            self.assertIn('--http1.1', command)
            path = Path(command[command.index('--output') + 1])
            pq.write_table(pa.table({'id': [1], 'flt_id': ['T1'], 'adep': ['EHAM'],
                                     'first_seen': [1], 'last_seen': [2]}), path)
            return subprocess.CompletedProcess(command, 0, stdout='200')
        with patch.object(fetch.subprocess, 'run', side_effect=serve):
            result = fetch.download(dict(kind='flight_list', name='test.parquet'))
        self.assertEqual(result['state'], 'downloaded')
        self.assertTrue((fetch.CACHE / 'flight_list/test.parquet').is_file())
        self.assertFalse((fetch.CACHE / 'flight_list/test.parquet.part').exists())
        self.assertEqual(result['rows'], 1)

    def test_interrupted_transfer_is_never_marked_complete(self):
        with patch.object(fetch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 18, stdout='200')):
            result = fetch.download(dict(kind='flight_list', name='test.parquet'))
        self.assertEqual(result['state'], 'failed')
        self.assertFalse((fetch.CACHE / 'flight_list/test.parquet').exists())

    def test_range_rejection_restarts_only_the_partial_file(self):
        calls = []
        def serve(command, **kwargs):
            calls.append(list(command))
            path = Path(command[command.index('--output') + 1])
            if len(calls) == 1:
                path.write_bytes(b'partial')
                return subprocess.CompletedProcess(command, 0, stdout='416')
            self.assertNotIn('--continue-at', command)
            self.assertFalse(path.exists())
            pq.write_table(pa.table({'id': [1], 'flt_id': ['T1'], 'adep': ['EHAM'],
                                     'first_seen': [1], 'last_seen': [2]}), path)
            return subprocess.CompletedProcess(command, 0, stdout='200')
        with patch.object(fetch.subprocess, 'run', side_effect=serve):
            result = fetch.download(dict(kind='flight_list', name='test.parquet'))
        self.assertEqual(result['state'], 'downloaded')
        self.assertTrue(result['restart_without_range'])
        self.assertEqual(result['discarded_partial_bytes'], 7)
        self.assertEqual(len(calls), 2)

    def test_only_documented_404_is_an_allowed_gap(self):
        with patch.object(fetch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 22, stdout='404')):
            known = fetch.download(dict(kind='flight_events', name=next(iter(fetch.KNOWN_MISSING))))
            unknown = fetch.download(dict(kind='flight_events', name='unexpected.parquet'))
        self.assertEqual(known['state'], 'known_missing_404')
        self.assertEqual(unknown['state'], 'failed')

    def test_transient_error_is_not_mislabeled_as_known_gap(self):
        with patch.object(fetch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 22, stdout='503')):
            result = fetch.download(dict(kind='flight_events', name=next(iter(fetch.KNOWN_MISSING))))
        self.assertEqual(result['state'], 'failed')

    def test_bad_existing_file_is_not_trusted(self):
        directory = fetch.CACHE / 'flight_list'
        directory.mkdir(parents=True)
        (directory / 'test.parquet').write_bytes(b'partial file')
        with patch.object(fetch.subprocess, 'run') as request:
            result = fetch.download(dict(kind='flight_list', name='test.parquet'))
        request.assert_not_called()
        self.assertEqual(result['state'], 'failed')


if __name__ == '__main__':
    unittest.main()
