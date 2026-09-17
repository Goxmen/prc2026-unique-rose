"""Five-file limits, immutable artifacts, score seals and exact template filling."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import overnight_submit as s
import overnight_models as models
from records import sha256, write_json

REAL_VALIDATE_RUN = s.validate_run


class FakeMinio:
    def __init__(self):
        self.objects, self.modified, self.calls = {}, {}, []

    def add(self, name, content, modified='2026-09-17T18:00:00Z'):
        self.objects[name], self.modified[name] = content, modified

    def __call__(self, *args, binary=False):
        self.calls.append((args, binary))
        if args[0] == 'ls':
            return '\n'.join(json.dumps(dict(key=name, type='file', status='success',
                                             lastModified=self.modified[name])) for name in self.objects)
        if args[0] == 'cat':
            value = self.objects[str(args[1]).removeprefix(s.BUCKET)]
            return value if binary else value.decode()
        if args[0] == 'cp':
            name = str(args[-1]).removeprefix(s.BUCKET)
            if name in self.objects:
                raise RuntimeError('Conditional PUT rejected a pre-existing object')
            if 'If-None-Match:*' not in args or '--disable-multipart' not in args:
                raise AssertionError('Upload must never overwrite a racing writer')
            self.add(name, Path(args[-2]).read_bytes())
            return ''
        raise AssertionError(f'Unexpected external request: {args}')


class OvernightSubmitTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.run = self.root / 'VALIDATION/runs/overnight_test'
        self.run.mkdir(parents=True)
        (self.root / 'data').mkdir()
        (self.root / 'submissions').mkdir()
        (self.root / 'model.py').write_text('frozen source\n')
        self.template = pa.table({'MVT_ID_mvt': pa.array([3., 1., 2.]),
                                  'TAXITIME_SEC_mvt': pa.array([None]*3, type=pa.int32())})
        pq.write_table(self.template, self.root / 'data/submitting.parquet')
        self.record = dict(signature={'source': {'model.py': sha256(self.root / 'model.py')}})
        self.progress = dict(refits={}, refit_models={})
        for recipe in s.SELECTED:
            path = self.run / f'{recipe}.parquet'
            pd.DataFrame(dict(mvt_id=[1., 2., 3.], pred_raw=[10.4, 20.6, 30.],
                              pred=[10.4, 20.6, 30.], rule_triggered=[False]*3)).to_parquet(path, index=False)
            self.progress['refits'][recipe] = {'path': path.name}
            self.progress['refit_models'][f'refit_2025__{recipe}__42'] = {'recipe': recipe}
        write_json(self.run / 'result.json', self.record)
        write_json(self.run / 'progress.json', self.progress)
        self.remote = FakeMinio()
        self.stack.enter_context(patch.object(s, 'ROOT', self.root))
        self.stack.enter_context(patch.object(s, 'EXPECTED_ROWS', 3))
        self.stack.enter_context(patch.object(s, 'PREPARATION_SOURCES', ['model.py']))
        self.stack.enter_context(patch.object(s, 'now', return_value='2026-09-17T20:00:00+00:00'))
        self.run_guard = self.stack.enter_context(patch.object(s, 'validate_run', return_value=(self.record, self.progress)))
        self.stack.enter_context(patch.object(s, 'mc', side_effect=self.remote))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def prepare(self):
        return s.prepare(self.run)

    def prepare_uploaded(self):
        self.prepare()
        self.assertTrue(s.upload(self.run))
        return s.read_json(self.run / s.MANIFEST)

    def add_results(self, batch):
        for number, item in enumerate(batch['submissions']):
            result = dict(status='Succeeded', file=item['file'], bucket='prc-2026-unique-rose',
                          used_pairs=3, score=340. + number)
            self.remote.add(item['file'] + '_result.json', json.dumps(result).encode())

    def frozen_selection(self):
        cv = self.run / 'cv_scores.json'
        write_json(cv, {'levels': {'placeholder': 1}})
        discovery = sha256(self.run / 'discovery_scores.json')
        frozen = dict(status='frozen', created_at='2026-09-17T20:00:00+00:00',
                      selected_validation='month_balanced', expected_confirmation=s.CONFIRMATION,
                      discovery_scores_sha256=discovery,
                      input_hashes=dict(cv_scores_sha256=sha256(cv), discovery_scores_sha256=discovery),
                      source_hashes={'model.py': sha256(self.root / 'model.py')})
        write_json(self.run / 'cv_selection_frozen.json', frozen)
        return frozen

    def test_prepare_preserves_template_and_reserves_past_all_known_versions(self):
        reports = self.root / 'VALIDATION/reports/sweeps'
        reports.mkdir(parents=True)
        write_json(reports / 'old_reservation.json', {'submissions': [{'file': 'unique-rose_v21.parquet'}]})
        self.remote.add('unique-rose_v16.parquet_result.json', b'{}', '2026-09-16T18:00:00Z')
        (self.root / 'submissions/unique-rose_v19.meta.json').write_text('{}')
        batch = self.prepare()
        self.assertEqual([item['version'] for item in batch['submissions']], [22, 23, 24, 25, 26])
        first = pq.read_table(self.root / 'submissions' / batch['submissions'][0]['file'])
        self.assertTrue(first['MVT_ID_mvt'].equals(self.template['MVT_ID_mvt']))
        self.assertEqual(first['TAXITIME_SEC_mvt'].to_pylist(), [30, 10, 21])
        self.assertEqual(batch, self.prepare())
        self.assertFalse(any(call[0][0] == 'cp' for call in self.remote.calls))

    def test_upload_only_five_and_resume_does_not_upload_again(self):
        batch = self.prepare_uploaded()
        self.assertEqual(len([c for c in self.remote.calls if c[0][0] == 'cp']), 5)
        self.assertTrue(all(item['upload_intents'] and item['remote_sha256'] == item['sha256']
                            for item in batch['submissions']))
        self.assertTrue(s.upload(self.run))
        self.assertEqual(len([c for c in self.remote.calls if c[0][0] == 'cp']), 5)

    def test_interrupted_upload_restores_remote_timestamp_without_rewriting_known_time(self):
        batch = self.prepare()
        for index, item in enumerate(batch['submissions']):
            content = (self.root / 'submissions' / item['file']).read_bytes()
            self.remote.add(item['file'], content, '2026-09-17T20:10:12.345+02:00')
            if index == 1:
                item['uploaded_at'] = '2026-09-17T18:09:59+00:00'
        write_json(self.run / s.MANIFEST, batch)
        self.assertTrue(s.upload(self.run))
        actual = s.read_json(self.run / s.MANIFEST)['submissions']
        self.assertEqual(actual[0]['uploaded_at'], '2026-09-17T18:10:12+00:00')
        self.assertEqual(actual[0]['uploaded_at_source'], 'remote_last_modified')
        self.assertEqual(actual[0]['remote_last_modified'], '2026-09-17T20:10:12.345+02:00')
        self.assertEqual(actual[1]['uploaded_at'], '2026-09-17T18:09:59+00:00')
        self.assertNotIn('uploaded_at_source', actual[1])
        self.assertTrue(all(item['remote_sha256'] == item['sha256'] for item in actual))
        self.assertFalse(any(c[0][0] == 'cp' for c in self.remote.calls))

    def test_quota_checked_before_each_upload_and_no_wait(self):
        self.prepare()
        for i in range(4):
            self.remote.add(f'other_v{i}.parquet', b'existing')
        self.assertFalse(s.upload(self.run))
        self.assertEqual(len([c for c in self.remote.calls if c[0][0] == 'cp']), 1)
        batch = s.read_json(self.run / s.MANIFEST)
        self.assertEqual([len(check['existing_uploads_today']) for check in batch['quota_checks']], [4, 5])
        self.assertIn('no automatic quota reset', batch['blocked'])

    def test_remote_collision_and_artifact_source_changes_rejected(self):
        batch = self.prepare()
        self.remote.add(batch['submissions'][0]['file'], b'unrelated existing file')
        with self.assertRaisesRegex(ValueError, 'Remote version collision'):
            s.upload(self.run)
        self.assertFalse(any(c[0][0] == 'cp' for c in self.remote.calls))
        (self.root / 'model.py').write_text('changed source')
        with self.assertRaisesRegex(ValueError, 'fingerprinted file'):
            s.upload(self.run)

    def test_discovery_never_reads_confirmation_and_confirmation_requires_freeze(self):
        batch = self.prepare_uploaded()
        self.add_results(batch)
        self.assertTrue(s.refresh(self.run, 'discovery'))
        scores = s.read_json(self.run / 'discovery_scores.json')
        self.assertEqual(set(scores), set(s.DISCOVERY))
        result_reads = [str(call[0][1]) for call in self.remote.calls
                        if call[0][0] == 'cat' and str(call[0][1]).endswith('_result.json')]
        self.assertEqual(len(result_reads), 3)
        confirmation_names = [item['file'] for item in batch['submissions'] if item['role'] == 'confirmation']
        self.assertFalse(any(name in key for name in confirmation_names for key in result_reads))
        previous = len(self.remote.calls)
        with self.assertRaisesRegex(ValueError, 'Freeze CV selection'):
            s.refresh(self.run, 'confirmation')
        self.assertEqual(len(self.remote.calls), previous)
        self.frozen_selection()
        self.assertTrue(s.refresh(self.run, 'confirmation'))
        revealed = s.read_json(self.run / 'confirmation_scores.json')
        self.assertEqual(set(revealed), set(s.CONFIRMATION))
        self.assertTrue(s.read_json(self.run / s.MANIFEST)['confirmation_seal_sha256'])
        frozen = s.read_json(self.run / 'cv_selection_frozen.json')
        frozen['selected_validation'] = 'changed_after_reveal'
        write_json(self.run / 'cv_selection_frozen.json', frozen)
        with self.assertRaisesRegex(ValueError, 'changed after confirmation'):
            s.refresh(self.run, 'confirmation')

    def test_pending_scores_remain_pending_and_bad_used_pairs_rejected(self):
        batch = self.prepare_uploaded()
        self.assertFalse(s.refresh(self.run, 'discovery'))
        self.assertEqual(s.read_json(self.run / 'discovery_scores.json'), {})
        self.add_results(batch)
        key = batch['submissions'][0]['file'] + '_result.json'
        result = json.loads(self.remote.objects[key])
        result['used_pairs'] = 2
        self.remote.objects[key] = json.dumps(result).encode()
        with self.assertRaisesRegex(ValueError, 'used_pairs'):
            s.refresh(self.run, 'discovery')
        self.assertIsNone(s.read_json(self.run / s.MANIFEST)['submissions'][0]['score'])

    def test_frozen_cv_inputs_cannot_change_before_confirmation(self):
        batch = self.prepare_uploaded()
        self.add_results(batch)
        s.refresh(self.run, 'discovery')
        self.frozen_selection()
        write_json(self.run / 'cv_scores.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'CV/discovery inputs changed'):
            s.refresh(self.run, 'confirmation')

    def test_template_guard_rejects_reordering_wrong_type_and_nonfinite(self):
        table = s.fill_template(self.template, [1., 2., 3.], [100., 200., 300.])
        reordered = table.take(pa.array([1, 0, 2]))
        with self.assertRaisesRegex(ValueError, 'IDs/order'):
            s.validate_submission(reordered, self.template)
        wrong = table.set_column(1, 'TAXITIME_SEC_mvt', pa.array([1., 2., 3.]))
        with self.assertRaisesRegex(ValueError, 'schema'):
            s.validate_submission(wrong, self.template)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            s.fill_template(self.template, [1., 2., 3.], [100., np.inf, 300.])

    def test_complete_refit_provenance_required_and_changes_rejected(self):
        source = {'model.py': sha256(self.root / 'model.py')}
        (self.run / 'source').mkdir()
        (self.run / 'source/model.py').write_bytes((self.root / 'model.py').read_bytes())
        filenames = [f'training_2025-{i:02}-01.parquet' for i in range(1, 13)] + ['ranking.parquet']
        data = []
        for name in filenames:
            path = self.root / 'data' / name
            path.write_bytes(b'fixture data')
            data.append(dict(file=name, sha256=sha256(path)))
        config = dict(models.BASE_CONFIG, sample=1., smoke=False, recipes=list(models.RECIPES))
        jobs = s.runner.schedule()
        signature = dict(protocol=s.runner.PROTOCOL, config=config, jobs=jobs,
                         source=source, data_identity=data)
        record = dict(status='complete', config=config, signature=signature, jobs=jobs)
        progress = dict(models={}, predictions={}, refits={}, refit_models={})
        for recipe in s.SELECTED:
            prediction = self.run / f'{recipe}.parquet'
            model_path = self.run / f'{recipe}.model.txt'
            model_path.write_text('fixture model')
            model = dict(recipe=recipe, seed=42, path=model_path.name,
                         artifacts={model_path.name: sha256(model_path)})
            progress['refit_models'][f'refit_2025__{recipe}__42'] = model
            meta_path = self.run / f'{recipe}.meta.json'
            meta = dict(recipe=recipe, seed=42, rounds=800, freeze='2025-12-31', source_validated=True,
                        signature_sha256=s.runner.signature_digest(signature),
                        validation_jobs=[job['id'] for job in jobs], validation_seeds=[42, 142, 242],
                        config=config, model=model, ranking_rows=3, ranking_months=['2026-01', '2026-07'],
                        prediction_rule='rome_4h', predictions_sha256=sha256(prediction),
                        model_source=source, data_identity=data)
            write_json(meta_path, meta)
            progress['refits'][recipe] = dict(recipe=recipe, rows=3, path=prediction.name,
                metadata=meta_path.name, artifacts={prediction.name: sha256(prediction), meta_path.name: sha256(meta_path)})
        write_json(self.run / 'result.json', record)
        write_json(self.run / 'progress.json', progress)
        with patch.object(s.runner, 'validation_complete', return_value=True), \
                patch.object(s.runner, 'source_signature', return_value=source):
            actual, _ = REAL_VALIDATE_RUN(self.run, all_artifacts=False)
            self.assertEqual(actual['status'], 'complete')
            original = (self.root / 'data/ranking.parquet').read_bytes()
            (self.root / 'data/ranking.parquet').write_bytes(b'changed ranking')
            with self.assertRaisesRegex(ValueError, 'fingerprinted file'):
                REAL_VALIDATE_RUN(self.run, all_artifacts=False)
            (self.root / 'data/ranking.parquet').write_bytes(original)
            meta = s.read_json(self.run / 'no_month.meta.json')
            meta['seed'] = 142
            write_json(self.run / 'no_month.meta.json', meta)
            progress['refits']['no_month']['artifacts']['no_month.meta.json'] = sha256(self.run / 'no_month.meta.json')
            write_json(self.run / 'progress.json', progress)
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                REAL_VALIDATE_RUN(self.run, all_artifacts=False)
            record['config']['smoke'] = True
            write_json(self.run / 'result.json', record)
            with self.assertRaisesRegex(ValueError, 'nonsmoke'):
                REAL_VALIDATE_RUN(self.run, all_artifacts=False)


if __name__ == '__main__':
    unittest.main()
