"""Isolation, resume integrity and full-year gate checks for the CV study."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import overnight_run as r
from VALIDATION.tests.test_recovery import movements


class OvernightRunnerTests(unittest.TestCase):
    def frames(self):
        times = [f'2025-{month:02}-{day:02} 12:00' for month in range(1, 13)
                 for day in (1, 2, 15, 16)]
        dep = movements(times, np.arange(len(times)))
        arr = movements(times, np.arange(1000, 1000 + len(times)))
        dep['taxi_sec'] = 600 + np.arange(len(dep))
        arr['taxi_sec'] = 400.
        for frame in (dep, arr):
            frame['raw_flight_id_missing'] = frame.FLIGHT_ID_mvt.isna()
            frame['raw_plan_block_missing'] = frame.ADEP_flt.isna()
            frame['raw_aobt3_missing'] = frame.AOBT_3_flt.isna()
        return dep, arr

    def test_schedule_has_exact_original_roster_and_seasonal_isolation(self):
        jobs = r.schedule()
        self.assertEqual(len(jobs), 9)
        self.assertEqual(sum(len(j['evals']) for j in jobs), 20)
        self.assertEqual([len(j['train']) for j in jobs], [6] * 6 + [10, 11, 11])
        folds = {e['fold'] for j in jobs if j['scheme'] == 'v3' for e in j['evals']}
        self.assertEqual(folds, {f'A{i}' for i in range(1, 7)} | {f'B{i}' for i in range(1, 11)})
        for job in jobs:
            self.assertFalse(set(job['train']) & {e['month'] for e in job['evals']})
            if job['scheme'] == 'v3':
                self.assertEqual(job['freeze'], job['end'])
                self.assertTrue(all(e['month'] > job['end'][:7] for e in job['evals']))
            else:
                self.assertEqual(job['freeze'], '2025-12-31')
        fitted, predicted = r.expected_keys(jobs, ['control', 'no_month'], [42, 142, 242])
        self.assertEqual(len(set(fitted)), 54)
        self.assertEqual(len(set(predicted)), 120)

    def test_resume_replays_missing_prediction_without_refit_and_scores_extreme_row(self):
        dep, arr = self.frames()
        job = deepcopy(r.schedule()[0])
        job['evals'] = job['evals'][:2]
        extreme = dep.MVT_TIME_UTC_mvt.dt.month.eq(7)
        dep.loc[extreme, 'taxi_sec'] = 1_000_000.
        config = dict(r.models.BASE_CONFIG, seeds=[42], rounds=2, smoke=True,
                      threads=1, prediction_threads=1)
        progress = dict(models={}, predictions={})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            r.execute(dep, arr, [job], ['control'], config, directory, progress)
            r.check_artifacts(directory, progress)
            self.assertEqual(len(progress['models']), 1)
            self.assertEqual(len(progress['predictions']), 2)
            first = r.prediction_key(job, job['evals'][0], 'control', 42)
            entry = progress['predictions'].pop(first)
            original = pd.read_parquet(directory / entry['path'])
            self.assertTrue(original.y.eq(1_000_000).all())
            self.assertEqual(len(original), 4)
            self.assertTrue(np.isfinite(original.pred).all())
            (directory / entry['path']).unlink()
            with patch.object(r.models, 'fit_model', side_effect=AssertionError('Unexpected refit')), \
                 patch.object(r, 'prepare_training', side_effect=AssertionError('Unneeded training preparation')):
                r.execute(dep, arr, [job], ['control'], config, directory, progress)
            replay = pd.read_parquet(directory / progress['predictions'][first]['path'])
            pd.testing.assert_frame_equal(original, replay)
            r.check_artifacts(directory, progress)
            ids = pd.read_parquet(directory / next(iter(progress['models'].values()))['train_ids'])
            self.assertFalse(set(ids.mvt_id) & set(original.mvt_id))
            self.assertTrue(ids.used_for_fit.all())

    def test_completed_job_does_not_prepare_data(self):
        job = r.schedule()[0]
        models, predictions = r.expected_keys([job], ['control'], [42])
        progress = dict(models={key: {} for key in models},
                        predictions={key: {} for key in predictions})
        with patch.object(r, 'split_job', side_effect=AssertionError('Completed job read data')):
            r.execute(None, None, [job], ['control'], {'seeds': [42]}, Path('/unused'), progress)

    def test_seasonal_import_retains_exact_predictions_without_fitting(self):
        dep, arr = self.frames()
        job = deepcopy(r.schedule()[6])
        cfg = dict(r.models.BASE_CONFIG, seeds=[42], rounds=1, smoke=False,
                   threads=1, prediction_threads=1)
        td, ta, validation = r.split_job(dep, arr, job)
        X, y, cats, raw = r.prepare_training(td, ta, cfg)
        booster, history = r.models.fit_model(X, y, raw, 'control', 42, rounds=1, threads=1)
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source, target = Path(source_dir), Path(target_dir)
            path = source / 'model.txt'
            booster.save_model(str(path))
            r.write_json(path.with_suffix('.loss.json'), history)
            original = dict(models={'10m_baseline_42': dict(path=path.name, rows=len(y), minutes=.1,
                artifacts=r.artifact_hashes(source, path, path.with_suffix('.loss.json')))}, predictions={})
            wanted = {}
            for ev in job['evals']:
                vd, va = validation[ev['month']]
                V = r.modeling.validation_matrix(td, ta, vd, va, r.FREEZE, cats, cfg)
                pred = r.models.predict_model(booster, V, 'control', threads=1)
                final, audit = r.apply_rule(pred, vd, 'rome_4h')
                frame = r.seasonal.prediction_frame(vd, V, job, ev['month'])
                frame['parent_pred'], frame['Q0-4h'] = pred, final
                frame['Q0-4h_triggered'] = audit.rule_triggered.to_numpy()
                output = source / (ev['month'] + '.parquet')
                r.save_frame(output, frame)
                original['predictions'][f"10m_baseline_{ev['month']}_42"] = dict(
                    path=output.name, artifacts=r.artifact_hashes(source, output))
                wanted[ev['month']] = final
            r.write_json(source / 'progress.json', original)
            progress = dict(models={}, predictions={})
            with patch.object(r, 'SEASONAL_CONTROL', source), \
                 patch.object(r.models, 'fit_model', side_effect=AssertionError('Import refitted model')):
                r.import_seasonal_control(dep, arr, [job], cfg, target, progress, {'directory': 'test_archive'})
            self.assertTrue(r.validation_complete(progress, [job], ['control'], [42]))
            r.check_artifacts(target, progress)
            model = next(iter(progress['models'].values()))
            self.assertEqual(model['imported_from']['model_key'], '10m_baseline_42')
            self.assertEqual(model['params']['num_leaves'], 255)
            for entry in progress['predictions'].values():
                frame = pd.read_parquet(target / entry['path'])
                np.testing.assert_array_equal(frame.pred, wanted[entry['month']])

    def test_changed_completed_artifact_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'model.txt'
            path.write_text('initial')
            progress = dict(models={'m': {'artifacts': {'model.txt': r.sha256(path)}}}, predictions={})
            r.check_artifacts(directory, progress)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed completed artifact'):
                r.check_artifacts(directory, progress)
            progress['models']['m']['artifacts'] = {'../outside.txt': 'irrelevant'}
            with self.assertRaisesRegex(ValueError, 'escapes'):
                r.check_artifacts(directory, progress)

    def test_refit_requires_full_validation_and_rejects_smoke(self):
        cfg = dict(r.models.BASE_CONFIG, smoke=True)
        with self.assertRaisesRegex(ValueError, 'Smoke validation'):
            r.execute_refit(None, None, None, None, ['control'], cfg, None, None, None)
        cfg['smoke'] = False
        with self.assertRaisesRegex(ValueError, 'Complete all validation'):
            r.execute_refit(None, None, None, None, ['control'], cfg, None,
                            dict(models={}, predictions={}), dict(jobs=r.schedule()))

    def test_source_snapshot_and_fingerprint_reject_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            signature = {'VALIDATION/overnight_models.py': r.sha256(r.ROOT / 'VALIDATION/overnight_models.py')}
            r.snapshot_sources(directory, signature)
            frozen = directory / 'source/VALIDATION/overnight_models.py'
            frozen.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Changed frozen source'):
                r.snapshot_sources(directory, signature)
            before = r.signature_digest({'config': {'rounds': 800}, 'source': signature})
            after = r.signature_digest({'config': {'rounds': 801}, 'source': signature})
            self.assertNotEqual(before, after)


if __name__ == '__main__':
    unittest.main()
