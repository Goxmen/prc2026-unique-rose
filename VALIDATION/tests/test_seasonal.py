"""Seasonal holdout isolation, retrospective history and LB comparison checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seasonal as s
import seasonal_report as sr
from records import sha256
from reporting import Loss
from VALIDATION.tests.test_recovery import movements
from VALIDATION.tests.test_v3 import config, loss


class SeasonalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        blob, _, _ = s.model_source()
        cls.backend = s.load_backend(blob, Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def frames(self):
        times = [f'{month}-15 12:00' for month in s.MONTHS]
        dep = movements(times, np.arange(12))
        arr = movements([f'{month}-15 13:00' for month in s.MONTHS], np.arange(100, 112))
        return dep, arr

    def test_split_membership_and_fit_reuse(self):
        dep, arr = self.frames()
        jobs = s.schedule('both')
        self.assertEqual([len(job['train']) for job in jobs], [10, 11, 11])
        self.assertEqual(sum(len(job['val']) for job in jobs), 4)
        for job in jobs:
            td, ta, validation = s.split_frames(dep, arr, job)
            self.assertEqual(len(td), len(job['train']))
            self.assertEqual(len(ta), len(job['train']))
            for month, (vd, va) in validation.items():
                self.assertEqual(len(vd), 1)
                self.assertEqual(len(va), 1)
                self.assertFalse(set(td.mvt_id) & set(vd.mvt_id))
                self.assertNotIn(month, td.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').tolist())
        with self.assertRaisesRegex(ValueError, 'overlap'):
            s.split_frames(dep, arr, dict(train=['2025-01'], val=['2025-01']))

    def test_heldout_labels_and_block_times_cannot_change_features(self):
        dep, arr = self.frames()
        dep['BLOCK_TIME_UTC_mvt'] = dep.MVT_TIME_UTC_mvt - pd.Timedelta(minutes=10)
        for job in s.schedule('both'):
            altered = dep.copy()
            heldout = altered.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').isin(job['val'])
            altered.loc[heldout, 'taxi_sec'] = -1e15
            altered.loc[heldout, 'BLOCK_TIME_UTC_mvt'] = pd.Timestamp('1900-01-01', tz='UTC')
            td, ta, validation = s.split_frames(dep, arr, job)
            changed_td, changed_ta, changed_validation = s.split_frames(altered, arr, job)
            pd.testing.assert_frame_equal(td, changed_td)
            for mode in ('off', 'history'):
                cfg = config(recovery=mode)
                X, y, cats = self.backend.training_matrix(td, ta, cfg)
                changed_X, changed_y, _ = self.backend.training_matrix(changed_td, changed_ta, cfg)
                pd.testing.assert_frame_equal(X, changed_X)
                np.testing.assert_array_equal(y, changed_y)
                self.assertEqual(len(X.columns), 27)
                for month, (vd, va) in validation.items():
                    V = self.backend.validation_matrix(td, ta, vd, va, s.FREEZE, cats, cfg)
                    changed_vd, changed_va = changed_validation[month]
                    other = self.backend.validation_matrix(changed_td, changed_ta, changed_vd,
                                                           changed_va, s.FREEZE, cats, cfg)
                    pd.testing.assert_frame_equal(V, other)

    def test_training_history_causal_and_validation_frozen_on_training_only(self):
        dep = movements(['2025-01-15', '2025-02-01', '2025-02-02',
                         '2025-07-15', '2025-12-15', '2025-12-16'], np.arange(6))
        dep['taxi_sec'] = [7000, 100, 300, 6000, 500, 900]
        arr = movements(['2025-01-14', '2025-02-01', '2025-07-14', '2025-12-14'], np.arange(100, 104))
        td, ta, validation = s.split_frames(dep, arr, s.schedule('10m')[0])
        X, _, cats = self.backend.training_matrix(td, ta, config())
        self.assertTrue(np.isnan(X.op_taxi_30d.iloc[0]))
        self.assertEqual(X.op_taxi_30d.iloc[1], 100)
        for vd, va in validation.values():
            V = self.backend.validation_matrix(td, ta, vd, va, s.FREEZE, cats, config())
            self.assertEqual(V.op_taxi_30d.iloc[0], 700)

    def test_missing_holdout_context_not_bridged_by_date_range(self):
        raw = movements(['2025-02-01', '2025-08-01'], [1, 2])
        matrix = pd.DataFrame({'arr_taxiin_7d': [np.nan, np.nan]})
        job = s.schedule('10m')[0]
        for i, month in enumerate(['2025-02', '2025-08']):
            frame = s.prediction_frame(raw.iloc[[i]], matrix.iloc[[i]], job, month)
            self.assertEqual(frame.arrival_history_days_available.iloc[0], 0)

    def test_completed_work_skipped_and_modified_artifact_rejected(self):
        job = s.schedule('10m')[0]
        progress = {'models': {}, 'predictions': {
            f"10m_{parent}_{month}_42": {} for parent in ('baseline', 'recovery') for month in job['val']}}
        backend = Mock()
        s.execute(None, None, [job], {'seeds': [42]}, backend, Path(self.directory.name), progress)
        backend.training_matrix.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.parquet'
            path.write_bytes(b'original')
            progress = {'models': {}, 'predictions': {'p': {'artifacts': {path.name: sha256(path)}}}}
            s.check_artifacts(Path(directory), progress)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed completed artifact'):
                s.check_artifacts(Path(directory), progress)


class SeasonalScoringTests(unittest.TestCase):
    def test_seed_loss_average_month_weights_and_noise_tolerance(self):
        first = loss('2025-01', [1], [[3], [-3]])
        second = loss('2025-07', [2, 3], [[4, 4], [-4, -4]])
        self.assertEqual(sr.rmse(first), 3)  # Averaging predictions would give zero.
        self.assertAlmostEqual(sr.combined(3, 4, .25), np.sqrt(.25*9 + .75*16))
        losses = {name: {'2025-01': first, '2025-07': second} for name in s.RECIPES}
        items = {name: {'score': 10 + i} for i, name in enumerate(s.RECIPES)}
        result = sr.evaluate(losses, items, .25, dict(bootstrap_reps=30, smoke=False, negligible_sec=1))
        scores = result['modes']['standard']['levels']['Q0']['scores']
        self.assertEqual(scores['2025-01'], 3)
        self.assertEqual(scores['2025-07'], 4)
        self.assertAlmostEqual(scores['combined'], np.sqrt(14.25))
        comparison = result['modes']['standard']['comparisons'][0]['metrics']['combined']
        self.assertEqual(comparison['ci95'], [0, 0])
        self.assertEqual(sr.agreement(-.65, .17, 1), 'LB change within tolerance')
        self.assertEqual(sr.agreement(-4, -10, 1), 'agree')
        self.assertEqual(sr.agreement(4, -10, 1), 'disagree')


if __name__ == '__main__':
    unittest.main()
