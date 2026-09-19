"""Replay integrity checks without loading competition data or training boosters."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifact_control_run as replay


def sample():
    t = pd.Timestamp('2025-07-02 12:00', tz='UTC')
    raw = pd.DataFrame({name: [None] * 4 for name in replay.RAW_AUDIT}, index=[2, 4, 6, 8])
    raw['mvt_id'] = [100, 101, 102, 103]
    raw['taxi_sec'] = [20000.0, 1000.0, 10000.0, 900.0]
    raw['ADEP_mvt'] = ['LIRF', 'EDDF', 'LFPG', 'EGLL']
    raw['FLIGHT_ID_mvt'] = [np.nan, np.nan, 11.0, 12.0]
    raw['MVT_TIME_UTC_mvt'] = t
    raw['SCHED_TIME_UTC_mvt'] = [t - pd.Timedelta(seconds=s) for s in [20000, 7000, 10000, 900]]
    raw['LOBT_flt'] = pd.to_datetime([None, None, t - pd.Timedelta(seconds=10000), t - pd.Timedelta(seconds=900)], utc=True)
    raw['AOBT_3_flt'] = raw.LOBT_flt
    for name in ['IOBT_flt', 'EOBT_1_flt']:
        raw[name] = pd.to_datetime(raw[name], utc=True)
    raw['raw_aobt3_missing'] = raw.AOBT_3_flt.isna()
    raw['raw_flight_id_missing'] = raw.FLIGHT_ID_mvt.isna()
    raw['raw_plan_block_missing'] = raw.FLIGHT_ID_mvt.isna()
    V = pd.DataFrame(0.0, index=raw.index, columns=replay.FEATURE_AUDIT)
    V['mvt_minus_sched_sec'] = [20000, 7000, 10000, 900]
    V['mvt_minus_lobt_sec'] = [np.nan, np.nan, 10000, 900]
    V['mvt_minus_aobt3_sec'] = [np.nan, np.nan, 10000, 900]
    V['arr_taxiin_7d'] = 300.0
    job = dict(id='v3_2025-06', scheme='v3', start='2025-01-01', end='2025-06-30',
               train=['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06'])
    ev = dict(fold='A1', month='2025-07')
    return raw, V, job, ev


class PredictionTests(unittest.TestCase):
    def test_replay_has_correct_rule_window_and_cap_order_without_label_inputs(self):
        raw, V, job, ev = sample()
        output = dict(final=np.array([20000., 6000., 1000., 900.]),
                      ordinary=np.array([900., 6000., 1000., 900.]), p=np.array([1., 0., 0., 0.]),
                      blended=np.array([20000., 6000., 1000., 900.]),
                      rule_triggered=np.array([True, False, False, False]), late_rows=np.zeros(4, bool))
        with patch.object(replay.signals, 'predict', return_value=output) as predict:
            frame = replay.prediction_frame(raw, V, job, ev, 42, 'ordinary', 'classifier',
                                            {'late_mean': 1100.0}, {'prediction_threads': 20})
            np.testing.assert_array_equal(frame.pred_noid_cap, [20000., 3400., 6340., 900.])
            self.assertEqual(predict.call_args.kwargs['extra'], replay.ORDINARY)
            self.assertEqual(predict.call_args.kwargs['classifier_extra'], replay.CLASSIFIER)
            self.assertEqual(predict.call_args.kwargs['threads'], 20)
            changed = raw.assign(taxi_sec=-99999., BLOCK_TIME_UTC_mvt=pd.Timestamp('1900-01-01'))
            other = replay.prediction_frame(changed, V, job, ev, 42, 'ordinary', 'classifier',
                                            {'late_mean': 1100.0}, {'prediction_threads': 20})
        np.testing.assert_array_equal(other.pred_noid_cap, frame.pred_noid_cap)
        np.testing.assert_array_equal(frame.mvt_id, raw.mvt_id)
        self.assertFalse(any('BLOCK_TIME' in c for c in frame.columns))
        self.assertEqual(frame.seed.unique().tolist(), [42])

    def test_sparse_training_audit_uses_only_requested_rows_and_preserves_order(self):
        raw, V, _, _ = sample()
        frame = replay.training_audit(raw, V, raw.taxi_sec.to_numpy())
        self.assertEqual(frame.mvt_id.tolist(), [100, 102])
        self.assertEqual(frame.y.tolist(), [20000., 10000.])
        self.assertEqual(frame.raw_flight_id_missing.tolist(), [True, False])
        V.loc[8, 'mvt_minus_aobt3_sec'] = 3600.0
        self.assertEqual(replay.training_audit(raw, V, raw.taxi_sec.to_numpy()).mvt_id.tolist(), [100, 102, 103])
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            replay.training_audit(raw, V.iloc[::-1], raw.taxi_sec.to_numpy())


class IntegrityTests(unittest.TestCase):
    def test_resume_rejects_changed_source_data_and_saved_training_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / 'audit.parquet'
            path.write_bytes(b'original')
            config, signature, data = {'rounds': 800}, {'runner': 'abc'}, [{'file': 'x', 'sha256': 'def'}]
            record = dict(config=config, signature=signature, data_identity=data)
            progress = dict(models={}, predictions={}, training_audits={
                'window': dict(artifacts={'audit.parquet': replay.sha256(path)})})
            replay.check_resume(record, progress, directory, config, signature, data)
            with self.assertRaisesRegex(ValueError, 'source or configuration'):
                replay.check_resume(record, progress, directory, {'rounds': 801}, signature, data)
            with self.assertRaisesRegex(ValueError, 'input data'):
                replay.check_resume(record, progress, directory, config, signature, [])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Missing/changed'):
                replay.check_resume(record, progress, directory, config, signature, data)

    def test_default_is_all_folds_three_seeds_and_only_selected_boosters(self):
        args = replay.parse_args([])
        historical = replay.historical_recipe()
        config = replay.configuration(args, historical)
        self.assertEqual(config['seeds'], [42, 142, 242])
        self.assertEqual(config['rounds'], 800)
        self.assertEqual(config['threads'], 6)
        self.assertEqual(config['prediction_threads'], 20)
        self.assertEqual(config['fit_workers'], 1)
        self.assertEqual(config['ordinaries'], {'stand': list(replay.ORDINARY)})
        self.assertEqual(config['classifiers'], {'stand': list(replay.CLASSIFIER)})
        jobs = [job for job in replay.overnight_run.schedule() if job['scheme'] == 'v3']
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(len(job['evals']) for job in jobs), 16)
        self.assertEqual(config['late'], True)
        self.assertEqual(config['dayshift'], False)


if __name__ == '__main__':
    unittest.main()
