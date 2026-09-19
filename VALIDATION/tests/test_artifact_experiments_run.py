"""Synthetic checks for correction fitting, context isolation and resumable scoring."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifact_experiments_run as run


def frame():
    at = pd.Timestamp('2025-07-02 12:00', tz='UTC')
    out = pd.DataFrame({name: [None] * 4 for name in run.KEEP})
    out['mvt_id'] = [1, 2, 3, 4]
    out['ADEP_mvt'] = out['airport'] = ['LFPG', 'LFPG', 'LIRF', 'EGLL']
    out['FLIGHT_mvt'] = ['AFR100', 'AFR100', 'ITY10', 'BAW100']
    out['FLIGHT_ID_mvt'] = [np.nan, 55.0, np.nan, 77.0]
    out['MVT_TIME_UTC_mvt'] = out['mvt_time'] = [at, at + pd.Timedelta(seconds=15), at, at]
    out['SCHED_TIME_UTC_mvt'] = [at - pd.Timedelta(hours=2)] * 2 + [at - pd.Timedelta(hours=19), at - pd.Timedelta(minutes=20)]
    out['LOBT_flt'] = pd.to_datetime([None, at - pd.Timedelta(hours=2), None, at - pd.Timedelta(minutes=20)], utc=True)
    out['AOBT_3_flt'] = out.LOBT_flt
    out['y'] = [900.0, 7300.0, 87300.0, 1000.0]
    out[run.CONTROL] = [900.0, 3500.0, 68400.0, 1100.0]
    out['ordinary'] = [900.0, 3500.0, 1000.0, 1100.0]
    out['p'] = [0.0, 0.0, 0.8, 0.0]
    out['gap_sec'] = [7200., 7215., 68400., 1200.]
    out['handler_arr_sched_share'] = 0.25
    out['handler_arr_n'] = 50.
    out['has_flight_plan'] = out.FLIGHT_ID_mvt.notna()
    out['raw_flight_id_missing'] = out.FLIGHT_ID_mvt.isna()
    out['arrival_history_available'] = True
    out['arrival_history_days_available'] = 7
    out['rule_triggered'] = [False, False, True, False]
    out['late_rows'] = False
    out['seed'], out['fold'], out['month'] = 42, 'A1', '2025-07'
    out['STAND_mvt'] = ['A1', 'A2', 'B1', 'C1']
    return out


class DerivationTests(unittest.TestCase):
    def test_control_is_unchanged_and_twin_context_is_explicit(self):
        raw = frame()
        parameters = run.corrections.fit(raw, late_mean=1000.)
        mask = run.record_fixes.twin_rows(raw)
        self.assertEqual(mask.tolist(), [False, True, False, False])
        derived = run.derive(raw, parameters, mask)
        np.testing.assert_array_equal(derived[run.CONTROL], raw[run.CONTROL])
        self.assertEqual(derived.pred_twins.iloc[1], 7215. - 330.)
        self.assertEqual(derived.mask_twins.tolist(), mask.tolist())
        # The partner outside the supplied context cannot create a twin.
        restricted = raw.iloc[[1, 2, 3]].copy()
        scoped = run.derive(restricted, parameters, run.record_fixes.twin_rows(restricted))
        self.assertFalse(scoped.mask_twins.any())
        self.assertEqual(scoped.pred_twins.iloc[0], raw[run.CONTROL].iloc[1])
        changed_labels = raw.assign(y=1.)
        other = run.derive(changed_labels, parameters, mask)
        for column in run.corrections.CANDIDATES.values():
            np.testing.assert_array_equal(derived[column], other[column])

    def test_training_parameters_reject_audit_rows_outside_the_training_window(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            raw = frame()
            source = dict(training_audits={'model': dict(path='audit.parquet', rows=len(raw))},
                parameters={'model': {'late_mean': 1234.0}},
                folds={'A1': dict(model_id='model', train_start='2025-01-01', train_end='2025-06-30')})
            raw.to_parquet(directory / 'audit.parquet', index=False)
            with self.assertRaisesRegex(ValueError, 'outside its window'):
                run.fit_parameters(source, directory, 'model')
            raw.MVT_TIME_UTC_mvt -= pd.Timedelta(days=30)
            raw.to_parquet(directory / 'audit.parquet', index=False)
            with patch.object(run.corrections, 'fit', return_value={'fitted': True}) as fit:
                result = run.fit_parameters(source, directory, 'model')
            self.assertEqual(result, {'fitted': True})
            self.assertEqual(fit.call_args.kwargs['late_mean'], 1234.0)
            self.assertEqual(fit.call_args.args[0].mvt_id.tolist(), raw.mvt_id.tolist())

    def test_summary_deduplicates_affected_movements_across_seeds_and_folds(self):
        raw = frame()
        parameters = run.corrections.fit(raw, late_mean=1000.)
        derived = run.derive(raw, parameters, run.record_fixes.twin_rows(raw))
        stats, changed = run.change_stats(derived)
        self.assertEqual(stats['twins']['rows_changed'], 1)
        self.assertEqual(stats['twins']['closer'], 1)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            progress = {'predictions': {}}
            for fold in ('A1', 'B4'):
                for seed in (42, 142, 242):
                    path = f'changed_{fold}_{seed}.parquet'
                    changed.assign(fold=fold, seed=seed).to_parquet(directory / path, index=False)
                    progress['predictions'][f'{fold}__{seed}'] = dict(changes_path=path, statistics=stats)
            summary = run.summarize_changes(directory, progress)['candidates']['twins']
            self.assertEqual(summary['rows_changed'], 6)
            self.assertEqual(summary['unique_movements'], 1)
            self.assertEqual(summary['unique_utc_dates'], 1)
            self.assertEqual(summary['by_seed']['42']['rows_changed'], 2)
            self.assertEqual(summary['by_seed']['42']['unique_movements'], 1)


class ResumeTests(unittest.TestCase):
    def test_completed_comparison_is_loaded_without_recomputing(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            payload = {'report': {'scores': {'R_A': 123.}}}
            path = directory / 'comparison_twins_block1.json'
            path.write_text(json.dumps(payload))
            progress = {'comparisons': {'twins__block1': {'path': path.name}}}
            with patch.object(run.reporting, 'report', side_effect=AssertionError('must reuse comparison')):
                self.assertEqual(run.comparison({}, directory, progress, 'twins', 1), payload)

    def test_parent_fingerprint_change_and_output_tampering_reject_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / 'output.parquet'
            path.write_bytes(b'original')
            config, signature, parent = {'candidate': 'frozen'}, {'code': 'abc'}, {'parent': 'xyz'}
            record = dict(config=config, signature=signature, parent_input_fingerprints=parent,
                          runtime_identity=run.runtime_identity())
            progress = dict(predictions={'one': dict(artifacts={path.name: run.sha256(path)})},
                            comparisons={}, summaries={})
            run.verify_resume(record, progress, directory, signature, parent, config)
            with self.assertRaisesRegex(ValueError, 'parent artifacts'):
                run.verify_resume(record, progress, directory, signature, {'parent': 'changed'}, config)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed artifact'):
                run.verify_resume(record, progress, directory, signature, parent, config)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                run.verify_artifacts(directory, {'../outside': 'digest'})


if __name__ == '__main__':
    unittest.main()
