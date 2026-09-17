"""Recovery safety: hidden data, collisions, raw precedence and fold isolation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modeling
from recovery import PLAN_FIELDS, public, recover_exact, recover_frame
from recovery_history import impute_static
from VALIDATION.tests.test_v3 import departures, config


def movements(times, ids, *, missing=False):
    df = departures(times)
    df['mvt_id'] = ids
    df['FLIGHT_ID_mvt'] = np.asarray(ids, dtype=float) + 100
    df['FLIGHT_mvt'] = 'ABC123'
    df['AIRCRAFT_TYPE_mvt'] = 'A320'
    for field in PLAN_FIELDS:
        if field not in df:
            df[field] = df.MVT_TIME_UTC_mvt if field.startswith('ARVT') else 'x'
    df['ADEP_flt'], df['ADES_flt'] = 'LIRF', 'EGLL'
    if missing:
        df['FLIGHT_ID_mvt'] = np.nan
        for field in PLAN_FIELDS:
            df.loc[:, field] = pd.NaT if pd.api.types.is_datetime64_any_dtype(df[field]) else None
    return df


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.dep = movements(['2025-06-30 22:00'], [1], missing=True)
        self.arr = movements(['2025-06-30 23:00'], [2])

    def test_exact_and_raw_precedence(self):
        self.dep['MARKET_SEGMENT_flt'] = 'keep'
        out, audit = recover_frame(self.dep, self.arr)
        self.assertEqual(audit.stage.tolist(), ['exact'])
        self.assertEqual(out.MARKET_SEGMENT_flt.iloc[0], 'keep')
        self.assertEqual(out.AOBT_3_flt.iloc[0], self.arr.AOBT_3_flt.iloc[0])
        untouched = [c for c in self.dep if c not in PLAN_FIELDS]
        pd.testing.assert_frame_equal(out[untouched], self.dep[untouched])
        self.assertTrue(audit.anchor_recovered.iloc[0])
        self.assertTrue(audit.raw_plan_block_missing.iloc[0])

    def test_hidden_data_are_removed_before_matching(self):
        expected, audit = recover_frame(self.dep, self.arr)
        changed = self.dep.copy()
        changed['taxi_sec'] = -1e20
        changed['BLOCK_TIME_UTC_mvt'] = pd.Timestamp('2030-01-01', tz='UTC')
        actual, other = recover_frame(changed, self.arr)
        pd.testing.assert_frame_equal(expected[PLAN_FIELDS], actual[PLAN_FIELDS])
        pd.testing.assert_frame_equal(audit, other)
        with self.assertRaisesRegex(ValueError, 'observable'):
            recover_exact(changed, self.arr)

    def test_ambiguity_collisions_and_owned_plans(self):
        duplicate = pd.concat([self.arr, self.arr.assign(mvt_id=3)], ignore_index=True)
        _, audit = recover_frame(self.dep, duplicate)
        self.assertEqual(audit.reason.iloc[0], 'ambiguous')
        dep = pd.concat([self.dep, self.dep.assign(mvt_id=4)], ignore_index=True)
        _, audit = recover_frame(dep, self.arr)
        self.assertEqual(set(audit.reason), {'donor_collision'})
        owner = movements(['2025-06-30 21:50'], [5])
        owner['FLIGHT_ID_mvt'] = self.arr.FLIGHT_ID_mvt.iloc[0]
        _, audit = recover_frame(self.dep, self.arr, pd.concat([self.dep, owner]))
        self.assertEqual(audit.reason.iloc[0], 'plan_already_owned')

    def test_origin_destination_aircraft_and_existing_id_guards(self):
        for col, value, reason in [('ADEP_flt', 'LFPG', 'incompatible_route'),
                                    ('ADES_flt', 'LFPG', 'incompatible_route'),
                                    ('AIRCRAFT_TYPE_mvt', 'B738', 'incompatible_aircraft')]:
            arr = self.arr.copy()
            arr[col] = value
            _, audit = recover_frame(self.dep, arr)
            self.assertEqual(audit.reason.iloc[0], reason)
        dep = self.dep.assign(FLIGHT_ID_mvt=999.)
        _, audit = recover_frame(dep, self.arr)
        self.assertEqual(audit.reason.iloc[0], 'incompatible_plan_id')
        dep['FLIGHT_ID_mvt'] = self.arr.FLIGHT_ID_mvt.iloc[0]
        _, audit = recover_frame(dep, self.arr)
        self.assertEqual(audit.stage.iloc[0], 'exact')
        self.assertFalse(audit.raw_flight_id_missing.iloc[0])
        self.assertTrue(audit.raw_plan_block_missing.iloc[0])

    def test_timing_boundaries_and_missing_true_twin(self):
        for delta, recovered in [(0, True), (8*3600, True), (-1, False), (8*3600+1, False)]:
            arr = self.arr.copy()
            arr['MVT_TIME_UTC_mvt'] = self.dep.MVT_TIME_UTC_mvt.iloc[0] + pd.Timedelta(seconds=delta)
            _, audit = recover_frame(self.dep, arr)
            self.assertEqual(audit.stage.iloc[0] == 'exact', recovered)
        out, audit = recover_frame(self.dep, self.arr.iloc[:0])
        pd.testing.assert_frame_equal(out, self.dep)
        self.assertEqual(audit.reason.iloc[0], 'no_candidate')

    def test_evaluation_arrivals_cannot_change_training(self):
        # The June departure's twin lies in July and is forbidden for fitting.
        arr = self.arr.copy()
        arr['MVT_TIME_UTC_mvt'] = pd.to_datetime(['2025-07-01 00:00'], utc=True)
        train_arr = modeling.month_rows(arr, '2025-06')
        cfg = config(recovery='exact')
        audits = []
        X, _, cats = modeling.training_matrix(self.dep, train_arr, cfg, audit_out=audits)
        self.assertTrue(X.mvt_minus_aobt3_sec.isna().all())
        val = movements(['2025-07-01 00:30'], [10], missing=True)
        _ = modeling.validation_matrix(self.dep, train_arr, val, arr, '2025-06-30', cats, cfg)
        again, _, _ = modeling.training_matrix(self.dep, train_arr, cfg)
        pd.testing.assert_frame_equal(X, again)

    def test_off_path_ignores_recovery_only_fields(self):
        dep = self.dep.copy()
        arr = self.arr.copy()
        X, y, cats = modeling.training_matrix(dep, arr, config())
        basic = dep[['mvt_id', 'taxi_sec'] + modeling.RAW_NAMES]
        expected, targets, categories = modeling.training_matrix(basic, arr, config())
        pd.testing.assert_frame_equal(X, expected)
        np.testing.assert_array_equal(y, targets)
        self.assertEqual(set(cats), set(categories))

    def test_relaxed_a_cross_field_identifier(self):
        arr = self.arr.assign(FLIGHT_mvt='different', CALLSIGN_flt='ABC123')
        _, audit = recover_frame(self.dep, arr, mode='relaxed')
        self.assertEqual(audit.stage.iloc[0], 'relaxed_A')

    def test_relaxed_b_learns_prefix_only_from_reference(self):
        ref = movements(pd.date_range('2025-01-01', periods=30, freq='D'), range(100, 130))
        ref['FLIGHT_mvt'], ref['CALLSIGN_flt'] = 'AB001', 'ABC1'
        dep = self.dep.assign(FLIGHT_mvt='AB0099')
        arr = self.arr.assign(FLIGHT_mvt='ABC99', CALLSIGN_flt='ABC99')
        _, audit = recover_frame(dep, arr, mode='relaxed', reference_dep=ref,
                                  reference_arr=arr.iloc[:0])
        self.assertEqual(audit.stage.iloc[0], 'relaxed_B')
        _, empty = recover_frame(dep, arr, mode='relaxed', reference_dep=ref.iloc[:0],
                                  reference_arr=arr.iloc[:0])
        self.assertEqual(empty.stage.iloc[0], 'unrecovered')

    def test_relaxed_c_requires_training_route_support(self):
        ref = movements(pd.date_range('2025-01-01', periods=20, freq='D'), range(100, 120))
        refarr = ref.copy()
        refarr['mvt_id'] += 1000
        refarr['MVT_TIME_UTC_mvt'] += pd.Timedelta(hours=1)
        arr = self.arr.assign(FLIGHT_mvt='unrelated', CALLSIGN_flt='unrelated')
        _, audit = recover_frame(self.dep, arr, mode='relaxed', reference_dep=ref, reference_arr=refarr)
        self.assertEqual(audit.stage.iloc[0], 'relaxed_C')
        _, empty = recover_frame(self.dep, arr, mode='relaxed', reference_dep=ref.iloc[:19], reference_arr=refarr)
        self.assertEqual(empty.stage.iloc[0], 'unrecovered')

    def test_history_uses_only_earlier_days_for_training(self):
        hist = movements(['2025-06-28 10:00', '2025-06-29 10:00', '2025-06-30 10:00'], [10, 11, 12])
        _, audit = recover_frame(self.dep, self.arr.iloc[:0])
        filled, result = impute_static(public(self.dep), public(hist), audit, causal=True)
        self.assertTrue(filled.AIRCRAFT_OPERATOR_flt.isna().all())  # two earlier days: below support=3
        hist.loc[2, 'MVT_TIME_UTC_mvt'] = pd.Timestamp('2025-06-29 12:00', tz='UTC')
        filled, result = impute_static(public(self.dep), public(hist), audit, causal=True)
        self.assertEqual(filled.AIRCRAFT_OPERATOR_flt.iloc[0], 'x')
        self.assertTrue(result.static_imputed.iloc[0])
        self.assertTrue(filled.AOBT_3_flt.isna().all())
        self.assertTrue(filled.FLIGHT_ID_mvt.isna().all())
        self.assertTrue(filled.ADEP_flt.isna().all())

    def test_history_raw_precedence_and_excluded_flight_type_prefix(self):
        hist = movements(pd.date_range('2025-01-01', periods=12, freq='D'), range(100, 112))
        hist['FLIGHT_mvt'] = 'ABC999'
        dep = self.dep.assign(MARKET_SEGMENT_flt='raw')
        _, audit = recover_frame(dep, self.arr.iloc[:0])
        filled, audit = impute_static(public(dep), public(hist), audit, causal=False)
        self.assertEqual(filled.MARKET_SEGMENT_flt.iloc[0], 'raw')
        self.assertEqual(filled.AIRCRAFT_OPERATOR_flt.iloc[0], 'x')
        self.assertTrue(filled.FLIGHT_TYPE_flt.isna().all())

    def test_duplicate_plan_claims_with_different_arrivals_are_rejected(self):
        dep = pd.concat([self.dep, self.dep.assign(mvt_id=4, FLIGHT_mvt='ABC999')], ignore_index=True)
        arr = pd.concat([self.arr, self.arr.assign(mvt_id=5, FLIGHT_mvt='ABC999')], ignore_index=True)
        _, audit = recover_frame(dep, arr)
        self.assertEqual(set(audit.reason), {'donor_collision'})

    def test_recovery_reuses_training_without_changing_validation_values(self):
        train = movements(['2025-06-01 12:00'], [10], missing=True)
        train_arr = movements(['2025-06-01 13:00'], [11])
        val = movements(['2025-07-01 12:00'], [20], missing=True)
        val_arr = movements(['2025-07-01 13:00'], [21])
        cfg = config(recovery='exact')
        prepared = []
        _, _, cats = modeling.training_matrix(train, train_arr, cfg, prepared_out=prepared)
        fresh = modeling.validation_matrix(train, train_arr, val, val_arr, '2025-06-30', cats, cfg)
        reused = modeling.validation_matrix(train, train_arr, val, val_arr, '2025-06-30', cats, cfg,
                                            prepared_training=prepared[0])
        pd.testing.assert_frame_equal(fresh, reused)


if __name__ == '__main__':
    unittest.main()
