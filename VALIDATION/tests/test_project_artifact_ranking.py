"""Synthetic ranking projection/scenario checks; no competition data is loaded."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'tools')]
import project_artifact_ranking as project


def departures():
    at = pd.Timestamp('2026-07-02 12:00', tz='UTC')
    raw = pd.DataFrame({name: [None] * 3 for name in project.DEPARTURE_COLUMNS})
    raw['MVT_ID_mvt'] = [1, 2, 3]
    raw['ADEP_mvt'] = ['LFPG', 'LIRF', 'EGLL']
    raw['ADES_mvt'] = ['EGLL', 'LFPG', 'LFPG']
    raw['FLIGHT_mvt'] = ['AFR100', 'ITY10', 'BAW100']
    raw['FLIGHT_ID_mvt'] = [10., np.nan, 20.]
    raw['MVT_TIME_UTC_mvt'] = at
    raw['SCHED_TIME_UTC_mvt'] = [at - pd.Timedelta(hours=14), at - pd.Timedelta(hours=19), at - pd.Timedelta(minutes=20)]
    raw['LOBT_flt'] = pd.to_datetime([at - pd.Timedelta(hours=14), None, at - pd.Timedelta(minutes=20)], utc=True)
    raw['AOBT_3_flt'] = raw.LOBT_flt
    raw['STAND_mvt'] = ['A1', 'B1', 'C1']
    raw['AIRCRAFT_TYPE_mvt'] = 'A320'
    return raw


def parameters(weight):
    return dict(settings=project.corrections.SETTINGS.copy(), late_mean=1000.,
        clock=dict(pooled_residual_sec=-100., airport_residual_sec={'LFPG': -120.}),
        day=dict(pooled_weight=weight, handler_weights={'low': weight, 'high': weight}),
        rule=dict(weight=0.1))


class ProjectionTests(unittest.TestCase):
    def test_departure_poison_labels_and_blocks_never_enter_projection(self):
        raw = departures()
        raw['PHASE_mvt'] = 'DEP'
        raw['TAXITIME_SEC_mvt'] = 9999999.
        raw['BLOCK_TIME_UTC_mvt'] = pd.Timestamp('1900-01-01', tz='UTC')
        arrival = raw.iloc[[0]].copy()
        arrival['PHASE_mvt'] = 'ARR'
        arrival['MVT_ID_mvt'] = 4
        arrival['TAXITIME_SEC_mvt'] = 600.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ranking.parquet'
            pd.concat([raw, arrival]).to_parquet(path, index=False)
            dep, arr = project.projected_ranking(path)
        self.assertEqual(dep.mvt_id.tolist(), [1, 2, 3])
        self.assertFalse(project.FORBIDDEN & set(dep))
        self.assertEqual(arr.taxi_sec.tolist(), [600.])

    def test_control_alignment_is_by_id_and_rejects_missing_or_duplicate_ids(self):
        raw = departures().rename(columns={'MVT_ID_mvt': 'mvt_id'})
        submitted = pd.DataFrame({'MVT_ID_mvt': [3, 1, 2], 'TAXITIME_SEC_mvt': [1100., 47000., 68400.]})
        aligned, control = project.align_control(raw, submitted, expected_rows=3)
        self.assertEqual(aligned.mvt_id.tolist(), [3, 1, 2])
        np.testing.assert_array_equal(control, [1100., 47000., 68400.])
        with self.assertRaisesRegex(ValueError, 'IDs differ'):
            project.align_control(raw, submitted.assign(MVT_ID_mvt=[3, 1, 4]), expected_rows=3)
        with self.assertRaisesRegex(ValueError, 'unique movement'):
            project.align_control(raw, submitted.assign(MVT_ID_mvt=[3, 1, 1]), expected_rows=3)

    def test_handler_rates_use_each_months_own_arrivals_only(self):
        january = pd.Timestamp('2026-01-02 12:00', tz='UTC')
        july = pd.Timestamp('2026-07-02 12:00', tz='UTC')
        raw = pd.DataFrame({'ADEP_mvt': ['LIRF', 'LIRF'], 'FLIGHT_mvt': ['ITY10', 'ITY10'],
                            'MVT_TIME_UTC_mvt': [january, july]})
        arrivals = pd.DataFrame({'ADES_mvt': ['LIRF'] * 3, 'FLIGHT_mvt': ['ITY11'] * 3,
            'MVT_TIME_UTC_mvt': [january, july, pd.Timestamp('2025-07-02', tz='UTC')],
            'SCHED_TIME_UTC_mvt': [january, july, pd.Timestamp('2025-07-02', tz='UTC')],
            'taxi_sec': [0., 600., 0.]})
        out = project.add_handler_signals(raw, arrivals)
        self.assertEqual(out.handler_arr_sched_share.tolist(), [1., 0.])
        self.assertEqual(out.handler_arr_n.tolist(), [1., 1.])
        with self.assertRaisesRegex(ValueError, 'Departure labels'):
            project.add_handler_signals(raw.assign(y=1.), arrivals)


class SensitivityTests(unittest.TestCase):
    def test_missing_probabilities_equal_any_probability_for_the_selected_recipes(self):
        raw = departures().rename(columns={'MVT_ID_mvt': 'mvt_id'})
        midgap = raw.iloc[[1]].copy()
        midgap['mvt_id'] = 4
        midgap['SCHED_TIME_UTC_mvt'] = midgap.MVT_TIME_UTC_mvt - pd.Timedelta(hours=6)
        raw = pd.concat([raw, midgap], ignore_index=True)
        raw['handler_arr_sched_share'], raw['handler_arr_n'] = 0.25, 50.
        control = np.array([47000., 68400., 1100., 21600.])
        twin = np.zeros(len(raw), bool)
        missing, _ = project.independent_predictions(control, raw, parameters(0.3), twin)
        with_probability, _ = project.corrections.predict(control, raw, parameters(0.3),
            probability=np.full(len(raw), 0.01), twin_mask=twin)
        self.assertNotEqual(with_probability['pred_rule_confidence'][-1], control[-1])
        for name in project.CANDIDATES:
            np.testing.assert_array_equal(missing[name], with_probability[project.corrections.CANDIDATES[name]])

    def test_six_windows_are_separate_and_selected_candidates_need_no_probability(self):
        raw = departures().rename(columns={'MVT_ID_mvt': 'mvt_id'})
        raw['handler_arr_sched_share'], raw['handler_arr_n'] = 0.25, 50.
        control = np.array([47000., 68400., 1100.])
        sets = {f'v3_2025-{month:02}': parameters((month - 5) / 10) for month in range(6, 12)}
        result = project.sensitivity(raw, control, sets, scenarios=True, total_rows=3)
        self.assertEqual(set(result), set(project.CANDIDATES))
        self.assertNotIn('rule_confidence', result)
        self.assertNotIn('combined', result)
        self.assertEqual(len(result['day_shrunk']['by_window']), 6)
        self.assertEqual(result['day_shrunk']['window_ranges']['changed_rows'], [1, 1])
        self.assertEqual(result['clock12']['window_ranges']['eligible_rows'], [1, 1])
        self.assertEqual(result['day_shrunk']['top_affected_examples'][0]['mvt_id'], 2)
        self.assertEqual(result['day_shrunk']['top_affected_examples'][0]['candidate_sec_range'], [70300., 79800.])

    def test_conditional_rmse_arithmetic_is_exact_and_marks_impossible_total(self):
        result = project.conditional_score_change(np.array([3., 100.]), np.array([2., 500.]),
            np.array([1., 5000.]), np.array([True, False]), score=10., total_rows=2)
        self.assertEqual(result['delta_sse'], -3.)
        self.assertAlmostEqual(result['delta_rmse_sec'], np.sqrt(100. - 1.5) - 10.)
        impossible = project.conditional_score_change(np.array([100.]), np.array([0.]),
            np.array([0.]), np.array([True]), score=1., total_rows=1)
        self.assertFalse(impossible['compatible_with_total_sse'])
        self.assertIsNone(impossible['delta_rmse_sec'])

    def test_positive_new_radicand_does_not_make_negative_unaffected_sse_feasible(self):
        result = project.conditional_score_change(np.array([100.]), np.array([120.]),
            np.array([0.]), np.array([True]), score=1., total_rows=1)
        self.assertGreater(1. + result['delta_sse'], 0.)
        self.assertEqual(result['assumed_control_sse'], 10000.)
        self.assertEqual(result['historical_total_sse'], 1.)
        self.assertEqual(result['implied_unaffected_sse'], -9999.)
        self.assertFalse(result['compatible_with_total_sse'])
        self.assertIsNone(result['delta_rmse_sec'])

    def test_plan_window_bound_matches_both_endpoints(self):
        raw = departures().rename(columns={'MVT_ID_mvt': 'mvt_id'})
        old, new, selected = np.array([47000., 68400., 1100.]), np.array([50280., 68400., 1100.]), np.array([True, False, False])
        bound = project.plan_window_change_bounds(old, new, raw, selected, score=1000., total_rows=100)
        low, high = project.plan_window.window(raw)
        endpoints = [project.conditional_score_change(old, new, y, selected, score=1000., total_rows=100)['delta_rmse_sec'] for y in (low, high)]
        self.assertAlmostEqual(bound['minimum_delta_rmse_sec'], min(endpoints))
        self.assertAlmostEqual(bound['maximum_delta_rmse_sec'], max(endpoints))

    def test_plan_window_endpoints_exceeding_known_total_are_unavailable(self):
        raw = departures().rename(columns={'MVT_ID_mvt': 'mvt_id'})
        old = np.array([47000., 68400., 1100.])
        new = np.array([50280., 68400., 1100.])
        selected = np.array([True, False, False])
        bound = project.plan_window_change_bounds(old, new, raw, selected, score=1., total_rows=1)
        self.assertIsNone(bound['minimum_delta_rmse_sec'])
        self.assertIsNone(bound['maximum_delta_rmse_sec'])
        self.assertFalse(bound['minimum_endpoint_compatible_with_total_sse'])
        self.assertFalse(bound['maximum_endpoint_compatible_with_total_sse'])


if __name__ == '__main__':
    unittest.main()
