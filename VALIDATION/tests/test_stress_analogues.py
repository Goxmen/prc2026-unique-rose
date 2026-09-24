"""Reject superficially close but unsupported stress cases; selection ignores error."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stress_analogues as match


def frame(ids):
    n = len(ids)
    return pd.DataFrame(dict(mvt_id=ids, airport=['EHAM'] * n, raw_flight_id_missing=False,
        gap_band=3, mvt_time=pd.to_datetime(['2025-01-01 10:00'] * n, utc=True),
        dis_snow_6h_fraction=.5, dis_snow_24h_fraction=.3,
        dis_arr_60m_count=10., dis_arr_180m_count=30.,
        dis_arr_60m_mean_sec=1800., dis_arr_180m_mean_sec=1800.,
        dis_arr_60m_over30_share=.3, dis_arr_180m_over30_share=.3,
        dis_arr_60m_over60_share=.1, dis_arr_180m_over60_share=.1,
        stress_arr_60m_uncapped_mean_sec=1800., stress_arr_180m_uncapped_mean_sec=1800.,
        stress_arr_60m_over120_share=0., stress_arr_180m_over120_share=0.))


class StressAnalogueTests(unittest.TestCase):
    def test_exact_condition_matches_without_outcomes(self):
        pairs, states = match.match_targets(frame([1, 2]), frame([3]), 'snow_and_congestion')
        self.assertEqual(set(pairs.source_id), {1, 2})
        self.assertEqual(states.reason.tolist(), ['matched'])

    def test_nearest_is_rejected_above_observed_snow(self):
        target = frame([3]); target.dis_snow_24h_fraction = .31
        pairs, states = match.match_targets(frame([1]), target, 'prolonged_snow')
        self.assertTrue(pairs.empty)
        self.assertEqual(states.reason.tolist(), ['beyond_observed_severity'])

    def test_joint_mismatch_is_rejected_inside_marginal_ranges(self):
        source = frame([1, 2]); source.loc[0, 'dis_snow_6h_fraction'] = .1
        source.loc[1, 'dis_arr_60m_mean_sec'] = 1000.
        pairs, states = match.match_targets(source, frame([3]), 'prolonged_snow')
        self.assertTrue(pairs.empty)
        self.assertEqual(states.reason.tolist(), ['no_match_within_limits'])

    def test_wrong_airport_id_or_clock_class_cannot_match(self):
        for column, value in [('airport', 'EDDF'), ('raw_flight_id_missing', True), ('gap_band', 8)]:
            source = frame([1]); source[column] = value
            pairs, _ = match.match_targets(source, frame([3]), 'prolonged_snow')
            self.assertTrue(pairs.empty)

    def test_missing_measurements_not_silently_imputed(self):
        target = frame([3]); target.dis_arr_60m_mean_sec = np.nan
        pairs, states = match.match_targets(frame([1]), target, 'prolonged_snow')
        self.assertTrue(pairs.empty)
        self.assertEqual(states.reason.tolist(), ['missing_measurements'])

    def test_label_changes_cannot_affect_matches(self):
        source, target = frame([1, 2]), frame([3])
        source['y'], source['pred_disruption'] = [1., 99999.], [8000., 20.]
        first, _ = match.match_targets(source, target, 'prolonged_snow')
        source['y'], source['pred_disruption'] = [99999., 1.], [20., 8000.]
        second, _ = match.match_targets(source, target, 'prolonged_snow')
        pd.testing.assert_frame_equal(first, second)

    def test_sparse_arrival_window_is_not_heavy_group(self):
        source = frame([1]); source.dis_arr_60m_count = 4; source.dis_arr_180m_count = 9
        self.assertFalse(match.cohort(source, 'heavy_arrival_congestion').any())

    def test_capped_inputs_cannot_disguise_larger_uncapped_target(self):
        target = frame([3]); target.stress_arr_60m_uncapped_mean_sec = 2400.
        pairs, status = match.match_targets(frame([1]), target, 'heavy_arrival_congestion')
        self.assertTrue(pairs.empty)
        self.assertEqual(status.reason.tolist(), ['beyond_observed_severity'])

    def test_float32_rounding_does_not_reject_exact_one_hour_snow_difference(self):
        source, target = frame([1]), frame([3])
        source.dis_snow_24h_fraction = np.float32(16/48)
        target.dis_snow_24h_fraction = np.float32(14/48)
        pairs, status = match.match_targets(source, target, 'prolonged_snow')
        self.assertEqual(len(pairs), 1)
        self.assertEqual(status.reason.tolist(), ['matched'])

    def test_uncapped_check_uses_only_completed_arrivals(self):
        raw = pd.DataFrame({'ADEP_mvt': ['EHAM'], 'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-01-01 12:00'], utc=True)})
        # First completes at 11:30; second completes exactly at takeoff and is unavailable.
        arrivals = pd.DataFrame({'mvt_id': [1, 2], 'ADES_mvt': ['EHAM', 'EHAM'], 'taxi_sec': [10800., 1800.],
            'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-01-01 08:30', '2025-01-01 11:30'], utc=True)})
        actual = match.arrival_tail_checks(raw, arrivals)
        self.assertEqual(actual.stress_arr_60m_uncapped_mean_sec.iloc[0], 10800.)
        self.assertEqual(actual.stress_arr_180m_over120_share.iloc[0], 1.)


class StressScoringTests(unittest.TestCase):
    def test_metric_averages_squared_losses_before_taking_root(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        import score_stress_analogues as score
        sample = frame([1, 2]).set_index('mvt_id')
        sample['retrospective'] = True
        for seed, square in zip(score.SEEDS, [0., 4., 16.]):
            sample[f'old_sq_{seed}'] = square
            sample[f'new_sq_{seed}'] = 9.
        result = score.paired_score(sample)
        self.assertAlmostEqual(result['control'], np.sqrt(20 / 3))
        self.assertEqual(result['candidate'], 3.)
        self.assertAlmostEqual(result['difference'], 3 - np.sqrt(20 / 3))
        self.assertEqual(result['evidence'], 'too_few_flights_or_dates')

    def test_examples_are_not_chosen_for_improvement(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        import score_stress_analogues as score
        view = frame([1, 2]).set_index('mvt_id')
        view['flight'], view['retrospective'], view['gap_sec'], view['y'] = 'TEST', False, 1000., 100.
        view['pred_control_mean'], view['pred_disruption_mean'] = 100., [10000., 100.]
        for seed in score.SEEDS:
            view[f'old_sq_{seed}'], view[f'new_sq_{seed}'] = 0., [9900.**2, 0.]
        target = frame([3]); target['gap_sec'] = 1000.
        pairs = pd.DataFrame(dict(target_id=[3, 3], source_id=[1, 2], distance=[0., .1], cohort='prolonged_snow'))
        chosen = score.choose_examples(pairs, view, target, maximum=1)
        self.assertEqual(chosen.source_id.tolist(), [1])
        self.assertGreater(chosen.mse_change.iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
