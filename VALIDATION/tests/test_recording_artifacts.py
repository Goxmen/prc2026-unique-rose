"""Synthetic checks for the fold-fitted recording corrections; no data files are read."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plan_window
import recording_artifacts as artifacts


def departures(*specs):
    """Readable synthetic movements; every time gap is takeoff minus that clock."""
    rows = []
    for number, supplied in enumerate(specs, 1):
        spec = dict(supplied)
        takeoff = pd.Timestamp(spec.pop('takeoff', '2025-07-02 12:00'), tz='UTC')
        stamp = lambda gap: pd.NaT if gap is None else takeoff - pd.Timedelta(seconds=gap)
        target = spec.pop('y', 1000.0)
        row = dict(
            mvt_id=number, ADEP_mvt='EGLL', FLIGHT_ID_mvt=float(number),
            FLIGHT_mvt=f'TST{number}', MVT_TIME_UTC_mvt=takeoff,
            SCHED_TIME_UTC_mvt=stamp(spec.pop('gap', 1800)),
            LOBT_flt=stamp(spec.pop('lobt_gap', 1200)),
            AOBT_3_flt=stamp(spec.pop('aobt_gap', 1200)),
            handler_arr_sched_share=0.25, handler_arr_n=30.0,
            y=target, BLOCK_TIME_UTC_mvt=stamp(target),
        )
        row.update(spec)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt',
                   'AOBT_3_flt', 'BLOCK_TIME_UTC_mvt']:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def day_row(**changes):
    return dict(dict(ADEP_mvt='LIRF', FLIGHT_ID_mvt=np.nan, gap=54000,
                     lobt_gap=None, aobt_gap=None, y=87400), **changes)


def empty_parameters():
    """A valid fold with no examples eligible for either fitted correction."""
    return artifacts.fit(departures({}), late_mean=1000)


def predictions(frame, parameters=None, control=None, probability=None, twin_mask=None):
    n = len(frame)
    return artifacts.predict(
        np.full(n, 1000.0) if control is None else control, frame,
        empty_parameters() if parameters is None else parameters,
        np.full(n, 0.5) if probability is None else probability,
        np.zeros(n, bool) if twin_mask is None else twin_mask,
    )


class IsolationTests(unittest.TestCase):
    def test_prediction_does_not_read_departure_targets_or_block_times(self):
        frame = departures(
            dict(lobt_gap=86400, aobt_gap=86400),
            day_row(),
            day_row(gap=18000),
            dict(FLIGHT_mvt='TWIN', lobt_gap=7000, aobt_gap=7000),
            dict(FLIGHT_mvt='TWIN', FLIGHT_ID_mvt=np.nan, lobt_gap=None, aobt_gap=None),
        )
        parameters = artifacts.fit(departures(day_row()), late_mean=1000)
        control = np.array([82740, 54000, 18000, 5000, 1000], float)
        probability = np.array([0.5, 0.5, 0.01, 0.5, 0.5])
        expected, masks = artifacts.predict(control, frame, parameters, probability)
        changed = frame.copy()
        changed['y'] = [1, 200000, -10000, np.nan, np.inf]
        changed['BLOCK_TIME_UTC_mvt'] = pd.Timestamp('1900-01-01', tz='UTC')
        changed['taxi_sec'] = changed.y
        for candidate in [changed, changed.drop(columns=['y', 'taxi_sec', 'BLOCK_TIME_UTC_mvt'])]:
            actual, actual_masks = artifacts.predict(control, candidate, parameters, probability)
            for column in expected:
                np.testing.assert_array_equal(actual[column], expected[column])
            for name in masks:
                np.testing.assert_array_equal(actual_masks[name], masks[name])
        self.assertEqual(list(artifacts.observable(changed)), artifacts.RAW + artifacts.SIGNALS)

    def test_fit_rejects_repeated_or_missing_movement_ids(self):
        frame = departures(day_row(), day_row(takeoff='2025-07-03 12:00'))
        for ids in [[1, 1], [1, np.nan]]:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'unique training movements'):
                artifacts.fit(frame.assign(mvt_id=ids), 1000)

    def test_fitted_parameters_do_not_use_block_stamps_or_ineligible_labels(self):
        frame = departures(dict(lobt_gap=8000, aobt_gap=8000, y=8300), day_row(), {})
        expected = artifacts.fit(frame, 1000)
        changed = frame.copy()
        changed['BLOCK_TIME_UTC_mvt'] = pd.NaT
        changed.loc[2, 'y'] = 199999
        self.assertEqual(artifacts.fit(changed, 1000), expected)
        self.assertEqual(artifacts.fit(frame.iloc[::-1], 1000), expected)

    def test_fold_fits_are_independent_and_prediction_does_not_mutate_parameters(self):
        first = departures(dict(lobt_gap=8000, aobt_gap=8000, y=8300), day_row())
        parameters = artifacts.fit(first, 1000)
        saved = copy.deepcopy(parameters)
        other_fold = departures(dict(lobt_gap=9000, aobt_gap=9000, y=6000), day_row(y=54000))
        other_parameters = artifacts.fit(other_fold, 1000)
        self.assertNotEqual(parameters, other_parameters)
        predictions(departures(day_row()), parameters)
        self.assertEqual(parameters, saved)
        self.assertEqual(artifacts.fit(first, 1000), saved)

    def test_fit_retains_the_control_label_filter_and_finite_late_mean(self):
        for target in [0, -1, 200001, np.nan, np.inf]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                artifacts.fit(departures({}).assign(y=target), 1000)
        for mean in [None, np.nan, np.inf]:
            with self.subTest(mean=mean), self.assertRaises(ValueError):
                artifacts.fit(departures({}), mean)


class ClockTests(unittest.TestCase):
    def test_clock_masks_require_id_both_clocks_and_agreement(self):
        frame = departures(
            dict(lobt_gap=43200, aobt_gap=43200),
            dict(lobt_gap=43200, aobt_gap=45000),
            dict(lobt_gap=43200, aobt_gap=45001),
            dict(lobt_gap=43200, aobt_gap=43199),
            dict(lobt_gap=None, aobt_gap=50000),
            dict(lobt_gap=50000, aobt_gap=None),
            dict(lobt_gap=50000, aobt_gap=50000, FLIGHT_ID_mvt=np.nan),
            dict(lobt_gap=7200, aobt_gap=7200),
            dict(lobt_gap=7200, aobt_gap=7199),
        )
        _, masks = predictions(frame)
        self.assertEqual(masks['clock12'].tolist(), [True, True, False, False, False, False, False, False, False])
        self.assertEqual(masks['clock2'].tolist(), [True, True, False, True, False, False, False, True, False])

    def test_clock_offsets_shrink_toward_zero_and_unknown_airport_uses_pool(self):
        training = departures(dict(lobt_gap=8000, aobt_gap=8000, y=8300))
        parameters = artifacts.fit(training, 1000)
        pooled = 300 / 26
        airport = (300 + 25 * pooled) / 26
        self.assertEqual(parameters['clock']['rows'], 1)
        self.assertAlmostEqual(parameters['clock']['pooled_residual_sec'], pooled)
        self.assertAlmostEqual(parameters['clock']['airport_residual_sec']['EGLL'], airport)
        self.assertTrue(0 < pooled < airport < 300)
        query = departures(dict(lobt_gap=50000, aobt_gap=50000),
                           dict(ADEP_mvt='EDDF', lobt_gap=50000, aobt_gap=50000))
        actual, _ = predictions(query, parameters)
        np.testing.assert_allclose(actual['pred_clock12'], [50000 + airport, 50000 + pooled])

    def test_clock_residuals_are_clipped_before_fitting(self):
        training = departures(dict(lobt_gap=8000, aobt_gap=8000, y=50000),
                              dict(lobt_gap=8000, aobt_gap=8000, y=1000))
        fitted = artifacts.fit(training, 1000)
        self.assertEqual(fitted['clock']['pooled_residual_sec'], 0)
        self.assertEqual(fitted['clock']['airport_residual_sec']['EGLL'], 0)

    def test_clock_prediction_stays_inside_lobt_window_at_either_edge(self):
        query = departures(dict(lobt_gap=50000, aobt_gap=51800),
                           dict(lobt_gap=50000, aobt_gap=48200))
        low, high = plan_window.window(query)
        for offset, expected in [(3660, high), (-3660, low)]:
            parameters = empty_parameters()
            parameters['clock']['pooled_residual_sec'] = offset
            actual, _ = predictions(query, parameters)
            np.testing.assert_array_less(low - 1, actual['pred_clock12'])
            np.testing.assert_array_less(actual['pred_clock12'], high + 1)
            edge_row = 0 if offset > 0 else 1
            self.assertEqual(actual['pred_clock12'][edge_row], expected[edge_row])


class DayTests(unittest.TestCase):
    def test_day_gate_uses_rome_dates_across_dst_and_midnight(self):
        frame = departures(
            # Spring change makes these two local midnights 23 elapsed hours apart.
            day_row(takeoff='2025-03-30 22:30', gap=23 * 3600),
            # Autumn change makes them 25 hours apart; both still cross one local date.
            day_row(takeoff='2025-10-26 23:30', gap=25 * 3600),
            # UTC dates differ, but schedule and takeoff are on the same Rome date.
            day_row(takeoff='2025-07-02 20:00', gap=22 * 3600),
            # Exactly 24 UTC hours, but two Rome dates across the spring change.
            day_row(takeoff='2025-03-30 22:30', gap=24 * 3600),
            day_row(takeoff='2025-07-02 08:00', gap=14 * 3600),
            day_row(takeoff='2025-07-02 08:00', gap=14 * 3600 - 1),
            day_row(takeoff='2025-07-02 08:00', gap=None),
            day_row(ADEP_mvt='LFPG'),
            day_row(FLIGHT_ID_mvt=99),
        )
        raw = artifacts.observable(frame)
        self.assertEqual(artifacts.day_mask(raw, artifacts.clocks(raw)).tolist(),
                         [True, True, False, False, True, False, False, False, False])

    def test_day_weight_shrinks_and_respects_zero_one_bounds(self):
        # One perfect shifted example against five prior rows gives weight 1/6.
        parameters = artifacts.fit(departures(day_row()), 1000)
        self.assertAlmostEqual(parameters['day']['pooled_weight'], 1 / 6)
        actual, _ = predictions(departures(day_row()), parameters, control=[54000])
        self.assertAlmostEqual(actual['pred_day_shrunk'][0], 54000 + (87400 - 54000) / 6)
        below = artifacts.fit(departures(day_row(y=1000)), 1000)
        self.assertEqual(below['day']['pooled_weight'], 0)
        above = artifacts.fit(departures(*(day_row(y=200000, takeoff=f'2025-07-{day:02} 12:00')
                                          for day in range(2, 12))), 1000)
        self.assertEqual(above['day']['pooled_weight'], 1)

    def test_no_eligible_training_examples_keep_the_rule(self):
        parameters = empty_parameters()
        self.assertEqual(parameters['day']['rows'], 0)
        self.assertEqual(parameters['day']['pooled_weight'], 0)
        actual, _ = predictions(departures(day_row()), parameters, control=[54000])
        self.assertEqual(actual['pred_day_shrunk'][0], 54000)
        self.assertEqual(actual['pred_day_handler'][0], 54000)

    def test_handler_support_counts_distinct_dates_not_rows(self):
        training = departures(*(day_row() for _ in range(10)))
        parameters = artifacts.fit(training, 1000)
        self.assertEqual(parameters['day']['handler_support']['low'], dict(rows=10, dates=1))
        self.assertEqual(parameters['day']['handler_weights']['low'], parameters['day']['pooled_weight'])
        self.assertEqual(parameters['day']['handler_weights']['high'], parameters['day']['pooled_weight'])

    def test_one_supported_handler_group_still_uses_pooled_weight(self):
        # Without evidence for both groups, a second shrinkage level must not
        # simply weaken the pooled regularizer by reusing the same examples.
        training = departures(*(day_row(takeoff=f'2025-07-{day:02} 12:00') for day in range(2, 7)))
        parameters = artifacts.fit(training, 1000)
        day = parameters['day']
        self.assertEqual(day['handler_support']['low']['dates'], 5)
        self.assertEqual(day['handler_support']['high']['dates'], 0)
        self.assertEqual(day['handler_weights']['low'], day['pooled_weight'])
        self.assertEqual(day['handler_weights']['high'], day['pooled_weight'])

    def test_supported_handler_groups_shrink_toward_pool_and_missing_support_falls_back(self):
        specs = [day_row(takeoff=f'2025-07-{day:02} 12:00') for day in range(2, 7)]
        specs += [day_row(takeoff=f'2025-07-{day:02} 12:00', y=54000,
                          handler_arr_sched_share=0.5) for day in range(7, 12)]
        parameters = artifacts.fit(departures(*specs), 1000)
        day = parameters['day']
        self.assertAlmostEqual(day['pooled_weight'], 1 / 3)
        self.assertAlmostEqual(day['handler_weights']['low'], 2 / 3)
        self.assertAlmostEqual(day['handler_weights']['high'], 1 / 6)
        self.assertEqual(day['handler_support']['low']['dates'], 5)
        query = departures(day_row(handler_arr_n=20),
                           day_row(handler_arr_sched_share=0.5, handler_arr_n=20),
                           day_row(handler_arr_n=19),
                           day_row(handler_arr_sched_share=np.nan),
                           day_row(handler_arr_n=np.nan))
        actual, _ = predictions(query, parameters, control=np.full(5, 54000))
        expected_weights = [2 / 3, 1 / 6, 1 / 3, 1 / 3, 1 / 3]
        np.testing.assert_allclose(actual['pred_day_handler'], 54000 + 33400 * np.array(expected_weights))


class RuleShrinkTests(unittest.TestCase):
    def test_one_real_taxi_shrinks_the_rule_weight_toward_zero(self):
        training = departures(day_row(gap=18000, y=1000))
        parameters = artifacts.fit(training, 1000)
        self.assertEqual(parameters['rule']['rows'], 1)
        self.assertEqual(parameters['rule']['dates'], 1)
        self.assertAlmostEqual(parameters['rule']['weight'], 1 / 6)
        actual, _ = predictions(training, parameters, control=[18000])
        self.assertAlmostEqual(actual['pred_rule_shrunk'][0], 18000 - 17000 / 6)

    def test_schedule_recorded_and_absent_training_cases_keep_the_hard_rule(self):
        query = departures(day_row(gap=18000))
        for training in [departures(day_row(gap=18000, y=18000)), departures({}),
                         departures(day_row(gap=18000, y=50000))]:
            with self.subTest(target=training.y.tolist()):
                parameters = artifacts.fit(training, 1000)
                self.assertEqual(parameters['rule']['weight'], 0)
                actual, _ = predictions(query, parameters, control=[18000])
                self.assertEqual(actual['pred_rule_shrunk'][0], 18000)

    def test_fit_and_prediction_share_the_narrow_midgap_population(self):
        frame = departures(
            day_row(gap=14400, y=1000), day_row(gap=43199, y=1000),
            day_row(gap=43200, y=1000), day_row(gap=14399, y=1000),
            day_row(gap=18000, y=1000, ADEP_mvt='EGLL'),
            day_row(gap=18000, y=1000, FLIGHT_ID_mvt=99), day_row(gap=None, y=1000),
            # The separate date-shift population must not influence this fit.
            day_row(),
        )
        parameters = artifacts.fit(frame, 1000)
        just_band = artifacts.fit(frame.iloc[:2].copy(), 1000)
        self.assertEqual(parameters['rule'], just_band['rule'])
        self.assertEqual(parameters['rule']['rows'], 2)
        control = np.arange(len(frame), dtype=float) + 10000
        actual, masks = predictions(frame, parameters, control)
        self.assertEqual(masks['rule_band'].tolist(), [True, True] + [False] * 6)
        np.testing.assert_array_equal(actual['pred_rule_shrunk'][2:], control[2:])

    def test_midgap_prediction_is_independent_of_classifier_probability(self):
        query = departures(day_row(gap=18000), day_row(gap=20000))
        parameters = artifacts.fit(departures(day_row(gap=18000, y=1000)), 1000)
        first, _ = predictions(query, parameters, control=[18000, 20000], probability=[0.001, 0.01])
        second, _ = predictions(query, parameters, control=[18000, 20000], probability=[0.99, np.nan])
        for column in ['pred_rule_shrunk', 'pred_combined_midgap']:
            np.testing.assert_array_equal(first[column], second[column])
        self.assertFalse(np.array_equal(first['pred_rule_confidence'], second['pred_rule_confidence']))

    def test_midgap_combination_replaces_confidence_branch_and_preserves_other_priorities(self):
        training = departures(day_row(), day_row(gap=18000, y=1000))
        parameters = artifacts.fit(training, 1000)
        query = departures(day_row(), day_row(gap=18000),
                           dict(lobt_gap=50000, aobt_gap=50000),
                           dict(lobt_gap=7000, aobt_gap=7000), {})
        control = [54000, 18000, 46340, 4000, 950]
        actual, _ = predictions(query, parameters, control, [0.5, 0.01, 0.5, 0.5, 0.5],
                                [False, False, True, True, False])
        np.testing.assert_allclose(actual['pred_combined_midgap'], [
            actual['pred_day_handler'][0], actual['pred_rule_shrunk'][1],
            actual['pred_clock12'][2], actual['pred_twins'][3], control[4],
        ])
        self.assertNotEqual(actual['pred_combined_midgap'][1], actual['pred_combined'][1])


class PredictionCompositionTests(unittest.TestCase):
    def test_high_confidence_rule_has_narrow_observable_gate(self):
        frame = departures(
            day_row(gap=14400), day_row(gap=43199), day_row(gap=43200),
            day_row(gap=14399), day_row(gap=18000), day_row(gap=18000),
            day_row(gap=18000), day_row(gap=18000),
            day_row(gap=18000, ADEP_mvt='EGLL'), day_row(gap=18000, FLIGHT_ID_mvt=1),
        )
        probability = [0, 0.049, 0.01, 0.01, 0.05, np.nan, -0.01, np.inf, 0.01, 0.01]
        control = np.arange(10) + 15000
        with np.errstate(invalid='ignore'):
            actual, masks = predictions(frame, control=control, probability=probability)
        self.assertEqual(masks['rule_confidence'].tolist(), [True, True] + [False] * 8)
        self.assertEqual(actual['pred_rule_confidence'][0], 1000)
        self.assertAlmostEqual(actual['pred_rule_confidence'][1], 0.049 * 43199 + 0.951 * 1000)
        np.testing.assert_array_equal(actual['pred_rule_confidence'][2:], control[2:])

    def test_combination_uses_each_disjoint_regime_and_clock_overrides_twin(self):
        frame = departures(day_row(), day_row(gap=18000),
                           dict(lobt_gap=50000, aobt_gap=50000),
                           dict(lobt_gap=7000, aobt_gap=7000), {})
        parameters = artifacts.fit(departures(day_row()), 1000)
        control = np.array([54000, 18000, 46340, 4000, 950], float)
        twin = [False, False, True, True, False]
        actual, masks = predictions(frame, parameters, control, [0.5, 0.01, 0.5, 0.5, 0.5], twin)
        np.testing.assert_allclose(actual['pred_combined'], [
            actual['pred_day_handler'][0], actual['pred_rule_confidence'][1],
            actual['pred_clock12'][2], actual['pred_twins'][3], control[4],
        ])
        self.assertEqual(actual['pred_twins'][2], 50000 - 330)
        self.assertEqual(actual['pred_combined'][2], 50000)
        self.assertFalse((masks['day'] & masks['rule_confidence']).any())

    def test_individual_candidates_preserve_every_unselected_prediction(self):
        frame = departures(day_row(), day_row(gap=18000),
                           dict(lobt_gap=50000, aobt_gap=50000),
                           dict(lobt_gap=8000, aobt_gap=8000), {},
                           dict(lobt_gap=None, aobt_gap=None, gap=None))
        control = np.array([54000, 18000, 47000, 9000, 1700, 1900], float)
        actual, masks = predictions(frame, artifacts.fit(departures(day_row()), 1000), control,
                                    [0.5, 0.01, 0.5, 0.5, 0.5, np.nan], [False, False, False, True, False, False])
        candidates = dict(twins='twins', clock12='clock12', clock2='clock2',
                          day_shrunk='day', day_handler='day', rule_confidence='rule_confidence')
        for name, mask in candidates.items():
            with self.subTest(name=name):
                np.testing.assert_array_equal(actual[artifacts.CANDIDATES[name]][~masks[mask]], control[~masks[mask]])
        untouched = ~(masks['twins'] | masks['clock12'] | masks['day'] | masks['rule_confidence'])
        np.testing.assert_array_equal(actual['pred_combined'][untouched], control[untouched])

    def test_actual_twin_detection_changes_only_the_id_record(self):
        frame = departures(dict(FLIGHT_mvt='PAIR', lobt_gap=7000, aobt_gap=7000),
                           dict(FLIGHT_mvt='PAIR', FLIGHT_ID_mvt=np.nan, lobt_gap=None, aobt_gap=None), {})
        control = np.array([4000, 1000, 900], float)
        actual, masks = artifacts.predict(control, frame, empty_parameters(), np.full(3, 0.5))
        self.assertEqual(masks['twins'].tolist(), [True, False, False])
        np.testing.assert_array_equal(actual['pred_twins'], [6670, 1000, 900])
        np.testing.assert_array_equal(actual['pred_combined'], [6670, 1000, 900])

    def test_rejects_misaligned_vectors_and_changed_fitted_settings(self):
        frame = departures({})
        for control, probability, twins in [([1000, 1000], [0.5], [False]),
                                            ([1000], [0.5, 0.5], [False]),
                                            ([1000], [0.5], [False, False]),
                                            ([np.nan], [0.5], [False])]:
            with self.subTest(control=control, probability=probability, twins=twins), self.assertRaises(ValueError):
                artifacts.predict(control, frame, empty_parameters(), probability, twins)
        parameters = empty_parameters()
        parameters['settings']['clock_agreement_sec'] = 0
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            predictions(frame, parameters)


if __name__ == '__main__':
    unittest.main()
