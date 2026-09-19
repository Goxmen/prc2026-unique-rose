"""Check the selected submission's wiring at the recording-regime boundaries."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import artifact_submit as submit
import recording_artifacts as artifacts
from test_recording_artifacts import departures, day_row


class SelectedRecipeTests(unittest.TestCase):
    def setUp(self):
        self.parameters = artifacts.fit(departures(day_row()), late_mean=1000)

    def apply(self, raw, base, probability=None):
        X = raw[artifacts.SIGNALS].copy()
        p = np.full(len(raw), 0.5) if probability is None else probability
        return submit.apply_corrections(base, raw, X, self.parameters, p)

    def test_selected_recipe_routes_each_regime_in_the_validated_order(self):
        raw = departures(
            dict(lobt_gap=50000, aobt_gap=50000),
            day_row(),
            day_row(gap=18000),
            dict(FLIGHT_mvt='TWIN', lobt_gap=7000, aobt_gap=7000),
            dict(FLIGHT_mvt='TWIN', FLIGHT_ID_mvt=np.nan, lobt_gap=None, aobt_gap=None),
            dict(FLIGHT_ID_mvt=np.nan, gap=7200, lobt_gap=None, aobt_gap=None),
            {},
        )
        raw.index = np.arange(len(raw)) * 3 + 5
        frame, masks = self.apply(raw, [1000, 54000, 18000, 5000, 1000, 8000, 1000],
                                  [0.5, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(frame.pred_combined,
            [50000, 54000 + 33400 / 6, 1170, 6670, 1000, 3600, 1000])
        self.assertEqual(frame.pred_control.iloc[0], 46340)
        self.assertEqual(frame.pred_control.iloc[5], 3600)
        self.assertEqual(masks['rule_confidence'].sum(), 1)
        self.assertEqual(masks['twins'].sum(), 1)
        np.testing.assert_array_equal(frame.mvt_id, raw.mvt_id)

    def test_real_classifier_probability_controls_the_confidence_exception(self):
        raw = departures(day_row(gap=18000), day_row(gap=18000))
        frame, masks = self.apply(raw, [18000, 18000], [0.049, 0.05])
        np.testing.assert_allclose(frame.pred_combined, [1833, 18000])
        self.assertEqual(masks['rule_confidence'].tolist(), [True, False])

    def test_full_year_handler_weights_are_used_when_supported(self):
        self.parameters['day'].update(handler_stratified=True,
                                     handler_weights={'low': 0.2, 'high': 0.8})
        raw = departures(day_row(handler_arr_sched_share=0.1),
                         day_row(handler_arr_sched_share=0.9))
        frame, _ = self.apply(raw, [54000, 54000])
        np.testing.assert_allclose(frame.pred_combined, [60680, 80720])

    def test_labels_and_departure_block_times_cannot_change_predictions(self):
        raw = departures(day_row(), dict(lobt_gap=50000, aobt_gap=50000))
        expected, _ = self.apply(raw, [54000, 1000])
        raw['y'] = [np.inf, -999999]
        raw['BLOCK_TIME_UTC_mvt'] = None
        actual, _ = self.apply(raw, [54000, 1000])
        np.testing.assert_array_equal(actual.pred_combined, expected.pred_combined)

    def test_misaligned_features_are_rejected(self):
        raw = departures(day_row(), {})
        with self.assertRaises(ValueError):
            submit.apply_corrections([54000, 1000], raw, raw[artifacts.SIGNALS].iloc[::-1],
                                     self.parameters, [0.5, 0.5])


if __name__ == '__main__':
    unittest.main()
