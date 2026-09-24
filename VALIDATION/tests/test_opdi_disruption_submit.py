"""Check selected submission wiring without fitting or uploading."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import opdi_disruption_submit as submit
import recording_artifacts as corrections
from test_recording_artifacts import departures, day_row


class DisruptionSubmissionTests(unittest.TestCase):
    def test_parameter_guard_allows_roundoff_but_rejects_changed_recipe(self):
        expected = {'rows': 19, 'weight': .32498911877812825}
        submit.check_parameters({'rows': 19, 'weight': .3249891187781282}, expected)
        with self.assertRaises(AssertionError):
            submit.check_parameters({'rows': 19, 'weight': .3249892}, expected)
        with self.assertRaises(ValueError):
            submit.check_parameters({'rows': 18, 'weight': expected['weight']}, expected)

    def test_corrections_retain_rule_rows_and_all_arrival_donors(self):
        raw = departures(day_row(), day_row(gap=18000, y=1170),
                         dict(lobt_gap=8000, aobt_gap=8000, y=8300), {}, dict(y=200001))
        raw['taxi_sec'] = raw.y
        raw.index = [5, 8, 11, 19, 20]
        arrivals = pd.DataFrame({'donor': [1, 2]})

        def rates(rows, donors):
            self.assertIs(donors, arrivals)
            return raw.loc[rows.index, corrections.SIGNALS]

        with patch.object(submit.signals, 'handler_rates', side_effect=rates):
            fitted, audit = submit.fit_corrections(raw, arrivals, {'max_taxi_sec': 200000}, 1000)
        self.assertEqual(audit.mvt_id.tolist(), [1, 2, 3])
        self.assertEqual(fitted, corrections.fit(raw.iloc[:4], 1000))
        self.assertEqual(fitted['rule']['rows'], 1)
        self.assertEqual(fitted['day']['rows'], 1)

    def test_prediction_uses_new_ordinary_inputs_and_unchanged_classifier_and_rules(self):
        raw = departures(dict(lobt_gap=50000, aobt_gap=50000), day_row(), day_row(gap=18000))
        raw.index = [5, 8, 15]
        V = raw[corrections.SIGNALS].copy()
        V['op_ground_sec'], V['op_matched'] = [np.nan, 1000, 900], 1
        for name in submit.features.FEATURES:
            V[name] = [0., 1., 2.]
        fitted = corrections.fit(departures(day_row()), 1000)
        out = dict(final=np.array([1000., 54000., 18000.]), ordinary=np.full(3, 1000.),
                   p=np.array([.5, .5, .01]), rule_triggered=np.array([False, True, True]))
        with patch.object(submit.signals, 'predict', return_value=out) as mock:
            frame, masks = submit.predict('ordinary', 'classifier', V, raw, {'late_mean': 1000}, fitted)
        kw = mock.call_args.kwargs
        self.assertEqual(kw['extra'], submit.cv.EXTRA)
        self.assertEqual(kw['classifier_extra'], submit.cv.CLASSIFIER)
        self.assertFalse(set(submit.features.FEATURES) & set(kw['classifier_extra']))
        self.assertIsNone(kw['dayshift_weight'])
        np.testing.assert_allclose(frame.pred, [50000, 54000 + 33400 / 6, 1170])
        np.testing.assert_array_equal(frame.mvt_id, raw.mvt_id)
        self.assertEqual(masks['rule_confidence'], 1)
        for name in submit.features.FEATURES:
            np.testing.assert_array_equal(frame[name], V[name])


if __name__ == '__main__':
    unittest.main()
