"""Keep v40's exclusion out of correction fitting, categories and event context."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import opdi_disruption_run as run


class OpdiDisruptionTests(unittest.TestCase):
    def prepared(self, exclude):
        raw = pd.DataFrame(dict(mvt_id=[10., 11., 12., 13.], ADEP_mvt=['LIRF', 'LIRF', 'EHAM', 'LIRF'],
            FLIGHT_ID_mvt=[np.nan, np.nan, np.nan, 4.],
            MVT_TIME_UTC_mvt=pd.to_datetime(['2025-07-01 10:00'] * 4, utc=True),
            SCHED_TIME_UTC_mvt=pd.to_datetime(['2025-07-01 06:00', '2025-07-01 07:00',
                                               '2025-07-01 04:00', '2025-07-01 05:00'], utc=True)),
            index=[20, 30, 40, 50])
        X, y, cats = pd.DataFrame({'base': [1., 2., 3., 4.]}, index=raw.index), np.arange(4.), {'airport': 'all'}
        arrivals = pd.DataFrame({'mvt_id': [22.]})
        expected = dict(late_mean=9.)
        seen = {}

        def fit_parameters(matrix, labels, rows):
            pd.testing.assert_frame_equal(rows, raw)
            np.testing.assert_array_equal(labels, y)
            seen['parameters'] = True
            return expected

        def add_inputs(matrix, rows, reports, deps, arrs):
            self.assertTrue(seen['parameters'])
            self.assertIs(deps, raw)
            self.assertIs(arrs, arrivals)
            self.assertEqual(rows.mvt_id.tolist(), [11., 12., 13.] if exclude else raw.mvt_id.tolist())
            self.assertTrue(matrix.index.equals(rows.index))
            return matrix

        def add_features(matrix, rows, arrs, grid):
            self.assertIs(arrs, arrivals)
            return matrix

        with patch.object(run.overnight_run, 'prepare_training', return_value=(X, y, cats, raw)), \
             patch.object(run.signals, 'fit_parameters', side_effect=fit_parameters), \
             patch.object(run.recipe, 'add_inputs', side_effect=add_inputs), \
             patch.object(run.features, 'add_features', side_effect=add_features), \
             patch.object(run.rome_hybrid, 'substituted', side_effect=lambda X, y, r: r.mvt_id.eq(11.).to_numpy()):
            result = run.training_inputs(raw, arrivals, dict(exclude_rule_rows=exclude, prediction_rule='rome_4h'),
                                         None, None, expected)
        self.assertIs(result[2], cats)
        audit = result[-1].set_index('mvt_id')
        self.assertEqual(bool(audit.loc[10., 'excluded_by_rule']), exclude)
        self.assertEqual(bool(audit.loc[10., 'used_by_classifier']), not exclude)
        self.assertFalse(audit.loc[11., 'used_by_ordinary'])
        self.assertTrue(audit.loc[11., 'used_by_classifier'])
        self.assertTrue(audit.loc[12., 'used_by_ordinary'])
        self.assertFalse(audit.loc[12., 'used_by_classifier'])

    def test_v39_keeps_rule_rows_and_original_context(self):
        self.prepared(False)

    def test_v40_excludes_at_four_hours_after_fitting_corrections(self):
        self.prepared(True)

    def test_candidate_changes_only_ordinary_feature_roster(self):
        self.assertEqual(run.EXTRA, run.ORDINARY + run.features.FEATURES)
        self.assertEqual(len(run.EXTRA), len(set(run.EXTRA)))
        self.assertFalse(set(run.features.FEATURES) & set(run.CLASSIFIER))
        self.assertTrue(set(run.opdi.OPDI_FEATURES) <= set(run.CLASSIFIER))


if __name__ == '__main__':
    unittest.main()
