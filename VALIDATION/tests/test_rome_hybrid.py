"""Checks for the Rome hybrid blend, its training label and the four-hour override."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import overnight_models as models
import rome_hybrid as h


class Stub:
    def __init__(self, values):
        self.values = np.asarray(values, float)
        self.seen = []

    def predict(self, X, num_iteration=None, num_threads=None):
        self.seen.append(list(X.columns))
        return self.values[X.index.to_numpy()]


def frames(gaps, airports, id_missing):
    n = len(gaps)
    takeoff = pd.Timestamp('2025-07-01 12:00', tz='UTC')
    raw = pd.DataFrame({
        'ADEP_mvt': airports,
        'FLIGHT_ID_mvt': [np.nan if m else 1.0 for m in id_missing],
        'MVT_TIME_UTC_mvt': [takeoff] * n,
        'SCHED_TIME_UTC_mvt': [takeoff - pd.Timedelta(seconds=g) for g in gaps],
        'raw_flight_id_missing': id_missing, 'raw_plan_block_missing': id_missing,
    })
    names = h.regressor_features('clock')
    V = pd.DataFrame(0.0, index=raw.index, columns=names)
    V[h.GAP] = np.asarray(gaps, float)
    return V, raw


class RomeHybridTests(unittest.TestCase):
    def test_label_marks_lirf_rows_logged_at_schedule(self):
        V, raw = frames([1200, 1200, 1200, 1200], ['LIRF', 'LIRF', 'LIRF', 'EGLL'], [True] * 4)
        s = h.substituted(V, [1200, 1140, 1139, 1200], raw)
        self.assertEqual(s.tolist(), [True, True, False, False])

    def test_blend_at_lirf_and_four_hour_rule_override(self):
        # rows: LIRF 3h no ID (blend), LIRF 5h no ID (rule), LIRF 5h with ID (blend),
        #       EGLL 5h no ID (ordinary), LIRF 20 min (blend barely moves)
        gaps = [10800, 18000, 18000, 18000, 1200]
        V, raw = frames(gaps, ['LIRF', 'LIRF', 'LIRF', 'EGLL', 'LIRF'], [True, True, False, True, False])
        ordinary = Stub([900] * 5)
        classifier = Stub([0.6, 0.9, 0.5, 0.99, 0.25])
        out = h.predict(ordinary, classifier, V, raw, 'original', threads=1)
        np.testing.assert_allclose(out['p'], [0.6, 0.9, 0.5, 0.0, 0.25])
        np.testing.assert_allclose(out['blended'], [6840, 16290, 9450, 900, 975])
        np.testing.assert_allclose(out['final'], [6840, 18000, 9450, 900, 975])
        self.assertEqual(out['rule_triggered'].tolist(), [False, True, False, False, False])
        self.assertEqual(ordinary.seen[0], models.ORIGINAL_FEATURES)
        self.assertEqual(classifier.seen[0], models.ORIGINAL_FEATURES + h.FLAGS)

    def test_clock_kind_uses_all_45_inputs_without_duplicate_flags(self):
        V, raw = frames([1200], ['LIRF'], [True])
        X = h.classifier_matrix(V, raw, 'clock')
        self.assertEqual(X.columns.tolist(), models.ORIGINAL_FEATURES + models.CLOCK_FEATURES)
        self.assertTrue(X.columns.is_unique)


if __name__ == '__main__':
    unittest.main()
