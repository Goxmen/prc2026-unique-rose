"""Checks for the in-between guess on possibly wrong-date Rome rule rows."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dayshift_hedge as dh


def departures(rows):
    """(airport, flight, flight_id, schedule, takeoff), times in UTC."""
    frame = pd.DataFrame(rows, columns=['ADEP_mvt', 'FLIGHT_mvt', 'FLIGHT_ID_mvt', 'SCHED_TIME_UTC_mvt',
                                        'MVT_TIME_UTC_mvt'])
    for column in ['SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt']:
        frame[column] = pd.to_datetime(frame[column], utc=True, format='ISO8601')
    return frame


# 2025 training rows: two Jet2 (one each way), three ITY (two wrong date, one planned time), one WMT planned time.
TRAIN = departures([
    ('LIRF', 'EXSXS3AD', np.nan, '2025-01-24 11:30', '2025-01-25 03:13:56'),
    ('LIRF', 'EXSXS1GT', np.nan, '2025-01-24 10:55', '2025-01-25 03:36:58'),
    ('LIRF', 'ITYTY680', np.nan, '2025-07-29 20:30', '2025-07-30 12:46:00'),
    ('LIRF', 'ITYTY630', np.nan, '2025-09-28 08:35', '2025-09-28 22:38:00'),
    ('LIRF', 'ITYTY674', np.nan, '2025-08-26 20:05', '2025-08-27 10:17:00'),
    ('LIRF', 'WMTT5232', np.nan, '2025-07-13 18:40', '2025-07-14 12:33:00'),
])
LABELS = np.array([56633.0, 87177.0, 86400 + 800.0, 86400 + 900.0, 51120.0, 64380.0])


class HedgeTests(unittest.TestCase):
    def setUp(self):
        self.hist = dh.history(TRAIN, LABELS)

    def test_history_counts(self):
        self.assertEqual(self.hist['rows'], 6)
        self.assertEqual(self.hist['wrong_date'], 3)
        self.assertEqual(self.hist['format'], {'mangled': [3, 6], 'clean': [0, 0]})
        self.assertEqual(self.hist['airline']['ITY'], [2, 3])
        self.assertEqual(self.hist['airline']['WMT'], [0, 1])
        self.assertEqual(self.hist['taxi_sec'], 800.0)

    def test_only_eligible_rows_move(self):
        rank = departures([
            ('LIRF', 'WMTT3508', np.nan, '2026-01-02 19:45', '2026-01-03 11:27:00'),  # eligible
            ('LIRF', 'TAY9LA', np.nan, '2026-01-22 05:45', '2026-01-22 20:37:00'),    # same local day
            ('LIRF', 'ITY100', 1.0, '2026-07-09 20:10', '2026-07-10 13:12:00'),       # has a flight ID
            ('LIRF', 'RYR1', np.nan, '2026-07-09 20:10', '2026-07-10 06:12:00'),      # 10 h late
            ('LFPG', 'AFR1', np.nan, '2026-07-09 20:10', '2026-07-10 13:12:00'),      # not Rome
        ])
        pred = np.array([56520.0, 53520.0, 900.0, 36120.0, 900.0])
        out = dh.hedge(pred, rank, self.hist)
        self.assertNotEqual(out[0], pred[0])
        np.testing.assert_array_equal(out[1:], pred[1:])

    def test_weighted_mean_of_the_two_labels(self):
        rank = departures([
            ('LIRF', 'WMTT3508', np.nan, '2026-01-02 19:45', '2026-01-03 11:27:00'),
            ('LIRF', 'ITYTY672', np.nan, '2026-07-09 20:10', '2026-07-10 13:12:02'),
            ('LIRF', 'SVII3006', np.nan, '2026-07-16 15:30', '2026-07-17 08:20:56'),
        ])
        gap = np.array([56520.0, 61322.0, 60656.0])
        p = dh.weights(rank, self.hist)
        np.testing.assert_allclose(p, [(0 + 2 * 0.5) / 3, (2 + 2 * 0.5) / 5, 0.5])
        out = dh.hedge(gap, rank, self.hist)
        np.testing.assert_allclose(out, p * (86400 + 800.0) + (1 - p) * gap)

    def test_small_groups_never_reach_certainty(self):
        hist = dict(rows=1, wrong_date=1, format={'mangled': [0, 0], 'clean': [1, 1]}, airline={'QFA': [0, 0]},
                    taxi_sec=900.0)
        rank = departures([('LIRF', 'QFAQFA6', np.nan, '2026-09-05 09:00', '2026-09-06 00:14:00')])
        overall = (1 + 2 * 0.5) / 3
        clean = (1 + 2 * overall) / 3
        np.testing.assert_allclose(dh.weights(rank, hist), [(0 + 2 * clean) / 2])
        self.assertLess(dh.weights(rank, hist)[0], 1.0)

    def test_no_history_keeps_predictions(self):
        rank = departures([('LIRF', 'WMTT3508', np.nan, '2026-01-02 19:45', '2026-01-03 11:27:00')])
        empty = dh.history(TRAIN.iloc[:0], LABELS[:0])
        np.testing.assert_array_equal(dh.hedge(np.array([56520.0]), rank, empty), [56520.0])


if __name__ == '__main__':
    unittest.main()
