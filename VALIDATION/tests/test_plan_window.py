"""Checks for the flight-plan window."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plan_window as pw


def departures(rows):
    """(flight_id, takeoff, LOBT)."""
    frame = pd.DataFrame(rows, columns=['FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt'])
    for column in ['MVT_TIME_UTC_mvt', 'LOBT_flt']:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


class PlanWindowTests(unittest.TestCase):
    def setUp(self):
        self.dep = departures([
            (1.0, '2026-07-17 21:51', '2026-07-16 21:44'),  # ITY340: takeoff a day after LOBT
            (2.0, '2026-01-05 12:00', '2026-01-05 09:00'),  # three hours after LOBT
            (3.0, '2026-01-05 12:00', '2026-01-05 11:40'),  # ordinary
            (np.nan, '2026-07-17 21:51', None),             # no flight ID: Rome rule rows look like this
            (4.0, '2026-07-17 21:51', None),                # flight ID without LOBT
            (5.0, '2026-01-05 12:00', '2026-01-05 13:30'),  # LOBT 90 min after takeoff
        ])

    def test_window_is_takeoff_minus_lobt_plus_minus_margin(self):
        low, high = pw.window(self.dep)
        centre = (24 * 60 + 7) * 60
        self.assertEqual(low[0], centre - pw.MARGIN_SEC)
        self.assertEqual(high[0], centre + pw.MARGIN_SEC)
        self.assertTrue(np.isneginf(low[3]) and np.isposinf(high[3]))
        self.assertTrue(np.isneginf(low[4]) and np.isposinf(high[4]))

    def test_predictions_move_to_the_nearest_edge_only_when_outside(self):
        pred = np.array([12924.0, 1200.0, 600.0, 56520.0, 700.0, 300.0])
        out = pw.apply(pred, self.dep)
        self.assertEqual(out[0], (24 * 60 + 7) * 60 - pw.MARGIN_SEC)  # raised to the low edge
        self.assertEqual(out[1], 3 * 3600 - pw.MARGIN_SEC)            # raised
        self.assertEqual(out[2], 600.0)                                # inside: unchanged
        self.assertEqual(out[3], 56520.0)                              # no flight ID: unchanged
        self.assertEqual(out[4], 700.0)                                # no LOBT: unchanged
        self.assertEqual(out[5], 0.0)                                  # window below zero: floored

    def test_high_edge_lowers_a_prediction(self):
        out = pw.apply(np.array([5000.0]), self.dep.iloc[[2]])
        self.assertEqual(out[0], 20 * 60 + pw.MARGIN_SEC)


if __name__ == '__main__':
    unittest.main()
