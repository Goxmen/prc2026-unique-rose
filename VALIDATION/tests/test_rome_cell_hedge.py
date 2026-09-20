"""Checks for the in-between guess on the clean four-to-six-hour Rome rule cell."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rome_cell_hedge as rch


def departures(rows):
    """(airport, flight, flight_id, schedule, takeoff), times in UTC."""
    frame = pd.DataFrame(rows, columns=['ADEP_mvt', 'FLIGHT_mvt', 'FLIGHT_ID_mvt',
                                        'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt'])
    for column in ['SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt']:
        frame[column] = pd.to_datetime(frame[column], utc=True, format='ISO8601')
    return frame


# Real 2025 rows: three clean 4-6 h (two typed in, one real off-block), one mangled
# 4-6 h, one clean 6-14 h, one clean row only three hours late.
TRAIN = departures([
    ('LIRF', 'RYR68SF', np.nan, '2025-07-28 16:45', '2025-07-28 21:00:01'),   # real off-block, y 723
    ('LIRF', 'ACA891', np.nan, '2025-07-21 10:25', '2025-07-21 14:32:01'),    # typed in, y 14821
    ('LIRF', 'FIN4NC', np.nan, '2025-07-21 16:55', '2025-07-21 21:32:00'),    # typed in, y 16674
    ('LIRF', 'WMTMT6AN', np.nan, '2025-07-27 19:30', '2025-07-27 23:38:00'),  # mangled, out of the cell
    ('LIRF', 'TAR753', np.nan, '2025-08-13 09:15', '2025-08-13 21:39:00'),    # 12 h late, out of the cell
    ('LIRF', 'ABR468', np.nan, '2025-07-19 03:30', '2025-07-19 07:30:00'),    # 4 h exactly, in the cell
])
TRAIN_Y = np.array([723.0, 14821.0, 16674.0, 14939.0, 44698.0, 14400.0])


class Eligibility(unittest.TestCase):
    def test_only_the_clean_four_to_six_hour_rome_cell(self):
        self.assertEqual(list(rch.eligible(TRAIN)), [True, True, True, False, False, True])

    def test_a_flight_id_or_another_airport_is_never_eligible(self):
        rows = departures([
            ('LIRF', 'RYR68SF', 12345.0, '2025-07-28 16:45', '2025-07-28 21:00:01'),
            ('EHAM', 'RYR68SF', np.nan, '2025-07-28 16:45', '2025-07-28 21:00:01'),
        ])
        self.assertEqual(list(rch.eligible(rows)), [False, False])

    def test_the_band_is_closed_below_and_open_above(self):
        rows = departures([
            ('LIRF', 'AAA111', np.nan, '2025-07-01 00:00', '2025-07-01 03:59:59'),
            ('LIRF', 'AAA112', np.nan, '2025-07-01 00:00', '2025-07-01 04:00:00'),
            ('LIRF', 'AAA113', np.nan, '2025-07-01 00:00', '2025-07-01 05:59:59'),
            ('LIRF', 'AAA114', np.nan, '2025-07-01 00:00', '2025-07-01 06:00:00'),
        ])
        self.assertEqual(list(rch.eligible(rows)), [False, True, True, False])


class History(unittest.TestCase):
    def test_counts_and_taxi_come_from_eligible_labelled_rows_only(self):
        hist = rch.history(TRAIN, TRAIN_Y)
        self.assertEqual(hist['rows'], 4)          # RYR68SF, ACA891, FIN4NC, ABR468
        self.assertEqual(hist['planned'], 3)
        self.assertEqual(hist['taxi_sec'], 723.0)

    def test_an_empty_window_falls_back_without_failing(self):
        hist = rch.history(TRAIN.iloc[3:5], TRAIN_Y[3:5])
        self.assertEqual(hist, dict(rows=0, planned=0, taxi_sec=rch.TAXI_FALLBACK_SEC))

    def test_a_label_that_is_neither_face_is_ignored(self):
        rows = departures([('LIRF', 'AAA111', np.nan, '2025-07-01 00:00', '2025-07-01 05:00:00')])
        self.assertEqual(rch.history(rows, np.array([9000.0]))['rows'], 0)


class Hedge(unittest.TestCase):
    def test_eligible_rows_move_to_the_weighted_mean_and_others_do_not(self):
        hist = rch.history(TRAIN, TRAIN_Y)
        pred = np.arange(len(TRAIN), dtype=float)
        out = rch.hedge(pred, TRAIN, hist)
        p = (3 + rch.PRIOR_WEIGHT * 0.5) / (4 + rch.PRIOR_WEIGHT)
        gap = (pd.to_datetime(TRAIN.MVT_TIME_UTC_mvt) - pd.to_datetime(TRAIN.SCHED_TIME_UTC_mvt)
               ).dt.total_seconds().to_numpy(float)
        np.testing.assert_allclose(out[[0, 1, 2, 5]], p * gap[[0, 1, 2, 5]] + (1 - p) * 723.0)
        np.testing.assert_array_equal(out[[3, 4]], pred[[3, 4]])

    def test_the_guess_lies_between_the_two_faces(self):
        hist = rch.history(TRAIN, TRAIN_Y)
        out = rch.hedge(np.zeros(len(TRAIN)), TRAIN, hist)
        gap = (pd.to_datetime(TRAIN.MVT_TIME_UTC_mvt) - pd.to_datetime(TRAIN.SCHED_TIME_UTC_mvt)
               ).dt.total_seconds().to_numpy(float)
        for i in [0, 1, 2, 5]:
            self.assertGreater(out[i], hist['taxi_sec'])
            self.assertLess(out[i], gap[i])

    def test_shrinkage_keeps_the_probability_off_the_ends(self):
        rows = TRAIN.iloc[[1]]
        for planned, total in [(0, 1), (1, 1)]:
            hist = dict(rows=total, planned=planned, taxi_sec=1000.0)
            p = rch.weights(rows, hist)[0]
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)

    def test_no_history_leaves_every_prediction_untouched(self):
        pred = np.arange(len(TRAIN), dtype=float)
        out = rch.hedge(pred, TRAIN, dict(rows=0, planned=0, taxi_sec=rch.TAXI_FALLBACK_SEC))
        np.testing.assert_array_equal(out, pred)

    def test_the_input_predictions_are_not_mutated(self):
        pred = np.arange(len(TRAIN), dtype=float)
        rch.hedge(pred, TRAIN, rch.history(TRAIN, TRAIN_Y))
        np.testing.assert_array_equal(pred, np.arange(len(TRAIN), dtype=float))


if __name__ == '__main__':
    unittest.main()
