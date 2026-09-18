"""Checks for the stand clock and dirty-day inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stand_day as sd


def times(values):
    return pd.to_datetime(values, utc=True)


def departures(rows):
    """(mvt_id, airport, stand, takeoff, schedule, flight_id, type, flight) plus a label that must never be read."""
    frame = pd.DataFrame(rows, columns=['mvt_id', 'ADEP_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                                        'FLIGHT_ID_mvt', 'AIRCRAFT_TYPE_mvt', 'FLIGHT_mvt'])
    for column in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']:
        frame[column] = times(frame[column])
    frame['taxi_sec'] = 99999.0
    return frame


def arrivals(rows):
    """(airport, stand, landing, taxi-in, schedule, type, flight)."""
    frame = pd.DataFrame(rows, columns=['ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'taxi_sec',
                                        'SCHED_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt', 'FLIGHT_mvt'])
    for column in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']:
        frame[column] = times(frame[column])
    return frame


class StandClockTests(unittest.TestCase):
    def setUp(self):
        self.arr = arrivals([
            ('LIRF', '711', '2025-07-01 20:12', 480, '2025-07-01 19:00', 'A320', 'ISR341'),  # in-block 20:20, own plane
            ('LIRF', '711', '2025-07-01 17:00', 300, '2025-07-01 17:05', 'B738', 'RYR100'),  # in-block 17:05
            ('LIRF', '711', '2025-07-01 21:40', 600, '2025-07-01 21:50', 'B738', 'RYR200'),  # in-block 21:50 = takeoff
            ('LSZH', '711', '2025-07-01 21:00', 60, '2025-07-01 21:00', 'A320', 'ISR999'),   # same stand name, other airport
            ('EGLL', '331', '2025-07-01 18:40', 180, '2025-07-01 18:40', 'A320', 'BAW703E'),  # in-block 18:43
        ])
        self.dep = departures([
            (1, 'LIRF', '711', '2025-07-01 21:50', '2025-07-01 19:20', np.nan, 'A320', 'ISR342'),
            (2, 'EGLL', '331', '2025-07-01 19:09', '2025-07-01 18:30', 7.0, 'A35K', 'BAW5VY'),
            (3, 'EGLL', '999', '2025-07-01 19:09', '2025-07-01 18:30', 8.0, 'A35K', 'BAW5VZ'),  # no arrival
            (4, 'EGLL', None, '2025-07-01 19:09', '2025-07-01 18:30', 9.0, None, None),         # no stand
        ])

    def test_worked_example(self):
        f = sd.stand_features(self.dep, self.arr)
        # ISR342: own plane parked at 20:20, takeoff 21:50; the 21:50 in-block is not before takeoff.
        self.assertEqual(f.loc[0, 'st_last_arr_sec'], 90 * 60)
        self.assertEqual(f.loc[0, 'st_prev_arr_sec'], (21 * 60 + 50 - (17 * 60 + 5)) * 60)
        self.assertEqual(f.loc[0, 'st_same_type'], 1.0)
        self.assertEqual(f.loc[0, 'st_same_airline'], 1.0)
        self.assertEqual(f.loc[0, 'st_inblock_minus_sched_sec'], 60 * 60)
        # BAW5VY: another plane parked 26 min before takeoff.
        self.assertEqual(f.loc[1, 'st_last_arr_sec'], 26 * 60)
        self.assertTrue(np.isnan(f.loc[1, 'st_prev_arr_sec']))
        self.assertEqual(f.loc[1, 'st_same_type'], 0.0)
        self.assertEqual(f.loc[1, 'st_same_airline'], 1.0)
        self.assertEqual(f.loc[1, 'st_inblock_minus_sched_sec'], 13 * 60)
        self.assertTrue(f.loc[2:].isna().all().all())

    def test_missing_type_gives_missing_match(self):
        dep = self.dep.copy()
        dep.loc[0, 'AIRCRAFT_TYPE_mvt'] = None
        self.assertTrue(np.isnan(sd.stand_features(dep, self.arr).loc[0, 'st_same_type']))

    def test_row_order_does_not_matter(self):
        f = sd.stand_features(self.dep, self.arr)
        shuffled = self.dep.iloc[[3, 1, 0, 2]]
        pd.testing.assert_frame_equal(sd.stand_features(shuffled, self.arr.iloc[::-1]), f.loc[shuffled.index])


class DirtyDayTests(unittest.TestCase):
    def setUp(self):
        d = '2025-07-01'
        self.dep = departures([
            (1, 'LTFM', 'A1', f'{d} 10:00', f'{d} 09:00', np.nan, None, 'THY1'),   # no ID, no type, 60 min
            (2, 'LTFM', 'A2', f'{d} 10:30', f'{d} 10:10', 2.0, 'B738', 'THY2'),     # 20 min
            (3, 'LTFM', 'A3', f'{d} 11:00', f'{d} 10:00', 3.0, 'A321', 'THY3'),     # 60 min, late
            (4, 'LTFM', 'A4', f'{d} 12:00', f'{d} 11:50', 4.0, 'A321', 'THY4'),     # 10 min
            (5, 'LTFM', 'A5', '2025-07-02 10:00', '2025-07-02 09:00', np.nan, 'A321', 'THY5'),  # next day
            (6, 'EHAM', 'B1', f'{d} 10:00', f'{d} 09:00', np.nan, 'B738', 'KLM1'),  # other airport
        ])
        self.arr = arrivals([
            ('LTFM', 'C1', f'{d} 08:00', 600, f'{d} 08:10', 'A321', 'THY9'),        # in-block = schedule
            ('LTFM', 'C2', f'{d} 08:00', 600, f'{d} 08:00', 'A321', 'THY8'),        # 10 min late
            ('LTFM', 'C3', '2025-06-30 23:55', 600, '2025-07-01 00:05', 'A321', 'THY7'),  # landed the day before
        ])

    def test_worked_example(self):
        f = sd.day_features(self.dep, self.dep, self.arr)
        # Row 1 sees rows 2-4: none without an ID or type, one of three late.
        self.assertEqual(f.loc[0, 'dd_noid_share'], 0.0)
        self.assertEqual(f.loc[0, 'dd_notype_share'], 0.0)
        self.assertAlmostEqual(f.loc[0, 'dd_late_share'], 1 / 3)
        # Row 2 sees rows 1, 3 and 4: one without an ID and type; row 3 is its only late ID row of two others.
        self.assertAlmostEqual(f.loc[1, 'dd_noid_share'], 1 / 3)
        self.assertAlmostEqual(f.loc[1, 'dd_notype_share'], 1 / 3)
        self.assertAlmostEqual(f.loc[1, 'dd_late_share'], 1 / 2)
        self.assertEqual(f.loc[2, 'dd_late_share'], 0.0)
        # Mean delay of the other ID departures (20, 60 and 10 min).
        self.assertEqual(f['dd_delay_mean_sec'].iloc[:4].tolist(), [30 * 60, 35 * 60, 15 * 60, 40 * 60])
        # Arrivals: one of two that landed on 1 July blocked in on schedule.
        self.assertTrue((f.loc[0:3, 'dd_arr_sched_share'] == 0.5).all())
        # Row 5 is alone on its day; row 6 alone at its airport.
        self.assertTrue(np.isnan(f.loc[4, 'dd_noid_share']))
        self.assertTrue(np.isnan(f.loc[4, 'dd_arr_sched_share']))
        self.assertTrue(np.isnan(f.loc[5, 'dd_noid_share']))

    def test_window_share_excludes_the_instant(self):
        f = sd.day_features(self.dep, self.dep, self.arr)
        # 10:00 sees 10:30 and 11:00 (11:30 limit), both with an ID; EHAM's 10:00 row is another airport.
        self.assertEqual(f.loc[0, 'dh_noid_share_pm90'], 0.0)
        # 11:00 sees 10:00 (no ID), 10:30 and 12:00.
        self.assertAlmostEqual(f.loc[2, 'dh_noid_share_pm90'], 1 / 3)

    def test_same_result_whether_described_rows_are_events(self):
        described = self.dep.iloc[[1]]
        others = self.dep.drop(index=1)
        pd.testing.assert_frame_equal(sd.day_features(described, self.dep, self.arr),
                                      sd.day_features(described, others, self.arr))


class BoundaryTests(unittest.TestCase):
    def test_departure_labels_are_never_read(self):
        t = DirtyDayTests()
        t.setUp()
        a = sd.stand_day_features(t.dep, t.dep, t.arr)
        without = t.dep.drop(columns=['taxi_sec'])
        pd.testing.assert_frame_equal(a, sd.stand_day_features(t.dep, without, t.arr))
        changed = t.dep.assign(taxi_sec=1.0)
        pd.testing.assert_frame_equal(a, sd.stand_day_features(changed, changed, t.arr))

    def test_add_checks_alignment_and_duplicates(self):
        t = DirtyDayTests()
        t.setUp()
        X = pd.DataFrame({'x': np.arange(len(t.dep))}, index=t.dep.index)
        out = sd.add_stand_day(X, t.dep, t.dep, t.arr)
        self.assertEqual(list(out.columns), ['x', *sd.ALL_FEATURES])
        self.assertTrue((out.dtypes.iloc[1:] == 'float32').all())
        with self.assertRaises(ValueError):
            sd.add_stand_day(out, t.dep, t.dep, t.arr)
        with self.assertRaises(ValueError):
            sd.add_stand_day(X.iloc[::-1], t.dep, t.dep, t.arr)


if __name__ == '__main__':
    unittest.main()
