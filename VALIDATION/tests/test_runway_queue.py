"""Checks for the takeoff-anchored runway queue inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runway_queue as q


def times(values):
    return pd.to_datetime(values, utc=True)


def departures(rows):
    frame = pd.DataFrame(rows, columns=['ADEP_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt'])
    frame['MVT_TIME_UTC_mvt'] = times(frame['MVT_TIME_UTC_mvt'])
    frame['AOBT_3_flt'] = times(frame['AOBT_3_flt'])
    return frame


def arrivals(rows):
    frame = pd.DataFrame(rows, columns=['ADES_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt'])
    frame['MVT_TIME_UTC_mvt'] = times(frame['MVT_TIME_UTC_mvt'])
    return frame


class WorkedExampleTests(unittest.TestCase):
    def setUp(self):
        self.dep = departures([
            ('LFPG', '09L', '2025-07-01 09:20', '2025-07-01 09:00'),  # gap 1200, exactly 60 min before Q
            ('LFPG', '09L', '2025-07-01 10:00', '2025-07-01 09:45'),  # gap 900
            ('LFPG', '09L', '2025-07-01 10:10', '2025-07-01 09:50'),  # gap 1200
            ('LFPG', '09L', '2025-07-01 10:20', '2025-07-01 10:05'),  # Q itself, gap 900
            ('LFPG', '08R', '2025-07-01 10:20', None),                # same instant, other runway
            ('LFPG', '09L', '2025-07-01 10:25', '2025-07-01 10:15'),  # gap 600
            ('LFPG', '09L', '2025-07-01 10:50', '2025-07-01 08:00'),  # exactly +30 min; gap 10200 is filtered
            ('EGLL', '09L', '2025-07-01 10:15', '2025-07-01 10:10'),  # same runway name, other airport
        ])
        self.arr = arrivals([
            ('LFPG', '09L', '2025-07-01 09:45'),  # before the 30-minute window
            ('LFPG', '09L', '2025-07-01 10:05'),
            ('LFPG', '09L', '2025-07-01 10:20'),  # at Q's takeoff instant: excluded
            ('LFPG', '08R', '2025-07-01 10:00'),  # other runway
            ('EGLL', '09L', '2025-07-01 10:10'),
        ])
        self.queries = pd.concat([
            self.dep.iloc[[3, 7]],
            departures([('LFPG', None, '2025-07-01 10:30', None)]),
        ], ignore_index=True)
        self.queries.index = [100, 101, 102]

    def test_values(self):
        out = q.queue_features(self.queries, self.dep, self.arr)
        self.assertEqual(out.index.tolist(), [100, 101, 102])
        self.assertEqual(out.columns.tolist(), q.ALL_FEATURES)
        expected_q = dict(q_rwy_prev_takeoff_gap_sec=600, q_rwy_takeoffs_prev30=2, q_apt_takeoffs_prev30=2,
                          q_rwy_others_gap_prev60=1100, q_apt_others_gap_prev60=1100, q_queue_ahead_since_aobt3=1,
                          q_rwy_landings_prev30=1, q_rwy_next_takeoff_gap_sec=300, q_rwy_takeoffs_next30=2,
                          q_rwy_others_gap_pm30=900, q_apt_others_gap_pm30=900)
        for name, value in expected_q.items():
            self.assertAlmostEqual(float(out.loc[100, name]), value, msg=name)
        egll = out.loc[101]
        self.assertTrue(np.isnan(egll.q_rwy_prev_takeoff_gap_sec), 'LFPG 09L must not leak into EGLL 09L')
        self.assertEqual(egll.q_rwy_takeoffs_prev30, 0)
        self.assertEqual(egll.q_rwy_takeoffs_next30, 0)
        self.assertTrue(np.isnan(egll.q_rwy_others_gap_pm30))
        self.assertEqual(egll.q_rwy_landings_prev30, 1)
        self.assertEqual(egll.q_queue_ahead_since_aobt3, 0)
        no_runway = out.loc[102]
        for name in ['q_rwy_prev_takeoff_gap_sec', 'q_rwy_takeoffs_prev30', 'q_rwy_others_gap_prev60',
                     'q_rwy_landings_prev30', 'q_rwy_next_takeoff_gap_sec', 'q_rwy_takeoffs_next30',
                     'q_rwy_others_gap_pm30', 'q_queue_ahead_since_aobt3']:
            self.assertTrue(np.isnan(no_runway[name]), name)
        self.assertEqual(no_runway.q_apt_takeoffs_prev30, 5)
        self.assertAlmostEqual(float(no_runway.q_apt_others_gap_prev60), 900)
        self.assertAlmostEqual(float(no_runway.q_apt_others_gap_pm30), 900)

    def test_rows_need_not_be_events(self):
        with_self = q.queue_features(self.queries.iloc[:1], self.dep, self.arr)
        without_self = q.queue_features(self.queries.iloc[:1], self.dep.drop(index=3), self.arr)
        pd.testing.assert_frame_equal(with_self, without_self)

    def test_only_declared_columns_are_read(self):
        dep = self.dep.assign(BLOCK_TIME_UTC_mvt=times(['2020-01-01'] * len(self.dep)), TAXITIME_SEC_mvt=1e9)
        pd.testing.assert_frame_equal(q.queue_features(self.queries, dep, self.arr),
                                      q.queue_features(self.queries, self.dep, self.arr))

    def test_naive_timestamps_match_utc(self):
        naive = self.dep.assign(MVT_TIME_UTC_mvt=self.dep.MVT_TIME_UTC_mvt.dt.tz_localize(None),
                                AOBT_3_flt=self.dep.AOBT_3_flt.dt.tz_localize(None))
        pd.testing.assert_frame_equal(q.queue_features(self.queries, naive, self.arr),
                                      q.queue_features(self.queries, self.dep, self.arr))

    def test_add_queue_alignment_and_overlap(self):
        with self.assertRaises(ValueError):
            q.add_queue(pd.DataFrame({'a': [1.0, 2.0, 3.0]}), self.queries, self.dep, self.arr)
        X = q.add_queue(pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=self.queries.index), self.queries,
                        self.dep, self.arr, q.PAST_FEATURES)
        self.assertEqual(X.columns.tolist(), ['a'] + q.PAST_FEATURES)
        with self.assertRaises(ValueError):
            q.add_queue(X, self.queries, self.dep, self.arr, q.PAST_FEATURES)


class PastOnlyTests(unittest.TestCase):
    def test_past_inputs_ignore_later_movements(self):
        rng = np.random.default_rng(3)
        n = 600
        start = pd.Timestamp('2025-07-01', tz='UTC')
        takeoff = start + pd.to_timedelta(np.sort(rng.integers(0, 12 * 3600, n)), unit='s')
        aobt = takeoff - pd.to_timedelta(rng.integers(-300, 3000, n), unit='s')
        dep = pd.DataFrame({'ADEP_mvt': rng.choice(['LFPG', 'EDDF'], n), 'RUNWAY_mvt': rng.choice(['09L', '27R', None], n),
                            'MVT_TIME_UTC_mvt': takeoff, 'AOBT_3_flt': aobt.where(rng.random(n) > 0.1)})
        land = start + pd.to_timedelta(rng.integers(0, 12 * 3600, n), unit='s')
        arr = pd.DataFrame({'ADES_mvt': rng.choice(['LFPG', 'EDDF'], n), 'RUNWAY_mvt': rng.choice(['09L', '27R'], n),
                            'MVT_TIME_UTC_mvt': land})
        cutoff = start + pd.Timedelta(hours=6)
        queries = dep[dep.MVT_TIME_UTC_mvt < cutoff]
        full = q.queue_features(queries, dep, arr)
        truncated = q.queue_features(queries, dep[dep.MVT_TIME_UTC_mvt < cutoff], arr[arr.MVT_TIME_UTC_mvt < cutoff])
        pd.testing.assert_frame_equal(full[q.PAST_FEATURES], truncated[q.PAST_FEATURES])
        self.assertFalse(full[q.FUTURE_FEATURES].equals(truncated[q.FUTURE_FEATURES]),
                         'future inputs near the cutoff should see the removed movements')


if __name__ == '__main__':
    unittest.main()
