"""Checks for the no-ID cap and the twin-record fix."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import record_fixes as rf


def departures(rows):
    """(airport, flight, flight_id, takeoff, schedule, LOBT)."""
    frame = pd.DataFrame(rows, columns=['ADEP_mvt', 'FLIGHT_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt',
                                        'SCHED_TIME_UTC_mvt', 'LOBT_flt'])
    for column in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt']:
        frame[column] = pd.to_datetime(frame[column], utc=True, format='ISO8601')
    return frame


class NoIdCapTests(unittest.TestCase):
    def setUp(self):
        self.dep = departures([
            ('LSZH', 'NJE278K', np.nan, '2026-07-12 13:30:15', '2026-07-12 11:50', None),  # gap 100 min
            ('LSZH', 'QGA01A', np.nan, '2026-07-17 17:22:49', '2026-07-17 16:10', None),   # gap 73 min
            ('LTFM', 'THY3KY', np.nan, '2026-01-19 08:16:04', '2026-01-19 07:00', None),   # LTFM, gap 76 min
            ('LIRF', 'RYR12RR', np.nan, '2026-07-01 22:12:00', '2026-07-01 18:45', None),  # Rome: untouched
            ('LFPG', 'AFR1', 1.0, '2026-07-01 12:00', '2026-07-01 09:00', '2026-07-01 11:40'),  # has a flight ID
            ('EGLL', 'BAW2', np.nan, '2026-07-01 12:00', '2026-07-01 11:50', None),        # gap 10 min
        ])

    def test_ceilings(self):
        pred = np.array([13089.0, 9722.0, 5241.0, 11498.0, 9000.0, 1500.0])
        out = rf.noid_cap(pred, self.dep)
        self.assertEqual(out[0], 6015.0 - 3600)          # late cap
        self.assertEqual(out[1], 4369.0)                 # below 75 min: capped at the gap only
        self.assertEqual(out[2], 76 * 60 + 4)            # LTFM: gap cap only
        self.assertEqual(out[3], 11498.0)                # Rome keeps its prediction
        self.assertEqual(out[4], 9000.0)                 # flight ID rows keep theirs
        self.assertEqual(out[5], 1500.0)                 # gap under 30 min: no ceiling

    def test_only_lowers(self):
        pred = np.array([900.0, 900.0, 900.0, 900.0, 900.0, 900.0])
        np.testing.assert_array_equal(rf.noid_cap(pred, self.dep), pred)

    def test_late_cap_starts_above_75_minutes(self):
        dep = departures([('EDDF', 'X1', np.nan, '2026-07-01 12:16', '2026-07-01 11:00', None),   # gap 76 min
                          ('EDDF', 'X2', np.nan, '2026-07-01 12:15', '2026-07-01 11:00', None)])  # gap 75 min
        np.testing.assert_array_equal(rf.noid_cap(np.array([5000.0, 5000.0]), dep), [960.0, 4500.0])


class TwinRecordTests(unittest.TestCase):
    def setUp(self):
        self.dep = departures([
            ('LFPG', 'UAL914', np.nan, '2026-07-27 12:00:04', '2026-07-27 10:25', None),
            ('LFPG', 'UAL914', 2.0, '2026-07-27 12:00:21', '2026-07-27 10:25', '2026-07-27 10:25'),
            ('LFPG', 'AFR6T', 3.0, '2026-07-27 16:37:04', '2026-07-27 12:30', '2026-07-27 16:10'),
            ('LFPG', 'AFR6T', 3.0, '2026-07-27 16:46:59', '2026-07-27 12:30', '2026-07-27 16:10'),  # 10 min later
            ('LSZH', 'SCOUT11', np.nan, '2026-01-17 13:39:57', '2026-01-17 13:20', None),
            ('LSZH', 'SCOUT11', 4.0, '2026-01-17 13:40:05', '2026-01-17 13:20', '2026-01-17 13:20'),
            ('LSZH', 'SCOUT11', np.nan, '2026-01-17 13:40:08', '2026-01-17 13:20', None),  # three records
            ('EGLL', 'BAW1', 5.0, '2026-07-27 12:00:10', '2026-07-27 11:40', '2026-07-27 11:40'),
            ('EGLL', 'BAW2', np.nan, '2026-07-27 12:00:20', '2026-07-27 11:40', None),     # other flight number
        ])

    def test_only_the_id_row_of_a_two_record_pair(self):
        mask = rf.twin_rows(self.dep)
        self.assertEqual(mask.tolist(), [False, True, False, False, False, False, False, False, False])

    def test_prediction_is_takeoff_minus_lobt_minus_offset(self):
        pred = np.full(len(self.dep), 1000.0)
        out = rf.twin_records(pred, self.dep)
        self.assertEqual(out[1], 5721.0 - rf.TWIN_OFFSET_SEC)
        self.assertTrue((np.delete(out, 1) == 1000.0).all())

    def test_mask_from_a_larger_set(self):
        rows = [1, 2]
        mask = rf.twin_rows(self.dep)[rows]
        out = rf.twin_records(np.full(2, 1000.0), self.dep.iloc[rows].reset_index(drop=True), mask=mask)
        self.assertEqual(out.tolist(), [5721.0 - rf.TWIN_OFFSET_SEC, 1000.0])

    def test_input_order_does_not_matter(self):
        order = [8, 3, 1, 6, 0, 2, 5, 7, 4]
        mask = rf.twin_rows(self.dep.iloc[order].reset_index(drop=True))
        self.assertEqual([order[i] for i in np.flatnonzero(mask)], [1])


if __name__ == '__main__':
    unittest.main()
