"""Checks for the OPDI ground-track inputs."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import opdi

T0 = pd.Timestamp('2025-07-01 10:00', tz='UTC')
BIG = 2 ** 63 + 5            # stored as UBIGINT in an old flight list, as BIGINT -2**63+5 in events


def movements(rows):
    """(mvt_id, phase, takeoff minutes after T0, stand, callsign, flight number)."""
    frame = pd.DataFrame(rows, columns=['MVT_ID_mvt', 'PHASE_mvt', 'minutes', 'STAND_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt'])
    frame['ADEP_mvt'] = 'EDDF'
    frame['MVT_TIME_UTC_mvt'] = T0 + pd.to_timedelta(frame.pop('minutes'), unit='min')
    return frame


class OpdiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / 'events').mkdir()
        (root / 'flight_list').mkdir()
        movements([
            (1.0, 'DEP', 20, 'V155', 'DLH1', 'LH1'),     # seen leaving its own stand
            (2.0, 'DEP', 30, 'A10', None, 'EZY2'),       # no callsign: matched on the flight number
            (3.0, 'DEP', 40, 'B20', 'AFR3', 'AF3'),      # no OPDI flight
            (4.0, 'ARR', 25, 'V155', 'DLH9', 'LH9'),
        ]).to_parquet(root / 'training_2025-07-01.parquet')
        movements([(5.0, 'DEP', 50, 'C1', 'BAW5', 'BA5')]).to_parquet(root / 'ranking.parquet')
        flights = pd.DataFrame(dict(id=np.array([11, BIG], dtype='uint64'), flt_id=['DLH1  ', 'EZY2'], adep='EDDF',
                                    first_seen=[T0.tz_localize(None) + pd.Timedelta(minutes=2),
                                                T0.tz_localize(None) + pd.Timedelta(minutes=10)],
                                    last_seen=[T0.tz_localize(None) + pd.Timedelta(minutes=90)] * 2))
        flights.to_parquet(root / 'flight_list' / 'flight_list_202507.parquet')
        pd.DataFrame(dict(id=np.array([-7], dtype='int64'), flt_id=['BAW5'], adep='EDDF',
                          first_seen=[T0.tz_localize(None) + pd.Timedelta(minutes=200)],
                          last_seen=[T0.tz_localize(None) + pd.Timedelta(minutes=300)])
                     ).to_parquet(root / 'flight_list' / 'flight_list_202601.parquet')
        at = lambda m: (T0 + pd.Timedelta(minutes=m)).tz_localize(None)
        big_signed = BIG - 2 ** 64
        pd.DataFrame([
            (11, 'first_seen', at(2), 'EDDF', None),
            (11, 'exit-parking_position', at(4), 'EDDF', 'V 155'),   # own stand, spaces differ
            (11, 'exit-parking_position', at(6), 'EDDF', 'V160'),    # passing another stand
            (11, 'entry-taxiway', at(7), 'EDDF', 'N7'),
            (11, 'entry-deicing_pad', at(10), 'EDDF', 'D1'),
            (11, 'exit-deicing_pad', at(15), 'EDDF', 'D1'),
            (11, 'entry-runway', at(18), 'EDDF', '25C'),
            (11, 'exit-parking_position', at(25), 'EDDF', 'V155'),   # after takeoff: ignored
            (big_signed, 'first_seen', at(10), 'EDDF', None),
            (big_signed, 'entry-taxiway', at(22), 'EDDF', 'L'),
        ], columns=['flight_id', 'type', 'event_time', 'apt', 'osm_ref']).astype({'flight_id': 'int64'}).to_parquet(
            root / 'events' / 'flight_events_20250624_20250704.parquet')
        self.paths = dict(TRAIN_GLOB=root / 'training_*.parquet', RANKING=root / 'ranking.parquet',
                          LIST_GLOB=root / 'flight_list' / 'flight_list_*.parquet',
                          EVENTS_GLOB=root / 'events' / 'flight_events_*.parquet')
        self.saved = {name: getattr(opdi, name) for name in self.paths}
        for name, path in self.paths.items():
            setattr(opdi, name, path)
        opdi.build_features(path=root / 'features.parquet')
        self.table = pd.read_parquet(root / 'features.parquet').set_index('mvt_id')

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(opdi, name, value)
        self.tmp.cleanup()

    def test_one_row_per_departure_and_arrivals_ignored(self):
        self.assertEqual(sorted(self.table.index), [1.0, 2.0, 3.0, 5.0])

    def test_own_stand_first_stand_and_events_after_takeoff(self):
        row = self.table.loc[1.0]
        self.assertEqual(row.op_matched, 1)
        self.assertEqual(row.op_own_stand_sec, 16 * 60)       # left V155 at 10:04, took off 10:20
        self.assertEqual(row.op_first_stand_sec, 16 * 60)
        self.assertEqual(row.op_first_twy_sec, 13 * 60)
        self.assertEqual(row.op_rwy_sec, 2 * 60)
        self.assertEqual(row.op_deice_sec, 5 * 60)
        self.assertEqual(row.op_fs_sec, 18 * 60)
        self.assertEqual(row.op_ground_sec, 18 * 60)          # earliest ground sighting: first seen

    def test_flight_number_fallback_and_unsigned_ids(self):
        row = self.table.loc[2.0]
        self.assertEqual(row.op_matched, 1)
        self.assertEqual(row.op_fs_sec, 20 * 60)
        self.assertEqual(row.op_first_twy_sec, 8 * 60)
        self.assertEqual(row.op_deice_sec, 0)

    def test_unmatched_rows_are_missing_not_zero(self):
        row = self.table.loc[3.0]
        self.assertEqual(row.op_matched, 0)
        self.assertTrue(np.isnan(row.op_ground_sec) and np.isnan(row.op_deice_sec) and np.isnan(row.op_fs_ground))
        self.assertEqual(self.table.loc[5.0].op_matched, 0)   # its OPDI flight starts after its takeoff

    def test_neighbours_exclude_the_row_itself(self):
        # Row 3 took off 10:40; in the hour before it rows 1 (18 min) and 2 (20 min) were seen moving.
        self.assertEqual(self.table.loc[3.0].nb_op_ground_mean_60, 19 * 60)
        self.assertEqual(self.table.loc[3.0].nb_op_ground_n_60, 2)
        # Within 30 min either side of row 1 only row 2 counts, not row 1 itself.
        self.assertEqual(self.table.loc[1.0].nb_op_ground_mean_pm30, 20 * 60)
        self.assertEqual(self.table.loc[3.0].nb_op_deice_share_60, 0.5)

    def test_add_opdi_aligns_by_movement_id(self):
        raw = pd.DataFrame({'mvt_id': [3.0, 1.0]}, index=[7, 8])
        X = pd.DataFrame({'a': [0, 1]}, index=[7, 8])
        out = opdi.add_opdi(X, raw, self.table)
        self.assertEqual(out.loc[8, 'op_own_stand_sec'], 16 * 60)
        self.assertEqual(out.loc[7, 'op_matched'], 0)
        with self.assertRaises(ValueError):
            opdi.add_opdi(X, pd.DataFrame({'mvt_id': [3.0, 99.0]}, index=[7, 8]), self.table)
        with self.assertRaises(ValueError):
            opdi.add_opdi(out, raw, self.table)


if __name__ == '__main__':
    unittest.main()
