"""Leakage boundaries, temporal semantics and aircraft identity ambiguity."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aircraft_rotation as ar


def events(rows):
    frame = pd.DataFrame(rows, columns=['mvt_id','phase','t','hex','stand','inblock','taxiin'])
    for c in ['t','inblock']:
        frame[c] = pd.to_datetime(frame[c], utc=True)
    frame['arrival_schedule'] = frame.inblock - pd.Timedelta(minutes=5)
    frame['airport'] = 'EDDF'
    frame['month'] = frame.t.dt.strftime('%Y-%m')
    frame['cs'] = 'TEST'
    frame['no_id'] = False
    return frame


class AircraftRotationTests(unittest.TestCase):
    def test_previous_aircraft_is_distinct_from_previous_stand(self):
        frame = events([
            [1,'DEP','2025-01-10T12:00Z',100,'A',None,None],
            [2,'ARR','2025-01-10T09:00Z',100,'A','2025-01-10T09:10Z',600],
            [3,'ARR','2025-01-10T11:00Z',200,'A','2025-01-10T11:10Z',600],
            [4,'DEP','2025-01-10T08:00Z',100,'B',None,None]])
        got = ar.feature_table(frame, [1], ['2025-01']).loc[1]
        self.assertEqual(got.ar_aircraft, '000064')
        self.assertEqual(got.ar_prev_inblock_age_sec, 10200)
        self.assertEqual(got.ar_prev_same_stand, 1)
        self.assertEqual(got.ar_prev_minus_stand_age_sec, 7200)
        self.assertEqual(got.ar_stand_same_aircraft, 0)
        self.assertEqual(got.ar_departures_24h, 1)
        self.assertEqual(got.ar_prev_arrival_delay_sec, 300)

    def test_excluded_month_is_not_a_reference(self):
        frame = events([
            [1,'DEP','2025-04-01T01:00Z',100,'A',None,None],
            [2,'ARR','2025-03-31T23:00Z',100,'A','2025-03-31T23:10Z',600],
            [3,'DEP','2025-03-31T20:00Z',100,'B',None,None]])
        train = ar.feature_table(frame, [1], ['2025-04']).loc[1]
        self.assertTrue(np.isnan(train.ar_prev_inblock_age_sec))
        self.assertTrue(np.isnan(train.ar_stand_same_aircraft))
        self.assertTrue(np.isnan(train.ar_departures_24h))
        held = ar.feature_table(frame, [1], ['2025-03','2025-04']).loc[1]
        self.assertEqual(held.ar_prev_inblock_age_sec, 6600)
        self.assertEqual(held.ar_departures_24h, 1)

    def test_future_inblock_and_over_age_are_missing(self):
        frame = events([
            [1,'DEP','2025-01-10T12:00Z',100,'A',None,None],
            [2,'ARR','2025-01-10T11:55Z',100,'A','2025-01-10T12:05Z',600],
            [3,'DEP','2025-01-15T12:00Z',100,'A',None,None],
            [4,'DEP','2025-01-10T12:00Z',None,'A',None,None]])
        got = ar.feature_table(frame, [1,3,4], ['2025-01'])
        self.assertTrue(got.ar_prev_inblock_age_sec.isna().all())
        self.assertEqual(got.loc[1,'ar_aircraft'], '000064')
        self.assertEqual(got.loc[4,'ar_aircraft'], '<MISSING>')
        self.assertTrue(np.isnan(got.loc[4,'ar_departures_24h']))

    def test_unique_stable_callsign_and_position_are_required(self):
        t = pd.Timestamp('2025-01-10T12:00Z')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'positions').mkdir(); (path/'callsigns').mkdir()
            positions = pd.DataFrame({'ts':[t,t,t], 'airport':['EDDF']*3, 'hex':[100,200,300]})
            positions.to_parquet(path/'positions/2025-01-10.parquet')
            calls = pd.DataFrame({'hex':[100,200,300], 'callsign':['TEST','TEST','UNSTABLE'],
                'records':[3,3,2], 'first_ms':[t.value//1_000_000]*3,
                'last_ms':[t.value//1_000_000]*3})
            calls.to_parquet(path/'callsigns/2025-01-10.parquet')
            con = ar.connection()
            movements = pd.DataFrame({'mvt_id':[1,2], 'airport':['EDDF']*2,
                                      't':[t,t], 'cs':['TEST','UNSTABLE']})
            con.register('movements',movements)
            self.assertTrue(ar.match_day(con,'2025-01-10',path).empty)
            calls.loc[1,'callsign'] = 'DIFFERENT'
            calls.to_parquet(path/'callsigns/2025-01-10.parquet')
            got = ar.match_day(con,'2025-01-10',path)
            self.assertEqual(got.mvt_id.tolist(),[1])
            self.assertEqual(got.hex.tolist(),[100])
            con.close()


if __name__ == '__main__':
    unittest.main()
