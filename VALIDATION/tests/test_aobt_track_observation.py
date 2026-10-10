import sys
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aobt_track_observation as ao


class AobtObservationTest(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect()
        self.con.execute("SET TimeZone='UTC'")
        self.clock = int(pd.Timestamp('2025-07-01T12:00:00Z').timestamp() * 1000)
        self.q = pd.DataFrame(dict(mvt_id=[1], airport=['TEST'], stand=['A'],
            to_ms=[self.clock+600000], clock_ms=[self.clock], hex=[123]))
        self.coords = pd.DataFrame(dict(airport=['TEST'], stand=['A'], lat=[50.], lon=[4.]))

    def tearDown(self):
        self.con.close()

    def positions(self, offsets, speed, altitude=None):
        return pd.DataFrame(dict(ts=pd.to_datetime(self.clock+np.asarray(offsets)*1000,
            unit='ms', utc=True), airport='TEST', hex=123, lat=50000000, lon=4000000,
            alt=altitude or [-123]*len(offsets), gs=speed))

    def test_nearest_tie_and_valid_speed_windows(self):
        pos = self.positions([-300, -20, 20, 100, 300], [-1, 0, 100, 200, 0],
                             [-123, -123, -123, 100, -123])
        row = ao.observe(self.con, self.q, pos, self.coords).iloc[0]
        self.assertEqual(row.ao_point_offset_sec, -20)
        self.assertEqual(row.ao_speed_kt, 0)
        self.assertEqual(row.ao_stand_distance_m, 0)
        self.assertEqual(row.ao_on_ground, 1)
        self.assertEqual(row.ao_moving_before, 0)
        self.assertEqual(row.ao_moving_after, .5)

    def test_unavailable_clock_identity_and_old_clock_stay_missing(self):
        queries = pd.concat([self.q.assign(mvt_id=1, clock_ms=np.nan),
            self.q.assign(mvt_id=2, hex=np.nan),
            self.q.assign(mvt_id=3, to_ms=self.clock+14401000)])
        out = ao.observe(self.con, queries, self.positions([0], [0]), self.coords)
        self.assertTrue(out[ao.FEATURES].isna().all().all())
        self.assertEqual(out.mvt_id.tolist(), [1, 2, 3])

    def test_cutoff_and_nearest_tolerance(self):
        q = self.q.assign(to_ms=self.clock)
        out = ao.observe(self.con, q, self.positions([-121, 61], [0, 100]), self.coords).iloc[0]
        self.assertTrue(np.isnan(out.ao_speed_kt))
        self.assertTrue(np.isnan(out.ao_point_offset_sec))
        self.assertEqual(out.ao_moving_before, 0)
        self.assertTrue(np.isnan(out.ao_moving_after))


if __name__ == '__main__':
    unittest.main()
