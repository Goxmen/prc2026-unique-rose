"""Guard the new identity assignment and preservation of existing witnesses."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adsblol_kinematic as ak


class KinematicMatchingTests(unittest.TestCase):
    def setUp(self):
        self.ends = pd.DataFrame([
            dict(airport='TEST', runway='09', lat=0., lon=0.),
            dict(airport='TEST', runway='27', lat=0., lon=.03),
        ])
        self.tables = (pd.DataFrame(), self.ends, pd.DataFrame())
        self.calibration = dict(parameters=ak.PARAMETERS,
                                airports={'TEST': dict(offset_sec=-5)})

    def positions(self, aircraft=1, end=10000):
        return pd.DataFrame([
            dict(ts=end-10000, airport='TEST', hex=aircraft, lat=0, lon=5000,
                 alt=ak.ab.GROUND, gs=800),
            dict(ts=end, airport='TEST', hex=aircraft, lat=0, lon=6000,
                 alt=10, gs=850),
        ])

    def deps(self, ids=(1,), time=10000, cs='UNKNOWN'):
        return pd.DataFrame([dict(mvt_id=x, airport='TEST', to_ms=time, cs=cs,
                                   runway='09') for x in ids])

    def calls(self):
        return pd.DataFrame(columns=['hex', 'cs', 'first_ms', 'last_ms']).astype(
            {'hex': 'int64', 'first_ms': 'int64', 'last_ms': 'int64'})

    def test_unmatched_identity_can_be_recovered(self):
        recovered, original, *_ = ak.match_missing(self.deps(), self.positions(),
            self.calls(), self.calibration, self.tables)
        self.assertEqual(len(original), 0)
        self.assertEqual(recovered[['mvt_id', 'hex']].to_numpy().tolist(), [[1, 1]])

    def test_two_flights_cannot_claim_one_event(self):
        recovered, *_ = ak.match_missing(self.deps(ids=(1, 2)), self.positions(),
            self.calls(), self.calibration, self.tables)
        self.assertTrue(recovered.empty)

    def test_original_flight_reserves_the_event(self):
        calls = pd.DataFrame([dict(hex=1, cs='KNOWN', first_ms=0, last_ms=10000)])
        deps = pd.concat([self.deps(ids=(1,), cs='KNOWN'), self.deps(ids=(2,))])
        recovered, original, *_ = ak.match_missing(deps, self.positions(), calls,
                                                  self.calibration, self.tables)
        self.assertEqual(original.mvt_id.tolist(), [1])
        self.assertTrue(recovered.empty)

    def test_parallel_runway_is_rejected(self):
        positions = self.positions().assign(lat=5000)
        recovered, *_ = ak.match_missing(self.deps(), positions, self.calls(),
                                         self.calibration, self.tables)
        self.assertTrue(recovered.empty)

    def test_aircraft_near_reference_cannot_be_assigned_twice(self):
        calls = pd.DataFrame([dict(hex=1, cs='KNOWN', first_ms=0, last_ms=90000)])
        positions = pd.concat([self.positions(), self.positions(end=110000)])
        deps = pd.concat([self.deps(ids=(1,), cs='KNOWN'), self.deps(ids=(2,), time=110000)])
        recovered, original, *_ = ak.match_missing(deps, positions, calls,
                                                  self.calibration, self.tables)
        self.assertEqual(original.mvt_id.tolist(), [1])
        self.assertTrue(recovered.empty)

    def test_feature_overlay_never_changes_existing_ground_witness(self):
        X = pd.DataFrame({'ab_seg_sec': [100., np.nan], 'ab_matched': [1., 0.]})
        table = pd.DataFrame({'mvt_id': [1, 2], **{c: [999., 500.] for c in ak.SUMMARY_COLUMNS},
                              **{c: [1., 1.] for c in ak.CONFIDENCE_COLUMNS}})
        result = ak.add_features(X, [1, 2], table)
        self.assertEqual(result.loc[0, 'ab_seg_sec'], 100.)
        self.assertEqual(result.loc[0, 'ab_matched'], 1.)
        self.assertEqual(result.loc[0, 'ak_recovered'], 0.)
        self.assertEqual(result.loc[1, 'ab_seg_sec'], 500.)


if __name__ == '__main__':
    unittest.main()
