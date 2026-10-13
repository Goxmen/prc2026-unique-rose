"""Tiny structural/input-isolation checks; never prepares a real fold or fits."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "VALIDATION"), str(ROOT / "pipeline"), str(ROOT / "VALIDATION/tools")]
import final_month_inputs as inputs


class FinalMonthInputContract(unittest.TestCase):
    def test_exact_cyclic_one_month_embargo(self):
        for month, train in inputs.FOLDS.items():
            number = int(month[-2:])
            excluded = {number, (number - 2) % 12 + 1, number % 12 + 1}
            self.assertEqual(train, [f"2025-{m:02d}" for m in range(1, 13) if m not in excluded])
            self.assertEqual(len(train), 9)

    def test_hidden_target_poison_cannot_reach_feature_apis(self):
        frame = pd.DataFrame(dict(mvt_id=[2, 1], taxi_sec=[120., 600.],
            TAXITIME_SEC_mvt=[120., 600.], y=[120., 600.],
            BLOCK_TIME_UTC_mvt=pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:01Z"]),
            MVT_TIME_UTC_mvt=pd.to_datetime(["2025-01-01T00:02Z", "2025-01-01T00:11Z"]),
            AOBT_3_flt=pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:01Z"])))
        original = frame.copy(deep=True)
        poisoned = frame.assign(taxi_sec=-1e30, TAXITIME_SEC_mvt=1e30, y=42.,
                                BLOCK_TIME_UTC_mvt=pd.Timestamp("2099-01-01", tz="UTC"))
        A, B = inputs.observable_held(frame), inputs.observable_held(poisoned)
        pd.testing.assert_frame_equal(A, B, check_exact=True)
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
        pd.testing.assert_series_equal(A.AOBT_3_flt, original.AOBT_3_flt)
        self.assertTrue(A[["taxi_sec", "TAXITIME_SEC_mvt", "y", "BLOCK_TIME_UTC_mvt"]].isna().all().all())

    def test_adsb_neighbour_scope_excludes_unpermitted_movement(self):
        table = pd.DataFrame(dict(mvt_id=[1, 3, 2], airport=["EDDF"] * 3,
            to_ms=[0, 900000, 1800000], ab_seg_sec=[100., 100000., 300.],
            ab_matched=[1., 1., 1.], ab_stand_exit_sec=[200., 200000., 400.],
            ab_exit_gap_sec=[20., 20., 20.])).set_index("mvt_id")
        original = table.copy(deep=True)
        context = pd.DataFrame(dict(mvt_id=[2, 1], ADEP_mvt=["EDDF"] * 2,
            MVT_TIME_UTC_mvt=pd.to_datetime(["2025-01-01T00:30Z", "2025-01-01T00:00Z"])))
        scoped = inputs._scope_adsb(table, context)
        self.assertEqual(set(scoped.index), {1, 2})
        self.assertEqual(scoped.loc[2, "ab_nb_seg_mean_60"], 100.)
        self.assertEqual(scoped.loc[2, "ab_nb_exit_mean_60"], 200.)
        self.assertEqual(scoped.loc[2, "ab_nb_seg_n_60"], 1.)
        self.assertTrue(np.isnan(scoped.loc[1, "ab_nb_seg_mean_60"]))
        pd.testing.assert_frame_equal(table, original, check_exact=True)


if __name__ == "__main__":
    unittest.main()
