"""Leakage and arithmetic checks for the arrival-plan duration experiment."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arrival_plan_jan_jul as plan


def frame(month, duration, n=25):
    t = pd.Timestamp(month + '-01', tz='UTC')
    return pd.DataFrame(dict(ADEP_mvt=['AAA'] * n, ADES_mvt=['BBB'] * n,
        AIRCRAFT_TYPE_mvt=['X'] * n, month=[month] * n,
        MVT_TIME_UTC_mvt=[t + pd.Timedelta(minutes=20)] * n,
        EOBT_1_flt=[t] * n, AOBT_3_flt=[t + pd.Timedelta(minutes=5)] * n,
        ARVT_1_flt=[t + pd.Timedelta(seconds=duration)] * n,
        ARVT_3_flt=[t + pd.Timedelta(seconds=duration + 300)] * n))


def test_training_references_exclude_the_entire_query_month():
    train = pd.concat([frame('2025-01', 3600), frame('2025-02', 7200)], ignore_index=True)
    held = frame('2025-07', 6000, 1)
    a, b = plan.feature_pair(train, held)
    np.testing.assert_allclose(a.fp_block1_excess_sec.iloc[:25], -3600)
    np.testing.assert_allclose(a.fp_block1_excess_sec.iloc[25:], 3600)
    np.testing.assert_allclose(b.fp_block1_excess_sec, 600)
    np.testing.assert_allclose(b.fp_block_revision_sec, 0)
    assert list(a) == list(b) == plan.FEATURES


def test_unseen_route_uses_airport_and_unknown_airport_stays_missing():
    train = frame('2025-01', 3600)
    held = frame('2025-07', 6000, 2)
    held.loc[0, 'ADES_mvt'] = 'UNSEEN'
    held.loc[1, 'ADEP_mvt'] = 'UNSEEN'
    result = plan.deviations(train, held, plan.durations(train), plan.durations(held))
    assert result.loc[0, 'fp_block1_excess_sec'] == 2400
    assert np.isnan(result.loc[1, 'fp_block1_excess_sec'])
    assert result.loc[1, 'fp_block1_sec'] == 6000


def test_missing_clock_is_not_filled_from_other_clock():
    query = frame('2025-07', 6000, 1)
    query.loc[0, 'ARVT_1_flt'] = pd.NaT
    result = plan.durations(query)
    assert np.isnan(result.loc[0, 'fp_block1_sec'])
    assert np.isnan(result.loc[0, 'fp_block_revision_sec'])
    assert result.loc[0, 'fp_block3_sec'] == 6000
    assert result.loc[0, 'fp_air3_sec'] == 5100


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in [
        test_training_references_exclude_the_entire_query_month,
        test_unseen_route_uses_airport_and_unknown_airport_stays_missing,
        test_missing_clock_is_not_filled_from_other_clock,
    ])


if __name__ == '__main__':
    unittest.main()
