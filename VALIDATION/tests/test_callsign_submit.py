"""Bounded production checks: isolated timestamp views and allowed arrival scope."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import callsign_submit_features as f


def events(rows):
    frame = pd.DataFrame(rows, columns=['mvt_id', 'phase', 'airport', 't', 'inblock', 'hex'])
    frame['t'] = pd.to_datetime(frame.t, utc=True)
    frame['inblock'] = pd.to_datetime(frame.inblock, utc=True)
    frame['cs'] = 'ABC1'
    frame['stand'] = '1'
    frame['no_id'] = False
    frame['taxiin'] = np.where(frame.phase.eq('ARR'), 60., np.nan)
    frame['arrival_schedule'] = frame.inblock
    frame['month'] = frame.t.dt.strftime('%Y-%m')
    return frame


def shared(query, starts):
    ms = query.t.astype('datetime64[ms, UTC]').astype('int64').to_numpy()
    starts = pd.to_datetime(starts, utc=True).as_unit('ms').astype('int64').to_numpy()
    return pd.DataFrame(dict(mvt_id=query.mvt_id.to_numpy(), to_ms=ms, call_start_ms=starts,
        cc_call_start_sec=(ms-starts)/1000, cc_squawk_start_sec=(ms-starts)/1000,
        cc_previous_call_end_sec=np.nan, cc_ambiguous=0., cc_censored=0.)).set_index('mvt_id')


def test_old_final_same_id_uses_own_clock():
    old = events([(7, 'DEP', 'LIRF', '2026-01-03T12:00Z', None, 100)])
    final = events([(7, 'DEP', 'LIRF', '2026-01-03T12:00:12Z', None, 100)])
    start = ['2026-01-03T11:45Z']
    a, _ = f.ranking_features(old, old, shared(old, start), ['2026-01'])
    b, _ = f.ranking_features(final, final, shared(final, start), ['2026-01'])
    assert a.mvt_id.tolist() == b.mvt_id.tolist() == [7]
    assert a.cc_call_start_sec.iloc[0] == 900
    assert b.cc_call_start_sec.iloc[0] == 912
    assert a.cc_before_own_arrival.isna().all() and b.cc_before_own_arrival.isna().all()


def test_reference_month_and_strict_arrival_boundaries():
    all_events = events([
        (1, 'ARR', 'LIRF', '2025-12-31T23:00Z', '2025-12-31T23:10Z', 100),
        (2, 'DEP', 'LIRF', '2026-01-01T00:00Z', None, 100),
        (3, 'ARR', 'LIRF', '2026-01-31T23:00Z', '2026-01-31T23:10Z', 200),
        (4, 'DEP', 'LIRF', '2026-02-01T00:00Z', None, 200),
        # Landing is prior, but inblock after the queried takeoff is unavailable.
        (5, 'ARR', 'LIRF', '2026-02-01T00:10Z', '2026-02-01T00:30Z', 300),
        (6, 'DEP', 'LIRF', '2026-02-01T00:20Z', None, 300)])
    ranking = all_events.loc[all_events.month.str.startswith('2026')]
    query = ranking.loc[ranking.phase.eq('DEP')]
    clock = shared(query, ['2025-12-31T23:05Z', '2026-01-31T23:05Z', '2026-02-01T00:00Z'])
    result, refs = f.ranking_features(all_events, ranking, clock, ['2026-01', '2026-02'])
    assert refs == {m: f.TRAIN_MONTHS+[m] for m in ['2026-01', '2026-02']}
    result = result.set_index('mvt_id')
    assert result.loc[2, 'cc_before_own_arrival'] == 1.
    assert np.isnan(result.loc[4, 'cc_before_own_arrival'])
    assert np.isnan(result.loc[6, 'cc_before_own_arrival'])


def test_cross_year_conflict_cannot_mutate_training_cache():
    train = events([(1, 'DEP', 'LIRF', '2025-12-31T23:59:30Z', None, 100)])
    rank = events([(2, 'DEP', 'LIRF', '2026-01-01T00:00:30Z', None, 100)])
    before = train.copy(deep=True)
    identified = pd.DataFrame(dict(mvt_id=[2], hex=[100], near_ms=[0.], position_n=[3]))
    con = f.rotation.connection()
    try:
        con.register('movements', rank.drop(columns=['hex', 'month']))
        with unittest.TestCase().assertRaisesRegex(ValueError, 'frozen2025'):
            f.clean_rank_identities(con, identified, train)
    finally:
        con.close()
    pd.testing.assert_frame_equal(train, before)


def test_within_ranking_identity_conflict_rejected():
    train = events([(1, 'DEP', 'LIRF', '2025-12-01T12:00Z', None, 200)])
    rank = events([(2, 'DEP', 'LIRF', '2026-01-01T00:00Z', None, 100),
                   (3, 'DEP', 'LIRF', '2026-01-01T00:01Z', None, 100)])
    identified = pd.DataFrame(dict(mvt_id=[2, 3], hex=[100, 100], near_ms=[0., 0.], position_n=[3, 3]))
    con = f.rotation.connection()
    try:
        con.register('movements', rank.drop(columns=['hex', 'month']))
        result, n = f.clean_rank_identities(con, identified, train)
    finally:
        con.close()
    assert n == 2 and result.hex.isna().all() and set(result.mvt_id) == {2, 3}


def test_parent_replay_checks_float_and_rounded_template():
    import pyarrow as pa
    import callsign_submit as runner
    template = pa.table({'MVT_ID_mvt': pa.array([8, 7], type=pa.int64()),
                         'TAXITIME_SEC_mvt': pa.array([0, 0], type=pa.int32()),
                         'context': ['keep-eight', 'keep-seven']})
    frame = pd.DataFrame({'mvt_id': [7, 8], 'pred': [10.4, 11.6]})
    expected = template.set_column(1, template.schema.field(1), pa.array([12, 10], type=pa.int32()))
    runner.assert_parent_output(frame, template, expected, frame.copy())
    # A float change can leave rounded output identical; it must still fail.
    changed = frame.copy()
    changed.loc[0, 'pred'] = 10.45
    with unittest.TestCase().assertRaises(AssertionError):
        runner.assert_parent_output(changed, template, expected, frame)
    corrupted = expected.set_column(2, expected.schema.field(2), pa.array(['changed', 'keep-seven']))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'column differs'):
        runner.assert_parent_output(frame, template, corrupted, frame)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().copy().items()
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
