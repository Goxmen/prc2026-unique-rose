"""Boundary, sparse-history and identity/mask tests without data scans."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_track_sequence as s


def query(mvt_id=1, **changes):
    values = dict(mvt_id=mvt_id, airport='TEST', stand='A', runway='01',
        to_ms=int(pd.Timestamp('2025-01-10T00:01:00Z').timestamp()*1000),
        clock_ms=np.nan, hex=123, day='2025-01-10',
        stand_lat=np.nan, stand_lon=np.nan, runway_lat=50., runway_lon=4.,
        origin_lat=50., origin_lon=4.)
    values.update(changes)
    return values


def positions(times, **changes):
    frame = pd.DataFrame(dict(airport='TEST', hex=123, ts_ms=np.asarray(times, dtype=np.int64),
                              lat=50000000, lon=4000000, alt=-123, gs=50))
    for key, value in changes.items():
        frame[key] = value
    return frame


def test_inclusive_time_window_midnight_and_identity():
    q = query()
    t = q['to_ms']
    p = positions([t-7200001, t-7200000, t, t+60000, t+60001])
    qs = pd.DataFrame([q, query(2, hex=np.nan), query(3, airport='OTHER'), query(4, hex=456)])
    ix, raw = s.select_sequences(qs, p, {'2025-01-09', '2025-01-10'})
    assert ix['count'].tolist() == [3, 0, 0, 0]
    assert raw[:, 0].tolist() == [7200000, 0, -60000]
    assert not ix.left_source_missing.any() and not ix.right_source_missing.any()
    ix2, _ = s.select_sequences(pd.DataFrame([q]), p, {'2025-01-10'})
    assert ix2.left_source_missing.iloc[0] == 1


def test_sparse_points_order_duplicates_and_raw_gap():
    q = query()
    t = q['to_ms']
    p = positions([t, t-500000, t-500000], lat=[50000010, 50000002, 50000001])
    ix, raw = s.select_sequences(pd.DataFrame([q]), p, {'2025-01-09', '2025-01-10'})
    assert ix['count'].iloc[0] == 2
    assert raw[:, 1].tolist() == [50000001, 50000010]
    assert raw[:, 5].tolist() == [-1, 500000]
    ix2, raw2 = s.select_sequences(pd.DataFrame([q]), p.sample(frac=1, random_state=6), {'2025-01-09', '2025-01-10'})
    np.testing.assert_array_equal(raw, raw2)
    pd.testing.assert_frame_equal(ix, ix2)


def test_dense_first_last_and_previous_raw_gap_preserved():
    q = query()
    times = q['to_ms']-np.arange(1000, -1, -1)*1000
    ix, raw = s.select_sequences(pd.DataFrame([q]), positions(times), {'2025-01-09', '2025-01-10'})
    assert ix['raw_count'].iloc[0] == 1001 and ix['count'].iloc[0] == 128
    assert raw[0, 0] == 1000000 and raw[-1, 0] == 0
    assert np.all(np.diff(raw[:, 0]) < 0)
    assert raw[0, 5] == -1 and np.all(raw[1:, 5] == 1000)


def test_missing_masks_negative_raw_codes_and_units():
    q = query()
    p = positions([q['to_ms']-1000, q['to_ms']], alt=[-123, -5], gs=[50, -1])
    ix, raw = s.select_sequences(pd.DataFrame([q]), p, {'2025-01-09', '2025-01-10'})
    x, mask, quality = s.encoded_batch(ix, raw)
    assert mask.sum() == 2 and not x[:, 2:].any()
    col = {name: i for i, name in enumerate(s.CHANNEL_NAMES)}
    np.testing.assert_array_equal(x[0, :2, col['speed_known']], [1, 0])
    np.testing.assert_array_equal(x[0, :2, col['altitude_known']], [1, 0])
    np.testing.assert_array_equal(x[0, :2, col['ground']], [1, 0])
    assert np.isclose(x[0, 0, col['speed']], 5/300)
    assert not x[0, :, col['stand_known']].any()
    assert not x[0, :, col['actual_clock_known']].any()
    assert not x[0, :, col['actual_clock_offset']].any()
    assert np.isfinite(x).all() and np.isfinite(quality).all()


def test_actual_clock_transform_spatial_reference_and_bounds():
    q = query(clock_ms=1, stand_lat=50., stand_lon=4.)
    p = positions([q['to_ms']-1000, q['to_ms']], lat=[50010000, 50000000],
                  alt=[400, 1000], gs=[1000, 32000])
    ix, raw = s.select_sequences(pd.DataFrame([q]), p, {'2025-01-09', '2025-01-10'})
    x, _, _ = s.encoded_batch(ix, raw)
    col = {name: i for i, name in enumerate(s.CHANNEL_NAMES)}
    assert np.isclose(x[0, 0, col['north']], 1113.2/20000)
    assert np.isclose(x[0, 0, col['stand_distance']], 1113.2/40000)
    assert x[0, 1, col['stand_distance']] == 0
    assert np.all(x[0, :2, col['altitude']] == 1)
    assert x[0, 1, col['speed']] == 1
    assert np.all(x[0, :2, col['actual_clock_offset']] == 1)


def test_actual_clock_offset_sign_and_seconds():
    q = query()
    q['clock_ms'] = q['to_ms']-1800000
    p = positions([q['clock_ms']-1800000, q['clock_ms'], q['clock_ms']+1800000])
    ix, raw = s.select_sequences(pd.DataFrame([q]), p, {'2025-01-09', '2025-01-10'})
    x, _, _ = s.encoded_batch(ix, raw)
    actual = x[0, :3, s.CHANNEL_NAMES.index('actual_clock_offset')]
    np.testing.assert_allclose(actual, np.arcsinh([-1., 0., 1.])/8, atol=1e-8)


def test_timestamp_resolution_and_no_label_dependence():
    q = query()
    p = positions([q['to_ms']-1000, q['to_ms']])
    p['ts'] = pd.to_datetime(p.ts_ms, unit='ms', utc=True)
    p = p.drop(columns='ts_ms')
    first = s.canonical_positions(p)
    assert first.ts_ms.tolist() == [q['to_ms']-1000, q['to_ms']]
    base, values = s.select_sequences(pd.DataFrame([q]), p, {'2025-01-09', '2025-01-10'})
    # Unselected query columns must never affect the numeric model transforms.
    poisoned = dict(q, taxi_sec=1e20, BLOCK_TIME_UTC_mvt='2099-01-01')
    alternate, alt_values = s.select_sequences(pd.DataFrame([poisoned]), p, {'2025-01-09', '2025-01-10'})
    for left, right in zip(s.encoded_batch(base, values), s.encoded_batch(alternate, alt_values)):
        np.testing.assert_array_equal(left, right)


def test_prototype_initial_parameters_predictions_rng_and_fallback():
    import torch
    import nn_sparse_track_prototype as p
    torch.set_num_threads(2)
    result = p.self_check([4, 8], [torch.linspace(-3, 3, 8), torch.linspace(-2, 2, 6)], 5)
    assert result['initial_train_predictions_exact'] and result['dropout_rng_exact']


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
