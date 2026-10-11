"""Check time-pooling weights against raw integer timestamps, without labels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import nn_public_duration_pool as model
import sparse_track_sequence as seq
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261011_nn_public_duration_pool'


def audit():
    assert not (OUT / 'input_pooling_audit.json').exists(), 'Preserve completed audit'
    index = pd.read_parquet(seq.OUT / 'full_index.parquet')
    scopes = [index['count'].eq(0), index['count'].eq(1), index['count'].eq(2),
              index['count'].between(3, 127), index['count'].eq(128)]
    positions = np.unique(np.concatenate([np.linspace(0, len(index)-1, 2048, dtype=int),
                                          *[np.flatnonzero(mask)[:12] for mask in scopes]]))
    chosen = index.iloc[positions].reset_index(drop=True)
    points = np.memmap(seq.OUT / 'full_points.bin', dtype=np.int32, mode='r').reshape(-1, 6)
    encoded, mask, _ = seq.encoded_batch(chosen, points)
    actual = model.interval_weights(torch.from_numpy(encoded), torch.from_numpy(mask)).numpy()
    expected = np.zeros_like(actual, dtype=float)
    for i, row in enumerate(chosen.itertuples()):
        raw = points[row.start:row.start+row.count]
        for j in range(max(0, row.count-1)):
            gap = (int(raw[j, 0])-int(raw[j+1, 0]))/1000
            assert gap > 0
            value = min(gap, 60)/2
            expected[i, j] += value
            expected[i, j+1] += value
    tolerance = 2*np.finfo(np.float32).eps*7260
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    assert np.isfinite(actual).all() and (actual >= 0).all() and not actual[~mask].any()
    totals = actual.sum(1)
    valid = totals > 0
    normalized = actual[valid]/totals[valid, None]
    uniform = mask[valid]/mask[valid].sum(1)[:, None]
    l1 = np.abs(normalized-uniform).sum(1)
    result = dict(status='pass', auditor=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha256(__file__)),
        source=dict(path='VALIDATION/nn_public_duration_pool.py', sha256=sha256(model.__file__)),
        cache_manifest=dict(path=str((seq.OUT / 'full.json').relative_to(ROOT)), sha256=sha256(seq.OUT / 'full.json')),
        sampled_rows=len(chosen),
        sampling='2048 evenly spaced full2025 IDs plus count0/1/2/3..127/128 boundary samples',
        raw_integer_interval_oracle=True, maximum_absolute_seconds_error=float(np.max(abs(actual-expected))),
        float32_time_transform_error_bound_seconds=float(tolerance), finite_nonnegative_padding_zero=True,
        usable_rows=int(valid.sum()), nonuniform_rows=int((l1 > 1e-5).sum()),
        l1_distance_from_observation_uniform_quantiles=np.quantile(l1, [0, .25, .5, .75, 1]).tolist(),
        labels_used=False, no_fits=True, no_validation_claim=True)
    write_json(OUT / 'input_pooling_audit.json', result)
    print({key: result[key] for key in ['status', 'sampled_rows', 'maximum_absolute_seconds_error', 'nonuniform_rows']})


if __name__ == '__main__':
    with threadpool_limits(1): audit()
