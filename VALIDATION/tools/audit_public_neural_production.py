"""Reload public production models from the saved encoder and raw track cache."""
import gc
import os
from pathlib import Path
import pickle
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import nn_public_submit as run
from records import write_json


def audit():
    assert not (run.OUT / 'audit.json').exists(), 'Retain completed audit'
    m = run.guard()
    assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080'
    records = {k: run.completed(k) for k in range(3)}
    assert all(records.values()), 'All production members must be complete'
    parent = m['parent_packet']
    packet = run.old.packets.load_packet('old', parent['packet'])
    ids = np.asarray(packet['rank_ids'])
    folder = ROOT / parent['tracks']['old']['directory']
    index = pd.read_parquet(folder / 'full_index.parquet').set_index('mvt_id', drop=False).loc[ids]
    points = np.memmap(folder / 'full_points.bin', mode='r', dtype=np.int32).reshape(-1, 6)
    positions = [np.linspace(0, len(ids)-1, 512, dtype=int)]
    for mask in [index['count'].eq(0), index['count'].eq(1), index['count'].between(2, 31),
                 index['count'].ge(32), index.raw_ground_count.gt(0), index.clock_ms.isna()]:
        positions.append(np.flatnonzero(mask)[:8])
    for airport in sorted(index.airport.unique()):
        positions.append(np.flatnonzero(index.airport.eq(airport))[:8])
    at = np.unique(np.concatenate(positions))
    selected = index.iloc[at].reset_index(drop=True)
    chosen = ids[at]
    first = run.trained(0)
    with run.verify(first['preprocessing']).open('rb') as handle:
        pre = pickle.load(handle)
    assert pre['design'] == run.DESIGN
    X = packet['Vn'].iloc[at][pre['columns']].copy()
    del packet
    gc.collect()
    cat, num = pre['encoder'].transform(X)
    public = pd.read_parquet(run.verify(m['ranking_features'])).set_index('mvt_id').loc[chosen]
    extra = run.model_type.transform(public)
    # Recalculate the fixed public transform without calling its implementation.
    reference = np.nan_to_num(np.clip(np.arcsinh(
        public[run.model_type.features.FEATURES].to_numpy(float) /
        np.asarray(run.model_type.public.SCALES))/4, -5, 5), nan=0, posinf=5, neginf=-5).astype('float32')
    np.testing.assert_array_equal(extra, reference)
    values = [cat, num[:, pre['ple']], np.concatenate([num[:, pre['plain']], reference], axis=1)]
    tensors = [torch.from_numpy(np.ascontiguousarray(v)).cuda() for v in values]
    tracks = run.cv.single.base.tensor_seq.RaggedTracks(selected, points, 'cuda')
    track = tracks.batch(np.arange(len(chosen)))
    scalar = run.cv.single.base.seq.encoded_batch(selected.iloc[:32], points)
    for tensor, expected in zip(track, scalar):
        np.testing.assert_allclose(tensor[:32].cpu().numpy(), expected, rtol=2e-6, atol=2e-7)
    checks = []
    for k, record in records.items():
        trained = run.trained(k)
        assert trained['preprocessing'] == first['preprocessing']
        model = run.model_type.SparseFusion(pre['cards'], pre['bins'], len(pre['plain'])+12).cuda().eval()
        checkpoint = torch.load(run.verify(trained['model']), map_location='cuda', weights_only=False)
        assert checkpoint['design'] == run.DESIGN and checkpoint['config'] == parent['config']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        with torch.no_grad():
            standardized = model(*tensors, *track)[:, 0].cpu().numpy().astype(float)
        predicted = np.maximum(standardized*pre['sd']+pre['mu'], 0)
        saved = pd.read_parquet(run.verify(record['prediction']))
        np.testing.assert_array_equal(saved.mvt_id, ids)
        assert np.isfinite(saved.base).all() and saved.base.ge(0).all()
        expected = saved.base.to_numpy(float)[at]
        np.testing.assert_allclose(predicted, expected, rtol=0, atol=.01)
        checks.append(dict(member=k, seed=run.SEEDS[k], rows=len(at),
            max_difference_seconds=float(np.max(abs(predicted-expected))),
            record=run.art(run.paths(k)[2]), production_reload_all_rows=record['all_ranking_reload_exact']))
        del model, checkpoint, saved
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(status='pass', source=run.art(__file__), identity=run.art(run.MANIFEST),
        checks=checks, public_transform_values=int(reference.size), scalar_track_rows=32,
        tolerance_seconds=.01,
        scope='Independent input preparation on input-selected rows; every ranking prediction also reloaded during production')
    write_json(run.OUT / 'audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        audit()
