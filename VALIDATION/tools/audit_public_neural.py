"""Reload public-head neural models with retained input encoders and raw tracks."""
import argparse
import gc
import os
from pathlib import Path
import pickle
import sys
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'VALIDATION/tools')]
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import nn_public_scratch_jan_jul as single
from records import write_json


def audit(pool=False):
    single.DESIGN = dict(single.DESIGN, variant='public')
    identity = single.shared.ts.read(single.MANIFEST)
    single.verify(identity['source'])
    for source in identity['helpers']:
        single.verify(source)
    if pool:
        import nn_public_pool3 as run
        run.guard()
        destination = run.OUT
        members = [0, 1, 2]
        def record(month, k):
            return run.completed(month, k, identity)
    else:
        destination, members = single.OUT, [0]
        def record(month, k):
            return single.completed(month, identity)
    single.shared.recipe()
    index = pd.read_parquet(single.base.seq.OUT / 'full_index.parquet').set_index('mvt_id', drop=False)
    points = np.memmap(single.base.seq.OUT / 'full_points.bin', mode='r', dtype=np.int32).reshape(-1, 6)
    public = pd.read_parquet(single.verify(identity['public_features'])).set_index('mvt_id')
    checks = []
    for month in single.STOP:
        rec = record(month, 0)
        with single.verify(rec['preprocessing']).open('rb') as handle:
            pre = pickle.load(handle)
        data = pd.read_pickle(single.verify(identity['folds'][month]['matrix']))
        ids = np.asarray(data['ids'])
        scope = index.loc[ids]
        positions = np.linspace(0, len(ids)-1, 512, dtype=int)
        extra = [np.flatnonzero(mask)[:8] for mask in [scope['count'].eq(0), scope['count'].eq(1),
            scope['count'].between(2,31), scope['count'].ge(32), scope.raw_ground_count.gt(0), scope.clock_ms.isna()]]
        positions = np.unique(np.concatenate([positions, *extra]))
        chosen = ids[positions]
        X = data['V'][pre['columns']].iloc[positions].copy()
        del data, scope
        gc.collect()
        cat, num = pre['encoder'].transform(X)
        values = [cat, num[:, pre['ple']], np.concatenate([num[:, pre['plain']],
            single.airport_tune.transform(public.loc[chosen])], axis=1)]
        tensors = [torch.from_numpy(np.ascontiguousarray(v)).cuda() for v in values]
        selected_index = index.loc[chosen].reset_index(drop=True)
        tracks = single.base.tensor_seq.RaggedTracks(selected_index, points, 'cuda')
        track = tracks.batch(np.arange(len(chosen)))
        scalar = single.base.seq.encoded_batch(selected_index.iloc[:32], points)
        for tensor, expected in zip(track, scalar):
            np.testing.assert_allclose(tensor[:32].cpu().numpy(), expected, rtol=2e-6, atol=2e-7)
        for k in members:
            rec = record(month, k)
            assert rec is not None
            model = single.airport_tune.SparseFusion(pre['cards'], pre['bins'], len(pre['plain'])+12).cuda().eval()
            checkpoint = torch.load(single.verify(rec['model']), map_location='cuda', weights_only=False)
            assert checkpoint['design'] == (run.DESIGN if pool and k else single.DESIGN)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            with torch.no_grad():
                standardized = model(*tensors, *track)[:, 0].cpu().numpy().astype(float)
            prediction = np.maximum(standardized*pre['sd']+pre['mu'], 0)
            saved = pd.read_parquet(single.verify(rec['prediction'])).set_index('mvt_id').loc[chosen, 'base'].to_numpy(float)
            np.testing.assert_allclose(prediction, saved, rtol=0, atol=.01)
            checks.append(dict(month=month, seed=42+1000*k, rows=len(chosen), raw_scalar_rows=32,
                maximum_difference_seconds=float(np.max(np.abs(prediction-saved)))))
            del model, checkpoint
        del pre, X, tracks, track, tensors
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(status='pass', checks=checks, source=single.artifact(__file__),
        identity=single.artifact(destination / 'identity.json'), tolerance_seconds=.01,
        interpretation='Numerical reload audit; development months are reused')
    write_json(destination / 'audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        audit(args.pool)
