"""Saved-model reload and separate score arithmetic for the physical-scale screen."""
import gc
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'VALIDATION/tools'), str(ROOT / 'pipeline')]
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_track_physical_scales_jan_jul as runner
from records import write_json


def main():
    identity = runner.shared.ts.read(runner.MANIFEST)
    runner.verify(identity['source'])
    for source in identity['helpers']:
        runner.verify(source)
    runner.shared.recipe()
    score = runner.shared.ts.read(runner.OUT / 'score.json')
    frame = pd.read_parquet(runner.OUT / 'scored.parquet')
    if not frame.mvt_id.is_unique or len(frame) != 344419:
        raise ValueError('Held-out population differs')
    if not np.isfinite(frame[list(score['scores'])].to_numpy()).all():
        raise ValueError('Nonfinite row errors')
    for arm, expected in score['scores'].items():
        actual = frame.groupby('month')[arm].mean().pow(.5)
        for month, value in actual.items():
            np.testing.assert_allclose(value, expected[month], rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual.mean(), expected['cv'], rtol=0, atol=1e-10)
    index = pd.read_parquet(runner.base.seq.OUT / 'full_index.parquet').set_index('mvt_id', drop=False)
    points = np.memmap(runner.base.seq.OUT / 'full_points.bin', mode='r', dtype=np.int32).reshape(-1, 6)
    checks = []
    for month in runner.STOP:
        record = runner.completed(month, identity)
        if record is None:
            raise ValueError('Both models must be completed')
        with runner.verify(record['preprocessing']).open('rb') as handle:
            pre = pickle.load(handle)
        data = pd.read_pickle(runner.verify(identity['folds'][month]['matrix']))
        ids = np.asarray(data['ids'])
        positions = np.linspace(0, len(ids)-1, 512, dtype=int)
        scope = index.loc[ids]
        extra = [np.flatnonzero(mask)[:8] for mask in
                 [scope['count'].eq(0), scope['count'].eq(1),
                  scope['count'].between(2,31), scope['count'].ge(32),
                  scope.raw_ground_count.gt(0), scope.clock_ms.isna()]]
        positions = np.unique(np.concatenate([positions, *extra]))
        chosen = ids[positions]
        X = data['V'][pre['columns']].iloc[positions].copy()
        del data, scope
        gc.collect()
        cat, num = pre['encoder'].transform(X)
        tensors = [torch.from_numpy(np.ascontiguousarray(a)).cuda()
                   for a in [cat, num[:,pre['ple']], num[:,pre['plain']]]]
        selected_index = index.loc[chosen].reset_index(drop=True)
        tracks = runner.base.tensor_seq.RaggedTracks(selected_index, points, 'cuda')
        track = tracks.batch(np.arange(len(chosen)))
        scalar = runner.base.seq.encoded_batch(selected_index.iloc[:32], points)
        for tensor, reference in zip(track, scalar):
            np.testing.assert_allclose(tensor[:32].cpu().numpy(), reference, rtol=2e-6, atol=2e-7)
        model = runner.physical_scales.SparseFusion(pre['cards'], pre['bins'], len(pre['plain'])).cuda().eval()
        checkpoint = torch.load(runner.verify(record['model']), map_location='cuda', weights_only=False)
        if checkpoint['design'] != runner.DESIGN:
            raise ValueError('Checkpoint architecture differs')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        with torch.no_grad():
            standardized = model(*tensors, *track)[:,0].cpu().numpy().astype(float)
        prediction = np.maximum(standardized * pre['sd'] + pre['mu'], 0)
        saved_frame = pd.read_parquet(runner.verify(record['prediction']))
        if not np.isfinite(saved_frame.base).all() or saved_frame.base.lt(0).any():
            raise ValueError('Invalid saved predictions')
        saved = saved_frame.set_index('mvt_id').loc[chosen, 'base'].to_numpy(float)
        np.testing.assert_allclose(prediction, saved, rtol=0, atol=.01)
        checks.append(dict(month=month, reload_rows=len(chosen), raw_scalar_checks=32,
                           max_prediction_difference_sec=float(abs(prediction-saved).max())))
        del model, checkpoint, tensors, track, tracks, X, pre
        gc.collect()
        torch.cuda.empty_cache()
    audit = dict(status='pass', reviewer='root; separate reload and arithmetic, not independent development data',
                 heldout_rows=len(frame), checks=checks, score=runner.artifact(runner.OUT/'score.json'),
                 source=runner.artifact(Path(__file__)), finite_saved_predictions=True)
    write_json(runner.OUT / 'audit.json', audit)
    print(audit, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        main()
