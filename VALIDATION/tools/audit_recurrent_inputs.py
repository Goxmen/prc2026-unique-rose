"""Verify chronological prefix masks and benchmark the fixed GRU screen."""
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import nn_public_recurrent_jan_jul as runner
import sparse_track_tensors as tensor_seq
from records import write_json


def audit():
    out = runner.OUT/'input_benchmark.json'
    assert not out.exists(), 'Preserve completed check'
    identity = runner.guard()
    seq = runner.parent.single.base.seq
    index = pd.read_parquet(seq.OUT/'full_index.parquet')
    raw = np.memmap(seq.OUT/'full_points.bin', mode='r', dtype=np.int32).reshape(-1,6)
    assert index.mvt_id.is_unique and index['count'].between(0,seq.MAX_POINTS).all()
    assert index.start.ge(0).all() and (index.start+index['count']).le(len(raw)).all()
    scopes = [index['count'].eq(k) for k in [0,1,2,seq.MAX_POINTS]]
    positions = np.unique(np.concatenate([np.linspace(0,len(index)-1,2048,dtype=int),
        *[np.flatnonzero(v)[:12] for v in scopes]]))
    selected = index.iloc[positions].reset_index(drop=True)
    tracks = tensor_seq.RaggedTracks(selected,raw,'cuda')
    points, mask, quality = tracks.batch(np.arange(len(selected)))
    reference = np.zeros(mask.shape,dtype=bool)
    for i,row in enumerate(selected.itertuples()):
        for j in range(row.count): reference[i,j] = True
        original = raw[row.start:row.start+row.count,0]
        assert (np.diff(original.astype('int64')) < 0).all()
    np.testing.assert_array_equal(mask.cpu(),reference)
    encoded = seq.encoded_batch(selected,raw)
    for a,b in zip([points,mask,quality],encoded):
        np.testing.assert_allclose(a.cpu().numpy(),b,rtol=2e-6,atol=2e-7)
    # Count selects the last actual observation, not the last padded timestep.
    count = mask.sum(1)
    for i,n in enumerate(count.cpu().tolist()):
        if n: assert bool(mask[i,n-1]) and not bool(mask[i,n:].any())
    del tracks,points,mask,quality
    torch.cuda.empty_cache()
    rec = runner.control('2025-01',runner.parent.read(runner.parent.single.MANIFEST))
    with runner.parent.verify(rec['preprocessing']).open('rb') as f: pre = pickle.load(f)
    m = runner.model_type
    cards,bins,plain = pre['cards'],pre['bins'],len(pre['plain'])
    m.physical.parent.shared.recipe()
    torch.manual_seed(42)
    model = m.SparseFusion(cards,bins,plain+12).cuda().train()
    tab,track = m.physical.parent.synthetic(cards,bins,plain,2048,'cuda')
    tab = (*tab[:2],torch.cat([tab[2],torch.zeros(2048,12,device='cuda')],dim=1))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(*tab,*track).square().mean().backward(); optimizer.step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        model(*tab,*track).square().mean().backward(); optimizer.step()
    torch.cuda.synchronize()
    value=dict(status='pass',identity=runner.parent.art(runner.MANIFEST),
        auditor=runner.parent.art(__file__), all_index_rows=len(index),sampled_rows=len(selected),
        all_count_start_bounds_checked=True,independent_prefix_mask=True,
        chronological_raw_integer_times=True,scalar_tensor_inputs_agree=True,
        last_observed_index_checked=True,labels_used=False,
        benchmark=dict(gpu=torch.cuda.get_device_name(),batch=2048,
            milliseconds_per_update=1000*(time.monotonic()-start)/5,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            full_original_tabular_dimensions=True,synthetic_rows=True))
    write_json(out,value);print(value,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):audit()
