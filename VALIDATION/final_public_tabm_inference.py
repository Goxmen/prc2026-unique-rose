"""Inference only: retained three TabM models on separate old/final views.

Run on the same laptop/GPU/Torch after production fitting and artifact return.
Saved preprocessing is transformed directly, never refitted. Every old-view
prediction must replay exactly before any final-view prediction is written.
"""
import argparse
import gc
import json
import os
import pickle
import tempfile
import types
from pathlib import Path

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_tabm_pool3_submit as production
import final_public_features as public

first = production.first
old = first.old
ROOT = first.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_public_tabm_inference'
MANIFEST = OUT / 'identity.json'
art, read, verify, write = first.art, first.read, first.verify, first.write_json
DESIGN = dict(seeds=[42, 1042, 2042], epochs=15, views=['old', 'final'],
    preprocessing='Saved full-year Encoder.transform, numeric layout, bins and target scaling',
    public_inputs='Unchanged fixed twelve-input transform; audited separate final view',
    control='All three old-view predictions replay exactly on original laptop/GPU/Torch',
    prediction='Original 512-row inference chunks and head mean; no fitting',
    no_new_training=True, no_new_selection=True, no_version=True, no_upload=True)


def prepare():
    assert not MANIFEST.exists()
    outer = production.guard()
    assert read(production.OUT / 'export.json')['identity'] == art(production.OUT / 'identity.json')
    assert read(public.RECORDS / 'audit.json')['status'] == 'pass'
    public_manifest = read(public.OUT / 'manifest.json')
    assert public_manifest['output'] == art(public.OUT / 'features.parquet')
    for a in public_manifest['sources']:
        verify(a)
    physical = read(first.parent.physical.MANIFEST)
    assert 'final' in physical['packet']['views'] and 'final' in physical['tracks']
    members = []
    for k in range(3):
        ns = first.__dict__ if k == 0 else production.member(k)
        identity = read(ns['MANIFEST'])
        trained = ns['trained'](identity)
        record = ns['completed'](identity)
        assert trained is not None and record is not None
        assert trained['seed'] == DESIGN['seeds'][k] and trained['epochs'] == 15
        members.append(dict(seed=trained['seed'], identity=art(ns['MANIFEST']),
            trained=art(ns['OUT'] / 'trained/k0.json'), model=trained['model'],
            preprocessing=trained['preprocessing'], old_prediction=record['prediction'],
            design=ns['DESIGN']))
    execution = {k:trained[k] for k in ['host', 'gpu', 'torch']}
    for member in members:
        r = read(verify(member['trained']))
        assert {k:r[k] for k in execution} == execution
    paths = [Path(__file__), Path(public.__file__), Path(production.__file__), Path(first.__file__),
        Path(first.cv.__file__), Path(first.cv.model_type.__file__), Path(old.adapter.__file__),
        first.parent.physical.MANIFEST, production.OUT / 'identity.json', production.OUT / 'export.json',
        public.OUT / 'manifest.json', public.RECORDS / 'audit.json',
        public.OUT / 'features.parquet', ROOT / 'data/final_submitting.parquet']
    original = read(first.MANIFEST)
    for m in members:
        paths.extend(verify(m[k]) for k in ['identity','trained','model','preprocessing','old_prediction'])
    paths += [verify(original['resources']['ranking_features']),
              verify(physical['packet']['cached_training_inputs']), old.packets.packet_path('final')]
    for track in [physical['tracks'][v] for v in DESIGN['views']]:
        paths.extend(verify(a) for a in track['files'])
    value = dict(design=DESIGN, execution=execution, members=members,
        original_public=original['resources']['ranking_features'], final_public=public_manifest['output'],
        packet=physical['packet'], tracks={v:physical['tracks'][v] for v in DESIGN['views']},
        sources=[art(p) for p in sorted(set(paths))])
    OUT.mkdir(parents=True, exist_ok=True)
    write(MANIFEST, value)
    print(dict(prepared=True, members=3, final_rows=670790, no_training=True), flush=True)


def data_for_view(manifest, pre, view):
    packet = old.packets.load_packet(view, manifest['packet'])
    ids = np.asarray(packet['rank_ids'])
    frame = packet['Vn']
    assert list(frame) == pre['columns'] == manifest['packet']['columns']
    expected = 344841 if view == 'old' else 670790
    assert len(ids) == len(frame) == expected and pd.Index(ids).is_unique
    template = 'submitting.parquet' if view == 'old' else 'final_submitting.parquet'
    np.testing.assert_array_equal(ids, pd.read_parquet(ROOT / 'data' / template,
        columns=['MVT_ID_mvt']).MVT_ID_mvt)
    with tempfile.TemporaryDirectory(prefix=f'encode_{view}_', dir=OUT) as tmp:
        cat, num = old.adapter.encode(pre['encoder'], frame, tmp, 'inference')
        held = (old.adapter.tensor_columns(cat, None, 'cuda'),
                old.adapter.tensor_columns(num, pre['ple'], 'cuda'),
                old.adapter.tensor_columns(num, pre['plain'], 'cuda'))
        torch.cuda.synchronize()
        cat._mmap.close(); num._mmap.close()
    public_path = manifest['original_public'] if view == 'old' else manifest['final_public']
    extra = pd.read_parquet(verify(public_path)).set_index('mvt_id')
    assert extra.index.is_unique
    transformed = first.cv.model_type.transform(extra.loc[ids])
    held = (*held[:2], torch.cat([held[2], torch.from_numpy(transformed).cuda()], dim=1))
    d = types.SimpleNamespace(held=held, mu=pre['mu'], sd=pre['sd'])
    del packet, frame, extra, transformed, cat, num
    gc.collect()
    source = manifest['tracks'][view]
    tracks = old.load_tracks(ROOT / source['directory'], source['identity'], 'cuda', view)
    return ids, d, tracks, tracks.align(ids)


def infer():
    m = read(MANIFEST)
    assert m['design'] == DESIGN
    execution = dict(host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__))
    assert execution == m['execution'], 'Use the original production execution environment'
    for a in m['sources']:
        verify(a)
    with verify(m['members'][0]['preprocessing']).open('rb') as handle:
        pre = pickle.load(handle)
    for member in m['members'][1:]:
        with verify(member['preprocessing']).open('rb') as handle:
            comparison = pickle.load(handle)
        first.cv.original.base.preprocess_guard.equal_preprocessing(
            {k:v for k,v in pre.items() if k != 'design'},
            {k:v for k,v in comparison.items() if k != 'design'})
    (OUT / 'predictions').mkdir(exist_ok=True)
    for view in DESIGN['views']:
        receipt = OUT / f'{view}.json'
        if receipt.exists():
            prior = read(receipt); assert prior['identity'] == art(MANIFEST)
            for a in prior.get('predictions', []): verify(a)
            continue
        if view == 'final':
            assert read(OUT / 'old.json')['all_old_predictions_exact']
        ids, d, tracks, rows = data_for_view(m, pre, view)
        outputs = []
        for k, member in enumerate(m['members']):
            checkpoint = torch.load(verify(member['model']), map_location='cuda', weights_only=False)
            assert checkpoint['identity'] == member['identity'] and checkpoint['design'] == member['design']
            model = first.cv.model_type.PublicTabM(pre['cards'], pre['bins'], len(pre['plain']) + 12).cuda()
            model.load_state_dict(checkpoint['state_dict'])
            pred = first.cv.predict(model, d, tracks, rows).astype(np.float32)
            assert np.isfinite(pred).all() and (pred >= 0).all()
            if view == 'old':
                prior = pd.read_parquet(verify(member['old_prediction']))
                np.testing.assert_array_equal(ids, prior.mvt_id)
                np.testing.assert_array_equal(pred, prior.base)
            else:
                # Restore the weights independently and check every final row.
                restored = first.cv.model_type.PublicTabM(pre['cards'], pre['bins'], len(pre['plain']) + 12).cuda()
                restored.load_state_dict(torch.load(verify(member['model']), map_location='cuda', weights_only=False)['state_dict'])
                np.testing.assert_array_equal(pred, first.cv.predict(restored, d, tracks, rows).astype(np.float32))
                del restored
                path = OUT / 'predictions' / f'final_k{k}.parquet'
                value = pd.DataFrame(dict(mvt_id=ids, base=pred))
                if path.exists(): pd.testing.assert_frame_equal(pd.read_parquet(path), value, check_exact=True)
                else: value.to_parquet(path, index=False)
                outputs.append(art(path))
            del checkpoint, model, pred
        write(receipt, dict(identity=art(MANIFEST), view=view, rows=len(ids),
            all_old_predictions_exact=True, all_final_reloads_exact=view == 'final',
            predictions=outputs, no_fit=True))
        del ids, d, tracks, rows
        gc.collect(); torch.cuda.empty_cache()
        print(dict(view=view, verified=True, rows=read(receipt)['rows']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'infer'])
    torch.set_num_threads(2)
    with threadpool_limits(2): globals()[parser.parse_args().stage]()
