"""Retry portable callsign15 with an exactly equivalent bounded-memory adapter.

The failed original worker and every parent/helper source remain unchanged.
Only DirectData construction uses chunked original encoding and tensor copies.
All fifteen models still use the original shared.train at frozen15/21/16 epochs.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import callsign_submit_portable as portable
import chunked_direct_data as adapter

parent = portable.parent
OUT = portable.OUT / 'chunked'
IDENTITY = OUT / 'identity.json'


def guard():
    manifest, prepared = portable.validate_portable()
    identity = dict(source=parent.artifact(Path(__file__)), adapter=parent.artifact(Path(adapter.__file__)),
        portable_worker=portable.worker_record(), settings=adapter.SETTINGS,
        change='identical original encoding in chunks and per-column PLE quantiles; training unchanged',
        parity_test=parent.artifact(ROOT / 'VALIDATION/tests/test_chunked_direct_data.py'))
    OUT.mkdir(parents=True, exist_ok=True)
    if IDENTITY.exists():
        if parent.read(IDENTITY) != identity:
            raise ValueError('Chunked adapter/worker provenance changed')
    else:
        if list((parent.OUT / 'networks').glob('*.json')) or list((parent.OUT / 'models').glob('*.pt')):
            raise ValueError('Unexpected existing models before chunked-worker registration')
        parent.write_json(IDENTITY, identity)
    return manifest, prepared


def completed_member(group, k):
    record = portable.completed_member(group, k)
    if record is not None and record.get('chunked_adapter') != parent.artifact(IDENTITY):
        raise ValueError('Saved member does not belong to this audited chunked adapter')
    return record


def memory_probe(data):
    """Exercise the original maximum prediction batch before fitting a model."""
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        model = parent.cv.shared.nni.PleNet(data.cards, data.bins, data.train[2].shape[1]).to(data.device).eval()
        with torch.no_grad():
            output = model(*(value[:65536] for value in data.held))
        if not torch.isfinite(output).all():
            raise ValueError('Nonfinite memory-probe forward result')
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del output, model
    torch.cuda.empty_cache()
    result = dict(encoded_tensor_bytes=allocated, forward_peak_bytes=peak,
        prediction_rows=min(65536, len(data.yv)), device=torch.cuda.get_device_name(),
        device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        rng_preserved=True, probe_only_no_optimizer_or_training=True)
    parent.write_json(OUT / 'gpu_memory_probe.json', result)
    parent.log(f'Original65536 prediction batch passes: tensors{allocated/2**30:.2f}GiB, '
               f'peak{peak/2**30:.2f}GiB on{result["device_total_bytes"]/2**30:.2f}GiB GPU')


def fit():
    manifest, prepared = guard()
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated GPU required')
    if all(completed_member(g, k) is not None for g in parent.GROUPS for k in parent.NETS):
        parent.log('All15 chunked-worker models complete; skipped')
        return
    adapter.log('load verified original training cache')
    data = pd.read_pickle(portable.verify(manifest['cached_training_inputs']))
    adapter.log('original cache loaded; load feature/ID packets')
    saved_ids = pd.read_parquet(portable.verify(manifest['training_ids']))
    np.testing.assert_array_equal(saved_ids.original_index, data['Xn'].index)
    ids = saved_ids.mvt_id.to_numpy()
    if len(ids) != prepared['training_rows'] or len(ids) != len(data['yn']):
        raise ValueError('Original full-year training alignment changed')
    tables = {view: portable.load_feature_table(view, manifest) for view in ['train'] + parent.VIEWS}
    packets = {view: portable.load_packet(view, manifest, data=data) for view in parent.VIEWS}
    X, V, view_ids, view_sizes = portable.assemble_inputs(data, ids, manifest['columns'], tables, packets)
    parent.cv.ts.FOLDS.setdefault(parent.hs.ids.FULL, parent.features.TRAIN_MONTHS)
    item = dict(X=X, y=data['yn'], Vn=V, tmonth=data['tmonth'], meta=dict(month=parent.hs.ids.FULL),
                raw=pd.DataFrame({'taxi_sec': np.full(len(V), np.nan)}))
    cfg = parent.cv.shared.recipe()
    del data, tables, packets, saved_ids
    gc.collect()
    adapter.log('assembled original162 inputs; unused original frames released')
    with tempfile.TemporaryDirectory(prefix='encoding_', dir=OUT) as scratch:
        d = adapter.ChunkedDirectData(item, 'cuda', scratch)
    if np.any(d.offset) or np.any(d.val_offset) or d.info['inner_month'] != '2025-12':
        raise ValueError('Direct-target/full-year preprocessing contract changed')
    pre = parent.OUT / 'preprocessing/full_year.pkl'
    pre.parent.mkdir(parents=True, exist_ok=True)
    state = d.preprocessing()
    state['design'] = parent.DESIGN
    if pre.exists():
        parent.resume.retain_saved_preprocessing(pre, state)
    else:
        parent.cv.shared.atomic_pickle(pre, state)
    record = dict(design=parent.DESIGN, config=cfg, encoding=d.info, columns=list(X),
        preprocessing=parent.artifact(pre), prepared_sha256=parent.sha256(parent.OUT / 'prepared.json'),
        view_rows=view_sizes, view_order=parent.VIEWS, label_monitor='ranking labels absent; no epoch selection')
    pre_meta = pre.with_suffix('.json')
    if pre_meta.exists() and parent.read(pre_meta) != record:
        raise ValueError('Saved preprocessing metadata changed')
    parent.write_json(pre_meta, record)
    del X, V, item
    gc.collect()
    memory_probe(d)
    for group in parent.GROUPS:
        for k in parent.NETS:
            if completed_member(group, k) is not None:
                continue
            started = time.monotonic()
            model, pred, curve = parent.cv.shared.train(d, np.arange(len(d.y)), group+1000*k,
                parent.EPOCHS[group], (d.held, d.yv, d.val_offset))
            if not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError('Invalid portable ranking predictions')
            model_path, record_path = parent.member_paths(group, k)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temp = model_path.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), config=cfg, design=parent.DESIGN,
                            preprocessing=str(pre.relative_to(ROOT))), temp)
            temp.replace(model_path)
            predictions, offset = {}, 0
            for view in parent.VIEWS:
                n = view_sizes[view]
                dest = parent.OUT / 'networks' / f'{view}_s{group}_k{k}.parquet'
                parent.save_frame(dest, pd.DataFrame(dict(mvt_id=view_ids[view], base=pred[offset:offset+n].astype(np.float32))))
                predictions[view] = parent.artifact(dest)
                offset += n
            if offset != len(pred):
                raise ValueError('Ranking view slices did not cover the inference matrix')
            curve = [{name: value for name, value in row.items() if name != 'rmse'} for row in curve]
            parent.write_json(record_path, dict(design_sha256=parent.sha256(parent.OUT / 'design.json'),
                prepared_sha256=parent.sha256(parent.OUT / 'prepared.json'), seed=group+1000*k,
                epochs=parent.EPOCHS[group], model=parent.artifact(model_path), preprocessing=parent.artifact(pre),
                predictions=predictions, config=cfg, training_curve=curve,
                minutes=(time.monotonic()-started)/60, device=torch.cuda.get_device_name(),
                torch=str(torch.__version__), portable_worker=portable.worker_record(),
                chunked_adapter=parent.artifact(IDENTITY)))
            parent.log(f'Chunked full-year callsign group{group} k{k}, epoch{parent.EPOCHS[group]} saved')
            del model
            torch.cuda.empty_cache()


def audit():
    parent.validate_prepared()
    guard()
    if not all(completed_member(g, k) is not None for g in parent.GROUPS for k in parent.NETS):
        raise ValueError('All15 chunked-worker models must complete')
    parent.log('Original desktop sources and all15 chunked-worker artifacts verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'audit'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {'fit': fit, 'audit': audit}[args.stage]()


if __name__ == '__main__':
    main()
