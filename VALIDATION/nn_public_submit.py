"""Fit the public-feature physical neural pool only after its CV/audit pass.

Reuse the verified full-year 156-column packet, encoder and raw track stores.
Append the same twelve fixed transformed inputs as CV, keep all three seeds
on one desktop, and use the retained original 15-epoch full-year schedule.
Models precede inference on disk; completed fits are never repeated. No upload.
"""
import argparse
import gc
import os
from pathlib import Path
import pickle
import tempfile
import time
import types
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_pool3 as cv
import nn_track_physical_submit as physical
import rome_public_delay_submit as public_rome
from records import write_json

ROOT = physical.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_nn_public_submit'
MANIFEST = OUT / 'identity.json'
old = physical.old
model_type = cv.single.airport_tune
art, verify = physical.art, physical.verify
read = cv.single.shared.ts.read
SEEDS, EPOCHS = physical.SEEDS, physical.EPOCHS
DESIGN = dict(parent='v75 physical neural pool', change='12 public daily inputs through the passing residual head',
    columns=156, public_features=model_type.features.FEATURES, model=model_type.SETTINGS,
    seeds=SEEDS, epochs=EPOCHS, training='unchanged fixed-epoch production updates; model constructor replaced',
    original_preprocessing='exact retained full-year encoder; append public transform afterward',
    training_rows=2051327, ranking_view='old', ranking_rows=344841,
    machine='all new production members on desktop RTX3080',
    machine_caveat='retained original physical production pool used laptop; CV controls/candidates all desktop',
    no_new_stopping_or_weight_search=True, no_upload=True)


def trainer():
    proxy = types.SimpleNamespace(**dict(vars(old.fast.original), prototype=model_type))
    fn = types.FunctionType(old.fast.train_updates.__code__,
        dict(old.fast.train_updates.__globals__, original=proxy), 'public_train_updates',
        old.fast.train_updates.__defaults__, old.fast.train_updates.__closure__)
    assert old.fast.original.prototype is cv.single.base.prototype
    assert fn.__code__ is old.fast.train_updates.__code__
    return fn


def preflight():
    cv.single.shared.recipe()
    trainer()
    result = dict(training_bytecode_exact=True,
        checks=[model_type.check('cpu')], seeds=[cv.initial_check(seed) for seed in SEEDS],
        source=art(__file__))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'preflight.json'
    if path.exists():
        assert read(path) == result
    else:
        write_json(path, result)
    print('CPU initialization, RNG, masking, gradients and exact trainer checks pass', flush=True)


def prepare():
    cv.guard()
    result = read(cv.OUT / 'result.json')
    audit = read(cv.OUT / 'audit.json')
    score_audit = read(cv.OUT / 'score_audit.json')
    assert audit['status'] == 'pass'
    assert score_audit['status'] == 'pass' and score_audit['result'] == art(cv.OUT / 'result.json')
    verify(score_audit['predictions'])
    for name in ['pred_vs_control', 'transferred_vs_v75']:
        assert result['paired'][name]['decision'] == 'improvement', name
    assert read(OUT / 'preflight.json')['source'] == art(__file__)
    parent = physical.guard()
    assert parent['training_rows'] == DESIGN['training_rows']
    assert read(model_type.features.OUT / 'audit.json')['status'] == 'pass'
    assert read(public_rome.OUT / 'ranking_feature_audit.json')['status'] == 'pass'
    sources = [cv, cv.single, model_type, model_type.public, model_type.physical,
        model_type.features, physical, old, old.fast, old.adapter, public_rome]
    value = dict(design=DESIGN, source=art(__file__), helpers=[art(m.__file__) for m in sources],
        preflight=art(OUT / 'preflight.json'), cv_identity=art(cv.MANIFEST),
        cv_result=art(cv.OUT / 'result.json'), cv_audit=art(cv.OUT / 'audit.json'),
        cv_score_audit=art(cv.OUT / 'score_audit.json'),
        physical_identity=art(physical.MANIFEST), parent_packet=parent,
        training_features=art(model_type.features.OUT / 'features2025.parquet'),
        training_feature_audit=art(model_type.features.OUT / 'audit.json'),
        ranking_features=art(public_rome.OUT / 'ranking_features.parquet'),
        ranking_feature_audit=art(public_rome.OUT / 'ranking_feature_audit.json'))
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('**/*.pt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST, value)
    print('Passing CV and original production packet bound to fixed recipe', flush=True)


def guard():
    value = read(MANIFEST)
    assert value['source'] == art(__file__) and value['design'] == DESIGN
    for rec in value['helpers']:
        verify(rec)
    for key in ['preflight', 'cv_identity', 'cv_result', 'cv_audit', 'cv_score_audit', 'physical_identity',
                'training_features', 'training_feature_audit', 'ranking_features', 'ranking_feature_audit']:
        verify(value[key])
    assert read(verify(value['cv_audit']))['status'] == 'pass'
    parent = physical.guard()
    assert parent == value['parent_packet']
    return value


def paths(k):
    return OUT / 'models' / f'k{k}.pt', OUT / 'trained' / f'k{k}.json', OUT / 'networks' / f'k{k}.json'


def trained(k):
    model, record, _ = paths(k)
    if not record.exists():
        assert not model.exists(), 'Unrecorded model: inspect before resuming'
        return None
    rec = read(record)
    assert rec['identity'] == art(MANIFEST) and rec['seed'] == SEEDS[k] and rec['epochs'] == EPOCHS
    verify(rec['model'])
    verify(rec['preprocessing'])
    return rec


def completed(k):
    rec = trained(k)
    path = paths(k)[2]
    if not path.exists():
        return None
    assert rec is not None
    value = read(path)
    assert value['trained'] == art(paths(k)[1]) and value['rows'] == DESIGN['ranking_rows']
    assert value['all_ranking_reload_exact']
    verify(value['prediction'])
    return value


def fit():
    m = guard()
    device = torch.cuda.get_device_name()
    assert device == 'NVIDIA GeForce RTX 3080', 'Allocate all new production members to one desktop'
    for k in range(3):
        rec = trained(k)
        if rec is not None:
            assert rec['gpu'] == device
    pending = [k for k in range(3) if completed(k) is None]
    if not pending:
        return
    for name in ['models', 'trained', 'networks', 'preprocessing']:
        (OUT / name).mkdir(exist_ok=True)
    parent, packet = m['parent_packet'], m['parent_packet']['packet']
    data = pd.read_pickle(verify(packet['cached_training_inputs']))
    membership = pd.read_parquet(verify(packet['training_ids']))
    np.testing.assert_array_equal(membership.original_index, data['Xn'].index)
    tids = membership.mvt_id.to_numpy()
    assert len(tids) == len(data['yn']) == DESIGN['training_rows']
    rank = old.packets.load_packet('old', packet, data=data)
    ids = np.asarray(rank['rank_ids']).copy()
    assert len(ids) == DESIGN['ranking_rows'] and pd.Index(ids).is_unique
    names = packet['columns']
    assert len(names) == 156
    old.parent.cv.ts.FOLDS.setdefault(old.parent.hs.ids.FULL, old.parent.features.TRAIN_MONTHS)
    item = dict(X=data['Xn'][names], Vn=rank['Vn'][names], y=data['yn'], tmonth=data['tmonth'],
        meta=dict(month=old.parent.hs.ids.FULL), raw=pd.DataFrame({'taxi_sec': np.full(len(ids), np.nan)}))
    del data, membership, rank
    gc.collect()
    with tempfile.TemporaryDirectory(prefix='encoding_', dir=OUT) as scratch:
        d = old.adapter.ChunkedDirectData(item, 'cuda', scratch)
    state = d.preprocessing()
    state['design'] = old.DESIGN
    with verify(parent['reference_preprocessing']).open('rb') as handle:
        cv.single.base.preprocess_guard.equal_preprocessing(pickle.load(handle), state)
    state['design'] = DESIGN
    pre = OUT / 'preprocessing/full_year.pkl'
    if pre.exists():
        cv.single.base.preprocess_guard.retain_saved_preprocessing(pre, state)
    else:
        cv.single.shared.atomic_pickle(pre, state)
    for part, flight_ids, source in [('train', tids, m['training_features']), ('held', ids, m['ranking_features'])]:
        public = pd.read_parquet(verify(source)).set_index('mvt_id')
        extra = torch.from_numpy(model_type.transform(public.loc[flight_ids])).to(d.device)
        before = getattr(d, part)
        setattr(d, part, (*before[:2], torch.cat([before[2], extra], dim=1)))
        del public, extra, before
    del item, state
    gc.collect()
    stores = {name: old.load_tracks(ROOT / parent['tracks'][name]['directory'],
        parent['tracks'][name]['identity'], 'cuda', None if name == 'train' else name) for name in ['train', 'old']}
    train = stores.pop('train')
    rows = train.align(tids)
    views = old.fast.ranking_views(d, {'old': ids}, stores)
    for k in pending:
        started = time.monotonic()
        mp, record, prediction_record = paths(k)
        rec = trained(k)
        if rec is None:
            model, curve = trainer()(d, train, rows, views, SEEDS[k], EPOCHS)
            temporary = mp.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), design=DESIGN, config=parent['config'],
                preprocessing=str(pre.relative_to(ROOT))), temporary)
            temporary.replace(mp)
            write_json(record, dict(identity=art(MANIFEST), seed=SEEDS[k], epochs=EPOCHS,
                model=art(mp), preprocessing=art(pre), curve=curve,
                minutes=(time.monotonic()-started)/60, gpu=device, host=os.uname().nodename,
                torch=str(torch.__version__), training_ids=packet['training_ids']))
        else:
            model = model_type.SparseFusion(d.cards, d.bins, d.train[2].shape[1]).cuda()
            model.load_state_dict(torch.load(mp, map_location='cuda', weights_only=False)['state_dict'])
        path = OUT / 'networks' / f'old_k{k}.parquet'
        assert not path.exists(), 'Unrecorded prediction: inspect before resuming'
        pred = old.fast.predict_view(model, d, views[0]).astype(np.float32)
        restored = model_type.SparseFusion(d.cards, d.bins, d.train[2].shape[1]).cuda()
        restored.load_state_dict(torch.load(mp, map_location='cuda', weights_only=False)['state_dict'])
        reloaded = old.fast.predict_view(restored, d, views[0]).astype(np.float32)
        np.testing.assert_array_equal(pred, reloaded)
        assert np.isfinite(pred).all() and (pred >= 0).all()
        pd.DataFrame(dict(mvt_id=ids, base=pred)).to_parquet(path, index=False)
        write_json(prediction_record, dict(trained=art(record), prediction=art(path),
            rows=len(ids), all_ranking_reload_exact=True, ranking_view='old', gpu=device))
        del model, restored, pred, reloaded
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Public neural production k{k} retained and all ranking rows reloaded', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'prepare', 'fit'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        globals()[args.stage]()
