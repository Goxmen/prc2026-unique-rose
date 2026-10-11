"""Expand the passing public neural head to the retained three physical seeds.

Reuse completed seed42; add1042/2042 with the original21/17 epochs. Compare
equal-size physical pools inside the actual v75 recipe without tuning weights.
"""
import argparse
import gc
import os
from pathlib import Path
import pickle
import tempfile
import time
import types
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_scratch_jan_jul as single
import nn_sparse_track_pool3 as original
import nn_track_physical_pool3 as control
import rome_public_delay_cat as cat
from records import write_json

single.DESIGN = dict(single.DESIGN, variant='public')
ROOT = single.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_nn_public_pool3'
MANIFEST = OUT / 'identity.json'
DESIGN = dict(parent='prepared v75', change='public daily head in physical three-seed pool',
    seeds=[42, 1042, 2042], reused_seed=42, new_seeds=[1042, 2042], epochs=single.STOP,
    model=single.airport_tune.SETTINGS, network_share=.71, physical_share_of_neural=.5,
    main_tree='retained callsign LB1', rome='selected public Cat3/original R20',
    machine='all public and original physical seeds on desktop RTX3080',
    no_stopping_or_weight_search=True, selection='declared after seed42 passed; reused development months')


def training(runtime):
    fn = original.compile_training(runtime)
    return types.FunctionType(fn.__code__, {**fn.__globals__, 'prototype': single.airport_tune},
        fn.__name__, fn.__defaults__, fn.__closure__)


def guard():
    assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080'
    assert single.shared.ts.read(single.OUT / 'audit.json')['status'] == 'pass'
    score = single.shared.ts.read(single.OUT / 'score.json')
    assert score['comparisons']['public_single_mix_vs_matched_physical_single_mix']['decision'] == 'improvement'
    identity = single.shared.ts.read(single.MANIFEST)
    for month in single.STOP:
        assert single.completed(month, identity) is not None
    value = dict(design=DESIGN, source=single.artifact(__file__),
        single_identity=single.artifact(single.MANIFEST), single_score=single.artifact(single.OUT / 'score.json'),
        single_audit=single.artifact(single.OUT / 'audit.json'),
        helpers=[single.artifact(m.__file__) for m in [single, original, control, single.airport_tune, cat]],
        control_identity=single.artifact(control.MANIFEST), control_audit=single.artifact(control.OUT / 'audit.json'),
        rome_result=single.artifact(cat.OUT / 'result.json'),
        rome_predictions=single.artifact(cat.OUT / 'scored.parquet'),
        seed_specialization=original.check_specialization())
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert single.shared.ts.read(MANIFEST) == value
    else:
        write_json(MANIFEST, value)
    return identity


def paths(month, k):
    if k == 0:
        return single.paths(month)
    return OUT / 'networks' / f'{month}_k{k}.parquet', OUT / 'models' / f'{month}_k{k}.pt', OUT / 'networks' / f'{month}_k{k}.json'


def completed(month, k, identity):
    if k == 0:
        return single.completed(month, identity)
    path, model, record = paths(month, k)
    if not record.exists():
        assert not path.exists() and not model.exists(), 'Unrecorded outputs'
        return None
    rec = single.shared.ts.read(record)
    assert rec['manifest'] == single.artifact(MANIFEST)
    assert rec['seed'] == 42+1000*k and rec['epochs'] == single.STOP[month]
    for key in ['model', 'prediction', 'preprocessing']:
        single.verify(rec[key])
    return rec


def initial_check(seed):
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])]*3, 5
    torch.manual_seed(seed)
    old = single.airport_tune.physical.SparseFusion(cards, bins, plain)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    new = single.airport_tune.SparseFusion(cards, bins, plain+12)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    assert not new.public[-1].weight.count_nonzero() and not new.public[-1].bias.count_nonzero()
    return dict(seed=seed, original_weights_rng_exact=True, new_head_zero=True)


def fit(identity):
    single.shared.recipe()
    meta, _ = single.metadata.load()
    index = pd.read_parquet(single.base.seq.OUT / 'full_index.parquet')
    points = np.memmap(single.base.seq.OUT / 'full_points.bin', mode='r', dtype=np.int32).reshape(-1, 6)
    tracks = single.base.tensor_seq.RaggedTracks(index, points, 'cuda')
    for name in ['models', 'networks', 'runtime']:
        (OUT / name).mkdir(exist_ok=True)
    for month in single.STOP:
        pending = [k for k in [1, 2] if completed(month, k, identity) is None]
        if not pending:
            continue
        data = pd.read_pickle(single.verify(identity['folds'][month]['matrix']))
        names = data['baseline']
        assert len(names) == 156
        tm, vm = meta.loc[data['train_ids']], meta.loc[data['ids']]
        assert set(tm.month.astype(str)) == set(single.shared.ts.FOLDS[month])
        assert set(vm.month.astype(str)) == {month}
        assert not np.intersect1d(data['train_ids'], data['ids']).size
        np.testing.assert_array_equal(data['y'], tm.taxi_sec)
        tids, ids = np.asarray(data['train_ids']).copy(), np.asarray(data['ids']).copy()
        item = dict(X=data['X'][names], Vn=data['V'][names], y=data['y'], raw=vm.reset_index(),
            tmonth=tm.month.astype(str).to_numpy(), meta=dict(month=month))
        del data, tm, vm
        gc.collect()
        with tempfile.TemporaryDirectory(prefix='encoding_', dir=OUT) as scratch:
            d = single.adapter.ChunkedDirectData(item, 'cuda', scratch)
        state = d.preprocessing()
        state['design'] = single.base.DESIGN
        pre = single.verify(identity['folds'][month]['preprocessing'])
        with pre.open('rb') as handle:
            single.base.preprocess_guard.equal_preprocessing(pickle.load(handle), state)
        public = pd.read_parquet(single.verify(identity['public_features'])).set_index('mvt_id')
        for part, flight_ids in [('train', tids), ('held', ids)]:
            extra = torch.from_numpy(single.airport_tune.transform(public.loc[flight_ids])).to(d.device)
            before = getattr(d, part)
            setattr(d, part, (*before[:2], torch.cat([before[2], extra], dim=1)))
        del item, public, extra, before
        gc.collect()
        tr, va = tracks.align(tids), tracks.align(ids)
        for k in pending:
            seed = 42+1000*k
            check = initial_check(seed)
            runtime = OUT / 'runtime' / f'{month}_k{k}'
            runtime.mkdir(exist_ok=True)
            start = time.monotonic()
            model, pred, curve = training(runtime)(d, tracks, tr, va, single.STOP[month], seed)
            assert np.isfinite(pred).all() and (pred >= 0).all()
            path, modelpath, record = paths(month, k)
            temp = modelpath.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), design=DESIGN, config=single.shared.recipe(),
                preprocessing=str(pre.relative_to(ROOT))), temp)
            temp.replace(modelpath)
            single.shared.nni.save_network(path, ids, pred)
            write_json(record, dict(manifest=single.artifact(MANIFEST), month=month, member=k, seed=seed,
                epochs=single.STOP[month], minutes=(time.monotonic()-start)/60, curve=curve, initial_check=check,
                model=single.artifact(modelpath), prediction=single.artifact(path), preprocessing=single.artifact(pre),
                train_rows=len(tids), held_rows=len(ids), gpu=torch.cuda.get_device_name(), host=os.uname().nodename))
            del model, pred
            gc.collect()
            torch.cuda.empty_cache()
        del d, tr, va
        gc.collect()
        torch.cuda.empty_cache()


def score(identity):
    r20 = cat.r20
    ph, replay = r20.physical, r20.physical.replay
    sc, rome, _, trees = replay.baseline.baseline()
    prior = pd.read_parquet(cat.OUT / 'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(r20.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(r20.constraints.exclusion.OUT / 'matches2025.parquet')
    old_identity = single.shared.ts.read(control.single.MANIFEST)
    frames = []
    for month in single.STOP:
        d = sc.d[month]
        ids = d['ids']
        old, new = [], []
        for k in [0, 1, 2]:
            for bank, rec in [(old, control.completed(month, k, old_identity)),
                              (new, completed(month, k, identity))]:
                assert rec is not None
                f = pd.read_parquet(single.verify(rec['prediction']))
                np.testing.assert_array_equal(f.mvt_id, ids)
                bank.append(f.base.to_numpy(float))
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        main = replay.values(replay.combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        tree = np.mean([main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
        p = .5*prior.loc[ids, 'newcat'].to_numpy()+.5*rome[month]
        reference = sc.final(month, .71*.5*(calls+np.mean(old, axis=0))+.29*tree, p)
        np.testing.assert_array_equal(reference, prior.loc[ids, 'cat_only'])
        pred = sc.final(month, .71*.5*(calls+np.mean(new, axis=0))+.29*tree, p)
        q = r20.constraints.matched_bound(raw.loc[ids].reset_index(),
            prior.loc[ids, 'v74'].to_numpy()+pred-prior.loc[ids, 'control'].to_numpy())
        permitted = matches[matches.donor_month.isin(r20.features.base.FOLDS[month]+[month])]
        q, _ = r20.constraints.exclusion.apply(ids, np.maximum(q, 0), permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'],
            control=reference, pred=pred, v75=prior.loc[ids, 'cat_only_transfer'].to_numpy(), transferred=q)))
    f = pd.concat(frames, ignore_index=True)
    f.to_parquet(OUT / 'scored.parquet', index=False)
    cols = ['control', 'pred', 'v75', 'transferred']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in cols} for m, g in f.groupby('month')}
    pairs = {f'{b}_vs_{a}': single.shared.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
        for a, b in [('control', 'pred'), ('v75', 'transferred')]}
    result = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([s[c] for s in scores.values()])) for c in cols}, paired=pairs)
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        identity = guard()
        globals()[args.stage](identity)
