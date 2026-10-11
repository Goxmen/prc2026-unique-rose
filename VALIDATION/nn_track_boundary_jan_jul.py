"""Fixed two-fold point-boundary supervision screen on the selected v64 stack.

Original156 inputs and the original sparse optimizer, batches and schedule are retained;
The fixed auxiliary loss trains a point boundary head on the physical track representation.
Same-desktop physical seed42 controls and fixed original21/17 epochs. No upload.
"""
import argparse
import math
import gc
import os
from pathlib import Path
import pickle
import platform
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

import nn_sparse_track_jan_jul as base
import nn_track_boundary as boundary
import nn_track_physical_scales_jan_jul as control_screen
import chunked_direct_data as adapter
import missing_plan_view as metadata
import sparse_track_combo_score as combo
from records import sha256, write_json

ROOT, shared = base.ROOT, base.shared
OUT = shared.ts.RUNS / 'v3_20261011_nn_track_boundary_jan_jul'
MANIFEST = OUT / 'identity.json'
STOP = dict(base.STOP)
DESIGN = dict(parent='selected half sparse3/half calls15 combination',
    change='dense pointwise before/after off-block supervision of physical-scale encoder',
    original_columns=156, added_columns=[], model=boundary.SETTINGS,
    original_model=base.DESIGN['model'], sequence=base.DESIGN['sequence'],
    seed=42, fixed_epochs=STOP, target=base.DESIGN['target'],
    training='original sparse training schedule plus fixed 0.1 pointwise auxiliary BCE',
    preprocessing='exact original156 preprocessing, checked against saved control',
    machine='desktop RTX3080; same-machine saved physical-scale seed42 controls',
    primary='half boundary single/half calls15 versus half physical single/half calls15',
    selected_reference='actual three-sparse-member v64; pool-size difference explicit',
    tree='selected callsign main LightGBM plus original airport and CatBoost, seed42',
    rome='selected half Cat3/half R20', network_share=.71, rules='unchanged',
    no_inner_stopping=True, no_weight_search=True, no_new_gate=True,
    held_labels='fixed-epoch monitoring and final development score only',
    exploratory_reused_development_months=True, production_or_upload=False)


def artifact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def verify(record):
    path = ROOT / record['path']
    if sha256(path) != record['sha256']:
        raise ValueError('Artifact changed: ' + str(path))
    return path


def training_function():
    return train


def train(d, tracks, train_rows, held_rows, stop):
    torch.manual_seed(42); np.random.seed(42)
    model=boundary.SparseFusion(d.cards,d.bins,d.train[2].shape[1]).to(d.device)
    opt=torch.optim.AdamW(model.parameters(),lr=shared.nni.NEW['ple']['lr'],
        weight_decay=shared.nni.NEW['ple']['weight_decay'])
    count,batch=len(d.y),shared.nni.NEW['batch']
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=shared.nni.NEW['ple']['lr'],
        total_steps=shared.nni.NEW['epochs']*math.ceil(count/batch))
    generator=torch.Generator(device=d.device).manual_seed(42);curve=[]
    for epoch in range(stop):
        started=time.monotonic();total=main_total=aux_total=0.
        perm=torch.randperm(count,device=d.device,generator=generator);cpu_perm=perm.cpu().numpy()
        for i in range(0,count,batch):
            rows=perm[i:i+batch];track_batch=tracks.batch(train_rows[cpu_perm[i:i+batch]])
            output=model(*(a[rows] for a in d.train),*track_batch)
            main=torch.nn.functional.mse_loss(output,d.ty[rows,None].expand_as(output))
            target_sec=d.ty[rows]*d.sd+d.mu
            auxiliary=boundary.auxiliary_loss(model,track_batch[0],track_batch[1],target_sec)
            loss=main+boundary.SETTINGS['auxiliary_weight']*auxiliary
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();schedule.step()
            total+=float(loss)*len(rows);main_total+=float(main)*len(rows);aux_total+=float(auxiliary)*len(rows)
        pred=base.predict(model,d,tracks,held_rows)
        curve.append(dict(epoch=epoch+1,train_loss=total/count,train_mse_std=main_total/count,
            auxiliary_bce=aux_total/count,rmse=shared.nni.rmse(d.yv,pred),seconds=time.monotonic()-started))
        shared.log(f'epoch {epoch+1}/{stop}: main {main_total/count:.4f}; auxiliary {aux_total/count:.4f}; '
            f'monitored {curve[-1]["rmse"]:.4f}; {curve[-1]["seconds"]:.1f}s')
    return model,pred,curve


def guard(cache):
    base.guard(cache)
    sources = {}
    for month in STOP:
        if control_screen.completed(month, shared.ts.read(control_screen.MANIFEST)) is None:
            raise ValueError('Both same-desktop physical controls required')
        sources[month] = dict(matrix=artifact(shared.baseline.OUT / 'inputs' / f'{month}.pkl'),
            control_record=artifact(control_screen.paths(month)[2]),
            control_prediction=artifact(control_screen.paths(month)[0]),
            preprocessing=artifact(base.OUT / 'preprocessing' / f'{month}.pkl'))
    modules = [control_screen, boundary.physical, base, boundary, adapter, metadata, base.prototype, base.tensor_seq,
               shared, shared.nni, shared.nnj, shared.hpo, combo, combo.parent]
    current = dict(design=DESIGN, source=artifact(Path(__file__)), cache=cache,
        helpers=[artifact(Path(module.__file__)) for module in modules], folds=sources,
        metadata=artifact(metadata.DEFAULT / 'manifest.json'),
        metadata_values=artifact(metadata.DEFAULT / 'metadata.parquet'),
        selected_result=artifact(combo.OUT / 'result.json'),
        selected_errors=artifact(combo.OUT / 'scored_errors.parquet'))
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        if shared.ts.read(MANIFEST) != current:
            raise ValueError('Frozen boundary recipe, source or inputs changed')
    else:
        if list(OUT.glob('**/*.pt')) or list(OUT.glob('**/*.parquet')):
            raise ValueError('Unrecorded output; inspect before running')
        write_json(MANIFEST, current)
    return current


def paths(month):
    return (OUT / 'networks' / f'{month}_s42.parquet',
            OUT / 'models' / f'{month}_s42.pt', OUT / 'networks' / f'{month}_s42.json')


def completed(month, identity):
    pred, model, record = paths(month)
    if not record.exists():
        if pred.exists() or model.exists():
            raise ValueError('Unrecorded model/prediction; do not refit blindly')
        return None
    value = shared.ts.read(record)
    config = shared.hpo.nn_config(shared.hpo.trial_spec('nn', 'N09')['overrides'])
    if (value['identity'] != artifact(MANIFEST) or value['design'] != DESIGN
            or value['seed'] != 42 or value['month'] != month
            or value['stop_epoch'] != STOP[month] or value['config'] != config
            or value['inputs'] != identity['folds'][month]):
        raise ValueError('Saved boundary model contract changed')
    for field in ['model', 'prediction', 'preprocessing']:
        verify(value[field])
    return value


def fit(months, identity):
    if torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3080':
        raise ValueError('Another GPU needs fresh matched controls')
    config = shared.recipe()
    checks = [boundary.check('cpu'), boundary.check('cuda')]
    training_function()
    write_json(OUT / 'preflight.json', dict(checks=checks, training_bytecode_exact=False,
                                          original_globals_unchanged=True, auxiliary_weight=boundary.SETTINGS["auxiliary_weight"]))
    meta, _ = metadata.load()
    index = pd.read_parquet(base.seq.OUT / 'full_index.parquet')
    points = np.memmap(base.seq.OUT / 'full_points.bin', mode='r', dtype=np.int32).reshape(-1, 6)
    if len(index) != identity['cache']['rows'] or len(points) != identity['cache']['stored_points']:
        raise ValueError('Sequence shape changed')
    tracks = base.tensor_seq.RaggedTracks(index, points, 'cuda')
    (OUT / 'runtime').mkdir(exist_ok=True)
    (OUT / 'networks').mkdir(exist_ok=True)
    for month in months:
        if completed(month, identity) is not None:
            continue
        data = pd.read_pickle(verify(identity['folds'][month]['matrix']))
        names = data['baseline']
        if len(names) != 156 or len(set(names)) != 156:
            raise ValueError('Original156 input contract differs')
        tm, vm = meta.loc[data['train_ids']], meta.loc[data['ids']]
        if (set(tm.month.astype(str)) != set(shared.ts.FOLDS[month])
                or set(vm.month.astype(str)) != {month}
                or np.intersect1d(data['train_ids'], data['ids']).size):
            raise ValueError('Outer fold membership changed')
        np.testing.assert_array_equal(data['y'], tm.taxi_sec)
        train_ids, held_ids = np.asarray(data['train_ids']).copy(), np.asarray(data['ids']).copy()
        item = dict(X=data['X'][names], Vn=data['V'][names], y=data['y'], raw=vm.reset_index(),
                    tmonth=tm.month.astype(str).to_numpy(), meta=dict(month=month))
        del data, tm, vm
        gc.collect()
        with tempfile.TemporaryDirectory(prefix='encoding_', dir=OUT) as scratch:
            d = adapter.ChunkedDirectData(item, 'cuda', scratch)
        state = d.preprocessing()
        state['design'] = base.DESIGN
        pre = verify(identity['folds'][month]['preprocessing'])
        with pre.open('rb') as handle:
            base.preprocess_guard.equal_preprocessing(pickle.load(handle), state)
        del item
        train_rows, held_rows = tracks.align(train_ids), tracks.align(held_ids)
        started = time.monotonic()
        model, prediction, curve = training_function()(d, tracks, train_rows, held_rows, STOP[month])
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('Invalid boundary predictions')
        pred_path, model_path, record_path = paths(month)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix('.pt.tmp')
        torch.save(dict(state_dict=model.state_dict(), config=config, design=DESIGN,
                        preprocessing=str(pre.relative_to(ROOT))), temporary)
        temporary.replace(model_path)
        shared.nni.save_network(pred_path, held_ids, prediction)
        write_json(record_path, dict(identity=artifact(MANIFEST), design=DESIGN, month=month,
            seed=42, stop_epoch=STOP[month], config=config, inputs=identity['folds'][month],
            model=artifact(model_path), prediction=artifact(pred_path), preprocessing=artifact(pre),
            curve=curve, minutes=(time.monotonic()-started)/60, train_rows=len(train_ids),
            held_rows=len(held_ids), host=platform.node(), gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__)))
        del d, model, prediction, train_rows, held_rows
        gc.collect()
        torch.cuda.empty_cache()


def score(identity):
    selected = shared.ts.read(verify(identity['selected_result']))
    for name, digest in selected['inputs'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Selected combination input changed: ' + name)
    sc, rome, _, trees = shared.baseline.baseline()
    calls = pd.read_parquet(combo.sequence.CALLS_PRED).set_index('mvt_id')
    pool = pd.read_parquet(combo.sequence.OUT / 'scored_pool3.parquet').set_index('mvt_id')
    reference = pd.read_parquet(verify(identity['selected_errors'])).set_index('mvt_id')
    arms = ['selected_sparse3_mix', 'matched_physical_single_mix', 'boundary_single_mix']
    errors, rows = {name: {} for name in arms}, []
    for month in STOP:
        record = completed(month, identity)
        if record is None:
            raise ValueError('Both candidate fits must finish')
        ids = sc.d[month]['ids']
        candidate = pd.read_parquet(verify(record['prediction']))
        control = pd.read_parquet(verify(identity['folds'][month]['control_prediction']))
        np.testing.assert_array_equal(candidate.mvt_id, ids)
        np.testing.assert_array_equal(control.mvt_id, ids)
        nn_calls = calls.loc[ids, 'candidate_nn'].to_numpy(float)
        neural = dict(selected_sparse3_mix=.5*(nn_calls+pool.loc[ids, 'candidate_nn'].to_numpy(float)),
            matched_physical_single_mix=.5*(nn_calls+control.base.to_numpy(float)),
            boundary_single_mix=.5*(nn_calls+candidate.base.to_numpy(float)))
        p = .5*combo.parent.cat.probabilities(month, ids)+.5*rome[month]
        members = {shared.hpo.MEMBER_OF[part]: trees[part][42][month] for part in trees}
        path = combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet'
        record = shared.ts.read(path.with_suffix('.json'))
        if sha256(path) != record['prediction_sha256']:
            raise ValueError('Selected main tree prediction changed')
        main = pd.read_parquet(path)
        np.testing.assert_array_equal(main.mvt_id, ids)
        members[shared.hpo.MEMBER_OF['lgb']] = main.pred.to_numpy(float)
        for name in arms:
            errors[name][month] = sc.se(month, 42, nn=neural[name], p=p, **members)
        np.testing.assert_array_equal(errors['selected_sparse3_mix'][month],
                                      reference.loc[ids, 'half_sparse3_combination'])
        rows.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=sc.d[month]['day'],
            candidate_nn=candidate.base.to_numpy(float), control_nn=control.base.to_numpy(float),
            **{name: errors[name][month] for name in arms})))
    frame = pd.concat(rows, ignore_index=True)
    scores = {name: sc.cv(value) for name, value in errors.items()}
    if abs(scores['selected_sparse3_mix']['cv']-selected['scores']['half_sparse3_combination']['cv']) > 1e-9:
        raise ValueError('Selected CV did not reproduce')
    comparisons = {f'boundary_single_mix_vs_{control}': shared.ts.paired_bootstrap(
        frame.day, frame.month, frame[control], frame.boundary_single_mix)
        for control in ['matched_physical_single_mix', 'selected_sparse3_mix']}
    frame.to_parquet(OUT / 'scored.parquet', index=False)
    write_json(OUT / 'score.json', dict(design=DESIGN, identity=artifact(MANIFEST),
        scores=scores, comparisons=comparisons,
        caution='Matched physical single models; selected reference has three original track members. Reused development months.'))
    shared.log(str(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    parser.add_argument('--months', nargs='+', choices=list(STOP), default=list(STOP))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        identity = guard(base.cache_identity())
        if args.stage == 'fit':
            fit(args.months, identity)
        else:
            score(identity)
