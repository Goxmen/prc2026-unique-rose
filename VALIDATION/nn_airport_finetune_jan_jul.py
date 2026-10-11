"""Fixed two-fold airport fine-tuning screen on the selected v64 stack.

Two fixed continuation epochs from saved physical seed42 fold models.
Global and airport-specific continuations share inputs, labels and optimizer.
Both variants run sequentially on the desktop. No upload.
"""
import argparse
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
import nn_airport_finetune as airport_tune
import nn_track_physical_scales_jan_jul as control_screen
import chunked_direct_data as adapter
import missing_plan_view as metadata
import sparse_track_combo_score as combo
from records import sha256, write_json

ROOT, shared = base.ROOT, base.shared
RUN_ROOT = shared.ts.RUNS / 'v3_20261011_nn_airport_finetune'
ARM = 'global'
OUT = RUN_ROOT / ARM
MANIFEST = OUT / 'identity.json'
STOP = dict(base.STOP)
DESIGN = dict(parent='selected half sparse3/half calls15 combination',
    change='two low-rate continuation epochs, global or separate complete airport networks',
    original_columns=156, added_columns=[], model=airport_tune.SETTINGS,
    original_model=base.DESIGN['model'], sequence=base.DESIGN['sequence'],
    seed=42, parent_epochs=STOP, finetune_epochs=2, target=base.DESIGN['target'],
    training='fixed two-epoch AdamW continuation; compare global with airport-specific updates',
    preprocessing='exact original156 preprocessing, checked against saved control',
    machine='desktop RTX3080; same-machine saved physical-scale seed42 controls',
    primary='airport continuation versus global continuation; both from identical saved physical models',
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


def train(d,tracks,train_rows,held_rows,parent_stop):
    month=next(m for m in STOP if STOP[m]==parent_stop)
    rec=shared.ts.read(control_screen.paths(month)[2])
    source=control_screen.verify(rec['model'])
    state=torch.load(source,map_location='cpu',weights_only=False)['state_dict']
    col=d.encoder.cats.index('airport')
    if ARM=='global':
        model=airport_tune.physical.SparseFusion(d.cards,d.bins,d.train[2].shape[1]).to(d.device)
        model.load_state_dict(state);jobs=[('global',model,torch.arange(len(d.y),device=d.device))]
    else:
        model=airport_tune.AirportFusion(d.cards,d.bins,d.train[2].shape[1],col).to(d.device)
        model.initialize(state)
        jobs=[(key,expert,torch.nonzero(d.train[0][:,col]==int(key)).flatten()) for key,expert in model.experts.items()]
        assert sum(len(idx) for _,_,idx in jobs)==len(d.y)
    curve=[];batch=airport_tune.SETTINGS['batch']
    for name,expert,indices in jobs:
        assert len(indices)>0
        torch.manual_seed(42);np.random.seed(42);expert.train()
        opt=torch.optim.AdamW(expert.parameters(),lr=airport_tune.SETTINGS['learning_rate'],
            weight_decay=shared.nni.NEW['ple']['weight_decay'])
        generator=torch.Generator(device=d.device).manual_seed(42)
        for epoch in range(airport_tune.SETTINGS['epochs']):
            started=time.monotonic();total=0.;perm=indices[torch.randperm(len(indices),device=d.device,generator=generator)]
            cpu_perm=perm.cpu().numpy()
            for i in range(0,len(perm),batch):
                ix=perm[i:i+batch];track_batch=tracks.batch(train_rows[cpu_perm[i:i+batch]])
                output=expert(*(x[ix] for x in d.train),*track_batch)
                loss=torch.nn.functional.mse_loss(output,d.ty[ix,None].expand_as(output))
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss.detach())*len(ix)
            curve.append(dict(expert=name,epoch=epoch+1,rows=len(indices),train_mse_std=total/len(indices),seconds=time.monotonic()-started))
            shared.log(f'{ARM} {month} expert {name} epoch {epoch+1}/2: {total/len(indices):.5f}; {curve[-1]["seconds"]:.1f}s')
    pred=base.predict(model,d,tracks,held_rows)
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
    modules = [control_screen, airport_tune.physical, base, airport_tune, adapter, metadata, base.prototype, base.tensor_seq,
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
            raise ValueError('Frozen airport_tune recipe, source or inputs changed')
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
            or value['parent_epoch'] != STOP[month] or value['config'] != config
            or value['inputs'] != identity['folds'][month]):
        raise ValueError('Saved airport_tune model contract changed')
    for field in ['model', 'prediction', 'preprocessing']:
        verify(value[field])
    return value


def fit(months, identity):
    if torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3080':
        raise ValueError('Another GPU needs fresh matched controls')
    config = shared.recipe()
    checks = [airport_tune.check('cpu'), airport_tune.check('cuda')]
    training_function()
    write_json(OUT / 'preflight.json', dict(checks=checks, training_bytecode_exact=False,
                                          original_globals_unchanged=True))
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
            raise ValueError('Invalid airport_tune predictions')
        pred_path, model_path, record_path = paths(month)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix('.pt.tmp')
        torch.save(dict(state_dict=model.state_dict(), config=config, design=DESIGN,
                        preprocessing=str(pre.relative_to(ROOT))), temporary)
        temporary.replace(model_path)
        shared.nni.save_network(pred_path, held_ids, prediction)
        write_json(record_path, dict(identity=artifact(MANIFEST), design=DESIGN, month=month,
            seed=42, parent_epoch=STOP[month], config=config, inputs=identity['folds'][month],
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
    arms = ['selected_sparse3_mix', 'matched_physical_single_mix', 'airport_tune_single_mix']
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
            airport_tune_single_mix=.5*(nn_calls+candidate.base.to_numpy(float)))
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
    comparisons = {f'airport_tune_single_mix_vs_{control}': shared.ts.paired_bootstrap(
        frame.day, frame.month, frame[control], frame.airport_tune_single_mix)
        for control in ['matched_physical_single_mix', 'selected_sparse3_mix']}
    frame.to_parquet(OUT / 'scored.parquet', index=False)
    write_json(OUT / 'score.json', dict(design=DESIGN, identity=artifact(MANIFEST),
        scores=scores, comparisons=comparisons,
        caution='Matched physical single models; selected reference has three original track members. Reused development months.'))
    shared.log(str(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    parser.add_argument('--arm',choices=['global','airport'],required=True)
    parser.add_argument('--months', nargs='+', choices=list(STOP), default=list(STOP))
    args = parser.parse_args()
    ARM=args.arm
    OUT=RUN_ROOT/ARM
    MANIFEST=OUT/'identity.json'
    DESIGN=dict(DESIGN,variant=ARM)
    torch.set_num_threads(2)
    with threadpool_limits(2):
        identity = guard(base.cache_identity())
        if args.stage == 'fit':
            fit(args.months, identity)
        else:
            score(identity)
