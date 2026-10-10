"""Fixed three-member continuation of the audited sparse-track fusion screen.

Preserve k0, original preprocessing, cache and runner bytes. Add only seeds1042
and2042, using original group42 epochs21/17 and the exact original train loop.
An isolated AST specialization changes its three seed literals to one argument;
the numerical training body is otherwise identical. No helper is monkeypatched.

Primary scores are matched3/fullparent15/fixedhalf and a fair matchedhalf.
Secondary, declared after seeing k0: use old15 unchanged for fewer than2points,
otherwise the same fixedhalf mixture; retain an identically gated direct control.
A fixedhalf mixture with existing calls15 is a separate diagnostic, not a search.
"""
import argparse
import ast
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
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
from records import sha256, write_json

ROOT, OUT, shared = base.ROOT, base.OUT, base.shared
NETS = [0, 1, 2]
MANIFEST = OUT/'sparse_pool3_manifest.json'
CALLS_OUT = shared.ts.RUNS/'v3_20261010_nn_callsign_clocks_jan_jul'
CALLS_SCORE = CALLS_OUT/'score_g42_142_242_k0_1_2_3_4.json'
CALLS_PRED = CALLS_OUT/'scored_g42_142_242_k0_1_2_3_4.parquet'
ARMS = ['v58', 'direct3', 'sparse3', 'half_sparse3_v58', 'half_direct3_v58',
        'available_half_sparse3_v58', 'available_half_direct3_v58', 'calls15',
        'half_sparse3_calls15', 'half_direct3_calls15']
DESIGN = dict(parent='v58', candidate='unchanged original156 plus frozen sparse-track fusion',
    group=42, nets=NETS, pool_size=3, inherited=[0], new_members=[1, 2], new_month_member_fits=4,
    seed_formula='42+1000*k', epochs=base.STOP, epoch_policy='original group42 fixed21/17; no inner fit',
    original_single_artifacts='read-only and reused exactly',
    numerical_training_change='only three RNG seed literals become the member seed',
    preprocessing='exact recursive semantic equality; retain original pickle bytes', neural_share=.71,
    arms=ARMS, primary=['sparse3', 'half_sparse3_v58'],
    secondary_availability=dict(choice_time='after k0 result, before k1/k2 fits',
        rule='raw sequence count>=2: fixedhalf sparse3/old15; otherwise originalold15',
        fair_control='same availability gate and fixedhalf historical3/old15', threshold_search=False),
    secondary_callsign=dict(rule='fixedhalf sparse3/calls15 on allrows',
        fair_control='fixedhalf historical3/calls15', weight_search=False),
    exploratory_reused_development_months=True)


def specialized_training_ast():
    """Make the only numerical recipe edits explicit and mechanically bounded."""
    tree = ast.parse(inspect.getsource(base.train))
    function = tree.body[0]
    if function.name != 'train' or [arg.arg for arg in function.args.args] != ['d', 'tracks', 'train_rows', 'held_rows', 'stop']:
        raise ValueError('Original training signature changed')
    function.args.args.append(ast.arg(arg='seed'))
    changed = []
    for node in ast.walk(function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ['manual_seed', 'seed']
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 42):
            changed.append(ast.unparse(node.func))
            node.args[0] = ast.Name(id='seed', ctx=ast.Load())
    if sorted(changed) != sorted(['torch.manual_seed', 'np.random.seed', 'torch.Generator(device=d.device).manual_seed']):
        raise ValueError('Expected exactly original model, NumPy and GPU-permutation seeds')
    return ast.fix_missing_locations(tree)


def compile_training(runtime_dir):
    namespace = dict(vars(base))
    namespace['OUT'] = Path(runtime_dir)
    exec(compile(specialized_training_ast(), '<sparse_pool3_original_train_seed_only>', 'exec'), namespace)
    return namespace['train']


def check_specialization():
    original = ast.parse(inspect.getsource(base.train))
    restored = copy.deepcopy(specialized_training_ast())
    restored.body[0].args.args.pop()
    restored_count = 0
    for node in ast.walk(restored):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ['manual_seed', 'seed'] and len(node.args) == 1
                and isinstance(node.args[0], ast.Name) and node.args[0].id == 'seed'):
            node.args[0] = ast.Constant(value=42)
            restored_count += 1
    assert restored_count == 3
    assert ast.dump(original, include_attributes=False) == ast.dump(restored, include_attributes=False)
    old_train, old_out = base.train, base.OUT
    function = compile_training(OUT/'pool3_runtime'/'synthetic_check')
    assert base.train is old_train and base.OUT == old_out
    assert function.__globals__ is not base.train.__globals__
    assert function.__globals__['OUT'] != base.OUT
    return dict(seed_substitutions=3, all_other_training_ast_exact=True, original_globals_unchanged=True,
        ast_sha256=hashlib.sha256(ast.dump(specialized_training_ast(), include_attributes=False).encode()).hexdigest())


def member_initial_check(d, seed):
    torch.manual_seed(seed)
    parent = shared.nni.PleNet(d.cards, d.bins, d.train[2].shape[1]).to(d.device)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    model = base.prototype.SparseFusion(d.cards, d.bins, d.train[2].shape[1]).to(d.device)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, tensor in parent.state_dict().items():
        assert torch.equal(tensor, model.tabular.state_dict()[name]), name
    tab, track = base.prototype.synthetic(d.cards, d.bins, d.train[2].shape[1], 24, d.device)
    for training in [False, True]:
        parent.train(training)
        model.train(training)
        torch.manual_seed(seed+17)
        old = parent(*tab)
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(d.device).clone()
        torch.manual_seed(seed+17)
        new = model(*tab, *track)
        assert torch.equal(old, new)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(d.device))
    return dict(seed=seed, initial_original_parameters_exact=True,
        initial_train_eval_predictions_exact=True, initialization_dropout_rng_exact=True)


def paths(month, k):
    return (OUT/'networks'/f'{month}_s42_k{k}.parquet', OUT/'models'/f'{month}_s42_k{k}.pt',
            OUT/'networks'/f'{month}_s42_k{k}.json')


def guard(cache):
    base.guard(cache)
    original = {}
    for month in base.STOP:
        record = base.validate_completed(month, cache)
        if record is None:
            raise ValueError('Continuation requires both complete original k0 fits')
        original[month] = dict(record_sha256=sha256(OUT/'networks'/f'{month}_s42.json'),
            prediction_sha256=record['prediction_sha256'], model_sha256=record['model_sha256'],
            preprocessing_sha256=record['preprocessing_sha256'], input_matrix_sha256=record['input_matrix_sha256'])
    calls_report = shared.ts.read(CALLS_SCORE)
    if abs(calls_report['scores']['callsign_clocks']['cv']-295.08866167274704) > 1e-7:
        raise ValueError('Saved full15 callsign reference differs')
    value = dict(design=DESIGN, wrapper_sha256=sha256(Path(__file__)), specialization=check_specialization(),
        original_runner_sha256=sha256(Path(base.__file__)), original_identity_sha256=sha256(OUT/'identity.json'),
        original_score_sha256=sha256(OUT/'score.json'), original=original, cache=cache,
        callsign_reference={str(path.relative_to(ROOT)): sha256(path) for path in
            [CALLS_SCORE, CALLS_PRED, CALLS_OUT/'callsign_pool15_manifest.json']})
    if MANIFEST.exists():
        if shared.ts.read(MANIFEST) != value:
            raise ValueError('Frozen continuation, original artifacts or reference changed')
    else:
        write_json(MANIFEST, value)
    return value


def validate_member(month, k, cache):
    if k == 0:
        return base.validate_completed(month, cache)
    prediction, model, record_path = paths(month, k)
    if not record_path.exists():
        if prediction.exists() or model.exists():
            raise ValueError('Unrecorded continuation artifact: '+str(record_path))
        return None
    record = shared.ts.read(record_path)
    cfg = shared.hpo.nn_config(shared.hpo.trial_spec('nn', 'N09')['overrides'])
    if (record['design'] != DESIGN or record['original_design'] != base.DESIGN or record['config'] != cfg
            or record['month'] != month or record['member'] != k or record['seed'] != 42+1000*k
            or record['stop_epoch'] != base.STOP[month] or record['cache'] != cache
            or record['expansion_manifest_sha256'] != sha256(MANIFEST)):
        raise ValueError('Continuation fit design or provenance changed')
    for path, key in [(prediction, 'prediction_sha256'), (model, 'model_sha256'),
                      (OUT/'preprocessing'/f'{month}.pkl', 'preprocessing_sha256'),
                      (shared.baseline.OUT/'inputs'/f'{month}.pkl', 'input_matrix_sha256')]:
        if sha256(path) != record[key]:
            raise ValueError('Continuation artifact changed: '+str(path))
    return record


def fit(cache):
    cfg = shared.recipe()
    raw = shared.weather_all.raw_lookup()
    index = pd.read_parquet(base.seq.OUT/'full_index.parquet')
    points = np.memmap(base.seq.OUT/'full_points.bin', mode='r', dtype=np.int32).reshape(-1, 6)
    if len(index) != cache['rows'] or len(points) != cache['stored_points']:
        raise ValueError('Full sequence shape differs')
    tracks = base.tensor_seq.RaggedTracks(index, points, 'cuda')
    for month in base.STOP:
        missing = [k for k in [1, 2] if validate_member(month, k, cache) is None]
        if not missing:
            shared.log(f'{month}: all three frozen members already complete')
            continue
        source = shared.baseline.OUT/'inputs'/f'{month}.pkl'
        data = pd.read_pickle(source)
        names = data['baseline']
        if len(names) != 156 or len(set(names)) != 156:
            raise ValueError('Original156 input contract changed')
        X, V = data['X'].loc[:, names], data['V'].loc[:, names]
        months_train = pd.to_datetime(raw.loc[data['train_ids'], 'MVT_TIME_UTC_mvt'], utc=True).dt.strftime('%Y-%m').to_numpy()
        assert set(months_train) == set(shared.ts.FOLDS[month])
        assert not np.intersect1d(data['train_ids'], data['ids']).size
        np.testing.assert_array_equal(data['y'], raw.loc[data['train_ids'], 'taxi_sec'].to_numpy())
        item = dict(X=X, Vn=V, y=data['y'], raw=raw.loc[data['ids']].reset_index(),
                    tmonth=months_train, meta=dict(month=month))
        shared.log(f'{month}: original156 encode once for remaining two fusion members')
        d = base.sensor.DirectData(item, 'cuda')
        state = d.preprocessing()
        state['design'] = base.DESIGN
        pre = OUT/'preprocessing'/f'{month}.pkl'
        before = sha256(pre)
        with pre.open('rb') as handle:
            base.preprocess_guard.equal_preprocessing(pickle.load(handle), state)
        if sha256(pre) != before:
            raise ValueError('Original preprocessing bytes changed')
        train_rows, held_rows = tracks.align(data['train_ids']), tracks.align(data['ids'])
        for k in missing:
            seed = 42+1000*k
            initial = member_initial_check(d, seed)
            runtime = OUT/'pool3_runtime'/f'{month}_s42_k{k}'
            runtime.mkdir(parents=True, exist_ok=True)
            trainer = compile_training(runtime)
            started = time.monotonic()
            model, pred, curve = trainer(d, tracks, train_rows, held_rows, base.STOP[month], seed)
            prediction_path, model_path, record_path = paths(month, k)
            temporary = model_path.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), config=cfg, design=base.DESIGN,
                expansion=DESIGN, preprocessing=str(pre.relative_to(ROOT)), cache=cache), temporary)
            temporary.replace(model_path)
            shared.nni.save_network(prediction_path, data['ids'], pred)
            write_json(record_path, dict(config=cfg, design=DESIGN, original_design=base.DESIGN,
                month=month, member=k, seed=seed, stop_epoch=base.STOP[month], cache=cache,
                initial_check=initial, minutes=(time.monotonic()-started)/60, held_curve=curve,
                expansion_manifest_sha256=sha256(MANIFEST), train_rows=len(d.y), held_rows=len(d.yv),
                input_matrix_sha256=sha256(source), preprocessing_sha256=sha256(pre),
                prediction_sha256=sha256(prediction_path), model_sha256=sha256(model_path)))
            shared.log(f'{month}: group42 k{k}, seed{seed}, frozen fusion saved')
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del d, data, item, X, V, train_rows, held_rows
        gc.collect()
        torch.cuda.empty_cache()


def neural_arms(parent, direct, candidate, calls, available):
    half = .5*(parent+candidate)
    control_half = .5*(parent+direct)
    return dict(v58=parent, direct3=direct, sparse3=candidate,
        half_sparse3_v58=half, half_direct3_v58=control_half,
        available_half_sparse3_v58=np.where(available, half, parent),
        available_half_direct3_v58=np.where(available, control_half, parent),
        calls15=calls, half_sparse3_calls15=.5*(calls+candidate), half_direct3_calls15=.5*(calls+direct))


def score(cache):
    sc, rome, parent, trees = shared.baseline.baseline()
    calls_frame = pd.read_parquet(CALLS_PRED, columns=['mvt_id', 'candidate_nn']).set_index('mvt_id')
    counts = pd.read_parquet(base.seq.OUT/'full_index.parquet', columns=['mvt_id', 'count']).set_index('mvt_id')['count']
    if not calls_frame.index.is_unique or not counts.index.is_unique:
        raise ValueError('Nonunique reference IDs')
    errors = {name: {} for name in ARMS}
    by_seed = {name: {} for name in ARMS}
    rows = []
    for month in base.STOP:
        ids = sc.d[month]['ids']
        direct, candidate = [], []
        for k in NETS:
            if validate_member(month, k, cache) is None:
                raise ValueError('Missing complete continuation member')
            frame = pd.read_parquet(paths(month, k)[0])
            np.testing.assert_array_equal(frame.mvt_id.to_numpy(), ids)
            candidate.append(frame.base.to_numpy(float))
            direct.append(shared.direct_member(sc, month, 42, k))
        dmean, cmean = np.mean(direct, axis=0), np.mean(candidate, axis=0)
        available = counts.loc[ids].to_numpy() >= 2
        calls = calls_frame.loc[ids, 'candidate_nn'].to_numpy(float)
        arms = neural_arms(parent[month], dmean, cmean, calls, available)
        month_errors = {name: [] for name in ARMS}
        for seed in shared.hpo.SEEDS:
            replacements = {shared.hpo.MEMBER_OF[p]: trees[p][seed][month] for p in trees}
            for name, neural in arms.items():
                squared = sc.se(month, seed, nn=neural, p=rome[month], **replacements)
                month_errors[name].append(squared)
                by_seed[name].setdefault(str(seed), {})[month] = float(np.sqrt(squared.mean()))
        for name in ARMS:
            errors[name][month] = np.mean(month_errors[name], axis=0)
        raw = sc.d[month]['held']['raw']
        rows.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=sc.d[month]['day'],
            airport=raw.ADEP_mvt.to_numpy(), y=sc.d[month]['y'], direct_nn=dmean,
            candidate_nn=cmean, sequence_available=available,
            **{f'se_{name}': errors[name][month] for name in ARMS})))
    report = dict(design=DESIGN, manifest_sha256=sha256(MANIFEST), scores={}, comparisons={})
    for name in ARMS:
        report['scores'][name] = dict(**sc.cv(errors[name]), by_tree_seed=by_seed[name])
    for name, expected in [('v58', 295.7414415741871), ('calls15', 295.08866167274704)]:
        if abs(report['scores'][name]['cv']-expected) > 1e-7:
            raise ValueError(name+' control did not reproduce')
    scored = pd.concat(rows, ignore_index=True)
    pairs = [('sparse3', 'direct3'), ('sparse3', 'v58'), ('half_sparse3_v58', 'v58'),
             ('half_sparse3_v58', 'half_direct3_v58'),
             ('available_half_sparse3_v58', 'v58'),
             ('available_half_sparse3_v58', 'available_half_direct3_v58'),
             ('half_sparse3_calls15', 'calls15'), ('half_sparse3_calls15', 'half_direct3_calls15')]
    for arm, control in pairs:
        report['comparisons'][f'{arm}_vs_{control}'] = shared.ts.paired_bootstrap(scored.day.to_numpy(),
            scored.month.to_numpy(), scored[f'se_{control}'].to_numpy(), scored[f'se_{arm}'].to_numpy())
    scored.to_parquet(OUT/'scored_pool3.parquet', index=False)
    write_json(OUT/'score_pool3.json', report)
    shared.log(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'fit', 'score'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'check':
        print(json.dumps(check_specialization(), indent=2))
    else:
        cache = base.cache_identity()
        guard(cache)
        with threadpool_limits(limits=1, user_api='blas'):
            if args.stage == 'fit':
                fit(cache)
            else:
                score(cache)
