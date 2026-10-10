"""Frozen direct N09 plus zero-initialized signed physical-clock gates.

Four already available physical clocks bypass numeric rank transforms at the
output. Nothing is appended to the original156 encoded inputs; direct targets,
ordinary rows and historical fixed21/17 epochs remain unchanged. No inner fit,
clock fallback, schedule/day anchor, target-derived input or search is used.
The extra correction is bounded by3600seconds and is zero if no clock qualifies.
The jointly trained original branch can still change on those rows.

Stages: check (tiny CPU synthetic), fit (only after resource allocation), score.
"""
import argparse
import gc
import json
import math
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

import nn_residual_jan_jul as shared
import nn_sensor_differences_jan_jul as sensor
import nn_aobt_observation_jan_jul as parent_contract
import nn_callsign_pool_resume as preprocess_guard
from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = shared.ts.RUNS / 'v3_20261010_nn_clock_bypass_jan_jul'
CLOCKS = ['mvt_minus_aobt3_sec', 'ab_stand_exit_sec', 'op_own_stand_sec', 'op_ground_sec']
STOP = {'2025-01': 21, '2025-07': 17}
MAX_CONTRAST = 3600.
GATE_SCALE = .25
DESIGN = dict(parent='v58', control='historical N09 seed42k0', original_inputs=156,
    added_features=[], clocks=CLOCKS, clock_interval_inclusive=[0, 10800],
    masks=['genuine AOBT flags', 'ab_matched==1 and ab_stand_n>0', 'op_matched==1', 'op_matched==1'],
    formula='p0 + sum(mask * .25*tanh(W*h+b) * clip(clock-p0,-3600,3600))',
    gate_initialization='all zero, no additional RNG consumption', extra_parameters=1028,
    maximum_absolute_extra_seconds=3600, missing='zero extra branch; shared trunk is jointly trained',
    target='unchanged min(y,10800), direct standardized MSE', stop_epochs=STOP,
    seed=42, member=0, inner_fits=0, schedule='original OneCycle24 horizon', network_share=.71,
    arms=['v58', 'direct_matched', 'clock_bypass', 'half_clock_half_v58', 'half_direct_half_v58'],
    exploratory_reused_development_months=True)


def physical_clocks(X):
    """Raw seconds and independent per-clock validity, without imputation."""
    value = X.loc[:, CLOCKS].to_numpy(dtype=np.float64, copy=True)
    masks = np.isfinite(value) & (value >= 0) & (value <= 10800)
    masks[:, 0] &= ((X.clock_aobt3_available.to_numpy() == 1)
                    & (X.clock_raw_aobt3_missing.to_numpy() == 0))
    masks[:, 1] &= ((X.ab_matched.to_numpy() == 1) & (X.ab_stand_n.to_numpy() > 0))
    for col in [2, 3]:
        masks[:, col] &= X.op_matched.to_numpy() == 1
    # Sanitize before tensor arithmetic; NaN * zero would still be NaN.
    value[~masks] = 0
    return np.ascontiguousarray(value, dtype=np.float32), np.ascontiguousarray(masks)


class ClockData(sensor.DirectData):
    def __init__(self, item, device):
        super().__init__(item, device)
        for attr, frame in [('train', item['X']), ('held', item['Vn'])]:
            clocks, masks = physical_clocks(frame)
            tensors = (torch.from_numpy(clocks).to(device), torch.from_numpy(masks).to(device))
            setattr(self, attr, (*getattr(self, attr), *tensors))

    def preprocessing(self):
        state = super().preprocessing()
        state['design'] = DESIGN
        return state


class ClockBypass(torch.nn.Module):
    def __init__(self, cards, bins, n_plain, target_mu, target_sd):
        super().__init__()
        self.tabular = shared.nni.PleNet(cards, bins, n_plain)
        width = self.tabular.mlp[-1].in_features
        self.gate_weight = torch.nn.Parameter(torch.zeros(len(CLOCKS), width))
        self.gate_bias = torch.nn.Parameter(torch.zeros(len(CLOCKS)))
        self.register_buffer('target_mu', torch.tensor(float(target_mu), dtype=torch.float32))
        self.register_buffer('target_sd', torch.tensor(float(target_sd), dtype=torch.float32))
        if target_sd <= 0:
            raise ValueError('Target standard deviation must be positive')

    def forward_parts(self, cat, g, plain, clocks, masks):
        hidden = self.tabular.inputs(cat, g, plain)
        for layer in list(self.tabular.mlp.children())[:-1]:
            hidden = layer(hidden)
        base = self.tabular.mlp[-1](hidden)
        gate = GATE_SCALE * torch.tanh(torch.nn.functional.linear(hidden, self.gate_weight, self.gate_bias))
        safe = torch.where(masks, clocks, torch.zeros_like(clocks))
        seconds = base * self.target_sd + self.target_mu
        contrast = (safe - seconds).clamp(-MAX_CONTRAST, MAX_CONTRAST)
        delta = (gate * contrast * masks).sum(1, keepdim=True) / self.target_sd
        return base + delta, delta, gate

    def forward(self, cat, g, plain, clocks, masks):
        return self.forward_parts(cat, g, plain, clocks, masks)[0]


def self_check(cards=None, bins=None, n_plain=3, device='cpu'):
    shared.recipe()
    cards = cards if cards is not None else [5, 3]
    bins = bins if bins is not None else [torch.tensor([-2., 0., 2.])] * 2
    torch.manual_seed(42)
    parent = shared.nni.PleNet(cards, bins, n_plain).to(device)
    initial_cpu = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = ClockBypass(cards, bins, n_plain, 1000., 500.).to(device)
    assert torch.equal(initial_cpu, torch.get_rng_state()), 'Initialization consumed extra RNG'
    for name, value in parent.state_dict().items():
        assert torch.equal(value, candidate.tabular.state_dict()[name]), name
    generator = torch.Generator(device=device).manual_seed(713)
    n = 12
    cat = torch.stack([torch.randint(c, (n,), device=device, generator=generator) for c in cards], 1)
    g = torch.randn(n, len(bins), device=device, generator=generator)
    plain = torch.randn(n, n_plain, device=device, generator=generator)
    clocks = torch.arange(n*4, device=device, dtype=torch.float32).reshape(n, 4)*250.
    masks = torch.ones((n, 4), device=device, dtype=torch.bool)
    masks[0] = False
    masks[1, 1:] = False
    clocks[~masks] = torch.nan
    target = torch.randn(n, 1, device=device, generator=generator)
    tab = (cat, g, plain)
    for training in [False, True]:
        parent.train(training)
        candidate.train(training)
        parent.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        torch.manual_seed(91)
        old = parent(*tab)
        torch.nn.functional.mse_loss(old, target).backward()
        rng_cpu = torch.get_rng_state().clone()
        rng_cuda = torch.cuda.get_rng_state(device).clone() if str(device).startswith('cuda') else None
        torch.manual_seed(91)
        new = candidate(*tab, clocks, masks)
        torch.nn.functional.mse_loss(new, target).backward()
        assert torch.equal(old, new), 'Initial prediction differs'
        assert torch.equal(rng_cpu, torch.get_rng_state()), 'Dropout CPU RNG differs'
        if rng_cuda is not None:
            assert torch.equal(rng_cuda, torch.cuda.get_rng_state(device)), 'Dropout CUDA RNG differs'
        for name, parameter in parent.named_parameters():
            other = dict(candidate.tabular.named_parameters())[name]
            assert torch.equal(parameter.grad, other.grad), 'Initial original gradient differs: '+name
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
    candidate.eval()
    parent.eval()
    with torch.no_grad():
        candidate.gate_bias.fill_(100)
        output, delta, _ = candidate.forward_parts(*tab, clocks, masks)
        assert torch.equal(output[:1], parent(*tab)[:1]), 'Missing-clock fallback differs'
        expected = (.25*(torch.where(masks, clocks, torch.zeros_like(clocks))
                    -(parent(*tab)*500.+1000.)).clamp(-3600, 3600)*masks).sum(1, keepdim=True)/500.
        torch.testing.assert_close(delta, expected, rtol=0, atol=0)
        assert (delta.abs()*500 <= 3600.001).all(), 'Correction exceeds fixed bound'
        candidate.gate_bias.fill_(-100)
        negative = candidate.forward_parts(*tab, clocks, masks)[1]
        torch.testing.assert_close(negative, -delta, rtol=0, atol=0)
    return dict(initial_parameters_exact=True, initial_eval_and_train_outputs_exact=True,
        initial_original_gradients_exact=True, initialization_and_dropout_rng_exact=True,
        missing_nan_clock_safe=True, missing_fallback_exact=True, signed_gate_and_bound_exact=True,
        device=str(device))


def train(d, stop):
    torch.manual_seed(42)
    np.random.seed(42)
    model = ClockBypass(d.cards, d.bins, d.train[2].shape[1], d.mu, d.sd).to(d.device)
    opt = torch.optim.AdamW(model.parameters(), lr=shared.nni.NEW['ple']['lr'],
                           weight_decay=shared.nni.NEW['ple']['weight_decay'])
    count, batch = len(d.y), shared.nni.NEW['batch']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=shared.nni.NEW['ple']['lr'],
        total_steps=shared.nni.NEW['epochs']*math.ceil(count/batch))
    generator = torch.Generator(device=d.device).manual_seed(42)
    curve = []
    for epoch in range(stop):
        started, total = time.monotonic(), 0.
        perm = torch.randperm(count, device=d.device, generator=generator)
        for i in range(0, count, batch):
            rows = perm[i:i+batch]
            output = model(*(a[rows] for a in d.train))
            loss = torch.nn.functional.mse_loss(output, d.ty[rows, None].expand_as(output))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            total += float(loss.detach())*len(rows)
        pred = shared.predict(model, d.held, d.sd, d.mu, d.val_offset)
        curve.append(dict(epoch=epoch+1, train_mse_std=total/count,
            rmse=shared.nni.rmse(d.yv, pred), seconds=time.monotonic()-started))
        shared.log(f'epoch {epoch+1}/{stop}: train {total/count:.4f}; monitored {curve[-1]["rmse"]:.4f}')
    return model, pred, curve


def identity():
    modules = [shared, sensor, parent_contract, preprocess_guard, shared.nni, shared.nnj, shared.hpo]
    helpers = {str(Path(m.__file__).relative_to(ROOT)): sha256(Path(m.__file__)) for m in modules}
    for name in ['VALIDATION/opdi.py', 'VALIDATION/adsblol.py', 'pipeline/clock_features.py']:
        helpers[name] = sha256(ROOT/name)
    return dict(source_sha256=sha256(Path(__file__)), helpers=helpers,
                design=DESIGN, control=parent_contract.control_identity())


def guard():
    value = identity()
    path = OUT/'identity.json'
    if path.exists():
        if shared.ts.read(path) != value:
            raise ValueError('Frozen source, helpers, controls or design changed')
    else:
        if list((OUT/'networks').glob('*.parquet')):
            raise ValueError('Existing predictions lack provenance')
        write_json(path, value)
        (OUT/'runner_initial.py').write_text(Path(__file__).read_text())


def validate_completed(month):
    path = OUT/'networks'/f'{month}_s42.json'
    if not path.exists():
        return None
    value = shared.ts.read(path)
    cfg = shared.hpo.nn_config(shared.hpo.trial_spec('nn', 'N09')['overrides'])
    if value['config'] != cfg or value['design'] != DESIGN or value['stop_epoch'] != STOP[month] or value['month'] != month:
        raise ValueError('Completed fit recipe differs')
    for source, key in [(OUT/'networks'/f'{month}_s42_k0.parquet', 'prediction_sha256'),
                        (OUT/'models'/f'{month}_s42_k0.pt', 'model_sha256'),
                        (OUT/'preprocessing'/f'{month}.pkl', 'preprocessing_sha256'),
                        (shared.baseline.OUT/'inputs'/f'{month}.pkl', 'input_matrix_sha256')]:
        if sha256(source) != value[key]:
            raise ValueError('Completed artifact changed: '+str(source))
    return value


def fit(months):
    cfg = shared.recipe()
    raw = shared.weather_all.raw_lookup()
    for month in months:
        if validate_completed(month) is not None:
            shared.log(f'{month}: completed frozen fit reused')
            continue
        dest = OUT/'networks'/f'{month}_s42_k0.parquet'
        model_path = OUT/'models'/f'{month}_s42_k0.pt'
        if dest.exists() or model_path.exists():
            raise ValueError('Unrecorded model/prediction; inspect interrupted save')
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
        shared.log(f'{month}: original156 encode {len(X):,} train/{len(V):,} held')
        d = ClockData(item, 'cuda')
        pre = OUT/'preprocessing'/f'{month}.pkl'
        state = d.preprocessing()
        if pre.exists():
            with pre.open('rb') as handle:
                preprocess_guard.equal_preprocessing(pickle.load(handle), state)
        else:
            shared.atomic_pickle(pre, state)
        initial = self_check(d.cards, d.bins, d.train[2].shape[1], 'cuda')
        started = time.monotonic()
        model, pred, curve = train(d, STOP[month])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix('.pt.tmp')
        torch.save(dict(state_dict=model.state_dict(), config=cfg, design=DESIGN,
                        preprocessing=str(pre.relative_to(ROOT))), temporary)
        temporary.replace(model_path)
        shared.nni.save_network(dest, data['ids'], pred)
        write_json(OUT/'networks'/f'{month}_s42.json', dict(config=cfg, design=DESIGN, month=month,
            stop_epoch=STOP[month], initial_check=initial, minutes=(time.monotonic()-started)/60,
            held_curve=curve, train_rows=len(d.y), held_rows=len(d.yv),
            physical_valid_train=d.train[-1].sum(0).cpu().tolist(), physical_valid_held=d.held[-1].sum(0).cpu().tolist(),
            input_matrix_sha256=sha256(source), preprocessing_sha256=sha256(pre),
            prediction_sha256=sha256(dest), model_sha256=sha256(model_path)))
        shared.log(f'{month}: frozen clock bypass saved')
        del model, d, data, X, V, item
        gc.collect()
        torch.cuda.empty_cache()


def score():
    sc, rome, pool, trees = shared.baseline.baseline()
    errors = {name: {} for name in DESIGN['arms']}
    by_seed = {name: {} for name in errors}
    rows = []
    for month in shared.ts.SPLITS:
        if validate_completed(month) is None:
            raise ValueError('No complete candidate fit')
        frame = pd.read_parquet(OUT/'networks'/f'{month}_s42_k0.parquet')
        np.testing.assert_array_equal(frame.mvt_id.to_numpy(), sc.d[month]['ids'])
        direct = shared.direct_member(sc, month, 42, 0)
        candidate = frame.base.to_numpy(float)
        arms = dict(v58=pool[month], direct_matched=direct, clock_bypass=candidate,
                    half_clock_half_v58=.5*(pool[month]+candidate), half_direct_half_v58=.5*(pool[month]+direct))
        month_errors = {name: [] for name in arms}
        for seed in shared.hpo.SEEDS:
            replacements = {shared.hpo.MEMBER_OF[p]: trees[p][seed][month] for p in trees}
            for name, neural in arms.items():
                squared = sc.se(month, seed, nn=neural, p=rome[month], **replacements)
                month_errors[name].append(squared)
                by_seed[name].setdefault(str(seed), {})[month] = float(np.sqrt(squared.mean()))
        for name in arms:
            errors[name][month] = np.mean(month_errors[name], axis=0)
        raw = sc.d[month]['held']['raw']
        rows.append(pd.DataFrame(dict(mvt_id=sc.d[month]['ids'], month=month, day=sc.d[month]['day'],
            airport=raw.ADEP_mvt.to_numpy(), y=sc.d[month]['y'], direct_nn=direct, candidate_nn=candidate,
            **{f'se_{name}': errors[name][month] for name in arms})))
    report = dict(design=DESIGN, scores={}, comparisons={})
    for name in errors:
        report['scores'][name] = dict(**sc.cv(errors[name]), by_tree_seed=by_seed[name])
    if abs(report['scores']['v58']['cv']-295.7414415741871) > 1e-7:
        raise ValueError('v58 control did not reproduce')
    scored = pd.concat(rows, ignore_index=True)
    for arm, control in [('clock_bypass', 'direct_matched'), ('clock_bypass', 'v58'),
                         ('half_clock_half_v58', 'v58'), ('half_clock_half_v58', 'half_direct_half_v58')]:
        report['comparisons'][f'{arm}_vs_{control}'] = shared.ts.paired_bootstrap(scored.day.to_numpy(),
            scored.month.to_numpy(), scored[f'se_{control}'].to_numpy(), scored[f'se_{arm}'].to_numpy())
    scored.to_parquet(OUT/'scored.parquet', index=False)
    write_json(OUT/'score.json', report)
    shared.log(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'fit', 'score'])
    parser.add_argument('--months', nargs='+', choices=list(STOP), default=list(STOP))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'check':
        print(json.dumps(self_check(), indent=2))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'networks').mkdir(exist_ok=True)
        guard()
        with threadpool_limits(limits=1, user_api='blas'):
            if args.stage == 'fit':
                fit(args.months)
            else:
                score()
