"""Untrained N09 + sparse temporal convolution prototype and bounded benchmark.

This module never loads training targets or fits a model. The tabular N09 branch
is created first with its original RNG stream. All new parameters are created in
a forked RNG context; a zero final residual head preserves the original output.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import torch
from torch import nn

import nn_residual_jan_jul as shared
import sparse_track_sequence as seq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tmp/research_20261010/model_diversity'
PREPROCESSING = ROOT / 'VALIDATION/runs/v3_20261010_nn_residual_jan_jul/preprocessing/2025-01.pkl'
SETTINGS = dict(sequence_channels=seq.CHANNEL_NAMES, quality_channels=seq.QUALITY_NAMES,
    width=32, residual_blocks=3, convolutions_per_block=2, kernel=3, dilations=[1, 4, 16],
    activation='SiLU', normalization='LayerNorm over channels independently per point',
    pooling='masked mean and max', fusion_hidden=128, final_residual_initialization='zero',
    usable='at least two distinct timestamps', original_inputs=156,
    extra_dropout=0, training=False, benchmark_batch=2048, benchmark_repeats=5)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])

    def forward(self, x, mask):
        value = x
        for conv, norm in zip(self.convs, self.norms):
            value = conv(value.transpose(1, 2)).transpose(1, 2)
            value = torch.nn.functional.silu(norm(value)) * mask
        return (value+x) * mask


class TrackEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(len(seq.CHANNEL_NAMES), SETTINGS['width'])
        self.blocks = nn.ModuleList([TemporalBlock(SETTINGS['width'], d) for d in SETTINGS['dilations']])

    def forward(self, points, mask):
        present = mask.unsqueeze(-1)
        value = torch.nn.functional.silu(self.stem(points * present)) * present
        for block in self.blocks:
            value = block(value, present)
        mean = value.sum(1)/mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum = value.masked_fill(~present, -torch.inf).amax(1)
        maximum = torch.where(mask.any(1).unsqueeze(1), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1)


class SparseFusion(nn.Module):
    def __init__(self, cards, bins, n_plain):
        super().__init__()
        # Calling code first applies the frozen N09 recipe exactly as its parent.
        self.tabular = shared.nni.PleNet(cards, bins, n_plain)
        with torch.random.fork_rng(devices=[]):
            self.track = TrackEncoder()
            self.fusion = nn.Sequential(nn.Linear(256+2*SETTINGS['width']+len(seq.QUALITY_NAMES),
                                                 SETTINGS['fusion_hidden']), nn.SiLU(),
                                        nn.Linear(SETTINGS['fusion_hidden'], 1))
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, cat, g, plain, points, mask, quality):
        hidden = self.tabular.inputs(cat, g, plain)
        for layer in list(self.tabular.mlp.children())[:-1]:
            hidden = layer(hidden)
        base = self.tabular.mlp[-1](hidden)
        representation = self.track(points, mask)
        delta = self.fusion(torch.cat([hidden, representation, quality], dim=1))
        return base + delta * (mask.sum(1) >= 2).unsqueeze(1)


def synthetic(cards, bins, n_plain, batch, device):
    """Mixed absent, single, sparse and dense sequences; no real training rows."""
    generator = torch.Generator(device=device).manual_seed(1729)
    cat = torch.stack([torch.randint(card, (batch,), generator=generator, device=device) for card in cards], dim=1)
    g = torch.randn(batch, len(bins), device=device, generator=generator)
    plain = torch.randn(batch, n_plain, device=device, generator=generator).clamp(-3, 3)
    points = torch.rand(batch, seq.MAX_POINTS, len(seq.CHANNEL_NAMES), device=device, generator=generator)
    counts = torch.tensor([0, 1, 2, 8, 32, 128], device=device).repeat((batch+5)//6)[:batch]
    mask = torch.arange(seq.MAX_POINTS, device=device).unsqueeze(0) < counts.unsqueeze(1)
    points *= mask.unsqueeze(-1)
    quality = torch.rand(batch, len(seq.QUALITY_NAMES), device=device, generator=generator)
    return (cat, g, plain), (points, mask, quality)


def self_check(cards, bins, n_plain, device='cpu'):
    shared.recipe()
    torch.manual_seed(42)
    baseline = shared.nni.PleNet(cards, bins, n_plain).to(device)
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(42)
    model = SparseFusion(cards, bins, n_plain).to(device)
    assert torch.equal(expected_state, torch.get_rng_state()), 'New initialization consumed original CPU RNG'
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, model.tabular.state_dict()[key]), key
    tab, track = synthetic(cards, bins, n_plain, 24, device)
    for training in [False, True]:
        baseline.train(training)
        model.train(training)
        torch.manual_seed(222)
        old = baseline(*tab)
        old_cpu = torch.get_rng_state().clone()
        old_gpu = torch.cuda.get_rng_state(device).clone() if str(device).startswith('cuda') else None
        torch.manual_seed(222)
        new = model(*tab, *track)
        assert torch.equal(old, new), f'Initial prediction differs, training={training}'
        assert torch.equal(old_cpu, torch.get_rng_state()), 'Dropout CPU RNG differs'
        if old_gpu is not None:
            assert torch.equal(old_gpu, torch.cuda.get_rng_state(device)), 'Dropout CUDA RNG differs'
    model.eval()
    baseline.eval()
    with torch.no_grad():
        model.fusion[-1].bias.fill_(1)
        output = model(*tab, *track)
        expected = baseline(*tab)+(track[1].sum(1) >= 2).unsqueeze(1)
        assert torch.equal(output, expected), 'No-sequence fallback changed'
        altered = track[0].clone()
        altered[~track[1]] = 10000
        assert torch.equal(model.track(track[0], track[1]), model.track(altered, track[1])), 'Padding influenced representation'
        assert torch.isfinite(model.track(*track[:2])).all(), 'Empty max pool is nonfinite'
    return dict(initial_parameters_exact=True, initial_eval_predictions_exact=True,
                initial_train_predictions_exact=True, dropout_rng_exact=True,
                absent_and_one_point_fallback_exact=True, padded_values_ignored=True,
                device=str(device))


def benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU benchmark requires the allocated desktop CUDA slot')
    torch.set_num_threads(2)
    shared.recipe()
    with PREPROCESSING.open('rb') as handle:
        prep = pickle.load(handle)
    assert len(prep['columns']) == 156
    cards, bins, n_plain = prep['cards'], prep['bins'], len(prep['plain'])
    result = dict(settings=SETTINGS, source_sha256=shared.sha256(Path(__file__)),
        sequence_source_sha256=shared.sha256(Path(seq.__file__)),
        preprocessing=str(PREPROCESSING.relative_to(ROOT)), preprocessing_sha256=shared.sha256(PREPROCESSING),
        original_dimensions=dict(columns=156, categoricals=len(cards), ple=len(bins), plain=n_plain),
        gpu=torch.cuda.get_device_name(0), check=self_check(cards, bins, n_plain, 'cuda'), measurements={})
    for name in ['n09', 'sparse_fusion']:
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        model = (shared.nni.PleNet(cards, bins, n_plain) if name == 'n09' else
                 SparseFusion(cards, bins, n_plain)).cuda().train()
        tab, track = synthetic(cards, bins, n_plain, SETTINGS['benchmark_batch'], 'cuda')
        args = tab if name == 'n09' else (*tab, *track)
        target = torch.zeros((SETTINGS['benchmark_batch'], 1), device='cuda')
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            (model(*args)-target).square().mean().backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(SETTINGS['benchmark_repeats']):
            model.zero_grad(set_to_none=True)
            (model(*args)-target).square().mean().backward()
        torch.cuda.synchronize()
        result['measurements'][name] = dict(batch=SETTINGS['benchmark_batch'],
            forward_backward_ms=1000*(time.perf_counter()-started)/SETTINGS['benchmark_repeats'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in model.parameters()),
            finite_gradients=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
        del model, tab, track, args, target
    result['forward_backward_ratio'] = result['measurements']['sparse_fusion']['forward_backward_ms']/result['measurements']['n09']['forward_backward_ms']
    result['limitations'] = ['Synthetic batches only; no optimizer updates or training labels.',
        'Memory excludes full encoded tabular matrices, optimizer moments and real sequence collation.',
        'Runtime excludes CPU ragged gathers and transfer; this does not predict validation gain.']
    shared.write_json(OUT/'sparse_track_benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', required=True)
    args = parser.parse_args()
    benchmark()
