"""One fixed context-conditioned attention residual on the original track model.

All original parameters and initial predictions are preserved. Four query
heads attend over the existing 32-channel track representation. No new inputs,
dropout, target, missingness gate or stopping choice is introduced.
"""
import math
import torch
from torch import nn

import nn_sparse_track_prototype as parent

SETTINGS = dict(heads=4, head_width=8, hidden=128,
                query='original 256-dimensional tabular hidden state',
                keys_values='existing 32-channel temporal convolution output',
                pooling='masked dot-product softmax; empty context exactly zero',
                output='additional zero-initialized residual; same count>=2 gate',
                original_branch='all weights and initialization RNG retained',
                additional_dropout=0)


class SparseFusion(parent.SparseFusion):
    def __init__(self, cards, bins, n_plain):
        super().__init__(cards, bins, n_plain)
        with torch.random.fork_rng(devices=[]):
            self.attention_query = nn.Linear(256, 32)
            self.attention_key = nn.Linear(32, 32)
            self.attention_value = nn.Linear(32, 32)
            self.attention_output = nn.Sequential(
                nn.Linear(256 + 32 + len(parent.seq.QUALITY_NAMES), 128), nn.SiLU(),
                nn.Linear(128, 1))
            nn.init.zeros_(self.attention_output[-1].weight)
            nn.init.zeros_(self.attention_output[-1].bias)

    def context(self, hidden, value, mask):
        n, length, _ = value.shape
        query = self.attention_query(hidden).reshape(n, 4, 8)
        keys = self.attention_key(value).reshape(n, length, 4, 8)
        values = self.attention_value(value).reshape(n, length, 4, 8)
        logits = torch.einsum('bhd,blhd->blh', query, keys) / math.sqrt(8)
        # Finite masked logits avoid all-empty softmax NaNs. Normalize valid
        # weights again so empty tracks have exactly zero context.
        weights = logits.masked_fill(~mask.unsqueeze(-1), -1e4).softmax(1)
        weights = weights * mask.unsqueeze(-1)
        weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-12)
        return torch.einsum('blh,blhd->bhd', weights, values).reshape(n, 32)

    def forward(self, cat, g, plain, points, mask, quality):
        hidden = self.tabular.inputs(cat, g, plain)
        for layer in list(self.tabular.mlp.children())[:-1]:
            hidden = layer(hidden)
        base = self.tabular.mlp[-1](hidden)
        # Exact original TrackEncoder operations, retaining its per-point output
        # for attention. Neither original parameters nor its pooled path change.
        present = mask.unsqueeze(-1)
        value = torch.nn.functional.silu(self.track.stem(points * present)) * present
        for block in self.track.blocks:
            value = block(value, present)
        mean = value.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum = value.masked_fill(~present, -torch.inf).amax(1)
        maximum = torch.where(mask.any(1).unsqueeze(1), maximum, torch.zeros_like(maximum))
        representation = torch.cat([mean, maximum], dim=1)
        delta = self.fusion(torch.cat([hidden, representation, quality], dim=1))
        extra = self.attention_output(torch.cat([hidden, self.context(hidden, value, mask), quality], dim=1))
        return base + (delta + extra) * (mask.sum(1) >= 2).unsqueeze(1)


def check(device='cpu'):
    cards, bins, plain_width = [3, 5], [torch.tensor([-1., 0., 1., 2.])] * 3, 5
    parent.shared.recipe()
    torch.manual_seed(42)
    old = parent.SparseFusion(cards, bins, plain_width).to(device)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    new = SparseFusion(cards, bins, plain_width).to(device)
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    tab, track = parent.synthetic(cards, bins, plain_width, 12, device)
    points, mask, quality = track
    mask[0] = False
    mask[1] = False
    mask[1, 0] = True
    # Exercise the preserved pooled path with a nonzero original residual.
    with torch.no_grad():
        old.fusion[-1].weight.fill_(.01)
        old.fusion[-1].bias.fill_(.03)
        new.load_state_dict(old.state_dict(), strict=False)
    for training in [False, True]:
        old.train(training); new.train(training)
        torch.manual_seed(913)
        expected = old(*tab, *track)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(device).clone() if device != 'cpu' else None
        torch.manual_seed(913)
        actual = new(*tab, *track)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if gpu_rng is not None:
            assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
    new.eval(); old.eval()
    with torch.no_grad():
        new.attention_output[-1].weight.fill_(.02)
        new.attention_output[-1].bias.fill_(.04)
    actual = new(*tab, *track)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[:2], old(*tab, *track)[:2], rtol=0, atol=0)
    changed_points = points.clone()
    changed_points[~mask] = 123.0
    torch.testing.assert_close(new(*tab, changed_points, mask, quality), actual, rtol=0, atol=0)
    actual.square().mean().backward()
    grad = new.attention_query.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    return dict(device=device, original_parameters_rng_and_nonzero_pool_path_exact=True,
                train_eval_and_dropout_rng_exact=True, zero_one_point_gate_exact=True,
                padding_invariant=True, query_gradient_finite_nonzero=True)
