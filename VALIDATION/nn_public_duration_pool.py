"""Fixed time-weighted track pooling alongside the existing mean/max pool.

Adjacent retained observations receive half of their interval at each endpoint,
with each interval capped at 60 seconds. These are pooling weights, not measured
dwell times or a claim that missing intervals were observed.
"""
import torch
from torch import nn

import nn_public_scratch as parent

physical, features, transform = parent.physical, parent.features, parent.transform
SETTINGS = dict(parent=parent.SETTINGS,
    change='add centered time-weighted track pool through zero linear projection',
    weights='half adjacent positive retained intervals, each capped at60seconds',
    missing='only pairs with both points present; no usable interval falls back to ordinary mean',
    projection='Linear(width32,pooled64,bias=False), zero initialization,2048weights',
    interpretation='sampling-time pooling, not observed taxi or dwell duration',
    inputs='unchanged168 tabular/public inputs and cached128-point sequences',
    training='same original sparse update,21/17 epochs,seed42,original batch and schedule')


def interval_weights(points, mask):
    x = points * mask.unsqueeze(-1)
    seconds = (x[:, :-1, 0] - x[:, 1:, 0]) * 7200
    valid = mask[:, :-1] & mask[:, 1:] & (seconds > 0)
    intervals = seconds.clamp(0, 60) * valid / 2
    return nn.functional.pad(intervals, (0, 1)) + nn.functional.pad(intervals, (1, 0))


class DurationTrackEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.stem, self.blocks, self.physics_stem = original.stem, original.blocks, original.physics_stem
        width = physical.parent.SETTINGS['width']
        self.duration_projection = nn.Linear(width, 2*width, bias=False)
        nn.init.zeros_(self.duration_projection.weight)

    def forward(self, points, mask):
        present = mask.unsqueeze(-1)
        value = nn.functional.silu(self.stem(points*present)
                + self.physics_stem(physical.physical_features(points, mask))) * present
        for block in self.blocks: value = block(value, present)
        mean = value.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum = value.masked_fill(~present, -torch.inf).amax(1)
        maximum = torch.where(mask.any(1).unsqueeze(1), maximum, torch.zeros_like(maximum))
        weights = interval_weights(points, mask)
        total = weights.sum(1, keepdim=True)
        weighted = (value*weights.unsqueeze(-1)).sum(1) / total.clamp(min=1e-12)
        centered = torch.where(total > 0, weighted-mean, torch.zeros_like(mean))
        return torch.cat([mean, maximum], dim=1) + self.duration_projection(centered)


class SparseFusion(parent.SparseFusion):
    def __init__(self, cards, bins, n_plain):
        super().__init__(cards, bins, n_plain)
        with torch.random.fork_rng(devices=[]):
            self.track = DurationTrackEncoder(self.track)


def pooling_check(device):
    # Independent hand calculations include irregular cadence, a capped gap,
    # isolated/absent observations and a deliberately nonincreasing timestamp.
    times = torch.tensor([[100, 90, 60, -100], [10, 20, 5, 0], [9, 8, 7, 6], [0, 0, 0, 0]],
                         dtype=torch.float64, device=device)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                        dtype=torch.bool, device=device)
    points = torch.zeros(4, 4, 16, dtype=torch.float64, device=device)
    points[:, :, 0] = times / 7200
    expected = torch.tensor([[5, 20, 45, 30], [0, 7.5, 7.5, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                            dtype=torch.float64, device=device)
    weights = interval_weights(points, mask)
    torch.testing.assert_close(weights, expected, rtol=0, atol=1e-12)
    values = torch.tensor([1., 2., 4., 8.], dtype=torch.float64, device=device)
    torch.testing.assert_close((weights[0]*values).sum()/weights[0].sum(),
                               torch.tensor(4.65, dtype=torch.float64, device=device), rtol=0, atol=1e-12)
    altered = points.clone(); altered[~mask] = 999
    torch.testing.assert_close(interval_weights(altered, mask), weights, rtol=0, atol=0)
    return dict(hand_interval_and_weighted_mean_exact=True, capped_gap_and_invalid_pair_checked=True,
                isolated_absent_and_padding_checked=True)


def check(device='cpu'):
    physical.parent.shared.recipe()
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])]*3, 5
    torch.manual_seed(42); old = parent.SparseFusion(cards, bins, plain+12).to(device)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42); new = SparseFusion(cards, bins, plain+12).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items(): assert torch.equal(value, new.state_dict()[name]), name
    assert new.track.duration_projection.weight.numel() == 2048
    tab, track = physical.parent.synthetic(cards, bins, plain, 24, device)
    tab = (*tab[:2], torch.cat([tab[2], torch.linspace(-2, 2, 12, device=device).repeat(24, 1)], dim=1))
    with torch.no_grad():
        for model in [old, new]:
            model.public[-1].weight.fill_(.02)
            model.fusion[-1].weight.fill_(.01)
            model.track.physics_stem.weight.fill_(.02)
    maximum_gradient_error = 0.
    for training in [False, True]:
        old.train(training); new.train(training)
        torch.manual_seed(912); expected = old(*tab, *track)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(device).clone() if device != 'cpu' else None
        torch.manual_seed(912); actual = new(*tab, *track)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if gpu_rng is not None: assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
        old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
        expected.square().mean().backward(); actual.square().mean().backward()
        parameters = dict(new.named_parameters())
        for name, value in old.named_parameters():
            other = parameters[name]
            assert (value.grad is None) == (other.grad is None), name
            if value.grad is not None:
                tolerance = 0.
                # The extra zero branch changes CUDA reduction accumulation
                # within the track graph. Keep tabular/public gradients exact.
                if device != 'cpu' and name.startswith('track.'):
                    tolerance = 8*torch.finfo(value.grad.dtype).eps*float(torch.maximum(value.grad.abs(), other.grad.abs()).max())
                torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=tolerance)
                maximum_gradient_error = max(maximum_gradient_error, float((value.grad-other.grad).abs().max()))
        gradient = new.track.duration_projection.weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    old.eval(); new.eval()
    with torch.no_grad():
        new.track.duration_projection.weight.fill_(.03)
        actual = new(*tab, *track); reference = old(*tab, *track)
        assert (actual-reference).abs().max() > 0
        absent = track[1].sum(1) < 2
        torch.testing.assert_close(actual[absent], reference[absent], rtol=0, atol=0)
        altered = track[0].clone(); altered[~track[1]] = 999
        torch.testing.assert_close(new(*tab, altered, *track[1:]), actual, rtol=0, atol=0)
    return dict(device=device, original_weights_outputs_and_rng_exact=True,
        original_gradients_checked=True, gradient_max_absolute_error=maximum_gradient_error,
        cuda_track_gradient_tolerance='8 * float epsilon * tensor maximum; all other gradients exact',
        nonzero_projection_effect_and_gradient=True, missing_track_fallback_and_padding_exact=True,
        pooling=pooling_check(device), added_parameters=2048)
