"""Add a ground-only mean/max readout to the physical track encoder.

All original parameters and RNG streams are retained. The additional linear
readout starts at zero and uses the existing ground flag, without new data.
"""
import torch
from torch import nn
import nn_track_physical_scales as physical

SETTINGS = dict(parent=physical.SETTINGS, change='ground-only mean/max readout',
    ground_channel=9, ground_threshold=.5, output_width=64,
    initialization='zero linear readout without bias', extra_dropout=0)


class GroundTrackEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.stem = original.stem
        self.blocks = original.blocks
        self.physics_stem = original.physics_stem
        self.ground_readout = nn.Linear(64, 64, bias=False)
        nn.init.zeros_(self.ground_readout.weight)

    @staticmethod
    def pool(value, mask):
        present = mask.unsqueeze(-1)
        mean = (value * present).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum = value.masked_fill(~present, -torch.inf).amax(1)
        maximum = torch.where(mask.any(1).unsqueeze(1), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1)

    def forward(self, points, mask):
        present = mask.unsqueeze(-1)
        value = torch.nn.functional.silu(self.stem(points * present) +
                    self.physics_stem(physical.physical_features(points, mask))) * present
        for block in self.blocks:
            value = block(value, present)
        ground = mask & (points[:, :, SETTINGS['ground_channel']] > SETTINGS['ground_threshold'])
        return self.pool(value, mask) + self.ground_readout(self.pool(value, ground))


class SparseFusion(physical.SparseFusion):
    def __init__(self, cards, bins, n_plain):
        super().__init__(cards, bins, n_plain)
        with torch.random.fork_rng(devices=[]):
            self.track = GroundTrackEncoder(self.track)


def check(device='cpu'):
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])] * 3, 5
    physical.parent.shared.recipe()
    torch.manual_seed(42)
    old = physical.SparseFusion(cards, bins, plain).to(device)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    new = SparseFusion(cards, bins, plain).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    tab, track = physical.parent.synthetic(cards, bins, plain, 24, device)
    with torch.no_grad():
        old.fusion[-1].weight.fill_(.01)
        old.fusion[-1].bias.fill_(.03)
        old.track.physics_stem.weight.fill_(.02)
        new.load_state_dict(old.state_dict(), strict=False)
    for training in [False, True]:
        old.train(training)
        new.train(training)
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
    old.eval()
    new.eval()
    with torch.no_grad():
        new.track.ground_readout.weight.fill_(.02)
    actual = new(*tab, *track)
    absent = track[1].sum(1) < 2
    torch.testing.assert_close(actual[absent], old(*tab, *track)[absent], rtol=0, atol=0)
    altered = track[0].clone()
    altered[~track[1]] = 999
    torch.testing.assert_close(new(*tab, altered, *track[1:]), actual, rtol=0, atol=0)
    no_ground = track[0].clone()
    no_ground[:, :, 9] = 0
    torch.testing.assert_close(new(*tab, no_ground, *track[1:]),
                               old(*tab, no_ground, *track[1:]), rtol=0, atol=0)
    actual.square().mean().backward()
    gradient = new.track.ground_readout.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    return dict(device=device, original_weights_and_rng_exact=True,
                nonzero_physical_initial_path_exact=True, dropout_rng_exact=True,
                absent_single_gate_exact=True, padding_invariant=True,
                no_ground_fallback_exact=True, finite_nonzero_gradient=True)
