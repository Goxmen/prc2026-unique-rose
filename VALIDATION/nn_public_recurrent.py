"""Fixed recurrent summary of the retained physical track representation.

One forward GRU reads the chronological local representations and exposes its
last observed state through a zero-initialized projection into the original
mean/max pool. Original initialization, inputs, missing-track gate and update
remain intact. Prefix masks are guaranteed by the retained ragged cache.
"""
import torch
from torch import nn

import nn_public_scratch as parent

physical, features, transform = parent.physical, parent.features, parent.transform
SETTINGS = dict(parent=parent.SETTINGS,
    change='add one forward width32 GRU over physical convolution states',
    summary='state at last observed point, zero for absent track; prefix masks',
    projection='Linear32to64 biasFalse zero; original masked mean/max unchanged',
    added_parameters=8384, recurrent_layers=1, recurrent_dropout=0,
    initialization='forked RNG, original parameters and outputs unchanged',
    training='unchanged168 inputs,seed42,21/17 epochs,original MSE and optimizer',
    hypothesis='long sequence order beyond fixed local convolutions and symmetric pooling',
    documentation='https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html')


def last_observed(recurrent, value, mask):
    states, _ = recurrent(value * mask.unsqueeze(-1))
    count = mask.sum(1)
    selected = states[torch.arange(len(states), device=states.device), (count-1).clamp(min=0)]
    return selected * (count > 0).unsqueeze(-1)


class RecurrentTrackEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.stem, self.blocks, self.physics_stem = original.stem, original.blocks, original.physics_stem
        width = physical.parent.SETTINGS['width']
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.recurrent_projection = nn.Linear(width, 2*width, bias=False)
        nn.init.zeros_(self.recurrent_projection.weight)

    def forward(self, points, mask):
        present = mask.unsqueeze(-1)
        value = nn.functional.silu(self.stem(points*present)
            + self.physics_stem(physical.physical_features(points, mask))) * present
        for block in self.blocks: value = block(value, present)
        mean = value.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum = value.masked_fill(~present, -torch.inf).amax(1)
        maximum = torch.where(mask.any(1).unsqueeze(1), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1) + self.recurrent_projection(
            last_observed(self.recurrent, value, mask))


class SparseFusion(parent.SparseFusion):
    def __init__(self, cards, bins, n_plain):
        super().__init__(cards, bins, n_plain)
        with torch.random.fork_rng(devices=[]):
            self.track = RecurrentTrackEncoder(self.track)


def recurrence_check(device):
    """Independent gate equations and per-row stopping, including padded rows."""
    torch.manual_seed(613)
    recurrent = nn.GRU(3, 4, batch_first=True).to(device=device, dtype=torch.float64)
    values = torch.arange(6*9*3, device=device, dtype=torch.float64).reshape(6, 9, 3)/97-1
    counts = torch.tensor([0, 1, 2, 4, 8, 9], device=device)
    mask = torch.arange(9, device=device)[None, :] < counts[:, None]
    actual = last_observed(recurrent, values, mask)
    expected = []
    for row, count in enumerate(counts.tolist()):
        h = torch.zeros(4, device=device, dtype=torch.float64)
        for step in range(count):
            ir, iz, inc = (recurrent.weight_ih_l0 @ values[row, step] + recurrent.bias_ih_l0).chunk(3)
            hr, hz, hnc = (recurrent.weight_hh_l0 @ h + recurrent.bias_hh_l0).chunk(3)
            reset, update = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
            proposal = torch.tanh(inc + reset*hnc)
            h = (1-update)*proposal + update*h
        expected.append(h)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=2e-15)
    altered = values.clone(); altered[~mask] = 999
    torch.testing.assert_close(last_observed(recurrent, altered, mask), actual, rtol=0, atol=0)
    reverse = values.clone()
    for row, count in enumerate(counts.tolist()): reverse[row, :count] = values[row, :count].flip(0)
    assert (last_observed(recurrent, reverse, mask)[2:] - actual[2:]).abs().max() > .01
    return dict(manual_gate_equations=True, max_error=float((actual-torch.stack(expected)).abs().max().detach()),
        empty_single_short_full_lengths=True, padded_values_ignored=True, sequence_order_changes_summary=True)


def check(device='cpu'):
    physical.parent.shared.recipe()
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])]*3, 5
    torch.manual_seed(42); old = parent.SparseFusion(cards, bins, plain+12).to(device)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42); new = SparseFusion(cards, bins, plain+12).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items(): assert torch.equal(value, new.state_dict()[name]), name
    added = sum(v.numel() for v in new.parameters()) - sum(v.numel() for v in old.parameters())
    assert added == SETTINGS['added_parameters']
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
        if training:
            old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
            expected.square().mean().backward(); actual.square().mean().backward()
            parameters = dict(new.named_parameters())
            for name, value in old.named_parameters():
                other = parameters[name]
                assert (value.grad is None) == (other.grad is None), name
                if value.grad is not None:
                    tolerance = 0.
                    if device != 'cpu' and name.startswith('track.'):
                        tolerance = 8*torch.finfo(value.grad.dtype).eps*float(torch.maximum(value.grad.abs(),other.grad.abs()).max())
                    torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=tolerance)
                    maximum_gradient_error = max(maximum_gradient_error, float((value.grad-other.grad).abs().max()))
            gradient = new.track.recurrent_projection.weight.grad
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    old.eval(); new.eval()
    with torch.no_grad():
        new.track.recurrent_projection.weight.fill_(.03)
        actual, reference = new(*tab, *track), old(*tab, *track)
        assert (actual-reference).abs().max() > 0
        absent = track[1].sum(1) < 2
        torch.testing.assert_close(actual[absent], reference[absent], rtol=0, atol=0)
        altered = track[0].clone(); altered[~track[1]] = 999
        torch.testing.assert_close(new(*tab, altered, *track[1:]), actual, rtol=0, atol=0)
    new.train(); new.zero_grad(set_to_none=True)
    new(*tab, *track).square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
               for v in new.track.recurrent.parameters())
    return dict(device=device, added_parameters=added, original_weights_outputs_and_rng_exact=True,
        original_gradients_checked=True, gradient_max_absolute_error=maximum_gradient_error,
        cuda_track_gradient_tolerance='8 * float epsilon * tensor maximum; other gradients exact',
        missing_track_fallback_and_padding_exact=True, recurrent_weights_receive_finite_gradients=True,
        recurrence=recurrence_check(device))
