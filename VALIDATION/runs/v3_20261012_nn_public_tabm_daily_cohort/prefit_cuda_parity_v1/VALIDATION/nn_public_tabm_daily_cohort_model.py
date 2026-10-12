"""One zero-initialized full UTC-day input-only context branch on selected TabM."""
import torch
from torch import nn

import nn_public_tabm as original
import daily_cohort_bank as bank

SETTINGS = dict(original=original.SETTINGS, bank=bank.SETTINGS,
    encoder='17 ->32 SiLU ->32 SiLU, no dropout',
    pooling='leave-query-and-alias-out mean32, second-moment32, log1p(count)/log853',
    projection='65 to unchanged joined-input width; weight and bias initially zero',
    position='addition immediately before original TabM ensemble view',
    no_new_random_stream='constructor uses fork_rng; no context dropout',
    empty='exactly zero context branch including learned projection bias',
    training='original individual32-head MSE, row permutation, microbatch and optimizer')


class DailyTabM(original.PublicTabM):
    def __init__(self, cards, bins, n_plain):
        super().__init__(cards, bins, n_plain)
        width = sum(self.inputs.chunks) + 12 + 64 + len(original.prototype.seq.QUALITY_NAMES)
        with torch.random.fork_rng(devices=[]):
            self.donor = nn.Sequential(nn.Linear(bank.DIM, 32), nn.SiLU(),
                nn.Linear(32, 32), nn.SiLU())
            self.context_projection = nn.Linear(65, width)
            nn.init.zeros_(self.context_projection.weight)
            nn.init.zeros_(self.context_projection.bias)
        self.context_disabled = False

    def context(self, packet):
        values, mask, groups, excluded = packet
        latent = self.donor(values) * mask.unsqueeze(-1)
        sums = latent.sum(1)
        squares = latent.square().sum(1)
        counts = mask.sum(1)
        qlatent = latent[groups[:, None], excluded.clamp(min=0)]
        valid = excluded >= 0
        assert torch.all(mask[groups[:, None], excluded.clamp(min=0)] | ~valid)
        removed = qlatent * valid.unsqueeze(-1)
        n = counts[groups] - valid.sum(1)
        assert torch.all(n >= 0)
        mean = (sums[groups] - removed.sum(1)) / n.clamp(min=1).unsqueeze(1)
        moment = (squares[groups] - (qlatent.square()*valid.unsqueeze(-1)).sum(1)) / n.clamp(min=1).unsqueeze(1)
        summary = torch.cat([mean, moment, torch.log1p(n).unsqueeze(1)/6.748759547491679], dim=1)
        summary = torch.where((n > 0).unsqueeze(1), summary, 0.)
        return self.context_projection(summary) * (n > 0).unsqueeze(1)

    def forward(self, cat, g, plain, points, mask, quality, packet=None):
        tab = self.inputs(cat, g, plain[:, :self.original_plain])
        representation = self.track(points, mask) * (mask.sum(1) >= 2).unsqueeze(1)
        joined = torch.cat([tab, plain[:, self.original_plain:], representation, quality], dim=1)
        if packet is not None and not self.context_disabled:
            joined = joined + self.context(packet)
        return self.output(self.backbone(self.view(joined))).squeeze(-1)


def synthetic_packet(n, device):
    values = torch.linspace(-1., 1., 3*7*bank.DIM, device=device).reshape(3, 7, bank.DIM)
    mask = torch.ones((3, 7), dtype=torch.bool, device=device)
    mask[1, 5:] = False
    groups = torch.arange(n, device=device) % 3
    excluded = torch.full((n, 2), -1, dtype=torch.long, device=device)
    excluded[:, 0] = 0
    excluded[::3, 1] = 1
    return values, mask, groups, excluded


def check(device='cpu'):
    original.prototype.shared.recipe()
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])]*3, 5
    torch.manual_seed(42)
    old = original.PublicTabM(cards, bins, plain+12).to(device)
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone() if device != 'cpu' else None
    torch.manual_seed(42)
    new = DailyTabM(cards, bins, plain+12).to(device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if gpu_rng is not None: assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    tab, track = original.prototype.synthetic(cards, bins, plain+12, 12, device)
    packet = synthetic_packet(12, device)
    target = torch.linspace(-1, 1, 12, device=device)
    for training in [False, True]:
        old.train(training); new.train(training)
        old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
        torch.manual_seed(913)
        before = old(*tab, *track)
        old_loss = original.loss(before, target); old_loss.backward()
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device != 'cpu' else None
        torch.manual_seed(913)
        after = new(*tab, *track, packet)
        new_loss = original.loss(after, target); new_loss.backward()
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        assert torch.equal(rng, torch.get_rng_state())
        if cuda_rng is not None: assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        old_parameters = dict(old.named_parameters()); new_parameters = dict(new.named_parameters())
        for name in old_parameters:
            torch.testing.assert_close(new_parameters[name].grad, old_parameters[name].grad, rtol=0, atol=0)
        assert new.context_projection.weight.grad.abs().sum() > 0
        for p in new.donor.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0
    # One identical AdamW step preserves every original parameter. The new
    # projection then enables finite nonzero donor-encoder gradients.
    a = torch.optim.AdamW(old.parameters(), lr=.002, weight_decay=.0003)
    b = torch.optim.AdamW(new.parameters(), lr=.002, weight_decay=.0003)
    a.step(); b.step()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    new.eval(); new.zero_grad(set_to_none=True)
    original.loss(new(*tab, *track, packet), target).backward()
    for p in new.donor.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    expected = new(*tab, *track, packet)
    # Permute whole donor tuples and remap exclusion positions exactly.
    permutation = torch.tensor([6, 2, 4, 0, 5, 1, 3], device=device)
    inverse = torch.argsort(permutation)
    values, mask, groups, excluded = packet
    permuted = (values[:, permutation], mask[:, permutation], groups,
        torch.where(excluded >= 0, inverse[excluded.clamp(min=0)], -1))
    torch.testing.assert_close(new(*tab, *track, permuted), expected, rtol=2e-6, atol=2e-6)
    # Excluded and padded tuples cannot affect outputs, even with an active branch.
    altered = values.clone(); altered[:, 0] = 9876.; altered[0, 1] = -1234.
    altered[~mask] = 9999.
    # Numerically large excluded latents can cause subtraction cancellation;
    # bounded physical inputs are the real contract. Use bounded replacements.
    altered = altered.clamp(-5, 5)
    torch.testing.assert_close(new(*tab, *track, (altered, mask, groups, excluded)), expected, rtol=2e-6, atol=2e-6)
    empty = (values, torch.zeros_like(mask), groups, torch.full_like(excluded, -1))
    with torch.no_grad(): new.context_projection.bias.fill_(.3)
    new.context_disabled = True
    original_only = new(*tab, *track, empty)
    new.context_disabled = False
    torch.testing.assert_close(new(*tab, *track, empty), original_only, rtol=0, atol=0)
    return dict(device=device, initialization_exact=True, original_forward_exact=True,
        original_gradient_exact=True, original_dropout_and_constructor_RNG_exact=True,
        first_original_AdamW_update_exact=True, zero_initial_donor_gradients_expected=True,
        active_donor_gradients_finite_nonzero=True, complete_tuple_permutation=True,
        query_alias_and_padding_invariant=True, empty_branch_exact_zero=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    print(check('cpu'))
