"""Train-time withholding of the raw-track residual, with unchanged inference.

Keep the public physical model, its initialization and optimizer RNG streams.
A separate generator removes the complete sequence for 25% of training rows.
Tabular/public inputs remain available. This simulates losing the raw-track
branch, not losing all ADS-B-derived inputs or specifically losing ground data.
"""
import torch
import nn_public_scratch as parent

physical, features, transform = parent.physical, parent.features, parent.transform
SETTINGS = dict(parent=parent.SETTINGS, track_drop_probability=.25,
    generator='separate per-device generator; initialization seed plus104729',
    training='mask entire raw track; original missing-track gate; no inverse-probability rescaling',
    inference='all observed tracks; original forward computation',
    unchanged='tabular/public inputs, weights, optimizer, batch order, dropout RNG and epochs',
    limitation='Raw-track branch availability; not a complete ADS-B or ground-only missingness simulation')


class SparseFusion(parent.SparseFusion):
    def __init__(self, cards, bins, n_plain):
        super().__init__(cards, bins, n_plain)
        self.augmentation_seed = int(torch.initial_seed())+104729
        self.augmentation_generators = {}
        self.drop_probability = SETTINGS['track_drop_probability']

    def forward(self, cat, g, plain, points, mask, quality):
        if self.training:
            key = str(points.device)
            if key not in self.augmentation_generators:
                self.augmentation_generators[key] = torch.Generator(device=points.device).manual_seed(self.augmentation_seed)
            keep = torch.rand(len(mask), device=points.device,
                generator=self.augmentation_generators[key]) >= self.drop_probability
            mask = mask & keep.unsqueeze(1)
        return super().forward(cat, g, plain, points, mask, quality)


def check(device='cpu'):
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])]*3, 5
    physical.parent.shared.recipe()
    torch.manual_seed(42)
    old = parent.SparseFusion(cards, bins, plain+12).to(device)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    model = SparseFusion(cards, bins, plain+12).to(device)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert old.state_dict().keys() == model.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    tab, track = physical.parent.synthetic(cards, bins, plain, 48, device)
    public = torch.linspace(-2, 2, 12, device=device).repeat(len(track[1]), 1)
    extended = (*tab[:2], torch.cat([tab[2], public], dim=1))
    # Nonzero trained paths make the mask test meaningful at initialization.
    with torch.no_grad():
        old.fusion[-1].weight.fill_(.01)
        old.fusion[-1].bias.fill_(.03)
        old.track.physics_stem.weight.fill_(.02)
        old.public[-1].weight.fill_(.01)
        model.load_state_dict(old.state_dict())
    old.eval()
    model.eval()
    torch.testing.assert_close(model(*extended, *track), old(*extended, *track), rtol=0, atol=0)
    assert not model.augmentation_generators, 'Inference must not draw augmentation randomness'
    old.train()
    model.train()
    generator = torch.Generator(device=device).manual_seed(42+104729)
    keep = torch.rand(len(track[1]), device=device, generator=generator) >= .25
    usable = track[1].sum(1) >= 2
    assert (keep&usable).any() and (~keep&usable).any()
    torch.manual_seed(913)
    expected = old(*extended, track[0], track[1]&keep[:, None], track[2])
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone() if device != 'cpu' else None
    torch.manual_seed(913)
    actual = model(*extended, *track)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
    actual.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.track.physics_stem.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.drop_probability = 1.
    torch.manual_seed(914)
    actual = model(*extended, *track)
    torch.manual_seed(914)
    expected = old(*extended, track[0], torch.zeros_like(track[1]), track[2])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert all(p.grad is None or p.grad.eq(0).all() for p in model.track.parameters())
    assert model.public[-1].weight.grad.abs().sum() > 0
    model.eval()
    old.eval()
    model.drop_probability = .25
    expected = old(*extended, *track)
    torch.testing.assert_close(model(*extended, *track), expected, rtol=0, atol=0)
    altered = track[0].clone()
    altered[~track[1]] = 999
    torch.testing.assert_close(model(*extended, altered, *track[1:]), expected, rtol=0, atol=0)
    return dict(device=device, identical_parameters_and_initialization_rng=True,
        inference_exact=True, independent_augmentation_generator=True, dropout_rng_exact=True,
        masked_training_replay_exact=True, dropped_usable_rows=int((~keep&usable).sum()),
        kept_usable_rows=int((keep&usable).sum()), finite_gradient=True,
        all_dropped_track_gradient_zero=True, public_head_active_without_track=True, padding_invariant=True)
