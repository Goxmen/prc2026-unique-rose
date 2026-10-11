"""Public daily residual head trained jointly from the original initialization.

This is the full-training follow-up to the two-epoch continuation screen.
Original inputs/weights/RNG and the sparse optimizer update are preserved.
"""
import torch
import nn_public_delay as public

physical = public.physical
features = public.features
transform = public.transform
SETTINGS = dict(features=features.FEATURES, scales=public.SCALES,
    transform=public.SETTINGS['transform'], public_head=public.SETTINGS['public_head'],
    initialization='original physical seed42; zero new final head; no pretrained weights',
    training='original sparse train bytecode; fixed21/17 epochs; original AdamW and OneCycle')


class SparseFusion(public.PublicFusion):
    def __init__(self, cards, bins, n_plain):
        assert n_plain >= 12
        super().__init__(cards, bins, n_plain - 12)


def check(device='cpu'):
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])] * 3, 5
    physical.parent.shared.recipe()
    torch.manual_seed(42)
    old = physical.SparseFusion(cards, bins, plain).to(device)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    new = SparseFusion(cards, bins, plain + 12).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    tab, track = physical.parent.synthetic(cards, bins, plain, 24, device)
    extra = torch.linspace(-2, 2, 12, device=device).repeat(24, 1)
    extended = (*tab[:2], torch.cat([tab[2], extra], dim=1))
    with torch.no_grad():
        old.fusion[-1].weight.fill_(.01)
        old.track.physics_stem.weight.fill_(.02)
        new.initialize(old.state_dict())
    for training in [False, True]:
        old.train(training)
        new.train(training)
        torch.manual_seed(913)
        expected = old(*tab, *track)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(device).clone() if device != 'cpu' else None
        torch.manual_seed(913)
        actual = new(*extended, *track)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if gpu_rng is not None:
            assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
    new.eval()
    with torch.no_grad():
        new.public[-1].weight.fill_(.02)
    actual = new(*extended, *track)
    altered = track[0].clone()
    altered[~track[1]] = 999
    torch.testing.assert_close(new(*extended, altered, *track[1:]), actual, rtol=0, atol=0)
    actual.square().mean().backward()
    grad = new.public[-1].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    return dict(device=device, original_initialization_exact=True,
        original_dropout_rng_exact=True, nonzero_physical_path_exact=True,
        padding_invariant=True, finite_gradient=True)
