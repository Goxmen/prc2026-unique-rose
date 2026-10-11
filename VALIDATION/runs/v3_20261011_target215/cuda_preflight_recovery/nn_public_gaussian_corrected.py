"""Fixed Gaussian likelihood screen on the existing public physical network.

The mean architecture, inputs and inference stay unchanged. A zero-initialized
linear head predicts log variance from the same hidden/track/public evidence.
Twice Gaussian NLL equals the original MSE when log variance is zero.
"""
import ast
import copy
import inspect

import torch
from torch import nn

import nn_public_scratch as parent

physical, features, transform = parent.physical, parent.features, parent.transform
SETTINGS = dict(parent=parent.SETTINGS,
    change='learn observation variance with twice Gaussian negative log likelihood',
    mean='unchanged public physical architecture; predict conditional mean only',
    variance='Linear(hidden256 + track64 + quality + public12,1), initialized zero',
    log_variance_bounds=[-6., 4.],
    target='unchanged standardized min(y,10800); no new labels or target filtering',
    training='same batches, AdamW, OneCycle24, seed42 and fixed21/17 epochs; loss alone changes',
    selection='matched mean RMSE and one-of-three replacement, not likelihood')


def gaussian_loss(mean, target, log_variance):
    assert mean.shape == target.shape == log_variance.shape
    return (torch.exp(-log_variance) * (mean - target).square() + log_variance).mean()


class SparseFusion(parent.SparseFusion):
    def __init__(self, cards, bins, n_plain):
        super().__init__(cards, bins, n_plain)
        width = 256 + 2 * physical.parent.SETTINGS['width'] + len(physical.parent.seq.QUALITY_NAMES) + 12
        with torch.random.fork_rng(devices=[]):
            self.variance = nn.Linear(width, 1)
            nn.init.zeros_(self.variance.weight)
            nn.init.zeros_(self.variance.bias)

    def forward(self, cat, g, plain, points, mask, quality):
        public = plain[:, self.original_plain:]
        hidden = self.tabular.inputs(cat, g, plain[:, :self.original_plain])
        for layer in list(self.tabular.mlp.children())[:-1]:
            hidden = layer(hidden)
        base = self.tabular.mlp[-1](hidden)
        representation = self.track(points, mask)
        fusion_input = torch.cat([hidden, representation, quality], dim=1)
        delta = self.fusion(fusion_input)
        mean = base + delta * (mask.sum(1) >= 2).unsqueeze(1) + self.public(torch.cat([hidden, public], dim=1))
        self.log_variance = self.variance(torch.cat([fusion_input, public], dim=1)).clamp(-6., 4.)
        return mean


def training_function(original, namespace):
    """Change exactly the loss statement and its log name in the frozen trainer."""
    tree = ast.parse(inspect.getsource(original))
    modified = copy.deepcopy(tree)
    matches = [node for node in ast.walk(modified) if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == 'loss']
    assert len(matches) == 1
    expected = ast.parse('torch.nn.functional.mse_loss(output,d.ty[rows,None].expand_as(output))', mode='eval').body
    assert ast.dump(matches[0].value) == ast.dump(expected)
    replacement = ast.parse('prototype.gaussian_loss(output,d.ty[rows,None].expand_as(output),model.log_variance)', mode='eval').body
    matches[0].value = replacement
    names = [node for node in ast.walk(modified) if isinstance(node, ast.keyword) and node.arg == 'train_mse_std']
    assert len(names) == 1
    names[0].arg = 'train_gaussian_nll_std'
    # Reverse exactly the two permitted changes to check every other AST node.
    restored = copy.deepcopy(modified)
    for node in ast.walk(restored):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'loss':
            node.value = expected
        if isinstance(node, ast.keyword) and node.arg == 'train_gaussian_nll_std':
            node.arg = 'train_mse_std'
    assert ast.dump(restored) == ast.dump(tree)
    scope = dict(original.__globals__, **namespace)
    exec(compile(ast.fix_missing_locations(modified), __file__ + ':loss_only_train', 'exec'), scope)
    return scope[original.__name__]


def check(device='cpu'):
    physical.parent.shared.recipe()
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])] * 3, 5
    torch.manual_seed(42); old = parent.SparseFusion(cards, bins, plain + 12).to(device)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42); new = SparseFusion(cards, bins, plain + 12).to(device)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    tab, track = physical.parent.synthetic(cards, bins, plain, 24, device)
    tab = (*tab[:2], torch.cat([tab[2], torch.linspace(-2, 2, 12, device=device).repeat(24, 1)], dim=1))
    target = torch.linspace(-3, 4, 24, device=device)[:, None]
    with torch.no_grad():
        for model in [old, new]:
            model.public[-1].weight.fill_(.02)
            model.fusion[-1].weight.fill_(.01)
            model.track.physics_stem.weight.fill_(.02)
    gradient_max_error = 0.
    for training in [False, True]:
        old.train(training); new.train(training)
        torch.manual_seed(912); expected = old(*tab, *track)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(device).clone() if device != 'cpu' else None
        torch.manual_seed(912); actual = new(*tab, *track)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if gpu_rng is not None: assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
        assert torch.count_nonzero(new.log_variance) == 0
        a = nn.functional.mse_loss(expected, target)
        b = gaussian_loss(actual, target, new.log_variance)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
        a.backward(); b.backward()
        parameters = dict(new.named_parameters())
        for name, value in old.named_parameters():
            other = parameters[name]
            assert (value.grad is None) == (other.grad is None), name
            if value.grad is not None:
                # Identical-network CUDA convolution repeats also differ by
                # floating accumulation precision; all other checks stay exact.
                tolerance = 0.
                if device != 'cpu' and name.startswith('track.blocks.') and '.convs.' in name and name.endswith('.weight'):
                    scale = float(torch.maximum(value.grad.abs(), other.grad.abs()).max())
                    tolerance = 8 * torch.finfo(value.grad.dtype).eps * scale
                torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=tolerance)
                gradient_max_error = max(gradient_max_error, float((value.grad-other.grad).abs().max()))
        assert torch.isfinite(new.variance.weight.grad).all() and new.variance.weight.grad.abs().sum() > 0
    new.eval()
    with torch.no_grad():
        new.variance.weight.fill_(.03)
        actual = new(*tab, *track); variance = new.log_variance.clone()
        altered = track[0].clone(); altered[~track[1]] = 999
        torch.testing.assert_close(new(*tab, altered, *track[1:]), actual, rtol=0, atol=0)
        torch.testing.assert_close(new.log_variance, variance, rtol=0, atol=0)
        torch.testing.assert_close(actual, old.eval()(*tab, *track), rtol=0, atol=0)
    # Independent analytic gradient checks at nonzero variance.
    mean = torch.tensor([[.4], [-1.2], [3.]], dtype=torch.float64, device=device, requires_grad=True)
    logv = torch.tensor([[-2.], [.7], [2.3]], dtype=torch.float64, device=device, requires_grad=True)
    truth = torch.tensor([[1.], [2.], [-1.]], dtype=torch.float64, device=device)
    gaussian_loss(mean, truth, logv).backward()
    torch.testing.assert_close(mean.grad, 2 * (mean - truth) * torch.exp(-logv) / 3, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(logv.grad, (1 - (mean - truth).square() * torch.exp(-logv)) / 3, rtol=1e-14, atol=1e-14)
    return dict(device=device, original_parameters_mean_and_initial_loss_exact=True,
        original_gradients_checked=True, gradient_max_absolute_error=gradient_max_error,
        cuda_convolution_gradient_tolerance='8 * float epsilon * tensor maximum; all other gradients exact',
        initialization_and_dropout_rng_exact=True, variance_gradient_finite_nonzero=True,
        variance_does_not_change_mean_inference=True, padding_invariant_for_both_outputs=True,
        independent_nonzero_variance_gradients_pass=True)
