"""Fixed PLE neural classifier for Rome's recorded schedule-copy mechanism."""
import copy
import math
import os
import time
import types
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')

import numpy as np
import pandas as pd
import rtdl_num_embeddings
import torch
from torch import nn

import nn_jan_jul as original

ENCODING = dict(min_cat_rows=20, quantiles=256, clip=5.)
SETTINGS = dict(encoding=ENCODING, bins=32, d_ple=8, hidden=[256, 128], dropout=.1,
    batch=1024, max_epochs=48, lr=.001, weight_decay=.0001, gradient_clip=1.,
    seed=42, loss='unweighted binary cross-entropy with logits',
    stopping='lowest last-allowed-training-month log loss; refit full outer training',
    schedule='OneCycleLR over48 epochs in inner and full fits; stop full at chosen epoch')


def isolated(function):
    return types.FunctionType(function.__code__, {**function.__globals__, 'RECIPE': ENCODING},
        function.__name__, function.__defaults__, function.__closure__)


class Encoder(original.Encoder):
    # Reuse the retained numeric/category transformations with frozen settings,
    # independent of ordinary-model modules that change their own recipe.
    def __init__(self, X):
        isolated(original.Encoder.__init__)(self, X)

    def transform(self, X):
        return isolated(original.Encoder.transform)(self, X)


def preprocessing(X):
    enc = Encoder(X)
    cat, num = enc.transform(X)
    rank, plain, at = [], [], 0
    for name in enc.nums:
        rank.append(at)
        plain.append(at+1)
        at += 2
        if name in enc.flags:
            plain.append(at)
            at += 1
    assert at == num.shape[1]
    edges = np.quantile(num[:, rank], np.linspace(0, 1, SETTINGS['bins']+1), axis=0)
    ple, bins = [], []
    for k, col in enumerate(rank):
        edge = np.unique(edges[:, k].astype('float32'))
        if len(edge) >= 3:
            ple.append(col)
            bins.append(torch.from_numpy(edge))
        else:
            plain.append(col)
    state = dict(encoder=enc, cards=enc.cards, bins=bins, ple=ple, plain=plain,
        columns=list(X), settings=copy.deepcopy(SETTINGS))
    return state


def encode(state, X, device):
    assert list(X) == state['columns'] and state['settings'] == SETTINGS
    cat, num = state['encoder'].transform(X)
    assert np.isfinite(num).all()
    return tuple(torch.from_numpy(np.ascontiguousarray(a)).to(device)
        for a in [cat, num[:, state['ple']], num[:, state['plain']]])


class Classifier(nn.Module):
    def __init__(self, state):
        super().__init__()
        assert state['settings'] == SETTINGS
        dims = [min(24, max(2, round(1.6*k**.56))) for k in state['cards']]
        self.emb = nn.ModuleList(nn.Embedding(k, d) for k, d in zip(state['cards'], dims))
        self.ple = (rtdl_num_embeddings.PiecewiseLinearEmbeddings(state['bins'], SETTINGS['d_ple'],
            activation=False, version='B') if state['bins'] else None)
        width = sum(dims)+len(state['bins'])*SETTINGS['d_ple']+len(state['plain'])
        layers = []
        for hidden in SETTINGS['hidden']:
            layers += [nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(SETTINGS['dropout'])]
            width = hidden
        self.mlp = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, cat, g, plain):
        pieces = [emb(cat[:, k]) for k, emb in enumerate(self.emb)]
        if self.ple is not None:
            pieces.append(self.ple(g).flatten(1))
        return self.mlp(torch.cat(pieces+[plain], dim=1)).squeeze(1)


def probabilities(model, data):
    training = model.training
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(*(a[i:i+4096] for a in data)).sigmoid()
            for i in range(0, len(data[0]), 4096)]).cpu().numpy().astype('float64')
    model.train(training)
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    return pred


def fit(state, X, y, epochs, monitor=None, device='cuda'):
    assert 1 <= epochs <= SETTINGS['max_epochs']
    y = np.asarray(y, 'float32')
    assert set(np.unique(y)) == {0., 1.}
    data = encode(state, X, device)
    target = torch.from_numpy(y).to(device)
    torch.manual_seed(SETTINGS['seed'])
    model = Classifier(state).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
    batch = SETTINGS['batch']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SETTINGS['lr'],
        total_steps=SETTINGS['max_epochs']*math.ceil(len(y)/batch))
    generator = torch.Generator(device=device).manual_seed(SETTINGS['seed'])
    monitoring = None if monitor is None else encode(state, monitor[0], device)
    curve = []
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        perm = torch.randperm(len(y), device=device, generator=generator)
        total = 0.
        for i in range(0, len(y), batch):
            index = perm[i:i+batch]
            logits = model(*(a[index] for a in data))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target[index])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip'])
            opt.step()
            scheduler.step()
            total += float(loss.detach())*len(index)
        point = dict(epoch=epoch+1, training_logloss=total/len(y), seconds=time.monotonic()-started)
        if monitoring is not None:
            p = np.clip(probabilities(model, monitoring), 1e-12, 1-1e-12)
            label = np.asarray(monitor[1], float)
            point['inner_logloss'] = float(-np.mean(label*np.log(p)+(1-label)*np.log1p(-p)))
        curve.append(point)
        print(point, flush=True)
    return model, curve


def checks(device='cpu'):
    ordinary_recipe = copy.deepcopy(original.RECIPE)
    torch.manual_seed(91)
    f = pd.DataFrame(dict(category=pd.Categorical(['A']*25+['B']*25+[None]*4),
        varied=np.linspace(-3, 3, 54), constant=np.ones(54),
        missing=np.r_[np.arange(50), [np.nan]*4]))
    state = preprocessing(f)
    before = copy.deepcopy(state['encoder'].__dict__)
    data = encode(state, f, device)
    other = f.iloc[:3].copy()
    other['category'] = pd.Categorical(['new', None, 'A'])
    other['varied'] = [1e9, np.nan, -1e9]
    extra = encode(state, other, device)
    assert extra[0][:, 0].cpu().tolist() == [0, 0, 1]
    assert before['maps'] == state['encoder'].maps and before['scale'] == state['encoder'].scale
    torch.manual_seed(42)
    model = Classifier(state).to(device)
    target = torch.arange(len(f), device=device).remainder(2).float()
    loss = nn.functional.binary_cross_entropy_with_logits(model(*data), target)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    expected = probabilities(model, data)
    other_model = Classifier(state).to(device)
    other_model.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(expected, probabilities(other_model, data))
    order = np.arange(len(f))[::-1].copy()
    np.testing.assert_allclose(probabilities(model, tuple(a[order] for a in data)), expected[order], atol=1e-7, rtol=0)
    assert original.RECIPE == ordinary_recipe
    return dict(device=device, unknown_and_missing_category_zero=True, finite_extreme_inputs=True,
        transform_does_not_refit=True, finite_bce_gradients=True, state_reload_exact=True,
        inference_row_permutation=True, no_ordinary_recipe_mutation=True)
