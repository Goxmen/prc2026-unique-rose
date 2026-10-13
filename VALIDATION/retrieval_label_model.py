"""Small fixed train-label retrieval mechanism, separate from retained models.

Only the dictionary accepts targets. The query API has no label argument.
Both arms have identical parameters; labels_off zeros dictionary values.
"""
import torch
from torch import nn

SETTINGS = dict(category_dim=8, hidden=128, key_dim=64, context=16,
    encoder_dropout=0., predictor_dropout=.1, label_encoder_zero_init=True,
    epochs=10, batch=256, seed=42, lr=.001, weight_decay=.0003,
    arms=['labels_off', 'labels_on'], train_per_airport=4096,
    bank_per_airport=512)


def admissible(query_meta, query_clock, bank_meta, bank_clock):
    """Metadata order is id,time_ms,airport,stand,runway,HEX,flight_id_code."""
    q, d = query_meta, bank_meta
    assert q.shape[1] == d.shape[1] == 7
    dt = q[:, 1, None] - d[None, :, 1]
    keep = (q[:, 2, None] == d[None, :, 2])
    keep &= torch.div(q[:, 1, None], 86400000, rounding_mode='floor') != torch.div(
        d[None, :, 1], 86400000, rounding_mode='floor')
    alias = q[:, 0, None] == d[None, :, 0]
    alias |= (q[:, 6, None] > 0) & (q[:, 6, None] == d[None, :, 6])
    alias |= ((q[:, 5, None] >= 0) & (q[:, 5, None] == d[None, :, 5]) & (dt.abs() <= 120000))
    signature = ((q[:, 3, None] > 0) & (q[:, 4, None] > 0)
        & torch.isfinite(query_clock[:, None]) & (dt == 0)
        & (q[:, 3, None] == d[None, :, 3]) & (q[:, 4, None] == d[None, :, 4])
        & (query_clock[:, None] == bank_clock[None, :]))
    return keep & ~(alias | signature)


class Retrieval(nn.Module):
    def __init__(self, cards, n_num):
        super().__init__()
        h, k = SETTINGS['hidden'], SETTINGS['key_dim']
        self.categories = nn.ModuleList([nn.Embedding(n, SETTINGS['category_dim']) for n in cards])
        self.encoder = nn.Sequential(nn.Linear(len(cards)*SETTINGS['category_dim']+n_num, h),
            nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.key = nn.Linear(h, k)
        self.label = nn.Linear(1, h)
        nn.init.zeros_(self.label.weight)
        nn.init.zeros_(self.label.bias)
        self.difference = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, h))
        self.predictor = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
            nn.Dropout(SETTINGS['predictor_dropout']), nn.Linear(h, 1))

    def encode(self, cat, num):
        parts = [module(cat[:, i]) for i, module in enumerate(self.categories)] + [num]
        x = self.encoder(torch.cat(parts, dim=1))
        return x, self.key(x)

    def forward(self, cat, num, bank_cat, bank_num, bank_y, valid, labels_on):
        assert valid.shape == (len(cat), len(bank_cat))
        assert torch.all(valid.sum(1) >= SETTINGS['context'])
        # Dictionary rows must be in increasing movement-ID order. Stable sort
        # then gives exact ID tie-breaking without perturbing any distances.
        with torch.no_grad():
            _, bank_key = self.encode(bank_cat, bank_num)
        x, key = self.encode(cat, num)
        with torch.no_grad():
            distance = (key.square().sum(1, keepdim=True) + bank_key.square().sum(1)[None, :]
                        - 2*key @ bank_key.T)
            distance.masked_fill_(~valid, torch.inf)
            selected = torch.argsort(distance, dim=1, stable=True)[:, :SETTINGS['context']]
        _, context_key = self.encode(bank_cat[selected].flatten(0, 1), bank_num[selected].flatten(0, 1))
        context_key = context_key.reshape(len(cat), SETTINGS['context'], -1)
        logits = -(key[:, None] - context_key).square().sum(-1)
        weights = torch.softmax(logits, dim=1)
        values = bank_y[selected] if labels_on else torch.zeros_like(bank_y[selected])
        context = self.label(values[..., None]) + self.difference(key[:, None] - context_key)
        return self.predictor(x + (weights[..., None]*context).sum(1)).squeeze(1)


def synthetic(device='cpu'):
    """No real files, fits or data-dependent choices; invoked only after review."""
    torch.manual_seed(42)
    bank_n, query_n = 64, 7
    cards, width = [5, 7], 8
    bcat = torch.stack([torch.arange(bank_n) % n for n in cards], 1).to(device)
    bnum = torch.randn(bank_n, width, device=device)
    by = torch.linspace(-2, 2, bank_n, device=device)
    meta = torch.zeros(bank_n, 7, dtype=torch.int64, device=device)
    meta[:, 0] = torch.arange(bank_n, device=device)+100
    meta[:, 1] = torch.arange(bank_n, device=device)*86400000
    meta[:, 2] = 1
    meta[:, 5] = -1
    clock = torch.full((bank_n,), torch.nan, device=device, dtype=torch.float64)
    qmeta = meta[:query_n].clone()
    valid = admissible(qmeta, clock[:query_n], meta, clock)
    assert not valid[torch.arange(query_n, device=device), torch.arange(query_n, device=device)].any()
    model = Retrieval(cards, width).to(device).eval()
    args = (bcat[:query_n], bnum[:query_n], bcat, bnum, by, valid)
    off = model(*args, False)
    on = model(*args, True)
    torch.testing.assert_close(on, off, atol=0, rtol=0)
    (on.square().mean()).backward()
    assert model.label.weight.grad is not None and model.label.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.label.weight.fill_(.125)
    on = model(*args, True)
    poisoned = (args[0], args[1], args[2], args[3], by+17, valid)
    assert not torch.equal(on, model(*poisoned, True))
    torch.testing.assert_close(model(*args, False), model(*poisoned, False), atol=0, rtol=0)
    # An excluded target value cannot influence its own query.
    own_poison = by.clone(); own_poison[0] = 1e8
    alt = model(args[0], args[1], bcat, bnum, own_poison, valid, True)
    torch.testing.assert_close(on[:1], alt[:1], atol=0, rtol=0)
    # The interface canonicalizes donor IDs before search, including exact ties.
    permutation = torch.arange(bank_n-1, -1, -1, device=device)
    restore = torch.argsort(meta[permutation, 0], stable=True)
    ix = permutation[restore]
    torch.testing.assert_close(model(args[0], args[1], bcat[ix], bnum[ix], by[ix],
        valid[:, ix], True), on, atol=0, rtol=0)
    tie_num, tie_cat = torch.zeros_like(bnum), torch.zeros_like(bcat)
    tied = model(args[0], args[1], tie_cat, tie_num, by, valid, True)
    torch.testing.assert_close(model(args[0], args[1], tie_cat[ix], tie_num[ix], by[ix],
        valid[:, ix], True), tied, atol=0, rtol=0)
    # Alias boundary fixtures include cross-UTC-midnight aircraft aliases.
    q = torch.tensor([[999, 86400000+60000, 1, 3, 4, 11, 77]], device=device)
    d = torch.tensor([[998, 86400000-60000, 1, 8, 9, 11, 0],
                      [997, 86400000-60001, 1, 8, 9, 11, 0],
                      [996, 86400000+60000, 1, 3, 4, 12, 0],
                      [995, 2*86400000+60000, 2, 3, 4, 11, 0],
                      [994, 3*86400000+60000, 1, 8, 9, 99, 77]], device=device)
    qc = torch.tensor([17.], device=device, dtype=torch.float64)
    dc = torch.tensor([19., 19., 17., 17., 20.], device=device, dtype=torch.float64)
    assert admissible(q, qc, d, dc).tolist() == [[False, True, False, False, False]]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return dict(device=device, init_exact=True, label_gradient_nonzero=True,
        active_dictionary_labels_matter=True, labels_off_poison_invariant=True,
        excluded_own_target_invariant=True, canonical_bank_permutation_exact=True,
        alias_midnight_boundary_exact=True, query_label_argument_absent=True)
