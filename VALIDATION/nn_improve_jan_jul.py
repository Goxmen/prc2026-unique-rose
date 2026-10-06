"""Jan/Jul CV of three changes to v47's neural network, each blended with v46's LightGBM.

Research question (2026-10-03): which of these improves v47's 0.3 network / 0.7 LightGBM
blend on Jan/Jul CV?
  avg5  v47's network unchanged (`nn_jan_jul.fit_nn`), averaged over five seeds.
  ple   v47's network with each rank-gauss input replaced by a piecewise-linear embedding
        (Gorishniy et al. 2022), and the number of epochs chosen on the last training month.
  tabm  TabM (Gorishniy et al. 2024; 32 networks sharing most weights, trained as one) on the
        `ple` inputs, epochs chosen the same way.
avg5 and ple average five networks per evaluation seed; tabm, already an ensemble and about
eight times slower per epoch, uses one. ple and tabm are compared both with v47 and with avg5. Rows, inputs, LightGBM predictions, blend weight, Rome classifier,
corrections and rules are those of `nn_jan_jul.py`; scoring is in
`VALIDATION/tools/score_nn_improve_jan_jul.py`.

Frozen recipe (fixed before any run, not tuned on the held-out months):
  networks   evaluation seeds 42/142/242 (the LightGBM seeds); network k of seed s uses seed
             s + 1000k, k = 0..4 (avg5, ple) or k = 0 (tabm). avg5's network k = 0 is v47's
             saved network.
  inputs     v47's encoder (`nn_jan_jul.Encoder`) unchanged. ple/tabm: every rank-gauss input
             with at least two quantile bins goes through a piecewise-linear embedding (48
             training-quantile bins, 16 dimensions, version B of `rtdl_num_embeddings`); the
             clipped linear copies, missing flags and remaining rank-gauss inputs enter as they
             are; categoricals keep v47's embeddings.
  ple        v47's 512-512-256 SiLU network, dropout 0.1, AdamW lr 2e-3, weight decay 1e-5.
  tabm       `tabm` package backbone: arch 'tabm', k = 32, 2 blocks of 512, ReLU, dropout 0.1,
             'normal' first scaling initialised per input chunk; AdamW lr 2e-3, weight decay
             3e-4 (the package's defaults); TF32 matrix products. Loss: mean of the 32 members'
             MSE; prediction: their mean.
  epochs     ple/tabm: one-cycle schedule over 24 epochs, batch 4,096. First train on the
             training months except the last (November for January, December for July) with
             network seed s, and take the epoch with the lowest RMSE on that last month. Then fit
             every network of seed s on all training months with the same 24-epoch schedule and
             stop after that epoch. avg5 keeps v47's 8-epoch schedule and final epoch.
  blend      base = 0.3 * mean of the arm's networks + 0.7 * LightGBM, then the shared final
             prediction (weight fixed for v47 in advance; grid in the scorer is a diagnostic).
Held-out RMSE per epoch is logged as a diagnostic only; nothing reads it.

Run: .venv/bin/python VALIDATION/nn_improve_jan_jul.py [--arms avg5 ple tabm]
Each network's held-out prediction is saved as it finishes; finished networks are skipped.
No full-year fits, submissions or ledger edits.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# torch 2.14 routes batched-matmul gradients to Triton kernels that need Python headers this
# machine lacks; the built-in CUDA kernels compute the same thing.
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import rtdl_num_embeddings
import tabm
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import nn_jan_jul
import overnight_run
import rome_hybrid
import seasonal
import tuning_seasonal as ts
from records import RUNS, sha256, write_json

SPLITS_FROM = nn_jan_jul.SPLITS_FROM
V47_RUN = nn_jan_jul.OUT
OUT = RUNS / 'v3_20261003_nn_improve_jan_jul'
SEEDS = [42, 142, 242]
ARMS = ['avg5', 'ple', 'tabm']
NETWORKS = dict(avg5=5, ple=5, tabm=1)
NEW = dict(bins=48, d_ple=16, epochs=24, batch=4096,
           ple=dict(lr=2e-3, weight_decay=1e-5),
           tabm=dict(lr=2e-3, weight_decay=3e-4, k=32, n_blocks=2, d_block=512, dropout=0.1,
                     activation='ReLU', arch_type='tabm', start_scaling_init='normal'))


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def load_month(month, dep, arr, table):
    """`nn_jan_jul.load_month`, plus the calendar month of every training row."""
    d = ts.split_dir(SPLITS_FROM, month)
    meta = ts.read(d / 'meta.json')
    td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
    train, held = pd.read_pickle(d / 'after_drop.pkl'), pd.read_pickle(d / 'heldout.pkl')
    names = meta['regressor_features']
    s = rome_hybrid.substituted(train['X'], train['y'], ts.minimal_raw(train['airport'], train['X'].index))
    X = train['X'].loc[~s, names]
    ids = pd.DataFrame({'mvt_id': td.loc[X.index, 'mvt_id'].to_numpy()}, index=X.index)
    tmonth = pd.to_datetime(td.loc[X.index, 'MVT_TIME_UTC_mvt'], utc=True).dt.strftime('%Y-%m').to_numpy()
    X = adsblol.add_adsblol(X, ids, table, nn_jan_jul.FEATURES)
    V, raw = held['V'], held['raw']
    Vn = adsblol.add_adsblol(V.loc[:, names], raw[['mvt_id']].set_index(V.index), table, nn_jan_jul.FEATURES)
    return dict(meta=meta, X=X, y=np.asarray(train['y'], float)[~s], V=V, Vn=Vn, raw=raw, p=held['p'],
                tmonth=tmonth)


def emb_dims(cards):
    return [min(24, max(2, round(1.6 * k ** 0.56))) for k in cards]


class Inputs(nn.Module):
    """v47's categorical embeddings + piecewise-linear embeddings + plain numeric inputs."""

    def __init__(self, cards, bins, n_plain):
        super().__init__()
        dims = emb_dims(cards)
        self.emb = nn.ModuleList(nn.Embedding(k, d) for k, d in zip(cards, dims))
        self.ple = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, NEW['d_ple'], activation=False, version='B')
        self.chunks = dims + [NEW['d_ple']] * len(bins) + [1] * n_plain

    def forward(self, cat, g, plain):
        return torch.cat([e(cat[:, i]) for i, e in enumerate(self.emb)] + [self.ple(g).flatten(1), plain], dim=1)


class PleNet(nn.Module):
    def __init__(self, cards, bins, n_plain):
        super().__init__()
        self.inputs = Inputs(cards, bins, n_plain)
        layers, width = [], sum(self.inputs.chunks)
        for h in nn_jan_jul.RECIPE['hidden']:
            layers += [nn.Linear(width, h), nn.SiLU(), nn.Dropout(nn_jan_jul.RECIPE['dropout'])]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, g, plain):
        return self.mlp(self.inputs(cat, g, plain))            # (batch, 1): one member


class TabMNet(nn.Module):
    def __init__(self, cards, bins, n_plain):
        super().__init__()
        p = NEW['tabm']
        self.inputs = Inputs(cards, bins, n_plain)
        self.view = tabm.EnsembleView(k=p['k'])
        self.backbone = tabm.make_tabm_backbone(
            d_in=sum(self.inputs.chunks), n_blocks=p['n_blocks'], d_block=p['d_block'], dropout=p['dropout'],
            activation=p['activation'], k=p['k'], arch_type=p['arch_type'],
            start_scaling_init=p['start_scaling_init'], start_scaling_init_chunks=self.inputs.chunks)
        self.output = tabm.LinearEnsemble(p['d_block'], 1, k=p['k'])

    def forward(self, cat, g, plain):
        return self.output(self.backbone(self.view(self.inputs(cat, g, plain)))).squeeze(-1)   # (batch, k)


class Data:
    """v47's encoded inputs on the device, split into PLE / plain columns; bins from training rows."""

    def __init__(self, item, device):
        enc = nn_jan_jul.Encoder(item['X'])
        cat, num = enc.transform(item['X'])
        vcat, vnum = enc.transform(item['Vn'])
        g, plain, j = [], [], 0
        for c in enc.nums:
            g.append(j)
            plain.append(j + 1)                                  # clipped linear copy
            j += 2
            if c in enc.flags:
                plain.append(j)                                  # missing flag
                j += 1
        if j != num.shape[1]:
            raise ValueError('Encoder column layout changed')
        edges = np.quantile(num[:, g], np.linspace(0, 1, NEW['bins'] + 1), axis=0)
        bins, ple = [], []
        for i, col in enumerate(g):
            e = np.unique(edges[:, i].astype(np.float32))
            if len(e) >= 3:
                bins.append(torch.from_numpy(e))
                ple.append(col)
            else:
                plain.append(col)
        self.bins, self.cards = bins, enc.cards
        self.info = dict(numeric=len(enc.nums), ple_inputs=len(ple), plain_inputs=len(plain),
                         categoricals=enc.cats, cards=enc.cards, bins_per_input=[len(b) - 1 for b in bins])
        y = np.minimum(item['y'], nn_jan_jul.RECIPE['label_clip'])
        self.mu, self.sd = float(y.mean()), float(y.std())
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.train = (t(cat), t(num[:, ple]), t(num[:, plain]))
        self.held = (t(vcat), t(vnum[:, ple]), t(vnum[:, plain]))
        self.ty = t(((y - self.mu) / self.sd).astype(np.float32))
        self.y = np.asarray(item['y'], float)
        self.yv = item['raw'].taxi_sec.to_numpy(float)
        last = max(ts.FOLDS[item['meta']['month']])
        self.inner_val = np.flatnonzero(item['tmonth'] == last)
        self.inner_train = np.flatnonzero(item['tmonth'] != last)
        self.info.update(inner_month=last, inner_val_rows=len(self.inner_val), inner_train_rows=len(self.inner_train))
        self.device = device

    def subset(self, rows):
        r = torch.from_numpy(rows).to(self.device)
        return tuple(a[r] for a in self.train)


def predict(model, data, sd, mu, chunk):
    model.eval()
    with torch.no_grad():
        n = len(data[0])
        out = torch.cat([model(*(a[i:i + chunk] for a in data)).mean(1) for i in range(0, n, chunk)])
    model.train()
    return np.maximum(out.cpu().numpy().astype(float) * sd + mu, 0)


def rmse(y, p):
    return float(np.sqrt(((np.asarray(y, float) - np.asarray(p, float)) ** 2).mean()))


def train(arm, d, rows, seed, stop, monitor):
    """Fit on `rows` under a one-cycle schedule of NEW['epochs'] epochs, stopping after `stop`.

    `monitor` = (inputs, labels): RMSE of the network's prediction after every epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (PleNet if arm == 'ple' else TabMNet)(d.cards, d.bins, d.train[2].shape[1]).to(d.device)
    opt = torch.optim.AdamW(model.parameters(), lr=NEW[arm]['lr'], weight_decay=NEW[arm]['weight_decay'])
    batch, n = NEW['batch'], len(rows)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=NEW[arm]['lr'],
                                                total_steps=NEW['epochs'] * math.ceil(n / batch))
    gen = torch.Generator(device=d.device).manual_seed(seed)
    rows_t = torch.from_numpy(rows).to(d.device)
    chunk = 65536 if arm == 'ple' else 4096
    curve = []
    for epoch in range(stop):
        started, total = time.monotonic(), 0.0
        perm = rows_t[torch.randperm(n, device=d.device, generator=gen)]
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            out = model(*(a[b] for a in d.train))
            loss = nn.functional.mse_loss(out, d.ty[b, None].expand_as(out))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(b)
        pred = predict(model, monitor[0], d.sd, d.mu, chunk)
        curve.append(dict(epoch=epoch + 1, train_mse_std=total / n, rmse=rmse(monitor[1], pred),
                          seconds=time.monotonic() - started))
        log(f'    epoch {epoch + 1}: train {total / n:.4f}, monitored RMSE {curve[-1]["rmse"]:.2f} '
            f'({curve[-1]["seconds"]:.0f}s)')
    return model, pred, curve


def network_path(arm, month, seed, k):
    return OUT / 'networks' / f'{arm}_{month}_s{seed}_k{k}.parquet'


def save_network(path, ids, base):
    tmp = path.with_suffix('.tmp')
    pd.DataFrame({'mvt_id': ids, 'base': base.astype(np.float32)}).to_parquet(tmp, index=False)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--arms', nargs='+', default=ARMS, choices=ARMS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--months', nargs='+', default=ts.SPLITS, choices=ts.SPLITS)
    args = ap.parse_args()
    device = 'cuda'
    (OUT / 'networks').mkdir(parents=True, exist_ok=True)
    result_path = OUT / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else dict(months={})
    result.update(recipe=dict(new=NEW, networks=NETWORKS, v47=nn_jan_jul.RECIPE), device=torch.cuda.get_device_name(),
                  torch=torch.__version__, tabm=tabm.__version__ if hasattr(tabm, '__version__') else '0.0.3',
                  source_sha256=dict(runner=sha256(Path(__file__)), v47_runner=sha256(Path(nn_jan_jul.__file__)),
                                     tabm=sha256(Path(tabm.__file__)),
                                     rtdl_num_embeddings=sha256(Path(rtdl_num_embeddings.__file__))))
    v47 = pd.read_parquet(V47_RUN / 'predictions' / 'nn_seeds42_142_242.parquet')
    table = adsblol.load_all()
    dep, arr = overnight_run.load_inputs(1.)
    for month in args.months:
        item = load_month(month, dep, arr, table)
        ids = item['raw'].mvt_id.to_numpy()
        saved = v47[v47.month == month].set_index('mvt_id').reindex(ids)
        if saved.isna().any().any():
            raise ValueError(f'{month}: v47 predictions do not cover the held-out rows')
        log(f'{month}: training {item["X"].shape}, held-out {item["Vn"].shape}')
        mm = result['months'].setdefault(month, {})
        for arm in args.arms:
            am = mm.setdefault(arm, {})
            if arm == 'avg5':
                for seed in args.seeds:
                    sm = am.setdefault(str(seed), dict(networks={}))
                    path = network_path(arm, month, seed, 0)
                    if not path.exists():
                        save_network(path, ids, saved[f'nn_base_{seed}'].to_numpy(float))
                        sm['networks']['0'] = dict(source='v47 saved prediction')
                    for k in range(1, NETWORKS[arm]):
                        path = network_path(arm, month, seed, k)
                        if path.exists():
                            continue
                        log(f'{month} {arm} seed {seed} network {k}')
                        started = time.monotonic()
                        base, curve, info = nn_jan_jul.fit_nn(item, seed + 1000 * k, device)
                        save_network(path, ids, base)
                        sm['networks'][str(k)] = dict(curve=curve, minutes=(time.monotonic() - started) / 60,
                                                      heldout_base_rmse=rmse(item['raw'].taxi_sec, base))
                        write_json(result_path, result)
                continue
            torch.backends.cuda.matmul.allow_tf32 = arm == 'tabm'
            d = Data(item, device)
            am['inputs'] = d.info
            log(f'{month} {arm}: {d.info["ple_inputs"]} PLE + {d.info["plain_inputs"]} plain inputs; inner '
                f'{d.info["inner_month"]} {d.info["inner_val_rows"]:,} rows, rest {d.info["inner_train_rows"]:,}')
            inner_monitor = (d.subset(d.inner_val), d.y[d.inner_val])
            for seed in args.seeds:
                sm = am.setdefault(str(seed), dict(networks={}))
                if 'stop_epoch' not in sm:
                    log(f'{month} {arm} seed {seed}: epoch choice on {d.info["inner_month"]}')
                    started = time.monotonic()
                    _, _, curve = train(arm, d, d.inner_train, seed, NEW['epochs'], inner_monitor)
                    best = min(curve, key=lambda c: c['rmse'])['epoch']
                    sm.update(inner_curve=curve, stop_epoch=best, inner_minutes=(time.monotonic() - started) / 60)
                    write_json(result_path, result)
                    log(f'{month} {arm} seed {seed}: stop after epoch {best}')
                for k in range(NETWORKS[arm]):
                    path = network_path(arm, month, seed, k)
                    if path.exists():
                        continue
                    log(f'{month} {arm} seed {seed} network {k} ({sm["stop_epoch"]} epochs)')
                    started = time.monotonic()
                    _, base, curve = train(arm, d, np.arange(len(d.y)), seed + 1000 * k, sm['stop_epoch'],
                                           (d.held, d.yv))
                    save_network(path, ids, base)
                    sm['networks'][str(k)] = dict(curve=curve, minutes=(time.monotonic() - started) / 60,
                                                  heldout_base_rmse=rmse(d.yv, base))
                    write_json(result_path, result)
            torch.backends.cuda.matmul.allow_tf32 = False
            del d, inner_monitor
            gc.collect()
            torch.cuda.empty_cache()
        del item
        gc.collect()
        torch.cuda.empty_cache()
    write_json(result_path, result)
    log('done')


if __name__ == '__main__':
    main()
