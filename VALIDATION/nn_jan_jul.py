"""Jan/Jul CV of a tabular neural network beside v46's ordinary LightGBM regressor.

Research question (2026-10-02): does an MLP trained on exactly v46's ordinary-regressor
inputs add anything to v46 when blended? Both use the prepared Jan/Jul splits of
`adsblol_jan_jul.py` (same training rows, Rome-substituted rows left out, same held-out
rows); the LightGBM side is that runner's saved `stand` arm (v46's recipe) at each seed.
Every prediction goes through the same Rome classifier, corrections and rules
(`tuning_seasonal.final_prediction`) before it is scored.

Frozen recipe (fixed before any v46 run, not tuned on the held-out months):
  inputs   v46's 134 ordinary inputs. 11 categoricals -> embeddings (values with < 20
           training rows, unseen values and missing share index 0). Each numeric input ->
           rank-gauss value (1,024 training quantiles), a clipped linear copy (median/IQR
           scaled, clipped to +-5) and, if it has missing values in training, a missing flag;
           missing values are 0 in both copies.
  network  embeddings + numerics -> 512 -> 512 -> 256 -> 1, SiLU, dropout 0.1
  target   label clipped at 3 h, standardised; MSE loss
  training AdamW (lr 2e-3, weight decay 1e-5), one-cycle schedule, batch 4,096, 8 epochs,
           final epoch kept (no early stopping on held-out data)
Blend: base = w * NN + (1 - w) * LightGBM, then the shared final prediction. The decision
uses w = 0.3, fixed in advance (`VALIDATION/tools/score_nn_jan_jul.py`); the weight grid and
the cross-month choice printed here are diagnostics.

Held-out RMSE per epoch is printed as a diagnostic only.

Run: .venv/bin/python VALIDATION/nn_jan_jul.py [--seeds 42]
No full-year fits, submissions or ledger edits.
"""
import argparse
import gc
from pathlib import Path
from statistics import NormalDist
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import modeling
import overnight_run
import rome_hybrid
import seasonal
import tuning_seasonal as ts
from records import RUNS, sha256, write_json

SPLITS_FROM = RUNS / 'v3_20261002_adsblol_jan_jul'
OUT = RUNS / 'v3_20261002_nn_v46_jan_jul'
FEATURES = adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES   # v46's additions to the ordinary regressor
LGB_ARM = 'stand'                                            # adsblol_jan_jul.py's arm for v46's recipe
RECIPE = dict(min_cat_rows=20, quantiles=1024, clip=5.0, hidden=[512, 512, 256], dropout=0.1,
              label_clip=10800.0, lr=2e-3, weight_decay=1e-5, batch=4096, epochs=8)
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def load_month(month, dep, arr, table):
    """Training and held-out frames exactly as adsblol_jan_jul.load builds them for arm `stand`."""
    d = ts.split_dir(SPLITS_FROM, month)
    meta = ts.read(d / 'meta.json')
    td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
    train, held = pd.read_pickle(d / 'after_drop.pkl'), pd.read_pickle(d / 'heldout.pkl')
    names = meta['regressor_features']
    s = rome_hybrid.substituted(train['X'], train['y'], ts.minimal_raw(train['airport'], train['X'].index))
    X = train['X'].loc[~s, names]
    ids = pd.DataFrame({'mvt_id': td.loc[X.index, 'mvt_id'].to_numpy()}, index=X.index)
    X = adsblol.add_adsblol(X, ids, table, FEATURES)
    V, raw = held['V'], held['raw']
    Vn = adsblol.add_adsblol(V.loc[:, names], raw[['mvt_id']].set_index(V.index), table, FEATURES)
    return dict(meta=meta, X=X, y=np.asarray(train['y'], float)[~s], V=V, Vn=Vn, raw=raw, p=held['p'])


class Encoder:
    """Training-fitted transforms; applied unchanged to the held-out month."""

    def __init__(self, X):
        self.cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        self.nums = [c for c in X.columns if c not in self.cats]
        self.maps, self.q, self.scale, self.flags = {}, {}, {}, []
        for c in self.cats:
            counts = X[c].astype(object).value_counts()
            keep = counts.index[counts >= RECIPE['min_cat_rows']]
            self.maps[c] = {v: i + 1 for i, v in enumerate(sorted(map(str, keep)))}
        probs = (np.arange(RECIPE['quantiles']) + 0.5) / RECIPE['quantiles']
        normal = np.array([NormalDist().inv_cdf(p) for p in probs])
        for c in self.nums:
            v = X[c].to_numpy(float)
            v = v[np.isfinite(v)]
            qs = np.quantile(v, probs) if len(v) else np.zeros(1)
            xs, idx = np.unique(qs, return_inverse=True)
            ys = np.array([normal[idx == i].mean() for i in range(len(xs))]) if len(v) else np.zeros(1)
            self.q[c] = (xs, ys)
            med = float(np.median(v)) if len(v) else 0.0
            iqr = float(np.subtract(*np.quantile(v, [0.75, 0.25]))) if len(v) else 1.0
            self.scale[c] = (med, iqr if iqr > 0 else (float(v.std()) or 1.0))
            if X[c].isna().any():
                self.flags.append(c)
        self.cards = [len(self.maps[c]) + 1 for c in self.cats]

    def transform(self, X):
        cols = []
        for c in self.cats:
            lut = np.array([self.maps[c].get(str(v), 0) for v in X[c].cat.categories] + [0], np.int64)
            cols.append(lut[X[c].cat.codes.to_numpy()])   # code -1 (missing) reads the trailing 0
        cat = np.stack(cols, axis=1)
        parts = []
        for c in self.nums:
            v = X[c].to_numpy(float)
            bad = ~np.isfinite(v)
            xs, ys = self.q[c]
            g = np.interp(np.where(bad, xs[0], v), xs, ys) if len(xs) > 1 else np.zeros(len(v))
            med, s = self.scale[c]
            lin = np.clip((np.where(bad, med, v) - med) / s, -RECIPE['clip'], RECIPE['clip'])
            parts += [np.where(bad, 0, g), np.where(bad, 0, lin)]
            if c in self.flags:
                parts.append(bad.astype(float))
        return cat, np.stack(parts, axis=1).astype(np.float32)


class Net(nn.Module):
    def __init__(self, cards, n_num):
        super().__init__()
        dims = [min(24, max(2, round(1.6 * k ** 0.56))) for k in cards]
        self.emb = nn.ModuleList(nn.Embedding(k, d) for k, d in zip(cards, dims))
        layers, width = [], sum(dims) + n_num
        for h in RECIPE['hidden']:
            layers += [nn.Linear(width, h), nn.SiLU(), nn.Dropout(RECIPE['dropout'])]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, num):
        z = torch.cat([e(cat[:, i]) for i, e in enumerate(self.emb)] + [num], dim=1)
        return self.mlp(z).squeeze(1)


def fit_nn(item, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder(item['X'])
    cat, num = enc.transform(item['X'])
    vcat, vnum = enc.transform(item['Vn'])
    y = np.minimum(item['y'], RECIPE['label_clip'])
    mu, sd = float(y.mean()), float(y.std())
    tc, tn = torch.from_numpy(cat).to(device), torch.from_numpy(num).to(device)
    ty = torch.from_numpy(((y - mu) / sd).astype(np.float32)).to(device)
    vc, vn = torch.from_numpy(vcat).to(device), torch.from_numpy(vnum).to(device)
    del cat, num
    model = Net(enc.cards, tn.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    steps = RECIPE['epochs'] * ((len(ty) + RECIPE['batch'] - 1) // RECIPE['batch'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=RECIPE['lr'], total_steps=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    yv = item['raw'].taxi_sec.to_numpy(float)
    curve = []

    def predict():
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(vc[i:i + 65536], vn[i:i + 65536]) for i in range(0, len(vc), 65536)])
        model.train()
        return np.maximum(out.cpu().numpy().astype(float) * sd + mu, 0)

    for epoch in range(RECIPE['epochs']):
        started, total = time.monotonic(), 0.0
        perm = torch.randperm(len(ty), device=device, generator=gen)
        for i in range(0, len(ty), RECIPE['batch']):
            b = perm[i:i + RECIPE['batch']]
            loss = nn.functional.mse_loss(model(tc[b], tn[b]), ty[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(b)
        base = predict()
        curve.append(dict(epoch=epoch + 1, train_mse_std=total / len(ty),
                          heldout_base_rmse=float(np.sqrt(((base - yv) ** 2).mean())),
                          seconds=time.monotonic() - started))
        log(f'  epoch {epoch + 1}: train {total / len(ty):.4f}, held-out base RMSE {curve[-1]["heldout_base_rmse"]:.2f} '
            f'({curve[-1]["seconds"]:.0f}s)')
    return base, curve, dict(inputs=tn.shape[1], categoricals=enc.cats, cards=enc.cards, label_mean=mu, label_sd=sd)


def rmse(y, p):
    return float(np.sqrt(((np.asarray(y, float) - np.asarray(p, float)) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (OUT / 'predictions').mkdir(parents=True, exist_ok=True)
    table = adsblol.load_all()
    dep, arr = overnight_run.load_inputs(1.)
    result = dict(recipe=RECIPE, features=FEATURES, lgb_arm=LGB_ARM,
                  device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                  torch=torch.__version__, source_sha256=dict(runner=sha256(Path(__file__)),
                  features=sha256(adsblol.FEATURES_PATH), stand_features=sha256(adsblol.STAND_PATH)), months={})
    frames = []
    for month in ts.SPLITS:
        item = load_month(month, dep, arr, table)
        log(f'{month}: training {item["X"].shape}, held-out {item["Vn"].shape}')
        mm = result['months'][month] = dict(seeds={})
        y = item['raw'].taxi_sec.to_numpy(float)
        frame = pd.DataFrame({'mvt_id': item['raw'].mvt_id.to_numpy(), 'month': month})
        final = lambda base: ts.final_prediction(base, item['p'], item['V'], item['raw'],
                                                 item['meta']['parameters'], item['meta']['corrections'])
        for seed in args.seeds:
            booster = lgb.Booster(model_file=str(SPLITS_FROM / 'models' / f'{LGB_ARM}_{month}_seed{seed}.txt'))
            lgb_base = modeling.predict(booster, item['Vn'][booster.feature_name()], threads=ts.THREADS)
            saved = pd.read_parquet(SPLITS_FROM / 'predictions' / f'{LGB_ARM}_seed{seed}.parquet')
            saved = saved.set_index('mvt_id').r5000.reindex(frame.mvt_id).to_numpy(float)
            lgb_final = final(lgb_base)
            check = float(np.max(np.abs(lgb_final - saved)))
            if check > 0.05:
                raise ValueError(f'{month} seed {seed}: LightGBM does not reproduce v46 predictions ({check:.4f} s)')
            log(f'{month} seed {seed}: LightGBM reproduces saved v46 predictions to {check:.4f} s')
            started = time.monotonic()
            nn_base, curve, info = fit_nn(item, seed, device)
            minutes = (time.monotonic() - started) / 60
            grid = {float(w): rmse(y, final(w * nn_base + (1 - w) * lgb_base)) for w in WEIGHTS}
            mm['seeds'][seed] = dict(lgb_reproduction_max_abs=check, curve=curve, info=info, minutes=minutes,
                                     lgb_base_rmse=rmse(y, lgb_base), nn_base_rmse=rmse(y, nn_base),
                                     corr_base=float(np.corrcoef(lgb_base, nn_base)[0, 1]), blend_final_rmse=grid)
            frame[f'lgb_base_{seed}'] = lgb_base.astype(np.float32)
            frame[f'nn_base_{seed}'] = nn_base.astype(np.float32)
            log(f'{month} seed {seed}: LightGBM {grid[0.0]:.2f}, NN alone {grid[1.0]:.2f}, best blend '
                f'{min(grid.values()):.2f} at w={min(grid, key=grid.get)} ({minutes:.1f} min)')
        frames.append(frame)
        del item
        gc.collect()
        torch.cuda.empty_cache()
    del dep, arr
    out = pd.concat(frames, ignore_index=True)
    overnight_run.save_frame(OUT / 'predictions' / f'nn_seeds{"_".join(map(str, args.seeds))}.parquet', out)
    # Cross-month weight choice: pick w on one month, score on the other.
    for seed in args.seeds:
        g = {m: result['months'][m]['seeds'][seed]['blend_final_rmse'] for m in ts.SPLITS}
        best = {m: min(g[m], key=g[m].get) for m in ts.SPLITS}
        a, b = ts.SPLITS
        cross = {a: g[a][best[b]], b: g[b][best[a]]}
        result.setdefault('cross_month', {})[seed] = dict(
            weight_chosen_on_other_month={a: best[b], b: best[a]}, final_rmse=cross,
            average=float(np.mean(list(cross.values()))),
            lgb_average=float(np.mean([g[m][0.0] for m in ts.SPLITS])),
            nn_average=float(np.mean([g[m][1.0] for m in ts.SPLITS])))
        c = result['cross_month'][seed]
        log(f'seed {seed}: Jan/Jul CV LightGBM {c["lgb_average"]:.4f}, NN {c["nn_average"]:.4f}, '
            f'cross-month blend {c["average"]:.4f} (weights {c["weight_chosen_on_other_month"]})')
    write_json(OUT / f'result_seeds{"_".join(map(str, args.seeds))}.json', result)


if __name__ == '__main__':
    main()
