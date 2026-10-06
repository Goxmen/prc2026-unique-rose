"""Jan/Jul CV: v48's `ple` networks refitted with extra inputs from `ideas_jan_jul.py`.

The recipe is `nn_improve_jan_jul.py`'s `ple` arm unchanged (piecewise-linear embeddings, epoch
chosen on the last training month, five networks per seed 42/142/242); only the input columns
grow by the named groups (extra_features.GROUPS and/or `route` from taxi_route.py). Networks are
saved under the ideas run directory as `nn_<tag>_<month>_s<seed>_k<k>.parquet`;
`score_ideas_jan_jul.py --nets <tag>` blends them with an arm's LightGBM.

Run: .venv/bin/python VALIDATION/nn_features_jan_jul.py --groups arrtaxi peers --tag combo
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')     # as nn_improve_jan_jul.py, before torch loads
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import extra_features as ef
import nn_improve_jan_jul as nni
import nn_jan_jul
import overnight_run
import rome_hybrid
import seasonal
import taxi_route
import tuning_seasonal as ts
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261003_ideas_jan_jul'


def columns(groups):
    return [c for g in groups for c in (taxi_route.FEATURES if g == 'route' else ef.GROUPS[g])]


def load_month(month, dep, arr, table, extra, routes, groups):
    """nn_improve_jan_jul.load_month plus the named input groups."""
    d = ts.split_dir(nni.SPLITS_FROM, month)
    meta = ts.read(d / 'meta.json')
    td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
    train, held = pd.read_pickle(d / 'after_drop.pkl'), pd.read_pickle(d / 'heldout.pkl')
    names = meta['regressor_features']
    s = rome_hybrid.substituted(train['X'], train['y'], ts.minimal_raw(train['airport'], train['X'].index))
    X = train['X'].loc[~s, names]
    rows = td.loc[X.index]
    ids = rows.mvt_id.to_numpy()
    tmonth = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').to_numpy()
    X = adsblol.add_adsblol(X, pd.DataFrame({'mvt_id': ids}, index=X.index), table, nn_jan_jul.FEATURES)
    V, raw = held['V'], held['raw']
    Vn = adsblol.add_adsblol(V.loc[:, names], raw[['mvt_id']].set_index(V.index), table, nn_jan_jul.FEATURES)
    plain = [c for c in columns(groups) if c not in taxi_route.FEATURES]
    if plain:
        X, Vn = ef.add(X, ids, extra, plain), ef.add(Vn, raw.mvt_id.to_numpy(), extra, plain)
    if 'route' in groups:
        X = taxi_route.add(X, rows.ADEP_mvt, rows.STAND_mvt, rows.RUNWAY_mvt, routes)
        Vn = taxi_route.add(Vn, raw.ADEP_mvt, raw.STAND_mvt, raw.RUNWAY_mvt, routes)
    return dict(meta=meta, X=X, y=np.asarray(train['y'], float)[~s], V=V, Vn=Vn, raw=raw, p=held['p'], tmonth=tmonth)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--groups', nargs='+', required=True, choices=sorted(ef.GROUPS) + ['route'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=nni.SEEDS)
    args = ap.parse_args()
    (OUT / 'networks').mkdir(parents=True, exist_ok=True)
    record_path = OUT / f'nn_{args.tag}.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record.update(groups=args.groups, columns=columns(args.groups), recipe=dict(new=nni.NEW, v47=nn_jan_jul.RECIPE),
                  source=dict(runner=sha256(Path(__file__)), nn_improve=sha256(Path(nni.__file__)),
                              extra_features=sha256(ef.PATH)))
    table, extra = adsblol.load_all(), ef.load()
    routes = taxi_route.load() if 'route' in args.groups else None
    dep, arr = overnight_run.load_inputs(1.)
    for month in ts.SPLITS:
        item = load_month(month, dep, arr, table, extra, routes, args.groups)
        ids = item['raw'].mvt_id.to_numpy()
        d = nni.Data(item, 'cuda')
        mm = record.setdefault(month, dict(inputs=d.info))
        nni.log(f'{month}: {d.info["ple_inputs"]} PLE + {d.info["plain_inputs"]} plain inputs')
        inner = (d.subset(d.inner_val), d.y[d.inner_val])
        for seed in args.seeds:
            sm = mm.setdefault(str(seed), {})
            if 'stop_epoch' not in sm:
                _, _, curve = nni.train('ple', d, d.inner_train, seed, nni.NEW['epochs'], inner)
                sm.update(stop_epoch=min(curve, key=lambda c: c['rmse'])['epoch'], inner_curve=curve)
                write_json(record_path, record)
            for k in range(nni.NETWORKS['ple']):
                path = OUT / 'networks' / f'nn_{args.tag}_{month}_s{seed}_k{k}.parquet'
                if path.exists():
                    continue
                started = time.monotonic()
                _, base, curve = nni.train('ple', d, np.arange(len(d.y)), seed + 1000 * k, sm['stop_epoch'], (d.held, d.yv))
                nni.save_network(path, ids, base)
                sm[str(k)] = dict(minutes=(time.monotonic() - started) / 60, heldout_base_rmse=nni.rmse(d.yv, base))
                write_json(record_path, record)
            nni.log(f'{month} seed {seed}: stop {sm["stop_epoch"]}')
        del d, inner, item
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
