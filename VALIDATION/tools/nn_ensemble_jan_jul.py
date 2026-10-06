"""Extra idea: a larger network ensemble from networks already fitted (saved predictions only).

v48's recipe blends each LightGBM seed with that seed's five `ple` networks. Without new fits:
  ple15       the network part is the mean of all fifteen `ple` networks of the month (the three
              seed groups pooled), for every LightGBM seed
  ple5_tabm   the network part is 0.5 x that seed's five `ple` networks + 0.5 x its TabM network
  ple15_tabm  0.75 x all fifteen `ple` networks + 0.25 x the mean of the three TabM networks
Network share 0.6 as in v48. Paired day bootstrap against v48's recipe; all variants fixed
before scoring.

Run: .venv/bin/python VALIDATION/tools/nn_ensemble_jan_jul.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import score_ideas_jan_jul as sc
import tuning_seasonal as ts
from records import write_json


def main():
    months = {m: sc.load_month(m, [], sc.NETS) for m in ts.SPLITS}
    se = {}
    for m, d in months.items():
        ids = None
        tabm = {}
        for s in sc.SEEDS:
            f = pd.read_parquet(sc.NETS / f'tabm_{m}_s{s}_k0.parquet')
            tabm[s] = f.base.to_numpy(float)
        ple15 = np.mean([d['nn'][s] for s in sc.SEEDS], axis=0)
        tabm3 = np.mean(list(tabm.values()), axis=0)
        variants = {'v48': lambda s: d['nn'][s], 'ple15': lambda s: ple15,
                    'ple5_tabm': lambda s: 0.5 * d['nn'][s] + 0.5 * tabm[s],
                    'ple15_tabm': lambda s: 0.75 * ple15 + 0.25 * tabm3}
        for name, net in variants.items():
            se.setdefault(name, {})[m] = np.mean(
                [(d['y'] - d['final'](sc.W_NN * net(s) + (1 - sc.W_NN) * d['lgb'][s])) ** 2 for s in sc.SEEDS], axis=0)
    month = np.concatenate([np.full(len(months[m]['y']), m) for m in ts.SPLITS])
    day = np.concatenate([months[m]['day'] for m in ts.SPLITS])
    flat = {k: np.concatenate([v[m] for m in ts.SPLITS]) for k, v in se.items()}
    level = lambda e: float(np.mean([np.sqrt(e[month == m].mean()) for m in ts.SPLITS]))
    out = {k: dict(cv=level(v), change=None if k == 'v48' else ts.paired_bootstrap(day, month, flat['v48'], v))
           for k, v in flat.items()}
    write_json(sc.RUN / 'nn_ensemble.json', out)
    for k, v in out.items():
        c = v['change']
        print(k, round(v['cv'], 4), '' if c is None else (round(c['change'], 4), [round(x, 3) for x in c['ci95']], c['decision']))


if __name__ == '__main__':
    main()
