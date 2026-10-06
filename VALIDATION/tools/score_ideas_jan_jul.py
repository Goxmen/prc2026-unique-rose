"""Score `ideas_jan_jul.py` / `catboost_jan_jul.py` arms against v48's recipe, saved predictions only.

For every arm and seed the gradient-boosting part (GBM) is rebuilt and blended with the mean of the
five `ple` networks of that seed, at v48's weight: base = 0.6 network + 0.4 GBM, then the shared
Rome classifier mix, corrections and rules. Control: GBM = v46's LightGBM (v48's recipe).
  feature arms (arrtaxi, peers, rwycfg, route, combo): GBM = the arm's LightGBM
  airport:  GBM = 0.5 v46 global + 0.5 per-airport LightGBM
  catboost: GBM = 0.5 v46 LightGBM + 0.5 CatBoost
Squared errors are averaged over seeds within each month; intervals resample whole days within
each month. The GBM alone (before the network blend) is scored the same way.

Decision, fixed before any arm was scored: an arm passes if its blend change against the control
has a 95% interval below zero and all three seeds improve. Passing feature arms are then combined
into one arm. The cross-month weight readout (network and GBM shares chosen on the other month)
is a diagnostic.

Run: .venv/bin/python VALIDATION/tools/score_ideas_jan_jul.py --arms arrtaxi peers ...
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import tuning_seasonal as ts
from records import write_json

SPLITS_FROM = ts.RUNS / 'v3_20261002_adsblol_jan_jul'
V47 = ts.RUNS / 'v3_20261002_nn_v46_jan_jul' / 'predictions' / 'nn_seeds42_142_242.parquet'
NETS = ts.RUNS / 'v3_20261003_nn_improve_jan_jul' / 'networks'
RUN = ts.RUNS / 'v3_20261003_ideas_jan_jul'
SEEDS, W_NN = [42, 142, 242], 0.6
MIXED = {'airport': 0.5, 'catboost': 0.5}           # share of the arm inside the GBM part
NN_GRID = np.round(np.arange(0.3, 0.851, 0.05), 2)
MIX_GRID = np.round(np.arange(0, 1.001, 0.1), 2)


def load_month(month, arms, nets_dir, tag=None):
    d = ts.split_dir(SPLITS_FROM, month)
    held, meta = pd.read_pickle(d / 'heldout.pkl'), ts.read(d / 'meta.json')
    raw = held['raw']
    ids = raw.mvt_id.to_numpy()
    v47 = pd.read_parquet(V47)
    saved = v47[v47.month == month].set_index('mvt_id').reindex(ids)
    final = lambda base, held=held, raw=raw, meta=meta: ts.final_prediction(
        base, held['p'], held['V'], raw, meta['parameters'], meta['corrections'])
    out = dict(final=final, y=raw.taxi_sec.to_numpy(float), airport=raw.ADEP_mvt.astype(str).to_numpy(),
               day=pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy(),
               lgb={}, nn={}, arm={a: {} for a in arms})
    for s in SEEDS:
        out['lgb'][s] = saved[f'lgb_base_{s}'].to_numpy(float)
        nets = [pd.read_parquet(nets_dir / f'ple_{month}_s{s}_k{k}.parquet') for k in range(5)]
        if any(not np.array_equal(n.mvt_id.to_numpy(), ids) for n in nets):
            raise ValueError(f'{month}: network rows differ')
        out['nn'][s] = np.mean([n.base.to_numpy(float) for n in nets], axis=0)
        if tag:                                   # networks refitted with the arms' extra inputs
            alt = [pd.read_parquet(RUN / 'networks' / f'nn_{tag}_{month}_s{s}_k{k}.parquet') for k in range(5)]
            if any(not np.array_equal(n.mvt_id.to_numpy(), ids) for n in alt):
                raise ValueError(f'{month}: {tag} network rows differ')
            out.setdefault('nn_alt', {})[s] = np.mean([n.base.to_numpy(float) for n in alt], axis=0)
        for a in arms:
            f = pd.read_parquet(RUN / 'predictions' / f'{a}_seed{s}.parquet')
            f = f[f.month == month]
            if not np.array_equal(f.mvt_id.to_numpy(), ids):
                raise ValueError(f'{a} {month} seed {s}: rows differ')
            out['arm'][a][s] = f.base.to_numpy(float)
    return out


def nn(m, arm, s):
    return m['nn_alt'][s] if arm != 'control' and 'nn_alt' in m else m['nn'][s]


def gbm(m, arm, s):
    if arm == 'control':
        return m['lgb'][s]
    share = MIXED.get(arm, 1.0)
    return share * m['arm'][arm][s] + (1 - share) * m['lgb'][s]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--out', default='score.json')
    ap.add_argument('--nets', help='blend the arms with networks nn_<NETS>_* instead of the ple networks')
    args = ap.parse_args()
    months = {m: load_month(m, args.arms, NETS, args.nets) for m in ts.SPLITS}
    names = ['control'] + args.arms
    se = {(n, kind): {} for n in names for kind in ('blend', 'gbm')}
    by_seed = {}
    for m, d in months.items():
        for n in names:
            per_blend, per_gbm = [], []
            for s in SEEDS:
                g = gbm(d, n, s)
                per_blend.append((d['y'] - d['final'](W_NN * nn(d, n, s) + (1 - W_NN) * g)) ** 2)
                per_gbm.append((d['y'] - d['final'](g)) ** 2)
                by_seed.setdefault((n, s), {})[m] = float(np.sqrt(per_blend[-1].mean()))
            se[n, 'blend'][m] = np.mean(per_blend, axis=0)
            se[n, 'gbm'][m] = np.mean(per_gbm, axis=0)
    month = np.concatenate([np.full(len(months[m]['y']), m) for m in ts.SPLITS])
    day = np.concatenate([months[m]['day'] for m in ts.SPLITS])
    airport = np.concatenate([months[m]['airport'] for m in ts.SPLITS])
    flat = {k: np.concatenate([v[m] for m in ts.SPLITS]) for k, v in se.items()}
    level = lambda e: dict({m: float(np.sqrt(e[month == m].mean())) for m in ts.SPLITS},
                           average=float(np.mean([np.sqrt(e[month == m].mean()) for m in ts.SPLITS])))
    out = dict(w_nn=W_NN, mixed=MIXED, seeds=SEEDS, nets=args.nets, level={}, change={}, by_seed={}, by_airport={}, cross_month={})
    for n in names:
        out['level'][n] = dict(blend=level(flat[n, 'blend']), gbm=level(flat[n, 'gbm']))
        out['by_seed'][n] = {s: float(np.mean(list(by_seed[n, s].values()))) for s in SEEDS}
    for n in args.arms:
        out['change'][n] = dict(blend=ts.paired_bootstrap(day, month, flat['control', 'blend'], flat[n, 'blend']),
                                gbm=ts.paired_bootstrap(day, month, flat['control', 'gbm'], flat[n, 'gbm']))
        out['change'][n]['seeds_better'] = int(sum(out['by_seed'][n][s] < out['by_seed']['control'][s] for s in SEEDS))
        out['change'][n]['passes'] = bool(out['change'][n]['blend']['decision'] == 'improvement'
                                          and out['change'][n]['seeds_better'] == len(SEEDS))
        out['by_airport'][n] = {f'{m} {a}': float(np.sqrt(flat[n, 'blend'][(month == m) & (airport == a)].mean())
                                                  - np.sqrt(flat['control', 'blend'][(month == m) & (airport == a)].mean()))
                                for m in ts.SPLITS for a in sorted(set(airport))}
    # Cross-month weights: network share (and arm share in the GBM part) chosen on the other month.
    for n in names:
        mixes = MIX_GRID if n in MIXED else [MIXED.get(n, 1.0)]
        grid = {}
        for m, d in months.items():
            for w in NN_GRID:
                for mix in mixes:
                    errs = []
                    for s in SEEDS:
                        g = d['lgb'][s] if n == 'control' else mix * d['arm'][n][s] + (1 - mix) * d['lgb'][s]
                        errs.append(((d['y'] - d['final'](w * nn(d, n, s) + (1 - w) * g)) ** 2).mean())
                    grid[m, float(w), float(mix)] = float(np.sqrt(np.mean(errs)))
        best = {m: min((k for k in grid if k[0] == m), key=grid.get)[1:] for m in ts.SPLITS}
        j, k = ts.SPLITS
        out['cross_month'][n] = dict(best_per_month={m: list(v) for m, v in best.items()},
                                     cv=float(np.mean([grid[(j, *best[k])], grid[(k, *best[j])]])))
    write_json(RUN / args.out, out)
    print('Jan/Jul CV blend', {n: round(v['blend']['average'], 4) for n, v in out['level'].items()})
    print('Jan/Jul CV GBM alone', {n: round(v['gbm']['average'], 4) for n, v in out['level'].items()})
    for n, c in out['change'].items():
        print(f"{n}: blend {c['blend']['change']:+.3f} {[round(x, 3) for x in c['blend']['ci95']]} "
              f"{c['blend']['decision']}, seeds better {c['seeds_better']}/3, passes {c['passes']}; GBM alone "
              f"{c['gbm']['change']:+.3f} {[round(x, 3) for x in c['gbm']['ci95']]}")
    print('by seed', {n: {s: round(v, 3) for s, v in d.items()} for n, d in out['by_seed'].items()})
    print('cross-month', {n: (v['best_per_month'], round(v['cv'], 4)) for n, v in out['cross_month'].items()})
    for n in args.arms:
        worst = sorted(out['by_airport'][n].items(), key=lambda kv: kv[1])
        print(n, 'best airports', [(k, round(v, 2)) for k, v in worst[:3]], 'worst', [(k, round(v, 2)) for k, v in worst[-3:]])


if __name__ == '__main__':
    main()
