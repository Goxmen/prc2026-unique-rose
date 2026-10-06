"""Score composed candidates (network part + gradient-boosting part) against v48's recipe.

Saved predictions only. A candidate names its network part and its GBM members:
  network  ple5        that seed's five `ple` networks (v48)
           ple15       all fifteen `ple` networks of the month, pooled over the three seed groups
           <tag>15     all fifteen networks `nn_<tag>_*` of the ideas run, pooled
  GBM      v46         v46's LightGBM (v48's); any other name is an arm of the ideas run
           (`ideas_jan_jul.py` or `catboost_jan_jul.py`), at the same seed
Base = 0.6 x network + 0.4 x the equal-weight mean of the GBM members, then the Rome mix,
corrections and rules. Squared errors averaged over seeds 42/142/242 within each month; paired day
bootstrap against v48's recipe (ple5 + v46). The candidates in CANDIDATES were fixed before any of
them was scored; the cross-month network share is a diagnostic.

Run: .venv/bin/python VALIDATION/tools/score_candidates_jan_jul.py [--only C1 C2]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import score_ideas_jan_jul as sc
import tuning_seasonal as ts
from records import write_json

CANDIDATES = {
    'v48': ('ple5', ['v46']),
    'C1_ple15': ('ple15', ['v46']),
    'C2_features': ('ple15', ['combo', 'combo_airport']),
    'C3_features_catboost': ('ple15', ['combo', 'combo_airport', 'catboost']),
    'C4_all_nn_features': ('combo15', ['combo', 'combo_airport', 'catboost']),
}
NN_GRID = np.round(np.arange(0.4, 0.851, 0.05), 2)


def networks(month, ids, spec, d):
    if spec == 'ple5':
        return d['nn']
    if spec == 'ple15':
        pooled = np.mean([d['nn'][s] for s in sc.SEEDS], axis=0)
        return {s: pooled for s in sc.SEEDS}
    tag = spec[:-2]
    nets = [pd.read_parquet(sc.RUN / 'networks' / f'nn_{tag}_{month}_s{s}_k{k}.parquet') for s in sc.SEEDS for k in range(5)]
    if any(not np.array_equal(n.mvt_id.to_numpy(), ids) for n in nets):
        raise ValueError(f'{month}: {tag} network rows differ')
    pooled = np.mean([n.base.to_numpy(float) for n in nets], axis=0)
    return {s: pooled for s in sc.SEEDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--only', nargs='+')
    args = ap.parse_args()
    names = ['v48'] + [c for c in CANDIDATES if c != 'v48' and (not args.only or c in args.only)]
    arms = sorted({g for c in names for g in CANDIDATES[c][1] if g != 'v46'})
    months = {}
    for m in ts.SPLITS:
        d = sc.load_month(m, arms, sc.NETS)
        ids = pd.read_pickle(ts.split_dir(sc.SPLITS_FROM, m) / 'heldout.pkl')['raw'].mvt_id.to_numpy()
        d['nets'] = {c: networks(m, ids, CANDIDATES[c][0], d) for c in names}
        months[m] = d
    se, grid, per_seed, alone = {}, {}, {}, {}
    for c in names:
        spec, members = CANDIDATES[c]
        for m, d in months.items():
            g = {s: np.mean([d['lgb'][s] if a == 'v46' else d['arm'][a][s] for a in members], axis=0) for s in sc.SEEDS}
            errs = {s: (d['y'] - d['final'](sc.W_NN * d['nets'][c][s] + (1 - sc.W_NN) * g[s])) ** 2 for s in sc.SEEDS}
            se.setdefault(c, {})[m] = np.mean(list(errs.values()), axis=0)
            for s in sc.SEEDS:
                per_seed.setdefault((c, s), []).append(float(np.sqrt(errs[s].mean())))
            alone.setdefault(c, {})[m] = dict(
                network=float(np.sqrt(np.mean([((d['y'] - d['final'](d['nets'][c][s])) ** 2).mean() for s in sc.SEEDS]))),
                gbm=float(np.sqrt(np.mean([((d['y'] - d['final'](g[s])) ** 2).mean() for s in sc.SEEDS]))))
            for w in NN_GRID:
                grid[c, m, float(w)] = float(np.sqrt(np.mean(
                    [((d['y'] - d['final'](w * d['nets'][c][s] + (1 - w) * g[s])) ** 2).mean() for s in sc.SEEDS])))
    month = np.concatenate([np.full(len(months[m]['y']), m) for m in ts.SPLITS])
    day = np.concatenate([months[m]['day'] for m in ts.SPLITS])
    flat = {c: np.concatenate([v[m] for m in ts.SPLITS]) for c, v in se.items()}
    level = lambda e: dict({m: float(np.sqrt(e[month == m].mean())) for m in ts.SPLITS},
                           average=float(np.mean([np.sqrt(e[month == m].mean()) for m in ts.SPLITS])))
    out = {}
    j, k = ts.SPLITS
    for c in names:
        best = {m: min(NN_GRID, key=lambda w: grid[c, m, float(w)]) for m in ts.SPLITS}
        out[c] = dict(network=CANDIDATES[c][0], gbm=CANDIDATES[c][1], level=level(flat[c]),
                      change=None if c == 'v48' else ts.paired_bootstrap(day, month, flat['v48'], flat[c]),
                      by_seed={s: float(np.mean(per_seed[c, s])) for s in sc.SEEDS},
                      network_alone=float(np.mean([alone[c][m]['network'] for m in ts.SPLITS])),
                      gbm_alone=float(np.mean([alone[c][m]['gbm'] for m in ts.SPLITS])),
                      cross_month=dict(best={m: float(w) for m, w in best.items()},
                                       cv=float(np.mean([grid[c, j, float(best[k])], grid[c, k, float(best[j])]]))))
    previous = sc.RUN / 'candidates.json'
    if previous.exists():
        import json
        old = json.loads(previous.read_text())
        old.update(out)
        out = old
    write_json(previous, out)
    for c in names:
        v = out[c]
        ch = v['change']
        print(f"{c:22s} {v['level']['average']:.4f} ({v['level'][j]:.2f}/{v['level'][k]:.2f})",
              '' if ch is None else f"{ch['change']:+.3f} {[round(x, 3) for x in ch['ci95']]} {ch['decision']}",
              f"cross-month share {v['cross_month']['best']} -> {v['cross_month']['cv']:.4f}",
              f"| by seed {[round(x, 3) for x in v['by_seed'].values()]} | network alone {v['network_alone']:.2f}, "
              f"GBM alone {v['gbm_alone']:.2f}")


if __name__ == '__main__':
    main()
