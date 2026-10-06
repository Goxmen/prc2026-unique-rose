"""Score `nn_improve_jan_jul.py` (three changes to v47's network) from saved predictions only.

Every arm is blended 0.3 network / 0.7 LightGBM, v47's fixed weight, at the three LightGBM
seeds; squared errors are averaged over seeds within each fit and the interval resamples whole
days within each fit, as `score_nn_jan_jul.py` does. The control is v47 itself (its saved
single network per seed).

Decision, fixed before any arm was scored: an arm passes if its change against v47 has a 95%
interval below zero and every seed improves. ple or tabm is chosen over avg5 only if it also
improves on avg5 with an interval below zero; otherwise the simpler avg5 is the candidate.
Between ple and tabm (about eight times slower per epoch), tabm is chosen only if it improves on
ple with an interval below zero. (The tabm-versus-ple rule was added on 2026-10-03 after avg5 and
ple were scored and before tabm finished.)
The weight grid, single-network readout and per-airport changes are diagnostics. So is the
cross-month weight: each month is blended at the weight that was best on the other month (seed
average), and compared with the same arm at 0.3.
"""
import argparse
import json
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
RUN = ts.RUNS / 'v3_20261003_nn_improve_jan_jul'
SEEDS, WEIGHT = [42, 142, 242], 0.3
NETWORKS = dict(avg5=5, ple=5, tabm=1)
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def network(arm, month, seed, k, ids):
    f = pd.read_parquet(RUN / 'networks' / f'{arm}_{month}_s{seed}_k{k}.parquet')
    if not np.array_equal(f.mvt_id.to_numpy(), ids):
        raise ValueError(f'{arm} {month} seed {seed} network {k}: rows differ from the held-out rows')
    return f.base.to_numpy(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--arms', nargs='+', default=list(NETWORKS))
    args = ap.parse_args()
    v47 = pd.read_parquet(V47)
    parts, grids, kept = [], {}, {}
    for month in ts.SPLITS:
        d = ts.split_dir(SPLITS_FROM, month)
        held, meta = pd.read_pickle(d / 'heldout.pkl'), ts.read(d / 'meta.json')
        raw = held['raw']
        ids = raw.mvt_id.to_numpy()
        saved = v47[v47.month == month].set_index('mvt_id').reindex(ids)
        if saved.isna().any().any():
            raise ValueError(f'{month}: v47 rows differ from the held-out rows')
        final = lambda base, held=held, raw=raw, meta=meta: ts.final_prediction(   # bound now: reused after the loop
            base, held['p'], held['V'], raw, meta['parameters'], meta['corrections'])
        y = raw.taxi_sec.to_numpy(float)
        frame = pd.DataFrame({'mvt_id': ids, 'month': month, 'y': y, 'airport': raw.ADEP_mvt.astype(str).to_numpy(),
                              'day': pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy()})
        for s in SEEDS:
            lgb = saved[f'lgb_base_{s}'].to_numpy(float)
            nets = {'v47': saved[f'nn_base_{s}'].to_numpy(float)}
            for arm in args.arms:
                base = [network(arm, month, s, k, ids) for k in range(NETWORKS[arm])]
                nets[arm] = np.mean(base, axis=0)
                if arm == 'ple':
                    nets['ple1'] = base[0]                       # one network: the input/epoch change alone
            frame[f'lgb_{s}'] = final(lgb)
            kept[(month, s)] = (final, lgb, nets)
            for name, nn_base in nets.items():
                frame[f'{name}_{s}'] = final(WEIGHT * nn_base + (1 - WEIGHT) * lgb)
                frame[f'{name}_alone_{s}'] = final(nn_base)
                grids.setdefault(name, {}).setdefault(month, {})[s] = {
                    float(w): float(((y - final(w * nn_base + (1 - w) * lgb)) ** 2).mean()) for w in GRID}
        parts.append(frame)
    f = pd.concat(parts, ignore_index=True)
    month, day = f.month.to_numpy(), f.day.to_numpy()
    names = ['v47'] + list(args.arms) + (['ple1'] if 'ple' in args.arms else [])
    se = {a: np.mean([(f.y - f[f'{a}_{s}']) ** 2 for s in SEEDS], axis=0) for a in ['lgb'] + names}
    se_alone = {a: np.mean([(f.y - f[f'{a}_alone_{s}']) ** 2 for s in SEEDS], axis=0) for a in names}
    cv = lambda e: dict({m: float(np.sqrt(e[month == m].mean())) for m in ts.SPLITS},
                        average=float(np.mean([np.sqrt(e[month == m].mean()) for m in ts.SPLITS])))
    out = dict(weight=WEIGHT, seeds=SEEDS, networks=NETWORKS, level={a: cv(e) for a, e in se.items()},
               network_alone={a: cv(e) for a, e in se_alone.items()}, change={}, by_seed={}, weight_grid={},
               by_airport={})
    pairs = [('v47', a) for a in names if a != 'v47'] + [('avg5', a) for a in ('ple', 'tabm') if a in names
                                                          and 'avg5' in names]
    pairs += [('ple', 'tabm')] if {'ple', 'tabm'} <= set(names) else []
    for a, b in pairs:
        out['change'][f'{b}_vs_{a}'] = ts.paired_bootstrap(day, month, se[a], se[b])
    for s in SEEDS:
        out['by_seed'][s] = {a: float(np.mean([np.sqrt(((f.y - f[f'{a}_{s}']) ** 2)[month == m].mean())
                                               for m in ts.SPLITS])) for a in ['lgb'] + names}
    for a in names:
        mse = {m: {w: np.mean([grids[a][m][s][w] for s in SEEDS]) for w in map(float, GRID)} for m in ts.SPLITS}
        best = {m: min(mse[m], key=mse[m].get) for m in ts.SPLITS}
        j, k = ts.SPLITS
        out['weight_grid'][a] = dict(
            average={w: float(np.mean([np.sqrt(mse[m][w]) for m in ts.SPLITS])) for w in map(float, GRID)},
            best_per_month=best,
            cross_month=float(np.mean([np.sqrt(mse[j][best[k]]), np.sqrt(mse[k][best[j]])])))
        # Cross-month weight: blend each month at the other month's best weight.
        cross = np.zeros(len(f))
        for s in SEEDS:
            pred = np.concatenate([kept[(m, s)][0](best[k if m == j else j] * kept[(m, s)][2][a]
                                                   + (1 - best[k if m == j else j]) * kept[(m, s)][1])
                                   for m in ts.SPLITS])
            cross += (f.y.to_numpy() - pred) ** 2 / len(SEEDS)
        out['weight_grid'][a]['cross_month_vs_0.3'] = ts.paired_bootstrap(day, month, se[a], cross)
    for (m, ap_), g in f.groupby(['month', 'airport']):
        i = g.index.to_numpy()
        out['by_airport'].setdefault(m, {})[ap_] = {a: float(np.sqrt(se[a][i].mean())) for a in ['v47'] + names[1:]}
    result = json.loads((RUN / 'result.json').read_text())
    out['stop_epochs'] = {m: {a: {s: v.get('stop_epoch') for s, v in result['months'][m][a].items() if s.isdigit()}
                              for a in args.arms if a in result['months'][m]} for m in ts.SPLITS}
    write_json(RUN / 'score.json', out)
    print('Jan/Jul CV (blend at 0.3)', {a: round(v['average'], 4) for a, v in out['level'].items()})
    print('per month', {a: {m: round(v[m], 2) for m in ts.SPLITS} for a, v in out['level'].items()})
    print('network alone', {a: round(v['average'], 2) for a, v in out['network_alone'].items()})
    for k, v in out['change'].items():
        print(k, round(v['change'], 4), [round(x, 3) for x in v['ci95']], v['decision'])
    print('by seed', {s: {a: round(x, 3) for a, x in v.items()} for s, v in out['by_seed'].items()})
    for a, g in out['weight_grid'].items():
        c = g['cross_month_vs_0.3']
        print('weights', a, 'best per month', g['best_per_month'], 'cross-month CV', round(g['cross_month'], 4),
              'vs 0.3', round(c['change'], 3), [round(x, 3) for x in c['ci95']],
              {w: round(x, 3) for w, x in g['average'].items() if w in (0.3, 0.4, 0.5, 0.6)})
    print('stop epochs', out['stop_epochs'])
    for m in ts.SPLITS:
        print(m, {ap_: {a: round(v[a] - v['v47'], 2) for a in names[1:]} for ap_, v in out['by_airport'][m].items()})


if __name__ == '__main__':
    main()
