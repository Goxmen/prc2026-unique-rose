"""Extra idea: one network/LightGBM blend weight per airport instead of v48's single 0.6.

Saved predictions only (v48's recipe: five `ple` networks + v46 LightGBM, seeds 42/142/242). Each
airport's weight is chosen on the other month (grid 0.3-0.9 by 0.05, squared error averaged over
seeds) and applied to this month, so no month is scored with a weight chosen on its own labels.
Compared with v48's single 0.6 by the paired day bootstrap.

Run: .venv/bin/python VALIDATION/tools/blend_by_airport.py
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import score_ideas_jan_jul as sc
import tuning_seasonal as ts
from records import write_json

GRID = np.round(np.arange(0.3, 0.901, 0.05), 2)


def main():
    months = {m: sc.load_month(m, [], sc.NETS) for m in ts.SPLITS}
    se = {}
    for m, d in months.items():
        for w in GRID:
            se[m, float(w)] = np.mean([(d['y'] - d['final'](w * d['nn'][s] + (1 - w) * d['lgb'][s])) ** 2
                                       for s in sc.SEEDS], axis=0)
    chosen, per = {}, {}
    for m, d in months.items():
        other = [o for o in ts.SPLITS if o != m][0]
        o = months[other]
        out = np.zeros(len(d['y']))
        for ap in np.unique(d['airport']):
            ko = o['airport'] == ap
            w = min(GRID, key=lambda w: se[other, float(w)][ko].mean()) if ko.any() else 0.6
            chosen.setdefault(m, {})[ap] = float(w)
            k = d['airport'] == ap
            out[k] = se[m, float(w)][k]
        per[m] = out
    month = np.concatenate([np.full(len(months[m]['y']), m) for m in ts.SPLITS])
    day = np.concatenate([months[m]['day'] for m in ts.SPLITS])
    base = np.concatenate([se[m, 0.6] for m in ts.SPLITS])
    new = np.concatenate([per[m] for m in ts.SPLITS])
    level = lambda e: float(np.mean([np.sqrt(e[month == m].mean()) for m in ts.SPLITS]))
    out = dict(weights_applied=chosen, v48=level(base), per_airport=level(new),
               change=ts.paired_bootstrap(day, month, base, new))
    write_json(sc.RUN / 'blend_by_airport.json', out)
    print('v48', round(out['v48'], 4), 'per-airport', round(out['per_airport'], 4), 'change',
          round(out['change']['change'], 4), out['change']['ci95'], out['change']['decision'])
    print(chosen)


if __name__ == '__main__':
    main()
