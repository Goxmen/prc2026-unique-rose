"""Idea 6: a second-stage model trained on held-out errors, scored across months (saved predictions only).

v48's final prediction (0.6 x five `ple` networks + 0.4 x v46 LightGBM, then the Rome mix,
corrections and rules) leaves a residual y - final on each held-out flight. A small LightGBM learns
that residual on one month's held-out flights and corrects the other month: January's corrector
is trained on July and vice versa, so no month is corrected by a model that saw its labels. For a
submission it would be trained on both 2025 months and applied to 2026.

Frozen recipe (fixed before any run): inputs are the two model outputs and their blend, the Rome
probability p, airport, hour, whether a flight ID exists, and eight clock / traffic / ADS-B inputs
(STACK_INPUTS); target y - final clipped to +-3,600 s; LightGBM 15 leaves, learning rate 0.03,
400 rounds, min_data_in_leaf 1,000, lambda_l2 100, feature_fraction 0.8; the correction is added
in full and the result floored at 0. Same seeds as the base models (the corrector of seed s sees
only seed s's predictions). Paired day bootstrap against v48's recipe.

Run: .venv/bin/python VALIDATION/tools/stack_jan_jul.py
"""
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import score_ideas_jan_jul as sc
import tuning_seasonal as ts
from records import write_json

STACK_INPUTS = ['mvt_minus_aobt3_sec', 'mvt_minus_sched_sec', 'mvt_minus_eobt1_sec', 'clock_available_count',
                'q_rwy_takeoffs_prev30', 'dis_arr_60m_mean_sec', 'ab_seg_sec', 'ab_stand_exit_sec']
PARAMS = dict(objective='regression', num_leaves=15, learning_rate=0.03, min_data_in_leaf=1000, lambda_l2=100.0,
              feature_fraction=0.8, bagging_fraction=1.0, verbose=-1, num_threads=4)
ROUNDS, CLIP = 400, 3600


def main():
    table = adsblol.load_all()
    months = {m: sc.load_month(m, [], sc.NETS) for m in ts.SPLITS}
    frames = {}
    for m in ts.SPLITS:
        held = pd.read_pickle(ts.split_dir(sc.SPLITS_FROM, m) / 'heldout.pkl')
        V, raw = held['V'], held['raw']
        f = pd.DataFrame({c: V[c].to_numpy(float) for c in STACK_INPUTS if c in V.columns})
        ab = table.reindex(raw.mvt_id.to_numpy())
        for c in STACK_INPUTS:
            if c not in f:
                f[c] = ab[c].to_numpy(float)
        f['airport'] = pd.Categorical(raw.ADEP_mvt.astype(str).to_numpy())
        f['hour'] = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.hour.to_numpy()
        f['has_id'] = raw.FLIGHT_ID_mvt.notna().to_numpy().astype(float)
        f['p'] = np.asarray(held['p'], float)
        frames[m] = f
    cats = sorted(set().union(*[set(f.airport.cat.categories) for f in frames.values()]))
    for f in frames.values():
        f['airport'] = pd.Categorical(f.airport.astype(str), categories=cats)
    se = {'v48': {}, 'stacked': {}}
    by_seed = {}
    for m in ts.SPLITS:
        other = [o for o in ts.SPLITS if o != m][0]
        d, o = months[m], months[other]
        per = {'v48': [], 'stacked': []}
        for s in sc.SEEDS:
            def inputs(mm, ff):
                base = sc.W_NN * mm['nn'][s] + (1 - sc.W_NN) * mm['lgb'][s]
                final = mm['final'](base)
                g = ff.copy()
                g['nn'], g['lgb'], g['blend'], g['final'] = mm['nn'][s], mm['lgb'][s], base, final
                return g, final
            Xo, fo = inputs(o, frames[other])
            Xm, fm = inputs(d, frames[m])
            booster = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(Xo, np.clip(o['y'] - fo, -CLIP, CLIP),
                                                                  categorical_feature=['airport']), ROUNDS)
            corrected = np.maximum(fm + booster.predict(Xm), 0)
            per['v48'].append((d['y'] - fm) ** 2)
            per['stacked'].append((d['y'] - corrected) ** 2)
            by_seed.setdefault(s, {})[m] = dict(v48=float(np.sqrt(per['v48'][-1].mean())),
                                                stacked=float(np.sqrt(per['stacked'][-1].mean())))
        for k in per:
            se[k][m] = np.mean(per[k], axis=0)
    month = np.concatenate([np.full(len(months[m]['y']), m) for m in ts.SPLITS])
    day = np.concatenate([months[m]['day'] for m in ts.SPLITS])
    flat = {k: np.concatenate([v[m] for m in ts.SPLITS]) for k, v in se.items()}
    level = {k: dict({m: float(np.sqrt(v[month == m].mean())) for m in ts.SPLITS},
                     average=float(np.mean([np.sqrt(v[month == m].mean()) for m in ts.SPLITS]))) for k, v in flat.items()}
    out = dict(recipe=dict(inputs=STACK_INPUTS, params=PARAMS, rounds=ROUNDS, clip=CLIP), level=level,
               change=ts.paired_bootstrap(day, month, flat['v48'], flat['stacked']),
               by_seed={s: {k: float(np.mean([v[m][k] for m in ts.SPLITS])) for k in ('v48', 'stacked')}
                        for s, v in by_seed.items()})
    write_json(sc.RUN / 'stack.json', out)
    print('levels', {k: {m: round(x, 3) for m, x in v.items()} for k, v in level.items()})
    print('change', round(out['change']['change'], 4), out['change']['ci95'], out['change']['decision'])
    print('by seed', out['by_seed'])


if __name__ == '__main__':
    main()
