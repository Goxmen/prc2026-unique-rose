"""Cap training labels for the tree models at 3 hours (Jan/Jul practice test).

A handful of absurd labels dominate what the tree models learn: in the July split's training data,
two labels over 10 hours outside Rome hold 4.4% of all label variation (January: one label, 2.3%).
The networks already clip labels at 3 h (`label_clip` 10800); the LightGBMs and CatBoost do not.
This runner refits v54's main LightGBM (trial LB1: `max_cat_threshold` 8, learning rate 0.02,
10,000 rounds) with every training label capped at CAP seconds, and scores it inside v54's practice
recipe (C5: pooled N09 networks, Rome R20, per-airport P04, CatBoost K07) with only the main LightGBM
replaced, seed by seed. Held-out labels are never capped.

Run: .venv/bin/python VALIDATION/clip_labels_jan_jul.py [fit|score|all] [--part lgb|airport] [--seeds 42 ...]
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import hpo_jan_jul as hpo
import tuning_seasonal as ts
from records import write_json

OUT = ts.RUNS / 'v3_20261006_clip_labels_jan_jul'
CAP = 10800.0
TRIALS = dict(lgb='LB1', airport='P04')


def log(msg):
    print(time.strftime('[%H:%M:%S] ') + msg, flush=True)


def fit(part, seeds):
    import modeling
    ideas = hpo.load_trees()
    spec = hpo.trial_spec(part, TRIALS[part])
    rounds = int(hpo.winner(part)['checkpoint'][1:])
    for seed in seeds:
        path = OUT / 'predictions' / f'{part}_cap_s{seed}.parquet'
        if path.exists():
            continue
        params = ts.lgb_params(dict(hpo.V44, **spec['overrides']), seed, spec['learning_rate'])
        frames, info = [], {}
        for month in ts.SPLITS:
            item = ideas.DATA['items'][month]
            cols = item['names'] + ideas.COMBO
            y = np.minimum(np.asarray(item['y'], float), CAP)
            started = time.monotonic()
            pred = np.full(len(item['Vn']), np.nan)
            groups = ([(item['airport'] == ap, item['v_airport'] == ap) for ap in np.unique(item['airport'])]
                      if part == 'airport' else [(slice(None), np.ones(len(item['Vn']), bool))])
            for k, v in groups:
                booster = ideas.train(item['X'].loc[k, cols], y[k], params, rounds)
                if v.any():
                    pred[v] = modeling.predict(booster, item['Vn'].loc[v, cols], threads=ts.THREADS)
            info[month] = dict(minutes=(time.monotonic() - started) / 60, capped_rows=int((np.asarray(item['y']) > CAP).sum()))
            log(f'{part} capped s{seed} {month}: {info[month]}')
            frames.append(pd.DataFrame({'mvt_id': item['ids'], 'month': month, 'pred': pred.astype(np.float32)}))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
        write_json(OUT / f'{part}_cap_s{seed}.json', dict(part=part, trial=TRIALS[part], cap=CAP, rounds=rounds, info=info))


def score(part, seeds):
    sc = hpo.Scorer()
    wins = {p: hpo.winner(p) for p in ['rome', 'nn', 'catboost', 'lgb', 'airport']}
    use = {p: w for p, w in wins.items() if w and w['confirmed']}
    f = sc.frame('rome', use['rome']['trial'], hpo.SEED)
    rome = {m: f[m][use['rome']['checkpoint']].to_numpy(float) for m in ts.SPLITS}
    t = use['nn']['trial']
    frames = [sc.frame('nn', t, hpo.SEED), sc.frame('nn', t + '_k34', hpo.SEED)] + [sc.frame('nn', t, s) for s in hpo.SEEDS[1:]]
    pool = {m: np.concatenate([fr[m].to_numpy(float) for fr in frames], axis=1).mean(1) for m in ts.SPLITS}
    reps = {p: {s: hpo.member_preds(sc, p, use[p], s) for s in hpo.SEEDS} for p in ['catboost', 'lgb', 'airport']}
    res, se_c, se_n, days, months = {}, [], [], [], []
    for s in seeds:
        cap = pd.read_parquet(OUT / 'predictions' / f'{part}_cap_s{s}.parquet')
        r = {}
        for m in ts.SPLITS:
            c = cap[cap.month == m]
            assert np.array_equal(c.mvt_id.to_numpy(), sc.d[m]['ids'])
            kw = {hpo.MEMBER_OF[p]: reps[p][s][m] for p in reps}
            e0 = sc.se(m, s, nn=pool[m], p=rome[m], **kw)
            kw[hpo.MEMBER_OF[part]] = c.pred.to_numpy(float)
            e1 = sc.se(m, s, nn=pool[m], p=rome[m], **kw)
            r[m] = (float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean())))
            se_c.append(e0); se_n.append(e1); days.append(sc.d[m]['day']); months.append(np.full(len(e0), m))
        res[s] = dict(by_month=r, control=float(np.mean([v[0] for v in r.values()])), capped=float(np.mean([v[1] for v in r.values()])))
        log(f"seed {s}: C5 {res[s]['control']:.4f} -> capped {part} {res[s]['capped']:.4f}  {r}")
    ch = ts.paired_bootstrap(np.concatenate(days), np.concatenate(months), np.concatenate(se_c), np.concatenate(se_n))
    log(f'change {ch}')
    write_json(OUT / f'result_{part}.json', dict(seeds=res, change=ch))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['fit', 'score', 'all'])
    ap.add_argument('--part', default='lgb', choices=list(TRIALS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ('fit', 'all'):
        fit(a.part, a.seeds)
    if a.stage in ('score', 'all'):
        score(a.part, a.seeds)


if __name__ == '__main__':
    main()
