"""Rome classifier v2 on the Jan/Jul practice test: cost-weighted training and a flight-number copy input.

Arms, fixed before any is scored. All use R20's settings (the v53/v54 Rome classifier), 2,400 rounds,
probabilities saved every 100 rounds, seeds 42/142/242:
  C   control: R20 as is. Seed 42 must reproduce hpo_jan_jul's R20_s42 probabilities.
  W   each LIRF training flight weighted by what a wrong copy call costs:
      max(|takeoff - schedule - typical taxi|, 300 s)^2, normalised to mean 1. Typical taxi = median label of
      the training month's non-copy LIRF flights.
  F   + fn_copy_rate, fn_copy_n: the share of the flight number's departures (FLIGHT_mvt) whose label equals
      takeoff minus schedule, from the training months. A training row only sees the other training months;
      the rate is pulled toward the overall share with weight FN_PRIOR.
  WF  both.
Scoring is hpo_jan_jul's Rome scoring: C4's parts at the same seed, only the probability replaced. Each arm's
checkpoint is the best of its seed-averaged curve (the same rule for every arm); paired day bootstrap of the
seed-averaged squared errors against C at its own chosen checkpoint.

Training rows carry no flight number in `after_drop.pkl`, so the LIRF training rows are rebuilt from the
movement files in the split's own order (overnight_run.load_inputs -> seasonal.month_subset -> label filter
-> rule rows dropped) and checked against the saved labels and schedule gaps.

Run: .venv/bin/python VALIDATION/rome_cost_jan_jul.py [fit|score|all]
"""
import argparse
import gc
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import hpo_jan_jul as hpo
import overnight_run
import rome_hybrid
import rome_signals as signals
import seasonal
import tuning_seasonal as ts
from records import write_json

OUT = ts.RUNS / 'v3_20261006_rome_cost_jan_jul'
R20 = dict(num_leaves=127, feature_fraction=0.6, min_data_in_leaf=50, learning_rate=0.02)
ROUNDS, STEP = 2400, 100
SEEDS = [42, 142, 242]
ARMS = ['C', 'W', 'F', 'WF']
FN_PRIOR = 10.0
WEIGHT_FLOOR_SEC = 300.0
FN = ['fn_copy_rate', 'fn_copy_n']
THREADS = 16


def log(msg):
    print(time.strftime('[%H:%M:%S] ') + msg, flush=True)


def lirf_training_rows(dep, month):
    """Flight numbers and months of the split's LIRF training rows, in after_drop order."""
    sp = ts.split_dir(hpo.SPLITS_FROM, month)
    meta = ts.read(sp / 'meta.json')
    td = seasonal.month_subset(dep, meta['train_months'])
    kept = td.loc[td.taxi_sec.between(1, 200000)]
    rule = pd.read_pickle(sp / 'before_drop.pkl')['rule']
    if len(rule) != len(kept):
        raise ValueError(f'{month}: rule mask {len(rule)} vs rows {len(kept)}')
    kept = kept.loc[~np.asarray(rule)]
    return kept


def copy_rate(fn_train, month_train, s_train, fn_query, month_query=None):
    """Smoothed copy share per flight number; with month_query, the query row's own month is left out."""
    prior = float(np.mean(s_train))
    t = pd.DataFrame({'fn': fn_train, 'm': month_train, 's': np.asarray(s_train, float)})
    tot = t.groupby('fn').s.agg(['sum', 'count'])
    q = pd.DataFrame({'fn': fn_query})
    s_sum = q.fn.map(tot['sum']).fillna(0).to_numpy()
    n = q.fn.map(tot['count']).fillna(0).to_numpy()
    if month_query is not None:
        per = t.groupby(['fn', 'm']).s.agg(['sum', 'count'])
        key = pd.MultiIndex.from_arrays([q.fn, np.asarray(month_query)])
        own = per.reindex(key)
        s_sum = s_sum - own['sum'].fillna(0).to_numpy()
        n = n - own['count'].fillna(0).to_numpy()
    return (s_sum + FN_PRIOR * prior) / (n + FN_PRIOR), n


def build(month, dep, scorer):
    tr = pd.read_pickle(ts.split_dir(hpo.SPLITS_FROM, month) / 'after_drop.pkl')
    lirf = np.asarray(tr['airport'] == 'LIRF')
    XL, yL = tr['X'].loc[lirf], np.asarray(tr['y'])[lirf]
    rawL = ts.minimal_raw(np.full(len(XL), 'LIRF'), XL.index)
    del tr
    gc.collect()
    kept = lirf_training_rows(dep, month)
    if len(kept) != len(lirf):
        raise ValueError(f'{month}: rebuilt {len(kept)} rows vs saved {len(lirf)}')
    k = kept.loc[lirf]
    if not np.array_equal(k.ADEP_mvt.to_numpy(), np.full(len(k), 'LIRF')):
        raise ValueError(f'{month}: airport order differs')
    if not np.allclose(k.taxi_sec.to_numpy(float), yL):
        raise ValueError(f'{month}: labels differ')
    gap_raw = (pd.to_datetime(k.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(k.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()
    gap = XL[rome_hybrid.GAP].to_numpy(float)
    ok = np.isfinite(gap) & np.isfinite(gap_raw)
    if not np.allclose(gap[ok], gap_raw[ok], atol=1.0):
        raise ValueError(f'{month}: schedule gaps differ')
    s = rome_hybrid.substituted(XL, yL, rawL)
    fn_t = k.FLIGHT_mvt.fillna('').astype(str).to_numpy()
    m_t = pd.to_datetime(k.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').to_numpy()
    held = scorer.d[month]['held']
    V, raw = held['V'], held['raw']
    lh = (raw.ADEP_mvt.to_numpy() == 'LIRF') & np.isfinite(V[rome_hybrid.GAP].to_numpy(float))
    fn_v = raw.loc[lh, 'FLIGHT_mvt'].fillna('').astype(str).to_numpy()
    rt, nt = copy_rate(fn_t, m_t, s, fn_t, m_t)
    rv, nv = copy_rate(fn_t, m_t, s, fn_v)
    X = signals.classifier_matrix(XL, rawL, ts.KIND, ts.CLASSIFIER)
    Vm = signals.classifier_matrix(V.loc[lh], raw.loc[lh], ts.KIND, ts.CLASSIFIER)
    typical = float(np.median(yL[~s]))
    w = np.maximum(np.abs(np.nan_to_num(gap, nan=typical) - typical), WEIGHT_FLOOR_SEC) ** 2
    w = w / w.mean()
    info = dict(train_rows=int(len(XL)), copies=int(s.sum()), typical_sec=typical, weight_max=float(w.max()),
                weight_share_top1pct=float(np.sort(w)[-max(1, len(w) // 100):].sum() / w.sum()),
                fn_rate_train_mean=float(rt.mean()), heldout_rows=int(lh.sum()), heldout_fn_seen=float((nv > 0).mean()))
    log(f'{month}: {info}')
    return dict(X=X, V=Vm, s=s.astype(float), w=w, lh=lh, ids=scorer.d[month]['ids'],
                fn_X=pd.DataFrame({'fn_copy_rate': rt, 'fn_copy_n': nt}, index=X.index),
                fn_V=pd.DataFrame({'fn_copy_rate': rv, 'fn_copy_n': nv}, index=Vm.index), info=info)


def train(params, X, y, w, rounds):
    cat = signals._categorical(X.columns)
    return lgb.train(params, lgb.Dataset(X, y, weight=w, categorical_feature=cat), num_boost_round=rounds)


def fit(scorer):
    dep, _ = overnight_run.load_inputs(1.)
    data = {m: build(m, dep, scorer) for m in ts.SPLITS}
    del dep
    gc.collect()
    write_json(OUT / 'build.json', {m: d['info'] for m, d in data.items()})
    for arm in ARMS:
        for seed in SEEDS:
            path = OUT / 'predictions' / f'{arm}_s{seed}.parquet'
            if path.exists():
                continue
            started = time.monotonic()
            params = dict(rome_hybrid.CLASSIFIER_PARAMS, **R20, seed=seed, bagging_seed=seed + 1,
                          feature_fraction_seed=seed + 2, num_threads=THREADS)
            frames = []
            for m in ts.SPLITS:
                d = data[m]
                X, V = (pd.concat([d['X'], d['fn_X']], axis=1), pd.concat([d['V'], d['fn_V']], axis=1)) if 'F' in arm else (d['X'], d['V'])
                b = train(params, X, d['s'], d['w'] if 'W' in arm else None, ROUNDS)
                cols = {}
                for c in range(STEP, ROUNDS + 1, STEP):
                    p = np.zeros(len(d['lh']))
                    p[d['lh']] = b.predict(V, num_iteration=c, num_threads=THREADS)
                    cols[f'c{c}'] = p
                frames.append(pd.DataFrame({'mvt_id': d['ids'], 'month': m, **cols}))
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            log(f'{arm} seed {seed}: {(time.monotonic() - started) / 60:.1f} min')
            if arm == 'C' and seed == 42:
                ref = pd.read_parquet(hpo.OUT / 'predictions' / 'rome' / 'R20_s42.parquet')
                new = pd.read_parquet(path)
                same = bool(np.allclose(ref.c1700.to_numpy(), new.c1700.to_numpy(), atol=1e-9))
                log(f'C seed 42 reproduces R20_s42 at c1700: {same}')
                write_json(OUT / 'reproduction.json', dict(R20_s42_c1700=same))


def score(scorer):
    curves, se = {}, {}
    for arm in ARMS:
        frames = {s: pd.read_parquet(OUT / 'predictions' / f'{arm}_s{s}.parquet') for s in SEEDS}
        cols = [c for c in frames[SEEDS[0]].columns if c.startswith('c')]
        per = {}
        for c in cols:
            for s in SEEDS:
                for m in ts.SPLITS:
                    f = frames[s][frames[s].month == m]
                    per[c, s, m] = scorer.se(m, s, p=f[c].to_numpy(float))
        curve = {c: float(np.mean([np.mean([np.sqrt(per[c, s, m].mean()) for m in ts.SPLITS]) for s in SEEDS])) for c in cols}
        best = min(curve, key=curve.get)
        curves[arm] = dict(curve=curve, best=best, cv=curve[best], c1700=curve['c1700'],
                           by_seed={s: float(np.mean([np.sqrt(per[best, s, m].mean()) for m in ts.SPLITS])) for s in SEEDS},
                           by_month={m: float(np.mean([np.sqrt(per[best, s, m].mean()) for s in SEEDS])) for m in ts.SPLITS})
        se[arm] = np.concatenate([np.mean([per[best, s, m] for s in SEEDS], axis=0) for m in ts.SPLITS])
        log(f"{arm}: {curves[arm]['cv']:.4f} at {best} (c1700 {curves[arm]['c1700']:.4f}) by seed {[round(v, 3) for v in curves[arm]['by_seed'].values()]}")
    day = np.concatenate([scorer.d[m]['day'] for m in ts.SPLITS])
    month = np.concatenate([np.full(len(scorer.d[m]['y']), m) for m in ts.SPLITS])
    for arm in ARMS[1:]:
        ch = ts.paired_bootstrap(day, month, se['C'], se[arm])
        curves[arm]['change_vs_C'] = ch
        seeds_better = sum(curves[arm]['by_seed'][s] < curves['C']['by_seed'][s] for s in SEEDS)
        curves[arm]['seeds_better'] = int(seeds_better)
        log(f"{arm} vs C: {ch['change']:+.3f} {[round(x, 3) for x in ch['ci95']]} {ch['decision']}, seeds better {seeds_better}/3")
    write_json(OUT / 'result.json', curves)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['fit', 'score', 'all'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scorer = hpo.Scorer()
    if args.stage in ('fit', 'all'):
        fit(scorer)
    if args.stage in ('score', 'all'):
        score(scorer)


if __name__ == '__main__':
    main()
