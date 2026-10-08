"""Train only on the training flights nearest to the test flights (Jan/Jul practice test).

The user's idea (2026-10-07): every flight to be predicted picks its K nearest training flights at the
same airport; only picked training flights are kept, and the model is refitted on them. Inputs are known for
the flights to be predicted (only their labels are hidden), so this can be practised honestly on Jan/Jul CV
with each held-out month's inputs.

Distance: every regressor input (`month` and history averages included). Numbers become their percentile
among the airport's training flights (missing = -0.5, so missing matches missing); a category input counts 1
when it differs (missing is its own value). Each input's squared difference is weighted by its share of the
gain of a fast main LightGBM fitted on the split's training flights only, so inputs count as much as the
model uses them.

Arms, each a fast main LightGBM (v54's LB1 settings at learning rate 0.1, 1,000 rounds, seed 42) scored
against the same fast model on all training flights (`control`), alone and inside v54's practice recipe
(C5) at network share 0.71 (v58), with only the main LightGBM replaced:
  knn_k1 ... knn_k500  keep every training flight picked within some test flight's K nearest
  knn_w5 ... knn_w50   the same, each weighted by how many test flights picked it (the kept set mirrors the test mix)
  knn_far2.0           drop training flights farther than C typical 5th-neighbour distances from every test flight

Neighbours' taxi times (`neighbour-labels`): each held-out flight's K nearest training flights' mean label,
compared with v58's practice predictions (bias by group, and a blend weight fitted on the other month).

Run: .venv/bin/python VALIDATION/similar_rows_jan_jul.py knn      (GPU; set TORCH_DISABLE_NATIVE_JIT=1)
     .venv/bin/python VALIDATION/similar_rows_jan_jul.py scan
     .venv/bin/python VALIDATION/similar_rows_jan_jul.py scan-score
     .venv/bin/python VALIDATION/similar_rows_jan_jul.py neighbour-labels      (GPU)
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
import tuning_seasonal as ts
from records import write_json

OUT = ts.RUNS / 'v3_20261007_similar_rows_jan_jul'
SCAN = dict(learning_rate=0.1, rounds=1000)       # fast main LightGBM for choosing K
KNN_MAX = 500
KNN_WEIGHT_ROUNDS = 300
KNN_K = [1, 3, 5, 10, 20, 50, 100, 200, 500]     # 100-500 added 2026-10-07 after v59 (K = 50) lost on the leaderboard
KNN_WEIGHTED = [5, 20, 50]
KNN_FAR = [2.0]      # 4x and 8x kept every flight in both months


def log(msg):
    print(time.strftime('[%H:%M:%S] ') + msg, flush=True)


def train(ideas, X, y, params, rounds, weight=None):
    """`ideas_jan_jul.train` with optional row weights."""
    cats = ideas.signals._categorical(X.columns) + [c for c in ideas.ef.CATEGORICAL if c in X.columns]
    return lgb.train(params, lgb.Dataset(X, y, weight=weight, categorical_feature=cats), num_boost_round=rounds)


def fast_params():
    spec = hpo.trial_spec('lgb', 'LB1')
    return ts.lgb_params(dict(hpo.V44, **spec['overrides']), ts.SEED, SCAN['learning_rate'])


# ---------------------------------------------------------------- nearest neighbours

def percentiles(train, test):
    """Each column's percentile among the training values (ties at their middle); missing -> -0.5."""
    out_tr, out_te = np.full(train.shape, -0.5, np.float32), np.full(test.shape, -0.5, np.float32)
    for j in range(train.shape[1]):
        ref = np.sort(train[~np.isnan(train[:, j]), j])
        for src, dst in ((train, out_tr), (test, out_te)):
            v = src[:, j]
            ok = ~np.isnan(v)
            mid = (np.searchsorted(ref, v[ok], 'left') + np.searchsorted(ref, v[ok], 'right')) / 2
            dst[ok, j] = mid / max(len(ref), 1)
    return out_tr, out_te


def input_weights(ideas, item, cols):
    """Each input's gain share in a fast main LightGBM fitted on this split's training flights only."""
    booster = train(ideas, item['X'][cols], item['y'], fast_params(), KNN_WEIGHT_ROUNDS)
    g = pd.Series(booster.feature_importance('gain'), index=booster.feature_name()).reindex(cols).fillna(0)
    return g / g.sum()


def distance_space(X, V, weight):
    """Training and test rows of one airport as points: weighted percentiles for numbers, weighted one-hot
    for categories, so the squared distance is the weighted sum of per-input differences."""
    cols = [c for c in X.columns if weight[c] > 0]
    cats = [c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in cols if c not in cats]
    tr, te = percentiles(X[nums].to_numpy(float), V[nums].to_numpy(float))
    scale = np.sqrt(weight[nums].to_numpy(np.float32))
    parts_tr, parts_te = [tr * scale], [te * scale]
    for c in cats:
        a, b = X[c].astype(object).fillna('<missing>').astype(str), V[c].astype(object).fillna('<missing>').astype(str)
        levels = pd.Index(pd.concat([a, b]).unique())
        if len(levels) < 2:
            continue
        for src, dst in ((a, parts_tr), (b, parts_te)):
            hot = np.zeros((len(src), len(levels)), np.float32)
            hot[np.arange(len(src)), levels.get_indexer(src)] = np.sqrt(weight[c] / 2)     # a mismatch adds weight[c]
            dst.append(hot)
    return np.hstack(parts_tr), np.hstack(parts_te)


def neighbours(train, test):
    """For every training row: the best rank at which a flight to predict picks it (0 = never within
    KNN_MAX), how many flights to predict pick it within each K, and its distance to the nearest flight
    to predict. Also each test row's distance to its 5th nearest training row."""
    import torch
    R, T = torch.from_numpy(train).cuda(), torch.from_numpy(test).cuda()
    r2, t2 = (R * R).sum(1), (T * T).sum(1)
    first = torch.full((len(R),), KNN_MAX + 1, dtype=torch.int32, device='cuda')
    counts = {k: torch.zeros(len(R), dtype=torch.int32, device='cuda') for k in KNN_K}
    fifth = []
    k = min(KNN_MAX, len(R))
    ranks = torch.arange(1, k + 1, dtype=torch.int32, device='cuda')
    for a in range(0, len(T), 1024):
        c = T[a:a + 1024]
        d = t2[a:a + 1024, None] + r2[None, :] - 2 * c @ R.T
        vals, idx = d.topk(k, dim=1, largest=False)
        first.scatter_reduce_(0, idx.flatten(), ranks.expand(len(c), k).flatten(), reduce='amin')
        for kk, cnt in counts.items():
            sel = idx[:, :kk].flatten()
            cnt.scatter_add_(0, sel, torch.ones_like(sel, dtype=torch.int32))
        fifth.append(vals[:, min(4, k - 1)].clamp(min=0).sqrt())
        del d
    nearest = []
    for a in range(0, len(R), 4096):
        d = r2[a:a + 4096, None] + t2[None, :] - 2 * R[a:a + 4096] @ T.T
        nearest.append(d.min(1).values.clamp(min=0).sqrt())
        del d
    first = first.cpu().numpy()
    first[first > KNN_MAX] = 0
    out = (first, {kk: v.cpu().numpy() for kk, v in counts.items()}, torch.cat(nearest).cpu().numpy(),
           torch.cat(fifth).cpu().numpy())
    del R, T
    torch.cuda.empty_cache()
    return out


def run_knn(ideas):
    record = {}
    for month in ts.SPLITS:
        item = ideas.DATA['items'][month]
        cols = item['names'] + ideas.COMBO
        started = time.monotonic()
        weight = input_weights(ideas, item, cols)
        log(f'knn {month} weights: {list(weight.sort_values(ascending=False).head(8).round(3).items())}')
        first = np.zeros(len(item['X']), np.int32)
        counts = {k: np.zeros(len(item['X']), np.int32) for k in KNN_K}
        nearest = np.zeros(len(item['X']), np.float32)
        fifth = np.zeros(len(item['Vn']), np.float32)
        dims = {}
        for ap in np.unique(item['airport']):
            t, v = item['airport'] == ap, item['v_airport'] == ap
            tr, te = distance_space(item['X'].loc[t, cols], item['Vn'].loc[v, cols], weight)
            dims[ap] = tr.shape[1]
            first[t], cnt, nearest[t], fifth[v] = neighbours(tr, te)
            for k in KNN_K:
                counts[k][t] = cnt[k]
            del tr, te
            gc.collect()
        typical = float(np.median(fifth))
        long = item['y'] > 3600
        kept = {f'k{k}': (first > 0) & (first <= k) for k in KNN_K}
        kept.update({f'far{c}': nearest <= c * typical for c in KNN_FAR})
        record[month] = dict(minutes=(time.monotonic() - started) / 60, typical_5th_distance=typical, dimensions=dims,
                             weights={c: round(float(w), 5) for c, w in weight.sort_values(ascending=False).items()},
                             kept={a: dict(all=round(float(m.mean()), 4), over_1h=round(float(m[long].mean()), 4))
                                   for a, m in kept.items()},
                             max_picks={k: int(counts[k].max()) for k in KNN_K})
        log(f"knn {month}: {record[month]['minutes']:.1f} min, typical distance {typical:.3f}; kept {record[month]['kept']}")
        pd.DataFrame({'first_pick': first, 'nearest_test': nearest,
                      **{f'picks_k{k}': counts[k] for k in KNN_K}}).to_parquet(OUT / f'knn_{month}.parquet')
    write_json(OUT / 'knn.json', record)


def knn_rows(arm):
    def rows_of(month, n):
        f = pd.read_parquet(OUT / f'knn_{month}.parquet')
        if len(f) != n:
            raise ValueError(f'{month}: neighbour rows differ from training rows')
        if arm.startswith('w'):
            w = f[f'picks_k{arm[1:]}'].to_numpy(float)
            return w > 0, w[w > 0] / w[w > 0].mean()
        if arm.startswith('k'):
            keep = (f.first_pick > 0) & (f.first_pick <= int(arm[1:]))
        else:
            typical = ts.read(OUT / 'knn.json')[month]['typical_5th_distance']
            keep = f.nearest_test <= float(arm[3:]) * typical
        return keep.to_numpy(), None
    return rows_of


# ---------------------------------------------------------------- fast fits

def fit_arm(ideas, path, params, rounds, rows_of, label):
    """Main LightGBM per split on the rows/weights `rows_of(month, n)` gives; held-out predictions to `path`."""
    import modeling
    frames, info = [], {}
    for month in ts.SPLITS:
        item = ideas.DATA['items'][month]
        cols = item['names'] + ideas.COMBO
        keep, w = rows_of(month, len(item['X']))
        if len(keep) != len(item['X']):
            raise ValueError(f'{month}: kept rows differ from training rows')
        started = time.monotonic()
        booster = train(ideas, item['X'].loc[keep, cols], item['y'][keep], params, rounds, w)
        pred = modeling.predict(booster, item['Vn'][cols], threads=ts.THREADS)
        info[month] = dict(minutes=(time.monotonic() - started) / 60, rows=int(keep.sum()),
                           weight_ess_share=None if w is None else float(w.sum() ** 2 / (w ** 2).sum() / len(w)))
        log(f'{label} {month}: {info[month]}')
        frames.append(pd.DataFrame({'mvt_id': item['ids'], 'month': month, 'pred': pred.astype(np.float32)}))
        del booster
        gc.collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return info


def scan_arms():
    arms = {'control': lambda month, n: (np.ones(n, bool), None)}
    if (OUT / 'knn.json').exists():
        arms.update({f'knn_{a}': knn_rows(a) for a in [f'k{k}' for k in KNN_K] + [f'w{k}' for k in KNN_WEIGHTED]
                     + [f'far{c}' for c in KNN_FAR]})
    return arms


def run_scan(ideas):
    path_rec = OUT / 'scan.json'
    record = ts.read(path_rec) if path_rec.exists() else dict(settings=SCAN, arms={})
    for arm, rows_of in scan_arms().items():
        path = OUT / 'scan' / f'{arm}.parquet'
        if path.exists():
            continue
        record['arms'][arm] = fit_arm(ideas, path, fast_params(), SCAN['rounds'], rows_of, f'scan {arm}')
        write_json(path_rec, record)


# ---------------------------------------------------------------- scoring inside v54's practice recipe (C5)

def context():
    sc = hpo.Scorer()
    wins = {p: hpo.winner(p) for p in ['rome', 'nn', 'catboost', 'lgb', 'airport']}
    use = {p: w for p, w in wins.items() if w and w['confirmed']}
    f = sc.frame('rome', use['rome']['trial'], hpo.SEED)
    rome = {m: f[m][use['rome']['checkpoint']].to_numpy(float) for m in ts.SPLITS}
    t = use['nn']['trial']
    frames = [sc.frame('nn', t, hpo.SEED), sc.frame('nn', t + '_k34', hpo.SEED)] + [sc.frame('nn', t, s) for s in hpo.SEEDS[1:]]
    pool = {m: np.concatenate([fr[m].to_numpy(float) for fr in frames], axis=1).mean(1) for m in ts.SPLITS}
    reps = {p: {s: hpo.member_preds(sc, p, use[p], s) for s in hpo.SEEDS} for p in ['catboost', 'lgb', 'airport']}
    return dict(sc=sc, rome=rome, pool=pool, reps=reps)


def errors(ctx, seed, month, lgb_pred, share):
    """Squared errors with the main LightGBM's held-out prediction `lgb_pred`, alone or in the blend."""
    sc = ctx['sc']
    if share == 'alone':
        return (sc.d[month]['y'] - sc.final(month, lgb_pred, ctx['rome'][month])) ** 2
    hpo.W_NN = share
    kw = {hpo.MEMBER_OF[p]: ctx['reps'][p][seed][month] for p in ctx['reps']}
    kw['combo'] = lgb_pred
    return sc.se(month, seed, nn=ctx['pool'][month], p=ctx['rome'][month], **kw)


def by_month(ctx, path):
    f = pd.read_parquet(path)
    out = {}
    for m in ts.SPLITS:
        g = f[f.month == m]
        if not np.array_equal(g.mvt_id.to_numpy(), ctx['sc'].d[m]['ids']):
            raise ValueError(f'{path.name} {m}: rows differ')
        out[m] = g.pred.to_numpy(float)
    return out


def compare(ctx, control, new, share):
    """control/new: seed -> month -> main LightGBM prediction. Per-seed scores, paired change, body/tail split."""
    sc = ctx['sc']
    se_c, se_n, day, mon, body, seeds = [], [], [], [], [], {}
    for s in new:
        r = {}
        for m in ts.SPLITS:
            e0, e1 = errors(ctx, s, m, control[s][m], share), errors(ctx, s, m, new[s][m], share)
            r[m] = (float(np.sqrt(e0.mean())), float(np.sqrt(e1.mean())))
            se_c.append(e0); se_n.append(e1); day.append(sc.d[m]['day']); mon.append(np.full(len(e0), m))
            body.append(sc.d[m]['y'] <= 3600)
        seeds[s] = dict(by_month=r, control=float(np.mean([v[0] for v in r.values()])),
                        new=float(np.mean([v[1] for v in r.values()])))
    se_c, se_n, day, mon, body = map(np.concatenate, (se_c, se_n, day, mon, body))
    ch = ts.paired_bootstrap(day, mon, se_c, se_n)

    def cv(se):
        return float(np.mean([np.sqrt(se[mon == m].mean()) for m in ts.SPLITS]))
    ch['body_change'] = cv(np.where(body, se_n, se_c)) - cv(se_c)       # labels up to 1 h
    ch['tail_change'] = ch['change'] - ch['body_change']
    return dict(seeds=seeds, change=ch)


def run_scan_score():
    ctx = context()
    s = ts.SEED
    control = {s: by_month(ctx, OUT / 'scan' / 'control.parquet')}
    result = {}
    for arm in scan_arms():
        if arm == 'control' or not (OUT / 'scan' / f'{arm}.parquet').exists():
            continue
        new = {s: by_month(ctx, OUT / 'scan' / f'{arm}.parquet')}
        result[arm] = {str(share): compare(ctx, control, new, share) for share in ['alone', 0.71]}
        a, b = result[arm]['alone'], result[arm]['0.71']
        log(f"scan {arm}: alone {a['seeds'][s]['control']:.4f} -> {a['seeds'][s]['new']:.4f} "
            f"({a['change']['change']:+.3f} {a['change']['decision']}, body {a['change']['body_change']:+.3f}), "
            f"blend 0.71 {b['seeds'][s]['control']:.4f} -> {b['seeds'][s]['new']:.4f} "
            f"({b['change']['change']:+.3f}, body {b['change']['body_change']:+.3f})")
    write_json(OUT / 'scan_score.json', result)


# ---------------------------------------------------------------- neighbours' taxi times (user's question, 2026-10-07)
# Do held-out flights whose nearest training flights took long get under-predicted by v58's practice recipe?
# Each held-out flight's K nearest training flights (same distance as above): their mean label is a
# nearest-neighbour prediction. Compared with C5 at network share 0.71, seed 42, all held-out flights.

LABEL_K = [5, 20, 50, 200]
LABEL_BINS = [0, 600, 900, 1200, 1800, 3600, np.inf]


def neighbour_means(train, test, y):
    """Mean label of each test row's K nearest training rows, per K in LABEL_K."""
    import torch
    R, T, Y = torch.from_numpy(train).cuda(), torch.from_numpy(test).cuda(), torch.from_numpy(y.astype(np.float32)).cuda()
    r2, t2 = (R * R).sum(1), (T * T).sum(1)
    k = min(max(LABEL_K), len(R))
    out = {kk: [] for kk in LABEL_K}
    for a in range(0, len(T), 1024):
        d = t2[a:a + 1024, None] + r2[None, :] - 2 * T[a:a + 1024] @ R.T
        idx = d.topk(k, dim=1, largest=False).indices
        yy = Y[idx]
        for kk in LABEL_K:
            out[kk].append(yy[:, :min(kk, k)].mean(1))
        del d
    res = {kk: torch.cat(v).cpu().numpy().astype(float) for kk, v in out.items()}
    del R, T, Y
    torch.cuda.empty_cache()
    return res


def model_prediction(ctx, month, share=0.71, seed=ts.SEED):
    sc = ctx['sc']
    g = np.mean([ctx['reps'][p][seed][month] for p in ['lgb', 'airport', 'catboost']], axis=0)
    return sc.final(month, share * ctx['pool'][month] + (1 - share) * g, ctx['rome'][month])


def run_neighbour_labels(ideas):
    knn = {}
    for month in ts.SPLITS:
        item = ideas.DATA['items'][month]
        cols = item['names'] + ideas.COMBO
        weight = input_weights(ideas, item, cols)
        means = {k: np.full(len(item['Vn']), np.nan) for k in LABEL_K}
        for ap in np.unique(item['airport']):
            t, v = item['airport'] == ap, item['v_airport'] == ap
            tr, te = distance_space(item['X'].loc[t, cols], item['Vn'].loc[v, cols], weight)
            for k, m in neighbour_means(tr, te, item['y'][t]).items():
                means[k][v] = m
        knn[month] = means
        pd.DataFrame({'mvt_id': item['ids'], **{f'knn{k}': means[k] for k in LABEL_K}}).to_parquet(OUT / f'neighbour_labels_{month}.parquet')
        log(f'neighbour labels {month} done')
    del ideas.DATA['items']
    gc.collect()
    ctx = context()
    rec, cache = {}, {}
    for month in ts.SPLITS:
        if not np.array_equal(pd.read_parquet(OUT / f'neighbour_labels_{month}.parquet').mvt_id.to_numpy(), ctx['sc'].d[month]['ids']):
            raise ValueError(f'{month}: held-out rows differ')
        y, pred = ctx['sc'].d[month]['y'], model_prediction(ctx, month)
        cache[month] = (y, pred)
        r = dict(model_rmse=float(np.sqrt(((y - pred) ** 2).mean())),
                 knn_rmse={k: float(np.sqrt(((y - knn[month][k]) ** 2).mean())) for k in LABEL_K},
                 residual_corr={k: float(np.corrcoef(y - pred, knn[month][k] - pred)[0, 1]) for k in LABEL_K})
        b = pd.cut(knn[month][20], LABEL_BINS, right=False)
        f = pd.DataFrame({'bin': b, 'y': y, 'pred': pred, 'knn': knn[month][20]})
        g = f.groupby('bin', observed=True)
        r['by_knn20'] = {str(k): dict(n=int(len(v)), mean_label=float(v.y.mean()), mean_model=float(v.pred.mean()),
                                      mean_knn=float(v.knn.mean()), bias=float((v.y - v.pred).mean()),
                                      model_rmse=float(np.sqrt(((v.y - v.pred) ** 2).mean())))
                         for k, v in g}
        rec[month] = r
        log(f"{month}: model {r['model_rmse']:.2f}; knn alone {({k: round(v, 1) for k, v in r['knn_rmse'].items()})}; "
            f"residual corr {({k: round(v, 3) for k, v in r['residual_corr'].items()})}")
        for k, v in r['by_knn20'].items():
            log(f"   neighbours' mean {k}: n {v['n']:,}, label {v['mean_label']:.0f}, model {v['mean_model']:.0f}, "
                f"bias {v['bias']:+.1f}, model RMSE {v['model_rmse']:.0f}")
    # blend weight on (knn - model) fitted on one month, applied to the other
    rec['blend'] = {}
    for k in LABEL_K:
        out = {}
        for fit_m, use_m in [(ts.SPLITS[0], ts.SPLITS[1]), (ts.SPLITS[1], ts.SPLITS[0])]:
            yf, pf = cache[fit_m]
            df = knn[fit_m][k] - pf
            a = float(((yf - pf) * df).sum() / (df * df).sum())
            yu, pu = cache[use_m]
            new = pu + a * (knn[use_m][k] - pu)
            out[use_m] = dict(weight_from_other_month=a, rmse_before=float(np.sqrt(((yu - pu) ** 2).mean())),
                              rmse_after=float(np.sqrt(((yu - new) ** 2).mean())))
        out['cv_change'] = float(np.mean([v['rmse_after'] - v['rmse_before'] for v in out.values()]))
        rec['blend'][k] = out
        log(f"blend knn{k}: weights {[round(v['weight_from_other_month'], 3) for v in out.values() if isinstance(v, dict)]}, "
            f"Jan/Jul CV change {out['cv_change']:+.3f}")
    write_json(OUT / 'neighbour_labels.json', rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['knn', 'scan', 'scan-score', 'neighbour-labels'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ('knn', 'scan', 'neighbour-labels'):
        ideas = hpo.load_trees()
        if a.stage == 'knn':
            run_knn(ideas)
        elif a.stage == 'scan':
            run_scan(ideas)
        else:
            run_neighbour_labels(ideas)
    else:
        run_scan_score()


if __name__ == '__main__':
    main()
