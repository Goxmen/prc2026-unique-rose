"""Jan/Jul CV hyperparameter search for the parts of v51 (C4) that were never tuned.

Asked 2026-10-05. v51's main LightGBM uses v44's settings, which were searched on v43's inputs;
the per-airport LightGBMs copy them; CatBoost, the `ple` networks and the Rome classifier use
settings fixed once. This runner searches each part on the prepared Jan/Jul splits, one part at
a time, and scores every trial inside v51's blend:

    base  = 0.6 x networks + 0.4 x mean(combo LightGBM, per-airport combo LightGBMs, CatBoost)
    final = shared Rome mix, corrections and rules (tuning_seasonal.final_prediction)

Search trials use seed 42. A trial's part replaces the same part of C4 at seed 42 (networks: C4's
15 pooled networks; a network trial's three networks are compared with C4's first three seed-42
networks, which have the same seeds). Jan/Jul CV = mean of the two months' RMSE. The trial lists
below were fixed before any trial ran; the one combined trial per part (`*20`) is built by a
rule fixed here too: every single change that beat its part's control by at least MIN_GAIN,
the best value per setting.

Rounds, iterations and epochs:
  lgb, airport, catboost, rome  held-out predictions are saved at checkpoints; the best checkpoint
                                is chosen on Jan/Jul together with the settings, as v44's search
                                chose its rounds
  nn                            epoch chosen on the last training month, as in v51
Optimism: winners and their rounds are picked on the same two months that v44-v51 were chosen on.

`confirm` refits each part's winner at seeds 142/242 when it beats its control by at least
MIN_GAIN at seed 42 (networks: five per seed, as C4). `report` scores every confirmed part, and
all of them together (C5), against C4 at three seeds with the paired day bootstrap.

Run: .venv/bin/python VALIDATION/hpo_jan_jul.py {all,rome,nn,catboost,lgb,airport,confirm,report} [--smoke]
No full-year fits, submissions or ledger edits.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')     # as nn_improve_jan_jul.py, before torch loads
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import tuning_seasonal as ts
from records import RUNS, sha256, write_json

SPLITS_FROM = RUNS / 'v3_20261002_adsblol_jan_jul'
IDEAS = RUNS / 'v3_20261003_ideas_jan_jul'
OUT = RUNS / 'v3_20261005_hpo_jan_jul'
SEED, SEEDS = 42, [42, 142, 242]
W_NN = 0.6
MEMBERS = ['combo', 'combo_airport', 'catboost']
MIN_GAIN = 0.05                       # seconds of Jan/Jul CV at seed 42
STAGES = ['rome', 'nn', 'lgb', 'airport', 'catboost', 'confirm', 'report']   # CatBoost (longest, GPU) overnight
# 2026-10-05, user's request: CatBoost (search + its seed check) moved to the laptop, which runs
# `all --stages catboost confirm` in its own checkout and then writes LAPTOP_DONE; the desktop runs
# `all --stages airport confirm pull report`, where `pull` waits for that marker and copies the
# CatBoost record and predictions back. Each part keeps all its seeds on one machine.
LAPTOP = 'glaptop-wsl'
EXTRA_STAGES = ['pull']

# ---------------------------------------------------------------- trial lists (fixed in advance)

V44 = ts.read(RUNS / 'v3_20261001_jan_jul' / 'search' / 'chosen.json')['settings']

ROME = dict(           # LIRF classifier: overrides of rome_hybrid.CLASSIFIER_PARAMS; (overrides, max rounds)
    R00=({}, 1200),
    R01=(dict(num_leaves=15), 1200), R02=(dict(num_leaves=63), 1200), R03=(dict(num_leaves=127), 1200),
    R04=(dict(min_data_in_leaf=30), 1200), R05=(dict(min_data_in_leaf=300), 1200),
    R06=(dict(learning_rate=0.02), 2400), R07=(dict(feature_fraction=0.6), 1200),
    R08=(dict(lambda_l2=10.0), 1200), R09=(dict(num_leaves=63, min_data_in_leaf=50, learning_rate=0.02), 2400),
)
ROME_STEP = 100

NN_BASE = dict(hidden=[512, 512, 256], dropout=0.1, lr=2e-3, weight_decay=1e-5, batch=4096, epochs=24,
               bins=48, d_ple=16, label_clip=10800.0)
NN = dict(             # overrides of NN_BASE; N00 = C4's own seed-42 networks k0-k2 (no refit)
    N01=dict(dropout=0.2), N02=dict(dropout=0.0, weight_decay=1e-4), N03=dict(weight_decay=1e-4),
    N04=dict(lr=1e-3), N05=dict(lr=4e-3), N06=dict(hidden=[1024, 512, 256]),
    N07=dict(hidden=[512, 512, 512, 256]), N08=dict(hidden=[256, 256, 128]), N09=dict(batch=2048),
    N10=dict(batch=8192, lr=4e-3), N11=dict(bins=96, d_ple=24), N12=dict(d_ple=8), N13=dict(epochs=40),
    N14=dict(label_clip=21600.0), N15=dict(label_clip=5400.0),
)
NN_TRIAL_NETS, NN_FINAL_NETS = 3, 5

CB_BASE = dict(depth=8, learning_rate=0.05, l2_leaf_reg=3, border_count=254)
CB = dict(             # overrides of CB_BASE; (overrides, max iterations)
    K00=({}, 16000),
    K01=(dict(learning_rate=0.1), 10000),
    K02=(dict(learning_rate=0.1, depth=10), 8000),
    K03=(dict(learning_rate=0.1, depth=10, l2_leaf_reg=10), 8000),
    K04=(dict(learning_rate=0.1, l2_leaf_reg=10, random_strength=2), 10000),
    K05=(dict(learning_rate=0.15, depth=6), 10000),
    K06=(dict(learning_rate=0.1, l2_leaf_reg=30), 10000),
    K07=(dict(learning_rate=0.1, bootstrap_type='Bernoulli', subsample=0.7), 10000),
)
CB_STEP = 500

LGB = dict(            # global combo LightGBM, learning rate 0.05, 2,500 rounds: overrides of v44's settings
    L00={}, L01=dict(num_leaves=255), L02=dict(num_leaves=63), L03=dict(min_data_in_leaf=400),
    L04=dict(lambda_l2=300.0), L05=dict(feature_fraction=0.5), L06=dict(max_cat_threshold=8),
    L07=dict(cat_smooth=300.0), L08=dict(min_data_per_group=300, cat_l2=50.0), L09=dict(path_smooth=10.0),
    L10=dict(bagging_fraction=0.6),
)
LGB_LR, LGB_ROUNDS, LGB_STEP = 0.05, 2500, 250
LGB_LONG_LR, LGB_LONG_ROUNDS, LGB_LONG_STEP = 0.02, 10000, 500     # LB0 = v44 settings, LB1 = stage-A winner

AIRPORT = dict(        # per-airport combo LightGBMs, learning rate 0.02 up to 5,000 rounds
    P00={}, P01=dict(num_leaves=63), P02=dict(num_leaves=31, min_data_in_leaf=100),
    P03=dict(num_leaves=63, lambda_l2=300.0), P04=dict(num_leaves=63, min_data_in_leaf=50, lambda_l2=10.0),
    P05=dict(num_leaves=255), P06=dict(feature_fraction=0.5),
)
AIRPORT_LR, AIRPORT_ROUNDS, AIRPORT_STEP = 0.02, 5000, 500

SMOKE = False


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def out_dir():
    return OUT / 'smoke' if SMOKE else OUT


def record_path(part):
    return out_dir() / f'{part}.json'


def load_record(part):
    p = record_path(part)
    return json.loads(p.read_text()) if p.exists() else {}


def pred_path(part, trial, seed):
    return out_dir() / 'predictions' / part / f'{trial}_s{seed}.parquet'


def save(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def checkpoints(rounds, step):
    return list(range(step, rounds + 1, step))


def combined(single, scores, base_key):
    """Best value per setting among single changes that beat the control by MIN_GAIN."""
    control = scores[base_key]['cv']
    good = sorted([(scores[t]['cv'], t) for t in single if t in scores and control - scores[t]['cv'] >= MIN_GAIN])
    merged = {}
    for _, t in good:                                  # best first: its value for a key wins
        for k, v in single[t].items():
            merged.setdefault(k, v)
    return merged if len(good) >= 2 else None


# ---------------------------------------------------------------- scoring inside C4

class Scorer:
    """C4's parts per month and seed, and the shared final prediction."""

    def __init__(self):
        self.d = {}
        for month in ts.SPLITS:
            sp = ts.split_dir(SPLITS_FROM, month)
            held, meta = pd.read_pickle(sp / 'heldout.pkl'), ts.read(sp / 'meta.json')
            raw = held['raw']
            ids = raw.mvt_id.to_numpy()
            nets = {}
            for s in SEEDS:
                for k in range(5):
                    f = pd.read_parquet(IDEAS / 'networks' / f'nn_combo_{month}_s{s}_k{k}.parquet')
                    if not np.array_equal(f.mvt_id.to_numpy(), ids):
                        raise ValueError(f'{month}: network rows differ')
                    nets[s, k] = f.base.to_numpy(float)
            members = {}
            for a in MEMBERS:
                for s in SEEDS:
                    f = pd.read_parquet(IDEAS / 'predictions' / f'{a}_seed{s}.parquet')
                    f = f[f.month == month]
                    if not np.array_equal(f.mvt_id.to_numpy(), ids):
                        raise ValueError(f'{a} {month} seed {s}: rows differ')
                    members[a, s] = f.base.to_numpy(float)
            self.d[month] = dict(held=held, meta=meta, ids=ids, y=raw.taxi_sec.to_numpy(float),
                                 day=pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy(),
                                 nets=nets, members=members,
                                 pool=np.mean([nets[s, k] for s in SEEDS for k in range(5)], axis=0))

    def final(self, month, base, p=None):
        d = self.d[month]
        held = d['held']
        return ts.final_prediction(base, held['p'] if p is None else p, held['V'], held['raw'],
                                   d['meta']['parameters'], d['meta']['corrections'])

    def se(self, month, seed=SEED, nn=None, p=None, **replace):
        d = self.d[month]
        g = np.mean([replace[a] if a in replace else d['members'][a, seed] for a in MEMBERS], axis=0)
        base = W_NN * (d['pool'] if nn is None else nn) + (1 - W_NN) * g
        return (d['y'] - self.final(month, base, p)) ** 2

    def cv(self, by_month):
        """by_month: month -> squared errors; returns per-month RMSE and their mean."""
        r = {m: float(np.sqrt(e.mean())) for m, e in by_month.items()}
        r['cv'] = float(np.mean([r[m] for m in ts.SPLITS]))
        return r

    def frame(self, part, trial, seed):
        f = pd.read_parquet(pred_path(part, trial, seed))
        out = {}
        for m in ts.SPLITS:
            g = f[f.month == m]
            if not np.array_equal(g.mvt_id.to_numpy(), self.d[m]['ids']):
                raise ValueError(f'{part} {trial} s{seed} {m}: rows differ')
            out[m] = g.drop(columns=['mvt_id', 'month'])
        return out

    def score_checkpoints(self, part, trial, seed=SEED):
        """Jan/Jul CV of C4 with `part` replaced by each saved checkpoint column of a trial."""
        f = self.frame(part, trial, seed)
        cols = list(f[ts.SPLITS[0]].columns)
        res = {}
        for c in cols:
            by = {}
            for m in ts.SPLITS:
                v = f[m][c].to_numpy(float)
                if part == 'rome':
                    by[m] = self.se(m, seed, p=v)
                elif part == 'nn':
                    raise ValueError('networks are scored with score_nn')
                else:
                    by[m] = self.se(m, seed, **{MEMBER_OF[part]: v})
            res[c] = self.cv(by)
        best = min(res, key=lambda c: res[c]['cv'])
        return dict(by_checkpoint={c: r['cv'] for c, r in res.items()}, best_checkpoint=best, **res[best])

    def score_nn(self, trial, seed=SEED, nets=NN_TRIAL_NETS):
        if trial == 'N00':
            by = {m: self.se(m, seed, nn=np.mean([self.d[m]['nets'][seed, k] for k in range(nets)], axis=0))
                  for m in ts.SPLITS}
        else:
            f = self.frame('nn', trial, seed)
            by = {m: self.se(m, seed, nn=f[m][[f'n{k}' for k in range(nets)]].to_numpy(float).mean(1))
                  for m in ts.SPLITS}
        return self.cv(by)


MEMBER_OF = dict(lgb='combo', airport='combo_airport', catboost='catboost')


def control_cv(scorer, seed=SEED):
    return scorer.cv({m: scorer.se(m, seed) for m in ts.SPLITS})


# ---------------------------------------------------------------- Rome classifier

def run_rome(trials=None, seed=SEED):
    import rome_hybrid
    import rome_signals as signals
    scorer = Scorer()
    record = load_record('rome')
    record.setdefault('control', control_cv(scorer))
    trials = dict(trials or ROME)
    data = {}
    for month in ts.SPLITS:
        tr = pd.read_pickle(ts.split_dir(SPLITS_FROM, month) / 'after_drop.pkl')
        lirf = tr['airport'] == 'LIRF'
        XL, yL = tr['X'].loc[lirf], tr['y'][lirf]
        rawL = ts.minimal_raw(np.full(len(XL), 'LIRF'), XL.index)
        held = scorer.d[month]['held']
        V, raw = held['V'], held['raw']
        lh = (raw.ADEP_mvt.to_numpy() == 'LIRF') & np.isfinite(V[rome_hybrid.GAP].to_numpy(float))
        data[month] = dict(X=signals.classifier_matrix(XL, rawL, ts.KIND, ts.CLASSIFIER),
                           s=rome_hybrid.substituted(XL, yL, rawL).astype(float),
                           V=signals.classifier_matrix(V.loc[lh], raw.loc[lh], ts.KIND, ts.CLASSIFIER), lh=lh)
        del tr
        gc.collect()
    queue = list(trials)
    while queue:
        name = queue.pop(0)
        over, rounds = trials[name]
        path = pred_path('rome', name, seed)
        if not path.exists():
            started = time.monotonic()
            params = dict(rome_hybrid.CLASSIFIER_PARAMS, **over, seed=seed, bagging_seed=seed + 1,
                          feature_fraction_seed=seed + 2, num_threads=ts.THREADS)
            frames = []
            for month in ts.SPLITS:
                d = data[month]
                b = signals._train(params, d['X'], d['s'], rounds, f'rome {name} {month}')
                cols = {}
                for c in checkpoints(rounds, ROME_STEP):
                    p = np.zeros(len(d['lh']))
                    p[d['lh']] = b.predict(d['V'], num_iteration=c, num_threads=ts.THREADS)
                    cols[f'c{c}'] = p
                frames.append(pd.DataFrame({'mvt_id': scorer.d[month]['ids'], 'month': month, **cols}))
            save(path, pd.concat(frames, ignore_index=True))
            record.setdefault('trials', {})[name] = dict(overrides=over, rounds=rounds,
                                                         minutes=(time.monotonic() - started) / 60)
        if name == 'R00':
            f = scorer.frame('rome', 'R00', seed)
            same = all(np.allclose(f[m]['c300'].to_numpy(), scorer.d[m]['held']['p'], atol=1e-9) for m in ts.SPLITS)
            record['control_reproduced_at_300'] = bool(same)
            log(f'rome R00 at 300 rounds reproduces C4 probabilities: {same}')
        record.setdefault('scores', {})[name] = scorer.score_checkpoints('rome', name, seed)
        write_json(record_path('rome'), record)
        log(f"rome {name}: {record['scores'][name]['cv']:.4f} at {record['scores'][name]['best_checkpoint']} "
            f"(control {record['control']['cv']:.4f})")
        if name == list(ROME)[-1] and not SMOKE:
            # The other parts' combined-trial rule, applied to Rome after its single changes were scored.
            merged = combined({t: o for t, (o, _) in ROME.items()}, record['scores'], 'R00')
            record['R20'] = merged
            write_json(record_path('rome'), record)
            if merged:
                trials['R20'] = (merged, 2400 if merged.get('learning_rate', 0.05) < 0.05 else 1200)
                queue.append('R20')


# ---------------------------------------------------------------- networks

def nn_config(over):
    cfg = json.loads(json.dumps(NN_BASE))
    cfg.update(over)
    return cfg


def apply_nn(cfg, nn_jan_jul, nni):
    nn_jan_jul.RECIPE.update(hidden=list(cfg['hidden']), dropout=cfg['dropout'], label_clip=cfg['label_clip'])
    nni.NEW.update(bins=cfg['bins'], d_ple=cfg['d_ple'], epochs=cfg['epochs'], batch=cfg['batch'])
    nni.NEW['ple'] = dict(lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def nn_fit(jobs, seed_nets):
    """jobs: trial -> overrides. Fits seed_nets = {seed: [k...]} networks per trial and month."""
    import torch
    import adsblol
    import extra_features as ef
    import nn_features_jan_jul as nnf
    import nn_improve_jan_jul as nni
    import nn_jan_jul
    import overnight_run
    current = dict(hidden=nn_jan_jul.RECIPE['hidden'], dropout=nn_jan_jul.RECIPE['dropout'],
                   label_clip=nn_jan_jul.RECIPE['label_clip'], lr=nni.NEW['ple']['lr'],
                   weight_decay=nni.NEW['ple']['weight_decay'],
                   **{k: nni.NEW[k] for k in ('bins', 'd_ple', 'epochs', 'batch')})
    if current != NN_BASE:
        raise ValueError(f'C4 network recipe changed: {current}')
    record = load_record('nn')
    groups = ['arrtaxi', 'peers', 'rwycfg']
    if SMOKE:
        nni.NEW['epochs'] = 2
    todo = {t: o for t, o in jobs.items()
            if any(not pred_path('nn', t, s).exists() for s in seed_nets)}
    if not todo:
        return
    table, extra = adsblol.load_all(), ef.load()
    dep, arr = overnight_run.load_inputs(1.)
    parts = {}                                    # (trial, seed) -> month -> frame columns
    for month in ts.SPLITS:
        item = nnf.load_month(month, dep, arr, table, extra, None, groups)
        ids = item['raw'].mvt_id.to_numpy()
        key, d = None, None
        for trial in sorted(todo, key=lambda t: (nn_config(todo[t])['bins'], nn_config(todo[t])['label_clip'], t)):
            cfg = nn_config(todo[trial])
            if SMOKE:
                cfg['epochs'] = 2
            apply_nn(cfg, nn_jan_jul, nni)
            if (cfg['bins'], cfg['label_clip']) != key:
                del d
                torch.cuda.empty_cache()
                d = nni.Data(item, 'cuda')
                key = (cfg['bins'], cfg['label_clip'])
            tr = record.setdefault('trials', {}).setdefault(trial, dict(overrides=todo[trial], config=cfg))
            for seed, ks in seed_nets.items():
                if pred_path('nn', trial, seed).exists():
                    continue
                sm = tr.setdefault(month, {}).setdefault(str(seed), {})
                started = time.monotonic()
                if 'stop_epoch' not in sm:
                    _, _, curve = nni.train('ple', d, d.inner_train, seed, cfg['epochs'],
                                            (d.subset(d.inner_val), d.y[d.inner_val]))
                    sm.update(stop_epoch=min(curve, key=lambda c: c['rmse'])['epoch'],
                              inner_rmse=min(c['rmse'] for c in curve), inner_curve=[c['rmse'] for c in curve])
                cols = {}
                for k in ks:
                    _, base, _ = nni.train('ple', d, np.arange(len(d.y)), seed + 1000 * k, sm['stop_epoch'],
                                           (d.held, d.yv))
                    cols[f'n{k}'] = base
                parts.setdefault((trial, seed), {})[month] = pd.DataFrame({'mvt_id': ids, 'month': month, **cols})
                sm['minutes'] = (time.monotonic() - started) / 60
                log(f'nn {trial} s{seed} {month}: stop {sm["stop_epoch"]}, inner {sm["inner_rmse"]:.2f}, '
                    f'{sm["minutes"]:.1f} min')
                write_json(record_path('nn'), record)
        del d, item
        gc.collect()
        torch.cuda.empty_cache()
    for (trial, seed), by in parts.items():
        if len(by) == len(ts.SPLITS):
            save(pred_path('nn', trial, seed), pd.concat([by[m] for m in ts.SPLITS], ignore_index=True))
    apply_nn(NN_BASE, nn_jan_jul, nni)


def run_nn():
    trials = dict(list(NN.items())[:1]) if SMOKE else NN
    nn_fit(trials, {SEED: list(range(1 if SMOKE else NN_TRIAL_NETS))})
    scorer = Scorer()
    record = load_record('nn')
    nets = 1 if SMOKE else NN_TRIAL_NETS
    scores = record.setdefault('scores', {})
    scores['N00'] = scorer.score_nn('N00', nets=nets)
    for t in trials:
        scores[t] = scorer.score_nn(t, nets=nets)
        log(f"nn {t}: {scores[t]['cv']:.4f} (control N00 {scores['N00']['cv']:.4f})")
    write_json(record_path('nn'), record)
    merged = None if SMOKE else combined(NN, scores, 'N00')
    record['N20'] = merged
    write_json(record_path('nn'), record)
    if merged:
        log(f'nn N20 = {merged}')
        nn_fit({'N20': merged}, {SEED: list(range(NN_TRIAL_NETS))})
        record = load_record('nn')
        record['N20'] = merged
        record['scores']['N20'] = scorer.score_nn('N20')
        write_json(record_path('nn'), record)
        log(f"nn N20: {record['scores']['N20']['cv']:.4f}")


# ---------------------------------------------------------------- tree models (shared data)

def load_trees():
    import ideas_jan_jul as ideas
    ideas.ARMS = {'combo': ideas.COMBO}           # load v46's inputs plus the 22 combo inputs, no route
    ideas.load()
    return ideas


def cb_fit(ideas, name, over, iters, seed):
    from catboost import CatBoostRegressor, Pool
    import catboost_jan_jul as cbj
    import extra_features as ef
    import rome_signals as signals
    path = pred_path('catboost', name, seed)
    if path.exists():
        return None
    params = dict(CB_BASE, **over)
    combo = params.pop('combo_inputs', False)
    frames, info = [], {}
    for month in ts.SPLITS:
        item = ideas.DATA['items'][month]
        cols = item['names'] + (ideas.COMBO if combo else [])
        cats = signals._categorical(cols) + [c for c in ef.CATEGORICAL if c in cols]
        X, V = cbj.frame(item['X'][cols], cats), cbj.frame(item['Vn'][cols], cats)
        started = time.monotonic()
        model = CatBoostRegressor(loss_function='RMSE', iterations=iters, random_seed=seed, task_type='GPU',
                                  devices='0', verbose=1000, allow_writing_files=False, **params)
        model.fit(Pool(X, item['y'], cat_features=cats))
        fit_min = (time.monotonic() - started) / 60
        step = min(CB_STEP, iters)
        cols_out = {}
        for i, p in enumerate(model.staged_predict(Pool(V, cat_features=cats), eval_period=step), start=1):
            if i * step <= iters:
                cols_out[f'c{i * step}'] = np.maximum(p, 0).astype(np.float32)
        info[month] = dict(fit_minutes=fit_min, minutes=(time.monotonic() - started) / 60,
                           tree_count=int(model.tree_count_))
        log(f'catboost {name} s{seed} {month}: {info[month]["minutes"]:.1f} min')
        frames.append(pd.DataFrame({'mvt_id': item['ids'], 'month': month, **cols_out}))
        del model, X, V
        gc.collect()
    save(path, pd.concat(frames, ignore_index=True))
    return info


def lgb_fit(ideas, part, name, settings, lr, rounds, step, seed, keep=None):
    """Global (`lgb`) or per-airport (`airport`) combo LightGBM; held-out predictions at checkpoints."""
    import modeling
    path = pred_path(part, name, seed)
    if path.exists():
        return None
    params = ts.lgb_params(dict(V44, **settings), seed, lr)
    marks = keep or checkpoints(rounds, step)
    frames, info = [], {}
    for month in ts.SPLITS:
        item = ideas.DATA['items'][month]
        cols = item['names'] + ideas.COMBO
        started = time.monotonic()
        out = {f'c{c}': np.full(len(item['Vn']), np.nan) for c in marks}
        groups = ([(item['airport'] == ap, item['v_airport'] == ap) for ap in np.unique(item['airport'])]
                  if part == 'airport' else [(slice(None), np.ones(len(item['Vn']), bool))])
        for k, v in groups:
            booster = ideas.train(item['X'].loc[k, cols], item['y'][k], params, rounds)
            if v.any():
                for c in marks:
                    out[f'c{c}'][v] = modeling.predict(booster, item['Vn'].loc[v, cols], num_iteration=c,
                                                       threads=ts.THREADS)
            del booster
        if any(np.isnan(a).any() for a in out.values()):
            raise ValueError(f'{part} {name} {month}: held-out rows without a prediction')
        info[month] = dict(minutes=(time.monotonic() - started) / 60)
        log(f'{part} {name} s{seed} {month}: {info[month]["minutes"]:.1f} min')
        frames.append(pd.DataFrame({'mvt_id': item['ids'], 'month': month,
                                    **{c: a.astype(np.float32) for c, a in out.items()}}))
        gc.collect()
    save(path, pd.concat(frames, ignore_index=True))
    return info


def run_trees(part):
    ideas = load_trees()
    scorer = Scorer()
    record = load_record(part)
    record.setdefault('control', control_cv(scorer))
    scores = record.setdefault('scores', {})

    def done(name, over, info, **extra):
        if info is not None:
            record.setdefault('trials', {})[name] = dict(overrides=over, seed42=info, **extra)
        scores[name] = scorer.score_checkpoints(part, name)
        write_json(record_path(part), record)
        log(f"{part} {name}: {scores[name]['cv']:.4f} at {scores[name]['best_checkpoint']} "
            f"(C4 seed 42 {record['control']['cv']:.4f})")

    if part == 'catboost':
        trials = {'K00': ({}, 600)} if SMOKE else CB
        for name, (over, iters) in trials.items():
            try:
                done(name, over, cb_fit(ideas, name, over, iters, SEED), iterations=iters)
            except Exception as e:                      # e.g. GPU memory on a smaller card: record, go on
                record.setdefault('failed', {})[name] = repr(e)
                write_json(record_path(part), record)
                log(f'catboost {name} failed: {e!r}')
                gc.collect()
        if not SMOKE:
            best = min([t for t in CB if t in scores], key=lambda t: scores[t]['cv'])
            over, iters = CB[best]
            name = 'K10'                    # stage-A winner + the 22 combo inputs (an input change, kept apart)
            done(name, dict(over, combo_inputs=True), cb_fit(ideas, name, dict(over, combo_inputs=True), iters, SEED),
                 iterations=iters, parent=best)
    elif part == 'lgb':
        trials = {'L00': {}} if SMOKE else LGB
        rounds = 300 if SMOKE else LGB_ROUNDS
        for name, over in trials.items():
            done(name, over, lgb_fit(ideas, 'lgb', name, over, LGB_LR, rounds, LGB_STEP if not SMOKE else 100, SEED),
                 learning_rate=LGB_LR, rounds=rounds)
        if not SMOKE:
            merged = combined(LGB, scores, 'L00')
            record['L20'] = merged
            if merged:
                done('L20', merged, lgb_fit(ideas, 'lgb', 'L20', merged, LGB_LR, LGB_ROUNDS, LGB_STEP, SEED),
                     learning_rate=LGB_LR, rounds=LGB_ROUNDS)
            pool = {**LGB, **({'L20': merged} if merged else {})}
            best = min(pool, key=lambda t: scores[t]['cv'])
            longs = {'LB0': {}} | ({'LB1': pool[best]} if pool[best] else {})
            for name, over in longs.items():
                done(name, over, lgb_fit(ideas, 'lgb', name, over, LGB_LONG_LR, LGB_LONG_ROUNDS, LGB_LONG_STEP, SEED),
                     learning_rate=LGB_LONG_LR, rounds=LGB_LONG_ROUNDS, parent=best if name == 'LB1' else 'v44')
            five = scores['LB0']['by_checkpoint'].get('c5000')
            record['LB0_at_5000_vs_C4'] = None if five is None else five - record['control']['cv']
            write_json(record_path(part), record)
    elif part == 'airport':
        trials = {'P00': {}} if SMOKE else AIRPORT
        rounds = 300 if SMOKE else AIRPORT_ROUNDS
        for name, over in trials.items():
            done(name, over, lgb_fit(ideas, 'airport', name, over, AIRPORT_LR, rounds,
                                     AIRPORT_STEP if not SMOKE else 100, SEED),
                 learning_rate=AIRPORT_LR, rounds=rounds)
        if not SMOKE:
            merged = combined(AIRPORT, scores, 'P00')
            record['P20'] = merged
            if merged:
                done('P20', merged, lgb_fit(ideas, 'airport', 'P20', merged, AIRPORT_LR, AIRPORT_ROUNDS,
                                            AIRPORT_STEP, SEED), learning_rate=AIRPORT_LR, rounds=AIRPORT_ROUNDS)
            p25 = scores['P00']['by_checkpoint'].get('c2500')
            record['P00_at_2500_vs_C4'] = None if p25 is None else p25 - record['control']['cv']
            write_json(record_path(part), record)


# ---------------------------------------------------------------- confirmation and report

def winner(part):
    """(trial, checkpoint, gain over the part's control at seed 42) or None."""
    rec = load_record(part)
    scores = rec.get('scores', {})
    if not scores:
        return None
    control = scores['N00']['cv'] if part == 'nn' else rec['control']['cv']
    pool = [k for k in scores if not (SMOKE and k == 'N00')]      # smoke: exercise every confirm path
    t = min(pool, key=lambda k: scores[k]['cv'])
    gain = control - scores[t]['cv']
    return dict(trial=t, checkpoint=scores[t].get('best_checkpoint'), gain=gain, cv=scores[t]['cv'],
                control=control, confirmed=(SMOKE or gain >= MIN_GAIN) and t != 'N00')


def trial_spec(part, trial):
    rec = load_record(part)
    return rec['trials'][trial]


def run_confirm():
    wins = {p: winner(p) for p in ['rome', 'nn', 'catboost', 'lgb', 'airport']}
    write_json(out_dir() / 'winners.json', wins)
    for p, w in wins.items():
        log(f'{p}: {w}')
    w = wins['nn']
    if w and w['confirmed']:
        cfg_over = trial_spec('nn', w['trial'])['overrides']
        # seed 42 already has k0-k2 under the trial name: add k3-k4, then five per other seed
        nn_fit({w['trial'] + '_k34': cfg_over}, {SEED: [3, 4]})
        nn_fit({w['trial']: cfg_over}, {s: list(range(NN_FINAL_NETS)) for s in SEEDS[1:]})
    need_trees = any(wins[p] and wins[p]['confirmed'] for p in ['catboost', 'lgb', 'airport'])
    if not need_trees:
        return
    ideas = load_trees()
    for p in ['catboost', 'lgb', 'airport']:
        w = wins[p]
        if not (w and w['confirmed']):
            continue
        spec = trial_spec(p, w['trial'])
        n = int(w['checkpoint'][1:])
        for seed in SEEDS[1:]:
            if p == 'catboost':
                info = cb_fit(ideas, w['trial'], spec['overrides'], n, seed)
            else:
                info = lgb_fit(ideas, p, w['trial'], spec['overrides'], spec['learning_rate'], n, n, seed, keep=[n])
            rec = load_record(p)
            rec.setdefault('confirm', {})[str(seed)] = dict(trial=w['trial'], rounds=n, info=info)
            write_json(record_path(p), rec)


def member_preds(scorer, part, w, seed):
    """Held-out base of a confirmed winner at one seed, per month."""
    if seed == SEED:
        f = scorer.frame(part, w['trial'], seed)
        return {m: f[m][w['checkpoint']].to_numpy(float) for m in ts.SPLITS}
    f = scorer.frame(part, w['trial'], seed)
    return {m: f[m].iloc[:, -1].to_numpy(float) for m in ts.SPLITS}


def run_report():
    scorer = Scorer()
    wins = {p: winner(p) for p in ['rome', 'nn', 'catboost', 'lgb', 'airport']}
    use = {p: w for p, w in wins.items() if w and w['confirmed']}
    reps, report = {}, {}
    if 'rome' in use:
        f = scorer.frame('rome', use['rome']['trial'], SEED)
        reps['rome'] = {s: {m: f[m][use['rome']['checkpoint']].to_numpy(float) for m in ts.SPLITS} for s in SEEDS}
    if 'nn' in use:
        t = use['nn']['trial']
        frames = [scorer.frame('nn', t, SEED), scorer.frame('nn', t + '_k34', SEED)]
        frames += [scorer.frame('nn', t, s) for s in SEEDS[1:]]
        pool = {m: np.concatenate([f[m].to_numpy(float) for f in frames], axis=1).mean(1) for m in ts.SPLITS}
        report['networks_pooled'] = int(sum(f[ts.SPLITS[0]].shape[1] for f in frames))
        reps['nn'] = {s: pool for s in SEEDS}
    for p in ['catboost', 'lgb', 'airport']:
        if p in use:
            reps[p] = {s: member_preds(scorer, p, use[p], s) for s in SEEDS}

    def se_all(parts):
        out, per_seed = {}, {}
        for m in ts.SPLITS:
            errs = []
            for s in SEEDS:
                kw = {MEMBER_OF[p]: reps[p][s][m] for p in parts if p in MEMBER_OF}
                errs.append(scorer.se(m, s, nn=reps['nn'][s][m] if 'nn' in parts else None,
                                      p=reps['rome'][s][m] if 'rome' in parts else None, **kw))
                per_seed.setdefault(s, []).append(float(np.sqrt(errs[-1].mean())))
            out[m] = np.mean(errs, axis=0)
        return out, {s: float(np.mean(v)) for s, v in per_seed.items()}

    month = np.concatenate([np.full(len(scorer.d[m]['y']), m) for m in ts.SPLITS])
    day = np.concatenate([scorer.d[m]['day'] for m in ts.SPLITS])
    base_se, base_seed = se_all([])
    flat = lambda e: np.concatenate([e[m] for m in ts.SPLITS])
    report.update(winners=wins, C4=dict(**scorer.cv(base_se), by_seed=base_seed), parts={})
    arms = [[p] for p in use] + ([list(use)] if len(use) > 1 else [])
    for parts in arms:
        e, by_seed = se_all(parts)
        name = 'C5_all' if len(parts) > 1 else parts[0]
        report['parts'][name] = dict(parts=parts, **scorer.cv(e), by_seed=by_seed,
                                     change=ts.paired_bootstrap(day, month, flat(base_se), flat(e)),
                                     seeds_improving=sum(by_seed[s] < base_seed[s] for s in SEEDS))
        log(f"{name}: {report['parts'][name]['cv']:.4f} {report['parts'][name]['change']} "
            f"seeds improving {report['parts'][name]['seeds_improving']}/3")
    write_json(out_dir() / 'report.json', report)


# ---------------------------------------------------------------- driver

def run_pull():
    remote = '~/PRC/' + str(OUT.relative_to(ROOT))
    while subprocess.run(['ssh', LAPTOP, f'test -f {remote}/LAPTOP_DONE']).returncode != 0:
        time.sleep(300)
    (OUT / 'predictions' / 'catboost').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs' / 'laptop').mkdir(parents=True, exist_ok=True)
    for src, dst in [(f'{remote}/catboost.json', OUT / 'catboost.json'),
                     (f'{remote}/predictions/catboost/', OUT / 'predictions' / 'catboost'),
                     (f'{remote}/logs/', OUT / 'logs' / 'laptop')]:
        subprocess.run(['rsync', '-a', f'{LAPTOP}:{src}', str(dst)], check=True)
    log('CatBoost record and predictions copied from the laptop')


def run_all(smoke, stages=None):
    logs = out_dir() / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    path = out_dir() / 'stages.json'
    status = json.loads(path.read_text()) if path.exists() else {}
    for stage in stages or STAGES:
        started = time.monotonic()
        cmd = [sys.executable, str(Path(__file__).resolve()), stage] + (['--smoke'] if smoke else [])
        with open(logs / f'{stage}.log', 'a') as fh:
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        status[stage] = dict(exit=code, minutes=(time.monotonic() - started) / 60)
        write_json(out_dir() / 'stages.json', status)
        log(f'{stage}: exit {code} after {status[stage]["minutes"]:.1f} min')


def main():
    global SMOKE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['all'] + STAGES + EXTRA_STAGES)
    ap.add_argument('--stages', nargs='+', choices=STAGES + EXTRA_STAGES, help='with `all`: run only these, in order')
    ap.add_argument('--smoke', action='store_true', help='tiny trials into the run directory\'s smoke/ folder')
    args = ap.parse_args()
    SMOKE = args.smoke
    out_dir().mkdir(parents=True, exist_ok=True)
    write_json(out_dir() / 'source.json', dict(runner=sha256(Path(__file__)), smoke=SMOKE))
    if args.stage == 'all':
        run_all(SMOKE, args.stages)
    elif args.stage == 'pull':
        run_pull()
    elif args.stage == 'rome':
        run_rome(dict(R00=({}, 300)) if SMOKE else None)
    elif args.stage == 'nn':
        run_nn()
    elif args.stage in ('catboost', 'lgb', 'airport'):
        run_trees(args.stage)
    elif args.stage == 'confirm':
        run_confirm()
    elif args.stage == 'report':
        run_report()


if __name__ == '__main__':
    main()
