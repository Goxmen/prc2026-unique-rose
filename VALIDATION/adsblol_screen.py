"""Screen the ADSB.lol inputs on the Jan/Jul validation's July fit, with inputs available here.

This machine lacks the METAR, stand-map and OPDI caches that v42-v44 read, so the
control is the clock-feature ordinary regressor (27 original + 18 clock inputs) with
v44's LightGBM settings, and the candidate adds `adsblol.ADSB_FEATURES`. Everything
else is the July fit of `tuning_seasonal.py`: train January-March and September-
December 2025, hold out July 2025, history inputs frozen at 31 December. Training
excludes Rome's schedule-recorded rows as the ordinary regressor does; every held-out
departure is scored on the raw ordinary prediction (no Rome rule, classifier or
corrections). This measures the inputs' value against a weaker parent than v44, so
it is a screen, not a Jan/Jul CV result or a ledger entry.

Run: .venv/bin/python VALIDATION/adsblol_screen.py [--seeds 42 142] [--rounds 1600]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import modeling
import overnight_run
import rome_hybrid
import seasonal
from common import CATEGORICAL
from records import RUNS, write_json

OUT = RUNS / 'v3_20261002_adsblol_screen'
V43_RESULT = RUNS / 'v3_20260925_stand_coords/main/result.json'
TRAIN = ['2025-01', '2025-02', '2025-03', '2025-09', '2025-10', '2025-11', '2025-12']
HELD_OUT, FREEZE = '2025-07', '2025-12-31'
# v44's chosen ordinary-regressor settings (VALIDATION/runs/v3_20261001_jan_jul/search/chosen.json),
# at learning rate 0.05 for a quicker screen.
V44 = dict(num_leaves=127, min_data_in_leaf=200, lambda_l2=100.0, feature_fraction=0.7, bagging_fraction=0.8,
           bagging_freq=1, path_smooth=0.0, cat_smooth=100.0, cat_l2=10.0, min_data_per_group=100,
           max_cat_threshold=16)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def fit(X, y, Xv, yv, seed, rounds, learning_rate, threads):
    params = dict(modeling.PARAMS, **V44, learning_rate=learning_rate, seed=seed, bagging_seed=seed + 1,
                  feature_fraction_seed=seed + 2, num_threads=threads, verbosity=-1)
    cats = [c for c in X.columns if c in CATEGORICAL]
    train = lgb.Dataset(X, y, categorical_feature=cats)
    curve = {}
    booster = lgb.train(params, train, num_boost_round=rounds)
    for r in range(200, rounds + 1, 200):
        curve[r] = rmse(yv, booster.predict(Xv, num_iteration=r, num_threads=threads))
    return booster, curve


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def day_bootstrap(y, a, b, days, reps=2000, seed=0):
    """Paired whole-day resampling of RMSE(b) - RMSE(a)."""
    frame = pd.DataFrame(dict(day=days, ea=(y - a) ** 2, eb=(y - b) ** 2))
    per = frame.groupby('day').agg(ea=('ea', 'sum'), eb=('eb', 'sum'), n=('ea', 'size'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), (reps, len(per)))
    ea, eb, n = per.ea.to_numpy()[idx].sum(1), per.eb.to_numpy()[idx].sum(1), per.n.to_numpy()[idx].sum(1)
    diff = np.sqrt(eb / n) - np.sqrt(ea / n)
    return [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 142])
    ap.add_argument('--rounds', type=int, default=1600)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--threads', type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads(V43_RESULT.read_text())['config'], threads=args.threads,
                  prediction_threads=args.threads, fit_workers=1)
    table = adsblol.load_features()
    log('loading movements')
    dep, arr = overnight_run.load_inputs(1.)
    td, ta, validation = seasonal.split_frames(dep, arr, dict(train=TRAIN, val=[HELD_OUT]))
    vd, va = validation[HELD_OUT]
    X45, y, cats, raw = overnight_run.prepare_training(td, ta, config)
    y = np.asarray(y, float)
    keep = ~rome_hybrid.substituted(X45, y, raw)
    V = modeling.validation_matrix(td, ta, vd, va, FREEZE, cats, config)
    if not vd.index.equals(V.index):
        raise ValueError('Held-out raw rows and inputs are misaligned')
    base_cols = rome_hybrid.regressor_features('clock')
    X = adsblol.add_adsblol(X45.loc[:, base_cols], raw, table)
    Xv = adsblol.add_adsblol(V.loc[:, base_cols], vd, table)
    X, ytr = X.loc[keep], y[keep]
    yv = vd.taxi_sec.to_numpy(float)
    has_id = vd.FLIGHT_ID_mvt.notna().to_numpy()
    days = pd.to_datetime(vd.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m-%d').to_numpy()
    airport = vd.ADEP_mvt.astype(str).to_numpy()
    log(f'training rows {len(X):,} (excluded {int((~keep).sum())}), held-out {len(Xv):,}; '
        f'ADS-B coverage train {X.ab_matched.notna().mean():.3f}, held-out {Xv.ab_matched.notna().mean():.3f}')
    preds, curves = {}, {}
    for seed in args.seeds:
        for name, cols in (('control', base_cols), ('adsblol', base_cols + adsblol.ADSB_FEATURES)):
            started = time.monotonic()
            booster, curves[f'{name}/{seed}'] = fit(X[cols], ytr, Xv[cols], yv, seed, args.rounds,
                                                    args.learning_rate, args.threads)
            preds[f'{name}/{seed}'] = booster.predict(Xv[cols], num_threads=args.threads)
            booster.save_model(str(OUT / f'{name}_seed{seed}.txt'))
            log(f'{name} seed {seed}: RMSE {rmse(yv, preds[f"{name}/{seed}"]):.3f} '
                f'({time.monotonic() - started:.0f}s)')
            if name == 'adsblol':
                gain = booster.feature_importance('gain')
                top = pd.Series(gain / gain.sum(), index=booster.feature_name()).sort_values(ascending=False)
                (OUT / f'importance_seed{seed}.json').write_text(top.round(5).to_json(indent=1))
    frame = pd.DataFrame(dict(mvt_id=vd.mvt_id.to_numpy(), airport=airport, day=days, has_id=has_id, y=yv,
                              **{k.replace('/', '_'): v for k, v in preds.items()}))
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    # Seeds averaged in squared error, as the 16-fold CV does.
    def score(name, mask=slice(None)):
        return float(np.sqrt(np.mean([np.mean((yv[mask] - preds[f'{name}/{s}'][mask]) ** 2) for s in args.seeds])))
    a = np.mean([preds[f'control/{s}'] for s in args.seeds], axis=0)
    b = np.mean([preds[f'adsblol/{s}'] for s in args.seeds], axis=0)
    result = dict(
        seeds=args.seeds, rounds=args.rounds, learning_rate=args.learning_rate,
        all=dict(control=score('control'), adsblol=score('adsblol')),
        id=dict(control=score('control', has_id), adsblol=score('adsblol', has_id)),
        per_seed={s: dict(control=rmse(yv, preds[f'control/{s}']), adsblol=rmse(yv, preds[f'adsblol/{s}']))
                  for s in args.seeds},
        interval_seed_mean=day_bootstrap(yv, a, b, days),
        by_airport={ap: dict(n=int((airport == ap).sum()), control=score('control', airport == ap),
                             adsblol=score('adsblol', airport == ap)) for ap in sorted(set(airport))},
        curves=curves)
    result['all']['change'] = result['all']['adsblol'] - result['all']['control']
    write_json(OUT / 'result.json', result)
    log(f'July all: control {result["all"]["control"]:.3f} -> ADS-B {result["all"]["adsblol"]:.3f} '
        f'({result["all"]["change"]:+.3f}, day interval {result["interval_seed_mean"]})')
    for ap, r in result['by_airport'].items():
        log(f'  {ap}: {r["control"]:.1f} -> {r["adsblol"]:.1f} ({r["adsblol"] - r["control"]:+.1f}), n={r["n"]:,}')


if __name__ == '__main__':
    main()
