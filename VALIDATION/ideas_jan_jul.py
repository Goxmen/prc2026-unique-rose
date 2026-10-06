"""Jan/Jul CV of other teams' ideas on v48's recipe: new LightGBM inputs and model changes.

Control: v46's ordinary LightGBM (arm `stand` of `adsblol_jan_jul.py`, seeds 42/142/242; its base
predictions are the `lgb_base_*` columns saved by `nn_jan_jul.py`), which v48 blends 0.4 with
the five `ple` networks of `nn_improve_jan_jul.py`. No control refit is needed.

LightGBM arms refit the ordinary regressor with v44's settings (learning rate 0.02, 5,000
rounds) on the same rows; the Rome classifier, corrections and rules are shared:
  arrtaxi   + extra_features.ARRIVAL_FEATURES  (taxi-in of earlier arrivals at the stand / runway)
  peers     + extra_features.PEER_FEATURES     (other departures' AOBT_3 clocks around takeoff)
  rwycfg    + extra_features.RUNWAY_FEATURES   (runway sets in use, takeoff bursts, runway streak)
  airport   v46's inputs, one LightGBM per airport (same settings, 2,500 rounds); scored as
            0.5 airport + 0.5 v46 global before the network blend
Added 2026-10-03 after the four arms above were scored (all four passed):
  route          + taxi_route.FEATURES (OpenStreetMap route length and turning, stand to runway end)
  combo          + the three passing input groups together (arrtaxi, peers, rwycfg)
  combo_airport  combo's inputs, one LightGBM per airport (2,500 rounds)
Each arm's columns are fixed before it runs.
Every arm saves its held-out base prediction per seed; scoring is in
`VALIDATION/tools/score_ideas_jan_jul.py`.

Run: .venv/bin/python VALIDATION/ideas_jan_jul.py --arms arrtaxi peers rwycfg [--workers 2]
No full-year fits, submissions or ledger edits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import multiprocessing
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import extra_features as ef
import modeling
import rome_hybrid
import rome_signals as signals
import seasonal
import overnight_run
import taxi_route
import tuning_seasonal as ts
from records import RUNS, sha256, write_json

SPLITS_FROM = RUNS / 'v3_20261002_adsblol_jan_jul'
OUT = RUNS / 'v3_20261003_ideas_jan_jul'
CHOSEN = RUNS / 'v3_20261001_jan_jul' / 'search' / 'chosen.json'
SEEDS = [42, 142, 242]
V46 = adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES
COMBO = ef.ARRIVAL_FEATURES + ef.PEER_FEATURES + ef.RUNWAY_FEATURES
ARMS = {'arrtaxi': ef.ARRIVAL_FEATURES, 'peers': ef.PEER_FEATURES, 'rwycfg': ef.RUNWAY_FEATURES, 'airport': [],
        'route': taxi_route.FEATURES, 'combo': COMBO, 'combo_airport': COMBO}
ROUNDS = {'airport': 2500, 'combo_airport': 2500}
PER_AIRPORT = ('airport', 'combo_airport')
DATA = {}


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def every_extra():
    cols = [c for v in ARMS.values() for c in v if c not in taxi_route.FEATURES]
    return list(dict.fromkeys(cols))


def load():
    chosen = ts.read(CHOSEN)
    table = adsblol.load_all()
    extra = ef.load()
    routes = taxi_route.load() if any(c in taxi_route.FEATURES for v in ARMS.values() for c in v) else None
    dep, arr = overnight_run.load_inputs(1.)
    items = {}
    for month in ts.SPLITS:
        d = ts.split_dir(SPLITS_FROM, month)
        meta = ts.read(d / 'meta.json')
        td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
        train, held = pd.read_pickle(d / 'after_drop.pkl'), pd.read_pickle(d / 'heldout.pkl')
        names = meta['regressor_features']
        s = rome_hybrid.substituted(train['X'], train['y'], ts.minimal_raw(train['airport'], train['X'].index))
        X = train['X'].loc[~s, names]
        ids = td.loc[X.index, 'mvt_id'].to_numpy()
        X = adsblol.add_adsblol(X, pd.DataFrame({'mvt_id': ids}, index=X.index), table, V46)
        X = ef.add(X, ids, extra, every_extra())
        V, raw = held['V'], held['raw']
        Vn = adsblol.add_adsblol(V.loc[:, names], raw[['mvt_id']].set_index(V.index), table, V46)
        Vn = ef.add(Vn, raw.mvt_id.to_numpy(), extra, every_extra())
        if routes is not None:
            rows = td.loc[X.index]
            X = taxi_route.add(X, rows.ADEP_mvt, rows.STAND_mvt, rows.RUNWAY_mvt, routes)
            Vn = taxi_route.add(Vn, raw.ADEP_mvt, raw.STAND_mvt, raw.RUNWAY_mvt, routes)
        missing = {c: round(float(Vn[c].isna().mean()), 4) for c in every_extra()
                   + (taxi_route.FEATURES if routes is not None else [])}
        items[month] = dict(meta=meta, names=names + V46, X=X, y=np.asarray(train['y'], float)[~s], Vn=Vn,
                            airport=X.airport.astype(str).to_numpy(), v_airport=Vn.airport.astype(str).to_numpy(),
                            ids=raw.mvt_id.to_numpy())
        log(f'{month}: training {X.shape}, held-out {Vn.shape}; held-out missing shares {missing}')
        del train, held, td
        gc.collect()
    DATA.update(items=items, chosen=chosen)


def train(X, y, params, rounds):
    cats = signals._categorical(X.columns) + [c for c in ef.CATEGORICAL if c in X.columns]
    return lgb.train(params, lgb.Dataset(X, y, categorical_feature=cats), num_boost_round=rounds)


def fit_arm(arm, seed):
    chosen = DATA['chosen']
    params = ts.lgb_params(chosen['settings'], seed, chosen['learning_rate'])
    rounds = ROUNDS.get(arm, chosen['rounds'])
    frames, info = [], {}
    for month in ts.SPLITS:
        item = DATA['items'][month]
        cols = item['names'] + ARMS[arm]
        started = time.monotonic()
        base = np.full(len(item['Vn']), np.nan)
        if arm in PER_AIRPORT:
            for ap in np.unique(item['airport']):
                k, v = item['airport'] == ap, item['v_airport'] == ap
                booster = train(item['X'].loc[k, cols], item['y'][k], params, rounds)
                if v.any():
                    base[v] = modeling.predict(booster, item['Vn'].loc[v, cols], threads=ts.THREADS)
            imp = {}
        else:
            booster = train(item['X'][cols], item['y'], params, rounds)
            booster.save_model(str(OUT / 'models' / f'{arm}_{month}_seed{seed}.txt'))
            base = modeling.predict(booster, item['Vn'][cols], threads=ts.THREADS)
            gain = dict(zip(booster.feature_name(), booster.feature_importance('gain')))
            total = sum(gain.values())
            imp = {c: float(gain[c] / total) for c in ARMS[arm]}
        if np.isnan(base).any():
            raise ValueError(f'{arm} {month}: held-out rows without a prediction')
        info[month] = dict(seconds=time.monotonic() - started, gain_share=imp)
        frames.append(pd.DataFrame({'mvt_id': item['ids'], 'month': month, 'base': base.astype(np.float32)}))
        log(f'{arm} seed {seed} {month}: fitted in {info[month]["seconds"]:.0f}s')
        gc.collect()
    path = OUT / 'predictions' / f'{arm}_seed{seed}.parquet'
    overnight_run.save_frame(path, pd.concat(frames, ignore_index=True))
    return arm, seed, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--arms', nargs='+', required=True, choices=sorted(ARMS))
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = ap.parse_args()
    for sub in ('models', 'predictions'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    load()
    jobs = [(a, s) for s in args.seeds for a in args.arms
            if not (OUT / 'predictions' / f'{a}_seed{s}.parquet').exists()]
    log(f'{len(jobs)} arm/seed fits to run with {args.workers} workers')
    path = OUT / 'fits.json'
    record = json.loads(path.read_text()) if path.exists() else {}
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        for arm, seed, info in pool.map(fit_arm, *zip(*jobs)) if jobs else []:
            record.setdefault(arm, {})[str(seed)] = info
            record.setdefault('_source', {})[arm] = dict(runner=sha256(Path(__file__)), features=sha256(ef.PATH),
                                                         extra_features_code=sha256(ROOT / 'VALIDATION/extra_features.py'),
                                                         columns=ARMS[arm], rounds=ROUNDS.get(arm, DATA['chosen']['rounds']))
            write_json(path, record)
            log(f'{arm} seed {seed} complete ({sum(v["seconds"] for v in info.values()) / 60:.1f} min)')


if __name__ == '__main__':
    main()
