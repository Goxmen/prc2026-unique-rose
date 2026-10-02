"""Jan/Jul CV of the ADSB.lol inputs on v44 (the two fits of tuning_seasonal.py).

Control: v44's ordinary regressor (search/chosen.json: B00 settings, learning rate 0.02,
5,000 rounds) on splits prepared by `tuning_seasonal.py prepare` on this machine.
Candidate: the same plus `adsblol.ADSB_FEATURES`, in the ordinary regressor only. The
Rome classifier, corrections and rules are the prepared ones, shared by both arms, and
every held-out flight is scored on the final prediction, as the search scored v44.

Both arms are refitted here at seeds 42/142/242, so each pair comes from one machine.
v44's saved Apple M3 predictions (search/B00 and S_chosen_142/242 of
v3_20261001_jan_jul) are compared with the local control as a reproduction check only.

Arm `stand` adds `adsblol.STAND_FEATURES` (stand-exit witness from the public stand map) on top;
its comparison with the control is written to result_stand.json.

Run: .venv/bin/python VALIDATION/adsblol_jan_jul.py [--workers 2] [--arms control adsblol] [--seeds 42 142 242]
No 16-fold CV, full-year fits, submissions or ledger edits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import modeling
import overnight_run
import rome_hybrid
import seasonal
import tuning_seasonal as ts
from records import RUNS, sha256, write_json

OUT = RUNS / 'v3_20261002_adsblol_jan_jul'
MAC = RUNS / 'v3_20261001_jan_jul' / 'search'
SEEDS = [42, 142, 242]
MARKS = [1000, 2000, 3000, 4000, 5000]
ARMS = {'control': [], 'adsblol': adsblol.ADSB_FEATURES, 'stand': adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES,
        'cov': adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES + adsblol.COVERAGE_FEATURES}
DATA = {}


def load():
    """Prepared splits with ADS-B inputs attached; training rows mapped back to movement IDs."""
    chosen = ts.read(MAC / 'chosen.json')
    table = adsblol.load_all()
    every = ARMS['cov']
    dep, arr = overnight_run.load_inputs(1.)
    items = {}
    for month in ts.SPLITS:
        d = ts.split_dir(OUT, month)
        meta = ts.read(d / 'meta.json')
        td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
        train, held = pd.read_pickle(d / 'after_drop.pkl'), pd.read_pickle(d / 'heldout.pkl')
        names = meta['regressor_features']
        s = rome_hybrid.substituted(train['X'], train['y'], ts.minimal_raw(train['airport'], train['X'].index))
        X = train['X'].loc[~s, names]
        ids = pd.DataFrame({'mvt_id': td.loc[X.index, 'mvt_id'].to_numpy()}, index=X.index)
        X = adsblol.add_adsblol(X, ids, table, every)
        V, raw = held['V'], held['raw']
        Vn = adsblol.add_adsblol(V.loc[:, names], raw[['mvt_id']].set_index(V.index), table, every)
        items[month] = dict(meta=meta, names=names, X=X, y=train['y'][~s], V=V, Vn=Vn, raw=raw, p=held['p'])
        log(f'{month}: training {X.shape}, held-out {Vn.shape}; ADS-B rows missing: training '
            f'{int(X.ab_matched.isna().sum())}, held-out {int(Vn.ab_matched.isna().sum())}')
        del train, held, td
        gc.collect()
    DATA.update(items=items, chosen=chosen)


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def fit_arm(arm, seed):
    chosen = DATA['chosen']
    params = ts.lgb_params(chosen['settings'], seed, chosen['learning_rate'])
    preds, seconds = {k: [] for k in MARKS}, {}
    for month in ts.SPLITS:
        item = DATA['items'][month]
        cols = item['names'] + ARMS[arm]
        started = time.monotonic()
        booster = ts.train(item['X'][cols], item['y'], params, chosen['rounds'])
        seconds[month] = time.monotonic() - started
        booster.save_model(str(OUT / 'models' / f'{arm}_{month}_seed{seed}.txt'))
        for k in MARKS:
            base = modeling.predict(booster, item['Vn'][cols], num_iteration=k, threads=ts.THREADS)
            preds[k].append(ts.final_prediction(base, item['p'], item['V'], item['raw'],
                                                item['meta']['parameters'], item['meta']['corrections']))
        log(f'{arm} seed {seed} {month}: fitted in {seconds[month]:.0f}s')
        del booster
        gc.collect()
    frame = pd.DataFrame({'mvt_id': np.concatenate([DATA['items'][m]['raw'].mvt_id.to_numpy() for m in ts.SPLITS])})
    for k in MARKS:
        frame[f'r{k}'] = np.concatenate(preds[k]).astype(np.float32)
    path = OUT / 'predictions' / f'{arm}_seed{seed}.parquet'
    overnight_run.save_frame(path, frame)
    return arm, seed, seconds


def evaluate(arm='adsblol', seeds=SEEDS):
    items = DATA['items']
    raw = [items[m]['raw'] for m in ts.SPLITS]
    y = np.concatenate([r.taxi_sec.to_numpy(float) for r in raw])
    has_id = np.concatenate([r.FLIGHT_ID_mvt.notna().to_numpy() for r in raw])
    month = np.concatenate([np.full(len(r), m) for m, r in zip(ts.SPLITS, raw)])
    day = np.concatenate([pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy() for r in raw])
    airport = np.concatenate([r.ADEP_mvt.astype(str).to_numpy() for r in raw])
    mvt = np.concatenate([r.mvt_id.to_numpy() for r in raw])
    final = f'r{DATA["chosen"]["rounds"]}'
    pred = {(a, s): pd.read_parquet(OUT / 'predictions' / f'{a}_seed{s}.parquet') for a in ('control', arm)
            for s in seeds}
    if any(not np.array_equal(f.mvt_id.to_numpy(), mvt) for f in pred.values()):
        raise ValueError('Prediction rows differ from the prepared held-out rows')
    out = dict(chosen=DATA['chosen'], arm=arm, seeds=seeds, by_seed={}, reproduction={}, by_airport={})
    mac = {42: 'B00', 142: 'S_chosen_142', 242: 'S_chosen_242'}
    for s in seeds:
        a, b = pred['control', s][final].to_numpy(float), pred[arm, s][final].to_numpy(float)
        out['by_seed'][s] = dict(control=ts.scores(y, a, has_id, month), adsblol=ts.scores(y, b, has_id, month),
                                 change_all=ts.paired_bootstrap(day, month, (y - a) ** 2, (y - b) ** 2),
                                 change_id=ts.paired_bootstrap(day[has_id], month[has_id], ((y - a) ** 2)[has_id],
                                                               ((y - b) ** 2)[has_id]))
        saved = pd.read_parquet(MAC / f'{mac[s]}.parquet')
        if not np.array_equal(saved.mvt_id.to_numpy(), mvt):
            raise ValueError(f'{mac[s]}: row order differs')
        m = saved[final].to_numpy(float)
        out['reproduction'][s] = dict(mac=ts.scores(y, m, has_id, month), local=out['by_seed'][s]['control'],
                                      max_abs_prediction_difference=float(np.max(np.abs(m - a))))
    # Seeds averaged in squared error within each fit, as the search's headline did.
    se = {a: np.mean([(y - pred[a, s][final].to_numpy(float)) ** 2 for s in seeds], axis=0) for a in ('control', arm)}
    se['adsblol'] = se.pop(arm) if arm != 'adsblol' else se['adsblol']   # report keys stay control/adsblol
    def level(e, mask):
        per = [np.sqrt(e[mask & (month == m)].mean()) for m in ts.SPLITS]
        return dict(zip(ts.SPLITS, map(float, per)), average=float(np.mean(per)))
    every = np.ones(len(y), bool)
    out['seed_average'] = {arm: dict(all=level(se[arm], every), id=level(se[arm], has_id)) for arm in se}
    out['seed_average_change'] = dict(all=ts.paired_bootstrap(day, month, se['control'], se['adsblol']),
                                      id=ts.paired_bootstrap(day[has_id], month[has_id], se['control'][has_id],
                                                             se['adsblol'][has_id]))
    for ap in sorted(set(airport)):
        k = airport == ap
        out['by_airport'][ap] = {m: dict(n=int((k & (month == m)).sum()),
                                         control=float(np.sqrt(se['control'][k & (month == m)].mean())),
                                         adsblol=float(np.sqrt(se['adsblol'][k & (month == m)].mean())))
                                 for m in ts.SPLITS}
    out['source_sha256'] = dict(runner=sha256(Path(__file__)), adsblol=sha256(ROOT / 'VALIDATION/adsblol.py'),
                                features=sha256(adsblol.FEATURES_PATH), stand_features=sha256(adsblol.STAND_PATH))
    write_json(OUT / ('result.json' if arm == 'adsblol' and seeds == SEEDS else f'result_{arm}_{len(seeds)}seed.json'), out)
    c = out['seed_average_change']['all']
    log(f'Jan/Jul CV: control {out["seed_average"]["control"]["all"]["average"]:.4f} -> '
        f'ADS-B {out["seed_average"]["adsblol"]["all"]["average"]:.4f} ({c["change"]:+.4f}, {c["ci95"]}, {c["decision"]})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--arms', nargs='+', default=['control', 'adsblol'], choices=sorted(ARMS))
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = ap.parse_args()
    for sub in ('models', 'predictions'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    load()
    jobs = [(a, s) for s in args.seeds for a in args.arms
            if not (OUT / 'predictions' / f'{a}_seed{s}.parquet').exists()]
    log(f'{len(jobs)} arm/seed fits to run with {args.workers} workers')
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        for arm, seed, seconds in pool.map(fit_arm, *zip(*jobs)) if jobs else []:
            log(f'{arm} seed {seed} complete ({sum(seconds.values()) / 60:.1f} min)')
    for arm in [a for a in args.arms if a != 'control']:
        evaluate(arm, args.seeds)


if __name__ == '__main__':
    main()
