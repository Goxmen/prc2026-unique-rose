"""Jan/Jul CV: v54's main LightGBM with the ten ADSB.lol track-shape inputs (requested 2026-10-06).

The inputs are `adsb_track.TRACK_FEATURES` (distance travelled, straight-line distance, top taxi
speed, share of time moving, time standing still, total turning, first direction, first position,
last point's distance to the take-off runway). Only the main (combo) LightGBM gets them: v54's
settings (trial LB1: v44's settings with max_cat_threshold 8, learning rate 0.02, 10,000 rounds),
seed 42, on the prepared Jan/Jul splits. Every other part of v54 (networks, per-airport LightGBMs,
CatBoost, Rome classifier, corrections, rules) is unchanged.

Scored inside v54's recipe (C5) against C5 at seed 42, at v54's 10,000 rounds; other checkpoints
are printed for information only. Decision rule, fixed before the run: upload only if both the
Jan/Jul CV and its part from flights with labels up to 1 h improve.

Run: .venv/bin/python VALIDATION/adsb_track_jan_jul.py {cv,score}
No full-year fits, submissions or ledger edits.
"""
import argparse
import gc
import time

import numpy as np
import pandas as pd

import hpo_jan_jul as hpo          # sets sys.path for VALIDATION, pipeline and tools
import adsb_track as at
import adsblol
import modeling
import overnight_run
import tuning_seasonal as ts
from records import RUNS, write_json

OUT = RUNS / 'v3_20261006_adsb_track_jan_jul'
PRED = OUT / 'predictions' / 'track_lgb_s42.parquet'
SEED, ROUNDS, STEP, LR, THREADS = 42, 10_000, 1_000, 0.02, 14
SETTINGS = dict(hpo.V44, max_cat_threshold=8)                     # trial LB1, as v54
BODY = 3600


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def cv():
    track = at.load()
    plain = adsblol.add_adsblol
    v46 = adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES

    def with_track(X, raw, table, features=adsblol.ADSB_FEATURES):
        """ideas_jan_jul.load adds v46's ADSB.lol inputs to both matrices; append the track inputs there."""
        X = plain(X, raw, table, features)
        return plain(X, raw, track, at.TRACK_FEATURES) if list(features) == v46 else X
    adsblol.add_adsblol = with_track
    ideas = hpo.load_trees()
    adsblol.add_adsblol = plain
    params = dict(ts.lgb_params(SETTINGS, SEED, LR), num_threads=THREADS)
    marks = list(range(STEP, ROUNDS + 1, STEP))
    frames, info = [], {}
    for month in ts.SPLITS:
        item = ideas.DATA['items'][month]
        cols = item['names'] + ideas.COMBO + at.TRACK_FEATURES
        missing = [c for c in cols if c not in item['X'].columns or c not in item['Vn'].columns]
        if missing:
            raise ValueError(f'{month}: inputs missing {missing}')
        seen = {c: round(float(item['Vn'][c].notna().mean()), 3) for c in ('ab_path_m', 'ab_max_gs')}
        started = time.monotonic()
        booster = ideas.train(item['X'][cols], item['y'], params, ROUNDS)
        out = {f'c{c}': modeling.predict(booster, item['Vn'][cols], num_iteration=c, threads=THREADS).astype(np.float32)
               for c in marks}
        gain = dict(zip(booster.feature_name(), booster.feature_importance('gain')))
        total = sum(gain.values())
        info[month] = dict(minutes=(time.monotonic() - started) / 60, held_out_seen=seen,
                           gain_share={c: float(gain[c] / total) for c in at.TRACK_FEATURES})
        log(f'{month}: {info[month]["minutes"]:.1f} min; track gain share '
            f'{sum(info[month]["gain_share"].values()):.3f}')
        frames.append(pd.DataFrame({'mvt_id': item['ids'], 'month': month, **out}))
        del booster
        gc.collect()
    PRED.parent.mkdir(parents=True, exist_ok=True)
    overnight_run.save_frame(PRED, pd.concat(frames, ignore_index=True))
    write_json(OUT / 'fit.json', dict(settings=SETTINGS, learning_rate=LR, rounds=ROUNDS, seed=SEED,
                                      features=at.TRACK_FEATURES, months=info))


def score():
    sc = hpo.Scorer()
    wins = {p: hpo.winner(p) for p in ['rome', 'nn', 'catboost', 'lgb', 'airport']}
    rome = sc.frame('rome', wins['rome']['trial'], SEED)
    rome = {m: rome[m][wins['rome']['checkpoint']].to_numpy(float) for m in ts.SPLITS}
    t = wins['nn']['trial']
    fr = [sc.frame('nn', t, SEED), sc.frame('nn', t + '_k34', SEED)] + [sc.frame('nn', t, s) for s in hpo.SEEDS[1:]]
    nn = {m: np.concatenate([f[m].to_numpy(float) for f in fr], axis=1).mean(1) for m in ts.SPLITS}
    trees = {p: hpo.member_preds(sc, p, wins[p], SEED) for p in ['catboost', 'lgb', 'airport']}
    track = pd.read_parquet(PRED)

    def se(lgb):
        return {m: sc.se(m, SEED, nn=nn[m], p=rome[m], combo=lgb[m], combo_airport=trees['airport'][m],
                         catboost=trees['catboost'][m]) for m in ts.SPLITS}

    def cvs(e):
        return float(np.mean([np.sqrt(e[m].mean()) for m in ts.SPLITS]))

    def split(a, b):
        body = tail = 0.0
        for m in ts.SPLITS:
            y, d, r = sc.d[m]['y'], b[m] - a[m], np.sqrt(a[m].mean())
            body += d[y <= BODY].sum() / len(y) / (2 * r) / len(ts.SPLITS)
            tail += d[y > BODY].sum() / len(y) / (2 * r) / len(ts.SPLITS)
        return body, tail

    base = se(trees['lgb'])
    res = dict(control=dict(cv=cvs(base), **{m: float(np.sqrt(base[m].mean())) for m in ts.SPLITS}), checkpoints={})
    for c in [f'c{k}' for k in range(STEP, ROUNDS + 1, STEP)]:
        lgb = {}
        for m in ts.SPLITS:
            g = track[track.month == m]
            if not np.array_equal(g.mvt_id.to_numpy(), sc.d[m]['ids']):
                raise ValueError(f'{m}: rows differ')
            lgb[m] = g[c].to_numpy(float)
        e = se(lgb)
        body, tail = split(base, e)
        res['checkpoints'][c] = dict(cv=cvs(e), change=cvs(e) - res['control']['cv'], body=body, tail=tail,
                                     **{m: float(np.sqrt(e[m].mean())) for m in ts.SPLITS})
        if c == f'c{ROUNDS}':
            month = np.concatenate([np.full(len(sc.d[m]['y']), m) for m in ts.SPLITS])
            day = np.concatenate([sc.d[m]['day'] for m in ts.SPLITS])
            flat = lambda x: np.concatenate([x[m] for m in ts.SPLITS])
            res['bootstrap'] = ts.paired_bootstrap(day, month, flat(base), flat(e))
    chosen = res['checkpoints'][f'c{ROUNDS}']
    res['passes'] = bool(chosen['change'] < 0 and chosen['body'] < 0)
    write_json(OUT / 'score.json', res)
    log(f"C5 seed 42 {res['control']['cv']:.4f}")
    for c, r in res['checkpoints'].items():
        log(f"{c:>7s} {r['cv']:.4f} change {r['change']:+.3f} = body {r['body']:+.3f} + tail {r['tail']:+.3f} "
            f"(Jan {r['2025-01']:.2f}, Jul {r['2025-07']:.2f})")
    log(f"at {ROUNDS} rounds: bootstrap {res['bootstrap']}; passes: {res['passes']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=['cv', 'score'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cv() if args.action == 'cv' else score()


if __name__ == '__main__':
    main()
