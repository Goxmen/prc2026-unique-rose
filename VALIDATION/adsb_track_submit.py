"""Prepare a leaderboard file: v54 with its main LightGBM refitted with the ADSB.lol track inputs.

The ten inputs are `adsb_track.TRACK_FEATURES`; Jan/Jul CV of this change is in
`adsb_track_jan_jul.py` (run `v3_20261006_adsb_track_jan_jul`). The main (combo) LightGBM is refitted
on all of 2025 with v54's settings (trial LB1, learning rate 0.02, 10,000 rounds, seed 42) on v51's
full-year rows plus the track inputs. Networks, per-airport LightGBMs, CatBoost, the Rome
probabilities, corrections and rules are v54's saved parts, unchanged:

    base = 0.6 x networks + 0.4 x mean(track LightGBM, per-airport LightGBMs, CatBoost)

Checks: the training rows' flight IDs are recovered from the training departures and confirmed by
v46's ADSB.lol inputs already in the matrix; v54's final predictions are reproduced from its parts.

Run: .venv/bin/python VALIDATION/adsb_track_submit.py fit
     .venv/bin/python VALIDATION/adsb_track_submit.py prepare --version 58
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import hpo_jan_jul as hpo          # sets sys.path for VALIDATION, pipeline and tools
import adsb_track as at
import adsblol
import ideas_jan_jul as ideas
import ideas_submit as ids
import modeling
import nn_improve_submit as nis
import overnight_run
from records import RUNS, runtime_identity, sha256, source_snapshot, write_json
from rome_hybrid_submit import MONTHS, now, submission
from submit import fill_template
import tuning_seasonal as ts

OUT = RUNS / 'v3_20261006_adsb_track_submit'
WORK = OUT / 'work'
V54 = RUNS / 'v3_20261006_hpo_submit' / 'submission_v54'
V54_SHA = '367d4d19f365c497e7a408713513a4813a633e02d5f7f658233f68a53d77970e'
CV_RUN = RUNS / 'v3_20261006_adsb_track_jan_jul'
SEED, ROUNDS, LR, THREADS = 42, 10_000, 0.02, 14
SETTINGS = dict(hpo.V44, max_cat_threshold=8)                     # trial LB1, as v54
V46 = adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


def fit():
    started = time.monotonic()
    data = pd.read_pickle(ids.WORK / 'inputs.pkl')
    Xn, Vn = data['Xn'], data['Vn']
    dep, _ = overnight_run.load_inputs(1.)
    train_ids = dep.loc[Xn.index, 'mvt_id'].to_numpy()
    del dep
    table, track = adsblol.load_all(), at.load()
    if not same(Xn[V46].to_numpy(float), table.loc[train_ids, V46].to_numpy(float)):
        raise ValueError('Recovered training flight IDs do not match the ADSB.lol inputs already in the matrix')
    if not same(Vn[V46].to_numpy(float), table.loc[data['rank_ids'], V46].to_numpy(float)):
        raise ValueError('Ranking rows do not match the ADSB.lol inputs already in the matrix')
    log('training and ranking flight IDs confirmed by v46 ADSB.lol inputs')
    cols = data['names'] + ids.COMBO
    X = adsblol.add_adsblol(Xn[cols], pd.DataFrame({'mvt_id': train_ids}, index=Xn.index), track, at.TRACK_FEATURES)
    V = adsblol.add_adsblol(Vn[cols], pd.DataFrame({'mvt_id': data['rank_ids']}, index=Vn.index), track,
                            at.TRACK_FEATURES)
    seen = {c: dict(training=round(float(X[c].notna().mean()), 3), ranking=round(float(V[c].notna().mean()), 3))
            for c in ('ab_path_m', 'ab_max_gs')}
    params = dict(ts.lgb_params(SETTINGS, SEED, LR), num_threads=THREADS)
    with threadpool_limits(limits=1, user_api='blas'):
        booster = ideas.train(X, data['yn'], params, ROUNDS)
        WORK.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(WORK / 'track_lgb.txt'))
        pred = modeling.predict(booster, V, threads=THREADS)
    if not np.isfinite(pred).all():
        raise ValueError('ranking rows without a prediction')
    gain = dict(zip(booster.feature_name(), booster.feature_importance('gain')))
    total = sum(gain.values())
    path = WORK / 'track_lgb.parquet'
    overnight_run.save_frame(path, pd.DataFrame({'mvt_id': data['rank_ids'], 'pred': pred.astype(np.float32)}))
    write_json(WORK / 'track_lgb.json', dict(settings=SETTINGS, learning_rate=LR, rounds=ROUNDS, seed=SEED,
                                             features=at.TRACK_FEATURES, seen=seen,
                                             gain_share={c: float(gain[c] / total) for c in at.TRACK_FEATURES},
                                             minutes=(time.monotonic() - started) / 60, finished_at=now(),
                                             artifacts=overnight_run.artifact_hashes(WORK, path)))
    log(f'track LightGBM done in {(time.monotonic() - started) / 60:.1f} min; seen {seen}')


def prepare(version):
    started = time.monotonic()
    directory = OUT / f'submission_v{version}'
    if directory.exists() or submission(version).exists():
        raise ValueError(f'v{version} is already prepared; inspect it before continuing')
    if sha256(submission(54)) != V54_SHA:
        raise ValueError('v54 file differs from the scored upload')
    data = pd.read_pickle(ids.WORK / 'inputs.pkl')
    parameters, fitted = data['parameters'], data['fitted']
    part = pd.read_parquet(WORK / 'track_lgb.parquet').set_index('mvt_id').pred
    directory.mkdir(parents=True)
    pieces, summary = [], {}
    for m in MONTHS:
        d = data['months'][m]
        saved = pd.read_parquet(V54 / f'ranking_{m}.parquet')
        if not np.array_equal(saved.mvt_id.to_numpy(), d['raw'].mvt_id.to_numpy()):
            raise ValueError(f'{m}: v54 rows differ from the saved inputs')
        nn, p = saved.nn_base.to_numpy(float), saved.p.to_numpy(float)
        lgb_old, airport, cat = (saved[f'{q}_part'].to_numpy(float) for q in ('lgb', 'airport', 'catboost'))
        gbm_old = np.mean([lgb_old, airport, cat], axis=0)
        if np.max(np.abs(gbm_old - saved.gbm_base.to_numpy(float))) > 1e-3:
            raise ValueError(f'{m}: v54 trees are not the mean of its three parts')
        frame, _ = nis.final(0.6 * nn + 0.4 * saved.gbm_base.to_numpy(float), p, d['V'], d['raw'], parameters, fitted)
        if np.max(np.abs(frame.pred.to_numpy(float) - saved.pred.to_numpy(float))) > 1e-6:
            raise ValueError(f'{m}: v54 not reproduced')
        track = part.reindex(saved.mvt_id).to_numpy(float)
        gbm = np.mean([track, airport, cat], axis=0)
        base = 0.6 * nn + 0.4 * gbm
        frame, masks = nis.final(base, p, d['V'], d['raw'], parameters, fitted)
        frame['nn_base'], frame['gbm_base'], frame['blend_base'], frame['month'] = nn, gbm, base, m
        frame['lgb_part'], frame['airport_part'], frame['catboost_part'] = track, airport, cat
        change = frame.pred.to_numpy(float) - saved.pred.to_numpy(float)
        summary[m] = dict(rows=len(frame), lgb_corr_vs_v54=float(np.corrcoef(track, lgb_old)[0, 1]),
                          lgb_mean_change=float((track - lgb_old).mean()), mean_change=float(change.mean()),
                          mean_abs_change=float(np.abs(change).mean()), max_abs_change=float(np.abs(change).max()),
                          correction_masks=masks)
        overnight_run.save_frame(directory / f'ranking_{m}.parquet', frame)
        pieces.append(frame)
    log('v54 reproduced from its saved parts')
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(data['rank_ids']).reset_index()
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(nis.TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    nis.transfer.validate_submission(table, template)
    path = submission(version)
    pq.write_table(table, path)
    score = ts.read(CV_RUN / 'score.json')
    manifest = dict(version=version, state='prepared', file=str(path.relative_to(hpo.ROOT)), sha256=sha256(path),
                    parent='v54', change='main LightGBM refitted with the ten ADSB.lol track inputs',
                    fit=ts.read(WORK / 'track_lgb.json'), jan_jul_cv=score, summary=summary,
                    runtime=runtime_identity(), source=source_snapshot(directory),
                    minutes=(time.monotonic() - started) / 60, prepared_at=now())
    write_json(directory / f'submission_v{version}.json', manifest)
    log(json.dumps(summary, indent=1))
    log(f'prepared {path}; not uploaded')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=['fit', 'prepare'])
    ap.add_argument('--version', type=int)
    args = ap.parse_args()
    fit() if args.action == 'fit' else prepare(args.version)


if __name__ == '__main__':
    main()
