"""Prepare a Rome-only leaderboard check: v51 with the Rome classifier's searched settings.

Asked 2026-10-05 as a fast leaderboard check while the full search (`hpo_jan_jul.py`) runs.
Only the LIRF substitution classifier changes. Its settings are trial R20 of the Rome search
(`VALIDATION/runs/v3_20261005_hpo_jan_jul/rome.json`): 127 leaves, 60% of inputs per tree,
at least 50 rows per leaf, learning rate 0.02, 1,700 rounds (rounds chosen on Jan/Jul).
Jan/Jul CV on C4: 298.2138 -> 295.8929 (-2.32 [-4.96, -0.42]).

Everything else is v51 unchanged: its saved blend base per ranking flight (0.6 networks + 0.4
GBMs, `submission_v51/ranking_*.parquet`), and v46's corrections, late estimate and four-hour
rule (`nn_improve_submit.final`). The classifier is refitted on all 2025 LIRF departures with the
inputs v46's classifier used. Checks, in order:
  1. v51's final predictions are reproduced from its saved base and probabilities;
  2. v46's classifier settings refitted here reproduce v46's ranking probabilities;
  3. only LIRF flights change.
`--hold-v51 ID ...` keeps the named movements at v51's prediction. v53 holds ITA flight ITY108
(202681908, January, no flight ID, 4.95 h after schedule): the new probability falls below the
5% at which v46's rule-confidence correction swaps the four-hour rule's 17,816 s for an ordinary
taxi (1,585 s). That one flight could move the leaderboard by about 1.6 s either way, so the user
chose to hold it and let the leaderboard measure the classifier change elsewhere. A test, not
a recipe.
Writes submissions/unique-rose_v<N>.parquet and a manifest; uploading is a separate decision.

Run: .venv/bin/python VALIDATION/rome_hpo_submit.py --version 53 --hold-v51 202681908
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import nn_improve_submit as nis
import overnight_run
from records import RUNS, runtime_identity, sha256, source_snapshot, write_json
import rome_hybrid
from rome_hybrid_submit import KIND, MONTHS, SEED, now, submission
from submit import fill_template
import tuning_seasonal as ts

cv, features, signals, weather, v43s = nis.cv, nis.features, nis.signals, nis.weather, nis.v43s
IDEAS_SUBMIT = RUNS / 'v3_20261003_ideas_submit'
V51 = IDEAS_SUBMIT / 'submission_v51'
HPO = RUNS / 'v3_20261005_hpo_jan_jul'
OUT = RUNS / 'v3_20261005_rome_hpo_submit'
TRIAL, ROUNDS = 'R20', 1700


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def classifier(X, s, overrides, rounds):
    params = dict(rome_hybrid.CLASSIFIER_PARAMS, **overrides, seed=SEED, bagging_seed=SEED + 1,
                  feature_fraction_seed=SEED + 2, num_threads=ts.THREADS)
    return signals._train(params, X, s, rounds, f'classifier {overrides or "v46"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--hold-v51', type=float, nargs='*', default=[], help='movement IDs kept at v51')
    args = ap.parse_args()
    hold = np.asarray(args.hold_v51, float)
    started = time.monotonic()
    directory = OUT / f'submission_v{args.version}'
    if directory.exists() or submission(args.version).exists():
        raise ValueError(f'v{args.version} is already prepared; inspect it before continuing')
    hpo = json.loads((HPO / 'rome.json').read_text())
    overrides = hpo['R20']
    if hpo['scores'][TRIAL]['best_checkpoint'] != f'c{ROUNDS}' or hpo['trials'][TRIAL]['overrides'] != overrides:
        raise ValueError('Rome search result differs from the one this script was written for')
    if list(cv.CLASSIFIER) != list(ts.CLASSIFIER):
        raise ValueError('Classifier inputs differ between validation and submission recipes')

    data = pd.read_pickle(IDEAS_SUBMIT / 'work' / 'inputs.pkl')
    parameters, fitted = data['parameters'], data['fitted']
    saved = {m: pd.read_parquet(V51 / f'ranking_{m}.parquet') for m in MONTHS}
    # 1. v51 reproduced from its saved base and v46's probabilities
    for m in MONTHS:
        d = data['months'][m]
        if not np.array_equal(saved[m].mvt_id.to_numpy(), d['raw'].mvt_id.to_numpy()):
            raise ValueError(f'{m}: v51 rows differ from the saved inputs')
        frame, _ = nis.final(saved[m].blend_base.to_numpy(float), d['p'], d['V'], d['raw'], parameters, fitted)
        gap = float(np.max(np.abs(frame.pred.to_numpy(float) - saved[m].pred.to_numpy(float))))
        if gap > 1e-6:
            raise ValueError(f'{m}: v51 not reproduced ({gap})')
    log('v51 final predictions reproduced from saved parts')

    # Full-year training rows with v46's classifier inputs (as ideas_submit.inputs builds them)
    v46 = ts.read(nis.V46 / 'submission_v46.json')
    config = v46['identity']['config']
    expected = ts.read(v43s.PARENT_MANIFEST)['model']['parameters']
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr = overnight_run.load_inputs(1.)
    X, y, cats, raw, params, membership = cv.training_inputs(dep, arr, config, reports, grid, expected)
    if params != parameters:
        raise ValueError('Training parameters differ from v51')
    del dep, arr
    gc.collect()
    lirf = raw.ADEP_mvt.to_numpy() == 'LIRF'
    XL, rawL, yL = X.loc[lirf], raw.loc[lirf], np.asarray(y, float)[lirf]
    del X, raw
    gc.collect()
    M = signals.classifier_matrix(XL, rawL, KIND, cv.CLASSIFIER)
    s = rome_hybrid.substituted(XL, yL, rawL).astype(float)
    log(f'LIRF training rows {len(M):,}, schedule-recorded {int(s.sum()):,}')

    def probabilities(booster):
        return {m: signals.substitution_probability(booster, data['months'][m]['V'], data['months'][m]['raw'],
                                                    KIND, ts.THREADS, cv.CLASSIFIER) for m in MONTHS}

    # 2. v46's classifier settings reproduce v46's ranking probabilities
    old = probabilities(classifier(M, s, {}, rome_hybrid.CLASSIFIER_ROUNDS))
    repro = max(float(np.max(np.abs(old[m] - data['months'][m]['p']))) for m in MONTHS)
    if repro > 1e-9:
        raise ValueError(f'v46 classifier not reproduced ({repro})')
    log('v46 classifier reproduced')
    new = probabilities(classifier(M, s, overrides, ROUNDS))

    directory.mkdir(parents=True)
    write_json(directory / 'refit.json', dict(trial=TRIAL, overrides=overrides, rounds=ROUNDS, started_at=now(),
                                              runtime=runtime_identity(), source=source_snapshot(directory)))
    pieces, summary = [], {}
    for m in MONTHS:
        d = data['months'][m]
        frame, masks = nis.final(saved[m].blend_base.to_numpy(float), new[m], d['V'], d['raw'], parameters, fitted)
        frame['p_v51'], frame['blend_base'], frame['month'] = d['p'], saved[m].blend_base.to_numpy(float), m
        held = frame.mvt_id.isin(hold).to_numpy()
        frame['held_at_v51'] = held
        frame.loc[held, 'pred'] = saved[m].pred.to_numpy(float)[held]
        lirf_m = d['raw'].ADEP_mvt.to_numpy() == 'LIRF'
        change = frame.pred.to_numpy(float) - saved[m].pred.to_numpy(float)
        if np.any(change[~lirf_m] != 0):                                  # 3. only LIRF flights change
            raise ValueError(f'{m}: flights outside LIRF changed')
        summary[m] = dict(rows=len(frame), lirf_rows=int(lirf_m.sum()), changed=int((change != 0).sum()),
                          held_at_v51=frame.loc[held, 'mvt_id'].tolist(),
                          mean_change_lirf=float(change[lirf_m].mean()),
                          mean_abs_change_lirf=float(np.abs(change[lirf_m]).mean()),
                          max_abs_change=float(np.abs(change).max()),
                          mean_p=dict(v51=float(d['p'][lirf_m].mean()), new=float(new[m][lirf_m].mean())),
                          correction_masks=masks)
        overnight_run.save_frame(directory / f'ranking_{m}.parquet', frame)
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(data['rank_ids']).reset_index()
    if frame.held_at_v51.sum() != len(hold):
        raise ValueError('A held movement ID was not found')
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(nis.TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    nis.transfer.validate_submission(table, template)
    path = submission(args.version)
    pq.write_table(table, path)
    manifest = dict(version=args.version, state='prepared', file=str(path.relative_to(ROOT)), sha256=sha256(path),
                    parent='v51', change='Rome classifier settings from the Jan/Jul search (trial R20)',
                    held_at_v51=hold.tolist(),
                    jan_jul_cv=dict(c4=hpo['control']['cv'], r20_seed42=hpo['scores'][TRIAL]['cv']),
                    reproduction=dict(v51=True, v46_classifier=repro), summary=summary,
                    minutes=(time.monotonic() - started) / 60, prepared_at=now())
    write_json(directory / f'submission_v{args.version}.json', manifest)
    log(json.dumps(summary, indent=1))
    log(f'prepared {path} ({manifest["minutes"]:.1f} min); not uploaded')


if __name__ == '__main__':
    main()
