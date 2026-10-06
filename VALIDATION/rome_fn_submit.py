"""Prepare a leaderboard file: v54 with the Rome classifier given the flight-number copy input.

Arm F of `rome_cost_jan_jul.py` passed on Jan/Jul (three seeds, C4 parts): 296.5761 -> 295.0409
(-1.54 [-2.44, -0.75], every seed better; January -0.44, July -2.63), at 1,700 rounds like R20.
The classifier keeps R20's settings and inputs and adds two:
  fn_copy_rate  share of the flight number's 2025 LIRF departures whose label equals takeoff minus
                schedule; a training row only sees the other months; pulled toward the overall share
                with weight 10 (`rome_cost_jan_jul.copy_rate`)
  fn_copy_n     how many of those departures there were
Everything else is v54 unchanged: its saved blend base per ranking flight and v46's corrections, late
estimate and four-hour rule (`nn_improve_submit.final`). Checks, in order:
  1. v54's final predictions are reproduced from its saved base and Rome probabilities;
  2. R20 refitted here reproduces v54's Rome probabilities;
  3. only LIRF flights change, and the four-hour rule still fires on the same 43 flights.
Flight numbers never seen at LIRF in 2025 keep v54's probability (default; `--no-fallback` turns it
off). In July 2026, 48% of LIRF departures have such numbers (renumbered or new routes), against 2-3% in
the practice months, and the new input raised their copy chance for no reason. On Jan/Jul this fallback
scores 295.3014 against F's 295.0409 and the control's 296.7497 (all at 1,700 rounds, three seeds).
Flights whose prediction moves by more than an hour are listed in the manifest: one wrong flip of that
size can move the leaderboard by a second or more (see SUBMISSIONS.md, row 56).
Writes submissions/unique-rose_v<N>.parquet and a manifest; uploading is a separate decision.

Run: .venv/bin/python VALIDATION/rome_fn_submit.py --version 56
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
import rome_cost_jan_jul as rc
import rome_hybrid
from rome_hybrid_submit import KIND, MONTHS, SEED, now, submission
from submit import fill_template
import tuning_seasonal as ts

cv, features, signals, weather, v43s = nis.cv, nis.features, nis.signals, nis.weather, nis.v43s
IDEAS_SUBMIT = RUNS / 'v3_20261003_ideas_submit'
V54 = RUNS / 'v3_20261006_hpo_submit' / 'submission_v54'
V54_SHA = '367d4d19f365c497e7a408713513a4813a633e02d5f7f658233f68a53d77970e'
OUT = RUNS / 'v3_20261006_rome_fn_submit'
ROUNDS = 1700


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def classifier(X, s, rounds):
    params = dict(rome_hybrid.CLASSIFIER_PARAMS, **rc.R20, seed=SEED, bagging_seed=SEED + 1,
                  feature_fraction_seed=SEED + 2, num_threads=ts.THREADS)
    return signals._train(params, X, s, rounds, 'classifier R20' + (' + flight number' if 'fn_copy_rate' in X else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--no-fallback', action='store_true', help='use the new probability for unseen flight numbers too')
    args = ap.parse_args()
    started = time.monotonic()
    directory = OUT / f'submission_v{args.version}'
    if directory.exists() or submission(args.version).exists():
        raise ValueError(f'v{args.version} is already prepared; inspect it before continuing')
    if sha256(submission(54)) != V54_SHA:
        raise ValueError('v54 file differs from the scored upload')
    result = json.loads((rc.OUT / 'result.json').read_text())
    if result['F']['best'] != f'c{ROUNDS}' or result['F']['change_vs_C']['decision'] != 'improvement':
        raise ValueError('Arm F result differs from the one this script was written for')
    if list(cv.CLASSIFIER) != list(ts.CLASSIFIER):
        raise ValueError('Classifier inputs differ between validation and submission recipes')

    data = pd.read_pickle(IDEAS_SUBMIT / 'work' / 'inputs.pkl')
    parameters, fitted = data['parameters'], data['fitted']
    saved = {m: pd.read_parquet(V54 / f'ranking_{m}.parquet') for m in MONTHS}
    for m in MONTHS:                                                   # 1. v54 reproduced
        d = data['months'][m]
        if not np.array_equal(saved[m].mvt_id.to_numpy(), d['raw'].mvt_id.to_numpy()):
            raise ValueError(f'{m}: v54 rows differ from the saved inputs')
        frame, _ = nis.final(saved[m].blend_base.to_numpy(float), saved[m].p.to_numpy(float), d['V'], d['raw'], parameters, fitted)
        gap = float(np.max(np.abs(frame.pred.to_numpy(float) - saved[m].pred.to_numpy(float))))
        if gap > 1e-6:
            raise ValueError(f'{m}: v54 not reproduced ({gap})')
    log('v54 final predictions reproduced from saved parts')

    v46 = ts.read(nis.V46 / 'submission_v46.json')
    config = v46['identity']['config']
    expected = ts.read(v43s.PARENT_MANIFEST)['model']['parameters']
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr = overnight_run.load_inputs(1.)
    X, y, cats, raw, params, membership = cv.training_inputs(dep, arr, config, reports, grid, expected)
    if params != parameters:
        raise ValueError('Training parameters differ from v54')
    del dep, arr
    gc.collect()
    lirf = raw.ADEP_mvt.to_numpy() == 'LIRF'
    XL, rawL, yL = X.loc[lirf], raw.loc[lirf], np.asarray(y, float)[lirf]
    del X, raw
    gc.collect()
    M = signals.classifier_matrix(XL, rawL, KIND, cv.CLASSIFIER)
    s = rome_hybrid.substituted(XL, yL, rawL).astype(float)
    fn_t = rawL.FLIGHT_mvt.fillna('').astype(str).to_numpy()
    m_t = pd.to_datetime(rawL.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').to_numpy()
    rate, n = rc.copy_rate(fn_t, m_t, s, fn_t, m_t)
    MF = pd.concat([M, pd.DataFrame({'fn_copy_rate': rate, 'fn_copy_n': n}, index=M.index)], axis=1)
    log(f'LIRF training rows {len(M):,}, schedule-recorded {int(s.sum()):,}, months {sorted(set(m_t))}')

    def test_matrix(m, with_fn):
        d = data['months'][m]
        V, r = d['V'], d['raw']
        lh = (r.ADEP_mvt.to_numpy() == 'LIRF') & np.isfinite(V[rome_hybrid.GAP].to_numpy(float))
        Mv = signals.classifier_matrix(V.loc[lh], r.loc[lh], KIND, cv.CLASSIFIER)
        if with_fn:
            rv, nv = rc.copy_rate(fn_t, m_t, s, r.loc[lh, 'FLIGHT_mvt'].fillna('').astype(str).to_numpy())
            Mv = pd.concat([Mv, pd.DataFrame({'fn_copy_rate': rv, 'fn_copy_n': nv}, index=Mv.index)], axis=1)
        return lh, Mv

    def probabilities(booster, with_fn):
        out = {}
        for m in MONTHS:
            lh, Mv = test_matrix(m, with_fn)
            p = np.zeros(len(lh))
            p[lh] = booster.predict(Mv, num_threads=ts.THREADS)
            out[m] = p
        return out

    old = probabilities(classifier(M, s, ROUNDS), False)               # 2. R20 reproduced
    repro = max(float(np.max(np.abs(old[m] - saved[m].p.to_numpy(float)))) for m in MONTHS)
    if repro > 1e-9:
        raise ValueError(f'R20 classifier not reproduced ({repro})')
    log('R20 classifier reproduced')
    new = probabilities(classifier(MF, s, ROUNDS), True)
    seen, kept = {}, {}
    for m in MONTHS:
        lh, Mv = test_matrix(m, True)
        unseen = np.zeros(len(lh), bool)
        unseen[lh] = Mv.fn_copy_n.to_numpy() == 0
        seen[m] = float(1 - unseen[lh].mean())
        kept[m] = int(unseen.sum()) if not args.no_fallback else 0
        if not args.no_fallback:
            new[m] = np.where(unseen, old[m], new[m])
    log(f'unseen flight numbers keep v54 probabilities: {kept}')

    directory.mkdir(parents=True)
    write_json(directory / 'refit.json', dict(arm='F', settings=rc.R20, rounds=ROUNDS, fn_prior=rc.FN_PRIOR,
                                              fallback_unseen=not args.no_fallback, unseen_kept_at_v54=kept,
                                              started_at=now(), runtime=runtime_identity(), source=source_snapshot(directory)))
    pieces, summary, flips = [], {}, []
    for m in MONTHS:
        d = data['months'][m]
        frame, masks = nis.final(saved[m].blend_base.to_numpy(float), new[m], d['V'], d['raw'], parameters, fitted)
        frame['p_v54'], frame['blend_base'], frame['month'] = saved[m].p.to_numpy(float), saved[m].blend_base.to_numpy(float), m
        lirf_m = d['raw'].ADEP_mvt.to_numpy() == 'LIRF'
        change = frame.pred.to_numpy(float) - saved[m].pred.to_numpy(float)
        if np.any(change[~lirf_m] != 0):                                  # 3. only LIRF flights change
            raise ValueError(f'{m}: flights outside LIRF changed')
        big = np.abs(change) > 3600
        for i in np.flatnonzero(big):
            flips.append(dict(mvt_id=float(frame.mvt_id.iloc[i]), month=m, v54=float(saved[m].pred.iloc[i]),
                              new=float(frame.pred.iloc[i]), p_v54=float(saved[m].p.iloc[i]), p_new=float(new[m][i])))
        summary[m] = dict(rows=len(frame), lirf_rows=int(lirf_m.sum()), changed=int((change != 0).sum()),
                          mean_change_lirf=float(change[lirf_m].mean()), mean_abs_change_lirf=float(np.abs(change[lirf_m]).mean()),
                          max_abs_change=float(np.abs(change).max()), over_1h=int(big.sum()),
                          sum_sq_change=float((change ** 2).sum()), fn_seen_share=seen[m],
                          mean_p=dict(v54=float(saved[m].p.to_numpy()[lirf_m].mean()), new=float(new[m][lirf_m].mean())),
                          correction_masks=masks)
        overnight_run.save_frame(directory / f'ranking_{m}.parquet', frame)
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(data['rank_ids']).reset_index()
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(nis.TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    nis.transfer.validate_submission(table, template)
    path = submission(args.version)
    pq.write_table(table, path)
    manifest = dict(version=args.version, state='prepared', file=str(path.relative_to(ROOT)), sha256=sha256(path),
                    parent='v54', change='Rome classifier R20 + flight-number copy input (arm F)'
                    + ('' if args.no_fallback else '; unseen flight numbers keep v54'),
                    fallback_unseen=not args.no_fallback,
                    jan_jul_cv=dict(control=result['C']['cv'], arm_F=result['F']['cv'], change=result['F']['change_vs_C']),
                    reproduction=dict(v54=True, r20_classifier=repro), summary=summary, flips_over_1h=flips,
                    minutes=(time.monotonic() - started) / 60, prepared_at=now())
    write_json(directory / f'submission_v{args.version}.json', manifest)
    log(json.dumps(summary, indent=1))
    log(f'flips over 1 h: {json.dumps(flips, indent=1)}')
    log(f'prepared {path} ({manifest["minutes"]:.1f} min); not uploaded')


if __name__ == '__main__':
    main()
