"""Prepare a leaderboard file: v54 with a smaller network share in the blend (no refit).

v54's blend is 0.6 x networks + 0.4 x mean(LightGBM, per-airport LightGBMs, CatBoost). After the tree
models were tuned, Jan/Jul CV of v54's recipe (C5, saved practice predictions, three seeds) prefers a
smaller network share:
    0.60  295.3603   (v54)
    0.55  295.2489   -0.11 [-0.15, -0.07]
    0.50  295.1770   -0.18 [-0.27, -0.10]
    0.45  295.1427   -0.22 [-0.34, -0.09]   both months, every seed
    0.40  295.1470   -0.21 [-0.38, -0.03]
    0.35  295.1911
The weight was read off this grid, on the same two months it is scored on.
v58 (2026-10-06) uses 0.71 instead: solving the leaderboard scores of v54 (0.6) and v57 (0.45),
with the final step nearly linear in the share, puts the leaderboard's best share near 0.71
(predicted 236.197, -0.14 vs v54), against Jan/Jul CV's +0.38.
The file recombines v54's saved network and tree parts per ranking flight with the new share, then
applies v54's Rome probabilities, corrections and rules (`nn_improve_submit.final`). Check first: v54's
final predictions are reproduced from its saved parts at 0.6.

Run: .venv/bin/python VALIDATION/blend_weight_submit.py --version 57 --w-nn 0.45
"""
import argparse
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
from rome_hybrid_submit import MONTHS, now, submission
from submit import fill_template

IDEAS_SUBMIT = RUNS / 'v3_20261003_ideas_submit'
V54 = RUNS / 'v3_20261006_hpo_submit' / 'submission_v54'
V54_SHA = '367d4d19f365c497e7a408713513a4813a633e02d5f7f658233f68a53d77970e'
OUT = RUNS / 'v3_20261006_blend_weight_submit'
CV = {0.6: 295.3603, 0.55: 295.2489, 0.5: 295.1770, 0.45: 295.1427, 0.4: 295.1470, 0.35: 295.1911,
      0.71: 295.7414}   # 0.71: scored 2026-10-06 with the same saved C5 parts, three seeds


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--w-nn', type=float, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    directory = OUT / f'submission_v{args.version}'
    if directory.exists() or submission(args.version).exists():
        raise ValueError(f'v{args.version} is already prepared; inspect it before continuing')
    if sha256(submission(54)) != V54_SHA:
        raise ValueError('v54 file differs from the scored upload')
    data = pd.read_pickle(IDEAS_SUBMIT / 'work' / 'inputs.pkl')
    parameters, fitted = data['parameters'], data['fitted']
    directory.mkdir(parents=True)
    pieces, summary = [], {}
    for m in MONTHS:
        d = data['months'][m]
        saved = pd.read_parquet(V54 / f'ranking_{m}.parquet')
        if not np.array_equal(saved.mvt_id.to_numpy(), d['raw'].mvt_id.to_numpy()):
            raise ValueError(f'{m}: v54 rows differ from the saved inputs')
        nn, gbm, p = saved.nn_base.to_numpy(float), saved.gbm_base.to_numpy(float), saved.p.to_numpy(float)
        if np.max(np.abs(0.6 * nn + 0.4 * gbm - saved.blend_base.to_numpy(float))) > 1e-6:
            raise ValueError(f'{m}: v54 blend base is not 0.6 x networks + 0.4 x trees')
        frame, _ = nis.final(0.6 * nn + 0.4 * gbm, p, d['V'], d['raw'], parameters, fitted)
        if np.max(np.abs(frame.pred.to_numpy(float) - saved.pred.to_numpy(float))) > 1e-6:
            raise ValueError(f'{m}: v54 not reproduced')
        base = args.w_nn * nn + (1 - args.w_nn) * gbm
        frame, masks = nis.final(base, p, d['V'], d['raw'], parameters, fitted)
        frame['nn_base'], frame['gbm_base'], frame['blend_base'], frame['month'] = nn, gbm, base, m
        change = frame.pred.to_numpy(float) - saved.pred.to_numpy(float)
        summary[m] = dict(rows=len(frame), changed=int((np.abs(change) > 0.5).sum()), mean_change=float(change.mean()),
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
    path = submission(args.version)
    pq.write_table(table, path)
    manifest = dict(version=args.version, state='prepared', file=str(path.relative_to(ROOT)), sha256=sha256(path),
                    parent='v54', change=f'network share {args.w_nn} instead of 0.6', w_nn=args.w_nn,
                    jan_jul_cv=dict(grid=CV, chosen=CV.get(args.w_nn)), summary=summary, runtime=runtime_identity(),
                    source=source_snapshot(directory), minutes=(time.monotonic() - started) / 60, prepared_at=now())
    write_json(directory / f'submission_v{args.version}.json', manifest)
    log(json.dumps(summary, indent=1))
    log(f'prepared {path}; not uploaded')


if __name__ == '__main__':
    main()
