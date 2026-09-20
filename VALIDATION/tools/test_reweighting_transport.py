"""Does importance-weighted CV transport? Tested where the answer is known.

The design is frozen in `VALIDATION/docs/REWEIGHTING_TRANSPORT_TEST.md` and its
SHA-256 is asserted below, so the criteria cannot be chosen after the results.
If that document changes, this script refuses to run and the rerun is not
independent confirmation.

One fixed model -- `v3_2025-06`, trained January-June 2025 -- predicts five months
it never trained on. Each month in turn is the pretend ranking set; the other four
estimate its RMSE, and the estimate is compared with the truth. Two baselines run
on the same rows: no weighting, and matching the no-flight-ID share only. Level
and candidate-minus-control differences are both tested, because predicting a
score is not the same as ranking two models.

Saved predictions and local diagnostics only. No taxi-out refits, no submissions,
no fold removals, no model promotions. The real leaderboard score is not an input.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'VALIDATION/tools'), str(ROOT / 'pipeline')]
import analyze_adversarial_shift as adv  # noqa: E402
from records import write_json  # noqa: E402

RUN = ROOT / 'VALIDATION/runs/v3_20260919_artifact_experiments'
DESIGN = ROOT / 'VALIDATION/docs/REWEIGHTING_TRANSPORT_TEST.md'
DESIGN_SHA = 'c51bbda6feb27f4d9562429d87342dbf419fb15d09007e0d9dfc15a24b716501'
OUT = ROOT / 'VALIDATION/reports/reweighting_transport.json'
MODEL = 'v3_2025-06'
TARGETS = {'A1': '2025-07', 'B1': '2025-09', 'B2': '2025-10', 'B3': '2025-11', 'B4': '2025-12'}
CONTROL = 'pred_noid_cap'
HEADLINE = 'pred_combined'


def frozen_design():
    digest = hashlib.sha256(DESIGN.read_bytes()).hexdigest()
    if digest != DESIGN_SHA:
        raise SystemExit(
            f'{DESIGN.name} changed (sha256 {digest[:16]}..., expected {DESIGN_SHA[:16]}...).\n'
            'The design was frozen before results existed. Editing it and rerunning the same '
            'five months is not independent confirmation; start a new document and say so.')
    return digest


def squared_errors(record):
    """Per-flight squared error, averaged over seeds, for every saved column."""
    seeds = record['config']['seeds']
    out, columns = {}, None
    for fold in TARGETS:
        info = record['folds'][fold]
        assert info['model_id'] == MODEL, f'{fold} is not scored by {MODEL}'
        frames = []
        for seed in seeds:
            f = pd.read_parquet(RUN / info['predictions'][str(seed)]).set_index('mvt_id').sort_index()
            if columns is None:
                columns = [c for c in f.columns if c.startswith('pred_')]
            frames.append(pd.DataFrame(
                {c: (f.y.to_numpy() - f[c].to_numpy()) ** 2 for c in columns}, index=f.index))
        pooled = sum(frames) / len(frames)
        out[fold] = pooled
        print(f"  {fold} ({info['val']}): {len(pooled):,} flights, {len(columns)} columns")
    return out, columns


def weighted_rmse(sq, w):
    return float(np.sqrt(np.average(sq, weights=w)))


def main():
    digest = frozen_design()
    print(f'design frozen, sha256 {digest[:16]}...')
    record = json.loads((RUN / 'result.json').read_text())
    record = {**record, 'prediction_column': HEADLINE}
    print(f'\nloading saved predictions from the fixed {MODEL} model:')
    sq, columns = squared_errors(record)
    candidates = [c for c in columns if c != CONTROL]

    print('\nloading features...')
    df = adv.load()
    wanted = pd.Index(np.concatenate([s.index.to_numpy() for s in sq.values()]))
    df = df[df.mvt_id.isin(wanted)].reset_index(drop=True)
    print(f'  {len(df):,} rows across the five months')
    month_of = dict(zip(df.mvt_id, df.ym))

    results, level_rows, diff_rows = {}, [], []
    for target, month in TARGETS.items():
        print(f'\n=== target {target} ({month}) ===')
        source_folds = [f for f in TARGETS if f != target]
        source_ids = np.concatenate([sq[f].index.to_numpy() for f in source_folds])
        target_ids = sq[target].index.to_numpy()
        source = df.mvt_id.isin(source_ids).to_numpy()
        target_mask = df.mvt_id.isin(target_ids).to_numpy()
        ratios, counts, calib = adv.cross_fitted(
            df, pd.Series(source, index=df.index), pd.Series(target_mask, index=df.index), target)

        src_sq = pd.concat([sq[f] for f in source_folds])
        w = ratios.ratio.reindex(src_sq.index).to_numpy()
        assert np.isfinite(w).all() and (w > 0).all(), f'{target}: a source flight has no weight'
        cap = np.quantile(w, 0.99)
        # Baseline 2: match only the share of departures with no flight ID.
        noid = df.set_index('mvt_id').no_flight_id.reindex(src_sq.index).to_numpy() > 0.5
        p_target = float(df.loc[target_mask, 'no_flight_id'].mean())
        p_source = float(noid.mean())
        w_noid = np.where(noid, p_target / p_source, (1 - p_target) / (1 - p_source))

        entry = {'month': month, 'source_folds': source_folds,
                 'n_source': int(len(src_sq)), 'n_target': int(len(sq[target])),
                 'cross_fit_counts': counts, 'probability_calibration_heldout': calib,
                 'noid_share_source_pct': 100 * p_source, 'noid_share_target_pct': 100 * p_target,
                 'ess_pct': float(100 * w.sum() ** 2 / np.sum(w ** 2) / len(w)),
                 'level': {}, 'differences': {}}

        for column in (HEADLINE, CONTROL):
            truth = float(np.sqrt(sq[target][column].mean()))
            estimates = {
                'weighted': weighted_rmse(src_sq[column].to_numpy(), w),
                'weighted_clipped_p99': weighted_rmse(src_sq[column].to_numpy(), np.minimum(w, cap)),
                'unweighted': float(np.sqrt(src_sq[column].mean())),
                'noid_only': weighted_rmse(src_sq[column].to_numpy(), w_noid),
            }
            entry['level'][column] = {'truth': truth, **estimates,
                                      **{f'error_{k}': v - truth for k, v in estimates.items()}}
            if column == HEADLINE:
                print(f'  level ({column}): truth {truth:.3f} | weighted '
                      f"{estimates['weighted']:.3f} ({estimates['weighted'] - truth:+.3f}) | "
                      f"unweighted {estimates['unweighted']:.3f} "
                      f"({estimates['unweighted'] - truth:+.3f}) | no-ID only "
                      f"{estimates['noid_only']:.3f} ({estimates['noid_only'] - truth:+.3f})")
            for method in estimates:
                level_rows.append({'month': month, 'column': column, 'method': method,
                                   'error': estimates[method] - truth})

        control_truth = float(np.sqrt(sq[target][CONTROL].mean()))
        for column in candidates:
            truth = float(np.sqrt(sq[target][column].mean())) - control_truth
            deltas = {}
            for method, weights in (('weighted', w), ('unweighted', None), ('noid_only', w_noid)):
                if weights is None:
                    d = float(np.sqrt(src_sq[column].mean()) - np.sqrt(src_sq[CONTROL].mean()))
                else:
                    d = (weighted_rmse(src_sq[column].to_numpy(), weights) -
                         weighted_rmse(src_sq[CONTROL].to_numpy(), weights))
                deltas[method] = d
                diff_rows.append({'month': month, 'candidate': column, 'method': method,
                                  'estimate': d, 'truth': truth, 'error': d - truth,
                                  'sign_correct': bool(np.sign(d) == np.sign(truth)) if truth else None})
            entry['differences'][column] = {'truth': truth, **deltas}
        results[target] = entry

    level = pd.DataFrame(level_rows)
    diffs = pd.DataFrame(diff_rows)
    summary = {'level': {}, 'differences': {}}
    print('\n=== level, every month reported (error = estimate - truth, seconds) ===')
    for column in (HEADLINE, CONTROL):
        part = level[level.column.eq(column)]
        print(f'\n {column}')
        table = part.pivot(index='month', columns='method', values='error')
        print(table.to_string(float_format=lambda v: f'{v:+8.3f}'))
        summary['level'][column] = {
            m: {'per_month': {r.month: r.error for r in part[part.method.eq(m)].itertuples()},
                'mean_abs_error': float(part[part.method.eq(m)].error.abs().mean()),
                'worst_abs_error': float(part[part.method.eq(m)].error.abs().max())}
            for m in part.method.unique()}
        for m, v in summary['level'][column].items():
            print(f"   {m:<22} mean |error| {v['mean_abs_error']:7.3f}   "
                  f"worst {v['worst_abs_error']:7.3f}")

    print('\n=== candidate-minus-control differences ===')
    for m in ('weighted', 'unweighted', 'noid_only'):
        part = diffs[diffs.method.eq(m)]
        signs = part.sign_correct.dropna()
        summary['differences'][m] = {
            'n_pairs': int(len(part)), 'mean_abs_error': float(part.error.abs().mean()),
            'sign_correct': int(signs.sum()), 'sign_total': int(len(signs)),
            'per_month_mean_abs_error': {k: float(v) for k, v in
                                         part.groupby('month').error.apply(lambda s: s.abs().mean()).items()}}
        v = summary['differences'][m]
        print(f"   {m:<12} mean |error| {v['mean_abs_error']:.4f} s, "
              f"sign right {v['sign_correct']}/{v['sign_total']}")
        print(f"      per month: " + ', '.join(f'{k} {x:.3f}' for k, x in v['per_month_mean_abs_error'].items()))

    # Criteria fixed in the frozen design; evaluated, not chosen, here.
    lvl = summary['level'][HEADLINE]
    level_pass = (lvl['weighted']['mean_abs_error'] < min(lvl['unweighted']['mean_abs_error'],
                                                          lvl['noid_only']['mean_abs_error'])
                  and lvl['weighted']['worst_abs_error'] < min(lvl['unweighted']['worst_abs_error'],
                                                               lvl['noid_only']['worst_abs_error']))
    d = summary['differences']
    diff_pass = (d['weighted']['sign_correct'] >= d['unweighted']['sign_correct']
                 and d['weighted']['mean_abs_error'] < d['unweighted']['mean_abs_error'])
    verdict = {'level_transports': bool(level_pass), 'differences_transport': bool(diff_pass),
               'criteria': 'fixed in VALIDATION/docs/REWEIGHTING_TRANSPORT_TEST.md before results',
               'reading': ('both' if level_pass and diff_pass else
                           'neither' if not level_pass and not diff_pass else 'mixed')}
    print(f"\nVERDICT (criteria frozen in advance): level {'PASS' if level_pass else 'FAIL'}, "
          f"differences {'PASS' if diff_pass else 'FAIL'} → {verdict['reading']}")

    write_json(OUT, {'design_sha256': digest, 'model': MODEL, 'targets': TARGETS,
                     'control': CONTROL, 'headline': HEADLINE, 'candidates': candidates,
                     'per_target': results, 'summary': summary, 'verdict': verdict})
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
