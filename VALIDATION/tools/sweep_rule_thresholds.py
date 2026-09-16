"""Diagnostic Rome-rule threshold sweep on saved baseline and R3 predictions."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import ROOT, data_identity, read_run, sha256, write_json
from reporting import (Loss, boot_scores, bootstrap_calendar, group_loss, interval,
                       load_losses, rmse, slice_stats, validate_comparison)
sys.path.insert(0, str(ROOT / 'pipeline'))
from common import DATA, TRAIN_GLOB, connect
from prediction_rules import RULE_INPUTS, apply_rule

# Declared before evaluating this sensitivity check. No automatic promotion.
HOURS = (4, 5, 6, 7, 8)
GROUPS = {'R_A': [f'A{i}' for i in range(1, 7)], 'R_B_long': ['B3', 'B4', 'B7']}


def eligible_gaps(raw):
    """Eligibility uses original observable fields, never labels/recovered plans."""
    if raw.index.has_duplicates:
        raise ValueError('duplicate raw movement IDs')
    visible = raw.loc[:, RULE_INPUTS]
    gap = (pd.to_datetime(visible.MVT_TIME_UTC_mvt, utc=True)
           - pd.to_datetime(visible.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    return gap.where(visible.ADEP_mvt.eq('LIRF') & visible.FLIGHT_ID_mvt.isna())


def threshold_loss(original, gaps, hours):
    """A deterministic override commutes with averaging losses across origins."""
    if not np.isfinite(hours) or hours <= 0:
        raise ValueError('threshold must be finite and positive')
    aligned = gaps.reindex(original.meta.index).to_numpy()
    mask = aligned >= hours * 3600
    sq = original.sq.copy()
    sq[:, mask] = (aligned[mask] - original.meta.y.to_numpy()[mask]) ** 2
    return Loss(original.meta, sq, original.seeds), mask, aligned


def changed_rows(original, candidate, mask, gaps):
    rows = original.meta.loc[mask, ['mvt_time', 'y']].copy()
    rows['mvt_time'] = rows.mvt_time.astype(str)
    rows['rule_prediction'] = gaps[mask]
    rows['parent_mean_squared_error'] = original.sq[:, mask].mean(axis=0)
    rows['rule_squared_error'] = candidate.sq[:, mask].mean(axis=0)
    rows['delta_squared_error'] = rows.rule_squared_error - rows.parent_mean_squared_error
    return rows.reset_index()


def change_counts(rows):
    delta = rows.delta_squared_error
    return {'triggered': len(rows), 'helped': int(delta.lt(0).sum()),
            'harmed': int(delta.gt(0).sum()), 'unchanged': int(delta.eq(0).sum()),
            'delta_sse': float(delta.sum())}


def metrics(loss, changes):
    months = loss.meta.mvt_time.dt.strftime('%Y-%m')
    monthly = {}
    for month in sorted(months.unique()):
        rows = changes.loc[changes.mvt_time.str.startswith(month)]
        monthly[month] = {**slice_stats(loss, months.eq(month)), **change_counts(rows)}
    return {**slice_stats(loss), 'changes': change_counts(changes), 'monthly': monthly,
            'rmse_by_seed': dict(zip(map(str, loss.seeds), np.sqrt(loss.sq.mean(axis=1)))),
            'ordinary': slice_stats(loss, loss.meta.y.le(7200)),
            'tail': slice_stats(loss, loss.meta.y.gt(7200)),
            'at_most_12_seconds': slice_stats(loss, loss.meta.y.le(12))}


def draw_plot(candidates, output, hours):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.76, wspace=.32)
    fig.suptitle('Rome rule · threshold sensitivity', x=.07, ha='left', weight='bold', fontsize=16)
    fig.text(.07, .84, 'Saved predictions · three seeds · lower RMSE is better', fontsize=11)
    for ax, key, title in zip(axes, ['R_A', 'R_B_long', 'R_combined'],
                             ['Next month', 'Longer gap', 'Provisional combined']):
        for label, color in [('B0-800', '#087f8c'), ('R3', '#b34b36')]:
            values = [candidates[label]['thresholds'][str(h)]['scores'][key] for h in hours]
            ax.plot(hours, values, 'o-', color=color, label=label)
        ax.axvline(6, color='#999999', ls='--', lw=.8)
        ax.set(title=title, xlabel='Threshold (hours)', ylabel='RMSE (seconds)', xticks=hours)
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.text(.07, .06, 'Reused development folds; point estimates do not establish an optimal threshold.\n'
             'The existing six-hour Q0/Q1 recipes are unchanged.', fontsize=9, color='#555555')
    for extension in ['png', 'pdf']:
        fig.savefig(output.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='v3_20260916T095948645262Z')
    parser.add_argument('--r3', default='v3_20260916T113256202842Z')
    parser.add_argument('--q0', default='v3_20260916T123230629423Z')
    parser.add_argument('--q1', default='v3_20260916T123433565511Z')
    parser.add_argument('--hours', nargs='+', type=int, default=HOURS,
                        help='Predeclared positive integer hours; six-hour control is always included')
    parser.add_argument('--output', type=Path, default=ROOT / 'VALIDATION/reports/stage2_threshold_sweep')
    args = parser.parse_args()
    if any(h <= 0 for h in args.hours):
        parser.error('--hours must be positive')
    hours_grid = tuple(sorted(set(args.hours) | {6}))
    start = time.monotonic()
    parents = [read_run(args.baseline), read_run(args.r3)]
    references = [read_run(args.q0), read_run(args.q1)]
    records = [r for r, _ in parents + references]
    if any(r['status'] != 'complete' or not r['full_benchmark'] for r in records):
        raise ValueError('requires complete full benchmark parents and six-hour controls')
    identity = data_identity(DATA / item['file'] for item in parents[0][0]['data_identity'])
    if any(r['data_identity'] != identity for r in records):
        raise ValueError('input data differ from frozen runs')
    for key in ['rounds', 'params', 'bootstrap_reps', 'bootstrap_seed', 'block_days']:
        if parents[0][0]['config'][key] != parents[1][0]['config'][key]:
            raise ValueError(f'parent configuration differs in {key}')
    out = {'kind': 'threshold_sensitivity', 'at': datetime.now(timezone.utc).isoformat(),
           'threshold_hours': hours_grid, 'new_model_fits': 0, 'data_identity': identity,
           'protocol': parents[0][0]['protocol'], 'score_version': parents[0][0]['score_version'],
           'ranking_weights': parents[0][0]['ranking_weights'],
           'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
               [Path(__file__).resolve(), ROOT/'VALIDATION/reporting.py', ROOT/'pipeline/prediction_rules.py']},
           'interpretation': 'Sensitivity on reused development folds; pointwise intervals, no multiplicity '
                             'correction. No automatic threshold promotion or submission.',
           'candidates': {}}
    con = connect()
    con.execute('SET threads=2')
    raw = con.sql('''SELECT MVT_ID_mvt mvt_id, ADEP_mvt, FLIGHT_ID_mvt,
        MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt FROM read_parquet(?)
        WHERE PHASE_mvt='DEP' AND ADEP_mvt='LIRF' AND FLIGHT_ID_mvt IS NULL''',
        params=[TRAIN_GLOB]).df().set_index('mvt_id')
    con.close()
    gaps = eligible_gaps(raw)
    _, fixed_audit = apply_rule(np.zeros(len(raw)), raw, 'rome_6h')
    np.testing.assert_array_equal(gaps.ge(21600), fixed_audit.rule_triggered)
    all_changes, score_rows, monthly_rows, all_boots = [], [], [], {}
    baseline_losses = None
    w = out['ranking_weights']['jan_fraction']
    combined = lambda scores: np.sqrt(w*scores['R_A']**2 + (1-w)*scores['R_B_long']**2)
    for label, (parent, directory), (reference, ref_dir), recovery in zip(
            ['B0-800', 'R3'], parents, references, ['off', 'history']):
        if parent['config'].get('prediction_rule', 'off') != 'off' or parent['config']['recovery'] != recovery:
            raise ValueError('incorrect parent recipe')
        if reference['model_parent_id'] != parent['run_id'] or reference['config']['prediction_rule'] != 'rome_6h':
            raise ValueError('incorrect six-hour reference')
        hashes = {}
        for fold, info in parent['folds'].items():
            for seed, filename in info['predictions'].items():
                hashes[filename] = sha256(directory/filename)
                if hashes[filename] != reference['prediction_provenance'][f'{fold}/{seed}']['parent_sha256']:
                    raise ValueError('saved parent predictions changed since Stage 2')
        print(f'{label}: loading saved predictions once', flush=True)
        losses = load_losses(parent, directory)
        if baseline_losses is None:
            baseline_losses = losses
            cfg = parent['config']
            calendar, weights = bootstrap_calendar(losses, cfg['bootstrap_reps'],
                                                    cfg['bootstrap_seed'], cfg['block_days'])
            out['bootstrap'] = {k: cfg[k] for k in ['bootstrap_reps', 'bootstrap_seed', 'block_days']}
        else:
            validate_comparison(parent, parents[0][0], losses, baseline_losses)
        groups = {name: group_loss(losses, folds) for name, folds in GROUPS.items()}
        for loss in groups.values():
            ids = loss.meta.index.intersection(raw.index)
            if not loss.meta.loc[ids, 'airport'].eq('LIRF').all():
                raise ValueError('raw airports disagree with predictions')
            np.testing.assert_array_equal(loss.meta.loc[ids, 'mvt_time'], raw.loc[ids, 'MVT_TIME_UTC_mvt'])
        result = {'run_id': parent['run_id'], 'parent_record_sha256': sha256(directory/'result.json'),
                  'six_hour_reference': reference['run_id'], 'parent_prediction_sha256': hashes,
                  'model_source': parent['source'], 'config': parent['config'], 'thresholds': {}}
        candidate_boots = {}
        for hours in [None, *hours_grid]:
            key = 'off' if hours is None else str(hours)
            item, draws = {'scores': {}, 'groups': {}}, {}
            for name, original in groups.items():
                if hours is None:
                    candidate, mask, aligned = original, np.zeros(len(original.meta), bool), np.full(len(original.meta), np.nan)
                else:
                    candidate, mask, aligned = threshold_loss(original, gaps, hours)
                changes = changed_rows(original, candidate, mask, aligned)
                all_changes.append(changes.assign(parent=label, threshold_hours=hours, group=name))
                item['groups'][name] = metrics(candidate, changes)
                item['scores'][name] = rmse(candidate)
                draws[name] = boot_scores(candidate, calendar, weights)
                for month, stats in item['groups'][name]['monthly'].items():
                    monthly_rows.append({'parent': label, 'threshold': key, 'group': name, 'month': month, **stats})
                if hours == 6:
                    old = pd.read_csv(ref_dir/reference['changed_rows'])
                    np.testing.assert_array_equal(sorted(changes.mvt_id),
                        sorted(old.loc[old.fold.isin(GROUPS[name]), 'mvt_id'].unique()))
            item['scores']['R_combined'] = float(combined(item['scores']))
            draws['R_combined'] = combined(draws)
            if hours in [None, 6]:
                expected = (parent if hours is None else reference)['report']['scores']
                for name, score in item['scores'].items():
                    np.testing.assert_allclose(score, expected[name], rtol=0, atol=1e-9)
            result['thresholds'][key], candidate_boots[key] = item, draws
            score_rows.append({'parent': label, 'threshold': key, **item['scores'],
                               **{f'{group}_{count}': item['groups'][group]['changes'][count]
                                  for group in GROUPS for count in ['triggered', 'helped', 'harmed']}})
            print(f'{label} {key}h: {item["scores"]}', flush=True)
        for key, item in result['thresholds'].items():
            for target in ['off', *map(str, hours_grid)]:
                item[f'vs_{target}'] = {name: interval(
                    score-result['thresholds'][target]['scores'][name],
                    candidate_boots[key][name]-candidate_boots[target][name])
                    for name, score in item['scores'].items()}
        result['six_hour_and_unmodified_scores_reproduced'] = True
        out['candidates'][label], all_boots[label] = result, candidate_boots
    out['r3_vs_baseline'] = {key: {name: interval(
        out['candidates']['R3']['thresholds'][key]['scores'][name]
        - out['candidates']['B0-800']['thresholds'][key]['scores'][name],
        all_boots['R3'][key][name]-all_boots['B0-800'][key][name])
        for name in ['R_A', 'R_B_long', 'R_combined']} for key in ['off', *map(str, hours_grid)]}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(output.with_suffix('.csv'), index=False)
    pd.DataFrame(monthly_rows).to_csv(output.with_name(output.name+'_monthly.csv'), index=False)
    pd.concat(all_changes, ignore_index=True).to_csv(output.with_name(output.name+'_changed_rows.csv'), index=False)
    draw_plot(out['candidates'], output, hours_grid)
    out['minutes'] = round((time.monotonic()-start)/60, 3)
    write_json(output.with_suffix('.json'), out)
    print(f'Completed in {out["minutes"]} minutes: {output.with_suffix(".json")}', flush=True)


if __name__ == '__main__':
    main()
