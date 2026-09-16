"""Compare official scores with frozen three-seed and seed-42 CV readouts."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import numpy as np
from records import read_run, sha256, write_json

REPORTS = ROOT/'VALIDATION/reports'
METRICS = ['R_A', 'R_B_long', 'R_combined']
PAIRS = [('Q0', 'B0-800'), ('Q1', 'Q0'), ('Q0-4h', 'Q0'),
         ('Q1-4h', 'Q1'), ('Q1-4h', 'Q0-4h')]


def direction(value):
    if value is None:
        return 'pending'
    return 'improvement' if value < -0.00005 else 'regression' if value > 0.00005 else 'no change'


def main():
    manifest = REPORTS/'leaderboard_selected_20260916.json'
    batch = json.loads(manifest.read_text())
    sweep_path = REPORTS/'stage2_threshold_sweep.json'
    sweep = json.loads(sweep_path.read_text())
    items = {r['codename']: r for r in batch['submissions']}
    q0, _ = read_run(items['Q0']['validation_run'])
    intervals = [q0['report']['comparison']['scores'], sweep['r3_vs_baseline']['6'],
                 sweep['candidates']['B0-800']['thresholds']['4']['vs_6'],
                 sweep['candidates']['R3']['thresholds']['4']['vs_6'],
                 sweep['r3_vs_baseline']['4']]
    result = {'scores_checked_at': batch.get('scores_checked_at'),
              'manifest_sha256': sha256(manifest), 'sweep_sha256': sha256(sweep_path),
              'metric': 'RMSE seconds; candidate-minus-reference; negative is improvement',
              'levels': {}, 'comparisons': []}
    for code, item in items.items():
        score = item['score']
        result['levels'][code] = {
            'file': item['file'], 'official_status': item.get('official_status', item['state']),
            'official_rmse': score, 'validation_run': item['validation_run'],
            'standard': item['cv_standard'], 'seed42': item['cv_seed42'],
            'lb_minus_cv': {mode: {k: None if score is None else score-item[field][k]
                                  for k in METRICS}
                            for mode,field in [('standard','cv_standard'),('seed42','cv_seed42')]}}
    for (candidate, reference), ci in zip(PAIRS, intervals):
        a, b = items[candidate], items[reference]
        scored = a['score'] is not None and b['score'] is not None
        if scored and a['official_result'].get('used_pairs') != b['official_result'].get('used_pairs'):
            raise ValueError('Official scores use different row counts')
        delta_lb = a['score']-b['score'] if scored else None
        comparison = {'candidate': candidate, 'reference': reference,
                      'delta_lb': delta_lb,
                      'delta_lb_pct': 100*delta_lb/b['score'] if scored else None,
                      'lb_direction': direction(delta_lb)}
        for mode, field in [('standard', 'cv_standard'), ('seed42', 'cv_seed42')]:
            metrics = {}
            for key in METRICS:
                delta_cv = a[field][key]-b[field][key]
                if mode == 'standard' and not np.isclose(delta_cv, ci[key]['difference'], atol=1e-9, rtol=0):
                    raise ValueError('Saved paired interval does not match selected CV scores')
                cv_direction = direction(delta_cv)
                metrics[key] = {'delta_cv': delta_cv, 'delta_cv_pct': 100*delta_cv/b[field][key],
                                'cv_direction': cv_direction,
                                'agreement': 'pending' if not scored else
                                    'no change' if 'no change' in [cv_direction, direction(delta_lb)] else
                                    'agree' if cv_direction == direction(delta_lb) else 'disagree',
                                'delta_lb_minus_delta_cv': None if not scored else delta_lb-delta_cv}
                if mode == 'standard':
                    metrics[key].update(ci95=ci[key]['ci95'], cv_decision=ci[key]['decision'])
            comparison[mode] = metrics
        result['comparisons'].append(comparison)
    write_json(REPORTS/'leaderboard_cv_comparison.json', result)
    fmt = lambda x: 'pending' if x is None else f'{x:.4f}'
    lines = ['# Selected leaderboard results versus CV', '',
             f"Official team-bucket results checked **{result['scores_checked_at']}**. Lower RMSE is better.", '',
             'The five candidates were selected before these official scores. Final models use seed 42, '
             '800 rounds, all twelve 2025 months and the original 27 features. Two model fits supply all '
             'five files; no CV models were refitted. Six-hour Q0/Q1 identities are preserved.', '',
             '## Score levels and LB minus CV', '',
             'Standard CV averages squared error across three seeds and repeated origins before taking '
             'the square root. Seed-42 CV is a secondary comparison with the final model seed. '
             'Neither table averages predictions.', '']
    for mode in ['standard', 'seed42']:
        lines += [f'### {mode}', '',
                  '| Candidate / version | Official LB | CV next month | CV longer gap | CV combined | LB − next month | LB − longer gap | LB − combined |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for code, row in result['levels'].items():
            version = items[code]['version']
            values = [row['official_rmse']]+[row[mode][k] for k in METRICS]+[row['lb_minus_cv'][mode][k] for k in METRICS]
            lines.append(f'| {code} / v{version} | '+' | '.join(fmt(v) for v in values)+' |')
        lines.append('')
    lines += ['## Controlled changes', '',
              'All changes are candidate minus reference; negative means improvement. Percent changes use '
              'the matching reference score. “Gap change” is ΔLB − ΔCV, equivalently the change in LB-minus-CV. '
              'Agreement describes point-estimate direction only; it does not overturn an inconclusive CV interval.', '']
    for row in result['comparisons']:
        lines += [f"### {row['candidate']} − {row['reference']}", '',
                  f"**LB change: {fmt(row['delta_lb'])} s ({fmt(row['delta_lb_pct'])}%).**", '',
                  '| CV readout | ΔCV (s) | ΔCV (%) | CV 95% interval | CV assessment | Direction vs LB | Gap change (s) |',
                  '|---|---:|---:|---|---|---|---:|']
        for key, data in row['standard'].items():
            interval = ', '.join(fmt(v) for v in data['ci95'])
            lines.append(f"| {key} | {fmt(data['delta_cv'])} | {fmt(data['delta_cv_pct'])} | [{interval}] | {data['cv_decision']} | {data['agreement']} | {fmt(data['delta_lb_minus_delta_cv'])} |")
        lines += ['', '| Seed-42 readout | ΔCV (s) | ΔCV (%) | Direction vs LB | Gap change (s) |',
                  '|---|---:|---:|---|---:|']
        for key, data in row['seed42'].items():
            lines.append(f"| {key} | {fmt(data['delta_cv'])} | {fmt(data['delta_cv_pct'])} | {data['agreement']} | {fmt(data['delta_lb_minus_delta_cv'])} |")
        lines.append('')
    lines += ['## Interpretation limits and provenance', '',
              '- CV uses six-month training windows in 2025; the final fit uses all twelve months for January and July 2026. '
              'The evaluation periods and training size differ. LB versus CV gaps combine these effects.',
              '- Submission values are rounded to integer seconds in the original template order; CV uses unrounded predictions. '
              'Both ranking months are processed independently, with departure history frozen at 2025-12-31.',
              '- The provisional combined CV score uses the ranking month-size weights. It is not a calibrated LB forecast.',
              '- CV intervals reuse the original 2,000 paired, month-stratified calendar-day bootstrap draws. '
              'Four hours came from adaptive selection on reused development folds; its intervals are pointwise, '
              'without multiplicity correction. Small selected-sample comparisons do not establish a correlation or an optimum.',
              '- Pending and rejected submissions have no official score. The manifest retains official result payloads and timestamps.',
              '- [Frozen selection](../docs/LEADERBOARD_CANDIDATES.md), '
              '[submission/model/source/data manifest](leaderboard_selected_20260916.json), '
              '[complete comparison data](leaderboard_cv_comparison.json), '
              '[original threshold intervals](stage2_threshold_sweep.json).', '']
    (REPORTS/'LEADERBOARD_RESULTS.md').write_text('\n'.join(lines))
    if all(r['delta_lb'] is not None for r in result['comparisons']):
        plot(result)
    print(json.dumps({k:{'LB':v['official_rmse'], 'CV':v['standard']['R_combined'],
                          'gap':v['lb_minus_cv']['standard']['R_combined']} for k,v in result['levels'].items()}, indent=2))


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    fig.subplots_adjust(left=.07, right=.98, bottom=.24, top=.77, wspace=.28)
    fig.suptitle('Selected candidates: leaderboard change versus CV change', x=.07, ha='left', weight='bold')
    colors = ['#087f8c', '#b34b36', '#4969a6', '#9772a4', '#bf8626']
    for ax, key in zip(axes, METRICS):
        all_values = []
        for row, color in zip(result['comparisons'], colors):
            cv = row['standard'][key]
            x, y = cv['delta_cv'], row['delta_lb']
            low, high = cv['ci95']
            ax.errorbar(x, y, xerr=[[x-low], [high-x]], fmt='o', color=color, capsize=3,
                        label=row['candidate']+' − '+row['reference'])
            all_values += [low, high, y]
        lo, hi = min(all_values)-5, max(all_values)+5
        ax.plot([lo, hi], [lo, hi], '--', color='#888888', linewidth=.8)
        ax.axhline(0, color='#cccccc', linewidth=.8)
        ax.axvline(0, color='#cccccc', linewidth=.8)
        ax.set(title=key, xlabel='ΔCV (seconds)', ylabel='ΔLB (seconds)')
        ax.grid(alpha=.12)
        ax.spines[['top','right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.text(.07, .84, 'Negative = improvement · horizontal bars: paired CV 95% intervals · dashed line: equal change', fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(REPORTS/f'leaderboard_cv_changes.{ext}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
