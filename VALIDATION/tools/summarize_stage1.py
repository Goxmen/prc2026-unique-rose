"""Create the Stage 1 comparison figure from completed, paired run reports."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import read_run, write_json, ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control', required=True)
    ap.add_argument('--r1', required=True)
    ap.add_argument('--r2', required=True)
    ap.add_argument('--r3', required=True)
    args = ap.parse_args()
    control, _ = read_run(args.control)
    candidates = []
    for label, ident in [('R1 · exact twins', args.r1), ('R2 · + relaxed twins', args.r2),
                          ('R3 · + static history', args.r3)]:
        run, directory = read_run(ident)
        if run['status'] != 'complete':
            raise ValueError('All candidates must be complete')
        comparison = run['report'].get('comparison')
        if not comparison or comparison['baseline_id'] != args.control:
            import json
            path = directory / f'comparison_{args.control}_block1.json'
            comparison = json.loads(path.read_text())['report']['comparison']
        candidates.append(dict(label=label, run_id=ident, scores=run['report']['scores'],
                               vs_control=comparison['scores'],
                               vs_parent=run['report']['comparison']))
    out = ROOT / 'VALIDATION/reports'
    write_json(out/'stage1_comparison.json', dict(control=args.control,
               control_scores=control['report']['scores'], candidates=candidates))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.subplots_adjust(left=.23, right=.97, top=.76, bottom=.23, wspace=.32)
    fig.suptitle('Stage 1 · recovery versus the 800-round control', x=.03,
                 ha='left', fontsize=17, weight='bold')
    fig.text(.03, .85, 'Paired RMSE difference and 95% day-bootstrap interval · lower is better', fontsize=11)
    for ax, group, title in zip(axes, ['R_A','R_B_long'], ['Next month', 'Longer gap']):
        for y, candidate in enumerate(candidates):
            item = candidate['vs_control'][group]
            diff, (lo, hi) = item['difference'], item['ci95']
            color = '#087f8c' if item['decision']=='improvement' else '#b34b36' if item['decision']=='regression' else '#596273'
            ax.plot([lo, hi], [y, y], color=color, linewidth=2)
            ax.plot(diff, y, 'o', color=color, markersize=7)
        ax.axvline(0, color='#999999', linewidth=.8, linestyle='--')
        ax.set_title(f'{title} ({group})', loc='left', fontsize=12)
        ax.set_xlabel('Candidate − control RMSE (seconds)')
        ax.set_yticks(range(3), [c['label'] for c in candidates])
        ax.set_ylim(2.5, -.5)
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top','right','left']].set_visible(False)
    fig.text(.03,.06,'Three seeds · all validation labels · repeated-flight weighting · 2,000 paired UTC-day replicates\n'
             'Reused development folds; these are local validation results, not official leaderboard scores.', fontsize=9, color='#555555')
    for ext in ['png','pdf']:
        fig.savefig(out/f'stage1_comparison.{ext}', dpi=170)
    plt.close(fig)
    print(out/'stage1_comparison.json')


if __name__ == '__main__':
    main()
