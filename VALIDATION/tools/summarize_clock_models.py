"""Summarize completed Stage 3 runs and plot their already-scored comparisons."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import ROOT, read_run, write_json


def comparison(run, directory, baseline):
    pair = run['report'].get('comparison', {})
    if pair.get('baseline_id') == baseline:
        return pair
    return json.loads((directory/f'comparison_{baseline}_block1.json').read_text())['report']['comparison']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', default='v3_20260916T123230629423Z')
    parser.add_argument('--t1', required=True)
    parser.add_argument('--t2', required=True)
    args = parser.parse_args()
    runs = {label: read_run(identifier) for label, identifier in
            [('Q0', args.control), ('T1', args.t1), ('T2', args.t2)]}
    if any(run['status'] != 'complete' for run, _ in runs.values()):
        raise ValueError('all runs must be complete')
    t1, d1 = runs['T1']
    t2, d2 = runs['T2']
    for name in t1['models']:
        for key in ['train_start', 'train_end', 'train_rows', 'features']:
            if t1['models'][name][key] != t2['models'][name][key]:
                raise ValueError(f'T1/T2 differ in training representation: {name}/{key}')
    pairs = {'T1 − Q0': comparison(t1, d1, args.control),
             'T2 − T1': comparison(t2, d2, args.t1),
             'T2 − Q0': comparison(t2, d2, args.control)}
    out = {'runs': {}, 'comparisons': pairs, 'same_t1_t2_training_rows_and_features': True}
    for label, (run, _) in runs.items():
        out['runs'][label] = {'run_id': run['run_id'], 'config': run['config'],
                             'minutes': run['minutes'], 'scores': run['report']['scores'],
                             'groups': run['report']['groups'],
                             'monthly_next_month': {name: values for name, values in run['report']['folds'].items()
                                                   if name.startswith('A')}}
    dest = ROOT/'VALIDATION/reports'
    write_json(dest/'stage3_comparison.json', out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.subplots_adjust(left=.20, right=.97, top=.76, bottom=.23, wspace=.32)
    fig.suptitle('Stage 3 · clock features and residual prediction', x=.03, ha='left', fontsize=16, weight='bold')
    fig.text(.03, .85, 'Paired RMSE difference and 95% day-bootstrap interval · lower is better', fontsize=11)
    for ax, group, title in zip(axes, ['R_A', 'R_B_long'], ['Next month', 'Longer gap']):
        for y, (label, pair) in enumerate(pairs.items()):
            item = pair['scores'][group]
            color = '#087f8c' if item['decision'] == 'improvement' else '#b34b36' if item['decision'] == 'regression' else '#596273'
            ax.plot(item['ci95'], [y, y], color=color, linewidth=2)
            ax.plot(item['difference'], y, 'o', color=color, markersize=7)
        ax.axvline(0, color='#999999', linewidth=.8, linestyle='--')
        ax.set(title=title, xlabel='Candidate − reference RMSE (seconds)',
               yticks=range(len(pairs)), yticklabels=list(pairs), ylim=(2.5, -.5))
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.text(.03, .06, '800 rounds · three seeds · all validation labels · fixed six-hour rule · 2,000 paired UTC-day replicates\n'
             'Reused development folds; these are local results, not official leaderboard scores.', fontsize=9, color='#555555')
    for ext in ['png', 'pdf']:
        fig.savefig(dest/f'stage3_comparison.{ext}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    fig.suptitle('Stage 3 · training RMSE before clipping and rule', fontsize=15)
    for ax, name in zip(axes.flat, t1['models']):
        for label, run, directory, color in [('T1 direct', t1, d1, '#087f8c'), ('T2 residual', t2, d2, '#b34b36')]:
            curves = [json.loads((directory/details['training_loss']).read_text())['rmse']
                      for details in run['models'][name]['seeds'].values()]
            curves = np.asarray(curves)
            x = np.arange(1, curves.shape[1]+1)
            ax.plot(x, curves.mean(axis=0), label=label, color=color)
            ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=color, alpha=.15)
        ax.set(title=name, ylabel='Training RMSE (seconds)')
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False)
    for ax in axes[1]:
        ax.set_xlabel('Boosting round')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(dest/f'stage3_training_loss.{ext}', dpi=160)
    plt.close(fig)
    print(dest/'stage3_comparison.json')


if __name__ == '__main__':
    main()
