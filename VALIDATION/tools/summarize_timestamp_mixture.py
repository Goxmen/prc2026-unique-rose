"""Report completed Stage 4 comparisons and false high predictions from saved OOF."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import ROOT, read_run, write_json
from tools.summarize_clock_models import comparison


def flight_errors(run, directory, names):
    """Average each error/event over seeds and repeated origins, never predictions first."""
    meta = ['airport', 'y', 'raw_flight_id_missing', 'rule_triggered']
    probabilities = ['mixture_p_ordinary', 'mixture_p_schedule', 'mixture_p_day'] if (
        run['config'].get('model_mode') == 'timestamp_mixture') else []
    parts = []
    for name in names:
        base = None
        for seed in run['config']['seeds']:
            x = pd.read_parquet(directory/run['folds'][name]['predictions'][str(seed)],
                                columns=['mvt_id', *meta, 'pred', 'model_pred', *probabilities])
            x = x.set_index('mvt_id').sort_index()
            values = {}
            for output, col in [('final', 'pred'), ('model', 'model_pred')]:
                sq = (x.y-x[col])**2
                high = (x.y <= 7200) & (x[col] > 7200)
                values.update({f'{output}_sq': sq, f'{output}_mean': x[col],
                               f'{output}_high_rate': high.astype(float),
                               f'{output}_high_sq': sq.where(high, 0)})
            values.update({c: x[c] for c in probabilities})
            numeric = pd.DataFrame(values, index=x.index)
            if base is None:
                base = x[meta].copy()
                base[numeric.columns] = 0.
            else:
                pd.testing.assert_frame_equal(base[meta], x[meta])
            base[numeric.columns] += numeric/len(run['config']['seeds'])
        parts.append(base)
    joined = pd.concat(parts).groupby(level=0, sort=True)
    result = joined[meta].first()
    result[numeric.columns] = joined[list(numeric.columns)].mean()
    return result


def slice_stats(x):
    result = {'n': len(x)}
    for output in ['model', 'final']:
        result[output] = {
            'rmse': float(np.sqrt(x[f'{output}_sq'].mean())),
            'false_high_unique_any_seed_origin': int((x[f'{output}_high_rate'] > 0).sum()),
            'false_high_flight_equivalents': float(x[f'{output}_high_rate'].sum()),
            'false_high_sse': float(x[f'{output}_high_sq'].sum()),
            'total_sse': float(x[f'{output}_sq'].sum()),
        }
    return result


def make_plots(out, run, directory, dest):
    pairs = out['comparisons']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.subplots_adjust(left=.18, right=.97, top=.76, bottom=.23, wspace=.32)
    fig.suptitle('Stage 4 · probability mixture of timestamp patterns', x=.03, ha='left', fontsize=15, weight='bold')
    fig.text(.03, .85, 'Paired RMSE difference and 95% day-bootstrap interval · lower is better', fontsize=11)
    for ax, group, title in zip(axes, ['R_A', 'R_B_long'], ['Next month', 'Longer gap']):
        for y, (label, pair) in enumerate(pairs.items()):
            item = pair['scores'][group]
            color = {'improvement': '#087f8c', 'regression': '#b34b36'}.get(item['decision'], '#596273')
            ax.plot(item['ci95'], [y, y], color=color, linewidth=2)
            ax.plot(item['difference'], y, 'o', color=color, markersize=7)
        ax.axvline(0, color='#999999', linewidth=.8, linestyle='--')
        ax.set(title=title, xlabel='Candidate − reference RMSE (seconds)',
               yticks=range(len(pairs)), yticklabels=list(pairs), ylim=(len(pairs)-.5, -.5))
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.text(.03, .06, '800 rounds per component · three seeds · all validation labels · fixed six-hour rule\n'
             '2,000 paired UTC-day replicates · reused development folds, not official scores.', fontsize=9, color='#555555')
    for ext in ['png', 'pdf']:
        fig.savefig(dest/f'stage4_comparison.{ext}', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(17, 6.5), sharex=True, sharey='row')
    fig.suptitle('Stage 4 · component training losses (before mixture and rule)', fontsize=15)
    for col, (name, model) in enumerate(run['models'].items()):
        for row, component, metric, label in [(0, 'gate', 'multi_logloss', 'Probability log loss'),
                                              (1, 'ordinary', 'rmse', 'Ordinary/other RMSE (s)')]:
            curves = np.asarray([json.loads((directory/details['training_loss']).read_text())[
                'components'][component]['training'][metric] for details in model['seeds'].values()])
            ax = axes[row, col]
            x = np.arange(1, curves.shape[1]+1)
            ax.plot(x, curves.mean(axis=0), color='#087f8c')
            ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color='#087f8c', alpha=.15)
            if row == 0:
                ax.set_title(name)
            else:
                ax.set_xlabel('Round')
            if col == 0:
                ax.set_ylabel(label)
            ax.grid(alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(dest/f'stage4_training_loss.{ext}', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--t1', default='v3_20260916T132038893775Z')
    parser.add_argument('--q0', default='v3_20260916T123230629423Z')
    parser.add_argument('--q1', default='v3_20260916T123433565511Z')
    args = parser.parse_args()
    runs = {label: read_run(identifier) for label, identifier in
            [('Q0', args.q0), ('Q1', args.q1), ('T1', args.t1), ('M1', args.run)]}
    if any(r['status'] != 'complete' or not r.get('full_benchmark') for r, _ in runs.values()):
        raise ValueError('all inputs must be complete full benchmarks')
    run, directory = runs['M1']
    for name, model in run['models'].items():
        for field in ['train_start', 'train_end', 'train_rows', 'features']:
            if model[field] != runs['T1'][0]['models'][name][field]:
                raise ValueError(f'mixture/T1 training representation mismatch: {name}/{field}')
    out = {'runs': {}, 'same_t1_m1_features_and_total_training_rows': True,
           'comparisons': {f'M1 − {name}': comparison(run, directory, runs[name][0]['run_id'])
                           for name in ['T1', 'Q0', 'Q1']}, 'false_high_diagnostics': {}}
    for label, (record, _) in runs.items():
        out['runs'][label] = {key: record[key] for key in ['run_id', 'config', 'minutes']}
        out['runs'][label]['report'] = {key: record['report'][key] for key in ['scores', 'groups', 'folds']}
    out['training_components'] = {name: next(iter(model['seeds'].values()))['components']
                                  for name, model in run['models'].items()}
    for group, names in [('R_A', [f'A{i}' for i in range(1, 7)]), ('R_B_long', ['B3', 'B4', 'B7'])]:
        frames = {}
        for label in ['Q0', 'T1', 'M1']:
            record, folder = runs[label]
            x = flight_errors(record, folder, names)
            np.testing.assert_allclose(np.sqrt(x.final_sq.mean()), record['report']['scores'][group], atol=1e-9, rtol=0)
            if frames:
                pd.testing.assert_frame_equal(x[['airport', 'y', 'raw_flight_id_missing', 'rule_triggered']],
                    frames['Q0'][['airport', 'y', 'raw_flight_id_missing', 'rule_triggered']])
            frames[label] = x
        masks = {'ordinary_rome': x.airport.eq('LIRF') & x.y.le(7200),
                 'ordinary_all_airports': x.y.le(7200)}
        masks['ordinary_rome_raw_id_missing'] = masks['ordinary_rome'] & x.raw_flight_id_missing
        masks['ordinary_rome_rule_triggered'] = masks['ordinary_rome'] & x.rule_triggered
        masks['ordinary_rome_not_rule_triggered'] = masks['ordinary_rome'] & ~x.rule_triggered
        slices = {name: {label: slice_stats(frame.loc[mask]) for label, frame in frames.items()}
                  for name, mask in masks.items()}
        x = x.assign(q0_sq=frames['Q0'].final_sq, t1_sq=frames['T1'].final_sq,
                     q0_mean=frames['Q0'].final_mean, t1_mean=frames['T1'].final_mean)
        x['delta_sq_vs_q0'] = x.final_sq-x.q0_sq
        worst = x.loc[masks['ordinary_rome'] & x.final_high_rate.gt(0)].sort_values(
            ['delta_sq_vs_q0', 'mvt_id'], ascending=[False, True]).head(10)
        out['false_high_diagnostics'][group] = {'slices': slices,
            'worst_ordinary_rome_false_high_changes_vs_q0': worst.reset_index().to_dict('records')}
    out['diagnostic_weighting'] = (
        'Squared errors and false-high indicators averaged per flight over seeds and repeated origins; '
        'unique_any counts a flight once if any prediction exceeds 7200 s while y<=7200 s. '
        'Flight-equivalent count sums the averaged indicators. Mean predictions are descriptive only.')
    dest = ROOT/'VALIDATION/reports'
    write_json(dest/'stage4_comparison.json', out)
    make_plots(out, run, directory, dest)
    print(dest/'stage4_comparison.json')


if __name__ == '__main__':
    main()
