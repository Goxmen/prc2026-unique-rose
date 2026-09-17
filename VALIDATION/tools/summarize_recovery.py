"""Paired recovery slices and loss charts from completed runs; no model fitting."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline'))
from records import read_run, write_json
from common import connect, TRAIN_GLOB
from reporting import (Loss, load_losses, group_loss, slice_stats, bootstrap_calendar,
                       boot_scores, interval, validate_comparison)


def plot_training_loss(run, directory):
    cfg = run['config']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{run['label']} · training RMSE by window and seed", fontsize=16)
    for ax, (name, model) in zip(axes.flat, run['models'].items()):
        for seed in cfg['seeds']:
            curve = json.loads((directory / model['seeds'][str(seed)]['training_loss']).read_text())['rmse']
            ax.plot(np.arange(1, len(curve)+1), curve, label=str(seed), linewidth=1)
        ax.set_title(f"{name}: {model['train_start'][:7]}–{model['train_end'][:7]}")
        ax.set_ylabel('Training RMSE (seconds)')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel('Boosting round')
    axes[0, 0].legend(title='Seed')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(directory / f'training_loss.{ext}', dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--baseline', required=True)
    args = ap.parse_args()
    run, directory = read_run(args.run)
    base, base_dir = read_run(args.baseline)
    losses, controls = load_losses(run, directory), load_losses(base, base_dir)
    validate_comparison(run, base, losses, controls)
    cfg = run['config']
    calendar, weights = bootstrap_calendar(losses, cfg['bootstrap_reps'], cfg['bootstrap_seed'], cfg['block_days'])
    con = connect()
    con.execute('SET threads=2')
    schedule_gap = con.sql('''SELECT MVT_ID_mvt mvt_id,
        epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) gap_seconds
        FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''', params=[TRAIN_GLOB]).df().set_index('mvt_id').gap_seconds
    con.close()
    out = {'candidate': run['run_id'], 'baseline': base['run_id'], 'groups': {},
           'delayed_definition': 'Visible takeoff minus scheduled movement time >= 10800 seconds; diagnostic only.',
           'recovered_slice_definition': 'Flights recovered in any constituent fold; losses retain protocol weighting.'}
    for group, names in [('R_A', [f'A{i}' for i in range(1, 7)]), ('R_B_long', ['B3', 'B4', 'B7'])]:
        candidate, control = group_loss(losses, names), group_loss(controls, names)
        flags = []
        for name in names:
            p = pd.read_parquet(directory / run['folds'][name]['predictions'][str(cfg['seeds'][0])])
            p['recovered'] = ~p.recovery_stage.isin(['unrecovered', 'unchanged'])
            gap = p.mvt_id.map(schedule_gap)
            p['delayed_ge_3h'] = gap.ge(10800)
            p['matched_ordinary_not_delayed'] = ~p.raw_plan_block_missing & p.y.le(7200) & gap.lt(10800)
            flags.append(p.set_index('mvt_id')[['recovered', 'anchor_recovered',
                          'static_imputed', 'raw_flight_id_missing', 'raw_plan_block_missing',
                          'delayed_ge_3h', 'matched_ordinary_not_delayed']])
        flags = pd.concat(flags).groupby(level=0).any().reindex(candidate.meta.index)
        masks = {'recovered': flags.recovered, 'unrecovered': ~flags.recovered,
                 'anchor_recovered': flags.anchor_recovered,
                 'static_imputed': flags.static_imputed,
                 'delayed_ge_3h': flags.delayed_ge_3h,
                 'matched_ordinary_not_delayed': flags.matched_ordinary_not_delayed,
                 'raw_id_missing': flags.raw_flight_id_missing,
                 'raw_plan_block_missing': flags.raw_plan_block_missing,
                 'id_present_empty_plan': ~flags.raw_flight_id_missing & flags.raw_plan_block_missing,
                 'raw_rome_no_id': flags.raw_flight_id_missing & candidate.meta.airport.eq('LIRF'),
                 'raw_rome_no_plan': flags.raw_plan_block_missing & candidate.meta.airport.eq('LIRF'),
                 'at_most_12_seconds': candidate.meta.y.le(12),
                 'ordinary': candidate.meta.y.le(7200), 'tail': candidate.meta.y.gt(7200)}
        items = {}
        for name, mask in masks.items():
            mask = mask.to_numpy(bool)
            a, b = slice_stats(candidate, mask), slice_stats(control, mask)
            entry = {'candidate': a, 'baseline': b}
            if mask.any():
                ca = Loss(candidate.meta.loc[mask], candidate.sq[:, mask], candidate.seeds)
                cb = Loss(control.meta.loc[mask], control.sq[:, mask], control.seeds)
                entry['comparison'] = interval(a['rmse']-b['rmse'],
                    boot_scores(ca, calendar, weights)-boot_scores(cb, calendar, weights))
            items[name] = entry
        out['groups'][group] = items
    write_json(directory / 'recovery_slices.json', out)
    plot_training_loss(run, directory)
    print(directory / 'recovery_slices.json')


if __name__ == '__main__':
    main()
