"""January/July scores, paired uncertainty and alignment with frozen LB scores."""

from pathlib import Path

import numpy as np
import pandas as pd

from reporting import (COVERAGE, IDENTITY, Loss, boot_scores, bootstrap_calendar,
                       interval, rmse, same_flights, slice_stats, summarize)

PAIRS = [('Q0', 'B0-800'), ('Q1', 'Q0'), ('Q0-4h', 'Q0'),
         ('Q1-4h', 'Q1'), ('Q1-4h', 'Q0-4h')]
MONTHS = ('2025-01', '2025-07')


def combined(january, july, jan_fraction):
    return np.sqrt(jan_fraction * np.asarray(january)**2
                   + (1 - jan_fraction) * np.asarray(july)**2)


def agreement(delta_cv, delta_lb, tolerance):
    if abs(delta_lb) <= tolerance:
        return 'LB change within tolerance'
    if abs(delta_cv) <= tolerance:
        return 'CV change within tolerance'
    return 'agree' if np.sign(delta_cv) == np.sign(delta_lb) else 'disagree'


def load_candidate_losses(directory, record, progress, strategy):
    from seasonal import RECIPES
    seeds = record['signature']['config']['seeds']
    output = {name: {} for name in RECIPES}
    for month in MONTHS:
        for parent in ('baseline', 'recovery'):
            frames = []
            for seed in seeds:
                matches = [p for p in progress['predictions'].values()
                           if (p['strategy'], p['month'], p['parent'], p['seed'])
                           == (strategy, month, parent, seed)]
                if len(matches) != 1:
                    raise ValueError('Missing or duplicate seasonal fold/seed')
                frame = pd.read_parquet(directory / matches[0]['path'])
                if (frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any()
                        or not frame.seed.eq(seed).all() or not frame.month.eq(month).all()):
                    raise ValueError('Invalid prediction identity')
                frame = frame.set_index('mvt_id').sort_index()
                frame['mvt_time'] = pd.to_datetime(frame.mvt_time, utc=True)
                if not frame.mvt_time.dt.strftime('%Y-%m').eq(month).all():
                    raise ValueError('Prediction timestamp outside held-out month')
                if frames:
                    same_flights(frames[0], frame)
                frames.append(frame)
            for name, (model_parent, _) in RECIPES.items():
                if parent != model_parent:
                    continue
                sq = np.stack([(f[name].to_numpy() - f.y.to_numpy())**2 for f in frames])
                if not np.isfinite(sq).all():
                    raise ValueError('Nonfinite seasonal errors')
                meta = frames[0][IDENTITY + COVERAGE + ['raw_flight_id_missing']].copy()
                output[name][month] = Loss(meta, sq, tuple(seeds))
        for name in output:
            same_flights(output['B0-800'][month].meta, output[name][month].meta)
    return output


def alignment(levels, pairs, tolerance):
    gaps = np.array([row['lb_minus_cv']['combined'] for row in levels.values()])
    material = [row for row in pairs if abs(row['delta_lb']) > tolerance]
    return {
        'mean_absolute_score_gap': float(np.abs(gaps).mean()),
        'root_mean_square_score_gap': float(np.sqrt(np.mean(gaps**2))),
        'mean_signed_lb_minus_cv': float(gaps.mean()),
        'mean_absolute_delta_gap': float(np.mean([
            abs(row['metrics']['combined']['delta_lb_minus_delta_cv']) for row in pairs])),
        'material_lb_pairs': len(material),
        'direction_agree': sum(row['metrics']['combined']['agreement'] == 'agree' for row in material),
        'direction_disagree': sum(row['metrics']['combined']['agreement'] == 'disagree' for row in material),
        'cv_within_tolerance': sum(row['metrics']['combined']['agreement'] == 'CV change within tolerance'
                                   for row in material),
    }


def evaluate(losses, items, jan_fraction, config):
    calendar, weights = bootstrap_calendar(losses['B0-800'], config['bootstrap_reps'], 0)
    modes = {}
    for mode in ('standard', 'seed42'):
        levels, bootstrap = {}, {}
        for name, months in losses.items():
            selected = {month: loss if mode == 'standard' else
                        Loss(loss.meta, loss.sq[[loss.seeds.index(42)]], (42,))
                        for month, loss in months.items()}
            scores = {month: rmse(loss) for month, loss in selected.items()}
            scores['combined'] = float(combined(scores[MONTHS[0]], scores[MONTHS[1]], jan_fraction))
            draws = {month: boot_scores(loss, calendar, weights) for month, loss in selected.items()}
            draws['combined'] = combined(draws[MONTHS[0]], draws[MONTHS[1]], jan_fraction)
            bootstrap[name] = draws
            official = items[name]['score']
            levels[name] = dict(scores=scores, official_lb=official,
                                lb_minus_cv={key: official - value for key, value in scores.items()})
        comparisons = []
        for candidate, reference in PAIRS:
            a, b = levels[candidate], levels[reference]
            delta_lb = a['official_lb'] - b['official_lb']
            metrics = {}
            for key in a['scores']:
                delta = a['scores'][key] - b['scores'][key]
                metrics[key] = dict(interval(delta, bootstrap[candidate][key] - bootstrap[reference][key],
                                             eligible=not config['smoke']),
                                    delta_lb_minus_delta_cv=delta_lb - delta,
                                    agreement=agreement(delta, delta_lb, config['negligible_sec']))
            comparisons.append(dict(candidate=candidate, reference=reference, delta_lb=delta_lb,
                                    metrics=metrics,
                                    supported_improvement_both_months=all(
                                        metrics[month]['decision'] == 'improvement' for month in MONTHS)))
        modes[mode] = dict(levels=levels, comparisons=comparisons,
                           alignment=alignment(levels, comparisons, config['negligible_sec']))
    slices = {}
    for name, months in losses.items():
        slices[name] = {}
        for month, loss in months.items():
            stats = summarize(loss)
            stats['all_airports_aobt3'] = {
                label: slice_stats(loss, loss.meta.has_flight_plan.eq(present))
                for label, present in [('present', True), ('missing', False)]}
            slices[name][month] = stats
    return dict(modes=modes, slices=slices)


def old_alignment(items, mode, tolerance):
    field = 'cv_standard' if mode == 'standard' else 'cv_seed42'
    levels = {name: {'lb_minus_cv': {'combined': item['score'] - item[field]['R_combined']}}
              for name, item in items.items()}
    pairs = []
    for candidate, reference in PAIRS:
        a, b = items[candidate], items[reference]
        delta_lb, delta_cv = a['score'] - b['score'], a[field]['R_combined'] - b[field]['R_combined']
        pairs.append(dict(delta_lb=delta_lb, metrics={'combined': {
            'delta_lb_minus_delta_cv': delta_lb - delta_cv,
            'agreement': agreement(delta_cv, delta_lb, tolerance)}}))
    return alignment(levels, pairs, tolerance)


def report(directory, record, progress):
    config, items = record['signature']['config'], record['references']
    jan_fraction = record['signature']['ranking_weights']['jan_fraction']
    results = {}
    for strategy in sorted({j['strategy'] for j in record['signature']['jobs']}):
        losses = load_candidate_losses(directory, record, progress, strategy)
        results[strategy] = evaluate(losses, items, jan_fraction, config)
    old = {mode: old_alignment(items, mode, config['negligible_sec']) for mode in ('standard', 'seed42')}
    lines = ['# Seasonal validation against recorded leaderboard scores', '',
             '**EXECUTION CHECK ONLY — sampled data and five rounds. Do not rank strategies.**' if config['smoke'] else
             'Retrospective 2025 holdouts; the official scores below are fixed references, not new submissions.', '',
             f"Official results recorded {record['scores_checked_at']}. Seeds: {config['seeds']}.",
             'Official LB is pooled across both months; no official January-only or July-only scores are available.',
             'January and July are scored separately; combined RMSE weights monthly mean squared errors '
             f'by the 2026 ranking populations (January weight {jan_fraction:.8f}).',
             'Standard scores average squared errors across seeds, not predictions. Seed 42 matches the submission seed.',
             'Training features remain causal within retained months. Validation departure history is frozen '
             'at December 31, using training-month labels only. Future-in-calendar training is intentional; '
             'these scores do not measure forward-year forecasting.', '',
             f"Direction tolerance: {config['negligible_sec']:.2f} s, applied to both LB and CV deltas. "
             'Pointwise paired 95% intervals resample calendar days within each held-out month.',
             'These five related submissions are the calibration set; later candidates must test whether alignment generalizes.', '',
             '## Alignment with the same five LB references', '',
             '| Strategy | Seed readout | Mean absolute score gap | Mean absolute delta gap | Agree / material LB pairs | Disagree | CV within tolerance |',
             '|---|---|---:|---:|---:|---:|---:|']
    for strategy, modes in [('existing-v3', old)] + [(s, {m: r['alignment'] for m, r in v['modes'].items()})
                                                   for s, v in results.items()]:
        for mode, row in modes.items():
            lines.append(f"| {strategy} | {mode} | {row['mean_absolute_score_gap']:.4f} | "
                         f"{row['mean_absolute_delta_gap']:.4f} | {row['direction_agree']} / "
                         f"{row['material_lb_pairs']} | {row['direction_disagree']} | {row['cv_within_tolerance']} |")
    for strategy, result in results.items():
        for mode, values in result['modes'].items():
            lines += ['', f'## {strategy}: {mode}', '',
                      '| Recipe | January RMSE | July RMSE | Combined RMSE | Official LB | LB − combined |',
                      '|---|---:|---:|---:|---:|---:|']
            for name, row in values['levels'].items():
                scores = row['scores']
                lines.append(f"| {name} | {scores[MONTHS[0]]:.4f} | {scores[MONTHS[1]]:.4f} | "
                             f"{scores['combined']:.4f} | {row['official_lb']:.4f} | {row['lb_minus_cv']['combined']:.4f} |")
            lines += ['', '| Candidate − reference | ΔLB | ΔJanuary | ΔJuly | ΔCombined [95% interval] | Direction | Both months supported gains |',
                      '|---|---:|---:|---:|---|---|---|']
            for row in values['comparisons']:
                metrics, summary = row['metrics'], row['metrics']['combined']
                lo, hi = summary['ci95']
                lines.append(f"| {row['candidate']} − {row['reference']} | {row['delta_lb']:+.4f} | "
                             f"{metrics[MONTHS[0]]['difference']:+.4f} | {metrics[MONTHS[1]]['difference']:+.4f} | "
                             f"{summary['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {summary['agreement']} | "
                             f"{row['supported_improvement_both_months']} |")
    lines += ['', 'Full per-month intervals, seed variation, airport/tail/AOBT3 slices, '
              'and LB-minus-monthly-CV gaps are in `result.json`. Unrounded predictions, '
              'models and training-loss curves are saved beside it.', '']
    (Path(directory) / 'comparison.md').write_text('\n'.join(lines))
    return {'strategies': results, 'existing_v3_alignment': old}
