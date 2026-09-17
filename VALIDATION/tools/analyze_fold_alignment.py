"""Explore original v3 fold alignment using saved aggregates, without refitting.

Reconstructs repeated-flight weighting, all seed readouts, leave-one-fold/month
summaries and a small set of explicit exploratory subsets. Does not select or
change the validation protocol. New paired intervals require original predictions.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from records import sha256, write_json

REPORTS = ROOT / 'VALIDATION/reports'
MANIFEST = REPORTS / 'leaderboard_selected_20260916.json'
CODES = ['B0-800', 'Q0', 'Q1', 'Q0-4h', 'Q1-4h']
PAIRS = [('Q0', 'B0-800'), ('Q1', 'Q0'), ('Q0-4h', 'Q0'),
         ('Q1-4h', 'Q1'), ('Q1-4h', 'Q0-4h')]
A = [f'A{i}' for i in range(1, 7)]
LONG = ['B3', 'B4', 'B7']
ALL = A + [f'B{i}' for i in range(1, 11)]
MODES = ['standard', '42', '142', '242']


def pool(records, names, mode='standard'):
    """Average per-flight losses across same-month origins before pooling months.

    All folds for a given month evaluate identical populations in protocol v3.
    This makes their mean per-flight MSE reconstructible from total SSE and n;
    uncertainty and individual-row error distributions are not reconstructible.
    """
    if not names or len(names) != len(set(names)):
        raise ValueError('Expected nonempty, unique fold names')
    groups = defaultdict(list)
    base = records[CODES[0]]
    for fold in names:
        groups[base['folds'][fold]['val']].append(fold)
    scores = {}
    for code in CODES:
        total, n_total = 0., 0
        for folds in groups.values():
            parts = [records[code]['report']['folds'][f] for f in folds]
            if len({p['n'] for p in parts}) != 1:
                raise ValueError('Same-month validation populations have different counts')
            count = parts[0]['n']
            mse = [p['sse'] / count if mode == 'standard' else p['rmse_by_seed'][mode]**2 for p in parts]
            total += count * np.mean(mse)
            n_total += count
        scores[code] = float(np.sqrt(total / n_total))
    return scores


def score(records, spec, weight, mode='standard'):
    first = pool(records, spec['a'], mode)
    if not spec.get('b'):
        return first
    second = pool(records, spec['b'], mode)
    return {code: float(np.sqrt(weight * first[code]**2 + (1-weight) * second[code]**2))
            for code in CODES}


def metrics(scores, official, tolerance):
    gaps = np.array([scores[c] - official[c] for c in CODES])
    pairs = []
    for a, b in PAIRS:
        cv, lb = scores[a] - scores[b], official[a] - official[b]
        direction = ('LB within tolerance' if abs(lb) <= tolerance else
                     'CV within tolerance' if abs(cv) <= tolerance else
                     'agree' if np.sign(cv) == np.sign(lb) else 'disagree')
        pairs.append(dict(candidate=a, reference=b, delta_cv=cv, delta_lb=lb,
                          delta_error=cv-lb, direction=direction))
    errors = np.abs([p['delta_error'] for p in pairs])
    material = [p for p in pairs if abs(p['delta_lb']) > tolerance]
    return dict(scores=scores, cv_minus_lb=dict(zip(CODES, gaps.tolist())), pairs=pairs,
                level_mae=float(np.abs(gaps).mean()), signed_gap=float(gaps.mean()),
                delta_mae=float(errors.mean()),
                material_delta_mae=float(np.mean([abs(p['delta_error']) for p in material])),
                threshold_delta_mae=float(errors[[2, 3]].mean()),
                recovery_delta_mae=float(errors[[1, 4]].mean()),
                directions=sum(p['direction'] == 'agree' for p in material),
                disagree=sum(p['direction'] == 'disagree' for p in material),
                negligible_cv=sum(p['direction'] == 'CV within tolerance' for p in material),
                material_pairs=len(material))


def specifications(base):
    specs = {f: dict(a=[f], kind='fold') for f in ALL}
    specs.update({
        'current_combined': dict(a=A, b=LONG, kind='reference'),
        'A_only': dict(a=A, kind='existing_readout'),
        'B_long_only': dict(a=LONG, kind='existing_readout'),
        'all16_pooled': dict(a=ALL, kind='structural_diagnostic'),
        'A_Jul_Sep': dict(a=A[:3], kind='exploratory_subset'),
        'A_Oct_Dec': dict(a=A[3:], kind='exploratory_subset'),
        'A_Aug_Sep': dict(a=A[1:3], kind='exploratory_subset'),
        'A_Jul_Sep_plus_long': dict(a=A[:3], b=LONG, kind='exploratory_subset'),
        'A_Jul_Sep_plus_Dec_long': dict(a=A[:3], b=['B4', 'B7'], kind='exploratory_subset'),
    })
    for fold in A + LONG:
        specs['drop_' + fold] = dict(a=[f for f in A if f != fold],
                                     b=[f for f in LONG if f != fold], kind='leave_one_fold')
    for month in sorted({base['folds'][f]['val'] for f in A}):
        specs['drop_month_' + month] = dict(
            a=[f for f in A if base['folds'][f]['val'] != month],
            b=[f for f in LONG if base['folds'][f]['val'] != month], kind='leave_one_month')
    return specs


def month_weights(base, names):
    counts = {base['folds'][f]['val']: base['report']['folds'][f]['n'] for f in names}
    return {month: count / sum(counts.values()) for month, count in counts.items()}


def concentration(rows):
    # Each unique flight carries one mean error across its three seed predictions.
    gain = -rows.groupby('mvt_id').delta_squared_error.mean()
    total = float(gain.sum())
    ranked = gain.sort_values(ascending=False)
    return dict(flights=len(gain), helped=int((gain > 0).sum()), harmed=int((gain < 0).sum()),
                sse_gain=total,
                largest_gain_share_pct=float(100 * ranked.iloc[0] / total) if total > 0 else None,
                top3_gain_share_pct=float(100 * ranked.head(3).sum() / total) if total > 0 else None)


def diagnostics(records, locations):
    changes = {}
    files = {}
    for code in ('Q0', 'Q0-4h'):
        path = locations[code] / 'changed_rows.csv'
        changes[code] = pd.read_csv(path)
        files[str(path.relative_to(ROOT))] = sha256(path)
    result = {}
    for fold in ALL:
        baseline, q0, q1, four = [records[c]['report']['folds'][fold]
                                 for c in ('B0-800', 'Q0', 'Q1', 'Q0-4h')]
        six = concentration(changes['Q0'].loc[changes['Q0'].fold.eq(fold)])
        extra = changes['Q0-4h'].loc[changes['Q0-4h'].fold.eq(fold)
                                    & changes['Q0-4h'].rule_gap_sec.lt(21600)]
        four_more = concentration(extra)
        for changed, expected in [(six, baseline['sse']-q0['sse']),
                                  (four_more, q0['sse']-four['sse'])]:
            if not np.isclose(changed['sse_gain'], expected, rtol=1e-10, atol=.001):
                raise ValueError(f'Changed-row errors do not reproduce fold totals: {fold}')
        rome = four['airports']['LIRF']
        recovery_gain = q0['sse'] - q1['sse']
        result[fold] = dict(
            six_hour=six, extra_four_hour=four_more,
            q0_4h_ordinary_rmse=four['ordinary']['rmse'],
            q0_4h_tail=four['tail'],
            q0_4h_rome_sse_pct=100 * rome['sse']/four['sse'],
            q0_4h_non_rome_rmse=float(np.sqrt((four['sse']-rome['sse'])/(four['n']-rome['n']))),
            recovery_ordinary_change=q1['ordinary']['rmse']-q0['ordinary']['rmse'],
            recovery_tail_sse_change=q1['tail']['sse']-q0['tail']['sse'],
            recovery_rome_share_of_net_sse_gain_pct=(
                100 * (q0['airports']['LIRF']['sse']-q1['airports']['LIRF']['sse'])/recovery_gain
                if recovery_gain > 0 else None))
    return result, files


def figure(result, prefix):
    names = ['current_combined', 'A_only', 'B_long_only'] + ALL
    rows = [result['summaries'][name]['modes']['standard'] for name in names]
    levels = np.array([[row['cv_minus_lb'][c] for c in CODES] for row in rows])
    deltas = np.array([[p['delta_cv'] for p in row['pairs']] for row in rows])
    lb_delta = np.array([p['delta_lb'] for p in rows[0]['pairs']])
    fig, axes = plt.subplots(1, 2, figsize=(14, 10.8), layout='constrained')
    labels = ['Current combined', 'A only', 'B long only'] + [
        f"{f} · {result['folds'][f]['val'][5:]} · gap {result['folds'][f]['gap']}" for f in ALL]
    for ax, matrix, title, ticks, bound in [
        (axes[0], levels, 'Absolute score mismatch: CV − LB', CODES, 140),
        (axes[1], deltas-lb_delta, 'Change mismatch: ΔCV − ΔLB',
         ['6h rule\nbase', 'Recovery\nat 6h', '6h → 4h\nbase', '6h → 4h\nR3', 'Recovery\nat 4h'], 85),
    ]:
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
        ax.set_title(title, loc='left', fontsize=13, pad=14)
        ax.set_xticks(range(5), ticks)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f'{value:+.1f}', ha='center', va='center', fontsize=8.5,
                        color='white' if abs(value) > bound*.62 else '#202020')
        ax.axhline(2.5, color='black', linewidth=1)
        ax.axhline(8.5, color='black', linewidth=1)
        fig.colorbar(im, ax=ax, shrink=.65, label='Seconds; zero is closest')
    fig.suptitle('Original validation folds versus five recorded LB scores\n'
                 'Three-seed loss averages · descriptive comparisons, no new uncertainty intervals', fontsize=15)
    for ext in ('png', 'pdf'):
        fig.savefig(prefix.with_suffix('.' + ext), dpi=170)
    plt.close(fig)


def write_markdown(result, prefix):
    summaries, base = result['summaries'], result['folds']
    lines = ['# Original fold alignment with the five recorded leaderboard scores', '',
             'Exploratory analysis requested 2026-09-17. No fits, uploads or fold-definition changes.',
             f"Official reference timestamp: {result['scores_checked_at']}. All values are RMSE seconds.", '',
             '## What the current headline averages', '',
             'The combined score uses **A1–A6 and B3/B4/B7**. Other B folds are diagnostics. '
             'Losses are averaged across seeds and repeated predictions of a flight before pooling; '
             'fold RMSEs are never averaged arithmetically. The A/long-B weights remain '
             'the fixed January/July ranking-population weights.', '',
             '| Validation month | Effective combined squared-error weight |', '|---|---:|']
    for month, weight in result['headline_month_weights'].items():
        lines.append(f'| {month} | {100*weight:.3f}% |')
    lines += ['', '## Every individual fold', '',
              'Level error is mean |CV−LB| over five candidates. Change error is mean |ΔCV−ΔLB| '
              'over the five fixed pairs. Direction uses the same 1-second tolerance; only three LB '
              'changes exceed it. All three concern the Rome rule, so 3/3 is limited evidence.', '',
              '| Fold | Train | Validate / gap | In headline? | Level error | Change error | Directions / 3 |',
              '|---|---|---|---|---:|---:|---:|']
    for fold in ALL:
        meta, row = base[fold], summaries[fold]['modes']['standard']
        lines.append(f"| {fold} | {meta['train_start'][:7]}–{meta['train_end'][:7]} | {meta['val']} / {meta['gap']} | "
                     f"{'yes' if fold in A+LONG else 'diagnostic'} | {row['level_mae']:.2f} | {row['delta_mae']:.2f} | {row['directions']}/3 |")
    lines += ['', '## Raw scores and controlled changes', '',
              '| Fold | B0-800 | Q0 | Q1 | Q0-4h | Q1-4h |', '|---|---:|---:|---:|---:|---:|',
              '| Official LB | ' + ' | '.join(f"{result['official'][c]:.2f}" for c in CODES) + ' |']
    for fold in ALL:
        row = summaries[fold]['modes']['standard']
        lines.append(f"| {fold} | " + ' | '.join(f"{row['scores'][c]:.2f}" for c in CODES) + ' |')
    lines += ['', '| Fold | Six-hour rule | Recovery at 6h | 6h→4h baseline | 6h→4h recovery | Recovery at 4h |',
              '|---|---:|---:|---:|---:|---:|',
              '| Official LB | ' + ' | '.join(f"{result['official'][a]-result['official'][b]:+.2f}" for a,b in PAIRS) + ' |']
    for fold in ALL:
        row = summaries[fold]['modes']['standard']
        lines.append(f"| {fold} | " + ' | '.join(f"{p['delta_cv']:+.2f}" for p in row['pairs']) + ' |')
    lines += ['', '## Alternative summaries and deletion checks', '',
              'All subsets below are exploratory and chosen after inspecting these results. '
              'No optimized weights or exhaustive subset search was used. A+B variants preserve '
              'the existing A/B mixing weights; pooling within each group respects unique flights.', '',
              '| Summary | Level error | Change error | Threshold-change error | Recovery-change error | Directions / 3 | Seed-42 level / change | Fits × 3 seeds if run alone |',
              '|---|---:|---:|---:|---:|---:|---|---:|']
    for name, entry in summaries.items():
        if entry['spec']['kind'] == 'fold':
            continue
        row, one = entry['modes']['standard'], entry['modes']['42']
        lines.append(f"| {name} | {row['level_mae']:.2f} | {row['delta_mae']:.2f} | "
                     f"{row['threshold_delta_mae']:.2f} | {row['recovery_delta_mae']:.2f} | {row['directions']}/3 | "
                     f"{one['level_mae']:.2f} / {one['delta_mae']:.2f} | {entry['model_windows']*3} |")
    lines += ['', 'Fit counts refer to running only that summary. Removing B evaluations while retaining '
              'all A folds saves **no model fits**, since B reuses A models.', '',
              '## Concentration of baseline rule gains', '',
              'Counts are unique flights per fold; losses are averaged across seeds. The largest-row '
              'share is its contribution to the net squared-error gain, so harmed rows can make a '
              'share exceed 100%. The same flights recur in different training-origin folds.', '',
              '| Fold | Six-hour flights | Extra four-hour flights | Largest extra flight share of net gain | Q0-4h ordinary RMSE |',
              '|---|---:|---:|---:|---:|']
    for fold, detail in result['diagnostics'].items():
        extra = detail['extra_four_hour']
        share = extra['largest_gain_share_pct']
        lines.append(f"| {fold} | {detail['six_hour']['flights']} | {extra['flights']} | "
                     + ('—' if share is None else f'{share:.1f}%')
                     + f" | {detail['q0_4h_ordinary_rmse']:.2f} |")
    lines += ['', '## Interpretation', '',
              '- September A3 is the most promising individual fold on both score level and change size '
              'versus the current combined reference. August+September is a strong exploratory change '
              'readout. Its threshold result is sparse: A3 adds six flights, with one supplying about '
              '56% of net gain. The older-origin September B1 sees the same flights but much less gain.',
              '- November cannot detect the 4h-vs-6h threshold change; December has only two extra cases. '
              'That explains why the long-gap score underestimates this particular intervention.',
              '- July is much harder even on ordinary flights and outside Rome. Removing it improves '
              'some change metrics but substantially worsens absolute score closeness and removes '
              'our only summer July evaluation.',
              '- Recovery improves summer folds; A5/A6 and all three headline long-gap folds '
              '(B3/B4/B7) regress under every seed. '
              'Removing winter coverage risks exaggerating recovery, as the seasonal experiments did. '
              'The current mixture is especially close to the tiny observed recovery LB changes.',
              '- A-only is closer in levels and average changes than the current combined score, but '
              'has larger errors on the recovery comparisons. July–September is closest in levels '
              'while overstating recovery/rule gains. These are different tradeoffs, not one clear winner.',
              '- Retain the current folds while recording these alternate summaries. Treat R_A as '
              'the primary next-month readout and keep long-gap results as a separate guardrail; '
              'the protocol already treats their combined number as provisional. Confirm any proposed '
              'reduction with a later candidate before changing the benchmark.', '',
              '## Evidence limits and reproducibility', '',
              'The original prediction parquets are absent here. Saved per-fold SSE/count/seed summaries '
              'reproduce the existing A, long-B and combined values exactly. They support these point '
              'estimates and seed checks, but not new paired day-bootstrap intervals or a significance '
              'claim that one fold/subset aligns better. Restore prediction artifacts for those checks; '
              'no model refit is needed merely to inspect saved predictions.',
              'The five submissions use only two parent models and highly related rule variants. '
              'Selecting folds against them is calibration on those observations, not independent '
              'validation. Opposite signs on sub-second LB changes are not counted as failures, but '
              'large predicted recovery gains remain visible in change-error metrics.', '',
              'Reproduce: `.venv/bin/python VALIDATION/tools/analyze_fold_alignment.py`.',
              'All four seed readouts, individual deltas, exact subset memberships, source hashes '
              'and slice diagnostics are in `fold_alignment.json`; tabular metrics are in '
              '`fold_alignment.csv`.', '', '![Fold score and change mismatches](fold_alignment.png)', '']
    (prefix.parent / 'FOLD_ALIGNMENT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPORTS / 'fold_alignment')
    args = parser.parse_args()
    batch = json.loads(MANIFEST.read_text())
    items = {item['codename']: item for item in batch['submissions']}
    records, locations, hashes = {}, {}, {str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST)}
    for code in CODES:
        path = ROOT / 'VALIDATION/runs' / items[code]['validation_run'] / 'result.json'
        records[code], locations[code] = json.loads(path.read_text()), path.parent
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        r = records[code]
        if r['status'] != 'complete' or r['config']['seeds'] != [42,142,242] or r['config']['rounds'] != 800:
            raise ValueError('Expected the completed, matched 800-round three-seed recipes')
        if items[code]['official_result']['status'] != 'Succeeded':
            raise ValueError('Reference is not an official successful submission')
    base = records['B0-800']
    for r in records.values():
        if r['data_identity'] != base['data_identity']:
            raise ValueError('Candidate datasets differ')
        for fold in ALL:
            if r['folds'][fold]['val'] != base['folds'][fold]['val'] or r['report']['folds'][fold]['n'] != base['report']['folds'][fold]['n']:
                raise ValueError('Fold populations differ')
    weight = base['ranking_weights']['jan_fraction']
    specs = specifications(base)
    official = {code: items[code]['score'] for code in CODES}
    summaries = {}
    for name, spec in specs.items():
        selected = set(spec['a'] + spec.get('b', []))
        summaries[name] = dict(spec=spec, model_windows=len({base['folds'][f]['model_id'] for f in selected}),
                               modes={mode: metrics(score(records,spec,weight,mode),official,1.) for mode in MODES})
    for code in CODES:
        for name, metric in [('A_only','R_A'), ('B_long_only','R_B_long'), ('current_combined','R_combined')]:
            np.testing.assert_allclose(summaries[name]['modes']['standard']['scores'][code],
                                       items[code]['cv_standard'][metric], atol=1e-9, rtol=0)
            np.testing.assert_allclose(summaries[name]['modes']['42']['scores'][code],
                                       items[code]['cv_seed42'][metric], atol=1e-9, rtol=0)
    details, extra_hashes = diagnostics(records, locations)
    aw, bw = month_weights(base,A), month_weights(base,LONG)
    result = dict(scores_checked_at=batch['scores_checked_at'], official=official,
                  tolerance_seconds=1., source_hashes={**hashes, **extra_hashes, str(Path(__file__).relative_to(ROOT)):sha256(__file__)},
                  headline_month_weights={m:weight*aw.get(m,0)+(1-weight)*bw.get(m,0) for m in aw},
                  folds={f:dict(train_start=base['folds'][f]['train_start'],train_end=base['folds'][f]['train_end'],
                                val=base['folds'][f]['val'],gap=int(base['folds'][f]['val'][5:])-int(base['folds'][f]['train_end'][5:7])) for f in ALL},
                  summaries=summaries, diagnostics=details,
                  predictions_available=False, new_fits=0, protocol_changed=False)
    prefix = args.output
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(prefix.with_suffix('.json'), result)
    rows = []
    for name, item in summaries.items():
        for mode, row in item['modes'].items():
            flat = {k:v for k,v in row.items() if k not in ('scores','cv_minus_lb','pairs')}
            flat.update(summary=name,mode=mode,kind=item['spec']['kind'],model_windows=item['model_windows'])
            flat.update({f'rmse_{c}':row['scores'][c] for c in CODES})
            flat.update({f'delta_{p["candidate"]}_minus_{p["reference"]}':p['delta_cv'] for p in row['pairs']})
            rows.append(flat)
    pd.DataFrame(rows).to_csv(prefix.with_suffix('.csv'),index=False)
    figure(result,prefix)
    write_markdown(result,prefix)
    print(prefix.parent / 'FOLD_ALIGNMENT.md')
    for name in ['current_combined','A_only','A_Aug_Sep','A_Jul_Sep','B_long_only']:
        row = summaries[name]['modes']['standard']
        print(f"{name}: level MAE {row['level_mae']:.2f}, change MAE {row['delta_mae']:.2f}, "
              f"recovery change MAE {row['recovery_delta_mae']:.2f}, directions {row['directions']}/3")


if __name__ == '__main__':
    main()
