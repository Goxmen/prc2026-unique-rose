"""Compare fixed aggregation designs; no fold removal or weight fitting to LB."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_fold_alignment as audit
from records import sha256, write_json


def fold_weights(base, design):
    by_month = defaultdict(list)
    for fold in audit.ALL:
        by_month[base['folds'][fold]['val']].append(fold)
    weights = dict.fromkeys(audit.ALL, 0.)
    if design == 'current_combined':
        january_weight = base['ranking_weights']['jan_fraction']
        for names, group_weight in [(audit.A, january_weight), (audit.LONG, 1-january_weight)]:
            months = defaultdict(list)
            for fold in names:
                months[base['folds'][fold]['val']].append(fold)
            counts = {month:base['report']['folds'][folds[0]]['n'] for month,folds in months.items()}
            for month,folds in months.items():
                for fold in folds:
                    weights[fold] += group_weight*counts[month]/sum(counts.values())/len(folds)
    else:
        counts = {month:base['report']['folds'][folds[0]]['n'] for month,folds in by_month.items()}
        for month,folds in by_month.items():
            month_weight = counts[month]/sum(counts.values()) if design == 'unique_flight_pooled' else 1/len(by_month)
            if design in ('month_balanced','unique_flight_pooled'):
                groups = [folds]
            elif design == 'month_balanced_horizon_groups':
                groups = [[f for f in folds if f.startswith(letter)] for letter in ('A','B')]
                groups = [group for group in groups if group]
            else:
                raise ValueError('Unknown aggregation design')
            for group in groups:
                for fold in group:
                    weights[fold] = month_weight/len(groups)/len(group)
    np.testing.assert_allclose(sum(weights.values()),1.,atol=1e-14,rtol=0)
    if any(value < 0 for value in weights.values()):
        raise ValueError('Negative aggregation weight')
    return weights


def main():
    batch = json.loads(audit.MANIFEST.read_text())
    items = {item['codename']:item for item in batch['submissions']}
    records, hashes = {}, {str(audit.MANIFEST.relative_to(audit.ROOT)):sha256(audit.MANIFEST),
                           str(Path(__file__).relative_to(audit.ROOT)):sha256(__file__),
                           str(Path(audit.__file__).relative_to(audit.ROOT)):sha256(audit.__file__)}
    for code in audit.CODES:
        path = audit.ROOT/'VALIDATION/runs'/items[code]['validation_run']/'result.json'
        records[code] = json.loads(path.read_text())
        hashes[str(path.relative_to(audit.ROOT))] = sha256(path)
    base = records['B0-800']
    official = {code:items[code]['score'] for code in audit.CODES}
    designs = ['current_combined','month_balanced','unique_flight_pooled','month_balanced_horizon_groups']
    result = dict(source_hashes=hashes,official=official,reference_date=batch['scores_checked_at'],
                  recommended_candidate='month_balanced',protocol_changed=False,new_fits=0,designs={})
    for design in designs:
        weights = fold_weights(base,design)
        modes = {}
        for mode in audit.MODES:
            scores = {}
            for code in audit.CODES:
                r = records[code]
                if r['config']['seeds'] != [42,142,242] or r['data_identity'] != base['data_identity']:
                    raise ValueError('Reference recipes do not have matched seeds/data')
                mses = {}
                for fold in audit.ALL:
                    row = r['report']['folds'][fold]
                    if row['n'] != base['report']['folds'][fold]['n'] or r['folds'][fold]['val'] != base['folds'][fold]['val']:
                        raise ValueError('Fold populations differ')
                    mses[fold] = row['sse']/row['n'] if mode == 'standard' else row['rmse_by_seed'][mode]**2
                scores[code] = float(np.sqrt(sum(weights[f]*mses[f] for f in weights)))
            modes[mode] = audit.metrics(scores,official,1.)
        month_weights = defaultdict(float)
        for fold,value in weights.items():
            month_weights[base['folds'][fold]['val']] += value
        if design.startswith('month_balanced'):
            np.testing.assert_allclose(list(month_weights.values()),1/6,atol=1e-14,rtol=0)
        result['designs'][design] = dict(fold_weights=weights,month_weights=dict(month_weights),modes=modes,
                                        next_month_weight=sum(weights[f] for f in audit.A),
                                        long_gap_weight=sum(weights[f] for f in audit.LONG))
    old = json.loads((audit.REPORTS/'fold_alignment.json').read_text())
    for design,previous in [('current_combined','current_combined'),('unique_flight_pooled','all16_pooled')]:
        for mode in audit.MODES:
            for code in audit.CODES:
                np.testing.assert_allclose(result['designs'][design]['modes'][mode]['scores'][code],
                                           old['summaries'][previous]['modes'][mode]['scores'][code],atol=1e-9,rtol=0)
    write_json(audit.REPORTS/'aggregation_comparison.json',result)
    rows = []
    for design,entry in result['designs'].items():
        for mode,metrics in entry['modes'].items():
            for code,score in metrics['scores'].items():
                rows.append(dict(design=design,mode=mode,candidate=code,cv_rmse=score,
                                 official_lb=official[code],cv_minus_lb=score-official[code]))
    pd.DataFrame(rows).to_csv(audit.REPORTS/'aggregation_comparison.csv',index=False)
    lines = ['# Month-balanced aggregation candidate', '',
             'The user requested a better overall aggregation without dropping folds. '
             'The candidate below is defined by coverage, not by fitting weights to the five LB scores. '
             'The default harness and historical scores are unchanged.', '',
             '## Proposed overview: equal months, all available folds', '',
             '1. Average squared errors across seeds for each fold.',
             '2. Within each validation month, average those errors across available training origins.',
             '3. Give each of July–December one-sixth of the total MSE; take the square root once.', '',
             'All sixteen folds remain. December has five folds, but they share its 16.67% allocation. '
             'This is not an arithmetic average of fold RMSEs or an ensemble of model predictions.', '',
             '| Month | Current combined weight | Proposed weight |', '|---|---:|---:|']
    for month,weight in result['designs']['current_combined']['month_weights'].items():
        lines.append(f'| {month} | {weight*100:.2f}% | 16.67% |')
    lines += ['', '## Recorded-submission check', '',
              'Three-seed loss averages; errors are in seconds. The one-second direction tolerance '
              'is unchanged. These five related submissions are calibration observations, not an '
              'independent proof that an aggregation is universally better.', '',
              '| Design | Mean absolute LB gap | Mean change error | Recovery-change error | Material directions |',
              '|---|---:|---:|---:|---:|']
    for design,entry in result['designs'].items():
        m=entry['modes']['standard']
        lines.append(f"| {design} | {m['level_mae']:.2f} | {m['delta_mae']:.2f} | {m['recovery_delta_mae']:.2f} | {m['directions']}/3 |")
    lines += ['', '| Candidate | Current combined | Proposed month-balanced | Official LB |',
              '|---|---:|---:|---:|']
    for code in audit.CODES:
        current=result['designs']['current_combined']['modes']['standard']['scores'][code]
        proposed=result['designs']['month_balanced']['modes']['standard']['scores'][code]
        lines.append(f'| {code} | {current:.4f} | {proposed:.4f} | {official[code]:.4f} |')
    lines += ['', '## Horizon tradeoff and interpretation', '',
              'Equal months correct the late-year dominance, but the available roster cannot also '
              'give every month the same forecast horizons: July/August have only A folds. Within-month '
              'equal origins give A weight 100%/100%/50%/33.33%/25%/20% from July to December. '
              'Across the whole score, A receives 54.72%, all B 45.28%, and k≥5 B3/B4/B7 only 10.83%. '
              'That last weight was 55.71% in the old combined score.', '',
              'Therefore retain **R_A, R_B_long and matched-month A/B comparisons** alongside this '
              'candidate overview. A good overall number must not hide a supported long-gap regression. '
              'No January evaluation or long-gap summer evaluation is created by changing weights.', '',
              'Two fixed sensitivities are shown, without selecting a winner from their LB fit: '
              '`unique_flight_pooled` gives each unique flight equal total weight after averaging its '
              'available origins, producing about 15–18% per month. `month_balanced_horizon_groups` '
              'first gives A and B equal shares inside months that have both; it removes dependence '
              'on the number of B origins but adds another convention. Their broad results are similar.', '',
              'The proposed score is structurally more balanced as a general overview, while its '
              'observed change-size accuracy is essentially tied with the old score. Recovery-change '
              'calibration worsens, so separate horizon readouts remain necessary. No claim of '
              'statistically better LB prediction is made; original prediction files would be needed '
              'for new paired uncertainty calculations.', '',
              'Verification: weights sum to one; each proposed month has weight 1/6; existing combined '
              'and unique-flight-pooled results reproduce the previous audit within 1e-9 for all five '
              'recipes and all four seed readouts. No new fits or protocol changes.', '',
              'Reproduce: `.venv/bin/python VALIDATION/tools/compare_aggregations.py`.',
              'Full weights, per-seed scores and controlled changes are in `aggregation_comparison.json`.', '']
    (audit.REPORTS/'AGGREGATION_COMPARISON.md').write_text('\n'.join(lines))
    print(audit.REPORTS/'AGGREGATION_COMPARISON.md')
    for design,entry in result['designs'].items():
        m=entry['modes']['standard']
        print(f"{design}: LB gap {m['level_mae']:.2f}; change error {m['delta_mae']:.2f}; directions {m['directions']}/3")


if __name__ == '__main__':
    main()
