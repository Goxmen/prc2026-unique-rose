"""Score each completed v39/v40 pair with the unchanged adopted Weighted CV."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import opdi_disruption_run as run
import reporting
import score_disruption_experiment as scoring
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/reports/opdi_disruption_experiment'
REPORT = ROOT / 'VALIDATION/reports/OPDI_DISRUPTION_EXPERIMENT.md'


def read(path):
    return json.loads(Path(path).read_text())


def write_report():
    results = {name: read(OUT / name / 'results.json') for name in run.HISTORICAL
               if (OUT / name / 'results.json').exists()}
    lines = ['# New features on v39/v40', '',
        'Same 12 snow-duration and completed-arrival inputs as the v38 test. Lower is better.', '',
        '| Local comparison | Control | New features | Change |', '|---|---:|---:|---:|']
    for name, result in results.items():
        for mode, label in [('weighted', 'Weighted CV'), ('unweighted', 'Original CV')]:
            a, b = [result['scores'][arm][mode]['R_combined']['score'] for arm in ('control', 'candidate')]
            lines.append(f'| {name} {label} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
        a, b = [result['winter_scores'][arm]['rmse'] for arm in ('control', 'candidate')]
        lines.append(f'| {name} January check | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    lines += ['', 'Controls were rebuilt with the same local OPDI cache as their candidates. '
              'Historical cache bytes differ, so gains are measured against these matched controls.', '']
    for name, result in results.items():
        historical = result['historical_comparison']['historical']['weighted']['R_combined']
        control = result['scores']['control']['weighted']['R_combined']['score']
        intervals = '; '.join(f"{block}-day [{result['changes'][str(block)]['weighted']['R_combined']['ci95'][0]:+.3f}, "
                             f"{result['changes'][str(block)]['weighted']['R_combined']['ci95'][1]:+.3f}]"
                             for block in (1, 3))
        seed_changes = result['changes']['1']['weighted']['R_combined']['difference_by_seed']
        lines += [f'- **{name}:** historical Weighted CV {historical:.4f}; rebuilt {control:.4f}. '
                  f'Change intervals: {intervals} s. Seed changes: '
                  + ', '.join(f'{seed}: {delta:+.3f}' for seed, delta in seed_changes.items()) + ' s.']
    lines += ['', 'All 16 folds, adopted weights, rules and recording corrections remain. '
              'Only ordinary regressors receive the new features; each pair shares its classifier. '
              'v40 keeps its training exclusions. January uses June–November models and is outside Weighted CV.', '',
              'These reused historical conditions do not establish performance during January 2026’s longer snow. '
              'The package was tested together, so gains cannot be assigned to snow versus arrivals.', '',
              '[Design](../docs/OPDI_DISRUPTION_EXPERIMENT.md) · '
              '[Detailed results](opdi_disruption_experiment/)', '',
              'No full-year fits, uploads, promotions or `SUBMISSIONS.md` changes.']
    REPORT.write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=sorted(run.HISTORICAL))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args(argv)
    directory = (args.run or run.DEFAULT / args.variant).resolve()
    record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
    if record['status'] != 'complete' or record['fits'] != 54 or len(record['folds']) != 16:
        raise ValueError('Expected complete 54-fit/16-fold paired run')
    for name, digest in record['signature'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed experiment source: {name}')
    run.overnight_run.check_artifacts(directory, progress)
    adopted = read(ROOT / 'VALIDATION/reports/weighted_cv_leaderboard.json')['weights']
    weights, weight_hashes = {}, {}
    for name in ('R_A', 'R_B_long'):
        path = ROOT / adopted[name]['path']
        weight_hashes[name] = sha256(path)
        if weight_hashes[name] != adopted[name]['sha256']:
            raise ValueError('Adopted weights changed')
        weights[name] = pd.read_parquet(path).ratio
        if weights[name].index.duplicated().any() or not np.isfinite(weights[name]).all() or (weights[name] <= 0).any():
            raise ValueError('Invalid weights')
    losses = {arm: reporting.load_losses(dict(record, prediction_column=column), directory)
              for arm, column in [('control', 'pred_control'), ('candidate', 'pred_disruption')]}
    groups = {arm: reporting.score_groups(value) for arm, value in losses.items()}
    ratios = {}
    for name in ('R_A', 'R_B_long'):
        reporting.same_flights(groups['control'][name].meta, groups['candidate'][name].meta)
        ids = groups['control'][name].meta.index
        if len(ids) != len(weights[name]) or len(ids.difference(weights[name].index)):
            raise ValueError('Weight/scoring population differs')
        ratios[name] = weights[name].loc[ids].to_numpy(float)
    scores = {arm: scoring.levels(values, ratios) for arm, values in groups.items()}
    historical_path = ROOT / 'VALIDATION/reports/weighted_cv_recent_submissions.json'
    original = next(row for row in read(historical_path)['submissions'] if row['version'] == int(args.variant[1:]))
    reference = original['modes']['standard']
    historical = dict(historical={mode: reference[mode] for mode in ('weighted', 'unweighted')},
        local_minus_historical={mode: {name: scores['control'][mode][name]['score'] - reference[mode][name]
            for name in ('R_A', 'R_B_long', 'R_combined')} for mode in ('weighted', 'unweighted')},
        historical_record_sha256=sha256(historical_path), historical_cache=record['historical_opdi_identity'],
        local_cache=record['opdi_identity'], exact_historical_predictions_available=False)
    changes = scoring.compare(groups, ratios, scores)
    winter_record = dict(record, folds={'JAN_RETRO': record['january_holdout']})
    winter = {arm: reporting.load_losses(dict(winter_record, prediction_column=column), directory)['JAN_RETRO']
              for arm, column in [('control', 'pred_control'), ('candidate', 'pred_disruption')]}
    winter_scores = {arm: reporting.summarize(loss) for arm, loss in winter.items()}
    winter_changes = {}
    for block in (1, 3):
        calendar, draws = reporting.bootstrap_calendar(winter, 2000, 0, block_days=block)
        estimate = reporting.rmse(winter['candidate']) - reporting.rmse(winter['control'])
        winter_changes[str(block)] = reporting.interval(estimate,
            reporting.boot_scores(winter['candidate'], calendar, draws) - reporting.boot_scores(winter['control'], calendar, draws))
    sliced = []
    for name in [f'A{i}' for i in range(1, 7)] + ['JAN_RETRO']:
        info = record['january_holdout'] if name == 'JAN_RETRO' else record['folds'][name]
        extra = pd.read_parquet(directory / info['predictions']['42'],
                                columns=['mvt_id', 'raw_flight_id_missing', 'dis_snow_24h_fraction'])
        left, right = ((winter['control'], winter['candidate']) if name == 'JAN_RETRO'
                       else (losses['control'][name], losses['candidate'][name]))
        sliced.extend(scoring.slices(left, right, extra, info['val']))
    concentration, influential = scoring.concentration(groups, ratios)
    all_folds = {name: {arm: reporting.summarize(losses[arm][name]) for arm in losses} for name in record['folds']}
    output = OUT / args.variant
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sliced).to_csv(output / 'slices.csv', index=False)
    influential.to_csv(output / 'influential_rows.csv')
    result = dict(variant=args.variant, run=str(directory.relative_to(ROOT)), design_sha256=record['design_sha256'],
        source_sha256=sha256(Path(__file__)), scoring_helpers_sha256=sha256(Path(scoring.__file__)),
        run_record_sha256=sha256(directory / 'result.json'), weights_sha256=weight_hashes,
        historical_comparison=historical, scores=scores, changes=changes, winter_scores=winter_scores,
        winter_changes=winter_changes, folds=all_folds, concentration=concentration,
        caveats=['Matched controls reconstructed; original OPDI cache/model files unavailable.',
                 'Reused development folds; no independent leaderboard confirmation.',
                 'Bootstrap holds learned weights fixed.', 'January retrospective is outside Weighted CV.',
                 'This historical test cannot establish performance under much longer 2026 snow.',
                 'Snow and arrival features tested as a package; no separate attribution.'])
    write_json(output / 'results.json', result)
    write_report()
    print(json.dumps(dict(variant=args.variant, scores=scores, historical=historical,
        primary=changes['1']['weighted']['R_combined'], winter_changes=winter_changes), indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
