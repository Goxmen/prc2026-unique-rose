"""Score the frozen disruption experiment with adopted weights and paired days."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import disruption_run as run
import reporting
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/reports/disruption_experiment'
REPORT = ROOT / 'VALIDATION/reports/DISRUPTION_EXPERIMENT.md'
ALPHA = 152719 / 344841


def read(path):
    return json.loads(path.read_text())


def mse(loss, weights):
    return (loss.sq @ weights) / weights.sum()


def levels(groups, ratios):
    output = {}
    for mode in ('unweighted', 'weighted'):
        per_seed = {name: mse(groups[name], ratios[name] if mode == 'weighted' else np.ones(len(groups[name].meta)))
                    for name in ('R_A', 'R_B_long')}
        per_seed['R_combined'] = ALPHA * per_seed['R_A'] + (1 - ALPHA) * per_seed['R_B_long']
        output[mode] = {name: dict(score=float(np.sqrt(values.mean())),
            by_seed=dict(zip(map(str, groups['R_A'].seeds), np.sqrt(values).tolist()))) for name, values in per_seed.items()}
    return output


def bootstrap_mse(loss, ratios, calendar, draws):
    index = calendar.get_indexer(loss.meta.mvt_time.dt.normalize())
    if (index < 0).any():
        raise ValueError('Missing bootstrap day')
    denominators = np.bincount(index, weights=ratios, minlength=len(calendar))
    numerators = np.bincount(index, weights=ratios * loss.sq.mean(axis=0), minlength=len(calendar))
    with np.errstate(divide='ignore', invalid='ignore'):
        return (draws @ numerators) / (draws @ denominators)


def compare(groups, ratios, scores):
    results = {}
    for block in (1, 3):
        # One calendar and draw matrix shared between arms AND overlapping groups.
        calendar, draws = reporting.bootstrap_calendar(groups['control'], 2000, 0, block_days=block)
        results[str(block)] = {}
        for mode in ('unweighted', 'weighted'):
            boots = {}
            for arm in groups:
                boots[arm] = {name: bootstrap_mse(groups[arm][name],
                    ratios[name] if mode == 'weighted' else np.ones(len(groups[arm][name].meta)), calendar, draws)
                    for name in ('R_A', 'R_B_long')}
                boots[arm]['R_combined'] = ALPHA * boots[arm]['R_A'] + (1 - ALPHA) * boots[arm]['R_B_long']
            results[str(block)][mode] = {}
            for name in ('R_A', 'R_B_long', 'R_combined'):
                estimate = scores['candidate'][mode][name]['score'] - scores['control'][mode][name]['score']
                result = reporting.interval(estimate, np.sqrt(boots['candidate'][name]) - np.sqrt(boots['control'][name]))
                result['difference_by_seed'] = {seed: value - scores['control'][mode][name]['by_seed'][seed]
                    for seed, value in scores['candidate'][mode][name]['by_seed'].items()}
                results[str(block)][mode][name] = result
    return results


def slices(left, right, extra, label):
    reporting.same_flights(left.meta, right.meta)
    extra = extra.set_index('mvt_id').reindex(left.meta.index)
    if not extra.index.equals(left.meta.index) or extra.raw_flight_id_missing.isna().any():
        raise ValueError('Slice metadata misaligned')
    scopes = {'all': np.ones(len(extra), bool)}
    for airport in sorted(left.meta.airport.unique()):
        scopes[airport] = left.meta.airport.eq(airport).to_numpy()
    categories = {'all': np.ones(len(extra), bool), 'missing_id': extra.raw_flight_id_missing.to_numpy(bool),
        'id_present': ~extra.raw_flight_id_missing.to_numpy(bool),
        'snow24h_any': extra.dis_snow_24h_fraction.gt(0).to_numpy(),
        'snow24h_prolonged': extra.dis_snow_24h_fraction.gt(.25).to_numpy(),
        'snow24h_none': extra.dis_snow_24h_fraction.eq(0).to_numpy(),
        'snow24h_unknown': extra.dis_snow_24h_fraction.isna().to_numpy()}
    output = []
    for scope, scope_mask in scopes.items():
        for category, category_mask in categories.items():
            mask = scope_mask & category_mask
            n = int(mask.sum())
            a, b = reporting.slice_stats(left, mask), reporting.slice_stats(right, mask)
            output.append(dict(period=label, airport=scope, category=category, n=n,
                control_rmse=a['rmse'], candidate_rmse=b['rmse'],
                difference=(b['rmse'] - a['rmse']) if n else None, sse_change=b['sse'] - a['sse']))
    return output


def concentration(groups, ratios):
    pieces = []
    for name, alpha in [('R_A', ALPHA), ('R_B_long', 1 - ALPHA)]:
        left, right = groups['control'][name], groups['candidate'][name]
        weights = ratios[name]
        pieces.append(pd.Series(alpha * weights * (right.sq.mean(axis=0) - left.sq.mean(axis=0)) / weights.sum(),
                                index=left.meta.index))
    contributions = pd.concat(pieces).groupby(level=0).sum()
    positive = contributions[contributions > 0].sort_values(ascending=False)
    negative = (-contributions[contributions < 0]).sort_values(ascending=False)
    rows = pd.concat([positive.head(10), -negative.head(10)]).rename('weighted_mse_change').to_frame()
    return dict(total_mse_change=float(contributions.sum()), gross_increase=float(positive.sum()),
        gross_decrease=float(negative.sum()),
        top10_share_of_gross_increase=float(positive.head(10).sum() / positive.sum()) if positive.sum() else 0,
        top10_share_of_gross_decrease=float(negative.head(10).sum() / negative.sum()) if negative.sum() else 0), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=run.DEFAULT)
    args = parser.parse_args(argv)
    directory = args.run.resolve()
    record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
    if record['status'] != 'complete' or len(record['folds']) != 16 or record['fits'] != 18:
        raise ValueError('Expected complete 18-fit/16-fold run')
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
            raise ValueError('Adopted weight bytes changed')
        weights[name] = pd.read_parquet(path).ratio
        if weights[name].index.duplicated().any() or not np.isfinite(weights[name]).all() or (weights[name] <= 0).any():
            raise ValueError('Invalid adopted weights')
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
    scores = {arm: levels(values, ratios) for arm, values in groups.items()}
    reference = read(run.V38 / 'result.json')['candidate_scores']['combined']['scores']
    for name in ('R_A', 'R_B_long', 'R_combined'):
        np.testing.assert_allclose(scores['control']['unweighted'][name]['score'], reference[name], atol=1e-8, rtol=0)
    # The established score is also checked below to its published precision.
    np.testing.assert_allclose(scores['control']['weighted']['R_combined']['score'], 267.5814, atol=.00005, rtol=0)
    changes = compare(groups, ratios, scores)
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
        columns = ['mvt_id', 'raw_flight_id_missing', 'dis_snow_24h_fraction']
        extra = pd.read_parquet(directory / info['predictions']['42'], columns=columns)
        left, right = (winter['control'], winter['candidate']) if name == 'JAN_RETRO' else (losses['control'][name], losses['candidate'][name])
        sliced.extend(slices(left, right, extra, info['val']))
    effects, influential = concentration(groups, ratios)
    all_fold_scores = {name: {arm: reporting.summarize(losses[arm][name]) for arm in losses} for name in record['folds']}
    primary = changes['1']['weighted']['R_combined']
    promising = all(changes[str(block)]['weighted']['R_combined']['ci95'][1] < 0 for block in (1, 3)) and all(
        x < 0 for x in primary['difference_by_seed'].values())
    regression = all(changes[str(block)]['weighted']['R_combined']['ci95'][0] > 0 for block in (1, 3))
    verdict = 'promising_on_reused_folds' if promising else ('regression' if regression else 'inconclusive')
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sliced).to_csv(OUT / 'slices.csv', index=False)
    influential.to_csv(OUT / 'influential_rows.csv')
    result = dict(run=str(directory.relative_to(ROOT)), design_sha256=record['design_sha256'],
        scoring_source_sha256=sha256(Path(__file__)), run_record_sha256=sha256(directory / 'result.json'),
        weights_sha256=weight_hashes, scores=scores, changes=changes, winter_scores=winter_scores,
        winter_changes=winter_changes, folds=all_fold_scores, concentration=effects, verdict=verdict,
        caveats=['Reused development folds; not independent leaderboard confirmation.',
                 'Bootstrap holds learned weights fixed.', 'January is retrospective and excluded from Weighted CV.',
                 'Compares against v38; v40 artifacts unavailable locally.', 'No extreme January 2026 departure labels available.'])
    write_json(OUT / 'results.json', result)
    lines = ['# Snow and arrival-congestion experiment', '',
        '**Verdict: ' + verdict.replace('_', ' ') + '.** One fixed package of 12 inputs added to v38.', '',
        '| Score (seconds; lower is better) | v38 | New features | Change |', '|---|---:|---:|---:|']
    for mode, name, label in [('weighted', 'R_combined', 'Weighted CV'), ('unweighted', 'R_combined', 'Original combined CV'),
                              ('weighted', 'R_A', 'Weighted next-month'), ('weighted', 'R_B_long', 'Weighted longer-gap')]:
        a, b = (scores[arm][mode][name]['score'] for arm in ('control', 'candidate'))
        lines.append(f'| {label} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    a, b = (winter_scores[arm]['rmse'] for arm in ('control', 'candidate'))
    lines.append(f'| January retrospective check | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    lines += ['', 'Weighted CV change, 95% paired intervals: ' + '; '.join(
        f"{block}-day blocks [{changes[str(block)]['weighted']['R_combined']['ci95'][0]:+.2f}, "
        f"{changes[str(block)]['weighted']['R_combined']['ci95'][1]:+.2f}] s" for block in (1, 3)) + '.',
        'Seed changes: ' + ', '.join(f'{seed}: {value:+.2f} s' for seed, value in primary['difference_by_seed'].items()) + '.', '',
        'The model uses past snow duration and taxi times of arrivals completed before takeoff. '
        'The classifier, rules, corrections, weights and all 16 folds stayed fixed. '
        f"All 48 saved v38 prediction files reproduced within 1e-8 s; {record['fits']} ordinary fits completed.", '',
        'January uses June–November models and excludes January departure labels from training. '
        'It is a retrospective diagnostic, outside Weighted CV. Its snow is much milder than January 2026. '
        'These reused folds cannot establish leaderboard transfer. This compares against v38, not v40.', '',
        '[Frozen design](../docs/DISRUPTION_EXPERIMENT.md) · '
        '[Scores and intervals](disruption_experiment/results.json) · '
        '[Airport/month/snow slices](disruption_experiment/slices.csv) · '
        '[Largest individual effects](disruption_experiment/influential_rows.csv)', '',
        'No upload, full-year fit, model promotion or submission-ledger entry.']
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verdict=verdict, scores=scores, primary=primary, winter_changes=winter_changes), indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
