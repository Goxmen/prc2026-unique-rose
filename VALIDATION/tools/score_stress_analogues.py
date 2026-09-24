"""Paired errors on genuinely bounded test-like historical conditions."""
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import opdi_disruption_run as run
import prepare_stress_analogues as prepare
import reporting
import stress_analogues as matching
import stress_holdout_run as supplement
from records import sha256, write_json

SEEDS = (42, 142, 242)
REPORT = ROOT / 'VALIDATION/reports/OPDI_DISRUPTION_STRESS_CHECK.md'


def read(path):
    return json.loads(Path(path).read_text())


def load_view(variant, inputs):
    directory = run.DEFAULT / variant
    record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
    extra = read(supplement.DEFAULT / 'result.json')
    if record['status'] != 'complete' or extra['status'] != 'complete':
        raise ValueError('Paired predictions incomplete')
    choices = [(name, record['folds'][name], directory, progress) for name in [f'A{i}' for i in range(1, 7)]]
    choices.append(('JAN_RETRO', record['january_holdout'], directory, progress))
    choices += [(name, info, supplement.DEFAULT / variant, extra['variants'][variant])
                for name, info in extra['variants'][variant]['folds'].items()]
    frames = []
    meta = inputs.set_index('mvt_id')
    for name, info, path, artifacts in choices:
        first, old, new, predictions_old, predictions_new = None, [], [], [], []
        for seed in SEEDS:
            run.prior.checked_file(path, artifacts['predictions'][f'{name}__{seed}'])
            columns = ['mvt_id', 'mvt_time', 'airport', 'y', 'seed', 'fold', 'raw_flight_id_missing',
                       'pred_control', 'pred_disruption', *run.features.FEATURES]
            frame = pd.read_parquet(path / info['predictions'][str(seed)], columns=columns).set_index('mvt_id').sort_index()
            if frame.index.duplicated().any() or not frame.seed.eq(seed).all() or not frame.fold.eq(name).all():
                raise ValueError('Wrong supplemental row/seed/fold population')
            if not np.isfinite(frame[['y', 'pred_control', 'pred_disruption']].to_numpy()).all():
                raise ValueError('Nonfinite paired error')
            if first is None:
                first = frame.copy()
            else:
                if not first.index.equals(frame.index):
                    raise ValueError('Seed populations differ')
                pd.testing.assert_frame_equal(first[['y', *run.features.FEATURES]], frame[['y', *run.features.FEATURES]])
            old.append((frame.pred_control.to_numpy() - frame.y.to_numpy()) ** 2)
            new.append((frame.pred_disruption.to_numpy() - frame.y.to_numpy()) ** 2)
            predictions_old.append(frame.pred_control.to_numpy())
            predictions_new.append(frame.pred_disruption.to_numpy())
        out = meta.loc[first.index].copy()
        np.testing.assert_array_equal(pd.to_datetime(out.mvt_time, utc=True), pd.to_datetime(first.mvt_time, utc=True))
        np.testing.assert_array_equal(out.airport, first.airport)
        np.testing.assert_array_equal(out.raw_flight_id_missing, first.raw_flight_id_missing)
        # Match on the inputs actually available in this held-out evaluation,
        # including its restricted arrival donor months at month boundaries.
        same_donors = (out[matching.COUNTS].to_numpy() == first[matching.COUNTS].to_numpy()).all(axis=1)
        # Allowed arrivals are a subset of the full-history cache; equal counts
        # prove equal donors. Reject boundary rows where that is not established.
        out.loc[~same_donors, matching.DIAGNOSTICS] = np.nan
        out[run.features.FEATURES] = first[run.features.FEATURES]
        out['y'] = first.y
        out['fold'], out['model_id'] = name, info['model_id']
        out['retrospective'] = name.startswith(('JAN_', 'RETRO_'))
        out['pred_control_mean'] = np.mean(predictions_old, axis=0)
        out['pred_disruption_mean'] = np.mean(predictions_new, axis=0)
        for i, seed in enumerate(SEEDS):
            out[f'old_sq_{seed}'], out[f'new_sq_{seed}'] = old[i], new[i]
        frames.append(out)
    result = pd.concat(frames).sort_index()
    if result.index.duplicated().any():
        raise ValueError('Repeated historical flight in stress scoring')
    return result


def paired_score(frame):
    n = len(frame)
    dates = int(pd.to_datetime(frame.mvt_time, utc=True).dt.normalize().nunique())
    if not n:
        return dict(rows=0, dates=0, control=None, candidate=None, difference=None, evidence='no_matched_rows')
    old = frame[[f'old_sq_{s}' for s in SEEDS]].to_numpy().T
    new = frame[[f'new_sq_{s}' for s in SEEDS]].to_numpy().T
    scores = np.sqrt([old.mean(), new.mean()])
    change = float(scores[1] - scores[0])
    losses = {name: reporting.Loss(frame[['mvt_time']].copy(), squares, SEEDS)
              for name, squares in [('control', old), ('candidate', new)]}
    intervals = {}
    for block in (1, 3):
        calendar, draws = reporting.bootstrap_calendar(losses, 2000, 0, block_days=block)
        intervals[str(block)] = reporting.interval(change,
            reporting.boot_scores(losses['candidate'], calendar, draws) - reporting.boot_scores(losses['control'], calendar, draws))
    seed_changes = np.sqrt(new.mean(axis=1)) - np.sqrt(old.mean(axis=1))
    enough = n >= 100 and dates >= 10
    evidence = 'too_few_flights_or_dates' if not enough else 'inconclusive'
    if enough and all(intervals[str(b)]['ci95'][1] < 0 for b in (1, 3)) and (seed_changes < 0).all():
        evidence = 'supported_gain_on_matched_historical_rows'
    elif enough and all(intervals[str(b)]['ci95'][0] > 0 for b in (1, 3)) and (seed_changes > 0).all():
        evidence = 'supported_regression_on_matched_historical_rows'
    difference = new.mean(axis=0) - old.mean(axis=0)
    return dict(rows=n, dates=dates, control=float(scores[0]), candidate=float(scores[1]), difference=change,
        by_seed=dict(zip(map(str, SEEDS), seed_changes.tolist())), intervals=intervals, evidence=evidence,
        improved=int((difference < 0).sum()), worsened=int((difference > 0).sum()), unchanged=int((difference == 0).sum()),
        retrospective_rows=int(frame.retrospective.sum()))


def choose_examples(pairs, view, target, maximum=6):
    if pairs.empty:
        return pd.DataFrame()
    # Selection depends on input distance and IDs, never the direction/size of error.
    candidates = pairs.sort_values(['distance', 'target_id', 'source_id'], kind='stable').copy()
    sources = view.loc[candidates.source_id]
    candidates['airport'] = sources.airport.to_numpy()
    candidates['date'] = pd.to_datetime(sources.mvt_time, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
    chosen = candidates.drop_duplicates(['airport', 'date']).head(maximum)
    output = []
    target = target.set_index('mvt_id')
    for pair in chosen.itertuples():
        old, new = view.loc[pair.source_id], target.loc[pair.target_id]
        item = dict(cohort=pair.cohort, distance=pair.distance, source_id=pair.source_id,
            target_id=pair.target_id, airport=old.airport, source_flight=old.flight,
            source_time=old.mvt_time, target_time=new.mvt_time, retrospective=bool(old.retrospective),
            actual_taxi_sec=old.y, before_prediction_mean_sec=old.pred_control_mean,
            after_prediction_mean_sec=old.pred_disruption_mean,
            mse_change=float(np.mean([old[f'new_sq_{s}'] - old[f'old_sq_{s}'] for s in SEEDS])))
        for column in matching.MATCH_COLUMNS + ['raw_flight_id_missing', 'gap_sec']:
            item['source_' + column], item['target_' + column] = old[column], new[column]
        output.append(item)
    return pd.DataFrame(output)


def main():
    inputs, input_manifest = prepare.inputs()
    target = inputs.loc[inputs.population.eq('ranking')]
    selection = read(prepare.OUT / 'selection.json')
    views = {variant: load_view(variant, inputs) for variant in run.HISTORICAL}
    common = ['mvt_time', 'airport', 'y', 'month', 'raw_flight_id_missing', 'gap_band',
              *run.features.FEATURES, *matching.DIAGNOSTICS]
    pd.testing.assert_frame_equal(views['v39'][common], views['v40'][common])
    # Deliberately exclude all outcome/prediction columns from the matcher.
    observable = views['v39'].reset_index()[['mvt_id', 'mvt_time', 'airport', 'month',
        'raw_flight_id_missing', 'gap_band', 'fold', 'model_id', 'retrospective',
        *run.features.FEATURES, *matching.DIAGNOSTICS]]
    run.overnight_run.save_frame(prepare.CACHE / 'heldout_observables.parquet', observable)
    coverage, _ = prepare.save_matches(observable, target, 'heldout')
    write_json(prepare.OUT / 'heldout_selection.json', dict(coverage=coverage,
        selection_sha256=sha256(prepare.OUT / 'selection.json'), selection_uses_departure_outcomes=False,
        matcher_sha256=sha256(Path(matching.__file__)),
        heldout_observables_sha256=sha256(prepare.CACHE / 'heldout_observables.parquet')))
    results, flat, examples = {}, [], []
    for variant, view in views.items():
        results[variant] = {}
        for name in matching.COHORTS:
            pairs = pd.read_parquet(prepare.CACHE / f'heldout_{name}_pairs.parquet')
            selected = view.loc[view.index.isin(pairs.source_id)]
            results[variant][name] = paired_score(selected)
            for airport in ['all', *sorted(target.airport.unique())]:
                part = selected if airport == 'all' else selected.loc[selected.airport.eq(airport)]
                metrics = paired_score(part) if airport != 'all' else results[variant][name]
                flat.append(dict(variant=variant, cohort=name, airport=airport,
                                 **{k: v for k, v in metrics.items() if k not in ('intervals', 'by_seed')}))
            example = choose_examples(pairs, view, target)
            if len(example):
                example['variant'] = variant
                examples.append(example)
    pd.DataFrame(flat).to_csv(prepare.OUT / 'paired_errors.csv', index=False)
    example_frame = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame(columns=[
        'variant', 'cohort', 'source_id', 'target_id', 'airport', 'source_time', 'target_time',
        'actual_taxi_sec', 'before_prediction_mean_sec', 'after_prediction_mean_sec', 'mse_change'])
    example_frame.to_csv(prepare.OUT / 'examples.csv', index=False)
    example_lines = ['# Matched historical examples', '',
        'Examples are chosen by input similarity, including cases where predictions worsen. '
        'Snow and arrival pairs below read **historical / test**. Predictions average three seeds; all times are seconds unless marked.', '',
        '| Conditions; historical flight | 24h snow % | Recent arrival taxi mean (min) | Actual taxi-out | v39 before → after | v40 before → after |',
        '|---|---:|---:|---:|---:|---:|']
    if len(example_frame):
        for name in matching.COHORTS:
            for item in example_frame.loc[example_frame.variant.eq('v39') & example_frame.cohort.eq(name)].head(2).itertuples():
                paired = example_frame.loc[example_frame.variant.eq('v40') & example_frame.cohort.eq(name)
                    & example_frame.source_id.eq(item.source_id) & example_frame.target_id.eq(item.target_id)].iloc[0]
                label = name.replace('_', ' ') + '; ' + str(item.source_flight) + ' ' + item.airport + ' ' + str(item.source_time)[:16]
                snow = f'{100*item.source_dis_snow_24h_fraction:.1f} / {100*item.target_dis_snow_24h_fraction:.1f}'
                arrivals = f'{item.source_stress_arr_60m_uncapped_mean_sec/60:.1f} / {item.target_stress_arr_60m_uncapped_mean_sec/60:.1f}'
                example_lines.append(f'| {label} | {snow} | {arrivals} | {item.actual_taxi_sec:.0f} | '
                    f'{item.before_prediction_mean_sec:.1f} → {item.after_prediction_mean_sec:.1f} | '
                    f'{paired.before_prediction_mean_sec:.1f} → {paired.after_prediction_mean_sec:.1f} |')
    else:
        example_lines += ['', 'No comparable scored examples met the fixed limits.']
    example_lines += ['', 'These examples cover only matched conditions; they do not stand in for unmatched extreme test rows. '
        'The linked CSV contains the test timestamps, all matching inputs and up to six examples per group.', '',
        '[All example inputs and predictions](opdi_disruption_stress/examples.csv) · '
        '[Coverage and error summary](OPDI_DISRUPTION_STRESS_CHECK.md)']
    (ROOT / 'VALIDATION/reports/OPDI_DISRUPTION_STRESS_EXAMPLES.md').write_text('\n'.join(example_lines) + '\n')
    historical_eham = inputs.loc[inputs.population.eq('training') & inputs.airport.eq('EHAM')
                                & inputs.mvt_time.dt.year.eq(2025)]
    target_eham = target.loc[target.airport.eq('EHAM') & target.month.eq('2026-01')
                             & matching.cohort(target, 'prolonged_snow')]
    historical_max = float(historical_eham.dis_snow_24h_fraction.max())
    statuses = pd.read_parquet(prepare.CACHE / 'heldout_prolonged_snow_status.parquet').set_index('mvt_id')
    snow_pairs = pd.read_parquet(prepare.CACHE / 'heldout_prolonged_snow_pairs.parquet')
    eham_matches = views['v39'].loc[views['v39'].airport.eq('EHAM')
        & views['v39'].index.isin(snow_pairs.source_id)]
    eham = dict(target_prolonged_rows=len(target_eham), historical_24h_snow_fraction_max=historical_max,
        target_24h_snow_fraction_max=float(target_eham.dis_snow_24h_fraction.max()) if len(target_eham) else None,
        target_rows_above_historical_snow_max=int(target_eham.dis_snow_24h_fraction.gt(historical_max + 1e-7).sum()),
        target_rows_with_scored_analogue=int(statuses.loc[target_eham.mvt_id, 'reason'].eq('matched').sum()),
        matched_historical_rows=len(eham_matches),
        matched_historical_dates=sorted(eham_matches.mvt_time.dt.strftime('%Y-%m-%d').unique().tolist()))
    final = dict(design_sha256=sha256(prepare.DESIGN), source_sha256=sha256(Path(__file__)),
        selection_sha256=sha256(prepare.OUT / 'selection.json'),
        heldout_selection_sha256=sha256(prepare.OUT / 'heldout_selection.json'),
        supplemental_sha256=sha256(supplement.DEFAULT / 'result.json'),
        model_records={v: sha256(run.DEFAULT / v / 'result.json') for v in run.HISTORICAL},
        all_2025_coverage=selection['all_2025'], heldout_coverage=coverage, results=results, eham_january_support=eham,
        caveats=['Selection uses bounded observable-input similarity, not errors.',
                 'Matches do not establish equivalence on every model input.',
                 'All scores are on historical held-out rows, not unknown ranking labels.',
                 'January and any February-May additions are retrospective June-November-model checks.',
                 'June has no existing held-out model and is not scored.',
                 'Overlapping cohorts must not be added together.',
                 'Intervals are pointwise across multiple correlated models and cohorts, on reused historical data.',
                 'These selected subsets do not alter adopted Weighted CV.'])
    write_json(prepare.OUT / 'results.json', final)
    titles = {'prolonged_snow': 'Prolonged snow', 'heavy_arrival_congestion': 'Long recent arrival taxi times',
              'snow_and_congestion': 'Both'}
    lines = ['# Test-like snow and congestion', '',
        'Historical rows must meet every fixed similarity limit. Unmatched extreme test rows remain unassessed.', '',
        f"**Amsterdam January:** {eham['target_rows_above_historical_snow_max']:,} of "
        f"{eham['target_prolonged_rows']:,} prolonged-snow test rows exceed the entire 2025 snow-duration maximum. "
        f"{eham['target_rows_with_scored_analogue']:,} have a historical match with held-out predictions. "
        f"Those matches use {eham['matched_historical_rows']:,} historical flights from only "
        f"{len(eham['matched_historical_dates'])} day(s).", '',
        '| Conditions | Test flights | Match in 2025 | Match with predictions | Historical flights / days |',
        '|---|---:|---:|---:|---:|']
    for name, c in coverage.items():
        a = selection['all_2025'][name]
        lines.append(f"| {titles[name]} | {c['target_rows']:,} | {a['matched_target_rows']:,} | "
                     f"{c['matched_target_rows']:,} | {c['historical_rows']:,} / {c['historical_dates']} |")
    lines += ['', '| Model | Conditions | Before RMSE | After RMSE | Change | Evidence |', '|---|---|---:|---:|---:|---|']
    for variant in run.HISTORICAL:
        for name, result in results[variant].items():
            if not result['rows']:
                lines.append(f'| {variant} | {titles[name]} | — | — | — | No comparable scored rows |')
            else:
                labels = {'too_few_flights_or_dates': 'Too little data', 'inconclusive': 'Uncertain',
                    'supported_gain_on_matched_historical_rows': 'Supported on these rows',
                    'supported_regression_on_matched_historical_rows': 'Regression on these rows'}
                lines.append(f"| {variant} | {titles[name]} | {result['control']:.2f} | {result['candidate']:.2f} | "
                             f"{result['difference']:+.2f} | {labels[result['evidence']]} |")
    lines += ['', 'Subset scores are unweighted RMSE in seconds, averaging squared errors over three seeds. Example predictions average the three predictions. '
              'January and any extra February–May months are retrospective checks, using models that did not train on those departure rows. '
              'The selected conditions overlap and are not a new validation headline.', '',
              '“Match with predictions” counts test flights with a similar historical flight scored by a model that did not train on it. '
              'Test labels are unavailable. Matches require the same airport, missing-ID state and clock-gap band, '
              'plus similar snow and arrival histories. No match is forced.', '',
              '[Overall Weighted CV](OPDI_DISRUPTION_EXPERIMENT.md) · '
              '[Input-selected before/after examples](OPDI_DISRUPTION_STRESS_EXAMPLES.md) · '
              '[Airport results](opdi_disruption_stress/paired_errors.csv) · '
              '[Coverage, uncertainty and limits](opdi_disruption_stress/results.json) · '
              '[Fixed matching rules](../docs/OPDI_DISRUPTION_STRESS_CHECK.md)']
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(coverage=coverage, results=results), indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
