"""Read-only ranking sensitivity to six saved CV-window correction parameter sets.

This is not a full-year recipe, an estimate of hidden errors, or a submission.
The existing scored v34 file stays fixed. Only compact JSON/Markdown summaries
are written; no model is fitted and no prediction/submission parquet is written.

Run after the derivative validation completes:
  .venv/bin/python VALIDATION/tools/project_artifact_ranking.py --run <derived run>
Add --scenarios for explicitly conditional day-label arithmetic, not forecasts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import RANKING
import plan_window
import recording_artifacts as corrections
import record_fixes
from records import data_identity, read_run, sha256, write_json
import rome_signals

ROWS = 344841
BASE_SCORE = 280.4658
MANIFEST = ROOT / 'VALIDATION/runs/v3_20260919T104157403241Z/submission_v34.json'
CONTROL = ROOT / 'tmp/recording_artifacts_20260919/unique-rose_v34.parquet'
CANDIDATES = ('twins', 'clock12', 'clock2', 'day_shrunk', 'day_handler', 'rule_shrunk', 'combined_midgap')
MASK = dict(twins='twins', clock12='clock12', clock2='clock2', day_shrunk='day', day_handler='day',
            rule_shrunk='rule_band')
DEPARTURE_COLUMNS = ['MVT_ID_mvt', *corrections.RAW, 'ADES_mvt', 'FLIGHT_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt']
ARRIVAL_COLUMNS = ['ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
FORBIDDEN = {'TAXITIME_SEC_mvt', 'taxi_sec', 'y', 'label', 'BLOCK_TIME_UTC_mvt'}
SOURCES = ['VALIDATION/tools/project_artifact_ranking.py', 'VALIDATION/recording_artifacts.py',
           'VALIDATION/record_fixes.py', 'VALIDATION/plan_window.py', 'VALIDATION/rome_signals.py',
           'VALIDATION/records.py', 'pipeline/common.py']


def projected_ranking(path):
    """Departure target/block columns are absent from SQL and the returned frame."""
    con = duckdb.connect()
    try:
        con.execute('SET threads=2')
        con.execute("SET TimeZone='UTC'")
        departures = con.execute(
            f"SELECT {', '.join(DEPARTURE_COLUMNS)} FROM read_parquet(?) WHERE PHASE_mvt='DEP'",
            [str(path)]).df().rename(columns={'MVT_ID_mvt': 'mvt_id'})
        arrivals = con.execute(
            f"SELECT {', '.join(ARRIVAL_COLUMNS)}, TAXITIME_SEC_mvt AS taxi_sec "
            "FROM read_parquet(?) WHERE PHASE_mvt='ARR'", [str(path)]).df()
    finally:
        con.close()
    if FORBIDDEN & set(departures) or any('BLOCK_TIME' in name for name in departures):
        raise ValueError('Hidden departure fields entered the ranking projection')
    if departures.mvt_id.isna().any() or departures.mvt_id.duplicated().any():
        raise ValueError('Ranking departures must have unique movement IDs')
    return departures, arrivals


def align_control(raw, submitted, expected_rows=ROWS):
    if set(submitted) != {'MVT_ID_mvt', 'TAXITIME_SEC_mvt'}:
        raise ValueError('Expected the two-column scored v34 file')
    if (len(raw) != expected_rows or len(submitted) != expected_rows
            or submitted.MVT_ID_mvt.isna().any() or submitted.MVT_ID_mvt.duplicated().any()
            or raw.mvt_id.isna().any() or raw.mvt_id.duplicated().any()):
        raise ValueError('Ranking and scored control must have the complete unique movement roster')
    left = pd.Index(raw.mvt_id)
    right = pd.Index(submitted.MVT_ID_mvt)
    if len(left.difference(right)) or len(right.difference(left)):
        raise ValueError('Ranking movement IDs differ from the scored control')
    aligned = raw.set_index('mvt_id').reindex(right).reset_index().rename(columns={'MVT_ID_mvt': 'mvt_id'})
    control = submitted.TAXITIME_SEC_mvt.to_numpy(float)
    if not np.isfinite(control).all() or (control < 0).any():
        raise ValueError('Scored control predictions must be finite and nonnegative')
    return aligned, control


def add_handler_signals(raw, arrivals):
    """Each ranking month sees only its own supplied arrivals; no departure labels."""
    if FORBIDDEN & set(raw):
        raise ValueError('Departure labels cannot enter handler construction')
    departure_month = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    arrival_month = pd.to_datetime(arrivals.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    output = raw.copy()
    for name in corrections.SIGNALS:
        output[name] = np.nan
    for month in sorted(departure_month.unique()):
        selected = departure_month.eq(month)
        rates = rome_signals.handler_rates(raw.loc[selected], arrivals.loc[arrival_month.eq(month)])
        output.loc[selected, corrections.SIGNALS] = rates[corrections.SIGNALS].to_numpy()
    return output


def independent_predictions(control, raw, parameters, twin_mask):
    # Missing probabilities switch off only the p-dependent experiment, which is
    # excluded here. The seven selected recipes have no dependence on p.
    predictions, masks = corrections.predict(control, raw, parameters,
        probability=np.full(len(raw), np.nan), twin_mask=twin_mask)
    return {name: predictions[corrections.CANDIDATES[name]] for name in CANDIDATES}, masks


def conditional_score_change(control, candidate, assumed_y, selected, score=BASE_SCORE, total_rows=ROWS):
    """Change in pooled RMSE under stated labels for selected rows; never infer labels."""
    selected = np.asarray(selected, bool)
    control, candidate, assumed_y = (np.asarray(x, float)[selected] for x in (control, candidate, assumed_y))
    old_sse = float(np.sum((control - assumed_y) ** 2))
    historical_sse = score ** 2 * total_rows
    delta = float(np.sum((candidate - assumed_y) ** 2 - (control - assumed_y) ** 2))
    radicand = score ** 2 + delta / total_rows
    compatible = old_sse <= historical_sse and radicand >= 0
    return dict(assumed_rows=int(selected.sum()), delta_sse=delta,
                assumed_control_sse=old_sse, historical_total_sse=historical_sse,
                implied_unaffected_sse=historical_sse - old_sse,
                delta_rmse_sec=float(np.sqrt(radicand) - score) if compatible else None,
                compatible_with_total_sse=bool(compatible))


def plan_window_change_bounds(control, candidate, raw, selected, score=BASE_SCORE, total_rows=ROWS):
    """Plan-window endpoint bounds, reported only when compatible with known total SSE.

    If an extremizing endpoint label assignment requires negative unaffected SSE,
    that endpoint bound is unavailable. This helper does not optimize over the
    smaller feasible subset in that case.
    """
    selected = np.asarray(selected, bool)
    if not selected.any():
        return dict(assumed_rows=0, minimum_delta_rmse_sec=0., maximum_delta_rmse_sec=0.,
                    minimum_endpoint_compatible_with_total_sse=True, maximum_endpoint_compatible_with_total_sse=True)
    low, high = plan_window.window(raw)
    low = np.maximum(low, 0)
    old, new = np.asarray(control)[selected], np.asarray(candidate)[selected]
    if not np.isfinite(np.r_[low[selected], high[selected]]).all():
        raise ValueError('A conditional plan-window bound needs finite window endpoints')
    endpoints = np.stack([(new - y[selected]) ** 2 - (old - y[selected]) ** 2 for y in (low, high)])
    assumed_min, assumed_max = np.zeros(len(raw)), np.zeros(len(raw))
    choose_low = endpoints[0] <= endpoints[1]
    assumed_min[selected] = np.where(choose_low, low[selected], high[selected])
    assumed_max[selected] = np.where(choose_low, high[selected], low[selected])
    minimum = conditional_score_change(control, candidate, assumed_min, selected, score, total_rows)
    maximum = conditional_score_change(control, candidate, assumed_max, selected, score, total_rows)
    return dict(assumed_rows=int(selected.sum()), minimum_delta_rmse_sec=minimum['delta_rmse_sec'],
                maximum_delta_rmse_sec=maximum['delta_rmse_sec'],
                minimum_endpoint_compatible_with_total_sse=minimum['compatible_with_total_sse'],
                maximum_endpoint_compatible_with_total_sse=maximum['compatible_with_total_sse'])


def delta_summary(control, candidate, eligible):
    delta = np.asarray(candidate, float) - np.asarray(control, float)
    changed = delta != 0
    absolute = np.abs(delta[changed])
    return dict(eligible_rows=int(np.asarray(eligible).sum()), changed_rows=int(changed.sum()),
                raised=int((delta > 0).sum()), lowered=int((delta < 0).sum()),
                mean_abs_change_sec=float(absolute.mean()) if changed.any() else 0.,
                max_abs_change_sec=float(absolute.max()) if changed.any() else 0.,
                min_change_sec=float(delta[changed].min()) if changed.any() else 0.,
                max_change_sec=float(delta[changed].max()) if changed.any() else 0.)


def examples(raw, control, low_delta, high_delta, limit=5):
    maximum = np.maximum(np.abs(low_delta), np.abs(high_delta))
    changed = np.flatnonzero(maximum > 0)
    ordered = changed[np.lexsort((raw.mvt_id.to_numpy()[changed], -maximum[changed]))[:limit]]
    movement = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
    gap = lambda column: (movement - pd.to_datetime(raw[column], utc=True)).dt.total_seconds().to_numpy(float)
    gaps = {name: gap(column) for name, column in [('schedule', 'SCHED_TIME_UTC_mvt'), ('lobt', 'LOBT_flt'), ('aobt3', 'AOBT_3_flt')]}
    out = []
    for i in ordered:
        item = raw.iloc[i]
        text = lambda value: str(value) if pd.notna(value) else None
        out.append(dict(mvt_id=int(item.mvt_id), airport=text(item.ADEP_mvt), flight=text(item.FLIGHT_mvt),
            destination=text(item.ADES_mvt), stand=text(item.STAND_mvt), time_utc=movement.iloc[i].isoformat(),
            has_flight_id=bool(pd.notna(item.FLIGHT_ID_mvt)), control_sec=float(control[i]),
            candidate_sec_range=[float(control[i] + low_delta[i]), float(control[i] + high_delta[i])],
            change_sec_range=[float(low_delta[i]), float(high_delta[i])],
            clock_gap_sec={name: float(values[i]) if np.isfinite(values[i]) else None for name, values in gaps.items()}))
    return out


def sensitivity(raw, control, parameter_sets, scenarios=False, total_rows=ROWS):
    twin = np.zeros(len(raw), bool)
    months = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy()
        twin[mask] = record_fixes.twin_rows(raw.loc[mask])
    windows = {name: {} for name in CANDIDATES}
    lower = {name: np.full(len(raw), np.inf) for name in CANDIDATES}
    upper = {name: np.full(len(raw), -np.inf) for name in CANDIDATES}
    for model, parameters in sorted(parameter_sets.items()):
        predictions, masks = independent_predictions(control, raw, parameters, twin)
        gaps = corrections.clocks(corrections.observable(raw))
        for name, candidate in predictions.items():
            eligible = (masks['twins'] | masks['day'] | masks['rule_band'] | masks['clock12']
                        if name == 'combined_midgap' else masks[MASK[name]])
            item = delta_summary(control, candidate, eligible)
            if name == 'clock12':
                item['conditional_plan_window_bounds'] = plan_window_change_bounds(
                    control, candidate, raw, eligible, total_rows=total_rows)
            if scenarios and name in ('day_shrunk', 'day_handler'):
                item['conditional_scenarios'] = {
                    'labels_equal_schedule_gap': conditional_score_change(control, candidate, gaps['schedule'], eligible, total_rows=total_rows),
                    'labels_equal_one_day_plus_window_late_mean': conditional_score_change(
                        control, candidate, np.full(len(raw), corrections.DAY + parameters['late_mean']), eligible, total_rows=total_rows),
                }
            windows[name][model] = item
            delta = candidate - control
            lower[name] = np.minimum(lower[name], delta)
            upper[name] = np.maximum(upper[name], delta)
    candidates = {}
    for name in CANDIDATES:
        values = list(windows[name].values())
        ranges = {key: [min(item[key] for item in values), max(item[key] for item in values)]
                  for key in ('eligible_rows', 'changed_rows', 'mean_abs_change_sec', 'max_abs_change_sec')}
        candidates[name] = dict(window_ranges=ranges, by_window=windows[name],
            top_affected_examples=examples(raw, control, lower[name], upper[name]))
    return candidates


def validate_run(record):
    if (record.get('status') != 'complete' or record.get('protocol') != 'v3'
            or set(record.get('correction_parameters', {})) != {f'v3_2025-{month:02}' for month in range(6, 12)}):
        raise ValueError('A completed derivative run with all six CV-window coefficient sets is required')
    for name in ('VALIDATION/recording_artifacts.py', 'VALIDATION/record_fixes.py', 'VALIDATION/plan_window.py'):
        if record['signature'].get(name) != sha256(ROOT / name):
            raise ValueError(f'Correction source changed since validation: {name}')
    name = 'VALIDATION/rome_signals.py'
    if record['source_signature'].get(name) != sha256(ROOT / name):
        raise ValueError('Handler feature source differs from the validated parent')
    if any(item['settings'] != corrections.SETTINGS for item in record['correction_parameters'].values()):
        raise ValueError('Saved coefficients use different correction settings')


def hypotheses(record, candidates, total_rows):
    cv_count = record['changes']['candidates']['clock12']['unique_movements']
    cv_total = record['report']['groups']['R_A']['n']
    ranking_count = candidates['clock12']['window_ranges']['eligible_rows'][0]
    by_window = candidates['clock12']['by_window'].values()
    bound = [item['conditional_plan_window_bounds']['minimum_delta_rmse_sec'] for item in by_window]
    complete_bounds = all(value is not None for value in bound)
    return dict(clock12=dict(
        status='Hypothesis only: larger affected-population frequency may make its leaderboard gain larger than CV; tiny headroom and only one observed case per population do not establish transfer.',
        ranking_eligible_movements=ranking_count, ranking_departures=total_rows,
        cv_unique_changed_movements=cv_count, cv_unique_departures=cv_total,
        ranking_per_100k=100000 * ranking_count / total_rows, cv_per_100k=100000 * cv_count / cv_total,
        frequency_ratio=(ranking_count / total_rows) / (cv_count / cv_total) if cv_count else None,
        conditional_largest_improvement_sec=-min(bound) if bound and complete_bounds else None,
        bound_assumption='Hidden labels lie within the empirical LOBT +/-3660-second plan window and the assumed affected-row control SSE does not exceed the historical total. Ranking labels remain hidden.'),
        day=dict(status='Leaderboard gain remains uncertain. Ranking day-pattern coverage does not reveal whether its labels follow schedule substitution or a one-day shift.',
                 ranking_eligible_movements=candidates['day_shrunk']['window_ranges']['eligible_rows'][0]))


def markdown(payload):
    lines = ['# Ranking correction sensitivity', '',
        'These are six CV-window coefficient sets applied separately to the existing scored v34 predictions. '
        '**This is not a full-year recipe or a prediction of hidden errors.** No model or submission file was created.', '',
        'Departure labels and block timestamps are excluded by an explicit SQL projection. Handler signals use only '
        "each ranking month's own arrivals. Prediction changes are unrounded.", '',
        '| Candidate | Eligible rows | Changed rows across windows | Mean absolute change on changed rows (s) | Largest change (s) |',
        '|---|---:|---:|---:|---:|']
    fmt = lambda values, digits=1: f'{values[0]:.{digits}f}–{values[1]:.{digits}f}'
    for name, item in payload['candidates'].items():
        ranges = item['window_ranges']
        lines.append(f"| {name} | {fmt(ranges['eligible_rows'], 0)} | {fmt(ranges['changed_rows'], 0)} | "
                     f"{fmt(ranges['mean_abs_change_sec'])} | {fmt(ranges['max_abs_change_sec'])} |")
    clock = payload['hypotheses']['clock12']
    lines += ['', f"**Clock12 transfer hypothesis:** {clock['ranking_eligible_movements']} ranking movement(s) among "
        f"{clock['ranking_departures']:,}, versus {clock['cv_unique_changed_movements']} unique changed CV movement(s) "
        f"among {clock['cv_unique_departures']:,}. A higher affected frequency may produce a larger leaderboard gain than CV. "
        'The sample is too small to establish that.', '']
    if clock['conditional_largest_improvement_sec'] is not None:
        lines.append(f"Its headroom is small: at most {clock['conditional_largest_improvement_sec']:.4f} s improvement across these "
                     'coefficient sets **if** the hidden labels obey the empirical LOBT ±3660 s window and the endpoint assumptions '
                     'are compatible with v34’s total squared error. This is a conditional bound.')
    lines += ['', '**Day-pattern gains remain uncertain.** We cannot tell which ranking labels follow the schedule and which have a date shift.']
    if payload['scenario_assumptions'] is not None:
        lines += ['', '**Conditional arithmetic only — not forecasts.** Assume the affected day-row labels equal the named clock. '
            'Keep all other squared errors fixed at the historical v34 total (280.4658 s; 344,841 rows). '
            'Assumptions whose affected-row control error exceeds that total are incompatible and have no score change. '
            'The table shows candidate-minus-v34 RMSE ranges across six coefficient sets.', '',
            '| Day candidate | Labels = takeoff − schedule | Labels = 86400 + window late mean |', '|---|---:|---:|']
        for name in ('day_shrunk', 'day_handler'):
            values = []
            for scenario in ('labels_equal_schedule_gap', 'labels_equal_one_day_plus_window_late_mean'):
                changes = [item['conditional_scenarios'][scenario]['delta_rmse_sec'] for item in payload['candidates'][name]['by_window'].values()]
                finite = [value for value in changes if value is not None]
                values.append(fmt([min(finite), max(finite)], 3) if finite else 'incompatible with total SSE')
            lines.append(f"| {name} | {values[0]} s | {values[1]} s |")
    lines += ['', '| Candidate | Example flight | Airport / stand | v34 (s) | Candidate range (s) |', '|---|---|---|---:|---:|']
    for name, item in payload['candidates'].items():
        for example in item['top_affected_examples'][:2]:
            lines.append(f"| {name} | {example['flight']} ({example['mvt_id']}) | {example['airport']} / {example['stand']} | "
                         f"{example['control_sec']:.0f} | {fmt(example['candidate_sec_range'])} |")
    lines += ['', 'The JSON companion retains every window summary, up to five examples per candidate, input hashes and coefficient fingerprints.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--ranking', type=Path, default=RANKING)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'VALIDATION/reports/recording_artifacts_20260919')
    parser.add_argument('--scenarios', action='store_true')
    args = parser.parse_args(argv)
    record, directory = read_run(args.run)
    validate_run(record)
    manifest = json.loads(MANIFEST.read_text())
    if (manifest['state'] != 'scored' or manifest['version'] != 34 or manifest['score'] != BASE_SCORE
            or manifest['rows'] != ROWS or sha256(args.control) != manifest['sha256']):
        raise ValueError('Control file must reproduce the tracked, scored v34 file byte for byte')
    ranking_identity = next(item for item in record['data_identity'] if item['file'] == RANKING.name)
    if sha256(args.ranking) != ranking_identity['sha256']:
        raise ValueError('Ranking data differs from the validated run')
    raw, arrivals = projected_ranking(args.ranking)
    raw, control = align_control(raw, pd.read_parquet(args.control))
    months = sorted(pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').unique())
    if months != ['2026-01', '2026-07']:
        raise ValueError('Expected the two supplied January/July 2026 ranking months')
    raw = add_handler_signals(raw, arrivals)
    candidates = sensitivity(raw, control, record['correction_parameters'], scenarios=args.scenarios)
    payload = dict(generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), run_id=record['run_id'],
        interpretation='Sensitivity to six CV-window coefficient sets; no full-year recipe, hidden-error estimate, fit, prediction parquet or submission.',
        rows=ROWS, historical_v34_rmse=BASE_SCORE, candidates=candidates,
        scenario_assumptions=('Only affected day-row labels are replaced with the specified hypothetical clock; all other SSE stays fixed. '
            'Not observed labels or score forecasts.') if args.scenarios else None,
        hypotheses=hypotheses(record, candidates, ROWS),
        projection=dict(departure_columns=['mvt_id' if name == 'MVT_ID_mvt' else name for name in DEPARTURE_COLUMNS],
                        forbidden_departure_fields_absent=True, arrival_scope='each own ranking UTC month',
                        arrival_taxi_labels_used=True, classifier_probabilities_used=False),
        source_signature={name: sha256(ROOT / name) for name in SOURCES},
        input_identity=data_identity([args.ranking, args.control, MANIFEST, directory / 'result.json']),
        coefficient_fingerprints={name: hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
                                  for name, parameters in record['correction_parameters'].items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'ranking_sensitivity.json', payload)
    (args.output_dir / 'ranking_sensitivity.md').write_text(markdown(payload))
    print(f'Wrote compact ranking_sensitivity.json/.md in {args.output_dir}; no fit or submission artifact.', flush=True)


if __name__ == '__main__':
    main()
