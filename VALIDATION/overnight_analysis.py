"""Prespecified CV comparison, calibration freeze and held-out submission checks.

No training or uploads. ``prepare`` reads prediction artifacts sequentially and
retains daily sufficient statistics, rather than keeping hundreds of files in
memory. ``select`` may see only the three discovery leaderboard scores;
``confirm`` evaluates the two previously unseen scores against frozen forecasts.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import A_FOLDS, B_FOLDS
from records import clean, sha256, write_json
from reporting import IDENTITY, COVERAGE, Loss, bootstrap_calendar, interval

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'VALIDATION/runs/overnight_20260917'
MANIFEST = ROOT / 'VALIDATION/reports/leaderboard_selected_20260916.json'
SEASONAL = ROOT / 'VALIDATION/runs/seasonal_jan_jul/result.json'
CONTROL = 'control'
ANCHOR = 'Q0-4h'
DISCOVERY = ('no_month', 'no_history', 'clock_features')
CONFIRMATION = ('small_trees', 'rule_routing')
RECIPES = (CONTROL,) + DISCOVERY + CONFIRMATION
OLD = ('B0-800', 'Q0', 'Q1', ANCHOR, 'Q1-4h')
SEEDS = (42, 142, 242)
A = tuple(f.name for f in A_FOLDS)
LONG = ('B3', 'B4', 'B7')
ALL = A + tuple(f.name for f in B_FOLDS)
FOLD_MONTHS = {f.name: f.val for f in A_FOLDS + B_FOLDS}
MONTHS = ('2025-01', '2025-07')
METRICS = ('R_combined', 'R_A', 'R_B_long', 'month_balanced',
           'unique_flight', 'seasonal_10m', 'seasonal_11m',
           'blend_25', 'blend_50', 'blend_75')
POLICIES = METRICS + tuple(metric+'__mse' for metric in METRICS)
TOLERANCE = 1.0
VERSION = 'overnight-calibration-v2-fixed-mse-transport'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest_json(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), ROOT / 'VALIDATION/reporting.py', ROOT / 'VALIDATION/protocol.py')}


def read_json(path):
    return json.loads(Path(path).read_text())


def add_blends(scores):
    for weight in (.25, .5, .75):
        scores[f'blend_{int(weight * 100)}'] = float(np.sqrt(
            weight * scores['month_balanced']**2 + (1-weight) * scores['seasonal_11m']**2))
    return scores


def aggregate(folds, seasonal, jan_fraction, mode='standard'):
    """Score aligned populations; repeated origins average losses, never votes.

    ``folds`` maps fold name to month/n/mse/mse_by_seed. Seasonal maps strategy
    and month to the same summaries. Works equally for saved historical totals
    and newly calculated prediction summaries.
    """
    if set(folds) != set(ALL):
        raise ValueError('Expected all sixteen v3 folds')

    def mse(row):
        return row['mse'] if mode == 'standard' else row['mse_by_seed'][str(mode)]

    def pool(names, equal_months=False):
        groups = defaultdict(list)
        for name in names:
            groups[folds[name]['month']].append(folds[name])
        counts, values = [], []
        for rows in groups.values():
            if len({r['n'] for r in rows}) != 1:
                raise ValueError('Same-month origins have different flight populations')
            counts.append(rows[0]['n'])
            values.append(np.mean([mse(row) for row in rows]))
        return float(np.sqrt(np.average(values, weights=None if equal_months else counts)))

    result = {'R_A': pool(A), 'R_B_long': pool(LONG),
              'month_balanced': pool(ALL, True), 'unique_flight': pool(ALL)}
    result['R_combined'] = float(np.sqrt(jan_fraction * result['R_A']**2
                                        + (1-jan_fraction) * result['R_B_long']**2))
    for strategy in ('10m', '11m'):
        result[f'seasonal_{strategy}'] = float(np.sqrt(
            jan_fraction * mse(seasonal[strategy][MONTHS[0]])
            + (1-jan_fraction) * mse(seasonal[strategy][MONTHS[1]])))
    return add_blends(result)


def historical():
    batch, season = read_json(MANIFEST), read_json(SEASONAL)
    items = {row['codename']: row for row in batch['submissions']}
    hashes = {str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
              str(SEASONAL.relative_to(ROOT)): sha256(SEASONAL)}
    scores, official, weight = {}, {}, None
    for code in OLD:
        item = items[code]
        if item['official_result']['status'] != 'Succeeded':
            raise ValueError('Historical submission lacks a successful official score')
        path = ROOT / 'VALIDATION/runs' / item['validation_run'] / 'result.json'
        record = read_json(path)
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        w = record['ranking_weights']['jan_fraction']
        if weight is not None and weight != w:
            raise ValueError('Historical ranking weights differ')
        weight = w
        if tuple(record['config']['seeds']) != SEEDS:
            raise ValueError('Historical seeds differ')
        folds = {name: dict(month=record['folds'][name]['val'], n=row['n'],
                           mse=row['sse']/row['n'],
                           mse_by_seed={s: v*v for s, v in row['rmse_by_seed'].items()})
                 for name, row in record['report']['folds'].items() if name in ALL}
        seasonal = {}
        for strategy in ('10m', '11m'):
            slices = season['report']['strategies'][strategy]['slices'][code]
            seasonal[strategy] = {m: dict(n=r['n'], mse=r['sse']/r['n'],
                mse_by_seed={s: v*v for s, v in r['rmse_by_seed'].items()}) for m, r in slices.items()}
        scores[code] = {mode: aggregate(folds, seasonal, weight, mode)
                        for mode in ('standard', *map(str, SEEDS))}
        for key in ('R_A', 'R_B_long', 'R_combined'):
            np.testing.assert_allclose(scores[code]['standard'][key],
                                       record['report']['scores'][key], atol=1e-9, rtol=0)
        for strategy in ('10m', '11m'):
            expected = season['report']['strategies'][strategy]['modes']['standard']['levels'][code]['scores']['combined']
            np.testing.assert_allclose(scores[code]['standard'][f'seasonal_{strategy}'],
                                       expected, atol=1e-9, rtol=0)
        official[code] = float(item['score'])
    if weight != season['signature']['ranking_weights']['jan_fraction']:
        raise ValueError('Seasonal and historical ranking weights differ')
    return dict(scores=scores, official=official, jan_fraction=weight, hashes=hashes)


def frame_digest(frame, columns):
    # Object strings give a stable identity hash across parquet category encodings.
    meta = frame[columns].copy()
    for column in meta.select_dtypes(include=['category']).columns:
        meta[column] = meta[column].astype(object)
    return hashlib.sha256(pd.util.hash_pandas_object(meta, index=True).values.tobytes()).hexdigest()


def prediction_entries(progress):
    entries = progress['predictions']
    return list(entries.values()) if isinstance(entries, dict) else entries


def checked_path(run_root, entry):
    path = run_root / entry['path']
    if not path.resolve().is_relative_to(run_root.resolve()):
        raise ValueError('Prediction artifact escapes run directory')
    expected = entry.get('sha256')
    artifacts = entry.get('artifacts', {})
    if expected is None and isinstance(artifacts, dict):
        expected = artifacts.get(entry['path']) or artifacts.get('sha256')
        if isinstance(expected, dict):
            expected = expected.get('sha256')
    if not expected:
        raise ValueError(f'Prediction entry has no SHA256: {entry["path"]}')
    actual = sha256(path)
    if expected != actual:
        raise ValueError(f'Prediction artifact hash mismatch: {path}')
    return path, actual


def summarize_entries(run_root, entries, populations, hashes, seeds=SEEDS):
    by_seed = {int(e['seed']): e for e in entries}
    if len(entries) != len(seeds) or set(by_seed) != set(seeds):
        raise ValueError(f'Each fold must contain exactly seeds {seeds}')
    reference, daily, summary = None, [], None
    for seed in seeds:
        entry = by_seed[seed]
        path, digest = checked_path(run_root, entry)
        hashes[str(path.relative_to(run_root))] = digest
        frame = pd.read_parquet(path)
        if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
            raise ValueError('Duplicate or missing prediction movement ID')
        frame = frame.set_index('mvt_id').sort_index()
        frame['mvt_time'] = pd.to_datetime(frame.mvt_time, utc=True)
        month = entry['month']
        if (len(frame) != entry['rows'] or not frame.seed.eq(seed).all()
                or not frame.fold.eq(entry['fold']).all()
                or not frame.month.eq(month).all()
                or not frame.mvt_time.dt.strftime('%Y-%m').eq(month).all()):
            raise ValueError('Prediction fold, month, seed or row count differs from manifest')
        if not np.isfinite(frame[['y', 'pred']].to_numpy()).all():
            raise ValueError('Nonfinite prediction or target')
        identity = frame_digest(frame, IDENTITY)
        if month in populations and identity != populations[month]:
            raise ValueError(f'Predictions have different movement IDs/targets for {month}')
        populations[month] = identity
        within_seed = frame_digest(frame, IDENTITY + COVERAGE)
        if reference is not None and reference != within_seed:
            raise ValueError('Feature coverage or flight identity differs between seeds')
        reference = within_seed
        sq = (frame.y.to_numpy() - frame.pred.to_numpy())**2
        index = frame.mvt_time.dt.day.to_numpy() - 1
        n_days = pd.Period(month, 'M').days_in_month
        counts = np.bincount(index, minlength=n_days)
        daily.append(np.bincount(index, weights=sq, minlength=n_days))
        if summary is None:
            summary = dict(month=month, n=len(frame), count_by_day=counts.tolist(),
                           identity_sha256=identity, mse_by_seed={})
        summary['mse_by_seed'][str(seed)] = float(sq.mean())
    summary['mse'] = float(np.mean(list(summary['mse_by_seed'].values())))
    summary['sse_by_seed_day'] = np.stack(daily).tolist()
    return summary


def daily_group(folds, names):
    by_month = defaultdict(list)
    for name in names:
        by_month[folds[name]['month']].append(folds[name])
    result = {}
    for month, rows in by_month.items():
        count = np.asarray(rows[0]['count_by_day'])
        if any(not np.array_equal(count, row['count_by_day']) for row in rows[1:]):
            raise ValueError('Daily populations differ between origins')
        # Both training-origin and seed variability are conditioned on, not resampled.
        sse = np.mean([np.mean(row['sse_by_seed_day'], axis=0) for row in rows], axis=0)
        result[month] = (count, sse)
    return result


def bootstrap_scores(folds, seasonal, weight, calendar, draws):
    def score(parts, equal_months=False):
        numerators, denominators = [], []
        for month, (count, sse) in sorted(parts.items()):
            mask = calendar.strftime('%Y-%m') == month
            weights = draws[:, mask]
            numerators.append(weights @ sse)
            denominators.append(weights @ count)
        with np.errstate(divide='ignore', invalid='ignore'):
            mse = (np.mean(np.asarray(numerators)/denominators, axis=0) if equal_months
                   else np.sum(numerators, axis=0)/np.sum(denominators, axis=0))
            return np.sqrt(mse)

    result = {'R_A': score(daily_group(folds, A)), 'R_B_long': score(daily_group(folds, LONG)),
              'month_balanced': score(daily_group(folds, ALL), True),
              'unique_flight': score(daily_group(folds, ALL))}
    result['R_combined'] = np.sqrt(weight*result['R_A']**2 + (1-weight)*result['R_B_long']**2)
    for strategy in ('10m', '11m'):
        monthly = [score(daily_group({m: seasonal[strategy][m]}, [m])) for m in MONTHS]
        result[f'seasonal_{strategy}'] = np.sqrt(weight*monthly[0]**2 + (1-weight)*monthly[1]**2)
    for w in (.25, .5, .75):
        result[f'blend_{int(100*w)}'] = np.sqrt(w*result['month_balanced']**2
                                               + (1-w)*result['seasonal_11m']**2)
    return result


def prepare(run_root, bootstrap_reps=2000):
    run_root = Path(run_root)
    if (run_root / 'cv_selection_frozen.json').exists():
        raise ValueError('Selection is already frozen; do not replace its CV inputs')
    progress_path = run_root / 'progress.json'
    progress = read_json(progress_path)
    record = read_json(run_root / 'result.json')
    config = record['config']
    execution_only = bool(config.get('smoke') or config.get('sample') != 1
                          or config.get('rounds') != 800 or tuple(config.get('seeds', [])) != SEEDS)
    seeds = tuple(config['seeds'])
    if not seeds or record['status'] not in ('complete', 'execution_check_only'):
        raise ValueError('Runner must finish validation before preparing scores')
    entries = prediction_entries(progress)
    if set(e['recipe'] for e in entries) != set(RECIPES):
        raise ValueError('Expected control and exactly five candidate recipes')
    reference = historical()
    grouped = defaultdict(list)
    for entry in entries:
        if entry['scheme'] not in ('v3', '10m', '11m'):
            raise ValueError('Unknown validation scheme')
        key = (entry['recipe'], entry['scheme'], entry['fold'], entry['month'])
        grouped[key].append(entry)
    populations, hashes, folds_all, seasonal_all, scores = {}, {}, {}, {}, {}
    for recipe in RECIPES:
        folds, seasonal = {}, {'10m': {}, '11m': {}}
        seasonal_jobs = defaultdict(dict)
        for (name, scheme, fold, month), group in sorted(grouped.items()):
            if name != recipe:
                continue
            summary = summarize_entries(run_root, group, populations, hashes, seeds)
            if scheme == 'v3':
                if fold not in ALL or FOLD_MONTHS[fold] != month or fold in folds:
                    raise ValueError('Invalid or duplicated v3 fold')
                folds[fold] = summary
            else:
                if month not in MONTHS or month in seasonal[scheme]:
                    raise ValueError('Invalid or duplicated seasonal held-out month')
                seasonal[scheme][month] = summary
                jobs = {e['job'] for e in group}
                if len(jobs) != 1:
                    raise ValueError('Seeds for a seasonal holdout use different training jobs')
                seasonal_jobs[scheme][month] = next(iter(jobs))
        if any(set(seasonal[s]) != set(MONTHS) for s in seasonal):
            raise ValueError('Incomplete seasonal predictions')
        if (len(set(seasonal_jobs['10m'].values())) != 1
                or len(set(seasonal_jobs['11m'].values())) != 2):
            raise ValueError('10m must reuse one model; 11m must use two distinct monthly training jobs')
        scores[recipe] = {mode: aggregate(folds, seasonal, reference['jan_fraction'], mode)
                          for mode in ('standard', *map(str, seeds))}
        folds_all[recipe], seasonal_all[recipe] = folds, seasonal
        print(f'Prepared {recipe}: v3 {scores[recipe]["standard"]["R_combined"]:.4f}', flush=True)
    # Reuse the protocol's calendar draws and apply exact daily sufficient statistics.
    months = sorted(populations)
    dummy = Loss(pd.DataFrame({'mvt_time': pd.to_datetime([m+'-01' for m in months], utc=True)}),
                 np.zeros((1, len(months))), (42,))
    calendar, draws = bootstrap_calendar({'calendar': dummy}, bootstrap_reps, 0)
    boots = {recipe: bootstrap_scores(folds_all[recipe], seasonal_all[recipe],
                reference['jan_fraction'], calendar, draws) for recipe in RECIPES}
    uncertainty = {recipe: {metric: interval(
        scores[recipe]['standard'][metric]-scores[CONTROL]['standard'][metric],
        boots[recipe][metric]-boots[CONTROL][metric], eligible=not execution_only) for metric in METRICS}
        for recipe in DISCOVERY + CONFIRMATION}
    monthly_v3 = {}
    for recipe, folds in folds_all.items():
        monthly_v3[recipe] = {}
        for month in sorted(set(FOLD_MONTHS.values())):
            names = [name for name in ALL if FOLD_MONTHS[name] == month]
            monthly_v3[recipe][month] = dict(folds=names, n=folds[names[0]]['n'],
                rmse=float(np.sqrt(np.mean([folds[n]['mse'] for n in names]))),
                rmse_by_seed={str(seed): float(np.sqrt(np.mean([
                    folds[n]['mse_by_seed'][str(seed)] for n in names]))) for seed in seeds})
    payload = dict(version=VERSION, prepared_at=utcnow(), source_hashes=source_hashes(),
        execution_only=execution_only, runner_config=config,
        progress_sha256=sha256(progress_path), prediction_sha256=hashes,
        historical_hashes=reference['hashes'], jan_fraction=reference['jan_fraction'],
        old=reference['scores'], old_official=reference['official'], new=scores,
        folds=folds_all, monthly_v3=monthly_v3, seasonal=seasonal_all, paired_bootstrap=uncertainty,
        bootstrap=dict(repetitions=bootstrap_reps, seed=0, block_days=1,
            scope='Conditional paired calendar-day bootstrap within month; not LB uncertainty or an independent year.'),
        candidate_metrics=list(METRICS), discovery=list(DISCOVERY), confirmation=list(CONFIRMATION))
    write_json(run_root / 'cv_scores.json', payload)
    lines = ['# Overnight CV scores before new leaderboard outcomes', '',
        '**EXECUTION CHECK ONLY; cannot select a validation metric.**' if execution_only else
        'Full-data 800-round validation; three seeds.', '',
        'All sixteen chronological folds are retained. Seeds and repeated training origins average '
        'squared errors, not predictions. Seasonal holdouts are retrospective. '
        'The fixed blends combine month-balanced and eleven-month seasonal MSE; blend number '
        'is the percentage assigned to month-balanced MSE.', '',
        '| Recipe | v3 combined | A | B long | Month balanced | Unique flight | 10m | 11m |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for recipe in RECIPES:
        values = scores[recipe]['standard']
        lines.append('| '+recipe+' | '+' | '.join(f'{values[k]:.4f}' for k in METRICS[:7])+' |')
    lines += ['', 'The JSON retains every fold, monthly and per-seed score, artifact hash, '
        'and 2,000-draw paired interval (or the explicitly requested repetition count). '
        'No new official score is read by this command.', '']
    (run_root / 'cv_scores.md').write_text('\n'.join(lines))
    return payload


def validate_score_file(path, expected):
    scores = read_json(path)
    if set(scores) != set(expected):
        raise ValueError(f'Expected only these official scores: {", ".join(expected)}')
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not np.isfinite(v) or v <= 0 for v in scores.values()):
        raise ValueError('Official scores must be finite positive numbers')
    return scores


def alignment(predicted, official, anchor=ANCHOR):
    """All-pair decisions are descriptive and explicitly not independent trials."""
    if set(predicted) != set(official):
        raise ValueError('CV and official candidate sets differ')
    names = sorted(official)
    errors = {n: predicted[n]-official[n] for n in names if n != anchor}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        cv, lb = predicted[a]-predicted[b], official[a]-official[b]
        if abs(lb) <= TOLERANCE:
            decision = 'both_negligible' if abs(cv) <= TOLERANCE else 'false_separation'
        elif abs(cv) <= TOLERANCE:
            decision = 'missed_separation'
        else:
            decision = 'agree' if cv*lb > 0 else 'misordering'
        pairs.append(dict(candidate=a, reference=b, delta_forecast=cv, delta_lb=lb, decision=decision))
    winner = min(names, key=lambda n: (predicted[n], n))
    best = min(official.values())
    tied = [n for n in names if predicted[n] <= predicted[winner]+TOLERANCE]
    return dict(delta_mae=float(np.mean(np.abs(list(errors.values())))), errors=errors,
        prediction_scale='anchored leaderboard RMSE seconds; pair deltas are forecast changes, not raw CV changes',
        pairs=pairs, pair_counts={decision: sum(p['decision'] == decision for p in pairs)
            for decision in ('agree', 'misordering', 'missed_separation', 'false_separation', 'both_negligible')},
        cv_winner=winner, observed_selection_regret=official[winner]-best,
        cv_near_tied_winners=tied, worst_near_tie_selection_regret=max(official[n]-best for n in tied))


def policy_parts(policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown fixed forecast policy: {policy}')
    return policy.removesuffix('__mse'), 'mse_delta' if policy.endswith('__mse') else 'rmse_delta'


def anchored_forecast(raw, control, anchor_lb, calibration):
    if calibration == 'rmse_delta':
        return float(anchor_lb+raw-control)
    if calibration != 'mse_delta':
        raise ValueError(f'Unknown forecast calibration: {calibration}')
    radicand = float(anchor_lb**2 + raw**2 - control**2)
    if radicand < 0:
        # Silently clipping can create artificial ties and conceal extrapolation.
        raise ValueError(f'Negative MSE-delta forecast radicand ({radicand}); '
                         'the proposed error-floor transport is invalid here; no silent clipping')
    return float(np.sqrt(radicand))


def metric_forecasts(cv, policy, mode='standard'):
    metric, calibration = policy_parts(policy)
    anchor_lb = cv['old_official'][ANCHOR]
    old_anchor = cv['old'][ANCHOR][mode][metric]
    new_anchor = cv['new'][CONTROL][mode][metric]
    raw = {code: cv['old'][code][mode][metric] for code in OLD}
    raw.update({name: cv['new'][name][mode][metric] for name in DISCOVERY+CONFIRMATION})
    anchored = {code: anchored_forecast(raw[code], old_anchor, anchor_lb, calibration) for code in OLD}
    anchored.update({name: anchored_forecast(raw[name], new_anchor, anchor_lb, calibration)
                     for name in DISCOVERY+CONFIRMATION})
    return raw, anchored


def selection_payload(cv, discovery, inputs):
    if set(discovery) != set(DISCOVERY):
        raise ValueError('Confirmation outcomes must not enter calibration')
    if cv.get('execution_only'):
        raise ValueError('Smoke or incomplete-protocol CV scores cannot select a metric')
    official = {**cv['old_official'], **discovery}
    metrics, forecasts = {}, {}
    for metric in POLICIES:
        readout, calibration_kind = policy_parts(metric)
        raw, anchored = metric_forecasts(cv, metric)
        calibration = {name: anchored[name] for name in official}
        row = alignment(calibration, official)
        row['raw_level_mae'] = float(np.mean([abs(raw[n]-official[n]) for n in official]))
        offset = float(np.median([official[n]-raw[n] for n in official]))
        row['median_additive_offset'] = offset
        row['offset_calibration_level_mae'] = float(np.mean([abs(raw[n]+offset-official[n]) for n in official]))
        row['calibration_predictions'] = calibration
        row['raw_cv'] = {name: raw[name] for name in official}
        row['readout'], row['calibration'] = readout, calibration_kind
        row['raw_cv_delta_vs_control'] = {name: raw[name] - (
            cv['old'][ANCHOR]['standard'][readout] if name in OLD else
            cv['new'][CONTROL]['standard'][readout]) for name in official}
        row['forecast_delta_vs_control'] = {name: anchored[name]-official[ANCHOR] for name in official}
        seed_raw, seed_anchored = metric_forecasts(cv, metric, '42')
        seed_offset = float(np.median([official[n]-seed_raw[n] for n in official]))
        row['seed42_diagnostic'] = alignment({n: seed_anchored[n] for n in official}, official)
        row['seed42_diagnostic'].update(
            raw_level_mae=float(np.mean([abs(seed_raw[n]-official[n]) for n in official])),
            median_additive_offset=seed_offset,
            calibration_predictions={n: seed_anchored[n] for n in official})
        metrics[metric] = row
        forecasts[metric] = {name: dict(raw_cv=raw[name], anchored_lb=anchored[name],
            additive_lb=raw[name]+offset,
            raw_cv_delta_vs_control=raw[name]-cv['new'][CONTROL]['standard'][readout],
            cv_delta_vs_control=raw[name]-cv['new'][CONTROL]['standard'][readout],
            forecast_delta_vs_control=anchored[name]-official[ANCHOR],
            readout=readout, calibration=calibration_kind,
            seed42=dict(raw_cv=seed_raw[name], anchored_lb=seed_anchored[name],
                additive_lb=seed_raw[name]+seed_offset,
                raw_cv_delta_vs_control=seed_raw[name]-cv['new'][CONTROL]['42'][readout],
                cv_delta_vs_control=seed_raw[name]-cv['new'][CONTROL]['42'][readout],
                forecast_delta_vs_control=seed_anchored[name]-official[ANCHOR]))
            for name in CONFIRMATION}
    def choose(rows):
        best = min(rows, key=lambda m: (rows[m]['delta_mae'], POLICIES.index(m)))
        # A negligible calibration win is not enough to displace the incumbent.
        return 'R_combined' if rows['R_combined']['delta_mae'] <= rows[best]['delta_mae']+TOLERANCE else best
    selected = choose(metrics)
    jackknife = {}
    for removed in sorted(set(official)-{ANCHOR}):
        subset = {n: s for n, s in official.items() if n != removed}
        candidates = {m: alignment({n: metrics[m]['calibration_predictions'][n] for n in subset}, subset)
                      for m in POLICIES}
        jackknife[removed] = choose(candidates)
    # Correlated rule/recovery variants count as one family in this diagnostic.
    # This sensitivity never changes the preregistered primary selection above.
    families = {'historical_rule_recovery': [name for name in OLD if name != ANCHOR],
                **{name: [name] for name in DISCOVERY}}
    family_sensitivity = {}
    for family, removed in families.items():
        training = {n: score for n, score in official.items() if n not in removed}
        candidates = {m: alignment({n: metrics[m]['calibration_predictions'][n] for n in training}, training)
                      for m in POLICIES}
        winner = choose(candidates)
        heldout = {ANCHOR: official[ANCHOR], **{n: official[n] for n in removed}}
        evaluated = {m: alignment({n: metrics[m]['calibration_predictions'][n] for n in heldout}, heldout)
                     for m in POLICIES}
        family_sensitivity[family] = dict(
            training_candidates=list(training), heldout_candidates=removed,
            winning_metric=winner,
            training_delta_mae={m: candidates[m]['delta_mae'] for m in POLICIES},
            heldout_family_forecasts={n: dict(official_lb=official[n],
                forecast_lb=metrics[winner]['calibration_predictions'][n],
                error=metrics[winner]['calibration_predictions'][n]-official[n]) for n in removed},
            winning_metric_heldout=evaluated[winner],
            baseline_heldout=evaluated['R_combined'],
            primary_selected_metric_heldout=evaluated[selected],
            all_metric_heldout_delta_mae={m: evaluated[m]['delta_mae'] for m in POLICIES})
    stamp = utcnow()
    return dict(version=VERSION, status='frozen', frozen_at=stamp, created_at=stamp, input_hashes=inputs,
        discovery_scores_sha256=inputs['discovery_scores_sha256'], scorecard_sha256=inputs['cv_scores_sha256'],
        selected_validation=selected, expected_confirmation=list(CONFIRMATION),
        source_hashes=source_hashes(), selection_rule='Minimum mean absolute anchor-relative delta error on 7 non-anchor calibration observations; retain R_combined if its MAE is within 1 second of the best. Other exact ties use fixed metric order.',
        anchor=ANCHOR, anchor_lb=official[ANCHOR], tolerance_sec=TOLERANCE,
        calibration_official=official, discovery=list(DISCOVERY), confirmation=list(CONFIRMATION),
        candidate_metrics=list(POLICIES), candidate_readouts=list(METRICS),
        selected_metric=selected, selected_readout=policy_parts(selected)[0],
        selected_calibration=policy_parts(selected)[1], baseline_metric='R_combined',
        forecast_policy=dict(rmse_delta='anchor_LB + candidate_CV - control_CV',
            mse_delta='sqrt(anchor_LB^2 + candidate_CV^2 - control_CV^2)',
            negative_radicand='Reject invalid transport; never silently clip',
            justification='Fixed MSE-delta transport tests an additive excess-error floor; no slope is fitted. '
                'Policies were added before any new submission scores were revealed. '
                'Local CV intervals remain in the original readout scale.'),
        metrics=metrics, confirmation_forecasts=forecasts,
        selected_offset=metrics[selected]['median_additive_offset'],
        leave_one_calibration_candidate_out_winner=jackknife,
        leave_one_family_out=dict(diagnostic_only=True, anchor_always_retained=ANCHOR,
            families=family_sensitivity,
            interpretation='The four historical non-anchor rule/recovery recipes form one correlated family; each discovery candidate forms one further family. Selection uses the remaining families only, with the same incumbent tie rule. This retrospective sensitivity is not independent confirmation and does not alter the primary selection.'),
        interpretation='Calibration selects a hypothesis. Two subsequent outcomes assess it without retuning; all scores share the same hidden LB population. Pair counts are dependent descriptions, not independent evidence.')


def select(run_root, discovery_path):
    run_root = Path(run_root)
    cv_path, target = run_root/'cv_scores.json', run_root/'cv_selection_frozen.json'
    discovery = validate_score_file(discovery_path, DISCOVERY)
    inputs = {'cv_scores_sha256': sha256(cv_path), 'discovery_scores_sha256': sha256(discovery_path)}
    if target.exists():
        frozen = read_json(target)
        if frozen['input_hashes'] != inputs or frozen['source_hashes'] != source_hashes():
            raise ValueError('Existing immutable selection has different inputs or implementation')
        return frozen
    cv = read_json(cv_path)
    if cv['source_hashes'] != source_hashes():
        raise ValueError('Analysis implementation changed after CV preparation')
    frozen = selection_payload(cv, discovery, inputs)
    # Exclusive creation prevents a second process from replacing the freeze.
    with target.open('x') as handle:
        json.dump(clean(frozen), handle, indent=2, allow_nan=False)
        handle.write('\n')
    lines = ['# Frozen CV selection before confirmation scores', '',
        f"Frozen at {frozen['frozen_at']}. Selected: **{frozen['selected_metric']}**.", '',
        'Primary selection uses seven candidate-minus-Q0-4h changes: four historical alternatives '
        'and three discovery candidates. New candidates use the regenerated control; historical '
        'ones use historical Q0-4h. The control LB is 344.3114 seconds. '
        'The two confirmation LB outcomes were not supplied to this command.', '',
        'Twenty fixed forecast policies combine the ten unchanged CV readouts with either '
        'RMSE-delta transport or MSE-delta transport (`__mse`). MSE transport computes '
        '`sqrt(anchor_LB² + candidate_CV² − control_CV²)` and rejects a negative radicand. '
        'Raw CV changes and transported forecast changes are stored separately. '
        'This amendment was specified before new leaderboard outcomes were revealed.', '',
        '| Metric | Anchored delta MAE | Raw level MAE | Misorderings | Missed separations | Selection regret |',
        '|---|---:|---:|---:|---:|---:|']
    for metric, row in frozen['metrics'].items():
        lines.append(f"| {metric} | {row['delta_mae']:.4f} | {row['raw_level_mae']:.4f} | "
            f"{row['pair_counts']['misordering']} | {row['pair_counts']['missed_separation']} | "
            f"{row['observed_selection_regret']:.4f} |")
    lines += ['', '| Held-out recipe | Selected anchored LB forecast | Existing v3 anchored LB forecast |',
              '|---|---:|---:|']
    for recipe in CONFIRMATION:
        a = frozen['confirmation_forecasts'][frozen['selected_metric']][recipe]['anchored_lb']
        b = frozen['confirmation_forecasts']['R_combined'][recipe]['anchored_lb']
        lines.append(f'| {recipe} | {a:.4f} | {b:.4f} |')
    lines += ['', 'Leave-one-family-out sensitivity (diagnostic only; primary selection unchanged):', '',
        '| Family held out | Winner using other families | Held-out family delta MAE | Existing v3 held-out delta MAE |',
        '|---|---|---:|---:|']
    for family, row in frozen['leave_one_family_out']['families'].items():
        lines.append(f"| {family} | {row['winning_metric']} | {row['winning_metric_heldout']['delta_mae']:.4f} | "
                     f"{row['baseline_heldout']['delta_mae']:.4f} |")
    lines += ['', frozen['interpretation'], '',
        'Additive level calibration is stored separately and did not select the metric. '
        'Conditional CV bootstrap intervals do not quantify hidden-label LB uncertainty.', '']
    (run_root/'cv_selection_frozen.md').write_text('\n'.join(lines))
    return frozen


def confirmation_payload(frozen, official):
    if set(official) != set(CONFIRMATION):
        raise ValueError('Expected only the two held-out official scores')
    control_lb = frozen['anchor_lb']
    heldout = {ANCHOR: control_lb, **official}
    results = {}
    for metric, forecasts in frozen['confirmation_forecasts'].items():
        predictions = {ANCHOR: control_lb, **{name: forecasts[name]['anchored_lb'] for name in official}}
        row = alignment(predictions, heldout)
        row['anchored_level_mae'] = row['delta_mae']
        row['raw_level_mae'] = float(np.mean([abs(forecasts[n]['raw_cv']-official[n]) for n in official]))
        row['additive_level_mae'] = float(np.mean([abs(forecasts[n]['additive_lb']-official[n]) for n in official]))
        row['predictions'] = forecasts
        seed_predictions = {ANCHOR: control_lb, **{n: forecasts[n]['seed42']['anchored_lb'] for n in official}}
        row['seed42_diagnostic'] = alignment(seed_predictions, heldout)
        row['seed42_diagnostic']['raw_level_mae'] = float(np.mean([
            abs(forecasts[n]['seed42']['raw_cv']-official[n]) for n in official]))
        row['seed42_diagnostic']['additive_level_mae'] = float(np.mean([
            abs(forecasts[n]['seed42']['additive_lb']-official[n]) for n in official]))
        all_official = {**frozen['calibration_official'], **official}
        all_predictions = {**frozen['metrics'][metric]['calibration_predictions'], **predictions}
        row['all_observed_descriptive'] = alignment(all_predictions, all_official)
        results[metric] = row
    chosen, baseline = frozen['selected_metric'], frozen['baseline_metric']
    improvement = results[baseline]['delta_mae']-results[chosen]['delta_mae']
    chosen_errors, baseline_errors = results[chosen]['errors'], results[baseline]['errors']
    return dict(version=VERSION, evaluated_at=utcnow(), selected_metric=chosen,
        selected_readout=frozen['selected_readout'], selected_calibration=frozen['selected_calibration'],
        official=official, metrics=results, selected_minus_baseline_delta_mae=-improvement,
        improves_both_confirmation_errors=all(abs(chosen_errors[n]) < abs(baseline_errors[n]) for n in official),
        confirmatory_metrics=[chosen, baseline],
        interpretation='Only the frozen selected metric versus untouched R_combined is confirmatory. Other fixed readouts are descriptive. Two correlated candidates on one LB population cannot establish independent-year reliability; no metric is reselected here.')


def confirm(run_root, confirmation_path):
    run_root = Path(run_root)
    frozen_path = run_root/'cv_selection_frozen.json'
    frozen = read_json(frozen_path)
    if sha256(run_root/'cv_scores.json') != frozen['input_hashes']['cv_scores_sha256']:
        raise ValueError('CV scores changed after selection freeze')
    if frozen['source_hashes'] != source_hashes():
        raise ValueError('Analysis implementation changed after selection freeze')
    official = validate_score_file(confirmation_path, CONFIRMATION)
    result = confirmation_payload(frozen, official)
    result['frozen_sha256'] = sha256(frozen_path)
    result['confirmation_scores_sha256'] = sha256(confirmation_path)
    target = run_root/'confirmation_report.json'
    if target.exists():
        previous = read_json(target)
        for key in ('frozen_sha256', 'confirmation_scores_sha256'):
            if previous[key] != result[key]:
                raise ValueError('Existing confirmation report uses different frozen inputs')
        return previous
    write_json(target, result)
    lines = ['# Held-out leaderboard confirmation', '',
        f"Frozen selected metric: **{result['selected_metric']}**. Existing baseline: **R_combined**.", '',
        '| Recipe | Official LB | Selected forecast | Existing v3 forecast | Selected error | Existing error |',
        '|---|---:|---:|---:|---:|---:|']
    for name in CONFIRMATION:
        selected = result['metrics'][result['selected_metric']]
        base = result['metrics']['R_combined']
        lines.append(f"| {name} | {official[name]:.4f} | {selected['predictions'][name]['anchored_lb']:.4f} | "
            f"{base['predictions'][name]['anchored_lb']:.4f} | {selected['errors'][name]:+.4f} | {base['errors'][name]:+.4f} |")
    lines += ['', '| Metric | Held-out delta MAE | Raw level MAE | Frozen additive level MAE | Misorderings | Missed separations | Selection regret |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys((result['selected_metric'], 'R_combined')):
        row = result['metrics'][name]
        lines.append(f"| {name} | {row['delta_mae']:.4f} | {row['raw_level_mae']:.4f} | "
            f"{row['additive_level_mae']:.4f} | {row['pair_counts']['misordering']} | "
            f"{row['pair_counts']['missed_separation']} | {row['observed_selection_regret']:.4f} |")
    lines += ['', result['interpretation'], '',
        'The observed regret here compares choosing among the unchanged control and the two '
        'held-out candidates. Every pairwise decision and every fixed readout is retained in JSON; '
        'these overlapping pairs are not independent trials.', '']
    (run_root/'confirmation_report.md').write_text('\n'.join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'select', 'confirm'):
        child = sub.add_parser(command)
        child.add_argument('--run-root', type=Path, default=DEFAULT_ROOT)
        if command == 'prepare':
            child.add_argument('--bootstrap-reps', type=int, default=2000)
        elif command == 'select':
            child.add_argument('--discovery-scores', type=Path, required=True)
        else:
            child.add_argument('--confirmation-scores', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args.run_root, args.bootstrap_reps)
    elif args.command == 'select':
        result = select(args.run_root, args.discovery_scores)
        print(f"Frozen metric: {result['selected_metric']}")
    else:
        result = confirm(args.run_root, args.confirmation_scores)
        print(f"Confirmation selected-minus-v3 delta MAE: {result['selected_minus_baseline_delta_mae']:+.4f}")


if __name__ == '__main__':
    main()
