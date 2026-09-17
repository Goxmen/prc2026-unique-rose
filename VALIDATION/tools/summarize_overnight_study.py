"""Write the final five-submission CV report after sealed confirmation finishes.

Offline only: reads frozen local artifacts, generates Markdown and standalone
PNG/PDF figures, and never fetches scores, trains models, or changes default CV.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import overnight_analysis as analysis
from overnight_models import RECIPES
from records import sha256

DEFAULT_RUN = ROOT / 'VALIDATION/runs/overnight_20260917'
DEFAULT_AUDIT = ROOT / 'VALIDATION/reports/overnight_20260917'
DEFAULT_REPORT = ROOT / 'VALIDATION/reports/OVERNIGHT_SUMMARY_20260917.md'
SELECTED = analysis.DISCOVERY + analysis.CONFIRMATION
DISPLAY = ('R_combined', 'month_balanced', 'seasonal_10m', 'seasonal_11m')
REQUIRED = ('result.json', 'progress.json', 'submission_manifest.json', 'cv_scores.json',
            'cv_selection_frozen.json', 'confirmation_report.json',
            'discovery_scores.json', 'confirmation_scores.json')


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Audit chronology must contain timezone-aware timestamps')
    return stamp.astimezone(timezone.utc)


def number(value, signed=False, digits=4):
    if value is None or not np.isfinite(value):
        raise ValueError('A required final-report number is missing or nonfinite')
    return f'{float(value):+.{digits}f}' if signed else f'{float(value):.{digits}f}'


def relative_link(path, output, label=None):
    path = Path(path)
    relative = Path(os.path.relpath(path, output.parent)).as_posix()
    return f'[{label or path.name}]({relative})'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '|' + '|'.join('---' for _ in headers) + '|',
            *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows], '']


def validate_inputs(run_root, audit_root):
    """Reject partial, smoke, unverified or unsealed results before any write."""
    files = {name: run_root / name for name in REQUIRED}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError('Final summary requires completed artifacts: ' + ', '.join(missing))
    data = {name.removesuffix('.json'): read_json(path) for name, path in files.items()}
    run, progress, batch = data['result'], data['progress'], data['submission_manifest']
    cv, frozen, confirmation = data['cv_scores'], data['cv_selection_frozen'], data['confirmation_report']
    config = run['config']
    if (run['status'] != 'complete' or config.get('smoke') or config.get('sample') != 1.
            or config.get('rounds') != 800 or config.get('seeds') != [42, 142, 242]
            or cv.get('execution_only')):
        raise ValueError('Final summary requires the complete nonsmoke three-seed benchmark')
    if cv['runner_config'] != config:
        raise ValueError('CV scorecard config does not match the completed runner')
    items = batch['submissions']
    if len(items) != 5 or tuple(item['recipe'] for item in items) != SELECTED:
        raise ValueError('Expected exactly the five preregistered submissions in queue order')
    for item in items:
        result = item.get('official_result', {})
        score = item.get('score')
        if (item.get('state') != 'uploaded' or result.get('status') != 'Succeeded'
                or item.get('official_status') != 'Succeeded'
                or item.get('sha256') != item.get('remote_sha256')
                or not item.get('sha256') or result.get('used_pairs') != 344_841
                or result.get('file') != item['file'] or score != result.get('score')
                or isinstance(score, bool) or not isinstance(score, (float, int))
                or not np.isfinite(score) or score < 0):
            raise ValueError(f"Missing successful verified complete score: {item['recipe']}")
    official = {item['recipe']: item['score'] for item in items}
    discovery, heldout = data['discovery_scores'], data['confirmation_scores']
    if (set(discovery) != set(analysis.DISCOVERY) or set(heldout) != set(analysis.CONFIRMATION)
            or {**discovery, **heldout} != official or confirmation['official'] != heldout):
        raise ValueError('Stage score files/report disagree with successful submission results')
    frozen_hash = sha256(files['cv_selection_frozen.json'])
    if (frozen.get('status') != 'frozen'
            or frozen['input_hashes']['cv_scores_sha256'] != sha256(files['cv_scores.json'])
            or frozen['input_hashes']['discovery_scores_sha256'] != sha256(files['discovery_scores.json'])
            or confirmation['frozen_sha256'] != frozen_hash
            or confirmation['confirmation_scores_sha256'] != sha256(files['confirmation_scores.json'])
            or batch.get('confirmation_seal_sha256') != frozen_hash
            or confirmation['selected_metric'] != frozen['selected_metric']):
        raise ValueError('CV selection/confirmation seal or its frozen inputs changed')
    for source, digest in frozen['source_hashes'].items():
        if sha256(ROOT / source) != digest:
            raise ValueError(f'Frozen analysis source changed: {source}')
    if frozen['calibration_official'] != {**cv['old_official'], **discovery}:
        raise ValueError('Calibration outcomes differ from the old plus discovery observations')
    prereg_path = audit_root / 'preregistration.json'
    status_path = audit_root / 'status_before.json'
    prereg, before = read_json(prereg_path), read_json(status_path)
    if (tuple(prereg['discovery']) != analysis.DISCOVERY
            or tuple(prereg['confirmation']) != analysis.CONFIRMATION
            or prereg['authorized_new_uploads'] != 5
            or sha256(ROOT / prereg['plan_path']) != prereg['plan_sha256']):
        raise ValueError('Prerecorded study plan/roles changed')
    freeze_at = timestamp(frozen['frozen_at'])
    reveal_at = timestamp(batch['confirmation_reveal_authorized_at'])
    if not timestamp(prereg['frozen_at']) <= timestamp(run['started_at']) <= freeze_at <= reveal_at:
        raise ValueError('Prerecording, execution, CV freeze and reveal chronology is inconsistent')
    for item in items:
        observed = timestamp(item['score_checked_at'])
        if item['role'] == 'discovery' and observed > freeze_at:
            raise ValueError('Discovery score timestamp follows CV selection; inspect the audit history')
        if item['role'] == 'confirmation' and observed < reveal_at:
            raise ValueError('A confirmation score was recorded before its frozen reveal seal')
    if timestamp(confirmation['evaluated_at']) < max(timestamp(i['score_checked_at']) for i in items):
        raise ValueError('Confirmation report predates an official score record')
    data.update(official=official, preregistration=prereg, status_before=before,
                input_files=files, audit_root=audit_root, run_root=run_root)
    data['input_files']['preregistration.json'] = prereg_path
    data['input_files']['status_before.json'] = status_path
    amendment_path = audit_root / 'amendment1.json'
    amendment = read_json(amendment_path) if amendment_path.exists() else None
    if amendment is None and any(policy.endswith('__mse') for policy in frozen['metrics']):
        raise ValueError('MSE-delta policies require the recorded pre-outcome amendment')
    if amendment is not None:
        amended_at = timestamp(amendment['frozen_at'])
        first_upload = min(timestamp(min(item.get('upload_intents') or [item['uploaded_at']])) for item in items)
        if (amendment.get('original_preregistration_sha256') != sha256(prereg_path)
                or sha256(ROOT / amendment['doc_path']) != amendment.get('doc_sha256')
                or amendment.get('new_batch_results_seen') != 0
                or amendment.get('new_batch_uploads') != 0
                or amendment.get('models_and_submission_queue_changed') is not False
                or amendment.get('primary_selection_criterion_changed') is not False
                or amendment.get('forecast_policies') != 20
                or len(frozen['metrics']) != amendment['forecast_policies']
                or not timestamp(run['started_at']) <= amended_at < first_upload
                or amended_at > freeze_at):
            raise ValueError('Forecast-policy amendment identity or pre-upload chronology is invalid')
        data['input_files']['amendment1.json'] = amendment_path
        data['input_files']['amendment1_document.md'] = ROOT / amendment['doc_path']
    data['amendment1'] = amendment
    after_path = audit_root / 'status_after.json'
    data['status_after'] = read_json(after_path) if after_path.exists() else None
    if data['status_after'] is not None:
        data['input_files']['status_after.json'] = after_path
    checks_path = audit_root / 'checks.json'
    data['checks'] = read_json(checks_path) if checks_path.exists() else None
    if data['checks'] is not None:
        data['input_files']['checks.json'] = checks_path
    replay_path = audit_root / 'control_replay_final.json'
    data['control_replay_final'] = read_json(replay_path) if replay_path.exists() else None
    if data['control_replay_final'] is not None:
        replay = data['control_replay_final']
        if (replay.get('rows') != 344_841
                or not isinstance(replay.get('exact_matches'), int)
                or not 0 <= replay['exact_matches'] <= replay['rows']
                or not replay.get('historical_file')
                or not replay.get('refit_prediction_sha256') or not replay.get('reference_sha256')
                or any(not isinstance(replay.get(name), (int, float))
                       or not np.isfinite(replay[name]) or replay[name] < 0
                       for name in ('rmse_between_prediction_vectors',
                                    'max_abs_prediction_change', 'mean_abs_prediction_change'))):
            raise ValueError('Final control replay audit is incomplete or invalid')
        data['input_files']['control_replay_final.json'] = replay_path
    return data


def decision_pairs(row):
    """Specific materially wrong ordering/near-tie decisions, not pair counts."""
    return {(pair['candidate'], pair['reference']) for pair in row['pairs']
            if pair['decision'] in ('misordering', 'missed_separation')}


def policy_readout(frozen, policy):
    return frozen['metrics'][policy].get('readout', policy.removesuffix('__mse'))


def policy_calibration(frozen, policy):
    return frozen['metrics'][policy].get('calibration', 'mse_delta' if policy.endswith('__mse') else 'rmse_delta')


def verdict(data):
    chosen = data['cv_selection_frozen']['selected_metric']
    rows = data['confirmation_report']['metrics']
    baseline, selected = rows['R_combined'], rows[chosen]
    improvement = baseline['delta_mae'] - selected['delta_mae']
    new_mistakes = decision_pairs(selected) - decision_pairs(baseline)
    regret_not_worse = selected['observed_selection_regret'] <= baseline['observed_selection_regret'] + 1e-9
    near_tie_not_worse = (selected['worst_near_tie_selection_regret']
                         <= baseline['worst_near_tie_selection_regret'] + analysis.TOLERANCE)
    limited_progress = (chosen != 'R_combined' and improvement > analysis.TOLERANCE
                        and not new_mistakes and regret_not_worse and near_tie_not_worse)
    if limited_progress:
        headline = 'Limited prospective progress; more varied submissions are still needed.'
        explanation = (f'The frozen {chosen} forecast policy reduced error in the two held-out candidate changes '
            f'by {number(improvement, digits=2)} seconds on average, without adding a material '
            'misordering or missed separation or increasing observed selection regret. '
            'This is useful confirmation of a candidate CV/forecast adjustment, not established broad reliability.')
    elif chosen == 'R_combined':
        headline = 'No replacement CV was established; more varied submissions are needed.'
        explanation = ('The prespecified calibration rule retained the existing R_combined readout. '
            'The two held-out outcomes therefore test the incumbent and do not demonstrate a better replacement.')
    else:
        headline = 'No reliably better CV was established; more varied submissions are needed.'
        explanation = (f'The frozen {chosen} forecast policy changed held-out change-prediction MAE by '
            f'{number(-improvement, signed=True, digits=2)} seconds versus R_combined '
            '(negative is better). It did not improve prediction of the held-out changes, '
            'so the prospective comparison does not support replacing the current CV.')
    return dict(headline=headline, explanation=explanation, limited_progress=limited_progress,
                delta_mae_improvement=improvement, added_material_mistakes=sorted(new_mistakes),
                regret_not_worse=regret_not_worse, near_tie_not_worse=near_tie_not_worse)


def forecasts(data, metric, mode='standard'):
    """Frozen LB forecast deltas, which differ from raw CV deltas under MSE transport."""
    frozen = data['cv_selection_frozen']
    row = frozen['metrics'][metric]
    calibration = row['calibration_predictions'] if mode == 'standard' else row['seed42_diagnostic']['calibration_predictions']
    values = {name: calibration[name] for name in analysis.DISCOVERY}
    for name in analysis.CONFIRMATION:
        forecast = frozen['confirmation_forecasts'][metric][name]
        values[name] = (forecast if mode == 'standard' else forecast['seed42'])['anchored_lb']
    return {name: value - frozen['anchor_lb'] for name, value in values.items()}


def draw_chart(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selected = data['cv_selection_frozen']['selected_metric']
    official = data['official']
    anchor = data['cv_scores']['old_official'][analysis.ANCHOR]
    actual = {name: official[name] - anchor for name in SELECTED}
    metrics = ('R_combined', selected)
    predicted = {metric: forecasts(data, metric) for metric in metrics}
    values = list(actual.values()) + [v for row in predicted.values() for v in row.values()] + [0.]
    span = max(max(values) - min(values), 2.)
    lower, upper = min(values) - .16 * span, max(values) + .22 * span
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharex=True, sharey=True,
                             layout='constrained')
    colors = dict(zip(SELECTED, ('#2878b5', '#3b9e78', '#9563bd', '#d78225', '#cb4b4b')))
    offsets = ((7, 10), (7, -15), (7, 19), (7, -25), (7, 3))
    for axis, metric in zip(axes, metrics):
        axis.plot([lower, upper], [lower, upper], color='#777777', lw=1.0, ls='--', label='Exact delta forecast')
        axis.axhline(0, color='#dddddd', lw=.8)
        axis.axvline(0, color='#dddddd', lw=.8)
        for name, offset in zip(SELECTED, offsets):
            confirmation = name in analysis.CONFIRMATION
            axis.scatter(actual[name], predicted[metric][name], s=84 if confirmation else 65,
                marker='D' if confirmation else 'o', color=colors[name],
                facecolors=colors[name] if confirmation else 'none', linewidths=1.6, zorder=3)
            axis.annotate(name, (actual[name], predicted[metric][name]), xytext=offset,
                          textcoords='offset points', fontsize=8, color=colors[name])
        axis.set_title('Existing CV + RMSE-delta transport' if axis is axes[0] else f'Frozen selection: {selected}')
        axis.set_xlabel('Observed LB change versus Q0-4h (seconds)')
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.grid(alpha=.13)
        axis.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel('Forecast LB change versus Q0-4h (seconds)')
    fig.suptitle('Five new model changes: CV predictions versus official leaderboard', fontsize=13)
    handles = [plt.Line2D([], [], marker='o', markersize=7, linestyle='None',
                         markeredgecolor='#444444', markerfacecolor='none', label='Discovery'),
               plt.Line2D([], [], marker='D', markersize=7, linestyle='None', color='#444444',
                          label='Held-out confirmation')]
    axes[0].legend(handles=handles, loc='upper left', frameon=False, fontsize=8)
    png, pdf = output.with_suffix('.png'), output.with_suffix('.pdf')
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def paired_cell(row):
    low, high = row['ci95']
    return f"{number(row['difference'], True, 2)} [{number(low, True, 2)}, {number(high, True, 2)}]"


def report(data, output, chart_paths):
    run, progress = data['result'], data['progress']
    cv, frozen, confirmation = data['cv_scores'], data['cv_selection_frozen'], data['confirmation_report']
    selected, official = frozen['selected_metric'], data['official']
    selected_readout = policy_readout(frozen, selected)
    selected_calibration = policy_calibration(frozen, selected)
    anchor = cv['old_official'][analysis.ANCHOR]
    decision = verdict(data)
    items = {row['recipe']: row for row in data['submission_manifest']['submissions']}
    old_batch = read_json(analysis.MANIFEST)
    old_items = {row['codename']: row for row in old_batch['submissions']}
    lines = ['# Five new submissions and prospective CV assessment — 17 September 2026', '',
             '**' + decision['headline'] + '**', '', decision['explanation'], '',
             'Two held-out recipes on the same competition population cannot establish performance on '
             'another year or broad modelling changes. Keep the current default CV unchanged while '
             'collecting further varied submissions. No fold was dropped and no validation default was changed. '
             'A calibrated leaderboard forecast is distinct from the raw CV score.', '',
             f'All five authorized uploads succeeded. The best new model is **clock_features / v19: '
             f'{number(official["clock_features"])} seconds**, an improvement of '
             f'**{number(anchor - official["clock_features"])} seconds** over Q0-4h/v15. '
             'This is meaningful model progress; it does not validate a replacement CV.', '',
             '## What was tested', '',
             'Five distinct interventions were trained, validated and submitted. All use the fixed four-hour '
             'Rome rule, recovery off, direct regression, 800 rounds and full evaluation populations. '
             'Local validation uses seeds 42/142/242; submissions use seed 42 and all twelve 2025 months. '
             'The local control is Q0-4h; its already-scored official reference is '
             f'{number(anchor)} seconds. Negative changes mean improvement.', '']
    lines += table(['Recipe', 'Change', 'Evidence role', 'Submission'], [
        [name, RECIPES[name]['hypothesis'], items[name]['role'], items[name]['file']] for name in SELECTED])
    lines += ['Original CV retains all sixteen folds across six chronological origins. Seasonal validation '
              'uses the ten-month joint January/July holdout and two eleven-month single-month holdouts. '
              'The common feature matrix is scoped to each training window; evaluated departures never '
              'supply departure labels to training. Rule routing removes eligible rows only from model fitting, '
              'preserving history context and every evaluation row.', '', '## All ten official scores and local CV levels', '',
              'CV entries below average squared errors across three seeds. Repeated origins first average '
              'losses on the same flights. These are local scores, not averages of the submitted seed-42 predictions. '
              'Historical rows retain their original historical scores; the new rows use this execution.', '']
    rows = []
    for name in analysis.OLD + SELECTED:
        old = name in analysis.OLD
        scores = cv['old' if old else 'new'][name]['standard']
        item = old_items[name] if old else items[name]
        filename = item.get('file', item.get('filename', ''))
        if not filename:
            raise ValueError(f'Missing original submission filename: {name}')
        lb = cv['old_official'][name] if old else official[name]
        rows.append([name, filename, number(lb), *[number(scores[m]) for m in DISPLAY]])
    lines += table(['Recipe', 'Submission', 'LB', 'Original v3', 'Month balanced', 'Seasonal 10m', 'Seasonal 11m'], rows)
    control_scores = cv['new']['control']['standard']
    lines += ['The regenerated local control scores are ' + ', '.join(
        f'{metric} {number(control_scores[metric])}' for metric in DISPLAY) + '. '
        'It was not submitted again. New candidate deltas subtract this matched execution control; '
        'historical deltas subtract historical Q0-4h. This avoids attributing host or replay differences '
        'to the candidate intervention.', '']
    replay = data.get('control_replay_final')
    if replay is None:
        lines += ['Full-year control replay against the historical submitted prediction vector: '
                  '**not checked in the supplied audit artifacts**. No zero-drift assumption is made.', '']
    else:
        lines += ['The seed-42 full-year control was also replayed and compared, after submission rounding, '
                  f"with the historical **{replay['historical_file']}** prediction artifact. This uses "
                  'prediction vectors only; hidden leaderboard labels were not accessed.', '']
        lines += table(['Control replay measure', 'Value'], [
            ['Ranking rows compared', replay['rows']],
            ['Exactly matching rounded predictions', replay['exact_matches']],
            ['RMSE between prediction vectors, seconds', number(replay['rmse_between_prediction_vectors'], digits=6)],
            ['Mean absolute prediction change, seconds', number(replay['mean_abs_prediction_change'], digits=6)],
            ['Maximum absolute prediction change, seconds', number(replay['max_abs_prediction_change'], digits=6)]])
        lines += ['By the triangle inequality, the RMSE between two prediction vectors bounds the '
                  'absolute difference between their leaderboard RMSEs against any fixed target vector. '
                  'The control replay therefore bounds possible host/replay score drift at '
                  f"**{number(replay['rmse_between_prediction_vectors'], digits=6)} seconds**, without "
                  'knowing the hidden labels. It does not identify the sign of any drift or quantify '
                  '2025-to-2026 population shift. '
                  + relative_link(data['audit_root'] / 'control_replay_final.json', output, 'Replay audit and full hashes')
                  + '.', '']
    lines += ['## Changes versus the matched control', '',
        'Intervals are paired calendar-day bootstrap intervals on 2025 validation losses, conditional '
        'on the fitted seeds and origins. They do not estimate hidden-2026 leaderboard uncertainty.', '']
    rows = [[name, number(official[name] - anchor, True, 2),
             paired_cell(cv['paired_bootstrap'][name]['R_combined']),
             paired_cell(cv['paired_bootstrap'][name][selected_readout])]
            for name in SELECTED]
    lines += table(['Recipe', 'Official LB Δ', 'Original v3 Δ [95% interval]',
                    f'{selected_readout} raw CV Δ [95% interval]'], rows)
    lines += ['These intervals apply to raw local score changes. They are not uncertainty intervals '
              'for the calibrated leaderboard forecasts, including MSE-delta transport.', '']
    lines += ['## What the additional submissions taught us', '',
              '**The extra variety was useful, but five submissions did not establish a reliably better CV.** '
              'Removing month looked beneficial under the original combined CV and regressed on LB. '
              'Removing history had an inconclusive small local regression but improved on LB. '
              'Clock features improved LB much more than the original CV suggested. Training-row routing '
              'did improve LB, but by much less than every tested CV summary suggested.', '',
              'The frozen alternative used **R_A: the six next-month folds**, with squared-error-change '
              'calibration to the known LB baseline. It forecasts the smaller-tree result slightly more '
              'accurately, but overstates the routing improvement more than the current method does. '
              'A calibration can bring score levels closer without making model choices more reliable.', '']
    total_calibration_gain = sum(
        abs(frozen['metrics']['R_combined']['calibration_predictions'][name]
            - frozen['calibration_official'][name])
        - abs(frozen['metrics'][selected]['calibration_predictions'][name]
              - frozen['calibration_official'][name])
        for name in frozen['calibration_official'] if name != analysis.ANCHOR)
    b0_gain = (abs(frozen['metrics']['R_combined']['calibration_predictions']['B0-800']
                   - frozen['calibration_official']['B0-800'])
               - abs(frozen['metrics'][selected]['calibration_predictions']['B0-800']
                     - frozen['calibration_official']['B0-800']))
    if total_calibration_gain > 0:
        lines += [f'About **{number(100 * b0_gain / total_calibration_gain, digits=1)}%** of the '
                  'selected policy\'s calibration error reduction comes from fitting the single historical '
                  'B0-800 outcome more closely. Removing that observation changes the selected policy '
                  f'to **{frozen["leave_one_calibration_candidate_out_winner"]["B0-800"]}**. '
                  'This concentration helps explain why the calibration improvement was not broad '
                  'evidence of better prediction of new interventions.', '']
    family = frozen['leave_one_family_out']['families']['historical_rule_recovery']
    rows = []
    for metric in ('R_combined', selected, 'seasonal_11m', family['winning_metric']):
        row = frozen['metrics'][metric]
        discovery_mae = np.mean([abs(row['calibration_predictions'][name] - official[name])
                                 for name in analysis.DISCOVERY])
        rows.append([metric, number(discovery_mae),
                     number(family['all_metric_heldout_delta_mae'][metric]),
                     number(confirmation['metrics'][metric]['delta_mae'])])
    lines += table(['Forecast policy', 'Three feature changes: change MAE',
                    'Four historical rule/recovery changes: change MAE', 'Two confirmation changes: change MAE'], rows)
    lines += ['This table is a descriptive family comparison, not a new selection. Seasonal eleven-month '
              'validation tracks the three feature changes much better than the original CV, while the '
              'historical rule/recovery changes favor other summaries. When the entire historical family '
              f'is left out of calibration, the winner becomes **{family["winning_metric"]}**, '
              'a fixed blend with 25% month-balanced and 75% seasonal-eleven-month MSE, followed by '
              'MSE-change transport. Its apparent advantage does not transfer to the omitted historical '
              'family or the two confirmation cases. The best summary depends on which kinds of model '
              'changes we show it.', '']
    winners = {row['all_observed_descriptive']['cv_winner'] for row in confirmation['metrics'].values()}
    if winners == {'rule_routing'}:
        lines += [f'**All {len(confirmation["metrics"])} tested forecast policies choose rule_routing '
                  'over clock_features across the ten candidates.** On LB, clock_features is better by '
                  f'**{number(official["rule_routing"] - official["clock_features"])} seconds**. '
                  'Even choosing a different policy after seeing confirmation would not fix this '
                  'practical model-selection failure. These are related readouts of the same data, '
                  'not twenty independent experiments.', '']
    gap_bound = np.mean([abs(frozen['confirmation_forecasts'][selected][name]['anchored_lb']
                            - frozen['confirmation_forecasts']['R_combined'][name]['anchored_lb'])
                        for name in analysis.CONFIRMATION])
    lines += ['The confirmation was also weakly discriminating: smaller trees changed LB by only '
              f'{number(official["small_trees"] - anchor, True)} seconds, within the one-second '
              'tolerance. The selected and original forecasts differ so little on these two cases '
              f'that the largest mathematically possible MAE advantage was only {number(gap_bound)} '
              'seconds, regardless of their eventual scores. The report uses a conservative greater-than-'
              'one-second gain gate when labelling limited progress; the original design specified '
              'improved confirmation predictions and a one-second tolerance, without an explicit '
              'greater-than-one-second MAE clause. Here the selected policy actually worsened MAE, '
              'so the negative verdict does not depend on that stricter reporting gate.', '',
              '### How this connects to the earlier validation work', '',
              'The earlier [seasonal comparison](../runs/seasonal_jan_jul/comparison.md) found closer '
              'absolute levels but poorer change-size alignment on the five rule/recovery submissions. '
              'The [fold audit](FOLD_ALIGNMENT.md) found no defensible harmful-fold removal. '
              'The [optimism diagnosis](VALIDATION_OPTIMISM.md) identified a concrete population mismatch: '
              'November/December receive 69.17% of the original headline MSE weight, July receives little '
              'weight, and January is absent. Rare January errors can also make seasonal scores pessimistic. '
              'The [month-balanced proposal](AGGREGATION_COMPARISON.md) changes those weights, but this '
              'larger study still provides no confirmed replacement. The additional outcomes show why '
              'closeness on one model family was insufficient evidence.', '',
              '### Recommended next evidence', '',
              'Keep the current CV as the operational reference and inspect next-month, longer-gap and '
              'seasonal diagnostics alongside it. Treat disagreements as uncertainty; do not fit free '
              'fold weights to these ten scores or remove difficult evaluation rows. The next submission '
              'batch should include different model/loss or training-population approaches, not only '
              'more variants of the same rule. All five new models here still use LightGBM squared-error '
              'regression and the same four-hour rule.', '',
              'Before seeing new outcomes, freeze a small number of CV alternatives and reserve '
              'confirmation candidates whose predictions differ enough to distinguish those alternatives. '
              'Test across intervention families, with more than two confirmation cases. There is no '
              'defensible fixed number of additional submissions that guarantees success; another five '
              'would be additional evidence, not a promised solution. No further uploads were made or '
              'scheduled. This batch made progress in diagnosing CV limitations, not in establishing '
              'a reliably better CV.', '']
    amendment = data.get('amendment1')
    if amendment:
        lines += ['### Amendment before any new leaderboard outcome', '',
            f"The forecast-policy amendment was recorded at **{amendment['frozen_at']}**, after training "
            'had started but **before the first new upload or any new leaderboard outcome**. It added '
            'fixed MSE-delta transport alongside the original RMSE-delta transport for the same ten '
            'CV summaries. It changed neither the five models/submission queue nor the primary selection '
            'criterion. The original preregistration remains unchanged and is bound into the amendment '
            'by SHA256. This was an amendment during execution, not part of the initial preregistration. '
            + relative_link(ROOT / amendment['doc_path'], output, 'Amendment rationale and formulas')
            + '; ' + relative_link(data['audit_root'] / 'amendment1.json', output, 'Amendment identity and timing')
            + '.', '']
    lines += ['## Discovery selection, frozen before confirmation', '',
        'The five historical submissions plus the first three new submissions formed the calibration set. '
        'Selection compared seven non-anchor candidate changes. Ten fixed CV summaries were considered, '
        'each with RMSE-delta and MSE-delta transfer assumptions, giving twenty fixed forecast policies; '
        'there was no arbitrary fold-subset or free-weight search. The unchanged original readout wins '
        'when it is within one second of the lowest calibration change-error MAE.', '',
        'RMSE-delta transport assumes that the local RMSE change transfers: '
        '`forecast = L0 + (Rc − R0)`. MSE-delta transport instead assumes that the local squared-error '
        'change transfers: `forecast = sqrt(L0² + Rc² − R0²)`, provided the radicand is nonnegative '
        '(an invalid negative radicand is rejected). Here `L0` is the known Q0-4h LB, `Rc` the candidate '
        'CV score and `R0` its matched control CV score. Neither assumption changes training, fold '
        'weights or the underlying raw CV scores.', '',
        f"Frozen selection: **{selected}**; underlying readout **{selected_readout}**, "
        f"calibration **{selected_calibration}**, at **{frozen['frozen_at']}**. "
        f"The two confirmation outcomes were authorized for reading at **{data['submission_manifest']['confirmation_reveal_authorized_at']}**.", '']
    rows = []
    for metric, row in frozen['metrics'].items():
        rows.append([metric, number(row['delta_mae']), number(row['raw_level_mae']),
                     row['pair_counts']['misordering'], row['pair_counts']['missed_separation'],
                     row['pair_counts']['false_separation'], number(row['observed_selection_regret'])])
    lines += table(['Forecast policy', 'Anchored change MAE', 'Raw level MAE', 'Misorderings',
                    'Missed separations', 'False separations', 'Selection regret'], rows)
    jackknife = Counter(frozen['leave_one_calibration_candidate_out_winner'].values())
    lines += ['Dropping one calibration candidate at a time selected: ' + ', '.join(
        f'{metric} in {count}/7 cases' for metric, count in sorted(jackknife.items())) + '. '
        'This measures sensitivity on the same calibration observations and is not independent confirmation.', '',
        '## Held-out confirmation', '',
        'Only the frozen choice versus unchanged R_combined is confirmatory. Other summaries remain '
        'descriptive after these outcomes are revealed. Forecasts use the known Q0-4h LB and their '
        'frozen RMSE-delta or MSE-delta transfer assumption; they are not raw CV scores.', '']
    rows = []
    for name in analysis.CONFIRMATION:
        chosen = frozen['confirmation_forecasts'][selected][name]
        base = frozen['confirmation_forecasts']['R_combined'][name]
        rows.append([name, number(official[name]), number(chosen['anchored_lb']),
                     number(base['anchored_lb']), number(chosen['anchored_lb'] - official[name], True),
                     number(base['anchored_lb'] - official[name], True)])
    lines += table(['Recipe', 'Actual LB', 'Frozen selected forecast', 'Original-v3 forecast',
                    'Selected forecast error', 'Original forecast error'], rows)
    rows = []
    for metric in dict.fromkeys(('R_combined', selected)):
        row = confirmation['metrics'][metric]
        rows.append([metric, number(row['delta_mae']), number(row['raw_level_mae']),
                     number(row['additive_level_mae']), row['pair_counts']['misordering'],
                     row['pair_counts']['missed_separation'], number(row['observed_selection_regret']),
                     number(row['worst_near_tie_selection_regret'])])
    lines += table(['Forecast policy', 'Anchored change/level MAE', 'Raw level MAE', 'Frozen offset level MAE',
                    'Misorderings', 'Missed separations', 'Selection regret', 'Worst near-tie regret'], rows)
    lines += ['The selected-minus-original held-out change MAE is '
              f"**{number(-decision['delta_mae_improvement'], True)} seconds**. "
              'A negative value favors the selected forecast policy. Improvement in both individual absolute '
              f"forecast errors: **{confirmation['improves_both_confirmation_errors']}**. "
              f"New material ordering/near-tie mistakes: **{len(decision['added_material_mistakes'])}**.", '',
              'Raw score closeness and change prediction answer different questions. The additive '
              'offset is the calibration-set median of LB minus raw CV, frozen before confirmation; '
              'improved offset forecasts alone do not establish better candidate ranking.', '']
    rows = []
    for name in SELECTED:
        for metric in dict.fromkeys(('R_combined', selected)):
            if name in analysis.CONFIRMATION:
                prediction = frozen['confirmation_forecasts'][metric][name]
            else:
                raw = frozen['metrics'][metric]['raw_cv'][name]
                prediction = dict(raw_cv=raw,
                    anchored_lb=frozen['metrics'][metric]['calibration_predictions'][name],
                    additive_lb=raw + frozen['metrics'][metric]['median_additive_offset'])
            rows.append([name, metric, number(prediction['raw_cv']),
                         number(official[name] - prediction['raw_cv'], True),
                         number(official[name] - prediction['anchored_lb'], True),
                         number(official[name] - prediction['additive_lb'], True)])
    lines += table(['Recipe', 'Forecast policy', 'Raw CV', 'LB − raw CV', 'LB − anchored forecast',
                    'LB − frozen-offset forecast'], rows)
    lines += ['### Seed-42 diagnostic', '',
              'This matches the submission seed but did not select the validation readout. Each old '
              'candidate uses its old seed-42 anchor; each new candidate uses the regenerated '
              'seed-42 control, under the same fixed transfer assumption.', '']
    rows = []
    for metric in dict.fromkeys(('R_combined', selected)):
        row = confirmation['metrics'][metric]['seed42_diagnostic']
        rows.append([metric, number(row['delta_mae']), number(row['raw_level_mae']),
                     row['pair_counts']['misordering'], row['pair_counts']['missed_separation'],
                     number(row['observed_selection_regret'])])
    lines += table(['Forecast policy', 'Held-out change MAE', 'Raw level MAE', 'Misorderings',
                    'Missed separations', 'Selection regret'], rows)
    lines += ['### Candidate ranking across all observed submissions', '',
              'These comparisons use all ten results and are descriptive. The many overlapping pairs '
              'are dependent; they are not additional independent confirmation trials. A material LB '
              'difference paired with a local difference of at most one second counts as a missed separation.', '']
    rows = []
    for metric in dict.fromkeys(('R_combined', selected)):
        row = confirmation['metrics'][metric]['all_observed_descriptive']
        rows.append([metric, row['pair_counts']['agree'], row['pair_counts']['misordering'],
                     row['pair_counts']['missed_separation'], row['pair_counts']['false_separation'],
                     row['cv_winner'], number(row['observed_selection_regret'])])
    lines += table(['Forecast policy', 'Agreements', 'Misorderings', 'Missed separations', 'False separations',
                    'CV winner', 'Selection regret'], rows)
    png, pdf = chart_paths
    lines += ['![' + 'Predicted versus observed new-candidate changes' + ']('
              + Path(os.path.relpath(png, output.parent)).as_posix() + ')', '',
              relative_link(pdf, output, 'Standalone PDF') + '. Hollow circles are discovery recipes; '
              'filled diamonds are the two held-out recipes. No prediction bands or hidden-label '
              'uncertainty estimates are implied.', '', '## Execution, reuse and upload audit', '']
    fitted = list(progress['models'].values())
    imported = [entry for entry in fitted if 'imported_from' in entry]
    newly_fitted = [entry for entry in fitted if 'imported_from' not in entry]
    refits = list(progress['refit_models'].values())
    cv_minutes = sum(entry['minutes'] for entry in newly_fitted)
    refit_minutes = sum(entry['minutes'] for entry in refits)
    wall = (timestamp(run['updated_at']) - timestamp(run['started_at'])).total_seconds() / 60
    lines += table(['Execution measure', 'Recorded value'], [
        ['New validation fits', len(newly_fitted)], ['Imported completed seasonal-control fits', len(imported)],
        ['Total validation model/seed/origin combinations', len(fitted)],
        ['Validation prediction files', len(progress['predictions'])],
        ['Completed full-year refits / ranking outputs', f"{len(refits)} / {len(progress['refits'])}"],
        ['New validation fit minutes, summed', number(cv_minutes, digits=2)],
        ['Full-year refit minutes, summed', number(refit_minutes, digits=2)],
        ['Recorded runner wall time through last update, minutes', number(wall, digits=2)],
        ['Fit threads / prediction threads / simultaneous fits', '8 / 8 / 1']])
    lines += ['Fit-minute totals include each model save/provenance step and exclude shared feature preparation '
              'and prediction work. Imported fits contribute zero new fit minutes. Runner wall time may '
              'include any gap between validation and refit commands. All model/prediction/source-snapshot '
              'artifacts remain local and ignored; concise records and analysis source are retained.', '']
    checks = data.get('checks')
    if checks:
        lines += [f'Verification: **{checks["tests"]} tests passed** at {checks["checked_at"]}; '
                  f'the complete suite took {number(checks["seconds"], digits=3)} seconds. '
                  'The execution-only smoke run covered all six recipes and nine windows '
                  '(54 tiny fits and 120 prediction files). Smoke scores were never used to rank models. '
                  + relative_link(data['audit_root'] / 'checks.json', output, 'Recorded checks') + '.', '']
    rows = []
    for name in SELECTED:
        item = items[name]
        uploaded = item.get('uploaded_at') or (item.get('upload_intents') or [item['verified_at']])[0]
        rows.append([item['file'], name, item['role'], uploaded, item['score_checked_at'],
                     item['official_status'], number(item['score']), item['sha256'][:16]])
    lines += table(['File', 'Recipe', 'Role', 'Upload UTC', 'Score read UTC', 'Status', 'LB', 'SHA256 prefix'], rows)
    lines += ['All five submitted artifacts retain the original template IDs/order/schema, have '
              '344,841 scored pairs, and were verified against remote bytes. The submission manifest '
              'retains full hashes, quota checks and raw official result payloads.', '']
    lines += table(['Chronology checkpoint', 'Timestamp'], [
        ['Study preregistration', data['preregistration']['frozen_at']],
        ['Validation runner start', run['started_at']],
        *([['Forecast-policy amendment before any new upload', amendment['frozen_at']]] if amendment else []),
        ['CV selection freeze', frozen['frozen_at']],
        ['Confirmation reveal authorization', data['submission_manifest']['confirmation_reveal_authorized_at']],
        ['Confirmation evaluation', confirmation['evaluated_at']]])
    lines += ['The confirmation reveal seal matches the final frozen-selection SHA256, and the generator '
              'verified that every discovery score was recorded before the freeze and each confirmation '
              'score after reveal authorization. No additional upload or automatic daily-reset retry '
              'is part of this five-submission study.', '', '## Best, latest and reproducibility', '']
    known = {item['file']: item['score'] for item in data['submission_manifest']['submissions']}
    known.update({item.get('file', item.get('filename')): item['score'] for item in old_batch['submissions']})
    best = min(known, key=lambda name: (known[name], name))
    latest = data['submission_manifest']['submissions'][-1]
    lines += [f'Best among these ten scored candidates: **{best}, {number(known[best])} seconds**. '
              f"Latest in this batch: **{latest['file']}, {number(latest['score'])} seconds**.", '']
    before = data['status_before']
    lines += [f"Before-run live snapshot: {before['updatedAt']}; team best "
              f"{before['team']['bestFile']} at {number(before['team']['bestScore'])} seconds.", '']
    after = data['status_after']
    if after:
        team = after['team']
        latest_live = team['latest']
        latest_score = (number(latest_live['score']) + ' seconds' if latest_live.get('score') is not None
                        else 'pending/unscored')
        lines += [f"Final {after['snapshotSource']} snapshot: {after['updatedAt']}; team best "
                  f"**{team['bestFile']}, {number(team['bestScore'])} seconds**; latest "
                  f"**{latest_live['file']}, {latest_score}**; rank "
                  f"**{team['rank']}/{team['scoringTeamCount']}** scoring teams.", '']
    else:
        lines += ['No final global leaderboard snapshot was supplied to this offline generator. '
                  'The best/latest figures above are restricted to the ten investigated candidates.', '']
    runtime = run['signature']['runtime']
    lines += ['Runtime: ' + runtime['cpu'] + '; Python ' + runtime['python'] + '; ' + ', '.join(
        f'{name} {version}' for name, version in sorted(runtime['versions'].items())) + '.', '']
    links = [(ROOT / data['preregistration']['plan_path'], 'Prespecified design'),
             (data['audit_root'] / 'preregistration.json', 'Prerecording and plan hash'),
             (data['run_root'] / 'result.json', 'Frozen source/data/runtime/config record'),
             (data['run_root'] / 'submission_manifest.json', 'Five-upload manifest and full hashes'),
             (data['run_root'] / 'cv_scores.json', 'Every fold, seed, population and interval'),
             (data['run_root'] / 'cv_selection_frozen.json', 'Immutable discovery selection and forecasts'),
             (data['run_root'] / 'confirmation_report.json', 'Held-out result and dependent all-pair diagnostics'),
             (Path(__file__), 'Offline report generator')]
    if amendment:
        links.insert(2, (ROOT / amendment['doc_path'], 'Pre-outcome forecast-policy amendment'))
        links.insert(3, (data['audit_root'] / 'amendment1.json', 'Amendment hash and timing'))
    lines += [relative_link(path, output, label) for path, label in links]
    lines += ['', 'Report input hashes:', '']
    lines += table(['Artifact', 'SHA256'], [[name, sha256(path)] for name, path in data['input_files'].items()])
    lines += ['Reproduce after all five official scores and sealed confirmation are complete:', '',
              '```bash', '.venv/bin/python VALIDATION/tools/summarize_overnight_study.py', '```', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--audit-root', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--output', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    run_root, audit_root, output = args.run_root.resolve(), args.audit_root.resolve(), args.output.resolve()
    data = validate_inputs(run_root, audit_root)
    # No artifact is written until every required successful outcome and seal is checked.
    output.parent.mkdir(parents=True, exist_ok=True)
    charts = draw_chart(data, output)
    text = report(data, output, charts)
    temporary = output.with_suffix('.md.tmp')
    temporary.write_text(text)
    temporary.replace(output)
    print(f'Saved {output}')
    print(verdict(data)['headline'])


if __name__ == '__main__':
    main()
