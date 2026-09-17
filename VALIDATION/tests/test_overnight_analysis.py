"""Aggregation invariants and separation of discovery from confirmation."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import overnight_analysis as a
from records import write_json, sha256
from reporting import Loss, bootstrap_calendar


def summaries():
    folds = {}
    for i, name in enumerate(a.ALL):
        month = a.FOLD_MONTHS[name]
        days = pd.Period(month, 'M').days_in_month
        n = days * (int(month[-2:])-5)
        values = np.array([i+1., i+3., i+5.])**2
        folds[name] = dict(month=month, n=n, mse=float(values.mean()),
            mse_by_seed=dict(zip(map(str, a.SEEDS), values)),
            count_by_day=[n//days]*days,
            sse_by_seed_day=[np.full(days, value*n/days).tolist() for value in values])
    seasonal = {s: {} for s in ('10m', '11m')}
    for s in seasonal:
        for month in a.MONTHS:
            row = copy.deepcopy(folds['A1'])
            row['month'] = month
            seasonal[s][month] = row
    return folds, seasonal


def fake_cv(discovery_error=3.):
    official = dict(zip(a.OLD, [406.4281, 354.4303, 354.0004, 344.3114, 344.4821]))
    old = {code: {'standard': dict.fromkeys(a.METRICS, score-100)} for code, score in official.items()}
    new = {a.CONTROL: {'standard': dict.fromkeys(a.METRICS, official[a.ANCHOR]-90)}}
    discovery = {}
    for i, name in enumerate(a.DISCOVERY+a.CONFIRMATION, 1):
        target = official[a.ANCHOR]+i*2
        new[name] = {'standard': dict.fromkeys(a.METRICS, target-90)}
        new[name]['standard']['R_combined'] += discovery_error
        if name in a.DISCOVERY:
            discovery[name] = target
    for row in list(old.values()) + list(new.values()):
        row['42'] = dict(row['standard'])
    return dict(version=a.VERSION, old=old, old_official=official, new=new,
                source_hashes=a.source_hashes(), execution_only=False), discovery


def make_prediction(path, fold='A1', month='2025-07', seed=42, offset=2.):
    days = pd.date_range(month+'-01', pd.Period(month, 'M').end_time, freq='D', tz='UTC')
    n = len(days)
    frame = pd.DataFrame(dict(mvt_id=[month+'-'+str(i) for i in range(n)], mvt_time=days,
        airport='EHAM', y=np.full(n, 600.), has_flight_plan=True,
        arrival_history_available=True, arrival_history_days_available=7.,
        seed=seed, fold=fold, month=month, raw_flight_id_missing=False,
        pred_raw=np.full(n, 600.+offset), pred=np.full(n, 600.+offset), rule_triggered=False))
    frame.to_parquet(path, index=False)
    return frame


class OvernightAnalysisTests(unittest.TestCase):
    def test_origin_pooling_is_equal_per_flight_not_equal_per_fold(self):
        folds, seasonal = summaries()
        result = a.aggregate(folds, seasonal, .44)
        by_month = {}
        for month in sorted(set(a.FOLD_MONTHS.values())):
            rows = [r for r in folds.values() if r['month'] == month]
            by_month[month] = (rows[0]['n'], np.mean([r['mse'] for r in rows]))
        expected = sum(n*v for n, v in by_month.values())/sum(n for n, _ in by_month.values())
        self.assertAlmostEqual(result['unique_flight']**2, expected)
        self.assertAlmostEqual(result['month_balanced']**2, np.mean([v for _, v in by_month.values()]))
        long_nov = folds['B3']['mse']
        long_dec = np.mean([folds[f]['mse'] for f in ('B4', 'B7')])
        expected_long = np.average([long_nov, long_dec], weights=[folds['B3']['n'], folds['B4']['n']])
        self.assertAlmostEqual(result['R_B_long']**2, expected_long)
        naive = np.average([folds[f]['mse'] for f in a.LONG], weights=[folds[f]['n'] for f in a.LONG])
        self.assertNotAlmostEqual(expected_long, naive)

    def test_standard_is_mean_seed_loss_and_blends_are_mse_blends(self):
        folds, seasonal = summaries()
        standard = a.aggregate(folds, seasonal, .44)
        seeds = [a.aggregate(folds, seasonal, .44, str(seed)) for seed in a.SEEDS]
        for metric in a.METRICS:
            self.assertAlmostEqual(standard[metric]**2, np.mean([r[metric]**2 for r in seeds]))
        self.assertNotAlmostEqual(standard['R_A'], np.mean([r['R_A'] for r in seeds]))
        self.assertAlmostEqual(standard['blend_25']**2,
                              .25*standard['month_balanced']**2+.75*standard['seasonal_11m']**2)

    def test_daily_bootstrap_reproduces_all_point_scores_for_constant_daily_loss(self):
        folds, seasonal = summaries()
        dummy = Loss(pd.DataFrame({'mvt_time': pd.to_datetime(
            [m+'-01' for m in sorted(set(a.FOLD_MONTHS.values())|set(a.MONTHS))], utc=True)}), np.zeros((1, 7)), (42,))
        calendar, weights = bootstrap_calendar({'all': dummy}, 25, 0)
        boots = a.bootstrap_scores(folds, seasonal, .44, calendar, weights)
        scores = a.aggregate(folds, seasonal, .44)
        for metric in a.METRICS:
            np.testing.assert_allclose(boots[metric], scores[metric], rtol=1e-12)

    def test_alignment_counts_material_missed_separation(self):
        observed = {a.ANCHOR: 344.31, 'Q1': 354., 'other': 344.48}
        predicted = {a.ANCHOR: 344.31, 'Q1': 344.55, 'other': 341.}
        result = a.alignment(predicted, observed)
        self.assertEqual(result['pair_counts']['missed_separation'], 1)
        self.assertEqual(result['pair_counts']['false_separation'], 1)
        self.assertEqual(result['observed_selection_regret'], .17000000000001592)

    def test_host_shift_cancels_and_confirmation_outcomes_are_not_calibration(self):
        cv, discovery = fake_cv()
        frozen = a.selection_payload(cv, discovery, {'cv_scores_sha256': 'a', 'discovery_scores_sha256': 'b'})
        self.assertEqual(frozen['selected_metric'], 'R_A')
        self.assertEqual(len(frozen['metrics']['R_A']['errors']), 7)
        self.assertAlmostEqual(frozen['metrics']['R_A']['delta_mae'], 0.)
        self.assertEqual(set(frozen['calibration_official']), set(a.OLD+a.DISCOVERY))
        self.assertEqual(set(frozen['confirmation_forecasts']['R_A']), set(a.CONFIRMATION))
        with self.assertRaisesRegex(ValueError, 'Confirmation'):
            a.selection_payload(cv, {**discovery, 'small_trees': 350.}, {})

    def test_negligible_calibration_win_retains_existing_metric(self):
        cv, discovery = fake_cv(discovery_error=1.)
        frozen = a.selection_payload(cv, discovery, {'cv_scores_sha256': 'a', 'discovery_scores_sha256': 'b'})
        self.assertEqual(frozen['selected_metric'], 'R_combined')

    def test_mse_transport_math_and_common_error_floor_invariance(self):
        value = a.anchored_forecast(400., 300., 500., 'mse_delta')
        self.assertAlmostEqual(value, np.sqrt(320_000.))
        shifted = a.anchored_forecast(np.sqrt(400.**2+50_000.),
                                     np.sqrt(300.**2+50_000.), 500., 'mse_delta')
        self.assertAlmostEqual(value, shifted)
        for kind in ('rmse_delta', 'mse_delta'):
            self.assertAlmostEqual(a.anchored_forecast(300., 300., 500., kind), 500.)
            self.assertAlmostEqual(a.anchored_forecast(400., 300., 300., kind), 400.)

    def test_mse_transport_preserves_cohort_order_and_rejects_negative_radicand(self):
        raw = [200., 300., 400.]
        predicted = [a.anchored_forecast(value, 300., 500., 'mse_delta') for value in raw]
        self.assertEqual(list(np.argsort(predicted)), list(np.argsort(raw)))
        self.assertLess(predicted[0], 500.)
        self.assertGreater(predicted[-1], 500.)
        self.assertEqual(a.anchored_forecast(0., 100., 100., 'mse_delta'), 0.)
        with self.assertRaisesRegex(ValueError, 'Negative MSE-delta.*no silent clipping'):
            a.anchored_forecast(0., 1_000., 100., 'mse_delta')

    def test_fixed_twenty_policies_can_select_mse_without_changing_readout(self):
        cv, discovery = fake_cv(discovery_error=0.)
        floor = 60_000.
        for name, score in cv['old_official'].items():
            for mode in ('standard', '42'):
                cv['old'][name][mode] = dict.fromkeys(a.METRICS, float(np.sqrt(score**2-floor)))
        for i, name in enumerate((a.CONTROL,)+a.DISCOVERY+a.CONFIRMATION):
            score = cv['old_official'][a.ANCHOR]+i*2
            for mode in ('standard', '42'):
                cv['new'][name][mode] = dict.fromkeys(a.METRICS, float(np.sqrt(score**2-floor+5_000.)))
        frozen = a.selection_payload(cv, discovery, {'cv_scores_sha256': 'a', 'discovery_scores_sha256': 'b'})
        self.assertEqual(len(frozen['candidate_metrics']), 20)
        self.assertEqual(frozen['candidate_readouts'], list(a.METRICS))
        self.assertEqual(frozen['selected_metric'], 'R_combined__mse')
        self.assertEqual(frozen['selected_readout'], 'R_combined')
        self.assertEqual(frozen['selected_calibration'], 'mse_delta')
        self.assertAlmostEqual(frozen['metrics']['R_combined__mse']['delta_mae'], 0.)
        self.assertEqual(len(frozen['confirmation_forecasts']), 20)
        forecast = frozen['confirmation_forecasts']['R_combined__mse']['small_trees']
        self.assertAlmostEqual(forecast['forecast_delta_vs_control'], 8.)
        self.assertNotAlmostEqual(forecast['raw_cv_delta_vs_control'], 8.)
        self.assertEqual(forecast['cv_delta_vs_control'], forecast['raw_cv_delta_vs_control'])
        self.assertEqual(forecast['raw_cv'], frozen['confirmation_forecasts']['R_combined']['small_trees']['raw_cv'])
        self.assertEqual(forecast['additive_lb'], frozen['confirmation_forecasts']['R_combined']['small_trees']['additive_lb'])
        scores = {n: frozen['confirmation_forecasts']['R_combined__mse'][n]['anchored_lb'] for n in a.CONFIRMATION}
        result = a.confirmation_payload(frozen, scores)
        self.assertEqual(result['selected_readout'], 'R_combined')
        self.assertEqual(result['selected_calibration'], 'mse_delta')
        self.assertAlmostEqual(result['metrics']['R_combined__mse']['delta_mae'], 0.)

    def test_historical_mse_demo_reads_only_frozen_old_references(self):
        old = a.historical()
        anchor = old['official'][a.ANCHOR]
        control = old['scores'][a.ANCHOR]['standard']['R_combined']
        raw = old['scores']['B0-800']['standard']['R_combined']
        rmse = a.anchored_forecast(raw, control, anchor, 'rmse_delta')
        mse = a.anchored_forecast(raw, control, anchor, 'mse_delta')
        self.assertGreater(abs(rmse-old['official']['B0-800']), 10.)
        self.assertLess(abs(mse-old['official']['B0-800']), 1.)

    def test_family_sensitivity_removes_correlated_old_recipes_together(self):
        cv, discovery = fake_cv(discovery_error=0.)
        for name in a.OLD:
            if name == a.ANCHOR:
                continue
            for metric in a.METRICS:
                if metric != 'R_A':
                    cv['old'][name]['standard'][metric] += 12.
        for name in a.DISCOVERY:
            cv['new'][name]['standard']['R_A'] += 8.
        frozen = a.selection_payload(cv, discovery, {'cv_scores_sha256': 'a', 'discovery_scores_sha256': 'b'})
        self.assertEqual(frozen['selected_metric'], 'R_A')
        sensitivity = frozen['leave_one_family_out']
        self.assertTrue(sensitivity['diagnostic_only'])
        self.assertEqual(set(sensitivity['families']), {'historical_rule_recovery', *a.DISCOVERY})
        old = sensitivity['families']['historical_rule_recovery']
        self.assertEqual(set(old['heldout_candidates']), set(a.OLD)-{a.ANCHOR})
        self.assertEqual(set(old['training_candidates']), {a.ANCHOR, *a.DISCOVERY})
        self.assertEqual(old['winning_metric'], 'R_combined')
        self.assertAlmostEqual(old['winning_metric_heldout']['delta_mae'], 12.)
        self.assertAlmostEqual(old['primary_selected_metric_heldout']['delta_mae'], 0.)
        self.assertEqual(len(old['heldout_family_forecasts']), 4)
        for family, row in sensitivity['families'].items():
            self.assertIn(a.ANCHOR, row['training_candidates'])
            self.assertNotIn(a.ANCHOR, row['heldout_candidates'])
            self.assertFalse(set(row['training_candidates']) & set(row['heldout_candidates']))
            self.assertFalse(set(a.CONFIRMATION) & (set(row['training_candidates']) | set(row['heldout_candidates'])))
        self.assertEqual(len(frozen['leave_one_calibration_candidate_out_winner']), 7)

    def test_smoke_cannot_select(self):
        cv, discovery = fake_cv()
        cv['execution_only'] = True
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            a.selection_payload(cv, discovery, {})

    def test_freeze_is_idempotent_and_refuses_changed_inputs(self):
        cv, discovery = fake_cv()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'cv_scores.json', cv)
            write_json(root/'discovery_scores.json', discovery)
            first = a.select(root, root/'discovery_scores.json')
            digest = sha256(root/'cv_selection_frozen.json')
            self.assertEqual(first, a.select(root, root/'discovery_scores.json'))
            self.assertEqual(digest, sha256(root/'cv_selection_frozen.json'))
            discovery['no_month'] += 1
            write_json(root/'discovery_scores.json', discovery)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                a.select(root, root/'discovery_scores.json')

    def test_confirmation_uses_frozen_forecasts_without_reselection(self):
        cv, discovery = fake_cv()
        frozen = a.selection_payload(cv, discovery, {'cv_scores_sha256': 'a', 'discovery_scores_sha256': 'b'})
        scores = {n: frozen['confirmation_forecasts']['R_combined'][n]['anchored_lb'] for n in a.CONFIRMATION}
        result = a.confirmation_payload(frozen, scores)
        self.assertEqual(result['selected_metric'], 'R_A')
        self.assertAlmostEqual(result['metrics']['R_combined']['delta_mae'], 0.)
        self.assertGreater(result['selected_minus_baseline_delta_mae'], 0.)
        self.assertFalse(result['improves_both_confirmation_errors'])

    def test_artifact_target_alignment_and_sha_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = []
            for seed in a.SEEDS:
                path = root/f'{seed}.parquet'
                frame = make_prediction(path, seed=seed)
                entries.append(dict(path=path.name, fold='A1', month='2025-07', seed=seed,
                    rows=len(frame), artifacts={path.name: sha256(path)}))
            output = a.summarize_entries(root, entries, {}, {})
            self.assertEqual(output['mse'], 4.)
            frame = pd.read_parquet(root/'242.parquet')
            frame.loc[0, 'y'] += 1
            frame.to_parquet(root/'242.parquet', index=False)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                a.summarize_entries(root, entries, {}, {})
            entries[-1]['artifacts']['242.parquet'] = sha256(root/'242.parquet')
            with self.assertRaisesRegex(ValueError, 'different movement IDs/targets'):
                a.summarize_entries(root, entries, {}, {})

    def test_complete_roster_is_sixty_predictions_per_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = {}
            specs = [('v3', f, m, 'v3_'+f) for f, m in a.FOLD_MONTHS.items()]
            specs += [(s, s+'_'+m, m, s if s == '10m' else s+'_'+m)
                      for s in ('10m', '11m') for m in a.MONTHS]
            for i, recipe in enumerate(a.RECIPES):
                for scheme, fold, month, job in specs:
                    for seed in a.SEEDS:
                        filename = f'{recipe}_{scheme}_{fold}_{seed}.parquet'
                        frame = make_prediction(root/filename, fold, month, seed, 2.+i)
                        entries[filename] = dict(path=filename, recipe=recipe, scheme=scheme,
                            fold=fold, month=month, seed=seed, rows=len(frame), job=job,
                            artifacts={filename: sha256(root/filename)})
            self.assertEqual(len(entries), 6*60)
            write_json(root/'progress.json', {'models': {}, 'predictions': entries})
            write_json(root/'result.json', {'status': 'complete', 'config':
                dict(smoke=False, sample=1., rounds=800, seeds=list(a.SEEDS))})
            result = a.prepare(root, bootstrap_reps=10)
            self.assertFalse(result['execution_only'])
            self.assertEqual(set(result['new']), set(a.RECIPES))
            self.assertEqual(len(result['prediction_sha256']), 360)
            for i, recipe in enumerate(a.RECIPES):
                for metric in a.METRICS:
                    self.assertAlmostEqual(result['new'][recipe]['standard'][metric], i+2.)
            # Eleven-month holdouts cannot silently reuse the ten-month model.
            for entry in entries.values():
                if entry['scheme'] == '11m':
                    entry['job'] = '11m_shared_wrong'
            write_json(root/'progress.json', {'models': {}, 'predictions': entries})
            with self.assertRaisesRegex(ValueError, 'two distinct'):
                a.prepare(root, bootstrap_reps=10)


if __name__ == '__main__':
    unittest.main()
