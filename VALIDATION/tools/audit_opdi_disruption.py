"""Independently check score arithmetic, fit membership, features and correction stacking.

Run only after training finishes: this reads full saved prediction files.
"""
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import opdi_disruption_run as run
import recording_artifacts
from common import connect, TRAIN_GLOB
from records import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def audit_variant(variant, con, raw, opdi_table):
    directory = run.DEFAULT / variant
    record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
    if record['status'] != 'complete':
        raise ValueError('Training must finish before the full-data audit')
    output = ROOT / 'VALIDATION/reports/opdi_disruption_experiment' / variant
    scored = read(output / 'results.json')
    run.overnight_run.check_artifacts(directory, progress)
    prior = read(run.prior.DEFAULT / 'result.json')
    prior_progress = read(run.prior.DEFAULT / 'progress.json')
    if prior['status'] != 'complete':
        raise ValueError('Prior independently audited feature comparison is incomplete')
    feature_rows, stack_rows, training_rows = 0, 0, 0
    max_stack_error = 0.
    for name, info in dict(record['folds'], JAN_RETRO=record['january_holdout']).items():
        current = pd.read_parquet(directory / info['predictions']['42']).set_index('mvt_id').sort_index()
        previous_info = prior['january_holdout'] if name == 'JAN_RETRO' else prior['folds'][name]
        run.prior.checked_file(run.prior.DEFAULT, prior_progress['predictions'][f'{name}__42'])
        previous = pd.read_parquet(run.prior.DEFAULT / previous_info['predictions']['42'],
                                  columns=['mvt_id', 'y', *run.features.FEATURES]).set_index('mvt_id').sort_index()
        if not current.index.equals(previous.index):
            raise ValueError(f'Changed evaluation population: {name}')
        np.testing.assert_array_equal(current.y, previous.y)
        np.testing.assert_allclose(current[run.features.FEATURES], previous[run.features.FEATURES],
                                   rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(current[run.opdi.OPDI_FEATURES], opdi_table.loc[current.index],
                                   rtol=0, atol=0, equal_nan=True)
        feature_rows += len(current)
        rows = raw.loc[current.index].copy()
        np.testing.assert_array_equal(current.y, rows.TAXITIME_SEC_mvt)
        np.testing.assert_array_equal(pd.to_datetime(current.mvt_time, utc=True), rows.MVT_TIME_UTC_mvt)
        np.testing.assert_array_equal(current.airport, rows.ADEP_mvt)
        for col in recording_artifacts.SIGNALS:
            rows[col] = np.nan  # All historical correction windows use pooled handler weights.
        gap = (rows.MVT_TIME_UTC_mvt - rows.SCHED_TIME_UTC_mvt).dt.total_seconds()
        V = pd.DataFrame({run.rome_hybrid.GAP: gap}, index=rows.index)
        params = record['parameters'][info['model_id']]
        cp = record['correction_parameters'][info['model_id']]
        if cp['day']['handler_stratified']:
            raise ValueError('Handler features would be needed for this correction window')
        for ordinary, final in [('ordinary_control', 'pred_control'), ('ordinary_disruption', 'pred_disruption')]:
            combined = run.signals.combine(current[ordinary].to_numpy(), current.p_shared.to_numpy(), V, rows,
                                           late_mean=params['late_mean'], dayshift_weight=None)
            pred34 = run.recipe.post_process(combined['final'], rows)
            pred37, _ = recording_artifacts.predict(pred34, rows, cp, probability=current.p_shared.to_numpy())
            np.testing.assert_allclose(current[final], pred37['pred_combined'], atol=1e-8, rtol=0)
            max_stack_error = max(max_stack_error, float(np.max(np.abs(current[final] - pred37['pred_combined']))))
        np.testing.assert_array_equal(current.rule_triggered, run.recipe.rule_rows(rows, 'rome_4h'))
        stack_rows += len(current)
        del current, previous, rows, V
        gc.collect()
    jobs = {job['id']: job for job in run.overnight_run.schedule() if job['scheme'] == 'v3'}
    months = raw.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m')
    training_counts = {}
    for job, info in record['training_audits'].items():
        run.prior.checked_file(directory, info)
        audit = pd.read_parquet(directory / info['path']).set_index('mvt_id').sort_index()
        population = raw.loc[months.isin(jobs[job]['train']) & raw.TAXITIME_SEC_mvt.between(1, 200000)].sort_index()
        if not audit.index.equals(population.index):
            raise ValueError(f'Changed training population: {job}')
        drop = run.recipe.rule_rows(population, 'rome_4h') if variant == 'v40' else np.zeros(len(population), bool)
        gap = (population.MVT_TIME_UTC_mvt - population.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
        rome = population.ADEP_mvt.eq('LIRF').to_numpy()
        substituted = rome & (np.abs(gap - population.TAXITIME_SEC_mvt.to_numpy()) <= 60)
        np.testing.assert_array_equal(audit.excluded_by_rule, drop)
        np.testing.assert_array_equal(audit.used_by_ordinary, ~drop & ~substituted)
        np.testing.assert_array_equal(audit.used_by_classifier, ~drop & rome)
        training_rows += len(audit)
        training_counts[job] = dict(rows=len(audit), excluded=int(drop.sum()))
    # SQL averages squared errors across both origins and seeds before weighting;
    # it does not call the experiment's Loss/grouping/scoring helpers.
    score_checks = {}
    adopted = read(ROOT / 'VALIDATION/reports/weighted_cv_leaderboard.json')['weights']
    con.execute('CREATE OR REPLACE TEMP TABLE preds AS SELECT mvt_id, fold, y, pred_control, pred_disruption FROM read_parquet(?)',
                [[str(directory / path) for info in record['folds'].values() for path in info['predictions'].values()]])
    for group, folds in [('R_A', [f'A{i}' for i in range(1, 7)]), ('R_B_long', ['B3', 'B4', 'B7'])]:
        weights = pd.read_parquet(ROOT / adopted[group]['path']).ratio.rename('weight').rename_axis('mvt_id').reset_index()
        con.register('weights', weights)
        q = con.execute('''WITH loss AS (
            SELECT mvt_id, avg(pow(pred_control-y,2)) old_sq, avg(pow(pred_disruption-y,2)) new_sq
            FROM preds WHERE list_contains(?, fold) GROUP BY mvt_id)
            SELECT sqrt(avg(old_sq)), sqrt(avg(new_sq)),
                   sqrt(sum(old_sq*weight)/sum(weight)), sqrt(sum(new_sq*weight)/sum(weight)), count(*)
            FROM loss JOIN weights USING(mvt_id)''', [folds]).fetchone()
        if q[4] != len(weights):
            raise ValueError('SQL weight population mismatch')
        expected = [scored['scores'][arm][mode][group]['score']
                    for mode in ('unweighted', 'weighted') for arm in ('control', 'candidate')]
        np.testing.assert_allclose(q[:4], expected, atol=1e-8, rtol=0)
        score_checks[group] = dict(rows=q[4], scores=list(q[:4]), max_error=float(np.max(np.abs(np.array(q[:4]) - expected))))
    result = dict(status='passed', variant=variant, training_rows_checked=training_rows,
        training_counts=training_counts, feature_rows_checked=feature_rows,
        disruption_feature_values_checked=feature_rows * len(run.features.FEATURES),
        opdi_feature_values_checked=feature_rows * len(run.opdi.OPDI_FEATURES),
        correction_stack_rows_checked=stack_rows, maximum_stack_error=max_stack_error,
        independent_sql_scores=score_checks, source_sha256=sha256(Path(__file__)),
        run_record_sha256=sha256(directory / 'result.json'), scored_result_sha256=sha256(output / 'results.json'),
        original_feature_audit_sha256=sha256(ROOT / 'VALIDATION/reports/disruption_experiment/independent_audit.json'))
    write_json(output / 'independent_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    for variant in run.HISTORICAL:
        if read(run.DEFAULT / variant / 'result.json')['status'] != 'complete':
            raise ValueError('Wait until both experiments finish')
    con = connect()
    con.execute('SET threads=2')
    columns = ['MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'FLIGHT_ID_mvt',
               'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt']
    raw = con.execute(f"SELECT {','.join(columns)} FROM read_parquet('{TRAIN_GLOB}') WHERE PHASE_mvt='DEP'").df()
    if raw.MVT_ID_mvt.duplicated().any():
        raise ValueError('Duplicate raw departure IDs')
    raw = raw.set_index('MVT_ID_mvt')
    for col in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt'):
        raw[col] = pd.to_datetime(raw[col], utc=True)
    table = run.opdi.load_features()
    for variant in run.HISTORICAL:
        audit_variant(variant, con, raw, table)
    con.close()


if __name__ == '__main__':
    main()
