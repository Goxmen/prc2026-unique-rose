"""Review the final daily/tree package independently of its assembly helpers.

Reuse completed model/source audits, independently check tree arithmetic,
both serialized constraint boundaries and CV against raw historical truth.
No fit, prediction change, calibration, or upload.
"""
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from records import sha256, write_json

RUNS = ROOT / 'VALIDATION/runs'
DEST = RUNS / 'v3_20261011_daily_refinements_candidate/submission_v77'


def read(path):
    return json.loads(path.read_text())


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def audit():
    assert not (DEST / 'review.json').exists(), 'Preserve completed review'
    manifest = read(DEST / 'manifest.json')
    assert manifest['state'] == 'prepared' and manifest['version'] == 77
    for path, digest in manifest['sources'].items():
        assert sha256(ROOT / path) == digest, path
    package = read(DEST / 'audit.json')
    assert package['status'] == 'pass'
    assert package['file'] == art(ROOT / manifest['file'])
    component = RUNS / 'v3_20261011_daily_main_refinements_submit'
    component_audit = read(component / 'audit.json')
    assert component_audit['status'] == 'pass'
    assert component_audit['result'] == art(component / 'result.json')
    assert component_audit['all_daily_shifts_final_bounds_replayed']
    paths = [Path(__file__), DEST / 'audit.json', component / 'audit.json']

    def frame(run, filename, columns=None):
        path = RUNS / ('v3_20261011_' + run) / filename
        paths.append(path)
        return pd.read_parquet(path, columns=columns)

    submitted = pq.read_table(ROOT / manifest['file'])
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    assert submitted.schema == template.schema and submitted.num_rows == 344841
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt': assert submitted[name].equals(template[name])
    ids = submitted['MVT_ID_mvt'].to_numpy()
    assert pd.Index(ids).is_unique
    values = submitted['TAXITIME_SEC_mvt'].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    parts = frame('daily_main_refinements_submit', 'predictions.parquet')
    main = frame('daily_main_submit', 'predictions/old.parquet', ['mvt_id', 'pred'])
    airport = frame('public_daily_taxi_submit', 'predictions.parquet', ['mvt_id', 'airport_new'])
    cat = frame('catboost_full_inputs_submit', 'predictions/old.parquet', ['mvt_id', 'pred'])
    original = frame('public_neural_candidate', 'submission_v76/predictions.parquet', ['mvt_id', 'control'])
    aggregate = frame('public_refinements_aggregate_submit', 'predictions.parquet', ['mvt_id', 'new74_serialized'])
    for table in [parts, main, airport, cat, original, aggregate]:
        np.testing.assert_array_equal(table.mvt_id, ids)
    tree = np.column_stack([main.pred.to_numpy(float), airport.airport_new.to_numpy(float),
                            cat.pred.to_numpy(float)]).mean(axis=1)
    np.testing.assert_array_equal(tree, parts.candidate_tree)
    raw = frame('daily_delay', 'ranking_inputs.parquet', ['mvt_id', 'has_id', 'nm_gap']).set_index('mvt_id').loc[ids]
    matches = frame('recovered_exclusion_fixed', 'ranking_matches.parquet')
    accepted = matches.loc[matches.accepted]
    assert accepted.mvt_id.is_unique
    positions = {value: index for index, value in enumerate(ids)}
    known = raw.has_id.to_numpy(bool) & raw.nm_gap.notna().to_numpy()
    gap = raw.nm_gap.to_numpy(float)

    def bounds(incoming):
        result = incoming.copy()
        result[known] = np.maximum(0, np.minimum(np.maximum(result[known], gap[known]-3606), gap[known]+3606))
        result = np.maximum(result, 0)
        for row in accepted.itertuples():
            at = positions[row.mvt_id]
            lo, hi = row.recovered_gap-3606, row.recovered_gap+3606
            if lo < result[at] < hi:
                result[at] = min([x for x in [lo, hi] if x >= 0], key=lambda x: abs(x-result[at]))
        return result

    first = bounds(aggregate.new74_serialized.to_numpy(float) + parts.candidate_direct.to_numpy(float)
                   - original.control.to_numpy(float))
    np.testing.assert_array_equal(first, parts.candidate_refined_float)
    rounded = np.rint(first)
    np.testing.assert_array_equal(rounded, parts.candidate_refined)
    shift = np.where(parts.quality_eligible, parts.candidate_shift, 0.)
    np.testing.assert_array_equal(shift, parts.candidate_applied)
    final = bounds(rounded + shift)
    np.testing.assert_array_equal(final, parts.candidate_bounded)
    np.testing.assert_array_equal(np.rint(final).astype('int32'), values)
    assert sha256(ROOT / manifest['file']) == manifest['sha256'] == sha256(component / 'candidate.parquet')
    truth = RUNS / 'v3_20261011_daily_delay/rows2025.parquet'
    paths.append(truth)
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(DEST / 'cv.parquet')).create_view('c')
        con.read_parquet(str(truth)).create_view('t')
        n, unique, bad = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.pred is null or not isfinite(c.pred) or c.pred<0 or c.pred!=round_even(c.pred,0)
            then 1 else 0 end) from c join t using(mvt_id)''').fetchone()
        assert n == unique == 344419 and bad == 0
        scores = dict(con.execute('select month,sqrt(avg(pow(pred-y,2))) from c group by month').fetchall())
        for month, value in scores.items():
            np.testing.assert_allclose(value, manifest['cv_months'][month], atol=1e-8, rtol=0)
        cv = float(np.mean(list(scores.values())))
        np.testing.assert_allclose(cv, manifest['jan_jul_cv'], atol=1e-8, rtol=0)
    value = dict(status='pass', source=art(Path(__file__)), output_sha256=manifest['sha256'],
        ranking_rows=len(ids), cv_rows=n, cv=cv, tree_mean_exact=True,
        both_constraint_and_rounding_boundaries_exact=True, quality_gate_exact=True,
        raw_truth_calendar_and_sql_cv_exact=True, template_and_all_unchanged_columns_exact=True,
        byte_identical_audited_component=True, component_audit=art(component / 'audit.json'),
        scope='Separate root-authored arithmetic/constraint/CV review; completed source/model/daily-shift audits reused',
        reviewer='root; no independent subagent available')
    write_json(DEST / 'review.json', value)
    paths.append(DEST / 'review.json')
    manifest['sources'].update({str(path.relative_to(ROOT)): sha256(path) for path in paths})
    manifest['audit']['independent_review'] = value
    write_json(DEST / 'manifest.json', manifest)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    with threadpool_limits(1): audit()
