"""Independently audit a scored-basis candidate with reference-form SQL.

The MSE identity is checked using differences from v64, while construction
uses the retained matrix product. Output clipping/rounding enters the bound
through its measured RMS distance, without assuming hidden-label values.
"""
import argparse
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

OUT = ROOT / 'VALIDATION/runs/v3_20261011_scored_basis_candidate'


def read(path):
    return json.loads(path.read_text())


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def reference_expression(versions, weights):
    assert len(versions) == len(weights) and 64 in versions
    assert all(isinstance(version, int) and version > 0 for version in versions)
    assert np.isfinite(weights).all()
    return 'r.v64' + ''.join(f'+({float(weight)!r})*(r.v{version}-r.v64)'
                            for version, weight in zip(versions, weights) if version != 64)


def geometry_from_sql(con, versions, weights, scores):
    raw = reference_expression(versions, weights)
    expressions = ','.join(f'avg(pow(r.v{version}-r.v64,2))' for version in versions)
    diagonal = np.array(con.execute(f'select {expressions} from r').fetchone())
    count, unique, residual_second, distance, max_rounding_error = con.execute(f'''
        select count(*),count(distinct r.mvt_id),avg(pow(({raw})-r.v64,2)),
        sqrt(avg(pow(c.pred-({raw}),2))),max(abs(c.pred-greatest(({raw}),0)))
        from r join c using(mvt_id)''').fetchone()
    assert count == unique == con.execute('select count(*) from r').fetchone()[0]
    assert count == con.execute('select count(*) from c').fetchone()[0]
    assert max_rounding_error <= .5000001, 'Final output is not a nearest nonnegative integer'
    reference = versions.index(64)
    mse0 = scores[reference]**2
    mse = float(mse0 + weights @ (scores**2-mse0-diagonal) + residual_second)
    assert mse >= 0 and np.isfinite([mse, distance]).all()
    effective = weights.copy()
    effective[reference] = 1-float(np.sum(np.delete(weights, reference)))
    radius = float(np.abs(effective) @ (np.abs(scores)*1e-4+2.5e-9))
    lower, upper = np.sqrt(max(0, mse-radius)), np.sqrt(mse+radius)
    return dict(rows=count, projected_rmse=float(np.sqrt(mse)), serialization_rms=float(distance),
                conservative=[max(0., float(lower-distance)), float(upper+distance)],
                maximum_rounding_distance=float(max_rounding_error))


def audit(version):
    destination = OUT / f'submission_v{version}'
    assert version > 77 and not (destination / 'audit.json').exists()
    path = destination / 'manifest.json'
    manifest = read(path)
    assert manifest['state'] == 'prepared' and manifest['version'] == version
    for source, digest in manifest['sources'].items(): assert sha256(ROOT / source) == digest, source
    prepared = read(ROOT / manifest['prepared']['path'])
    assert art(ROOT / manifest['prepared']['path']) == manifest['prepared']
    result = read(ROOT / manifest['geometry']['path'])
    assert art(ROOT / manifest['geometry']['path']) == manifest['geometry']
    versions = prepared['versions']
    assert len(versions) == len(set(versions)) == 51
    weights = np.array([manifest['weights'][str(v)] for v in versions])
    assert np.isfinite(weights).all() and abs(weights.sum()-1) < 1e-8
    assert (weights >= -3.-1e-8).all() and (weights <= 4.+1e-8).all()
    assert manifest['weights'] == result['solution']['weights']
    scores = np.array([manifest['basis_scores'][f'unique-rose_v{v}.parquet'] for v in versions])
    snapshots = read(destination / 'official_snapshots.json')
    for v, score in zip(versions, scores):
        official = snapshots['fresh_public_basis'][f'unique-rose_v{v}.parquet']
        assert official['usedPairs'] == 344841 and abs(official['score']-score) < 1e-8
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    submitted = pq.read_table(ROOT / manifest['file'])
    assert submitted.schema == template.schema and submitted.num_rows == 344841
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt': assert submitted[name].equals(template[name])
    assert sha256(ROOT / manifest['file']) == manifest['sha256']
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(ROOT / prepared['ranking']['path'])).create_view('r')
        con.read_parquet(str(ROOT / manifest['file'])).create_view('submitted')
        con.execute('create view c as select MVT_ID_mvt mvt_id,TAXITIME_SEC_mvt pred from submitted')
        bad = con.execute('select count(*) from c where pred is null or not isfinite(pred) or pred<0 or pred!=round_even(pred,0)').fetchone()[0]
        assert bad == 0
        projected = geometry_from_sql(con, versions, weights, scores)
        assert projected['rows'] == 344841
        np.testing.assert_allclose(projected['projected_rmse'], manifest['projected_rmse'], rtol=0, atol=1e-7)
        np.testing.assert_allclose(projected['conservative'], manifest['conservative_rmse_interval'], rtol=0, atol=1e-6)
        con.read_parquet(str(ROOT / prepared['cv']['path'])).create_view('h')
        con.read_parquet(str(destination / 'cv.parquet')).create_view('v')
        truth_path = ROOT / 'VALIDATION/runs/v3_20261011_daily_delay/rows2025.parquet'
        con.read_parquet(str(truth_path)).create_view('t')
        expression = reference_expression(versions, weights).replace('r.', 'h.')
        count, unique, errors, maximum = con.execute(f'''select count(*),count(distinct v.mvt_id),
            sum(case when v.y is distinct from t.y or v.month is distinct from t.month
                or v.day is distinct from t.day or v.y is distinct from h.y
                or v.month is distinct from h.month or v.day is distinct from h.day
                or v.pred is null or not isfinite(v.pred) or v.pred<0 or v.pred!=round_even(v.pred,0)
                then 1 else 0 end),max(abs(v.pred-greatest(({expression}),0)))
            from v join h using(mvt_id) join t using(mvt_id)''').fetchone()
        assert count == unique == 344419 and errors == 0 and maximum <= .5000001
        months = dict(con.execute('select month,sqrt(avg(pow(pred-y,2))) from v group by month').fetchall())
        assert set(months) == {'2025-01', '2025-07'}
        for month, value in months.items():
            np.testing.assert_allclose(value, manifest['cv_months'][month], rtol=0, atol=1e-8)
        cv = float(np.mean(list(months.values())))
        np.testing.assert_allclose(cv, manifest['jan_jul_cv'], rtol=0, atol=1e-8)
    best = snapshots['current_best']['score']
    upper = projected['conservative'][1]
    assert best > 215. and upper < best and (upper <= 215. or best-upper >= .25)
    report = dict(status='pass', source=art(Path(__file__)), file=art(ROOT / manifest['file']),
        independent_reference_form_sql_geometry=projected, all_ranking_rows=344841,
        all_cv_rows=count, independent_raw_truth_calendar_weights_and_cv=True, cv=cv,
        actual_basis_scores_checked=True, clipping_and_rounding_bound_checked=True,
        template_schema_and_unchanged_columns_exact=True,
        scope='Separate root-authored arithmetic/SQL audit; leaderboard-selected weights, not independent validation')
    write_json(destination / 'audit.json', report)
    manifest['audit']['independent_final_audit_pending'] = False
    manifest['audit']['independent_review'] = report
    for source in [Path(__file__), truth_path, destination / 'audit.json']:
        manifest['sources'][str(source.relative_to(ROOT))] = sha256(source)
    write_json(path, manifest)
    print(dict(status='pass', version=version, cv=cv, projected=projected))


def check():
    # Known synthetic labels independently verify signed coefficients and a
    # genuinely active nonnegative clipping bound; no competition score is invented.
    rng = np.random.default_rng(57211)
    truth = rng.uniform(0, 100, 3000)
    values = np.rint(np.maximum(truth[:, None]+rng.normal(0, 90, (3000, 5)), 0))
    versions = [75, 64, 77, 70, 76]
    weights = np.array([1.5, -.8, .4, -.7, .6])
    scores = np.round(np.sqrt(np.mean((values-truth[:, None])**2, axis=0)), 4)
    raw = values @ weights
    frame = pd.DataFrame(values, columns=[f'v{v}' for v in versions])
    frame.insert(0, 'mvt_id', np.arange(len(frame)))
    candidate = pd.DataFrame(dict(mvt_id=frame.mvt_id, pred=np.rint(np.maximum(raw, 0))))
    assert (raw < 0).sum() > 100
    with duckdb.connect(config={'threads': 1}) as con:
        con.register('r', frame); con.register('c', candidate)
        result = geometry_from_sql(con, versions, weights, scores)
        actual = float(np.sqrt(np.mean((candidate.pred-truth)**2)))
        assert result['conservative'][0] <= actual <= result['conservative'][1]
        unrounded = float(np.sqrt(np.mean((raw-truth)**2)))
        assert abs(unrounded-result['projected_rmse']) < .001
        # A corrupted output beyond the rounding cell must be rejected.
        damaged = candidate.copy(); damaged.loc[0, 'pred'] += 3
        con.unregister('c'); con.register('c', damaged)
        try: geometry_from_sql(con, versions, weights, scores)
        except AssertionError: pass
        else: raise AssertionError('Corrupted output accepted')
    return dict(synthetic_hidden_target_oracle=True, signed_coefficients_and_reordered_versions=True,
                score_rounding_and_active_clipping_bound=True, corrupted_output_rejected=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'audit'])
    parser.add_argument('--version', type=int)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'check': print(check())
        else: audit(args.version)
