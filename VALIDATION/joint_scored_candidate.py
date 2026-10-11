"""Package a useful, actually scored 52-vector blend without fitting models.

The source diagnosis must contain successful official results for v75--v78.
No version is reserved until all selection/source checks pass. This module
does not upload; its output requires the separate final SQL audit first.
"""
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import joint_models_geometry as diagnosis
from records import sha256, write_json
from submit import fill_template

ROOT = diagnosis.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_joint_scored_candidate'
read, art, verify = diagnosis.read, diagnosis.art, diagnosis.verify
SELECTION = dict(target=215., minimum_conservative_gain=.25,
    rule='conservative upper bound improves current official best by0.25s or reaches215',
    cv='report fixed Jan/Jul replay regardless of change; leaderboard selection is explicit',
    serialization='preserve frozen diagnosis vectors; do not reround floating half-integer ties',
    no_new_fit=True, no_new_weight_search=True, no_score_imputation=True)


def weights_for(versions, solution):
    assert set(solution['weights']) == {str(version) for version in versions}
    weights = np.array([solution['weights'][str(version)] for version in versions], dtype=float)
    assert np.isfinite(weights).all() and abs(weights.sum()-1) < 1e-8
    assert (weights >= -3.-1e-8).all() and (weights <= 4.+1e-8).all()
    assert solution['constraints'] == dict(lower=-3., upper=4., ridge_mse=1.)
    return weights


def useful(upper, best):
    assert np.isfinite([upper, best]).all() and upper >= 0 and best >= 0
    return best > SELECTION['target'] and upper < best and (
        upper <= SELECTION['target'] or best-upper >= SELECTION['minimum_conservative_gain'])


def build(version):
    assert version > 78
    destination = OUT / f'submission_v{version}'
    output = ROOT / f'submissions/unique-rose_v{version}.parquet'
    assert not destination.exists() and not output.exists(), 'Preserve existing candidate'
    # Nothing is written before all official-score, source and selection checks.
    result_path = diagnosis.OUT / 'result.json'
    result = read(result_path)
    prepared_path = diagnosis.OUT / 'prepared.json'
    prepared = read(prepared_path)
    audit = read(diagnosis.OUT / 'audit.json')
    assert result['prepared'] == art(prepared_path) == audit['prepared']
    assert audit['status'] == 'pass' and prepared['design'] == diagnosis.DESIGN
    assert result['design'] == diagnosis.DESIGN and result['independent_mse_identity']
    for path, digest in prepared['sources'].items(): assert sha256(ROOT / path) == digest, path
    versions = prepared['versions']
    assert len(versions) == len(set(versions)) == 52 and {64, 75, 76, 77, 78}.issubset(versions)
    weights = weights_for(versions, result['solution'])
    scores = dict(prepared['known_scores'])
    live_records = {}
    for version_added in diagnosis.ADDITIONS:
        addition = prepared['additions'][str(version_added)]
        snapshot = read(verify(addition['snapshot']))
        current = read(ROOT / addition['live_manifest'])
        assert current['state'] == 'scored'
        assert current['file'] == snapshot['file'] and current['sha256'] == snapshot['sha256']
        assert current['remote_sha256'] == snapshot['sha256']
        official = current['official_result']
        assert official['status'] == 'Succeeded' and official['used_pairs'] == 344841
        assert current['score'] == official['score']
        assert result['official'][str(version_added)] == official
        scores[str(version_added)] = official['score']
        live_records[str(version_added)] = current
    public = diagnosis.transport.live.fetch_leaderboard()
    ours = {row['filename']: row for row in public if row['teamName'] == diagnosis.transport.live.TEAM}
    basis_scores = {f'unique-rose_v{version}.parquet': scores[str(version)] for version in versions}
    for name, score in basis_scores.items():
        assert ours[name]['usedPairs'] == 344841 and abs(ours[name]['score']-score) < 1e-8
    successful = [row for row in ours.values() if row.get('score') is not None and row.get('usedPairs') == 344841]
    current_best = min(successful, key=lambda row: row['score'])
    assert useful(result['conservative_rmse_interval'][1], current_best['score']), 'No useful conservative gain or goal already met'
    ranking = pd.read_parquet(verify(prepared['ranking']))
    held = pd.read_parquet(verify(prepared['cv']))
    diagnostic_rank = pd.read_parquet(verify(result['diagnostic_ranking']))
    diagnostic_cv = pd.read_parquet(verify(result['diagnostic_cv']))
    assert len(ranking) == 344841 and len(held) == 344419
    for frame in [ranking, held, diagnostic_rank, diagnostic_cv]: assert frame.mvt_id.is_unique
    np.testing.assert_array_equal(ranking.mvt_id, diagnostic_rank.mvt_id)
    np.testing.assert_array_equal(held.mvt_id, diagnostic_cv.mvt_id)
    columns = [f'v{version}' for version in versions]
    raw = ranking[columns].to_numpy(float) @ weights
    raw_cv = held[columns].to_numpy(float) @ weights
    np.testing.assert_allclose(raw, diagnostic_rank.raw, atol=1e-8, rtol=0)
    # Reordered floating reductions can land on opposite sides of an exact
    # half-integer. Preserve the already evaluated serialization and verify its
    # rounding cells, instead of silently creating a numerically different file.
    serialized = diagnostic_rank.submitted.to_numpy(float)
    serialized_cv = diagnostic_cv.pred.to_numpy(float)
    assert np.isfinite(serialized).all() and serialized.max() < np.iinfo(np.int32).max
    np.testing.assert_array_equal(serialized, np.rint(np.maximum(diagnostic_rank.raw.to_numpy(), 0)))
    assert np.isfinite(serialized_cv).all() and (serialized_cv >= 0).all()
    np.testing.assert_array_equal(serialized_cv, np.rint(serialized_cv))
    assert np.max(abs(serialized_cv-np.maximum(raw_cv, 0))) <= .5000001
    for name in ['y', 'month', 'day']: np.testing.assert_array_equal(held[name], diagnostic_cv[name])
    for name in columns: assert np.any(serialized != ranking[name].to_numpy()), 'Duplicate basis prediction'
    months = {month: float(np.sqrt(np.mean((frame.pred-frame.y)**2)))
              for month, frame in diagnostic_cv.groupby('month')}
    assert set(months) == {'2025-01', '2025-07'}
    cv = float(np.mean(list(months.values())))
    np.testing.assert_allclose(cv, result['cv'], atol=1e-8, rtol=0)
    template_path = ROOT / 'data/submitting.parquet'
    template = pq.read_table(template_path)
    table = fill_template(template, ranking.mvt_id.to_numpy(), serialized)
    assert table.schema == template.schema
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt': assert table[name].equals(template[name])
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(), serialized)
    paths = [Path(__file__), Path(diagnosis.__file__), result_path, prepared_path,
             diagnosis.OUT / 'audit.json', verify(prepared['ranking']), verify(prepared['cv']),
             verify(result['diagnostic_ranking']), verify(result['diagnostic_cv']), template_path]
    sources = dict(prepared['sources'])
    sources.update({str(path.relative_to(ROOT)): sha256(path) for path in paths})
    destination.mkdir(parents=True)
    pq.write_table(table, output)
    assert pq.read_table(output).equals(table)
    shutil.copyfile(verify(result['diagnostic_cv']), destination / 'cv.parquet')
    write_json(destination / 'official_snapshots.json', dict(additions=live_records,
        fresh_public_basis={name: ours[name] for name in basis_scores}, current_best=current_best))
    for path in [destination / 'cv.parquet', destination / 'official_snapshots.json']:
        sources[str(path.relative_to(ROOT))] = sha256(path)
    manifest = dict(version=version, state='prepared', parent=current_best['filename'],
        change='Reoptimize retained scored basis with actual public-model and daily/tree scores',
        file=str(output.relative_to(ROOT)), sha256=sha256(output), rows=344841,
        jan_jul_cv=cv, cv_months=months, cv_record=art(result_path), geometry=art(result_path),
        prepared=art(prepared_path), selection=SELECTION, sources=sources,
        basis_scores=basis_scores, weights=result['solution']['weights'],
        projected_rmse=result['solution']['projected_rmse'],
        conservative_rmse_interval=result['conservative_rmse_interval'],
        authorization='User authorized autonomous leaderboard uploads',
        limitations=['Weights selected using own leaderboard scores; not independent validation',
                    prepared['historical_cv_caveat'], 'Projection is not an official submitted score'],
        audit=dict(independent_final_audit_pending=True, original_diagnosis_replayed=True,
                   schema_ids_unchanged_columns_exact=True, byte_readback_exact=True))
    write_json(destination / 'manifest.json', manifest)
    print(dict(version=version, state='prepared; separate SQL audit required', cv=cv,
               projected=manifest['projected_rmse'], conservative=manifest['conservative_rmse_interval']))


def check():
    # Guard behavior has no dependence on future scores or a reserved version.
    assert useful(214.99, 215.05) and useful(220., 221.6881)
    assert not useful(221.6, 221.6881) and not useful(215., 214.9)
    assert not useful(222., 221.6881)
    solution = dict(weights={'64': .25, '75': .75}, constraints=dict(lower=-3., upper=4., ridge_mse=1.))
    np.testing.assert_array_equal(weights_for([75, 64], solution), [.75, .25])
    for changed in [dict(solution, weights={'64': 1.}),
                    dict(solution, weights={'64': .5, '75': .6}),
                    dict(solution, weights={'64': 5., '75': -4.})]:
        try: weights_for([64, 75], changed)
        except AssertionError: pass
        else: raise AssertionError('Malformed/misaligned weights accepted')
    return dict(selection_goal_and_gain_guards=True, missing_weight_sum_and_bounds_rejected=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'build'])
    parser.add_argument('--version', type=int)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'check': print(check())
        else: build(args.version)
