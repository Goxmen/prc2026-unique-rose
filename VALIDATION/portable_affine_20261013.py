"""Explicit new final-portable scored ensemble; never substitute missing models.

Freeze one audited basis before its single solve. Historical leaderboard scores
select weights; Jan/Jul development scores are reported separately. This base
has no later public corrections and is not the selected v80 recipe.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import final_public_corrections as common
import scored_submission_geometry as geometry
import tuning_seasonal as seasonal
from records import write_json
from submit import fill_template

ROOT = common.ROOT
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261013_portable_affine'
MANIFEST = OUT / 'identity.json'
BASE = RUNS / 'v3_20261012_target205_geometry'
V80 = RUNS / 'v3_20261012_tabm_pool3_other_runway_combo'
V80_PRODUCTION = RUNS / 'v3_20261012_tabm_pool3_other_runway_combo_submit'
art, read, verify = common.art, common.read, common.verify
DESIGN = dict(
    recipe='portable_affine_v1',
    basis='All audited catalog components in retained nonredundant51-vector basis, in retained order',
    exclusions='Missing components excluded explicitly; v66/v68 excluded as in retained51 basis',
    objective='Own official rounded leaderboard scores; fixed sum1 bounds[-3,4] ridge MSE1',
    reference=64, solve_count=1, output='rint(max(float64 matrix product,0))',
    cv='Retained matching Jan/Jul vectors, with original reconstruction and reused-development caveats',
    controls=[70,80], no_fit=True, no_new_calibration=True, no_upload=True,
    limitations='Different from v80; no final-month accuracy or final selection implied')


def freeze(catalog, snapshot):
    assert not MANIFEST.exists(), 'Preserve frozen basis; no post-result basis search'
    catalog, snapshot = Path(catalog).resolve(), Path(snapshot).resolve()
    c, live, prepared = read(catalog), read(snapshot), read(BASE / 'prepared.json')
    assert c['schema'] == 'final_public_corrections_basis_v1'
    audited = read(BASE / 'audit.json')
    assert audited['status'] == 'pass' and audited['prepared'] == art(BASE / 'prepared.json')
    versions = [v for v in prepared['versions'] if str(v) in c['components']]
    assert 64 in versions and 70 in versions and len(versions) >= 27
    controls = read(V80 / 'result.json')
    production_audit = read(V80_PRODUCTION / 'independent_audit.json')
    production_result = read(V80_PRODUCTION / 'result.json')
    production_identity = read(V80_PRODUCTION / 'identity.json')
    assert production_audit['status'] == 'pass'
    assert production_audit['identity'] == production_result['identity'] == art(V80_PRODUCTION/'identity.json')
    assert production_audit['result'] == art(V80_PRODUCTION/'result.json')
    assert production_identity['cv'] == production_result['cv_source'] == art(V80/'result.json')
    assert controls['identity'] == art(V80/'identity.json')
    assert controls['scored'] in production_identity['sources']
    sources = {Path(__file__), ROOT / 'VALIDATION/tools/audit_portable_affine_20261013.py',
               Path(geometry.__file__), Path(seasonal.__file__), Path(common.__file__),
               catalog, snapshot, BASE / 'prepared.json', BASE / 'audit.json',
               verify(prepared['ranking']), verify(prepared['cv']),
               V80 / 'identity.json', V80 / 'result.json', verify(controls['scored']),
               V80_PRODUCTION/'identity.json', V80_PRODUCTION/'result.json', V80_PRODUCTION/'independent_audit.json',
               ROOT / 'VALIDATION/submit.py', ROOT / 'data/ranking.parquet', ROOT / 'data/final_ranking.parquet',
               ROOT / 'data/submitting.parquet', ROOT / 'data/final_submitting.parquet'}
    for month in ['2025-01','2025-07']:
        paths = list((ROOT/'data').glob(f'training_{month}-01_*.parquet'))
        assert len(paths) == 1, (month, paths)
        sources.add(paths[0])
    selected, scores = {}, {}
    history = {r['file']: r for r in live['history']}
    for version in versions:
        record = c['components'][str(version)]
        audit = read(verify(record['audit']))
        assert audit['status'] == 'pass', version
        result = read(verify(record['result'])) if 'result' in record else None
        for view in ['old', 'final']:
            verify(record[view])
        assert common.audit_binds(audit, record['final'], record.get('result'), result), (version, 'final')
        # Some legacy audits bind final bytes but only report old SQL equality.
        # Pin old bytes here; run and the separate auditor independently require
        # every old value to match the original audited leaderboard basis.
        official = history[f'unique-rose_v{version}.parquet']
        assert official['status'] == 'Succeeded' and official['usedPairs'] == 344841
        scores[str(version)] = float(official['score'])
        selected[str(version)] = record
        for key in ['old','final','audit','result']:
            if key in record:
                sources.add(verify(record[key]))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, dict(design=DESIGN, versions=versions, components=selected,
        scores=scores, snapshot_at=live['updatedAt'], source=art(__file__),
        sources=[art(p) for p in sorted(sources)], catalog=art(catalog), snapshot=art(snapshot),
        old_basis=prepared['ranking'], cv_basis=prepared['cv'], control_cv=controls['scored'],
        excluded_from_catalog=sorted(set(map(int,c['components']))-set(versions)),
        unavailable_from_retained_basis=[v for v in prepared['versions'] if v not in versions],
        historical_cv_caveat=prepared['historical_cv_caveat']))
    print(dict(frozen=True, versions=versions, independent_review_required=True), flush=True)


def guard():
    identity = read(MANIFEST)
    assert identity['design'] == DESIGN and identity['source'] == art(__file__)
    for record in identity['sources']:
        verify(record)
    return identity


def run():
    identity = guard()
    assert not (OUT / 'result.json').exists(), 'Preserve completed solve'
    versions = identity['versions']
    ids = common.template('old')
    old = pd.read_parquet(verify(identity['old_basis']), columns=['mvt_id']+[f'v{v}' for v in versions])
    np.testing.assert_array_equal(old.mvt_id, ids)
    for v in versions:
        rec = identity['components'][str(v)]
        np.testing.assert_array_equal(old[f'v{v}'], common.vector(rec['old'], ids, rec['id_column'], rec['value_column']))
    matrix = old[[f'v{v}' for v in versions]].to_numpy(float)
    scores = np.array([identity['scores'][str(v)] for v in versions])
    checkpoint = OUT / 'geometry.json'
    if checkpoint.exists():
        previous = read(checkpoint)
        assert previous['identity'] == art(MANIFEST)
        solved = previous['solution']
        raw = matrix @ np.array([solved['weights'][str(v)] for v in versions])
    else:
        solved, raw = geometry.solve(matrix, scores, versions, -3., 4., 1.)
    weights = np.array([solved['weights'][str(v)] for v in versions])
    squared = float(weights @ scores**2 - weights @ np.mean(matrix**2, axis=0) + np.mean(raw**2))
    np.testing.assert_allclose(np.sqrt(squared), solved['projected_rmse'], atol=1e-8, rtol=0)
    if not checkpoint.exists():
        write_json(checkpoint, dict(identity=art(MANIFEST), solution=solved,
            expanded_mse=squared, no_independent_validation=True))
    else:
        np.testing.assert_allclose(read(checkpoint)['expanded_mse'], squared, atol=1e-8, rtol=0)
    # Save old outputs immediately after the one solve; no optimization follows.
    saved = common.serialized(raw)
    pd.DataFrame(dict(mvt_id=ids, raw=raw, submitted=saved)).to_parquet(OUT/'old.parquet',index=False)
    distance = float(np.sqrt(np.mean((saved.astype(float)-raw)**2)))
    lo, hi = solved['score_rounding_rmse_interval']
    del matrix, old
    held = pd.read_parquet(verify(identity['cv_basis']), columns=['mvt_id','month','day','airport','y']+[f'v{v}' for v in versions])
    control = pd.read_parquet(verify(identity['control_cv'])).set_index('mvt_id').loc[held.mvt_id]
    for name in ['month','day','y']:
        np.testing.assert_array_equal(held[name], control[name])
    c_raw = held[[f'v{v}' for v in versions]].to_numpy(float) @ weights
    cv = held[['mvt_id','month','day','airport','y','v70']].copy()
    cv['v80'] = control['three_tabm_runway'].to_numpy()
    cv['raw'] = c_raw
    cv['submitted'] = common.serialized(c_raw)
    cv.to_parquet(OUT/'cv.parquet',index=False)
    metrics = {m:{k:float(np.sqrt(np.mean((g[k]-g.y)**2))) for k in ['v70','v80','submitted']} for m,g in cv.groupby('month')}
    paired = {k:seasonal.paired_bootstrap(cv.day,cv.month,(cv[k]-cv.y)**2,(cv.submitted-cv.y)**2) for k in ['v70','v80']}
    final_ids = common.template('final')
    final = np.empty((len(final_ids),len(versions)),dtype=float)
    for j,v in enumerate(versions):
        rec = identity['components'][str(v)]
        final[:,j] = common.vector(rec['final'],final_ids,rec['id_column'],rec['value_column'])
    f_raw = final @ weights
    pd.DataFrame(dict(mvt_id=final_ids,raw=f_raw,submitted=common.serialized(f_raw))).to_parquet(OUT/'final.parquet',index=False)
    template = pq.read_table(ROOT/'data/final_submitting.parquet')
    table = fill_template(template, final_ids, f_raw)
    assert table.schema == template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    pq.write_table(table,OUT/'final_candidate.parquet')
    assert pq.read_table(OUT/'final_candidate.parquet').equals(table)
    write_json(OUT/'result.json',dict(identity=art(MANIFEST),geometry=art(OUT/'geometry.json'),
        predictions={view:art(OUT/f'{view}.parquet') for view in ['old','cv','final']},
        final_candidate=art(OUT/'final_candidate.parquet'),versions=versions,
        rows=dict(old=len(ids),cv=len(cv),final=len(final_ids)),cv={k:float(np.mean([d[k] for d in metrics.values()])) for k in ['v70','v80','submitted']},
        months=metrics,paired=paired,projected_old_rmse=solved['projected_rmse'],
        serialization_rms=distance,conservative_old_rmse_interval=[max(0,lo-distance),hi+distance],
        independent_audit_pending=True,selected_for_final=False,no_upload=True))
    print(dict(base_complete=True,projected_old_rmse=solved['projected_rmse'],cv=read(OUT/'result.json')['cv']),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','run'])
    p.add_argument('--catalog',type=Path);p.add_argument('--snapshot',type=Path)
    a=p.parse_args()
    with threadpool_limits(1):
        if a.stage=='freeze':
            assert a.catalog and a.snapshot
            freeze(a.catalog,a.snapshot)
        else:run()
