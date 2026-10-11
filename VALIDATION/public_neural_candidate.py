"""Build the audited public neural pool, optionally with the public main tree.

Use direct joint CV and retained production predictions. Reproduce the old
physical pool, v70 and v75 before changing anything. No fits or uploads.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import nn_public_submit as production
import public_main_candidate as main_candidate
import public_stack_cv as cv
from records import write_json, sha256
from submit import fill_template

ROOT = production.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_neural_candidate'
rome = main_candidate.rome
art, verify, read = production.art, production.verify, production.read


def selected_cv(with_main):
    cv.guard()
    audit = read(cv.OUT / 'audit.json')
    result = read(cv.OUT / 'result.json')
    assert audit['status'] == 'pass' and audit['result'] == art(cv.OUT / 'result.json')
    assert audit['predictions'] == result['predictions'] == art(cv.OUT / 'scored.parquet')
    column = 'combined_transfer' if with_main else 'neural_transfer'
    if with_main:
        for other in ['baseline', 'main', 'neural']:
            key = f'combined_transfer_vs_{other}_transfer'
            assert result['paired'][key]['decision'] == 'improvement', key
            assert all(s[column] < s[other+'_transfer'] for s in result['scores'].values())
    else:
        result_single = read(production.cv.OUT / 'result.json')
        assert result_single['paired']['transferred_vs_v75']['decision'] == 'improvement'
        assert result_single['cv']['transferred'] == result['cv'][column]
    return result, column


def pool_predictions(ids):
    identity = production.guard()
    audit = read(production.OUT / 'audit.json')
    assert audit['status'] == 'pass' and audit['identity'] == art(production.MANIFEST)
    assert [c['member'] for c in audit['checks']] == [0, 1, 2]
    old, new, paths = [], [], [production.MANIFEST, production.OUT / 'audit.json']
    for k in range(3):
        old_record = production.physical.completed(k, identity['parent_packet'])
        new_record = production.completed(k)
        assert old_record is not None and new_record is not None
        assert audit['checks'][k]['record'] == art(production.paths(k)[2])
        for bank, record in [(old, old_record['predictions']['old']), (new, new_record['prediction'])]:
            path = verify(record)
            frame = pd.read_parquet(path)
            np.testing.assert_array_equal(frame.mvt_id, ids)
            assert np.isfinite(frame.base).all() and frame.base.ge(0).all()
            bank.append(frame.base.to_numpy(float))
            paths.append(path)
        for runner in [production.physical, production]:
            paths.extend(runner.paths(k))
        paths.extend([verify(production.trained(k)['preprocessing']),
                      verify(production.physical.trained(k, identity['parent_packet'])['preprocessing'])])
    return np.mean(old, axis=0), np.mean(new, axis=0), paths


def build(version, with_main):
    dest = OUT / f'submission_v{version}'
    output = ROOT / f'submissions/unique-rose_v{version}.parquet'
    assert version > 75 and not dest.exists() and not output.exists(), 'Preserve existing candidates'
    result_cv, column = selected_cv(with_main)
    data = rome.physical.parent.ranking_data('old')
    ids = data['rank_ids']
    old_pool, new_pool, paths = pool_predictions(ids)
    p75 = rome.OUT / 'submission_v75'
    manifest75 = read(p75 / 'manifest.json')
    assert manifest75['version'] == 75
    assert read(p75 / 'audit.json')['status'] == 'pass'
    for path, digest in manifest75['sources'].items():
        assert sha256(ROOT / path) == digest, path
    assert sha256(ROOT / manifest75['file']) == manifest75['sha256']
    prior75 = pd.read_parquet(p75 / 'predictions.parquet')
    m70_path = rome.physical.OUT / 'submission_v70/manifest.json'
    manifest70 = read(m70_path)
    reference = pd.read_parquet(verify(manifest70['parts']))
    m64_path = rome.physical.selected.OUT / 'candidates/old/manifest.json'
    manifest64 = read(m64_path)
    original = pd.read_parquet(verify(manifest64['parts']))
    for frame in [prior75, reference, original]:
        np.testing.assert_array_equal(frame.mvt_id, ids)
    calls = original.calls_nn.to_numpy(float)
    neural_old, neural_new = .5*(calls+old_pool), .5*(calls+new_pool)
    np.testing.assert_array_equal(neural_old, reference.nn_base)
    baseline, masks = rome.physical.parent.final_frame(data, neural_old,
        reference.gbm_base.to_numpy(), reference.p.to_numpy())
    np.testing.assert_array_equal(baseline.pred, reference.pred)
    np.testing.assert_array_equal(baseline.pred, prior75.control)
    assert masks == manifest70['correction_masks']
    intermediate, _ = rome.physical.parent.final_frame(data, neural_old,
        reference.gbm_base.to_numpy(), prior75.p.to_numpy())
    np.testing.assert_array_equal(intermediate.pred, prior75.candidate)
    tree = reference.gbm_base.to_numpy().copy()
    if with_main:
        main = main_candidate.production
        main.guard()
        record = read(main.OUT / 'predictions/old.json')
        assert record['identity'] == art(main.OUT / 'identity.json')
        assert record['all_ranking_reload_exact'] and record['rows'] == len(ids)
        replacement = pd.read_parquet(verify(record['prediction']))
        old_record = read(main.old.OUT / 'predictions/old.json')
        old_main = pd.read_parquet(verify(old_record['prediction']))
        components, sources = main.old.parent_components('old', ids)
        _, old_tree = main.old.replace_main(components, ids, old_main)
        _, tree = main.old.replace_main(components, ids, replacement)
        np.testing.assert_array_equal(old_tree, reference.gbm_base)
        paths += [Path(main.__file__), main.OUT / 'identity.json', main.OUT / 'model.json',
                  main.OUT / 'predictions/old.json', verify(record['model']), verify(record['prediction']),
                  main.old.OUT / 'predictions/old.json', verify(old_record['prediction'])]
        paths += [verify(s) for s in sources]
    candidate, _ = rome.physical.parent.final_frame(data, neural_new, tree, prior75.p.to_numpy())
    parent = pd.read_parquet(ROOT / 'submissions/unique-rose_v74.parquet')
    np.testing.assert_array_equal(parent.MVT_ID_mvt, ids)
    raw_path = rome.features.base.OUT / 'ranking_inputs.parquet'
    match_path = rome.r20.constraints.exclusion.OUT / 'ranking_matches.parquet'
    raw = pd.read_parquet(raw_path).set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(match_path)
    p74 = parent.TAXITIME_SEC_mvt.to_numpy(float)
    control75 = main_candidate.constrained(ids, raw, p74, intermediate.pred.to_numpy(), baseline.pred.to_numpy(), matches)
    np.testing.assert_array_equal(control75, prior75.pred)
    actual75 = pq.read_table(ROOT / manifest75['file'])
    np.testing.assert_array_equal(np.rint(control75).astype('int32'), actual75['TAXITIME_SEC_mvt'].to_numpy())
    pred = main_candidate.constrained(ids, raw, p74, candidate.pred.to_numpy(), baseline.pred.to_numpy(), matches)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    template_path = ROOT / 'data/submitting.parquet'
    template = pq.read_table(template_path)
    result = fill_template(template, ids, pred)
    serialized = np.rint(pred).astype('int32')
    assert result.schema == template.schema and result['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    np.testing.assert_array_equal(result['TAXITIME_SEC_mvt'].to_numpy(), serialized)
    held = pd.read_parquet(verify(result_cv['predictions']))
    held = held[['mvt_id', 'month', 'day', 'y', 'baseline_transfer', column]].rename(columns={column: 'pred'})
    scores = {m: float(np.sqrt(np.mean((g.pred-g.y)**2))) for m, g in held.groupby('month')}
    cv_score = float(np.mean(list(scores.values())))
    assert abs(cv_score-result_cv['cv'][column]) < 1e-10
    dest.mkdir(parents=True)
    write_json(dest / 'parent_v75_snapshot.json', manifest75)
    held.to_parquet(dest / 'cv.parquet', index=False)
    write_json(dest / 'cv.json', dict(cv=cv_score, months=scores, column=column,
        source=art(cv.OUT / 'result.json'), with_main=with_main,
        caveat='Reused Jan/Jul development; new neural production on desktop, old physical production on laptop'))
    pd.DataFrame(dict(mvt_id=ids, parent74=p74, control=baseline.pred.to_numpy(),
        intermediate=intermediate.pred.to_numpy(), candidate=candidate.pred.to_numpy(),
        calls_nn=calls, physical_old=old_pool, physical_new=new_pool, neural_old=neural_old,
        neural_new=neural_new, tree=tree, p=prior75.p.to_numpy(), pred=pred, submitted=serialized)).to_parquet(
            dest / 'predictions.parquet', index=False)
    pq.write_table(result, output)
    assert pq.read_table(output).equals(result)
    paths += [Path(__file__), Path(production.__file__), Path(main_candidate.__file__), Path(cv.__file__),
        Path(rome.__file__), cv.MANIFEST, cv.OUT / 'audit.json', cv.OUT / 'result.json',
        verify(result_cv['predictions']), dest / 'parent_v75_snapshot.json', p75 / 'audit.json',
        p75 / 'predictions.parquet', ROOT / manifest75['file'], m70_path, verify(manifest70['parts']),
        m64_path, verify(manifest64['parts']), ROOT / 'submissions/unique-rose_v74.parquet', raw_path,
        match_path, template_path, dest / 'cv.parquet', dest / 'cv.json', dest / 'predictions.parquet']
    sources = dict(manifest75['sources'])
    sources.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    write_json(dest / 'manifest.json', dict(version=version, state='prepared', parent='v75',
        change='Public daily inputs in physical neural pool'+(' and main LightGBM' if with_main else ''),
        with_main=with_main, file=str(output.relative_to(ROOT)), sha256=sha256(output), rows=len(ids),
        jan_jul_cv=cv_score, sources=sources, cv_record=art(dest / 'cv.json'),
        changed_rows=int((serialized != actual75['TAXITIME_SEC_mvt'].to_numpy()).sum()),
        authorization='User authorized autonomous uploads', audit=dict(v70_replay_exact=True,
            v75_replay_exact=True, original_neural_pool_exact=True, production_neural_reload_all_rows=True,
            schema_and_ids_exact=True, readback_exact=True, independent_final_audit_pending=True)))
    print(dict(version=version, with_main=with_main, cv=cv_score,
        state='prepared; independent final audit pending'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--with-main', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(2):
        build(args.version, args.with_main)
