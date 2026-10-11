"""Assemble the audited main-LightGBM change with the already prepared v75.

No refitting or upload. First replay v70 and v75 exactly, then replace only
the main tree using the fixed validated ordinary/Rome interaction.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import public_delay_lgb_submit as production
import rome_public_delay_submit as rome
from records import write_json, sha256
from submit import fill_template

ROOT = production.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_main_candidate'
artifact, verify, read = production.artifact, production.verify, production.read


def constrained(ids, raw, parent, candidate, control, matches):
    pred = rome.r20.constraints.matched_bound(raw, parent+candidate-control)
    return rome.r20.constraints.exclusion.apply(ids, np.maximum(pred, 0), matches)[0]


def build(version):
    dest = OUT / f'submission_v{version}'
    output = ROOT / f'submissions/unique-rose_v{version}.parquet'
    assert not dest.exists() and not output.exists(), 'Preserve existing candidate'
    production.guard()
    audit = read(production.cv.OUT / 'audit.json')
    assert audit['combined']['paired']['transferred_vs_v75']['decision'] == 'improvement'
    model = read(production.OUT / 'model.json')
    rank = read(production.OUT / 'predictions/old.json')
    assert rank['identity'] == artifact(production.OUT / 'identity.json')
    assert model['identity'] == rank['identity'] and rank['model'] == model['model']
    assert rank['rows'] == 344841 and rank['all_ranking_reload_exact']
    verify(model['model'])
    replacement = pd.read_parquet(verify(rank['prediction']))
    p75 = rome.OUT / 'submission_v75'
    manifest75 = read(p75 / 'manifest.json')
    assert manifest75['version'] == 75
    for path, digest in manifest75['sources'].items():
        assert sha256(ROOT / path) == digest, path
    assert sha256(ROOT / manifest75['file']) == manifest75['sha256']
    prior75 = pd.read_parquet(p75 / 'predictions.parquet')
    manifest70 = read(rome.physical.OUT / 'submission_v70/manifest.json')
    reference = pd.read_parquet(verify(manifest70['parts']))
    data = rome.physical.parent.ranking_data('old')
    ids = data['rank_ids']
    for f in [replacement, prior75, reference]:
        np.testing.assert_array_equal(f.mvt_id, ids)
    baseline, masks = rome.physical.parent.final_frame(data, reference.nn_base.to_numpy(),
        reference.gbm_base.to_numpy(), reference.p.to_numpy())
    np.testing.assert_array_equal(baseline.pred, reference.pred)
    np.testing.assert_array_equal(baseline.pred, prior75.control)
    assert masks == manifest70['correction_masks']
    components, tree_sources = production.old.parent_components('old', ids)
    old_record = read(production.old.OUT / 'predictions/old.json')
    old_main = pd.read_parquet(verify(old_record['prediction']))
    _, old_mean = production.old.replace_main(components, ids, old_main)
    _, new_mean = production.old.replace_main(components, ids, replacement)
    np.testing.assert_array_equal(old_mean, reference.gbm_base)
    intermediate, _ = rome.physical.parent.final_frame(data, reference.nn_base.to_numpy(),
        old_mean, prior75.p.to_numpy())
    np.testing.assert_array_equal(intermediate.pred, prior75.candidate)
    candidate, _ = rome.physical.parent.final_frame(data, reference.nn_base.to_numpy(),
        new_mean, prior75.p.to_numpy())
    parent = pd.read_parquet(ROOT / 'submissions/unique-rose_v74.parquet')
    np.testing.assert_array_equal(parent.MVT_ID_mvt, ids)
    raw = pd.read_parquet(rome.features.base.OUT / 'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(rome.r20.constraints.exclusion.OUT / 'ranking_matches.parquet')
    control75 = constrained(ids, raw, parent.TAXITIME_SEC_mvt.to_numpy(float),
        intermediate.pred.to_numpy(), baseline.pred.to_numpy(), matches)
    np.testing.assert_array_equal(control75, prior75.pred)
    actual75 = pq.read_table(ROOT / manifest75['file'])
    np.testing.assert_array_equal(np.rint(control75).astype('int32'), actual75['TAXITIME_SEC_mvt'].to_numpy())
    pred = constrained(ids, raw, parent.TAXITIME_SEC_mvt.to_numpy(float),
        candidate.pred.to_numpy(), baseline.pred.to_numpy(), matches)
    template_path = ROOT / 'data/submitting.parquet'
    template = pq.read_table(template_path)
    result = fill_template(template, ids, pred)
    assert result.schema == template.schema and result['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    serialized = np.rint(np.maximum(pred, 0)).astype('int32')
    np.testing.assert_array_equal(result['TAXITIME_SEC_mvt'].to_numpy(), serialized)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    cv = pd.read_parquet(verify(audit['predictions']))
    scores = {m: float(np.sqrt(np.mean((g.transferred-g.y)**2))) for m, g in cv.groupby('month')}
    cv_score = float(np.mean(list(scores.values())))
    assert abs(cv_score-audit['combined']['cv']['transferred']) < 1e-10
    dest.mkdir(parents=True)
    # Upload/result bookkeeping mutates the live parent manifest. Bind a
    # reviewed snapshot and its immutable inputs so v75's later upload does
    # not invalidate this already-built prediction.
    write_json(dest / 'parent_v75_snapshot.json', manifest75)
    cv.to_parquet(dest / 'cv.parquet', index=False)
    write_json(dest / 'cv.json', dict(cv=cv_score, months=scores,
        paired=audit['combined']['paired']['transferred_vs_v75'], source=artifact(production.cv.OUT / 'audit.json'),
        caveat='Reused Jan/Jul development; actual combined interaction; prior main controls used laptop'))
    parts = pd.DataFrame(dict(mvt_id=ids, parent74=parent.TAXITIME_SEC_mvt.to_numpy(),
        parent75=actual75['TAXITIME_SEC_mvt'].to_numpy(), control=baseline.pred.to_numpy(),
        intermediate=intermediate.pred.to_numpy(), candidate=candidate.pred.to_numpy(),
        tree_old=old_mean, tree_new=new_mean, p=prior75.p.to_numpy(), pred=pred, submitted=serialized))
    parts.to_parquet(dest / 'predictions.parquet', index=False)
    pq.write_table(result, output)
    assert pq.read_table(output).equals(result)
    paths = [Path(__file__), Path(production.__file__), Path(rome.__file__), Path(production.old.__file__),
        production.OUT / 'identity.json', production.OUT / 'model.json', verify(model['model']),
        production.OUT / 'predictions/old.json', verify(rank['prediction']),
        production.cv.OUT / 'audit.json', verify(audit['predictions']),
        production.old.OUT / 'predictions/old.json', verify(old_record['prediction']),
        dest / 'parent_v75_snapshot.json', p75 / 'audit.json', p75 / 'predictions.parquet', ROOT / manifest75['file'],
        rome.physical.OUT / 'submission_v70/manifest.json', verify(manifest70['parts']),
        ROOT / 'submissions/unique-rose_v74.parquet', rome.features.base.OUT / 'ranking_inputs.parquet',
        rome.r20.constraints.exclusion.OUT / 'ranking_matches.parquet', template_path,
        dest / 'cv.json', dest / 'cv.parquet', dest / 'predictions.parquet']
    paths += [verify(r) for r in tree_sources]
    inherited_sources = dict(manifest75['sources'])
    inherited_sources.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    write_json(dest / 'manifest.json', dict(version=version, state='prepared', parent='v75',
        change='12 public daily inputs in the main LightGBM', file=str(output.relative_to(ROOT)),
        sha256=sha256(output), rows=len(ids), jan_jul_cv=cv_score, cv_record=artifact(dest / 'cv.json'),
        sources=inherited_sources,
        changed_rows=int((serialized != actual75['TAXITIME_SEC_mvt'].to_numpy()).sum()),
        authorization='User authorized autonomous uploads', audit=dict(v70_replay_exact=True, v75_replay_exact=True,
            original_tree_mean_exact=True, production_model_reload_all_rows=True,
            schema_and_ids_exact=True, readback_exact=True, independent_final_audit_pending=True)))
    print(dict(version=version, cv=cv_score, rows=len(ids), state='prepared; independent final audit pending'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    with threadpool_limits(2):
        build(args.version)
