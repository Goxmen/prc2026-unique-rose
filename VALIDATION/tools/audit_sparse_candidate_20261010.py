"""Independent all-row arithmetic and serialization audit of sparse combination.

Run only after the chosen recipe has passed CV, all three full-year models have
finished, and both candidates have been built. The inherited correction stack
is covered by the builder's exact parent replay; this auditor does not claim
an independent implementation of those rules. No fitting or upload exists.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import sparse_callsign_submit as candidate

parent = candidate.parent


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = ROOT / record['path']
    if set(record) not in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}):
        raise ValueError('Unexpected artifact record fields: ' + record['path'])
    expected = {name: record[name] for name in ['path', 'sha256']}
    if (parent.artifact(path) != expected
            or ('bytes' in record and path.stat().st_size != record['bytes'])):
        raise ValueError('Artifact changed: ' + record['path'])
    return path


def ordered(path, ids):
    frame = pd.read_parquet(path)
    if not frame.mvt_id.is_unique or not np.array_equal(frame.mvt_id, ids):
        raise ValueError('Prediction identity/order differs: ' + str(path))
    return frame


def audit():
    design_path = candidate.OUT / 'design.json'
    design = read(design_path)
    if design['design'] != candidate.DESIGN:
        raise ValueError('Chosen candidate recipe changed')
    verify(design['source'])
    for name, digest in design['helpers'].items():
        if parent.sha256(ROOT / name) != digest:
            raise ValueError('Frozen helper changed: ' + name)
    for key in ['sparse_identity', 'chosen_recipe', 'combined_cv',
                'combined_cv_predictions', 'parent_candidate_audit']:
        verify(design[key])
    base_audit = read(verify(design['parent_candidate_audit']))
    if not base_audit['passed']:
        raise ValueError('Parent candidate audit must pass first')
    sparse = candidate.sparse
    identity = read(verify(design['sparse_identity']))
    pooled_path = sparse.OUT / 'pooled.json'
    pooled = read(pooled_path)
    if (pooled['identity'] != design['sparse_identity'] or pooled['members'] != sparse.SEEDS
            or pooled['rule'] != 'equal mean of three float32 saved member predictions'):
        raise ValueError('Sparse pooling contract changed')
    members = [sparse.completed_member(k, identity) for k in range(3)]
    if any(member is None for member in members):
        raise ValueError('Three completed sparse members required')
    if pooled['models'] != [parent.artifact(sparse.member_paths(k)[1]) for k in range(3)]:
        raise ValueError('Sparse pool member provenance differs')
    result = dict(source=parent.artifact(Path(__file__)),
                  design=parent.artifact(design_path), pooled=parent.artifact(pooled_path), views={})
    for view in ['old', 'final']:
        manifest_path = candidate.OUT / 'candidates' / view / 'manifest.json'
        manifest = read(manifest_path)
        if (manifest['view'] != view or manifest['design'] != candidate.DESIGN
                or manifest['design_record'] != parent.artifact(design_path)
                or manifest['parent_audit'] != design['parent_candidate_audit']
                or not manifest['exact_weight_zero_raw_and_serialized']
                or manifest['cv_evidence'] != design['combined_cv']):
            raise ValueError('Candidate view/provenance or parent replay differs')
        verify(manifest['source'])
        verify(manifest['view_query'])
        base_checked = base_audit['views'][view]
        if manifest['parent_candidate'] != base_checked['manifest']:
            raise ValueError('Audited parent candidate changed')
        base_manifest = read(verify(manifest['parent_candidate']))
        if (base_manifest['view'] != view or base_manifest['output'] != base_checked['output']
                or base_manifest['template'] != manifest['template']):
            raise ValueError('Parent namespace/template changed')
        parts = pd.read_parquet(verify(manifest['parts']))
        ids = parts.mvt_id.to_numpy()
        if (len(ids) != parent.features.EXPECTED_ROWS[view] or len(ids) != manifest['rows']
                or not parts.mvt_id.is_unique):
            raise ValueError('Candidate row population differs')
        base = ordered(verify(base_checked['parts']), ids)
        frames = [ordered(verify(member['predictions'][view]), ids) for member in members]
        pool_frame = ordered(verify(pooled['outputs'][view]), ids)
        if (manifest['sparse_pool']['record'] != parent.artifact(pooled_path)
                or manifest['sparse_pool']['prediction'] != pooled['outputs'][view]):
            raise ValueError('Candidate sparse pool provenance differs')
        mean = np.mean([frame.base.to_numpy(float) for frame in frames], axis=0)
        np.testing.assert_array_equal(mean, pool_frame.base)
        np.testing.assert_array_equal(mean, parts.sparse_nn)
        for frame in frames:
            np.testing.assert_array_equal(frame.sequence_count, parts.sequence_count)
        np.testing.assert_array_equal(pool_frame.sequence_count, parts.sequence_count)
        np.testing.assert_array_equal(base.nn_base, parts.calls_nn)
        for column in ['gbm_base', 'p', 'main_lgb_original', 'main_lgb_callsign', 'month']:
            np.testing.assert_array_equal(base[column], parts[column])
        neural = .5 * (base.nn_base.to_numpy(float) + mean)
        np.testing.assert_array_equal(neural, parts.nn_base)
        ordinary = .71 * neural + (1 - .71) * base.gbm_base.to_numpy(float)
        np.testing.assert_array_equal(ordinary, parts.ordinary)
        output = pq.read_table(verify(manifest['output']))
        template = pq.read_table(verify(manifest['template']))
        if output.schema != template.schema or output.column_names != template.column_names:
            raise ValueError('Submission schema differs')
        for column in template.column_names:
            if column != 'TAXITIME_SEC_mvt' and not output[column].equals(template[column]):
                raise ValueError('Protected template column differs: ' + column)
        template_ids = template['MVT_ID_mvt'].to_numpy()
        expected = np.rint(pd.Series(parts.pred.to_numpy(), index=ids).loc[template_ids].to_numpy())
        np.testing.assert_array_equal(output['TAXITIME_SEC_mvt'].to_numpy(), expected)
        if not np.isfinite(expected).all() or (expected < 0).any():
            raise ValueError('Invalid serialized prediction')
        previous = pq.read_table(verify(base_checked['output']))['TAXITIME_SEC_mvt'].to_numpy()
        delta = expected - previous
        result['views'][view] = dict(rows=len(ids), manifest=parent.artifact(manifest_path),
            output=manifest['output'], parts=manifest['parts'],
            all_three_member_means_exact=True, all_callsign_parent_predictions_exact=True,
            all_half_neural_blends_exact=True, all_ordinary_blends_exact=True,
            unchanged_tree_and_rome_parts_exact=True, all_serialized_predictions_exact=True,
            protected_template_columns_exact=True, changed_predictions=int(np.count_nonzero(delta)),
            mean_change_sec=float(delta.mean()), rms_change_sec=float(np.sqrt(np.mean(delta**2))),
            maximum_absolute_change_sec=float(np.abs(delta).max()))
    result['passed'] = True
    result['scope'] = ('Independent component arithmetic and serialization on every row; '
                       'inherited corrections are covered by the builder\'s exact parent replay. '
                       'No hidden labels, fitting, or upload.')
    parent.write_json(candidate.OUT / 'candidate_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        audit()
