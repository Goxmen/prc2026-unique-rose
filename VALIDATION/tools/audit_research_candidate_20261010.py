"""Read-only arithmetic, saved-tree and serialization audit of the fixed candidate.

Run after callsign_lgb_submit builds both views and after the full-source and
fifteen-network reload audits. This does not fit, upload, or load training data.
The inherited correction stack is covered by exact parent replay in the builder;
this audit independently checks its input arithmetic and final serialization.
"""
import gc
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import callsign_lgb_submit as candidate

parent = candidate.parent


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = ROOT / record['path']
    if parent.artifact(path) != record:
        raise ValueError('Artifact changed: ' + record['path'])
    return path


def ordered(path, ids, columns=None):
    frame = pd.read_parquet(path, columns=columns)
    if not frame.mvt_id.is_unique or not np.array_equal(frame.mvt_id, ids):
        raise ValueError('Prediction identity/order differs: ' + str(path))
    return frame


def audit():
    candidate.guard()
    model_record = candidate.completed_model()
    if model_record is None:
        raise ValueError('Main tree is incomplete')
    reload_path = parent.OUT / 'model_reload_audit/result.json'
    reload = read(reload_path)
    if not reload['passed'] or reload['total_predictions'] != 15 * 2 * 512:
        raise ValueError('All fifteen network reload checks must pass first')
    verify(reload['source'])
    verify(reload['preprocessing'])
    for record in reload['samples'].values():
        verify(record)
    checks = {(row['group'], row['k'], row['view']): row for row in reload['checks']}
    if len(checks) != len(reload['checks']) or set(checks) != {
            (g, k, v) for g in parent.GROUPS for k in parent.NETS for v in parent.VIEWS}:
        raise ValueError('Neural reload coverage is incomplete or duplicated')
    for (group, k, view), check in checks.items():
        current = parent.completed_member(group, k)
        if (current is None or current['model'] != check['model']
                or current['predictions'][view] != check['predictions']
                or current['preprocessing'] != reload['preprocessing']
                or check['rows'] != 512 or check['max_abs_sec'] > .01):
            raise ValueError('Neural reload result does not cover the current member')
    model = lgb.Booster(model_file=str(verify(model_record['model'])), params={'num_threads': 2})
    if model.feature_name() != model_record['columns'] or model.current_iteration() != 10000:
        raise ValueError('Reloaded main tree recipe differs')
    result = dict(source=parent.artifact(Path(__file__)),
                  neural_reload=parent.artifact(reload_path),
                  main_model=parent.artifact(candidate.OUT / 'model.json'), views={})
    for view in parent.VIEWS:
        directory = candidate.OUT / 'candidates' / view
        manifest_path = directory / 'manifest.json'
        manifest = read(manifest_path)
        if manifest['view'] != view or manifest['design'] != candidate.DESIGN:
            raise ValueError('Candidate view/recipe differs')
        if (manifest['source_sha256'] != parent.sha256(Path(candidate.__file__))
                or manifest['design_sha256'] != parent.sha256(candidate.OUT / 'design.json')):
            raise ValueError('Candidate source/design differs')
        exact = manifest['exact_parent_control']
        if not (exact['exact_parent_reproduced'] and exact['exact_raw_predictions_reproduced']
                and manifest['exact_parent_tree_mean']):
            raise ValueError('Exact parent replay must precede candidate arithmetic')
        for record in [manifest['parent_prepared'], manifest['model_record'], manifest['main_prediction'],
                       manifest['cv_evidence'], *manifest['parent_parts'], *manifest['parent_tree_components']]:
            verify(record)
        parts = pd.read_parquet(directory / 'parts.parquet')
        ids = parts.mvt_id.to_numpy()
        if len(ids) != parent.features.EXPECTED_ROWS[view] or not parts.mvt_id.is_unique:
            raise ValueError('Candidate row population differs')
        network_values = []
        expected_records = []
        for group in parent.GROUPS:
            for k in parent.NETS:
                rec = parent.completed_member(group, k)
                if rec is None:
                    raise ValueError('Incomplete neural member')
                expected_records.append(parent.artifact(parent.member_paths(group, k)[1]))
                predictions = ordered(verify(rec['predictions'][view]), ids, ['mvt_id', 'base'])
                network_values.append(predictions.base.to_numpy(float))
        if expected_records != manifest['network_records']:
            raise ValueError('Candidate network record list differs')
        neural = np.mean(network_values, axis=0)
        np.testing.assert_array_equal(neural, parts.nn_base.to_numpy())
        del network_values

        components, component_sources = candidate.parent_components(view, ids)
        if component_sources != manifest['parent_tree_components']:
            raise ValueError('Tree component provenance differs')
        pred_record = read(verify(manifest['main_prediction']))
        main = ordered(verify(pred_record['prediction']), ids, ['mvt_id', 'pred']).pred.to_numpy(float)
        np.testing.assert_array_equal(components.lgb, parts.main_lgb_original)
        np.testing.assert_array_equal(main, parts.main_lgb_callsign)
        # Direct arithmetic, separate from the builder's replacement function.
        tree_values = [main if name == 'lgb' else components[name].to_numpy(float)
                       for name in parent.hs.GBM]
        tree = np.column_stack(tree_values).mean(axis=1)
        np.testing.assert_array_equal(tree, parts.gbm_base.to_numpy())
        ordinary = .71 * neural + (1 - .71) * tree
        np.testing.assert_array_equal(ordinary, parts.ordinary.to_numpy())

        if view == 'old':
            old_p = pd.concat([pd.read_parquet(parent.hs.OUT / 'submission_v54' / f'ranking_{m}.parquet',
                                              columns=['mvt_id', 'p'])
                               for m in parent.features.RANK_MONTHS[view]], ignore_index=True)
        else:
            old_p = pd.read_parquet(parent.fs.WORK / 'parts/rome.parquet', columns=['mvt_id', 'p'])
        np.testing.assert_array_equal(old_p.mvt_id, ids)
        rome_record = read(verify(manifest['rome']['record']))
        if rome_record != manifest['rome']['contents'] or rome_record['view'] != view:
            raise ValueError('Rome provenance differs')
        cat = ordered(verify(rome_record['prediction']), ids, ['mvt_id', 'p']).p.to_numpy(float)
        probability = .5 * old_p.p.to_numpy(float) + .5 * cat
        np.testing.assert_array_equal(probability, parts.p.to_numpy())

        sample_manifest = read(parent.OUT / 'model_reload_audit/samples.json')
        sample = pd.read_pickle(verify(sample_manifest['views'][view]['sample']))
        if list(sample['X']) != model.feature_name():
            raise ValueError('Main tree sample feature order differs')
        actual = np.maximum(model.predict(sample['X'], num_threads=2), 0).astype(np.float32)
        expected = pd.Series(main, index=ids).loc[sample['mvt_id']].to_numpy(np.float32)
        np.testing.assert_array_equal(actual, expected)

        output = pq.read_table(verify(manifest['output']))
        template = pq.read_table(verify(manifest['template']))
        if output.schema != template.schema or output.column_names != template.column_names:
            raise ValueError('Submission schema differs')
        for column in template.column_names:
            if column != 'TAXITIME_SEC_mvt' and not output[column].equals(template[column]):
                raise ValueError('Protected submission column differs: ' + column)
        template_ids = output['MVT_ID_mvt'].to_numpy()
        expected_serialized = np.rint(pd.Series(parts.pred.to_numpy(), index=ids).loc[template_ids].to_numpy())
        np.testing.assert_array_equal(output['TAXITIME_SEC_mvt'].to_numpy(), expected_serialized)
        if not np.isfinite(expected_serialized).all() or (expected_serialized < 0).any():
            raise ValueError('Nonfinite/negative serialized prediction')

        original = pq.read_table(verify(exact['parent_submission'])).to_pandas().set_index('MVT_ID_mvt')
        original = original.loc[template_ids, 'TAXITIME_SEC_mvt'].to_numpy(float)
        delta = expected_serialized - original
        result['views'][view] = dict(rows=len(ids), manifest=parent.artifact(manifest_path),
            output=manifest['output'], parts=parent.artifact(directory / 'parts.parquet'),
            all_network_means_exact=True, all_equal_tree_means_exact=True, all_ordinary_blends_exact=True,
            all_rome_probability_blends_exact=True, saved_tree_sample_rows=len(actual),
            saved_tree_sample_exact=True, template_protected_columns_exact=True,
            all_serialized_predictions_exact=True, changed_predictions=int(np.count_nonzero(delta)),
            mean_change_sec=float(delta.mean()), rms_change_sec=float(np.sqrt(np.mean(delta ** 2))),
            maximum_absolute_change_sec=float(np.abs(delta).max()))
        del parts, components, old_p, sample, output, template
        gc.collect()
    result['passed'] = True
    result['scope'] = ('Independent component arithmetic, saved-tree inference and serialization; '
                       'inherited corrections use the builder\'s exact raw/serialized parent replay. '
                       'No hidden labels or new fits.')
    parent.write_json(candidate.OUT / 'candidate_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        audit()
