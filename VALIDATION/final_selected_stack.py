"""Port the selected v78/v79/v80 recipe from audited four-month components.

Saved weights and integer component deltas are retained. Final construction
requires the complete historical basis, exact old replay, and upstream audits.
This module never fits, selects, uploads, or substitutes an absent component.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import final_public_corrections as corrections
import final_public_quality_components as quality
import final_joint_public_inference as joint
import final_correction_inputs as inputs
import target205_candidate as original79
from records import write_json
from submit import fill_template

ROOT = corrections.ROOT
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_selected_stack'
art, read, verify = corrections.art, corrections.read, corrections.verify
GEOMETRY = RUNS / 'v3_20261012_target205_geometry'
V79 = original79.OUT / 'submission_v79'
TABM = RUNS / 'v3_20261012_nn_public_tabm_pool3_submit_stack'
ROBUST = RUNS / 'v3_20261012_joint_public_other_runway_submit'
COMBINED = RUNS / 'v3_20261012_tabm_pool3_other_runway_combo_submit'
DESIGN = dict(
    v78='serialize(bound(v77 + 0.5 * saved_original_joint_increment))',
    v79='serialize(original_ordered51_integer_components @ saved_weights)',
    tabm_v78='serialize(bound(tabm_v77 + 0.5 * saved_original_joint_increment))',
    robust_v78='serialize(bound(v78 + 0.5 * saved_robust_posterior_delta))',
    v80='serialize(v79 + (tabm_v78-v78) + (robust_v78-v78))',
    serialization='numpy.rint(max(value,0)); no separate clipping of transported deltas',
    source_only=True, no_fit=True, no_weight_search=True, no_upload=True,
    final_gate='Complete audited48 historical components and exact old selected stages')


def aligned(frame, ids):
    assert frame.mvt_id.is_unique and len(frame) == len(ids)
    np.testing.assert_array_equal(frame.mvt_id, ids)
    return frame


def load_quality(view):
    folder = quality.OUT / view
    identity, result, audit = [read(folder / name) for name in
                              ['identity.json', 'result.json', 'audit.json']]
    assert identity['source'] == art(quality.__file__) and identity['view'] == view
    for record in identity['sources']:
        verify(record)
    assert result['identity'] == art(folder / 'identity.json') and result['view'] == view
    assert audit['status'] == 'pass' and audit['result'] == art(folder / 'result.json')
    assert audit['identity'] == art(folder / 'identity.json')
    assert audit['prediction'] == result['prediction']
    return pd.read_parquet(verify(result['prediction']))


def freeze(view):
    folder = OUT / view
    assert not (folder / 'identity.json').exists(), 'Preserve frozen view'
    ids = corrections.template(view)
    for frame in [corrections.load_stages(view), load_quality(view), joint.load_increments(view)]:
        aligned(frame, ids)
    prepared, diagnosis, manifest = [read(p) for p in [GEOMETRY / 'prepared.json',
        GEOMETRY / 'result.json', V79 / 'manifest.json']]
    assert manifest['prepared'] == art(GEOMETRY / 'prepared.json')
    assert manifest['geometry'] == art(GEOMETRY / 'result.json')
    assert diagnosis['prepared'] == art(GEOMETRY / 'prepared.json')
    assert manifest['weights'] == diagnosis['solution']['weights']
    versions = prepared['versions']
    weights = original79.weights_for(versions, diagnosis['solution'])
    assert len(versions) == 51 and 75 not in versions
    assert set(versions) - {72, 74, 76, 77, 78} == set(
        corrections.guard()['geometry71']['versions']) - {66, 68}
    sources = {Path(__file__), ROOT / 'VALIDATION/final_selected_stack_audit.py',
        Path(original79.__file__), GEOMETRY / 'prepared.json',
        GEOMETRY / 'result.json', V79 / 'manifest.json',
        verify(prepared['ranking']), verify(diagnosis['diagnostic_ranking']),
        ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')}
    for module in [corrections, quality, joint]:
        sources.add(Path(module.__file__))
        for name in ['identity.json', 'result.json', 'audit.json', 'stages.parquet']:
            path = module.OUT / view / name
            if path.exists():
                sources.add(path)
        if getattr(module, 'MANIFEST', None):
            sources.add(module.MANIFEST)
    sources.add(joint.OUT / view / 'increments.parquet')
    for path in [inputs.MANIFEST, inputs.OUT / 'audit.json', inputs.OUT / view / 'prepared.json',
                 inputs.OUT / view / 'daily.parquet', inputs.OUT / view / 'matches.parquet']:
        sources.add(path)
    if view == 'final':
        corrections.binding()
        sources.add(corrections.BINDING)
        old_audit = read(OUT / 'old/audit.json')
        assert old_audit['status'] == 'pass'
        assert old_audit['result'] == art(OUT / 'old/result.json')
        sources.update([OUT / 'old/identity.json', OUT / 'old/result.json', OUT / 'old/audit.json'])
    controls = {str(v): art(ROOT / f'submissions/unique-rose_v{v}.parquet')
                for v in [72, 74, 76, 77, 78, 79, 80]}
    for record in controls.values():
        sources.add(verify(record))
    for base in [TABM, ROBUST, COMBINED]:
        result = read(base / 'result.json')
        audit = read(base / 'audit.json')
        assert audit['status'] == 'pass' and audit['result'] == art(base / 'result.json')
        assert result['identity'] == art(base / 'identity.json')
        assert audit['identity'] == art(base / 'identity.json')
        assert corrections.contains_artifact(result, art(base / 'predictions.parquet'))
        sources.update([base / n for n in ['identity.json', 'result.json', 'audit.json', 'predictions.parquet']])
    # Pin every imported local helper used by the retained constraints and recipes.
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT / 'VALIDATION'):
                sources.add(path)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'identity.json', dict(design=DESIGN, view=view, source=art(__file__),
        sources=[art(p) for p in sorted(sources)], controls=controls, versions=versions,
        weights=weights.tolist(), old_basis=prepared['ranking'],
        old_affine=diagnosis['diagnostic_ranking'], rows=len(ids),
        inherited_provenance='Preserve every upstream reconstruction and October8 base-network refit caveat'))
    print(dict(view=view, frozen=True, sources=len(sources)), flush=True)


def guard(view):
    identity = read(OUT / view / 'identity.json')
    assert identity['design'] == DESIGN and identity['view'] == view
    assert identity['source'] == art(__file__)
    for record in identity['sources']:
        verify(record)
    return identity


def constrain(ids, rows, values, matches):
    return corrections.serialized(corrections.constrain(ids, rows, values, matches))


def build(view):
    folder = OUT / view
    identity = guard(view)
    assert not (folder / 'result.json').exists(), 'Preserve completed view'
    ids = corrections.template(view)
    public = aligned(corrections.load_stages(view), ids)
    daily = aligned(load_quality(view), ids)
    inferred = aligned(joint.load_increments(view), ids)
    rows = aligned(pd.read_parquet(inputs.OUT / view / 'daily.parquet'), ids)
    matches = pd.read_parquet(inputs.OUT / view / 'matches.parquet')
    v78 = constrain(ids, rows, daily.main_pred.to_numpy(float) + .5 * inferred.increment.to_numpy(), matches)
    tabm78 = constrain(ids, rows, daily.tabm_pred.to_numpy(float) + .5 * inferred.increment.to_numpy(), matches)
    np.testing.assert_array_equal(constrain(ids, rows, v78.astype(float), matches), v78)
    robust78 = constrain(ids, rows, v78.astype(float) + .5 * inferred.robust_delta.to_numpy(), matches)
    np.testing.assert_array_equal(robust78[~inferred.robust_active], v78[~inferred.robust_active])
    recent = {72: public.v72, 74: public.v74, 76: daily.v76, 77: daily.main_pred, 78: v78}
    if view == 'old':
        basis = aligned(pd.read_parquet(verify(identity['old_basis'])), ids)
        for version, values in recent.items():
            np.testing.assert_array_equal(values, basis[f'v{version}'])
        matrix = basis[[f'v{v}' for v in identity['versions']]].to_numpy(float)
    else:
        components = corrections.binding()['components']
        columns = []
        for version in identity['versions']:
            if version in recent:
                value = np.asarray(recent[version], dtype=float)
            else:
                record = components[str(version)]
                value = corrections.vector(record['final'], ids, record.get('id_column', 'mvt_id'),
                    record.get('value_column', 'submitted'))
            columns.append(value)
        matrix = np.column_stack(columns)
    np.testing.assert_array_equal(matrix, corrections.serialized(matrix))
    raw79 = matrix @ np.asarray(identity['weights'], dtype=float)
    v79 = corrections.serialized(raw79)
    old_values = 0
    if view == 'old':
        saved = aligned(pd.read_parquet(verify(identity['old_affine'])), ids)
        np.testing.assert_allclose(raw79, saved.raw, rtol=0, atol=1e-8)
        # Require every serialized cell to reproduce; a near-tie mismatch fails
        # closed and must be investigated rather than patched row by row.
        np.testing.assert_array_equal(v79, saved.submitted)
        old_values += len(ids)
    tabm_delta = tabm78.astype(float) - v78
    robust_delta = robust78.astype(float) - v78
    v80 = corrections.serialized(v79.astype(float) + tabm_delta + robust_delta)
    frame = pd.DataFrame(dict(mvt_id=ids, v72=public.v72, v74=public.v74, v76=daily.v76,
        v77=daily.main_pred, v78=v78, raw79=raw79, v79=v79, tabm_v78=tabm78,
        robust_v78=robust78, tabm_delta=tabm_delta, robust_delta=robust_delta, v80=v80))
    if view == 'old':
        for version, record in identity['controls'].items():
            np.testing.assert_array_equal(frame[f'v{version}'], corrections.vector(record, ids))
            old_values += len(ids)
        for base, pairs in [(TABM, [('tabm_v78', 'tabm_v78'), ('tabm_delta', 'serialized_delta')]),
                            (ROBUST, [('robust_v78', 'robust_v78'), ('robust_delta', 'serialized_delta')]),
                            (COMBINED, [('v80', 'pred')])]:
            prior = aligned(pd.read_parquet(base / 'predictions.parquet'), ids)
            for column, original in pairs:
                np.testing.assert_array_equal(frame[column], prior[original])
                old_values += len(ids)
    frame.to_parquet(folder / 'stages.parquet', index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(folder / 'stages.parquet'), frame, check_exact=True)
    template = pq.read_table(ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet'))
    table = fill_template(template, ids, v80)
    assert table.schema == template.schema
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt':
            assert table[name].equals(template[name])
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(), v80)
    pq.write_table(table, folder / 'candidate.parquet')
    assert pq.read_table(folder / 'candidate.parquet').equals(table)
    write_json(folder / 'result.json', dict(identity=art(folder / 'identity.json'), view=view,
        rows=len(ids), prediction=art(folder / 'stages.parquet'), candidate=art(folder / 'candidate.parquet'),
        exact_old_values=old_values, template_schema_ids_other_columns_exact=True,
        independent_audit_pending=True, no_fit=True, no_upload=True,
        caveat=identity['inherited_provenance']))
    print(dict(view=view, rows=len(ids), exact_old_values=old_values, independent_audit_pending=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'build'])
    parser.add_argument('--view', choices=['old', 'final'], required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage](args.view)
