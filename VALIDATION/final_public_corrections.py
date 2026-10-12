"""Frozen v71-v74 and t5/full-monthly correction port; inference only.

Old replay is independent of the incomplete final affine basis. Final inference
requires all 48 individually replayed and audited final integer components.
The later direct-model replacement belongs to the final stack assembler.
"""
import argparse
import json
import os
from pathlib import Path
import platform

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

import daily_heavy_tail as heavy
import public_constraints_candidate as constraints
import final_correction_inputs as inputs
from records import sha256, write_json

base = heavy.base
ROOT = base.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261012_final_public_corrections'
MANIFEST = OUT/'identity.json'
BINDING = OUT/'final_basis.json'
V71 = ROOT/'VALIDATION/runs/v3_20261011_physical_affine/submission_v71'
V72 = ROOT/'VALIDATION/runs/v3_20261011_daily_delay_submit/submission_v72'
V73 = ROOT/'VALIDATION/runs/v3_20261011_expanded_affine/submission_v73'
V74 = constraints.OUT/'submission_v74'
V76 = ROOT/'VALIDATION/runs/v3_20261011_public_neural_candidate/submission_v76'
AGGREGATE = ROOT/'VALIDATION/runs/v3_20261011_public_refinements_aggregate_submit'
art = base.artifact
DESIGN = dict(
    versions=[71, 72, 73, 74], views=['old', 'final'],
    affine='Saved ordered48/49 serialized component vectors and original float64 matrix products; no new solve',
    daily='Original UTC takeoff-day population, serialized v71, full2025 calibration, Gaussian and fixed t5 ordinary grid',
    daily_source='Existing airport/UTC takeoff-day public feature grouping; exact public-source cohort identity is not established',
    daily_serialization='rint(max(posterior,0)) before raw73 + saved_w72*(t5_int-gaussian_int)',
    monthly='Original local-calendar eligibility and saved full2025 coefficients; half shift then max(0,2*half-refined73)',
    missing_monthly_source='Preserve original apply: missing source cells receive zero shift',
    constraints='Matched bound, max zero, recovered-plan exclusion, rint(max(value,0))',
    later_transport='constrain(serialized74 + direct_candidate - direct_control), with one final serialization by assembler',
    final_gate='All48 explicitly bound audited integer vectors plus exact old stage replay and SQL audit',
    no_fit=True, no_new_selection=True, no_ranking_labels=True, no_upload=True)


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = ROOT/record['path']
    assert path.is_file() and sha256(path) == record['sha256'], record['path']
    return path


def contains_artifact(value, record):
    if isinstance(value, dict):
        return (value.get('path') == record['path'] and value.get('sha256') == record['sha256']) or any(
            contains_artifact(child, record) for child in value.values())
    return isinstance(value, list) and any(contains_artifact(child, record) for child in value)


def audit_binds(audit, final, result_record=None, result_value=None):
    """Recognize direct receipts or a verified audit -> result -> vector chain."""
    if contains_artifact(audit, final):
        return True
    return result_record is not None and result_value is not None and (
        contains_artifact(audit, result_record) or audit.get('result_sha256') == result_record['sha256']
    ) and contains_artifact(result_value, final)


def serialized(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    result = np.rint(np.maximum(values, 0))
    assert result.max(initial=0) <= np.iinfo(np.int32).max
    return result.astype(np.int32)


def template(view):
    assert view in ['old', 'final']
    path = ROOT/'data'/('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
    ids = pd.read_parquet(path, columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    assert len(ids) == (344841 if view == 'old' else 670790) and pd.Index(ids).is_unique
    return ids


def vector(record, ids, id_column='MVT_ID_mvt', value_column='TAXITIME_SEC_mvt'):
    frame = pd.read_parquet(verify(record), columns=[id_column, value_column])
    np.testing.assert_array_equal(frame[id_column], ids)
    values = frame[value_column].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    np.testing.assert_array_equal(values, serialized(values))
    return values


def source_basis(folder, versions):
    manifest = read(folder/'manifest.json')
    answer = {}
    for version in versions:
        paths = [p for p in manifest['sources'] if Path(p).name == f'unique-rose_v{version}.parquet']
        assert len(paths) == 1, (version, paths)
        answer[str(version)] = dict(path=paths[0], sha256=manifest['sources'][paths[0]])
        verify(answer[str(version)])
    return answer


def freeze():
    assert not MANIFEST.exists(), 'Preserve frozen sources'
    audit = read(inputs.OUT/'audit.json')
    assert audit['status'] == 'pass' and audit['identity'] == art(inputs.MANIFEST)
    correction_identity = read(inputs.MANIFEST)
    for record in correction_identity['sources']:
        verify(record)
    for view in DESIGN['views']:
        receipt = read(inputs.OUT/view/'prepared.json')
        assert receipt['identity'] == art(inputs.MANIFEST)
        for record in receipt['outputs']:
            verify(record)
    a71, a73 = read(V71/'geometry.json'), read(V73/'geometry.json')
    for folder in [V71, V73]:
        production = read(folder/'manifest.json')
        record = art(folder/'geometry.json')
        assert production['sources'][record['path']] == record['sha256'], record['path']
    assert len(a71['versions']) == 48 and a73['versions'] == a71['versions'] + [72]
    assert all(set(map(str, a['versions'])) == set(a['weights']) for a in [a71, a73])
    basis71, basis73 = source_basis(V71, a71['versions']), source_basis(V73, a73['versions'])
    assert all(basis73[k] == v for k, v in basis71.items())
    aggregate_audit = read(AGGREGATE/'audit.json')
    assert aggregate_audit['status'] == 'pass'
    aggregate_result = read(AGGREGATE/'result.json')
    assert aggregate_result['identity'] == art(AGGREGATE/'identity.json')
    assert aggregate_audit['result'] == art(AGGREGATE/'result.json')
    verify(aggregate_result['prediction'])
    aggregate_sources = {r['path']: r['sha256'] for r in read(AGGREGATE/'identity.json')['sources']}
    for path in [Path(heavy.__file__), Path(constraints.__file__), Path(constraints.monthly.__file__),
        Path(constraints.exclusion.__file__), V72/'parameters.json', V74/'parameters.json']:
        record = art(path)
        assert aggregate_sources[record['path']] == record['sha256'], record['path']
    for folder in [V72, V74]:
        production = read(folder/'manifest.json')
        for filename in ['parameters.json', 'predictions.parquet']:
            record = art(folder/filename)
            assert production['sources'][record['path']] == record['sha256'], record['path']
    sources = [Path(__file__), ROOT/'VALIDATION/final_public_corrections_audit.py',
        Path(inputs.__file__), Path(heavy.__file__), Path(heavy.aligned.__file__),
        Path(heavy.old.__file__), Path(base.__file__), Path(constraints.__file__),
        Path(constraints.monthly.__file__), Path(constraints.exclusion.__file__),
        ROOT/'VALIDATION/records.py', ROOT/'VALIDATION/physical_affine.py',
        ROOT/'VALIDATION/expanded_scored_affine.py', ROOT/'VALIDATION/daily_delay_submit.py',
        ROOT/'VALIDATION/public_refinements_aggregate_submit.py', ROOT/'VALIDATION/public_main_candidate.py',
        inputs.MANIFEST, inputs.OUT/'audit.json', ROOT/'data/submitting.parquet',
        ROOT/'data/final_submitting.parquet', V71/'geometry.json', V71/'manifest.json',
        V73/'geometry.json', V73/'manifest.json', V72/'manifest.json', V72/'parameters.json', V72/'predictions.parquet',
        V74/'manifest.json', V74/'parameters.json', V74/'predictions.parquet', V76/'predictions.parquet',
        AGGREGATE/'identity.json', AGGREGATE/'result.json', AGGREGATE/'audit.json',
        AGGREGATE/'predictions.parquet', base.SOURCE/'all_pre_departure_delays_2026.csv',
        base.SOURCE/'taxi_out_additional_time_2026.csv']
    sources += [verify(record) for record in correction_identity['sources']]
    for view in DESIGN['views']:
        sources += [inputs.OUT/view/'prepared.json']
        sources += [verify(r) for r in read(inputs.OUT/view/'prepared.json')['outputs']]
    sources += [verify(r) for r in basis73.values()]
    sources += [ROOT/f'submissions/unique-rose_v{v}.parquet' for v in [71, 72, 73, 74, 76]]
    identity = dict(design=DESIGN, sources=[art(p) for p in sorted(set(sources))],
        geometry71=dict(versions=a71['versions'], weights=a71['weights']),
        geometry73=dict(versions=a73['versions'], weights=a73['weights']), old_basis=basis71,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        gaussian_original_float_tolerance_sec=1e-8,
        inherited_provenance='Every final component retains its bound inference/reconstruction history; no claim that archived refits are original models')
    OUT.mkdir(parents=True, exist_ok=True)
    assert not list(OUT.glob('*/*.parquet'))
    write_json(MANIFEST, identity)
    print(dict(frozen=True, sources=len(identity['sources']), required_final_components=48), flush=True)
    return identity


def guard():
    identity = read(MANIFEST)
    assert identity['design'] == DESIGN
    for record in identity['sources']:
        verify(record)
    assert identity['runtime'] == dict(python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__)
    return identity


def binding():
    value = read(BINDING)
    assert value['identity'] == art(MANIFEST)
    verify(value['catalog'])
    for record in value['sources']:
        verify(record)
    return value


def bind(catalog):
    """Bind only a complete, reviewed catalog. Never synthesize a missing vector.

    Each component has old/final/audit artifact records and a provenance string.
    An optional result artifact supports audit -> result -> final hash chains.
    id_column/value_column default to mvt_id/submitted; optional old_id_column
    and old_value_column override only the replay vector's names.
    """
    identity = guard()
    assert not BINDING.exists(), 'Preserve final component binding'
    catalog = Path(catalog).resolve()
    value = read(catalog)
    assert value['schema'] == 'final_public_corrections_basis_v1'
    required = set(identity['old_basis'])
    assert set(value['components']) == required, dict(missing=sorted(required-set(value['components'])),
        extra=sorted(set(value['components'])-required))
    old_ids, final_ids = template('old'), template('final')
    sources = [catalog]
    for version, record in value['components'].items():
        assert isinstance(record['provenance'], str) and record['provenance'].strip()
        audit = read(verify(record['audit']))
        assert audit['status'] == 'pass', (version, 'component audit failed')
        result = read(verify(record['result'])) if 'result' in record else None
        assert audit_binds(audit, record['final'], record.get('result'), result), (version, 'audit does not bind final artifact')
        id_column = record.get('id_column', 'mvt_id')
        value_column = record.get('value_column', 'submitted')
        old_values = vector(record['old'], old_ids, record.get('old_id_column', id_column),
            record.get('old_value_column', value_column))
        expected = vector(identity['old_basis'][version], old_ids)
        np.testing.assert_array_equal(old_values, expected, err_msg=f'v{version} original replay')
        vector(record['final'], final_ids, id_column, value_column)
        sources += [verify(record[k]) for k in ['old', 'final', 'audit']]
        if 'result' in record:
            sources.append(verify(record['result']))
    result = dict(identity=art(MANIFEST), catalog=art(catalog), components=value['components'],
        sources=[art(p) for p in sorted(set(sources))], all_original_components_exact=True,
        all_final_ids_exact=True, all_final_vectors_integer_finite_nonnegative=True)
    write_json(BINDING, result)
    print(dict(bound=True, components=len(required), final_rows=len(final_ids)), flush=True)


def basis_matrix(view, ids, identity):
    versions = identity['geometry71']['versions']
    if view == 'old':
        return np.column_stack([vector(identity['old_basis'][str(v)], ids) for v in versions])
    components = binding()['components']
    return np.column_stack([vector(components[str(v)]['final'], ids,
        components[str(v)].get('id_column', 'mvt_id'),
        components[str(v)].get('value_column', 'submitted')) for v in versions])


def constrain(ids, rows, values, matches):
    values = constraints.matched_bound(rows, values)
    return constraints.exclusion.apply(ids, np.maximum(values, 0), matches)[0]


def transport(ids, rows, serialized74, candidate, control, matches):
    """Same arithmetic/order as public_main_candidate.constrained; no rounding."""
    parent = np.asarray(serialized74, dtype=np.float64)
    np.testing.assert_array_equal(parent, serialized(parent))
    return constrain(ids, rows, parent + np.asarray(candidate, float) - np.asarray(control, float), matches)


def load_stages(view):
    guard()
    folder = OUT/view
    receipt, audit = read(folder/'result.json'), read(folder/'audit.json')
    assert receipt['identity'] == art(MANIFEST) and audit['status'] == 'pass'
    assert audit['result'] == art(folder/'result.json') and audit['prediction'] == receipt['prediction']
    if view == 'final':
        binding()
        assert receipt['binding'] == art(BINDING)
    return pd.read_parquet(verify(receipt['prediction']))


def old_replay(frame, rows, matches):
    ids = frame.mvt_id.to_numpy()
    for version in [71, 72, 73, 74]:
        np.testing.assert_array_equal(frame[f'v{version}'], vector(art(ROOT/f'submissions/unique-rose_v{version}.parquet'), ids))
    p72, p74 = [pd.read_parquet(folder/'predictions.parquet') for folder in [V72, V74]]
    aggregate = pd.read_parquet(AGGREGATE/'predictions.parquet')
    direct76 = pd.read_parquet(V76/'predictions.parquet')
    for table in [p72, p74, aggregate, direct76]:
        np.testing.assert_array_equal(table.mvt_id, ids)
    np.testing.assert_allclose(frame.raw72, p72.pred, rtol=0, atol=1e-8)
    for column, reference in [('monthly74', 'monthly'), ('matched74', 'matched_bound'), ('raw74', 'pred')]:
        np.testing.assert_array_equal(frame[column], p74[reference])
    mapping = dict(v71='old71', raw72='old72', raw_refined72='new72', v72='old72_serialized',
        refined72='new72_serialized', raw73='raw73', v73='old73', refined73='new73',
        monthly74='old_monthly', half_monthly74='half_monthly', full_monthly74='full_monthly',
        raw74='old74', raw_refined74='new74', v74='old74_serialized', refined74='new74_serialized')
    for column, reference in mapping.items():
        np.testing.assert_array_equal(frame[column], aggregate[reference], err_msg=column)
    old76 = transport(ids, rows, frame.v74, direct76.candidate, direct76.control, matches)
    refined76 = transport(ids, rows, frame.refined74, direct76.candidate, direct76.control, matches)
    np.testing.assert_array_equal(old76, direct76.pred)
    np.testing.assert_array_equal(old76, aggregate.old76)
    np.testing.assert_array_equal(refined76, aggregate.aggregate76)
    np.testing.assert_array_equal(serialized(old76), vector(art(ROOT/'submissions/unique-rose_v76.parquet'), ids))
    np.testing.assert_array_equal(serialized(refined76), aggregate.submitted)
    return dict(all_serialized_v71_v72_v73_v74_v76_exact=True,
        all_aggregate_stages_and_transport_exact=True,
        original_v72_float_max_error=float(np.max(np.abs(frame.raw72-p72.pred))))


def infer(view):
    identity = guard()
    if view == 'final':
        load_stages('old')
        binding()
    folder = OUT/view
    assert not (folder/'result.json').exists(), 'Preserve completed inference'
    ids = template(view)
    rows, monthly_rows, matches = [pd.read_parquet(inputs.OUT/view/f'{name}.parquet')
        for name in ['daily', 'monthly', 'matches']]
    for table in [rows, monthly_rows]:
        np.testing.assert_array_equal(table.mvt_id, ids)
        assert 'y' not in table
    P = basis_matrix(view, ids, identity)
    w71 = np.array([identity['geometry71']['weights'][str(v)] for v in identity['geometry71']['versions']])
    raw71 = P@w71
    p71 = serialized(raw71)
    params = read(V72/'parameters.json')['parameters']
    source = base.public(2026).set_index(['airport', 'day'])
    folder.mkdir(parents=True, exist_ok=True)
    # Save completed posterior inference before downstream replay/audit so a
    # serialization diagnostic never causes this expensive step to be repeated.
    daily_receipt = folder/'daily_state.json'
    daily_binding = art(BINDING) if view == 'final' else None
    if daily_receipt.exists():
        checkpoint = read(daily_receipt)
        assert checkpoint['identity'] == art(MANIFEST) and checkpoint['binding'] == daily_binding
        previous = pd.read_parquet(verify(checkpoint['prediction']))
        np.testing.assert_array_equal(previous.mvt_id, ids)
        np.testing.assert_array_equal(previous.raw71, raw71)
        np.testing.assert_array_equal(previous.v71, p71)
        gaussian, student = previous.raw72.to_numpy(), previous.raw_refined72.to_numpy()
        logs = pd.read_parquet(verify(checkpoint['days'])).to_dict('records')
    else:
        gaussian, student = [np.full(len(ids), np.nan) for _ in range(2)]
        logs = []
        for (airport, day), at in rows.groupby(['airport', 'day']).indices.items():
            group = rows.iloc[at]
            public = source.loc[(airport, day)]
            assert np.isfinite([public.total, public.n_public]).all(), (airport, day)
            args = (group.gap.to_numpy(float), p71[at].astype(float), group.has_id.to_numpy(bool),
                group.nm_gap.to_numpy(float), airport, params[airport], float(public.total), float(public.n_public-len(group)))
            gaussian[at], ginfo = heavy.condition(*args, logprior=heavy.GAUSSIAN)
            student[at], sinfo = heavy.condition(*args)
            logs.append(dict(airport=airport, day=day, rows=len(group), public_total=float(public.total),
                public_count=float(public.n_public), gaussian_offset=ginfo['offset'], **sinfo))
        assert np.isfinite(gaussian).all() and np.isfinite(student).all()
        pd.DataFrame(dict(mvt_id=ids, raw71=raw71, v71=p71, raw72=gaussian,
            raw_refined72=student)).to_parquet(folder/'daily_state.parquet', index=False)
        pd.DataFrame(logs).to_parquet(folder/'days.parquet', index=False)
        write_json(daily_receipt, dict(identity=art(MANIFEST), binding=daily_binding,
            prediction=art(folder/'daily_state.parquet'), days=art(folder/'days.parquet')))
    p72, refined72 = serialized(gaussian), serialized(student)
    P73 = np.column_stack([P, p72.astype(float)])
    del P
    w73 = np.array([identity['geometry73']['weights'][str(v)] for v in identity['geometry73']['versions']])
    raw73 = P73@w73
    del P73
    p73 = serialized(raw73)
    refined73 = serialized(raw73 + identity['geometry73']['weights']['72']*(refined72.astype(float)-p72.astype(float)))
    monthly_params = read(V74/'parameters.json')['full_year']
    monthly_source = constraints.monthly.public(2026)
    # Float inputs preserve the original pandas groupby/mean arithmetic exactly.
    monthly74, old_cells = constraints.monthly.apply(monthly_rows, p73.astype(float), monthly_source, monthly_params)
    half74, refined_cells = constraints.monthly.apply(monthly_rows, refined73.astype(float), monthly_source, monthly_params)
    full74 = np.maximum(0, 2*half74-refined73.astype(float))
    matched74 = constraints.matched_bound(rows, monthly74)
    raw74 = constrain(ids, rows, monthly74, matches)
    raw_refined74 = constrain(ids, rows, full74, matches)
    frame = pd.DataFrame(dict(mvt_id=ids, raw71=raw71, v71=p71, raw72=gaussian, raw_refined72=student,
        v72=p72, refined72=refined72, raw73=raw73, v73=p73, refined73=refined73,
        monthly74=monthly74, half_monthly74=half74, full_monthly74=full74, matched74=matched74,
        raw74=raw74, raw_refined74=raw_refined74, v74=serialized(raw74), refined74=serialized(raw_refined74)))
    assert np.isfinite(frame.drop(columns='mvt_id').to_numpy()).all()
    replay = old_replay(frame, rows, matches) if view == 'old' else None
    frame.to_parquet(folder/'stages.parquet', index=False)
    pd.concat([old_cells.assign(branch='original'), refined_cells.assign(branch='refined')], ignore_index=True).to_parquet(folder/'monthly_cells.parquet', index=False)
    result = dict(identity=art(MANIFEST), view=view, rows=len(ids), months=rows.month.value_counts().sort_index().to_dict(),
        prediction=art(folder/'stages.parquet'), days=art(folder/'days.parquet'), monthly_cells=art(folder/'monthly_cells.parquet'),
        binding=art(BINDING) if view == 'final' else None, old_replay=replay, no_fit=True, no_upload=True)
    write_json(folder/'result.json', result)
    print(dict(view=view, rows=len(ids), airport_days=len(logs), old_replay=replay, independent_audit_pending=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'bind', 'infer'])
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--view', choices=['old', 'final'])
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'bind':
            assert args.catalog is not None
            bind(args.catalog)
        elif args.stage == 'infer':
            assert args.view is not None
            infer(args.view)
        else:
            freeze()
