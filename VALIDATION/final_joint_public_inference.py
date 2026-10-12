"""Exact original/robust public posterior ports using retained calibration.

Only final-month posteriors are evaluated. Original ranking inputs must replay
exactly and their saved original/robust posterior outputs are reused intact.
No affine basis or v77 prediction is needed until downstream transport.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

import joint_public_moments as model
import joint_public_other_runway as robust
import daily_offblock_alignment as alignment
import monthly_taxi_reconcile as monthly
import final_correction_inputs as correction_inputs
from records import sha256, write_json

ROOT = model.prior.base.ROOT
RUNS = ROOT/'VALIDATION/runs'
OUT = RUNS/'v3_20261012_final_joint_public_inference'
MANIFEST = OUT/'identity.json'
ORIGINAL = RUNS/'v3_20261011_joint_public_moments_submit'
ROBUST = RUNS/'v3_20261012_joint_public_other_runway_submit'
MODERN = RUNS/'v3_20261012_final_legacy_modern_replay'
QUALITY = RUNS/'v3_20261011_public_daily_taxi_submit'
LEGACY_AUDIT = RUNS/'v3_20261012_final_legacy_audit/audit.json'
FEATURES = dict(old=RUNS/'v3_20261011_public_daily_taxi_features/inputsranking.parquet',
    final=RUNS/'v3_20261012_final_daily_main_inference/features/inputsfinal.parquet')
MONTHS = dict(old=['2026-01', '2026-07'], final=['2026-01', '2026-02', '2026-06', '2026-07'])
AUDITOR = ROOT/'VALIDATION/final_joint_public_audit.py'
INPUTS = robust.pilot.INPUTS
art = model.prior.base.artifact
DESIGN = dict(original_settings=model.SETTINGS, robust_settings=robust.SETTINGS,
    months=MONTHS, input_columns=INPUTS,
    prior='Exact serialized v70 independently audited final view; retained old prior',
    assignments='Takeoff minus fixed v70 once; explicit interior days of each UTC takeoff month; exclude VFR',
    taxi='Original local takeoff day/runway, stand/runway eligibility, monthly invalid share strictly below0.05',
    source='Reuse complete cached parent taxi_source/delay_source, require all final months covered',
    calibration='Original saved full2025 taxi/delay/support parameters, exact original hashes; never refit',
    old='Reconstruct inputs exactly then reuse both retained posterior outputs; do not rerun',
    final='Original model.apply then unchanged robust.apply, separate per-month checkpoints',
    anchoring='robust.old_joint versus saved original finaljoint atol1e-8; anchor delta to saved joint, force zero for no extra cells',
    transfer='Export fulltemplate-ID increment and robust delta; half-weight/bounds/serialization only in later assembler',
    provenance='v70 final retains previously audited component reconstruction history',
    no_fit=True, no_departure_labels=True, no_new_selection=True, no_upload=True)


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = ROOT/record['path']
    assert path.is_file() and sha256(path) == record['sha256'], record['path']
    return path


def template(view):
    assert view in MONTHS
    path = ROOT/'data'/('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
    ids = pd.read_parquet(path, columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    assert len(ids) == (344841 if view == 'old' else 670790) and pd.Index(ids).is_unique
    return ids


def aligned(path, ids):
    frame = pd.read_parquet(path)
    assert frame.mvt_id.is_unique and len(frame) == len(ids)
    assert set(frame.mvt_id) == set(ids)
    return frame.set_index('mvt_id').loc[ids].reset_index()


def source_index(identity):
    return {record['path']: record['sha256'] for record in identity['sources']}


def freeze():
    assert not MANIFEST.exists(), 'Preserve frozen sources'
    original, robust_identity = read(ORIGINAL/'identity.json'), read(ROBUST/'identity.json')
    assert original['design']['change']['settings'] == model.SETTINGS
    assert robust_identity['design']['settings'] == robust.SETTINGS
    authorized = source_index(original)
    for path, digest in source_index(robust_identity).items():
        if path in authorized:
            assert authorized[path] == digest, path
        authorized[path] = digest
    numerical_sources = [Path(model.__file__), Path(robust.__file__), Path(robust.pilot.__file__),
        Path(model.prior.__file__), Path(model.prior.base.__file__), Path(model.taxi.__file__),
        Path(alignment.__file__), Path(monthly.__file__),
        ROOT/'VALIDATION/joint_public_moments_submit.py', ROOT/'VALIDATION/joint_public_other_runway_submit.py']
    for path in numerical_sources:
        record = art(path)
        assert authorized[record['path']] == record['sha256'], record['path']
    parameters = read(ORIGINAL/'parameters.json')
    assert parameters['allowed'] == [f'2025-{m:02d}' for m in range(1, 13)]
    saved = [ORIGINAL/name for name in ['inputs.parquet', 'parameters.json', 'taxi_source.parquet', 'delay_source.parquet']]
    for path in saved:
        record = art(path)
        assert source_index(original)[record['path']] == record['sha256'], record['path']
    sources = [Path(__file__), AUDITOR, Path(correction_inputs.__file__), ROOT/'VALIDATION/records.py',
        *numerical_sources, *saved, ORIGINAL/'identity.json', ROBUST/'identity.json',
        LEGACY_AUDIT, MODERN/'identity.json', MODERN/'old_v70.json', MODERN/'final_v70.json',
        MODERN/'old_v70.parquet', MODERN/'final_v70.parquet', ROOT/'submissions/unique-rose_v70.parquet',
        correction_inputs.MANIFEST, correction_inputs.OUT/'audit.json', QUALITY/'identity.json',
        model.prior.base.SOURCE/'taxi_out_additional_time_2026.csv',
        ROOT/'data/submitting.parquet', ROOT/'data/final_submitting.parquet']
    monthly_record = art(model.prior.base.SOURCE/'taxi_out_additional_time_2026.csv')
    # Original joint packets inherited these quality flags from the audited
    # daily-quality producer, which directly froze the complete monthly CSV.
    assert read(QUALITY/'identity.json')['sources'][monthly_record['path']] == monthly_record['sha256']
    for folder in [ORIGINAL, ROBUST]:
        audit, inference = read(folder/'audit.json'), read(folder/'inference.json')
        assert audit['status'] == 'pass' and audit['identity'] == art(folder/'identity.json')
        assert audit['result'] == art(folder/'result.json')
        assert inference['identity'] == art(folder/'identity.json')
        if folder == ROBUST:
            assert audit['inference'] == art(folder/'inference.json')
        else:
            # cells.parquet is authenticated through this original inference
            # receipt; the later robust identity directly pins the first two.
            for filename in ['inference.json', 'inference.parquet']:
                record = art(folder/filename)
                assert authorized[record['path']] == record['sha256'], record['path']
        for record in inference['artifacts']:
            sources.append(verify(record))
        result = read(folder/'result.json')
        assert result['identity'] == art(folder/'identity.json')
        assert art(folder/'predictions.parquet') in result['artifacts']
        for record in result['artifacts']:
            if record['path'].endswith('/predictions.parquet'):
                verify(record)
        sources += [folder/name for name in ['audit.json', 'inference.json', 'result.json', 'predictions.parquet']]
    legacy = read(LEGACY_AUDIT)
    assert legacy['status'] == 'pass'
    entry = [row for row in legacy['predictions'] if row['version'] == 70]
    assert len(entry) == 1 and entry[0]['old_exact_in_SQL']
    assert entry[0]['final_prediction'] == art(MODERN/'final_v70.parquet')
    for view in MONTHS:
        v70 = read(MODERN/f'{view}_v70.json')
        assert v70['identity'] == art(MODERN/'identity.json')
        assert v70['prediction'] == art(MODERN/f'{view}_v70.parquet')
        assert v70['original'] == art(ROOT/'submissions/unique-rose_v70.parquet')
        feature_manifest = FEATURES[view].parent/'manifest.json'
        feature_audit = FEATURES[view].parent/'audit.json'
        audit, manifest = read(feature_audit), read(feature_manifest)
        assert audit['status'] == 'pass' and audit['manifest'] == art(feature_manifest)
        assert manifest['records']['ranking' if view == 'old' else 'final']['inputs'] == art(FEATURES[view])
        prepared = read(correction_inputs.OUT/view/'prepared.json')
        assert prepared['identity'] == art(correction_inputs.MANIFEST)
        sources += [feature_manifest, feature_audit, FEATURES[view], correction_inputs.OUT/view/'prepared.json']
        sources += [verify(record) for record in prepared['outputs']]
        sources.append(ROOT/'data'/('ranking.parquet' if view == 'old' else 'final_ranking.parquet'))
    correction_audit = read(correction_inputs.OUT/'audit.json')
    assert correction_audit['status'] == 'pass' and correction_audit['identity'] == art(correction_inputs.MANIFEST)
    taxi_source, delay_source = [pd.read_parquet(ORIGINAL/name) for name in ['taxi_source.parquet', 'delay_source.parquet']]
    assert not taxi_source.duplicated(['local_day', 'runway']).any() and delay_source.day.is_unique
    assert set(MONTHS['final']).issubset(taxi_source.local_day.str[:7])
    assert set(MONTHS['final']).issubset(delay_source.day.str[:7])
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, dict(design=DESIGN, sources=[art(path) for path in sorted(set(sources))],
        parameters=art(ORIGINAL/'parameters.json'), taxi_source=art(ORIGINAL/'taxi_source.parquet'),
        delay_source=art(ORIGINAL/'delay_source.parquet'), source_month_coverage_verified=True,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)))
    print(dict(frozen=True, sources=len(read(MANIFEST)['sources']), no_fit=True), flush=True)


def guard():
    identity = read(MANIFEST)
    assert identity['design'] == DESIGN
    for record in identity['sources']:
        verify(record)
    assert identity['runtime'] == dict(python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__)
    return identity


def input_packet(view):
    ids = template(view)
    daily = aligned(correction_inputs.OUT/view/'daily.parquet', ids)
    clocks = aligned(correction_inputs.OUT/view/'clocks.parquet', ids)
    eligibility = aligned(correction_inputs.OUT/view/'monthly.parquet', ids)
    local = aligned(FEATURES[view], ids)
    v70 = aligned(MODERN/f'{view}_v70.parquet', ids)
    prior = v70.submitted.to_numpy(float)
    assert np.isfinite(prior).all() and (prior >= 0).all()
    np.testing.assert_array_equal(prior, np.rint(prior))
    if view == 'old':
        original = pd.read_parquet(ROOT/'submissions/unique-rose_v70.parquet')
        np.testing.assert_array_equal(original.MVT_ID_mvt, ids)
        np.testing.assert_array_equal(prior, original.TAXITIME_SEC_mvt)
    np.testing.assert_array_equal(local.takeoff, clocks.takeoff)
    np.testing.assert_array_equal(local.airport, daily.airport)
    assert clocks.takeoff.dtype.unit == 'us'
    epoch = clocks.takeoff.astype('int64').to_numpy()/1e6
    np.testing.assert_array_equal(epoch[::4096], np.array([t.timestamp() for t in clocks.takeoff.iloc[::4096]]))
    assigned = alignment.dates(epoch, prior)
    months = clocks.takeoff.dt.strftime('%Y-%m').to_numpy()
    assert set(months) == set(MONTHS[view])
    scope = np.zeros(len(ids), bool)
    for month in MONTHS[view]:
        at = np.flatnonzero(months == month)
        scope[at] = alignment.interior(assigned[at], month)
    scope &= ~clocks.fltrule.eq('V').to_numpy()
    quality_rows = local[['airport', 'local_day']].assign(local_month=local.local_day.str[:7]).merge(
        monthly.public(2026)[['airport', 'local_month', 'invalid']],
        on=['airport', 'local_month'], how='left', validate='many_to_one')
    quality = quality_rows.invalid.lt(.05).to_numpy()
    frame = pd.DataFrame(dict(mvt_id=ids, airport=daily.airport.to_numpy(), month=months,
        gap=daily.gap.to_numpy(), prior=prior, eligible=eligibility.eligible.to_numpy(), quality=quality,
        runway=local.runway.to_numpy(), local_day=local.local_day.to_numpy(), assigned_day=assigned,
        fltrule=clocks.fltrule.to_numpy(), delay_scope=scope))
    frame = frame[frame.airport.eq('LIRF')].drop(columns='airport').reset_index(drop=True)
    assert list(frame.columns) == INPUTS and frame.mvt_id.is_unique
    assert not {'y', 'true_day', 'BLOCK_TIME_UTC_mvt'}.intersection(frame.columns)
    if view == 'old':
        pd.testing.assert_frame_equal(frame, pd.read_parquet(ORIGINAL/'inputs.parquet'), check_exact=True)
    return frame


def prepare():
    guard()
    assert not (OUT/'prepared.json').exists(), 'Preserve prepared packets'
    outputs, rows = {}, {}
    for view in MONTHS:
        frame = input_packet(view)
        folder = OUT/view
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(folder/'inputs.parquet', index=False)
        outputs[view] = art(folder/'inputs.parquet')
        rows[view] = dict(rome_rows=len(frame), months=frame.month.value_counts().sort_index().to_dict())
    write_json(OUT/'prepared.json', dict(identity=art(MANIFEST), inputs=outputs, rows=rows,
        old_inputs_exact=True, no_departure_labels=True, no_fit=True))
    print(dict(prepared=True, rows=rows, independent_input_audit_pending=True), flush=True)


def input_guard():
    guard()
    prepared, audit = read(OUT/'prepared.json'), read(OUT/'input_audit.json')
    assert prepared['identity'] == art(MANIFEST)
    assert audit['status'] == 'pass' and audit['identity'] == art(MANIFEST)
    assert audit['prepared'] == art(OUT/'prepared.json')
    for record in prepared['inputs'].values():
        verify(record)
    return prepared


def checkpoint(month, kind):
    assert month in MONTHS['final'] and kind in ['joint', 'robust']
    path = OUT/'final/months'/month/f'{kind}.json'
    receipt = read(path)
    assert receipt['identity'] == art(MANIFEST) and receipt['prepared'] == art(OUT/'prepared.json')
    assert receipt['month'] == month and receipt['kind'] == kind
    for key in ['prediction', 'cells']:
        verify(receipt[key])
    if kind == 'robust':
        assert receipt['joint'] == art(path.parent/'joint.json')
        raw = read(verify(receipt['raw']))
        assert raw['identity'] == art(MANIFEST) and raw['prepared'] == art(OUT/'prepared.json')
        assert raw['month'] == month
        for key in ['prediction', 'cells']:
            verify(raw[key])
    return receipt


def infer(month):
    input_guard()
    assert month in MONTHS['final']
    rows = pd.read_parquet(OUT/'final/inputs.parquet', filters=[('month', '==', month)])
    assert len(rows) and rows.month.nunique() == 1
    params = read(ORIGINAL/'parameters.json')
    st, sd = [pd.read_parquet(ORIGINAL/name) for name in ['taxi_source.parquet', 'delay_source.parquet']]
    folder = OUT/'final/months'/month
    folder.mkdir(parents=True, exist_ok=True)
    for kind, function in [('joint', model.apply), ('robust', robust.apply)]:
        if (folder/f'{kind}.json').exists():
            checkpoint(month, kind)
            continue
        assert not (folder/f'{kind}.parquet').exists(), 'Preserve unreceipted output for diagnosis'
        start = time.monotonic()
        raw_receipt = folder/'robust_raw.json'
        if kind == 'robust' and raw_receipt.exists():
            raw = read(raw_receipt)
            assert raw['identity'] == art(MANIFEST) and raw['prepared'] == art(OUT/'prepared.json')
            assert raw['month'] == month
            frame, cells = [pd.read_parquet(verify(raw[key])) for key in ['prediction', 'cells']]
        else:
            if kind == 'robust':
                assert not (folder/'robust_raw.parquet').exists() and not (folder/'robust_raw_cells.parquet').exists(), 'Preserve unreceipted raw posterior for diagnosis'
            frame, cells = function(rows, st, sd, params['taxi'], params['delay'], params['support'])
            if kind == 'robust':
                frame.to_parquet(folder/'robust_raw.parquet', index=False)
                cells.to_parquet(folder/'robust_raw_cells.parquet', index=False)
                write_json(raw_receipt, dict(identity=art(MANIFEST), prepared=art(OUT/'prepared.json'),
                    month=month, prediction=art(folder/'robust_raw.parquet'),
                    cells=art(folder/'robust_raw_cells.parquet'), seconds=time.monotonic()-start))
        np.testing.assert_array_equal(frame.mvt_id, rows.mvt_id)
        extra = {}
        if kind == 'robust':
            joint_receipt = checkpoint(month, 'joint')
            original = pd.read_parquet(verify(joint_receipt['prediction'])).set_index('mvt_id').loc[frame.mvt_id]
            np.testing.assert_array_equal(frame.touched, original.touched)
            error = float(np.max(np.abs(frame.old_joint.to_numpy()-original.joint.to_numpy())))
            np.testing.assert_allclose(frame.old_joint, original.joint, atol=1e-8, rtol=0)
            frame['recomputed_old_joint'] = frame.old_joint
            frame['old_joint'] = original.joint.to_numpy()
            frame['new_joint'] = np.where(frame.extra_cells.gt(0), frame.new_joint, frame.old_joint)
            frame['delta'] = frame.new_joint-frame.old_joint
            assert not frame.loc[frame.extra_cells.eq(0), 'delta'].any()
            extra = dict(joint=art(folder/'joint.json'), raw=art(raw_receipt), old_joint_max_error_sec=error,
                delta_anchored_to_saved_joint=True, no_extra_cells_exact_zero=True)
        frame['month'] = month
        cells['month'] = month
        frame.to_parquet(folder/f'{kind}.parquet', index=False)
        cells.to_parquet(folder/f'{kind}_cells.parquet', index=False)
        write_json(folder/f'{kind}.json', dict(identity=art(MANIFEST), prepared=art(OUT/'prepared.json'),
            kind=kind, month=month, rows=len(frame), seconds=time.monotonic()-start,
            prediction=art(folder/f'{kind}.parquet'), cells=art(folder/f'{kind}_cells.parquet'), **extra))
        print(dict(month=month, kind=kind, rows=len(frame), seconds=time.monotonic()-start, **extra), flush=True)


def inference_frames(view):
    if view == 'old':
        for folder in [ORIGINAL, ROBUST]:
            receipt = read(folder/'inference.json')
            assert receipt['identity'] == art(folder/'identity.json')
            for record in receipt['artifacts']:
                verify(record)
        return (pd.read_parquet(ORIGINAL/'inference.parquet'), pd.read_parquet(ROBUST/'inference.parquet'),
            [art(ORIGINAL/'inference.json'), art(ROBUST/'inference.json')])
    records = [checkpoint(month, kind) for month in MONTHS['final'] for kind in ['joint', 'robust']]
    return (pd.concat([pd.read_parquet(verify(r['prediction'])) for r in records if r['kind'] == 'joint'], ignore_index=True),
        pd.concat([pd.read_parquet(verify(r['prediction'])) for r in records if r['kind'] == 'robust'], ignore_index=True),
        [art(OUT/'final/months'/month/f'{kind}.json') for month in MONTHS['final'] for kind in ['joint', 'robust']])


def package(view):
    input_guard()
    folder = OUT/view
    assert not (folder/'result.json').exists(), 'Preserve completed packet'
    rows = pd.read_parquet(folder/'inputs.parquet')
    joint, other, receipts = inference_frames(view)
    for table in [joint, other]:
        assert table.mvt_id.is_unique and len(table) == len(rows) and set(table.mvt_id) == set(rows.mvt_id)
    joint, other = [table.set_index('mvt_id').loc[rows.mvt_id] for table in [joint, other]]
    np.testing.assert_array_equal(joint.prior, rows.prior)
    np.testing.assert_array_equal(joint.touched, other.touched)
    np.testing.assert_array_equal(joint.joint, other.old_joint)
    np.testing.assert_array_equal(joint.increment, joint.joint-joint.dep_only)
    np.testing.assert_array_equal(other.delta, other.new_joint-other.old_joint)
    assert not joint.loc[~joint.touched, 'increment'].any()
    assert not other.loc[other.extra_cells.eq(0), 'delta'].any()
    ids = template(view)
    j, r = joint.reindex(ids), other.reindex(ids)
    frame = pd.DataFrame(dict(mvt_id=ids, increment=j.increment.fillna(0).to_numpy(),
        touched=j.touched.fillna(False).to_numpy(bool), robust_delta=r.delta.fillna(0).to_numpy(),
        robust_active=r.extra_cells.fillna(0).gt(0).to_numpy(), extra_cells=r.extra_cells.fillna(0).to_numpy(np.int32),
        weight=r.weight.fillna(1).to_numpy(), extra_q=r.extra_q.fillna(0).to_numpy()))
    assert np.isfinite(frame[['increment', 'robust_delta', 'weight', 'extra_q']]).all().all()
    if view == 'old':
        original = aligned(ORIGINAL/'predictions.parquet', ids)
        saved_robust = aligned(ROBUST/'predictions.parquet', ids)
        np.testing.assert_array_equal(frame.increment, original.increment)
        np.testing.assert_array_equal(frame.touched, original.touched)
        np.testing.assert_array_equal(frame.robust_delta, saved_robust.posterior_delta)
        np.testing.assert_array_equal(frame.robust_active, saved_robust.active)
    frame.to_parquet(folder/'increments.parquet', index=False)
    write_json(folder/'result.json', dict(identity=art(MANIFEST), prepared=art(OUT/'prepared.json'),
        input_audit=art(OUT/'input_audit.json'), view=view, rows=len(frame), rome_rows=len(rows),
        prediction=art(folder/'increments.parquet'), inference_receipts=receipts,
        old_saved_posteriors_reused=view == 'old', no_posterior_rerun_on_old=view == 'old',
        no_rounding_or_downstream_transport=True, no_fit=True, no_upload=True))
    print(dict(view=view, rows=len(frame), rome_rows=len(rows), independent_posterior_audit_pending=True), flush=True)


def load_increments(view):
    input_guard()
    folder = OUT/view
    result, audit = read(folder/'result.json'), read(folder/'audit.json')
    assert result['identity'] == art(MANIFEST) and audit['status'] == 'pass'
    assert audit['identity'] == art(MANIFEST) and audit['result'] == art(folder/'result.json')
    assert audit['prediction'] == result['prediction']
    for record in result['inference_receipts']:
        verify(record)
    return pd.read_parquet(verify(result['prediction']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'prepare', 'infer', 'package'])
    parser.add_argument('--month', choices=MONTHS['final'])
    parser.add_argument('--view', choices=list(MONTHS))
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'infer':
            assert args.month is not None
            infer(args.month)
        elif args.stage == 'package':
            assert args.view is not None
            package(args.view)
        else:
            globals()[args.stage]()
