"""Independent pre-score input/split audit for the fixed portable student.

Reads projected predictions and raw input fields only. It never imports the
producer, decodes taxi truth, solves coefficients, or computes accuracy scores.
All four immutable fit receipts and the complete blocked prediction must exist
before this audit; the true-label score stage requires its passed receipt.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/runs/v3_20261013_v80_teacher_distillation'
PRODUCER = ROOT/'VALIDATION/portable_v80_distill_20261013.py'
PLAN = ROOT/'VALIDATION/reports/cohort_learning_20261013/v80_distillation_fixed_design.json'
AFFINE = ROOT/'VALIDATION/runs/v3_20261013_portable_affine'
PUBLIC = ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
VERSIONS = [1,4,8,12,13,14,15,16,17,18,19,20,21,30,33,34,37,38,45,46,54,56,57,58,60,61,64,70]
MONTHS = ['2025-01', '2025-07']
META = ['mvt_id', 'month', 'day', 'airport']
PRED = META+['v82', 'v80', 'block', 'raw', 'submitted']
ALLOWED = {'2025-01':[f'2025-{m:02d}' for m in range(3,12)],
           '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
ACCESS = []


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            h.update(block)
    return {'path':str(path.relative_to(ROOT)), 'sha256':h.hexdigest()}


def verify(record):
    path = ROOT/record['path']
    assert art(path) == record, record['path']
    return path


def ids_hash(ids):
    values = np.asarray(ids)
    assert np.isfinite(values).all() and np.array_equal(values, values.astype(np.int64))
    return hashlib.sha256(values.astype(np.int64).tobytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def project(record, columns):
    # Reject truth-shaped columns before even opening a source footer.
    assert not {'y', 'TAXITIME_SEC_mvt'} & set(columns), columns
    path = verify(record)
    frame = pd.read_parquet(path, columns=columns)
    assert list(frame) == columns
    ACCESS.append({'artifact':record, 'columns':columns, 'rows':len(frame)})
    return frame


def from_spec(spec, columns):
    assert spec['columns'] == columns
    path = verify(spec['artifact']); pf = pq.ParquetFile(path)
    assert hashlib.sha256(str(pf.schema_arrow).encode()).hexdigest() == spec['schema_sha256']
    assert pf.metadata.num_rows == spec['rows']
    frame = project(spec['artifact'], columns)
    assert len(frame) == spec['rows']
    return frame


def align(frame, ids):
    assert frame.mvt_id.is_unique and frame.mvt_id.notna().all()
    assert len(frame) == len(ids) and set(frame.mvt_id) == set(ids)
    return frame.set_index('mvt_id').loc[ids].reset_index()


def input_source_check():
    """Check the reviewed inference call graph contains no direct truth reader."""
    tree = ast.parse(PRODUCER.read_text())
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ['infer', 'inputs', 'coefficient', 'block_masks', 'solve']:
        assert name in functions
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'score', 'load_prediction_truth'}
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                assert node.slice.value not in {'y', 'truth', 'TAXITIME_SEC_mvt'} or name == 'inputs'
    # inputs() has a separately authenticated production submission teacher;
    # infer() calls only the cv branch. Full call-graph reasoning is source review.
    return {'producer_AST_parses':True, 'inference_separated_from_score':True,
            'scope':'Supplement to independently reviewed, hash-pinned source; not a sandbox claim.'}


def provenance():
    m = read(OUT/'identity.json'); inference = read(OUT/'cv/inference.json')
    assert m['source'] == art(PRODUCER) and m['design'] == art(PLAN)
    assert m['input_auditor'] == art(__file__) and m['versions'] == VERSIONS
    assert m['review'] == art(OUT/'source_review.json')
    review = read(verify(m['review']))
    assert review['status'] == 'pass' and review['source'] == m['source']
    assert review['input_auditor'] == m['input_auditor'] and review['design'] == m['design']
    pinned = {r['path']:r for r in m['sources']}
    assert len(pinned) == len(m['sources'])
    for record in pinned.values():
        verify(record)
    for record in [m['source'],m['design'],m['input_auditor'],m['review'],m['fixture'],m['auditor'],m['metadata']]:
        assert pinned[record['path']] == record
    metadata = read(verify(m['metadata']))
    assert metadata['status']=='pass' and metadata['source']==m['source'] and metadata['design']==m['design']
    assert metadata['specs']==m['specs']
    assert {v:{view:r[view] for view in ['old','final']} for v,r in m['components'].items()}==metadata['components']
    fixture = read(verify(m['fixture']))
    assert fixture['status']=='pass' and fixture['source']==m['source'] and fixture['design']==m['design']
    plan = read(PLAN)
    assert plan['basis']['versions'] == VERSIONS
    for record in plan['metadata_refs']:
        assert pinned[record['path']] == record; verify(record)
    a = read(AFFINE/'identity.json'); result = read(AFFINE/'result.json'); audit = read(AFFINE/'independent_audit.json')
    assert a['versions'] == result['versions'] == VERSIONS
    assert audit['status'] == 'pass' and result['identity'] == audit['identity'] == art(AFFINE/'identity.json')
    assert audit['result'] == art(AFFINE/'result.json') and audit['predictions'] == result['predictions']
    assert m['historical_cv_caveat'] == a['historical_cv_caveat']
    assert 'original27' in m['historical_cv_caveat']
    parent = read(PUBLIC/'identity.json'); pr = read(PUBLIC/'cv/result.json'); pa = read(PUBLIC/'cv/audit.json')
    assert pa['status'] == 'pass' and pa['result'] == art(PUBLIC/'cv/result.json')
    assert pa['identity'] == pr['identity'] == art(PUBLIC/'identity.json')
    assert pa['prediction'] == pr['prediction'] == plan['basis']['rows']['cv']['anchor']
    spec = m['specs']['cv']
    assert spec['basis']['artifact'] == a['cv_basis'] == plan['basis']['CV']
    assert spec['teacher']['artifact'] == a['control_cv'] == plan['basis']['teacher_CV']
    assert spec['anchor']['artifact'] == pr['prediction']
    for key in ['raw', 'matches']:
        assert spec[key]['artifact'] == parent['specs']['cv'][key]
    assert spec['truth']['artifact'] == spec['teacher']['artifact']
    assert spec['truth']['columns'] == ['mvt_id','month','day','y']
    # Footer/provenance binding only: the truth projection is never decoded here.
    for value in spec.values():
        assert pinned[value['artifact']['path']] == value['artifact']
    assert inference['identity'] == art(OUT/'identity.json') and inference['rows'] == 344419
    assert inference['all_four_fits_before_truth'] and inference['no_taxi_labels_read']
    assert inference['allocation'] == art(OUT/'cv/allocation.json')
    allocation = read(verify(inference['allocation']))
    assert allocation['authority'] == '/root' and allocation['identity'] == inference['identity']
    assert allocation['state'] == 'released_for_distillation_cv'
    started = read(OUT/'cv/started.json')
    assert started['identity'] == inference['identity'] and started['allocation'] == inference['allocation']
    expected = [art(OUT/'cv/models'/f'{month}_{block}.json') for month in MONTHS for block in ['A','B']]
    assert inference['coefficients'] == inference['fit_receipts'] == expected
    assert inference['prediction'] == art(OUT/'cv/predictions.parquet')
    return m, inference, pinned


def audit():
    start = time.monotonic()
    assert not (OUT/'cv/input_audit.json').exists(), 'Preserve completed input audit'
    for path in ['score/started.json','cv/scored.parquet','cv/score.json','cv/audit.json']:
        assert not (OUT/path).exists(), f'Truth stage already started: {path}'
    source = input_source_check(); m, inference, pinned = provenance(); spec = m['specs']['cv']
    anchor = from_spec(spec['anchor'], META+['submitted','v80'])
    assert len(anchor) == 344419 and anchor.mvt_id.is_unique and anchor.mvt_id.notna().all()
    ids = anchor.mvt_id.to_numpy(); ids_hash(ids)
    assert set(anchor.month) == set(MONTHS)
    basis = align(from_spec(spec['basis'], META+[f'v{v}' for v in VERSIONS]), ids)
    teacher = align(from_spec(spec['teacher'], ['mvt_id','three_tabm_runway']), ids)
    for key in META:
        np.testing.assert_array_equal(basis[key], anchor[key])
    np.testing.assert_array_equal(teacher.three_tabm_runway, anchor.v80)
    matrix = basis[[f'v{v}' for v in VERSIONS]].to_numpy(float)
    controls = anchor[['submitted','v80']].to_numpy(float)
    for values in [matrix,controls]:
        assert np.isfinite(values).all() and (values>=0).all() and np.array_equal(values,np.rint(values))
    del matrix, basis, teacher
    raw_parts = []
    raw_columns = ['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','FLIGHT_ID_mvt','LOBT_flt']
    for month in MONTHS:
        matches = [r for p,r in pinned.items() if Path(p).parent==Path('data') and Path(p).name.startswith(f'training_{month}-01_')]
        assert len(matches) == 1
        f = project(matches[0], raw_columns)
        f = f[f.MVT_ID_mvt.isin(ids)].copy()
        assert f.PHASE_mvt.eq('DEP').all()
        t = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True)
        assert t.dt.strftime('%Y-%m').eq(month).all()
        raw_parts.append(pd.DataFrame({'mvt_id':f.MVT_ID_mvt,'month':t.dt.strftime('%Y-%m'),
            'day':t.dt.strftime('%Y-%m-%d'),'airport':f.ADEP_mvt,'has_id':f.FLIGHT_ID_mvt.notna(),
            'nm_gap':(t-pd.to_datetime(f.LOBT_flt,utc=True)).dt.total_seconds()}))
    raw = align(pd.concat(raw_parts,ignore_index=True), ids)
    for key in META:
        np.testing.assert_array_equal(raw[key],anchor[key])
    cached = from_spec(spec['raw'], ['mvt_id','month','has_id','nm_gap'])
    assert cached.mvt_id.is_unique
    cached = cached.set_index('mvt_id').loc[ids]
    for key in ['month','has_id','nm_gap']:
        np.testing.assert_array_equal(raw[key].to_numpy(),cached[key].to_numpy())
    matches = from_spec(spec['matches'], ['mvt_id','accepted','recovered_gap','donor_mvt_id','donor_month'])
    accepted = matches[matches.accepted]
    assert accepted.mvt_id.is_unique
    prediction = project(inference['prediction'],PRED)
    assert len(prediction)==344419 and prediction.mvt_id.is_unique
    for key in META:
        np.testing.assert_array_equal(prediction[key],anchor[key])
    np.testing.assert_array_equal(prediction.v82,anchor.submitted)
    np.testing.assert_array_equal(prediction.v80,anchor.v80)
    assert np.isfinite(prediction[['v82','v80','raw','submitted']]).all().all()
    assert prediction.submitted.ge(0).all() and np.array_equal(prediction.submitted,np.rint(prediction.submitted))
    scopes = []
    held_count = np.zeros(len(ids),dtype=int)
    for month in MONTHS:
        outer = np.flatnonzero(anchor.month.eq(month))
        dom = pd.to_datetime(anchor.iloc[outer].day,utc=True).dt.day.to_numpy()
        assert set(dom)==set(range(1,32))
        for block in ['A','B']:
            train = outer[dom>=17] if block=='A' else outer[dom<=14]
            held = outer[dom<=15] if block=='A' else outer[dom>=16]
            assert len(train) and len(held) and not np.intersect1d(train,held).size
            assert anchor.iloc[train].month.eq(month).all() and anchor.iloc[held].month.eq(month).all()
            rec = read(OUT/'cv/models'/f'{month}_{block}.json')
            assert rec['identity']==art(OUT/'identity.json') and rec['outer']==month and rec['block']==block
            assert rec['columns']==VERSIONS and rec['ridge']==1. and rec['intercept'] is False
            assert rec['train_rows']==len(train) and rec['held_rows']==len(held)
            assert rec['train_ids_sha256']==ids_hash(ids[train]) and rec['held_ids_sha256']==ids_hash(ids[held])
            assert np.asarray(rec['beta']).shape==(28,) and np.isfinite(rec['beta']).all()
            assert np.isfinite(rec['normal_equation_max_abs']) and rec['normal_equation_max_abs']>=0
            assert prediction.iloc[held].block.eq(block).all(); held_count[held]+=1
            permitted = accepted[accepted.mvt_id.isin(ids[held]) & accepted.donor_month.isin(ALLOWED[month]+[month])]
            scopes.append({'outer':month,'block':block,'train_rows':len(train),'held_rows':len(held),
                'train_ids_sha256':ids_hash(ids[train]),'held_ids_sha256':ids_hash(ids[held]),
                'one_complete_day_embargo':True,'same_outer_only':True,
                'permitted_recovered_constraints':len(permitted),'coefficient':art(OUT/'cv/models'/f'{month}_{block}.json')})
    np.testing.assert_array_equal(held_count,np.ones(len(ids),dtype=int))
    value = {'status':'pass','source':art(__file__),'identity':art(OUT/'identity.json'),
        'inference':art(OUT/'cv/inference.json'),'prediction':inference['prediction'],
        'coefficients':inference['coefficients'],'fit_receipts':inference['fit_receipts'],
        'rows':len(ids),'versions':VERSIONS,'scopes':scopes,'projected_reads':ACCESS,'source_checks':source,
        'exact_raw_ID_calendar_airport_constraint_inputs':True,'exact_teacher_anchor_and_basis_alignment':True,
        'all_rows_held_once':True,'all_four_coefficients_saved_before_truth':True,
        'no_taxi_truth_decoded':True,'no_fit_or_score':True,
        'inherited_model_scope':'Existing audited outer-month CV vectors authenticated; this audit independently checks student scope, not a refit of all historical component models.',
        'historical_cv_caveat':m['historical_cv_caveat'],
        'numerical_audit':'Coefficient normal equations, prediction arithmetic, constraints, actual truth and score gates are verified separately after this scope gate.',
        'seconds':time.monotonic()-start}
    write(OUT/'cv/input_audit.json',value)
    print(json.dumps({'status':'pass','rows':len(ids),'audit':art(OUT/'cv/input_audit.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
