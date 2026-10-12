"""Independent SQL bounds, ordered affine sum and integer v80 transport audit.

No producer functions, model inference, calibration or fit are invoked. Upstream
independent audits are hash-bound; old stages must equal actual selected files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_selected_stack'
PUBLIC = RUNS / 'v3_20261012_final_public_corrections'
QUALITY = RUNS / 'v3_20261012_final_public_quality_components'
JOINT = RUNS / 'v3_20261012_final_joint_public_inference'
INPUTS = RUNS / 'v3_20261012_final_correction_inputs'
GEOMETRY = RUNS / 'v3_20261012_target205_geometry'
TABM = RUNS / 'v3_20261012_nn_public_tabm_pool3_submit_stack'
ROBUST = RUNS / 'v3_20261012_joint_public_other_runway_submit'
COMBINED = RUNS / 'v3_20261012_tabm_pool3_other_runway_combo_submit'
MONTHS = {'old': ['2026-01', '2026-07'], 'final': ['2026-01', '2026-02', '2026-06', '2026-07']}
RAW_SHA = {'old': '9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
           'final': '4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f'}


def read(path):
    return json.loads(path.read_text())


def art(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest())


def verify(record):
    path = ROOT / record['path']
    assert art(path) == {k: record[k] for k in ['path', 'sha256']}, path
    if 'bytes' in record:
        assert path.stat().st_size == record['bytes']
    return path


def load_record(record, ids, id_column='mvt_id', value_column=None):
    frame = pd.read_parquet(verify(record))
    assert len(frame) == len(ids) and frame[id_column].is_unique
    np.testing.assert_array_equal(frame[id_column], ids)
    if value_column is None:
        return frame
    value = frame[value_column].to_numpy(float)
    assert np.isfinite(value).all() and (value >= 0).all()
    np.testing.assert_array_equal(value, np.rint(value))
    return value


def load_upstream(folder, view, ids, global_identity=False):
    manifest = folder / ('identity.json' if global_identity else view + '/identity.json')
    identity = read(manifest)
    for a in identity['sources']:
        verify(a)
    result, audit = read(folder / view / 'result.json'), read(folder / view / 'audit.json')
    assert audit['status'] == 'pass'
    assert audit['identity'] == result['identity'] == art(manifest)
    assert audit['result'] == art(folder / view / 'result.json')
    assert audit['prediction'] == result['prediction']
    return load_record(result['prediction'], ids)


def constrained(expression):
    bounded = f'''greatest(0.,CASE WHEN d.has_id AND isfinite(d.nm_gap)
        THEN greatest(0.,least(greatest(({expression}),d.nm_gap-3606.),d.nm_gap+3606.))
        ELSE ({expression}) END)'''
    lo, hi = '(w.recovered_gap-3606.)', '(w.recovered_gap+3606.)'
    exclusion = f'''CASE WHEN isfinite(w.recovered_gap) AND ({bounded})>{lo} AND ({bounded})<{hi}
        THEN CASE WHEN {lo}>=0 AND ({bounded})-{lo}<={hi}-({bounded}) THEN {lo} ELSE {hi} END
        ELSE ({bounded}) END'''
    return f'round_even(greatest(0.,({exclusion})),0)'


def audit(view):
    started = time.monotonic()
    folder = OUT / view
    assert not (folder / 'audit.json').exists()
    identity, result = read(folder / 'identity.json'), read(folder / 'result.json')
    assert art(Path(__file__)) in identity['sources']
    assert identity['source'] == art(ROOT / 'VALIDATION/final_selected_stack.py')
    assert identity['view'] == result['view'] == view
    assert result['identity'] == art(folder / 'identity.json') and result['no_fit'] and result['no_upload']
    for a in identity['sources']:
        verify(a)
    template_path = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
    raw_path = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
    assert art(raw_path)['sha256'] == RAW_SHA[view]
    template = pq.read_table(template_path)
    ids = template['MVT_ID_mvt'].to_numpy()
    n = 344841 if view == 'old' else 670790
    assert len(ids) == len(np.unique(ids)) == identity['rows'] == result['rows'] == n
    stages = load_record(result['prediction'], ids)
    expected_columns = ['mvt_id', 'v72', 'v74', 'v76', 'v77', 'v78', 'raw79', 'v79',
                        'tabm_v78', 'robust_v78', 'tabm_delta', 'robust_delta', 'v80']
    assert list(stages.columns) == expected_columns
    assert np.isfinite(stages[expected_columns[1:]]).all().all()
    integer_columns = [c for c in expected_columns if c not in ['mvt_id', 'raw79']]
    np.testing.assert_array_equal(stages[integer_columns], np.rint(stages[integer_columns]))
    assert stages[[c for c in integer_columns if c not in ['tabm_delta', 'robust_delta']]].ge(0).all().all()
    public = load_upstream(PUBLIC, view, ids, True)
    daily = load_upstream(QUALITY, view, ids)
    joint = load_upstream(JOINT, view, ids, True)
    ir = read(INPUTS / view / 'prepared.json')
    ia = read(INPUTS / 'audit.json')
    assert ia['status'] == 'pass' and ia['identity'] == ir['identity'] == art(INPUTS / 'identity.json')
    for a in ir['outputs']:
        verify(a)
    d = pd.read_parquet(INPUTS / view / 'daily.parquet')
    np.testing.assert_array_equal(d.mvt_id, ids)
    matches = pd.read_parquet(INPUTS / view / 'matches.parquet')
    witnesses = matches[matches.accepted][['mvt_id', 'recovered_gap']]
    assert witnesses.mvt_id.is_unique and set(witnesses.mvt_id).issubset(ids)
    versions, weights = identity['versions'], np.asarray(identity['weights'], float)
    prepared, solution = read(GEOMETRY / 'prepared.json'), read(GEOMETRY / 'result.json')
    assert versions == prepared['versions'] and len(versions) == len(set(versions)) == 51 and 75 not in versions
    assert solution['prepared'] == art(GEOMETRY / 'prepared.json')
    assert set(solution['solution']['weights']) == set(map(str, versions))
    np.testing.assert_array_equal(weights, [solution['solution']['weights'][str(v)] for v in versions])
    assert np.isfinite(weights).all() and abs(weights.sum()-1)<1e-8 and (weights >= -3-1e-8).all() and (weights <= 4+1e-8).all()
    recent = {72: public.v72.to_numpy(float), 74: public.v74.to_numpy(float),
              76: daily.v76.to_numpy(float), 77: daily.main_pred.to_numpy(float), 78: stages.v78.to_numpy(float)}
    if view == 'old':
        matrix = load_record(identity['old_basis'], ids)
        for v, values in recent.items():
            np.testing.assert_array_equal(matrix[f'v{v}'], values)
        matrix = matrix[[f'v{v}' for v in versions]].copy()
    else:
        old_audit = read(OUT / 'old/audit.json')
        assert old_audit['status'] == 'pass' and old_audit['result'] == art(OUT / 'old/result.json')
        binding = read(PUBLIC / 'final_basis.json')
        assert binding['identity'] == art(PUBLIC / 'identity.json')
        assert binding['all_original_components_exact'] and binding['all_final_ids_exact']
        for a in binding['sources']:
            verify(a)
        verify(binding['catalog'])
        components = binding['components']
        cols = {}
        for v in versions:
            if v in recent:
                cols[f'v{v}'] = recent[v]
            else:
                record = components[str(v)]
                cols[f'v{v}'] = load_record(record['final'], ids,
                    record.get('id_column', 'mvt_id'), record.get('value_column', 'submitted'))
        matrix = pd.DataFrame(cols)
    assert np.isfinite(matrix).all().all() and matrix.ge(0).all().all()
    np.testing.assert_array_equal(matrix, np.rint(matrix))
    matrix = matrix.astype(float)
    matrix.insert(0, 'mvt_id', ids)
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name, frame in [('s', stages), ('p', public), ('q', daily), ('j', joint), ('d', d), ('w', witnesses), ('b', matrix)]:
            con.register(name, frame)
        con.read_parquet(str(raw_path)).create_view('raw')
        assert con.execute('''SELECT count(*) FROM s JOIN raw r ON s.mvt_id=r.MVT_ID_mvt
            WHERE r.PHASE_mvt<>'DEP' OR strftime(r.MVT_TIME_UTC_mvt,'%Y-%m') NOT IN
            (SELECT DISTINCT month FROM d)''').fetchone()[0] == 0
        assert con.execute('SELECT count(*),count(distinct r.MVT_ID_mvt) FROM s JOIN raw r ON s.mvt_id=r.MVT_ID_mvt').fetchone() == (n, n)
        assert sorted(d.month.unique()) == MONTHS[view]
        assert con.execute('''SELECT count(*) FROM s JOIN p USING(mvt_id) JOIN q USING(mvt_id) JOIN j USING(mvt_id)
            WHERE s.v72 IS DISTINCT FROM p.v72 OR s.v74 IS DISTINCT FROM p.v74
            OR s.v76 IS DISTINCT FROM q.v76 OR s.v77 IS DISTINCT FROM q.main_pred
            OR s.tabm_delta IS DISTINCT FROM cast(s.tabm_v78 AS DOUBLE)-s.v78
            OR s.robust_delta IS DISTINCT FROM cast(s.robust_v78 AS DOUBLE)-s.v78
            OR s.v79 IS DISTINCT FROM round_even(greatest(0.,s.raw79),0)
            OR s.v80 IS DISTINCT FROM round_even(greatest(0.,cast(s.v79 AS DOUBLE)+s.tabm_delta+s.robust_delta),0)
            OR (NOT j.robust_active AND s.robust_v78 IS DISTINCT FROM s.v78)
            OR (s.tabm_delta=0 AND s.robust_delta=0 AND s.v80 IS DISTINCT FROM s.v79)''').fetchone()[0] == 0
        for column, expression in [('v78', 'cast(q.main_pred AS DOUBLE)+.5*j.increment'),
                                   ('tabm_v78', 'cast(q.tabm_pred AS DOUBLE)+.5*j.increment'),
                                   ('robust_v78', 'cast(s.v78 AS DOUBLE)+.5*j.robust_delta')]:
            assert con.execute(f'''SELECT count(*) FROM s JOIN q USING(mvt_id) JOIN j USING(mvt_id)
                JOIN d USING(mvt_id) LEFT JOIN w USING(mvt_id)
                WHERE s.{column} IS DISTINCT FROM {constrained(expression)}''').fetchone()[0] == 0, column
        expression = '(' + '+'.join(f'b.v{v}*cast({repr(float(weight))} AS DOUBLE)' for v, weight in zip(versions, weights)) + ')'
        con.execute(f'CREATE TEMP TABLE independent_sum AS SELECT b.mvt_id,{expression} AS raw_sql FROM b')
        error, uncertain = con.execute('''SELECT max(abs(s.raw79-a.raw_sql)),count(*) FILTER(
            WHERE abs(greatest(0.,s.raw79)-floor(greatest(0.,s.raw79))-.5)<=1e-8)
            FROM s JOIN independent_sum a USING(mvt_id)''').fetchone()
        assert error <= 1e-8, error
    old_values = 0
    if view == 'old':
        affine = load_record(identity['old_affine'], ids)
        np.testing.assert_allclose(stages.raw79, affine.raw, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(stages.v79, affine.submitted)
        for version, record in identity['controls'].items():
            np.testing.assert_array_equal(stages['v'+version], load_record(record, ids, 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'))
            old_values += n
        for base, pairs in [(TABM, [('tabm_v78', 'tabm_v78'), ('tabm_delta', 'serialized_delta')]),
                            (ROBUST, [('robust_v78', 'robust_v78'), ('robust_delta', 'serialized_delta')]),
                            (COMBINED, [('v80', 'pred')])]:
            r, a = read(base / 'result.json'), read(base / 'audit.json')
            assert r['identity'] == a['identity'] == art(base / 'identity.json')
            assert a['status'] == 'pass' and a['result'] == art(base / 'result.json')
            old = pd.read_parquet(base / 'predictions.parquet')
            np.testing.assert_array_equal(old.mvt_id, ids)
            for name, original in pairs:
                np.testing.assert_array_equal(stages[name], old[original])
                old_values += n
    candidate = pq.read_table(verify(result['candidate']))
    assert candidate.schema == template.schema and len(candidate) == n
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt':
            assert candidate[name].equals(template[name])
    np.testing.assert_array_equal(candidate['TAXITIME_SEC_mvt'].to_numpy(), stages.v80)
    value = dict(status='pass', source=art(Path(__file__)), identity=art(folder / 'identity.json'),
        result=art(folder / 'result.json'), prediction=result['prediction'], candidate=result['candidate'],
        view=view, rows=n, affine_components=51, old_values_exact=old_values,
        all_upstream_audits_and_original_weights_bound=True, all_SQL_bounds_integer_deltas_and_serialization_exact=True,
        independent_affine_sum_max_error_sec=float(error), affine_sum_atol_sec=1e-8,
        near_half_integer_rows=int(uncertain), half_integer_policy='Frozen producer serialization retained; old all-row integer equality required. No per-row patch or substituted rounding.',
        full_template_schema_ID_order_non_target_columns_exact=True,
        no_model_inference_fit_or_upload=True, inherited_provenance=result['caveat'], seconds=time.monotonic()-started)
    (folder / 'audit.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(dict(status='pass', view=view, rows=n, affine_error_sec=error, near_half_integer_rows=uncertain), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=['old', 'final'], required=True)
    with threadpool_limits(1):
        audit(parser.parse_args().view)
