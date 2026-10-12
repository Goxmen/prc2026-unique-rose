"""Independent SQL arithmetic/constraint audit for the frozen correction port."""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import final_public_corrections as port
from records import write_json


def sql_constraints(con, expression):
    # names stages/daily/matches are fixed local views, never shell/user SQL.
    return con.execute(f'''WITH bounded AS (
        SELECT s.mvt_id, greatest(0, CASE WHEN d.has_id AND isfinite(d.nm_gap)
            THEN greatest(0, least(greatest({expression},d.nm_gap-3606),d.nm_gap+3606))
            ELSE {expression} END) candidate_value, m.recovered_gap-3606 lo, m.recovered_gap+3606 hi
        FROM stages s JOIN daily d USING(mvt_id)
        LEFT JOIN (SELECT * FROM matches WHERE accepted) m USING(mvt_id))
        SELECT mvt_id, CASE WHEN candidate_value>lo AND candidate_value<hi THEN
            CASE WHEN lo>=0 AND candidate_value-lo<=hi-candidate_value THEN lo ELSE hi END
            ELSE candidate_value END AS predicted FROM bounded''').fetchdf().set_index('mvt_id')


def sql_monthly(con, parent_column, coef):
    assert parent_column in ['v73', 'refined73'] and len(coef) == 12
    airport_intercepts = 'CASE c.airport ' + ' '.join(
        f"WHEN '{a}' THEN {repr(float(coef[i]))}" for i, a in enumerate(port.constraints.monthly.ZONES)) + ' END'
    con.execute(f'''CREATE OR REPLACE TEMP VIEW independent_monthly AS
        WITH cells AS (
            SELECT m.airport,m.local_month,count(*) n,avg(CAST(s.{parent_column} AS DOUBLE)) pred_mean
            FROM stages s JOIN monthly_rows m USING(mvt_id)
            WHERE m.eligible AND s.{parent_column}<=7200 GROUP BY m.airport,m.local_month),
        c AS (SELECT cells.*,60*(p.TOTAL_REF_TIME_MIN+p.TOTAL_ADD_TIME_MIN)/p.TOTAL_REF_NB_FL mean_sec,
            (p.TF-p.VALID_FL)/p.TF invalid,p.NO_REF/p.TF noref
            FROM cells LEFT JOIN monthly_source p ON cells.airport=p.APT_ICAO
            AND cells.local_month=CAST(CAST(p.YEAR AS INTEGER) AS VARCHAR)||'-'||lpad(CAST(CAST(p.MONTH_NUM AS INTEGER) AS VARCHAR),2,'0')),
        calibrated AS (SELECT c.*, {airport_intercepts}+({repr(float(coef[10]))})*invalid+
            ({repr(float(coef[11]))})*noref composition FROM c)
        SELECT *,CASE WHEN invalid<0.05 AND n>=100 THEN
            0.5*least(greatest(mean_sec+composition-pred_mean,-120),120) ELSE 0 END shift FROM calibrated''')
    return con.execute(f'''SELECT s.mvt_id,greatest(0,CAST(s.{parent_column} AS DOUBLE)+
            CASE WHEN m.eligible AND s.{parent_column}<=7200 THEN coalesce(c.shift,0) ELSE 0 END) pred
        FROM stages s JOIN monthly_rows m USING(mvt_id)
        LEFT JOIN independent_monthly c USING(airport,local_month)''').fetchdf().set_index('mvt_id')


def preflight():
    """Small adversarial serialization/support fixture; no real inference."""
    final_record = dict(path='fixture/final.parquet', sha256='a'*64)
    result_record = dict(path='fixture/result.json', sha256='b'*64)
    final_result = dict(views={'final': {'30': {'prediction': final_record}}})
    assert port.audit_binds(dict(prediction=final_record), final_record)
    assert port.audit_binds(dict(result_sha256='b'*64), final_record, result_record, final_result)
    assert not port.audit_binds(dict(result_sha256='c'*64), final_record, result_record, final_result)
    assert not port.audit_binds(dict(result_sha256='b'*64), final_record, result_record, {})
    np.testing.assert_array_equal(port.serialized([-.5, .5, 1.5, 2.5, 3.5]), [0, 0, 2, 2, 4])
    ids = np.arange(8)
    daily = pd.DataFrame(dict(mvt_id=ids, has_id=[True, True, True, False, False, False, False, False],
        nm_gap=[0., 5000., -4000., np.nan, np.nan, np.nan, np.nan, np.nan]))
    stages = pd.DataFrame(dict(mvt_id=ids, x=[5000., 0., 100., 5000., 1500., 1394., 8606., 999.]))
    matches = pd.DataFrame(dict(mvt_id=[3, 4, 5, 6], accepted=[True]*4,
        recovered_gap=[5000., 2000., 5000., 5000.], donor_mvt_id=[13, 14, 15, 16]))
    expected = port.constrain(ids, daily, stages.x.to_numpy(), matches)
    np.testing.assert_array_equal(expected, [3606., 1394., 0., 1394., 5606., 1394., 8606., 999.])
    with duckdb.connect(config={'threads': 1}) as con:
        for name, frame in [('stages', stages), ('daily', daily), ('matches', matches)]:
            con.register(name, frame)
        actual = sql_constraints(con, 's.x').loc[ids, 'predicted'].to_numpy()
    np.testing.assert_array_equal(actual, expected)
    parent = np.full(8, 5000., dtype=float)
    candidate = np.arange(8, dtype=float)*.1
    control = candidate/2
    expected = port.constrain(ids, daily, parent+candidate-control, matches)
    np.testing.assert_array_equal(port.transport(ids, daily, parent, candidate, control, matches), expected)
    # Fixed fixture covers both cap signs, zero clipping, missing source, the
    # exact5% invalid boundary, 100-row threshold, and prediction truncation.
    airports = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD']
    n = 6*102
    mrows = pd.DataFrame(dict(mvt_id=np.arange(n), airport=np.repeat(airports, 102),
        local_month='2026-01', eligible=np.tile(np.r_[np.ones(101, bool), False], 6)))
    original = np.repeat([900., 5., 1000., 1000., 1000., 1000.], 102)
    original[4*102:4*102+2] = 7201
    source = pd.DataFrame(dict(airport=airports[:-1], local_month='2026-01',
        mean_sec=[2000., 0., 1000., 2000., 2000.], invalid=[0., 0., 0.05, 0.0499, 0.], noref=0.))
    params = dict(coefficients=[0.]*12)
    expected, _ = port.constraints.monthly.apply(mrows, original, source, params)
    raw_source = pd.DataFrame(dict(APT_ICAO=source.airport, YEAR=2026, MONTH_NUM=1,
        TOTAL_REF_TIME_MIN=source.mean_sec/60, TOTAL_ADD_TIME_MIN=0., TOTAL_REF_NB_FL=1.,
        TF=10000., VALID_FL=(1-source.invalid)*10000, NO_REF=0.))
    with duckdb.connect(config={'threads': 1}) as con:
        con.register('stages', pd.DataFrame(dict(mvt_id=mrows.mvt_id, v73=original)))
        con.register('monthly_rows', mrows)
        con.register('monthly_source', raw_source)
        actual = sql_monthly(con, 'v73', params['coefficients']).loc[mrows.mvt_id, 'pred'].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    result = dict(status='pass', source=port.art(Path(__file__)), producer=port.art(Path(port.__file__)),
        round_even_zero_clip=True, matched_including_negative_upper_bound=True,
        exclusion_tie_lower_negative_lower_and_strict_edges=True, transport_order=True,
        fixture_rows=8, monthly_fixture_rows=n, monthly_threshold_cap_missing_source_and_clip=True,
        direct_and_two_hop_audit_bindings_and_corruption_rejection=True, no_real_predictions=True)
    write_json(port.OUT/'preflight.json', result)
    print(result, flush=True)


def audit(view):
    identity = port.guard()
    folder = port.OUT/view
    assert not (folder/'audit.json').exists(), 'Preserve completed audit'
    receipt = port.read(folder/'result.json')
    assert receipt['identity'] == port.art(port.MANIFEST) and receipt['view'] == view
    for key in ['prediction', 'days', 'monthly_cells']:
        port.verify(receipt[key])
    if view == 'final':
        port.binding()
        assert receipt['binding'] == port.art(port.BINDING)
        port.load_stages('old')
    f = pd.read_parquet(port.verify(receipt['prediction']))
    ids = port.template(view)
    np.testing.assert_array_equal(f.mvt_id, ids)
    assert np.isfinite(f.drop(columns='mvt_id').to_numpy()).all()
    # A sequential scalar-column accumulation is deliberately independent of
    # the producer's original full-matrix BLAS reduction. Bound its roundoff,
    # and separately require every serialized value to agree exactly.
    raw71, raw73 = np.zeros(len(ids)), np.zeros(len(ids))
    components = port.binding()['components'] if view == 'final' else None
    for version in identity['geometry71']['versions']:
        key = str(version)
        if view == 'old':
            values = port.vector(identity['old_basis'][key], ids)
        else:
            component = components[key]
            values = port.vector(component['final'], ids, component.get('id_column', 'mvt_id'),
                component.get('value_column', 'submitted'))
        raw71 += values*identity['geometry71']['weights'][key]
        raw73 += values*identity['geometry73']['weights'][key]
    raw73 += f.v72.to_numpy(float)*identity['geometry73']['weights']['72']
    affine_error = {}
    for name, values in [('raw71', raw71), ('raw73', raw73)]:
        error = float(np.max(np.abs(values-f[name])))
        assert error <= 1e-8, (name, error)
        np.testing.assert_array_equal(port.serialized(values), f['v'+name[3:]])
        affine_error[name] = error
    coef = port.read(port.V74/'parameters.json')['full_year']['coefficients']
    assert len(coef) == 12
    # Parse original public CSV fields here rather than calling source helpers.
    daily_source = pd.read_csv(port.base.SOURCE/'all_pre_departure_delays_2026.csv')
    monthly_source = pd.read_csv(port.base.SOURCE/'taxi_out_additional_time_2026.csv')
    checks = {}
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as con:
        con.register('stages', f)
        con.register('daily_source', daily_source)
        con.register('monthly_source', monthly_source)
        for name, path in [('daily', port.inputs.OUT/view/'daily.parquet'),
            ('monthly_rows', port.inputs.OUT/view/'monthly.parquet'),
            ('matches', port.inputs.OUT/view/'matches.parquet'),
            ('days', folder/'days.parquet'), ('monthly_cells', folder/'monthly_cells.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, unique, errors = con.execute('''SELECT count(*), count(distinct s.mvt_id), sum(
            (s.v71 IS DISTINCT FROM round_even(greatest(s.raw71,0),0))::INT+
            (s.v72 IS DISTINCT FROM round_even(greatest(s.raw72,0),0))::INT+
            (s.refined72 IS DISTINCT FROM round_even(greatest(s.raw_refined72,0),0))::INT+
            (s.v73 IS DISTINCT FROM round_even(greatest(s.raw73,0),0))::INT+
            (s.refined73 IS DISTINCT FROM round_even(greatest(s.raw73+?*
                (CAST(s.refined72 AS DOUBLE)-CAST(s.v72 AS DOUBLE)),0),0))::INT+
            (s.full_monthly74 IS DISTINCT FROM greatest(0,2*s.half_monthly74-CAST(s.refined73 AS DOUBLE)))::INT+
            (s.v74 IS DISTINCT FROM round_even(greatest(s.raw74,0),0))::INT+
            (s.refined74 IS DISTINCT FROM round_even(greatest(s.raw_refined74,0),0))::INT)
            FROM stages s JOIN daily d USING(mvt_id) JOIN monthly_rows m USING(mvt_id)''',
            [identity['geometry73']['weights']['72']]).fetchone()
        assert n == unique == len(ids) and errors == 0, (n, unique, errors)
        checks['arithmetic_rows'] = n
        for expression, column in [('s.monthly74', 'raw74'), ('s.full_monthly74', 'raw_refined74')]:
            actual = sql_constraints(con, expression).loc[ids, 'predicted'].to_numpy()
            np.testing.assert_array_equal(actual, f[column])
        checks['independent_matched_and_excluded_rows_per_branch'] = n
        ngroups, bad = con.execute('''WITH actual AS (
            SELECT d.airport,d.day,count(*) n,
                sum(CASE WHEN d.gap<=3600 THEN greatest(0,d.gap-CAST(s.v71 AS DOUBLE)) ELSE 0 END) ordinary,
                count(*) FILTER(WHERE d.gap<=3600 AND d.gap>s.v71) active,
                count(*) FILTER(WHERE d.gap>3600) long_rows
            FROM stages s JOIN daily d USING(mvt_id) GROUP BY d.airport,d.day)
            SELECT count(*),sum((a.n IS DISTINCT FROM d.rows)::INT+
                (abs(a.ordinary-d.prior_ordinary)>1e-8)::INT+
                (a.active IS DISTINCT FROM d.active)::INT+(a.long_rows IS DISTINCT FROM d.long_rows)::INT+
                (d.public_count IS DISTINCT FROM p.FLT_DEP_IFR_2)::INT+
                (abs(d.public_total-60*CAST(p.DLY_ALL_PRE_2 AS DOUBLE))>1e-8)::INT)
            FROM actual a JOIN days d USING(airport,day)
            JOIN daily_source p ON p.APT_ICAO=a.airport AND p.FLT_DATE=a.day''').fetchone()
        expected_groups = con.execute('SELECT count(*) FROM (SELECT DISTINCT airport,day FROM daily)').fetchone()[0]
        log_count, unique_logs = con.execute('SELECT count(*),count(DISTINCT (airport,day)) FROM days').fetchone()
        assert log_count == unique_logs == expected_groups
        assert ngroups == expected_groups and bad == 0, (ngroups, expected_groups, bad)
        checks['independent_daily_source_and_prior_groups'] = ngroups
        monthly_errors = {}
        for branch, parent_column, result_column in [('original', 'v73', 'monthly74'),
            ('refined', 'refined73', 'half_monthly74')]:
            independent = sql_monthly(con, parent_column, coef)
            actual = independent.loc[ids, 'pred'].to_numpy()
            error = float(np.max(np.abs(actual-f[result_column])))
            assert error <= 1e-8, (branch, error)
            cells, cell_bad = con.execute('''SELECT count(*),sum((c.n IS DISTINCT FROM p.n)::INT+
                (abs(c.pred_mean-p.pred_mean)>1e-8)::INT+(abs(c.shift-p.shift)>1e-8)::INT)
                FROM independent_monthly c JOIN monthly_cells p USING(airport,local_month)
                WHERE p.branch=?''', [branch]).fetchone()
            source_cells = con.execute('SELECT count(*) FROM independent_monthly').fetchone()[0]
            saved_cells, unique_cells = con.execute('SELECT count(*),count(DISTINCT (airport,local_month)) FROM monthly_cells WHERE branch=?', [branch]).fetchone()
            assert cells == source_cells == saved_cells == unique_cells and cells > 0 and cell_bad == 0, (branch, cells, cell_bad)
            monthly_errors[branch] = error
        checks['independent_monthly_max_error_sec'] = monthly_errors
        if view == 'old':
            for name, path in [('original72', port.V72/'predictions.parquet'),
                ('original74', port.V74/'predictions.parquet'),
                ('aggregate', port.AGGREGATE/'predictions.parquet')]:
                con.read_parquet(str(path)).create_view(name)
            old_n, old_bad = con.execute('''SELECT count(*),sum(
                (abs(s.raw72-a.pred)>1e-8)::INT+(s.v72 IS DISTINCT FROM a.submitted)::INT+
                (s.monthly74 IS DISTINCT FROM b.monthly)::INT+(s.matched74 IS DISTINCT FROM b.matched_bound)::INT+
                (s.raw74 IS DISTINCT FROM b.pred)::INT+(s.v74 IS DISTINCT FROM b.submitted)::INT+
                (s.raw72 IS DISTINCT FROM g.old72)::INT+(s.raw_refined72 IS DISTINCT FROM g.new72)::INT+
                (s.raw73 IS DISTINCT FROM g.raw73)::INT+(s.refined73 IS DISTINCT FROM g.new73)::INT+
                (s.half_monthly74 IS DISTINCT FROM g.half_monthly)::INT+
                (s.full_monthly74 IS DISTINCT FROM g.full_monthly)::INT+
                (s.raw_refined74 IS DISTINCT FROM g.new74)::INT+(s.refined74 IS DISTINCT FROM g.new74_serialized)::INT)
                FROM stages s JOIN original72 a USING(mvt_id) JOIN original74 b USING(mvt_id)
                JOIN aggregate g USING(mvt_id)''').fetchone()
            assert old_n == len(ids) and old_bad == 0
            checks['independent_old_stage_rows_exact'] = old_n
    report = dict(status='pass', identity=port.art(port.MANIFEST), result=port.art(folder/'result.json'),
        prediction=receipt['prediction'], view=view, rows=len(ids), source=port.art(Path(__file__)),
        independent_SQL_all_rows=True, independent_affine_max_error_sec=affine_error,
        checks=checks, posterior_scope='Original preserved numerical functions; old posterior all-row replay exact, final input/cohort/source checks independent',
        no_model_fit=True, no_upload=True)
    write_json(folder/'audit.json', report)
    print(dict(status='pass', view=view, rows=len(ids), checks=checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'audit'])
    parser.add_argument('--view', choices=['old', 'final'])
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'preflight':
            preflight()
        else:
            assert args.view is not None
            audit(args.view)
