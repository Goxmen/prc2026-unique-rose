"""Independent raw-input, full-row transport and bounded posterior port audit.

Run ``inputs`` after prepare, then ``outputs --view old|final`` after package.
No model.apply, robust.apply, calibration fit, held labels or CV scores are read.
Final numeric checks reuse frozen support/ordinary moments but independently
assemble memberships, row projections, direct block solves and t5 means.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_joint_public_inference'
ORIGINAL = RUNS / 'v3_20261011_joint_public_moments_submit'
ROBUST = RUNS / 'v3_20261012_joint_public_other_runway_submit'
MODERN = RUNS / 'v3_20261012_final_legacy_modern_replay'
MONTHS = {'old': ['2026-01', '2026-07'], 'final': ['2026-01', '2026-02', '2026-06', '2026-07']}
INPUTS = ['mvt_id', 'month', 'gap', 'prior', 'eligible', 'quality', 'runway',
          'local_day', 'assigned_day', 'fltrule', 'delay_scope']
OUTPUTS = ['mvt_id', 'increment', 'touched', 'robust_delta', 'robust_active', 'extra_cells', 'weight', 'extra_q']
KEYS = ['local_day', 'runway', 'assigned_day']
RAW_SHA = {'old': '9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
           'final': '4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f'}


def read(path):
    return json.loads(Path(path).read_text())


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
        assert path.stat().st_size == record['bytes'], path
    return path


def write(path, value):
    assert not path.exists(), 'Preserve completed audit'
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def guard():
    identity = read(OUT / 'identity.json')
    assert art(Path(__file__)) in identity['sources']
    for record in identity['sources']:
        verify(record)
    assert identity['design']['months'] == MONTHS and identity['design']['input_columns'] == INPUTS
    assert identity['design']['no_fit'] and identity['design']['no_departure_labels']
    params = read(verify(identity['parameters']))
    assert params['allowed'] == [f'2025-{m:02d}' for m in range(1, 13)]
    original_sources = {a['path']: a['sha256'] for a in read(ORIGINAL / 'identity.json')['sources']}
    for field in ['parameters', 'taxi_source', 'delay_source']:
        a = identity[field]
        assert a['sha256'] == original_sources[a['path']]
    prepared = read(OUT / 'prepared.json')
    assert prepared['identity'] == art(OUT / 'identity.json')
    assert prepared['old_inputs_exact'] and prepared['no_departure_labels'] and prepared['no_fit']
    for a in prepared['inputs'].values():
        verify(a)
    return identity, prepared


def inputs():
    started = time.monotonic()
    identity, prepared = guard()
    assert not (OUT / 'input_audit.json').exists()
    csvs = [verify(a) for a in identity['sources'] if a['path'].endswith('/taxi_out_additional_time_2026.csv')]
    assert len(csvs) == 1
    checks = []
    with duckdb.connect(config={'threads': 1, 'memory_limit': '768MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.read_csv(str(csvs[0]), header=True).create_view('source')
        assert con.execute('''SELECT count(*) FROM (SELECT YEAR,MONTH_NUM,APT_ICAO,count(*) n
            FROM source GROUP BY ALL HAVING count(*)<>1)''').fetchone()[0] == 0
        for view in MONTHS:
            path = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
            assert art(path)['sha256'] == RAW_SHA[view]
            con.read_parquet(str(path)).create_view('raw', replace=True)
            con.read_parquet(str(verify(prepared['inputs'][view]))).create_view('i', replace=True)
            con.read_parquet(str(MODERN / f'{view}_v70.parquet')).create_view('v70', replace=True)
            assert [x[0] for x in con.execute('DESCRIBE i').fetchall()] == INPUTS
            con.execute('''CREATE OR REPLACE TEMP VIEW expected AS WITH clocks AS (
                SELECT r.MVT_ID_mvt mvt_id,strftime(r.MVT_TIME_UTC_mvt,'%Y-%m') AS month,
                    epoch(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt) AS gap,cast(v.submitted AS DOUBLE) AS prior,
                    r.STAND_mvt IS NOT NULL AND r.RUNWAY_mvt IS NOT NULL AS eligible,
                    r.RUNWAY_mvt AS runway,strftime(r.MVT_TIME_UTC_mvt AT TIME ZONE 'Europe/Rome','%Y-%m-%d') AS local_day,
                    strftime(to_timestamp(epoch(r.MVT_TIME_UTC_mvt)-v.submitted),'%Y-%m-%d') AS assigned_day,
                    r.FLIGHT_RULE_mvt AS fltrule FROM raw r JOIN v70 v ON r.MVT_ID_mvt=v.mvt_id
                    WHERE r.PHASE_mvt='DEP' AND r.ADEP_mvt='LIRF')
                SELECT c.*,coalesce((cast(s.TF AS DOUBLE)-s.VALID_FL)/s.TF<.05,false) AS quality,
                    substr(assigned_day,1,7)=month AND day(cast(assigned_day AS DATE))>1
                    AND cast(assigned_day AS DATE)<last_day(cast(assigned_day AS DATE))
                    AND coalesce(fltrule<>'V',true) AS delay_scope
                    FROM clocks c LEFT JOIN source s ON s.APT_ICAO='LIRF'
                    AND s.YEAR=cast(substr(local_day,1,4) AS INTEGER)
                    AND s.MONTH_NUM=cast(substr(local_day,6,2) AS INTEGER)''')
            predicate = ' OR '.join(f'i.{col} IS DISTINCT FROM e.{col}' for col in INPUTS[1:])
            n, unique = con.execute('SELECT count(*),count(distinct mvt_id) FROM i').fetchone()
            assert n == unique == con.execute("SELECT count(*) FROM raw WHERE PHASE_mvt='DEP' AND ADEP_mvt='LIRF'").fetchone()[0]
            assert con.execute(f'''SELECT count(*) FROM i FULL JOIN expected e USING(mvt_id)
                WHERE i.mvt_id IS NULL OR e.mvt_id IS NULL OR {predicate}''').fetchone()[0] == 0
            assert sorted(x[0] for x in con.execute('SELECT DISTINCT month FROM i').fetchall()) == MONTHS[view]
            assert con.execute('''SELECT count(*) FROM i WHERE prior<0 OR prior<>round_even(prior,0)
                OR NOT isfinite(prior) OR NOT isfinite(gap) OR prior IS NULL OR gap IS NULL''').fetchone()[0] == 0
            if view == 'old':
                pd.testing.assert_frame_equal(pd.read_parquet(OUT / view / 'inputs.parquet'),
                    pd.read_parquet(ORIGINAL / 'inputs.parquet'), check_exact=True)
            checks.append(dict(view=view, rome_rows=n, raw_values=n*10, all_fields_exact=True,
                utc_prior_assignment_local_takeoff_quality_and_interior_days_exact=True))
    write(OUT / 'input_audit.json', dict(status='pass', source=art(Path(__file__)),
        identity=art(OUT / 'identity.json'), prepared=art(OUT / 'prepared.json'), checks=checks,
        old_packet_exact=True, no_departure_labels_read=True, no_inference_or_fit=True,
        seconds=time.monotonic()-started))
    print(dict(input_audit='pass', checks=checks), flush=True)


def load_inference(view):
    frames = {'joint': [], 'robust': []}
    cells = {'joint': [], 'robust': []}
    refs = []
    if view == 'old':
        for kind, folder in [('joint', ORIGINAL), ('robust', ROBUST)]:
            receipt = read(folder / 'inference.json')
            assert receipt['identity'] == art(folder / 'identity.json')
            for a in receipt['artifacts']:
                verify(a)
            frames[kind].append(pd.read_parquet(folder / 'inference.parquet'))
            cells[kind].append(pd.read_parquet(folder / 'cells.parquet'))
            refs.append(art(folder / 'inference.json'))
    else:
        for month in MONTHS[view]:
            folder = OUT / view / 'months' / month
            for kind in frames:
                receipt = read(folder / f'{kind}.json')
                assert receipt['identity'] == art(OUT / 'identity.json')
                assert receipt['prepared'] == art(OUT / 'prepared.json')
                assert receipt['month'] == month and receipt['kind'] == kind
                f, c = [pd.read_parquet(verify(receipt[k])) for k in ['prediction', 'cells']]
                assert len(f) == receipt['rows'] and f.month.eq(month).all() and c.month.eq(month).all()
                if kind == 'robust':
                    assert receipt['joint'] == art(folder / 'joint.json')
                    raw_receipt = read(verify(receipt['raw']))
                    assert raw_receipt['identity'] == art(OUT / 'identity.json')
                    assert raw_receipt['prepared'] == art(OUT / 'prepared.json') and raw_receipt['month'] == month
                    raw, raw_cells = [pd.read_parquet(verify(raw_receipt[k])) for k in ['prediction', 'cells']]
                    np.testing.assert_array_equal(raw.mvt_id, f.mvt_id)
                    np.testing.assert_array_equal(raw.old_joint, f.recomputed_old_joint)
                    for name in ['touched', 'extra_cells', 'weight', 'extra_q']:
                        np.testing.assert_array_equal(raw[name], f[name])
                    np.testing.assert_array_equal(f.new_joint, np.where(f.extra_cells.gt(0), raw.new_joint, f.old_joint))
                    pd.testing.assert_frame_equal(raw_cells, c.drop(columns='month'), check_exact=True)
                    error = float(np.max(np.abs(f.recomputed_old_joint-f.old_joint)))
                    assert error == receipt['old_joint_max_error_sec'] and error <= 1e-8
                    assert receipt['delta_anchored_to_saved_joint'] and receipt['no_extra_cells_exact_zero']
                frames[kind].append(f)
                cells[kind].append(c)
                refs.append(art(folder / f'{kind}.json'))
    return ({k: pd.concat(v, ignore_index=True) for k, v in frames.items()},
            {k: pd.concat(v, ignore_index=True) for k, v in cells.items()}, refs)


def fixed_keys(keys):
    keys = sorted(set(keys))
    return [keys[k] for k in sorted(set([0, len(keys)//2, len(keys)-1]))] if keys else []


def numeric_checks(inputs, frames, cells):
    """All memberships, then at most six source-selected scalar targets/month."""
    import joint_public_moments as moments
    parameters = read(ORIGINAL / 'parameters.json')
    tx = pd.read_parquet(ORIGINAL / 'taxi_source.parquet').set_index(['local_day', 'runway'])
    ds = pd.read_parquet(ORIGINAL / 'delay_source.parquet').set_index('day')
    assert tx.index.is_unique and ds.index.is_unique
    summaries, scalars = [], []
    for month in MONTHS['final']:
        r = inputs[inputs.month.eq(month)].reset_index(drop=True)
        j = frames['joint'].set_index('mvt_id').loc[r.mvt_id].reset_index()
        b = frames['robust'].set_index('mvt_id').loc[r.mvt_id].reset_index()
        p, gap = r.prior.to_numpy(float), r.gap.to_numpy(float)
        taxi = (r.eligible & r.quality).to_numpy(bool)
        delay = (~r.fltrule.eq('V') & r.delay_scope).to_numpy(bool)
        v, c, prob = moments.normal_moments(p, gap)
        supports = {}
        for i in np.flatnonzero((taxi | delay) & (gap > 3600)):
            y, w, _ = moments.prior.support(gap[i], p[i], False, np.nan, 'LIRF', parameters['support'])
            valid = (y >= 0) & (y <= 7200)
            vector = np.column_stack([valid, np.where(valid, y, 0), np.maximum(gap[i]-y, 0)]).astype(float)
            v[i] = w @ vector
            centered = vector-v[i]
            c[i] = centered.T @ (w[:, None]*centered)
            supports[i] = y, w, vector
        taxi_groups, delay_groups = {}, {}
        for key in sorted(set(zip(r.local_day[taxi], r.runway[taxi]))):
            ix = np.flatnonzero(taxi & r.local_day.eq(key[0]).to_numpy() & r.runway.eq(key[1]).to_numpy())
            if key not in tx.index:
                continue
            src, count = tx.loc[key], float(v[ix, 0].sum())
            if len(ix) >= 20 and src.source_n >= 20 and count >= 20 and .5 <= src.source_n/count <= 1.2:
                taxi_groups[key] = ix
        for day in sorted(set(r.assigned_day[delay])):
            ix = np.flatnonzero(delay & r.assigned_day.eq(day).to_numpy())
            if day in ds.index and len(ix) >= 20 and np.isfinite(ds.loc[day, 'total']+ds.loc[day, 'n_public']):
                delay_groups[day] = ix
        active = taxi & delay & (gap > 3600) & (gap < 50400) & (p > parameters['support']['tau']) & (p < gap)
        touched = np.zeros(len(r), bool)
        groups = {}
        for i in np.flatnonzero(active):
            key = (r.local_day.iloc[i], r.runway.iloc[i], r.assigned_day.iloc[i])
            if key[:2] in taxi_groups and key[2] in delay_groups:
                touched[i] = True
                groups.setdefault(key, []).append(i)
        np.testing.assert_array_equal(touched, j.touched)
        np.testing.assert_array_equal(touched, b.touched)
        jc = cells['joint'][cells['joint'].month.eq(month)].set_index(KEYS)
        bc = cells['robust'][cells['robust'].month.eq(month)].set_index(KEYS)
        assert jc.index.is_unique and bc.index.is_unique
        assert set(jc.index) == set(bc.index) == set(groups)
        extras_for = {}
        for key, targets in groups.items():
            own, day = key[:2], key[2]
            ti, di = taxi_groups[own], delay_groups[day]
            extras = sorted(k for k, ix in taxi_groups.items() if k[0] == own[0] and k != own and np.intersect1d(ix, di).size)
            extras_for[key] = extras
            cc, bb = jc.loc[key], bc.loc[key]
            assert int(cc.taxi_rows) == len(ti) and int(cc.delay_rows) == len(di)
            assert int(cc.shared_rows) == len(np.intersect1d(ti, di))
            np.testing.assert_allclose([cc.taxi_count_mean, cc.taxi_sum_mean, cc.delay_mean],
                [v[ti, 0].sum(), v[ti, 1].sum(), v[di, 2].sum()], atol=1e-7, rtol=1e-13)
            assert np.isfinite(cc.covariance_eigen_min) and cc.covariance_eigen_min > 0
            assert int(bb.extra_cells) == len(extras) and int(bb.targets) == len(targets)
            assert list(bb.extra_runways) == [k[1] for k in extras] and bb.dimension == 2*len(extras)
            np.testing.assert_array_equal(b.extra_cells.iloc[targets], len(extras))
            np.testing.assert_array_equal(b.weight.iloc[targets], bb.weight)
            np.testing.assert_array_equal(b.extra_q.iloc[targets], bb.q)
            expected_weight = min(1., (5.+2*len(extras))/(3.+bb.q)) if extras else 1.
            assert bb.q >= 0 and abs(float(bb.weight)-expected_weight) <= 1e-15
        selected = sorted(set(fixed_keys(groups) + fixed_keys(k for k in groups if extras_for[k])))
        for key in selected:
            own, day = key[:2], key[2]
            extras = extras_for[key]
            runways = [own, *extras]
            di = delay_groups[day]
            dset = set(di)
            dim = 3+2*len(extras)
            mu, cov, observed, loading = np.zeros(dim), np.zeros((dim, dim)), np.zeros(dim), np.zeros(dim)
            offsets = {int(a): (0 if n == 0 else 3+2*(n-1)) for n, k in enumerate(runways) for a in taxi_groups[k]}
            union = sorted(set(offsets) | dset)
            for a in union:
                projection = np.zeros((dim, 3))
                if a in offsets:
                    offset = offsets[a]
                    projection[offset, 0] = projection[offset+1, 1] = 1
                if a in dset:
                    projection[2, 2] = 1
                mu += projection @ v[a]
                cov += projection @ c[a] @ projection.T
            for n, k in enumerate(runways):
                offset = 0 if n == 0 else 3+2*(n-1)
                ix, source = taxi_groups[k], tx.loc[k]
                bias = np.asarray(parameters['taxi']['coefficients'])[0] + len(ix)/100*np.asarray(parameters['taxi']['coefficients'])[1]
                observed[offset:offset+2] = np.array([source.source_n, source.source_n*source.source_mean])+bias
                cov[offset:offset+2, offset:offset+2] += len(ix)*np.asarray(parameters['taxi']['covariance_per_row'])+np.diag([1., 3600.])
                loading[offset+1] = v[ix, 0].sum()
            source = ds.loc[day]
            observed[2] = source.total-parameters['delay']['bias']
            cov[2, 2] += 3*(parameters['delay']['sigma']**2+(parameters['delay']['kappa']*(source.n_public-len(di)))**2)
            loading[2] = -sum(prob[a] for a in di if gap[a] <= 3600)
            cov += 225*np.outer(loading, loading)
            i = min(groups[key], key=lambda a: int(r.mvt_id.iloc[a]))
            assert i in dset and i in taxi_groups[own] and all(i not in taxi_groups[k] for k in extras)
            mean, base_cov = mu[:3]-v[i], cov[:3, :3]-c[i]
            y, w, vector = supports[i]

            def mean_from(mean_, covariance, columns):
                residual = observed[columns]-mean_[columns]-vector[:, columns]
                block = covariance[np.ix_(columns, columns)]
                scale = np.sqrt(np.diag(block))
                z = residual/scale
                standardized = block/np.outer(scale, scale)
                np.linalg.cholesky(standardized)
                quadratic = np.sum(z*np.linalg.solve(.6*standardized, z.T).T, axis=1)
                logw = np.log(np.maximum(w, 1e-300))-.5*(5+len(columns))*np.log1p(np.maximum(quadratic, 0)/5)
                return float(y @ np.exp(logw-logsumexp(logw)))

            original_values = [mean_from(mean, base_cov, [2]), mean_from(mean, base_cov, [0, 1, 2])]
            np.testing.assert_allclose(original_values, [j.dep_only.iloc[i], j.joint.iloc[i]], atol=2e-6, rtol=0)
            q, weight = 0., 1.
            if extras:
                cross, bcov, innovation = cov[:3, 3:], cov[3:, 3:], observed[3:]-mu[3:]
                q = float(innovation @ np.linalg.solve(bcov, innovation))
                weight = min(1., (5+len(innovation))/(3+q))
                mean = mean+weight*cross @ np.linalg.solve(bcov, innovation)
                conditioned = base_cov-weight*cross @ np.linalg.solve(bcov, cross.T)
                posterior = mean_from(mean, conditioned, [0, 1, 2])
            else:
                posterior = original_values[1]
            np.testing.assert_allclose(posterior, b.new_joint.iloc[i], atol=2e-6, rtol=0)
            np.testing.assert_allclose(q, b.extra_q.iloc[i], atol=1e-8, rtol=1e-11)
            np.testing.assert_allclose(weight, b.weight.iloc[i], atol=1e-11, rtol=0)
            scalars.append(dict(month=month, mvt_id=int(r.mvt_id.iloc[i]), key=list(key), extra_cells=len(extras),
                peer_rows=len(union), original_max_error_sec=float(np.max(np.abs(np.array(original_values)-[j.dep_only.iloc[i], j.joint.iloc[i]]))),
                robust_error_sec=abs(posterior-float(b.new_joint.iloc[i])), weight_error=abs(weight-float(b.weight.iloc[i])),
                q_error=abs(q-float(b.extra_q.iloc[i]))))
        summaries.append(dict(month=month, rows=len(r), touched=int(touched.sum()), cohorts=len(groups),
            extra_rows=int(b.extra_cells.gt(0).sum()), scalar_checks=len(selected)))
    return dict(membership_all_rows=summaries, scalar_checks=scalars,
        selection='First/middle/last sorted cohort keys, plus same within nonempty-extra subset; minimum movement ID. No posterior value selection.',
        posterior_atol_sec=2e-6, scope='Independent memberships, projection covariance, direct Schur solves and normalized t5 means; original frozen support/ordinary moments reused. No full posterior rerun.')


def outputs(view):
    started = time.monotonic()
    identity, prepared = guard()
    folder = OUT / view
    assert not (folder / 'audit.json').exists()
    ia = read(OUT / 'input_audit.json')
    assert ia['status'] == 'pass' and ia['identity'] == art(OUT / 'identity.json') and ia['prepared'] == art(OUT / 'prepared.json')
    result = read(folder / 'result.json')
    assert result['identity'] == art(OUT / 'identity.json') and result['prepared'] == art(OUT / 'prepared.json')
    assert result['input_audit'] == art(OUT / 'input_audit.json') and result['view'] == view
    assert result['no_rounding_or_downstream_transport'] and result['no_fit'] and result['no_upload']
    frame = pd.read_parquet(verify(result['prediction']))
    inputs = pd.read_parquet(verify(prepared['inputs'][view]))
    frames, cells, refs = load_inference(view)
    assert result['inference_receipts'] == refs
    ids = pd.read_parquet(ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet'), columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    expected_n = 344841 if view == 'old' else 670790
    assert len(ids) == len(np.unique(ids)) == len(frame) == result['rows'] == expected_n
    np.testing.assert_array_equal(frame.mvt_id, ids)
    assert list(frame.columns) == OUTPUTS and len(inputs) == result['rome_rows']
    for kind, table in frames.items():
        assert table.mvt_id.is_unique and len(table) == len(inputs) and set(table.mvt_id) == set(inputs.mvt_id)
        numeric = ['prior', 'dep_only', 'joint', 'increment'] if kind == 'joint' else ['old_joint', 'new_joint', 'delta', 'weight', 'extra_q']
        assert np.isfinite(table[numeric]).all().all()
    with duckdb.connect(config={'threads': 1, 'memory_limit': '768MB'}) as con:
        for name, table in [('o', frame), ('i', inputs), ('j', frames['joint']), ('b', frames['robust'])]:
            con.register(name, table)
        assert con.execute('''SELECT count(*) FROM i JOIN j USING(mvt_id) JOIN b USING(mvt_id)
            WHERE i.prior IS DISTINCT FROM j.prior OR j.touched IS DISTINCT FROM b.touched
            OR i.month IS DISTINCT FROM j.month OR i.month IS DISTINCT FROM b.month
            OR j.joint IS DISTINCT FROM b.old_joint OR j.increment IS DISTINCT FROM j.joint-j.dep_only
            OR b.delta IS DISTINCT FROM b.new_joint-b.old_joint
            OR j.dep_only<0 OR j.joint<0 OR b.new_joint<0
            OR (NOT j.touched AND (j.increment<>0 OR j.joint<>j.prior OR j.dep_only<>j.prior OR b.delta<>0 OR b.extra_cells<>0))
            OR (b.extra_cells=0 AND (b.delta<>0 OR b.new_joint<>b.old_joint OR b.weight<>1 OR b.extra_q<>0))
            OR b.extra_cells<0 OR b.extra_cells<>trunc(b.extra_cells) OR b.extra_q<0 OR b.weight<=0 OR b.weight>1
            OR abs(b.weight-CASE WHEN b.extra_cells=0 THEN 1 ELSE least(1.,(5.+2*b.extra_cells)/(3.+b.extra_q)) END)>1e-15''').fetchone()[0] == 0
        assert con.execute('''SELECT count(*) FROM o LEFT JOIN j USING(mvt_id) LEFT JOIN b USING(mvt_id)
            WHERE o.increment IS DISTINCT FROM coalesce(j.increment,0)
            OR o.touched IS DISTINCT FROM coalesce(j.touched,false)
            OR o.robust_delta IS DISTINCT FROM coalesce(b.delta,0)
            OR o.robust_active IS DISTINCT FROM coalesce(b.extra_cells>0,false)
            OR o.extra_cells IS DISTINCT FROM coalesce(b.extra_cells,0)
            OR o.weight IS DISTINCT FROM coalesce(b.weight,1)
            OR o.extra_q IS DISTINCT FROM coalesce(b.extra_q,0)''').fetchone()[0] == 0
        if view == 'old':
            con.read_parquet(str(ORIGINAL / 'predictions.parquet')).create_view('saved_j')
            con.read_parquet(str(ROBUST / 'predictions.parquet')).create_view('saved_b')
            assert con.execute('''SELECT count(*) FROM o FULL JOIN saved_j j USING(mvt_id) FULL JOIN saved_b b USING(mvt_id)
                WHERE o.mvt_id IS NULL OR j.mvt_id IS NULL OR b.mvt_id IS NULL
                OR o.increment IS DISTINCT FROM j.increment OR o.touched IS DISTINCT FROM j.touched
                OR o.robust_delta IS DISTINCT FROM b.posterior_delta OR o.robust_active IS DISTINCT FROM b.active''').fetchone()[0] == 0
    if view == 'old':
        assert result['old_saved_posteriors_reused'] and result['no_posterior_rerun_on_old']
        numeric = dict(scope='Old posterior caches and historical numerical audits reused exactly; no repeated old posterior checks.')
    else:
        numeric = numeric_checks(inputs, frames, cells)
    write(folder / 'audit.json', dict(status='pass', source=art(Path(__file__)), identity=art(OUT / 'identity.json'),
        result=art(folder / 'result.json'), prediction=result['prediction'], input_audit=art(OUT / 'input_audit.json'),
        view=view, rows=len(frame), rome_rows=len(inputs), all_row_SQL_increment_membership_and_weight_exact=True,
        all_template_IDs_order_exact=True, old_saved_components_exact=view == 'old', numeric=numeric,
        no_fit_or_departure_labels=True, seconds=time.monotonic()-started))
    print(dict(audit='pass', view=view, rows=len(frame), rome_rows=len(inputs), seconds=time.monotonic()-started), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inputs', 'outputs'])
    parser.add_argument('--view', choices=list(MONTHS))
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'inputs':
            inputs()
        else:
            assert args.view is not None
            outputs(args.view)
