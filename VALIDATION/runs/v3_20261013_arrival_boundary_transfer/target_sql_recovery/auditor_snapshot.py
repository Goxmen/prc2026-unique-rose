# Frozen independent auditor AST; exactly one explicit SQL AS alias repair.
"""Independent bounded raw-input/target and fixed-model ARR transfer audits.

No producer import, optimizer call, all-year point binary, January/July labels,
ranking input, or candidate construction. `prepare` precedes all four fits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261013_arrival_boundary_transfer'
DAYS = [f'2025-{m:02d}-{d:02d}' for m in [3, 9, 10, 11] for d in [5, 15, 25]]
HALVES = {'early': DAYS[:6], 'late': DAYS[6:]}
DIRECTIONS = {'A': ('early', 'late'), 'B': ('late', 'early')}
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
CHECKED = set()

def read(path):
    return json.loads(Path(path).read_text())

def art(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for value in iter(lambda: f.read(1 << 20), b''):
            h.update(value)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest())

def verify(record):
    path = (ROOT / record['path']).resolve()
    path.relative_to(ROOT)
    key = (str(path), record['sha256'])
    if key not in CHECKED:
        assert art(path) == record, path
        CHECKED.add(key)
    return path

def close(a, b, atol=2e-11, rtol=0):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

def base():
    ident = read(OUT / 'identity.json')
    assert ident['days'] == DAYS and ident['auditor'] == art(__file__)
    for r in ident['sources'] + ident['inherited_sources'] + ident['raw'] + ident['arrivals']:
        verify(r)
    assert {Path(r['path']).name[9:16] for r in ident['raw']} == {'2025-03', '2025-09', '2025-10', '2025-11'}
    for record in ident['chunks'].values():
        verify(record['receipt'])
        for r in record['files']:
            verify(r)
    inputs = read(OUT / 'inputs.json')
    prepared = read(OUT / 'prepared.json')
    assert inputs['identity'] == prepared['identity'] == art(OUT / 'identity.json')
    assert prepared['inputs'] == art(OUT / 'inputs.json')
    assert inputs['features_persisted_before_label_read']
    for p in prepared['packets']:
        for key in ['features', 'times', 'flights', 'targets']:
            verify(p[key])
    verify(inputs['flights'])
    verify(prepared['labels'])
    return (ident, inputs, prepared)

def independent_features(raw, latitude, longitude, airport, phase):
    """Independent point feature algebra; target/event clocks are absent."""
    p = np.asarray(raw, float)
    assert p.ndim == 2 and p.shape[1] == 5 and (len(p) <= 128) and np.isfinite(p).all()
    t = p[:, 0]
    assert len(t) < 2 or (np.diff(t) > 0).all()
    eligible = (p[:, 3] == -123) & (p[:, 4] >= 0)
    if np.count_nonzero(eligible) < 2:
        return (np.empty((0, 22)), np.empty(0, np.int64))
    lat, lon, speed = (p[:, 1] / 1000000.0, p[:, 2] / 1000000.0, p[:, 4] / 10)
    dy = (latitude - lat) * 111320.0
    dx = (longitude - lon) * 111320.0 * np.cos(np.deg2rad((latitude + lat) / 2))
    dist = np.hypot(dx, dy)
    dt = np.r_[0.0, np.diff(t) / 1000]
    paired = np.r_[False, eligible[1:] & eligible[:-1] & (dt[1:] > 0) & (dt[1:] <= 120)]
    dlat = np.deg2rad(lat[:-1]) - np.deg2rad(lat[1:])
    dlon = np.deg2rad(lon[:-1] - lon[1:])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat[1:])) * np.cos(np.deg2rad(lat[:-1])) * np.sin(dlon / 2) ** 2
    displacement = np.r_[0.0, 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))]
    out = np.zeros((len(p), 22))
    out[:, 0] = np.clip(speed / 300, 0, 1)
    out[:, 1] = np.clip(np.log1p(np.maximum(speed, 0)) / np.log(301), 0, 1)
    out[:, 2] = np.exp(-np.maximum(speed, 0) / 5)
    out[:, 3] = np.log1p(np.clip(dist, 0, 40000)) / np.log(40001)
    for j, scale in enumerate([50, 200, 1000], 4):
        out[:, j] = np.exp(-dist / scale)
    out[:, 7] = paired
    out[:, 8] = paired * np.log1p(np.clip(dt, 0, 120)) / np.log(121)
    radial = np.r_[0.0, np.diff(dist)] * (1 if phase == 'DEP' else -1) / np.maximum(dt, 1e-12) / 50
    out[:, 9] = paired * np.clip(radial, -1, 1)
    out[:, 10] = paired * np.clip(displacement / np.maximum(dt, 1e-12) / 50, 0, 1)
    out[:, 11] = paired * np.clip(np.r_[0.0, np.abs(np.diff(speed))] / 50, 0, 1)
    out[:, 12 + AIRPORTS.index(airport)] = 1
    return (out[eligible], t[eligible].astype(np.int64))

def prepare():
    started = time.monotonic()
    ident, inputs, prepared = base()
    c = duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'})
    c.execute("SET TimeZone='UTC'")
    extended = sorted({str((pd.Timestamp(d) + pd.Timedelta(days=k)).date()) for d in DAYS for k in [-1, 0, 1]})
    identity = next((r for r in ident['sources'] if r['path'].endswith('aircraft_rotation_20261010/identity.parquet')))
    anchor = c.execute("SELECT mvt_id,airport,hex,epoch_ms(t)::BIGINT event_ms FROM read_parquet(?)\n        WHERE hex IS NOT NULL AND strftime(t,'%Y-%m-%d') IN (SELECT unnest(?))\n        ORDER BY airport,hex,event_ms,mvt_id", [str(verify(identity)), extended]).df()
    anchors = {(str(a), int(h)): (g.mvt_id.to_numpy(), g.event_ms.to_numpy()) for (a, h), g in anchor.groupby(['airport', 'hex'], sort=False)}
    del anchor
    aq = pd.read_parquet(verify(ident['arrivals'][0]), columns=['mvt_id', 'airport', 'hex', 'landing_ms', 'stand_lat', 'stand_lon', 'day']).sort_values('mvt_id')
    assert aq.mvt_id.is_unique and set(aq.day) == set(DAYS)
    packet_map = {(p['day'], p['phase']): p for p in prepared['packets']}
    assert set(packet_map) == {(d, p) for d in DAYS for p in ['DEP', 'ARR']}
    saved_input_packets = {(p['day'], p['phase']): p for p in inputs['packets']}
    frames = []
    max_feature_error = 0.0
    coverage = []
    bounds = {'early': [], 'late': []}
    for day in DAYS:
        chunk = ident['chunks'][day]
        dep = pd.read_parquet(verify(chunk['files'][0]), columns=['mvt_id', 'airport', 'hex', 'to_ms', 'stand_lat', 'stand_lon', 'start', 'count', 'day']).sort_values('mvt_id')
        assert dep.mvt_id.is_unique and set(dep.day) == {day}
        binary = np.memmap(verify(chunk['files'][1]), mode='r', dtype='<i4').reshape(-1, 6)
        arrival = pd.read_parquet(verify(ident['arrivals'][1 + DAYS.index(day)]), columns=['airport', 'hex', 'ts_ms', 'lat', 'lon', 'alt', 'gs'])
        arrival = arrival.sort_values(['airport', 'hex', 'ts_ms', 'lat', 'lon', 'alt', 'gs'], kind='stable')
        arrival = arrival.drop_duplicates(['airport', 'hex', 'ts_ms'])
        groups = {(str(a), int(h)): g[['ts_ms', 'lat', 'lon', 'alt', 'gs']].to_numpy(np.int64) for (a, h), g in arrival.groupby(['airport', 'hex'], sort=False)}
        del arrival
        for phase, queries in [('DEP', dep), ('ARR', aq.loc[aq.day.eq(day)])]:
            p = packet_map[day, phase]
            old = saved_input_packets[day, phase]
            assert all((p[k] == old[k] for k in old))
            saved = pd.read_parquet(verify(p['flights']))
            x = np.load(verify(p['features']), mmap_mode='r')
            times = np.load(verify(p['times']), mmap_mode='r')
            row = 0
            cursor = 0
            counts = dict(day=day, phase=phase, source_rows=len(queries), missing_identity_or_map=0, other_movement=0, insufficient_points=0)
            for q in queries.itertuples(index=False):
                if pd.isna(q.hex) or not np.isfinite([q.stand_lat, q.stand_lon]).all():
                    counts['missing_identity_or_map'] += 1
                    continue
                event = int(q.to_ms if phase == 'DEP' else q.landing_ms)
                assert str(pd.Timestamp(event, unit='ms', tz='UTC').date()) == day
                ids, ts = anchors[q.airport, int(q.hex)]
                own = ts[ids == int(q.mvt_id)]
                np.testing.assert_array_equal(own, [event])
                left, right = (event - 7200000, event + 60000) if phase == 'DEP' else (event, event + 5400000)
                lo, hi = (np.searchsorted(ts, left, side='left'), np.searchsorted(ts, right, side='right'))
                if np.any(ids[lo:hi] != int(q.mvt_id)):
                    counts['other_movement'] += 1
                    continue
                if phase == 'DEP':
                    assert 0 <= q.count <= 128 and q.start >= 0 and (q.start + q.count <= len(binary))
                    raw = np.asarray(binary[q.start:q.start + q.count], np.int64)
                    raw = np.column_stack([event - raw[:, 0], raw[:, 1:5]])
                else:
                    raw = groups.get((q.airport, int(q.hex)), np.empty((0, 5), np.int64))
                    raw = raw[(raw[:, 0] >= left) & (raw[:, 0] <= right)]
                    if len(raw) > 128:
                        raw = raw[np.linspace(0, len(raw) - 1, 128).astype(np.int64)]
                assert not len(raw) or np.all((raw[:, 0] >= left) & (raw[:, 0] <= right))
                expected_x, expected_time = independent_features(raw, q.stand_lat, q.stand_lon, q.airport, phase)
                if len(expected_x) < 2:
                    counts['insufficient_points'] += 1
                    continue
                got = saved.iloc[row]
                half = 'early' if day in HALVES['early'] else 'late'
                expected = dict(mvt_id=int(q.mvt_id), phase=phase, day=day, airport=q.airport, hex=int(q.hex), event_ms=event, start=cursor, count=len(expected_x), half=half, input_hash=hashlib.sha256(f'arrival_boundary_v1|{int(q.mvt_id)}'.encode()).hexdigest())
                assert all((got[k] == v for k, v in expected.items())), (day, phase, q.mvt_id)
                part = slice(cursor, cursor + len(expected_x))
                error = float(np.max(np.abs(x[part] - expected_x), initial=0))
                max_feature_error = max(max_feature_error, error)
                close(x[part], expected_x)
                np.testing.assert_array_equal(times[part], expected_time)
                cursor += len(expected_x)
                row += 1
            assert row == len(saved) == p['rows'] and cursor == len(x) == len(times) == p['points']
            counts.update(eligible_flights=row, eligible_points=cursor)
            old_count = next((v for v in inputs['coverage'] if v['day'] == day and v['phase'] == phase))
            assert all((old_count[k] == v for k, v in counts.items()))
            coverage.append(counts)
            frames.append(saved)
            if cursor:
                bounds[half].append((int(times.min()), int(times.max())))
        del groups, binary
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 < 2048
    meta = pd.concat(frames, ignore_index=True)
    assert meta.mvt_id.is_unique
    meta['train_selected'] = False
    for _, g in meta.groupby(['half', 'phase']):
        ix = g.sort_values(['input_hash', 'mvt_id'], kind='stable').index[:5000]
        meta.loc[ix, 'train_selected'] = True
    pd.testing.assert_frame_equal(meta.reset_index(drop=True), pd.read_parquet(verify(inputs['flights'])), check_dtype=False)
    ranges = {h: None if not v else [min((t[0] for t in v)), max((t[1] for t in v))] for h, v in bounds.items()}
    assert ranges == inputs['point_time_ranges']
    if all((v is not None for v in ranges.values())):
        assert ranges['early'][1] < ranges['late'][0]
    assert inputs['temporal_range_proves_no_reused_observation']
    truth = c.execute("SELECT MVT_ID_mvt::BIGINT mvt_id,PHASE_mvt phase,\n        epoch_ms(MVT_TIME_UTC_mvt)::BIGINT event_ms,TAXITIME_SEC_mvt::DOUBLE taxi_sec,\n        CASE WHEN PHASE_mvt='DEP' THEN ADEP_mvt ELSE ADES_mvt END airport,\n        strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day FROM read_parquet(?)\n        WHERE PHASE_mvt IN ('DEP','ARR') AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN (SELECT unnest(?))", [[str(verify(r)) for r in ident['raw']], DAYS]).df()
    assert truth.mvt_id.is_unique
    cached = pd.read_parquet(verify(ident['arrivals'][0]), columns=['mvt_id', 'inblock_ms'])
    arrivals = truth.loc[truth.phase.eq('ARR')].merge(cached, on='mvt_id', how='left', validate='one_to_one')
    valid = np.isfinite(arrivals.taxi_sec) & arrivals.taxi_sec.ge(0)
    np.testing.assert_array_equal(arrivals.inblock_ms.notna(), valid)
    close(arrivals.loc[valid, 'inblock_ms'], arrivals.loc[valid, 'event_ms'] + 1000 * arrivals.loc[valid, 'taxi_sec'], atol=1.0)
    labeled = meta.merge(truth, on=['mvt_id', 'phase', 'day', 'airport', 'event_ms'], how='left', validate='one_to_one', indicator=True)
    assert labeled._merge.eq('both').all()
    labeled = labeled.drop(columns='_merge')
    labeled['block_ms'] = labeled.event_ms - 1000 * labeled.taxi_sec
    ix = labeled.phase.eq('ARR')
    labeled.loc[ix, 'block_ms'] = labeled.loc[ix, 'mvt_id'].map(cached.set_index('mvt_id').inblock_ms)
    labeled['label_available'] = np.isfinite(labeled.block_ms)
    pd.testing.assert_frame_equal(labeled, pd.read_parquet(verify(prepared['labels'])), check_dtype=False)
    by_id = labeled.set_index('mvt_id')
    for p in prepared['packets']:
        flights = pd.read_parquet(verify(p['flights']))
        times = np.load(verify(p['times']), mmap_mode='r')
        actual = np.load(verify(p['targets']), mmap_mode='r')
        expected = np.full(len(times), np.nan)
        for f in flights.itertuples():
            if by_id.loc[f.mvt_id, 'label_available']:
                sl = slice(f.start, f.start + f.count)
                z = (times[sl].astype(float) - by_id.loc[f.mvt_id, 'block_ms']) / 30000
                expected[sl] = expit(z if f.phase == 'DEP' else -z)
        close(actual, expected, atol=2e-16)
        assert p['finite_targets'] == np.isfinite(expected).sum()
        assert p['targets_below_001'] == (expected < 0.01).sum() and p['targets_above_099'] == (expected > 0.99).sum()
    counts = []
    passed = True
    for half in HALVES:
        airports = []
        for phase in ['DEP', 'ARR']:
            f = labeled.loc[labeled.half.eq(half) & labeled.phase.eq(phase)]
            supervised = f.loc[f.train_selected & f.label_available]
            counts.append(dict(half=half, phase=phase, eligible=len(f), sampled=int(f.train_selected.sum()), finite_supervised_sample=len(supervised), finite_evaluation=int(f.label_available.sum())))
            passed &= len(supervised) >= 500
            airports.append(set(supervised.airport))
        passed &= len(airports[0] & airports[1]) >= 2
    assert counts == prepared['counts'] and bool(passed) is prepared['coverage_gate']
    if not passed:
        closed = read(OUT / 'closed.json')
        assert closed['state'] == 'inconclusive_insufficient_coverage' and closed['prepared'] == art(OUT / 'prepared.json')
    assert prepared['arrival_raw_negative'] == int(arrivals.taxi_sec.lt(0).sum())
    assert prepared['arrival_raw_nonfinite'] == int((~np.isfinite(arrivals.taxi_sec)).sum())
    assert prepared['arrival_cache_null'] == int(arrivals.inblock_ms.isna().sum())
    c.close()
    save('prepare_audit.json', dict(status='pass', identity=art(OUT / 'identity.json'), prepared=art(OUT / 'prepared.json'), inputs=art(OUT / 'inputs.json'), auditor=art(__file__), flights=len(meta), packets=len(packet_map), max_feature_error=max_feature_error, coverage_gate=bool(passed), counts=counts, seconds=time.monotonic() - started, no_fit=True, no_all_year_point_binary=True, raw_input_and_target_reconstruction=True))

def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k in ['status', 'flights', 'gate', 'seconds']}))

def result():
    started = time.monotonic()
    ident, inputs, prepared = base()
    audit = read(OUT / 'prepare_audit.json')
    assert audit['status'] == 'pass' and audit['identity'] == art(OUT / 'identity.json')
    assert audit['prepared'] == art(OUT / 'prepared.json') and audit['auditor'] == art(__file__)
    fitted = read(OUT / 'fit.json')
    recorded = read(OUT / 'result.json')
    for r in [fitted, recorded]:
        assert r['identity'] == art(OUT / 'identity.json') and r['prepared'] == art(OUT / 'prepared.json')
    assert fitted['prepare_audit'] == art(OUT / 'prepare_audit.json')
    assert fitted['all_four_saved_before_scoring'] and set(fitted['models']) == {'A_control', 'A_arrival', 'B_control', 'B_arrival'}
    assert recorded['fit'] == art(OUT / 'fit.json')
    fit_start = read(OUT / 'fit_started.json')
    score_start = read(OUT / 'score_started.json')
    assert fit_start['identity'] == score_start['identity'] == art(OUT / 'identity.json')
    assert fit_start['prepared'] == art(OUT / 'prepared.json') and fit_start['prepare_audit'] == art(OUT / 'prepare_audit.json')
    verify(fit_start['allocation'])
    assert score_start['fit'] == art(OUT / 'fit.json')
    labels = pd.read_parquet(verify(prepared['labels']))
    expected_rows = []
    models = {}
    max_gradient_error = 0.0
    max_objective_error = 0.0
    for direction, (train_half, held_half) in DIRECTIONS.items():
        selected = labels.loc[labels.half.eq(train_half) & labels.train_selected & labels.label_available]
        depids = sorted(selected.loc[selected.phase.eq('DEP'), 'mvt_id'])
        arrids = sorted(selected.loc[selected.phase.eq('ARR'), 'mvt_id'])
        for arm in ['control', 'arrival']:
            fit = read(verify(fitted['models'][direction + '_' + arm]))
            beta = np.load(verify(fit['beta']))
            assert fit['identity'] == art(OUT / 'identity.json') and fit['prepared'] == art(OUT / 'prepared.json')
            assert fit['prepare_audit'] == art(OUT / 'prepare_audit.json')
            assert fit['direction'] == direction and fit['arm'] == arm and (fit['train_half'] == train_half)
            assert fit['success'] and beta.shape == (23,) and np.isfinite(beta).all()
            assert 0 <= fit['iterations'] <= 200 and fit['evaluations'] >= 1 and ('CONVERGENCE' in fit['message'].upper())
            assert fit['DEP_flights'] == len(depids) and fit['ARR_flights'] == (len(arrids) if arm == 'arrival' else 0)
            for phase, ids in [('DEP', depids), ('ARR', arrids if arm == 'arrival' else [])]:
                assert fit[phase + '_ids_sha256'] == hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()
            total = 0.5 * np.dot(beta[1:], beta[1:])
            gradient = np.r_[0.0, beta[1:]].copy()
            mass = 0.0
            for p in prepared['packets']:
                if p['day'] not in HALVES[train_half] or (arm == 'control' and p['phase'] != 'DEP'):
                    continue
                flights = pd.read_parquet(verify(p['flights']))
                x = np.load(verify(p['features']), mmap_mode='r')
                y = np.load(verify(p['targets']), mmap_mode='r')
                selected_ids = set(depids if p['phase'] == 'DEP' else arrids)
                phase_mass = 1.0 if arm == 'control' else 0.5 if p['phase'] == 'DEP' else 0.5 * len(depids) / len(arrids)
                for f in flights.loc[flights.mvt_id.isin(selected_ids)].itertuples():
                    sl = slice(f.start, f.start + f.count)
                    logits = beta[0] + x[sl] @ beta[1:]
                    total += phase_mass * np.mean(np.logaddexp(0, logits) - y[sl] * logits)
                    residual = (expit(logits) - y[sl]) * phase_mass / f.count
                    gradient[0] += residual.sum()
                    gradient[1:] += x[sl].T @ residual
                    mass += phase_mass
            close(total, fit['objective'], atol=2e-08)
            close(np.max(np.abs(gradient)), fit['gradient_max'], atol=2e-08)
            close(mass, fit['total_weight'], atol=2e-08)
            close(mass, len(depids), atol=2e-08)
            max_objective_error = max(max_objective_error, abs(total - fit['objective']))
            max_gradient_error = max(max_gradient_error, abs(float(np.max(np.abs(gradient))) - fit['gradient_max']))
            models[direction, arm] = beta
        motion = models[direction, 'control'].copy()
        motion[1:13] = models[direction, 'arrival'][1:13]
        for p in prepared['packets']:
            if p['day'] not in HALVES[held_half] or p['phase'] != 'DEP':
                continue
            flights = pd.read_parquet(verify(p['flights']))
            x = np.load(verify(p['features']), mmap_mode='r')
            y = np.load(verify(p['targets']), mmap_mode='r')
            for f in flights.itertuples():
                sl = slice(f.start, f.start + f.count)
                target = y[sl]
                if not np.isfinite(target).all():
                    assert np.isnan(target).all()
                    continue
                value = dict(direction=direction, mvt_id=f.mvt_id, day=f.day, airport=f.airport, points=f.count)
                for arm, beta in [('control', models[direction, 'control']), ('arrival', models[direction, 'arrival']), ('motion', motion)]:
                    logits = beta[0] + x[sl] @ beta[1:]
                    value[arm] = float(np.mean(np.logaddexp(0, logits) - target * logits))
                expected_rows.append(value)
    expected = pd.DataFrame(expected_rows)
    got = pd.read_parquet(verify(recorded['predictions']))
    assert expected.mvt_id.is_unique and set(expected.day) == set(DAYS)
    for name in ['direction', 'mvt_id', 'day', 'airport', 'points']:
        np.testing.assert_array_equal(expected[name], got[name])
    for name in ['control', 'arrival', 'motion']:
        close(expected[name], got[name])
    c = duckdb.connect(config={'threads': 1})
    c.register('losses', expected)
    daily = c.execute('SELECT direction,day,sum(control) control_sum,count(*) control_count,\n        sum(arrival) arrival_sum,count(*) arrival_count,sum(motion) motion_sum,count(*) motion_count\n        FROM losses GROUP BY direction,day ORDER BY direction,day').df()
    saved_daily = pd.read_parquet(verify(recorded['daily']))
    for column in daily:
        if column in ['direction', 'day']:
            np.testing.assert_array_equal(daily[column], saved_daily[column])
        else:
            close(daily[column], saved_daily[column], atol=2e-08)
    primary = True
    motion_gate = True
    metrics = {}
    for direction in DIRECTIONS:
        subset = expected.loc[expected.direction.eq(direction)]
        days = daily.loc[daily.direction.eq(direction)]
        means = {arm: float(subset[arm].mean()) for arm in ['control', 'arrival', 'motion']}
        assert len(days) == 6
        draws = np.random.default_rng(20261013).integers(0, 6, (2000, 6))
        saved_boot = np.load(verify(recorded['bootstraps'][direction]))
        intervals = {}
        for arm in ['arrival', 'motion']:
            delta = (days[arm + '_sum'] - days.control_sum).to_numpy()
            count = days.control_count.to_numpy()
            values = np.array([np.bincount(draw, minlength=6) @ delta / (np.bincount(draw, minlength=6) @ count) for draw in draws])
            close(values, saved_boot[arm])
            intervals[arm] = np.quantile(values, [0.025, 0.975]).tolist()
        relative = (means['control'] - means['arrival']) / means['control']
        p = relative >= 0.05 and intervals['arrival'][1] < 0
        m = means['motion'] < means['control']
        primary &= p
        motion_gate &= m
        report = recorded['metrics'][direction]
        assert report['rows'] == len(subset) and report['primary_pass'] is bool(p) and (report['motion_pass'] is bool(m))
        for arm in means:
            close(means[arm], report['means'][arm])
        for arm in intervals:
            close(intervals[arm], report['ci95'][arm])
        close(relative, report['relative_arrival_improvement'])
        metrics[direction] = dict(means=means, ci95=intervals, relative_improvement=float(relative))
    verify(recorded['airports'])
    expected_airports = expected.groupby(['direction', 'airport'])[['control', 'arrival', 'motion']].agg(['mean', 'size'])
    saved_airports = pd.read_csv(ROOT / recorded['airports']['path'], header=[0, 1], index_col=[0, 1])
    pd.testing.assert_frame_equal(expected_airports, saved_airports, check_dtype=False, check_exact=False, atol=2e-11, rtol=0)
    gate = dict(primary=bool(primary), motion=bool(motion_gate), passed=bool(primary and motion_gate))
    assert recorded['gate'] == gate
    c.close()
    save('independent_audit.json', dict(status='pass', identity=art(OUT / 'identity.json'), prepared=art(OUT / 'prepared.json'), prepare_audit=art(OUT / 'prepare_audit.json'), fit=art(OUT / 'fit.json'), result=art(OUT / 'result.json'), auditor=art(__file__), flights=len(expected), gate=gate, metrics=metrics, maximum_objective_difference=max_objective_error, maximum_gradient_difference=max_gradient_error, seconds=time.monotonic() - started, no_refits=True, all_held_loss_and_bootstrap_values_checked=True))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'result'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
