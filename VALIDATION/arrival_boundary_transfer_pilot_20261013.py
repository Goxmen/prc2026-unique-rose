"""Fixed training-date ARR-to-DEP boundary transfer falsifier; no taxi CV.

Stages: freeze, prepare, fit, score. Numerical stages require root allocation.
All four fixed models save before held loss scoring. No download, inference on
ranking, hyperparameter retry, neural training, production or upload is present.
`preflight` uses synthetic arrays only and is safe before numerical allocation.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import resource
import sys
import time

for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_name] = '1'

import duckdb
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import stand_geometry as geometry

REPORT = ROOT / 'VALIDATION/reports/privileged_learning_frontier_20261013'
PLAN = REPORT / 'arrival_boundary_transfer_design.json'
OUT = ROOT / 'VALIDATION/runs/v3_20261013_arrival_boundary_transfer'
AUDITOR = ROOT / 'VALIDATION/tools/audit_arrival_boundary_transfer_20261013.py'
ARR = ROOT / 'VALIDATION/runs/v3_20261010_arrival_clock_context_pilot'
DEP = ROOT / 'data/derived/sparse_track_sequence_20261010'
IDENTITY_CACHE = ROOT / 'data/derived/aircraft_rotation_20261010/identity.parquet'
DAYS = [f'2025-{m:02d}-{d:02d}' for m in [3, 9, 10, 11] for d in [5, 15, 25]]
HALVES = {'early': DAYS[:6], 'late': DAYS[6:]}
DIRECTIONS = {'A': ('early', 'late'), 'B': ('late', 'early')}
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
FEATURES = 22
CAP = 5000


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def art(p):
    p = Path(p).resolve()
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def verify(a):
    p = ROOT / a['path']
    assert art(p) == a, a['path']
    return p


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), f'Refuse overwrite: {p}'
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def peak_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    assert value < 2048, f'2GB peak RSS cap exceeded: {value}'
    return value


def runtime():
    return dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, duckdb=duckdb.__version__,
                machine=platform.machine(), host=platform.node(), threads=1)


def connection():
    return duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'})


def freeze():
    """Metadata and byte identities only; no numerical arrays decoded."""
    plan = read(PLAN)
    review = read(OUT / 'source_review.json')
    assert review['status'] == 'pass'
    assert review['source'] == art(__file__) and review['auditor'] == art(AUDITOR)
    assert review['design'] == art(PLAN)
    for a in plan['sources']:
        verify(a)
    manifest = read(DEP / 'full.json')
    chunks = {}
    for day in DAYS:
        receipt = DEP / 'full_chunks' / f'{day}.json'
        assert sha(receipt) == manifest['chunks'][day]
        r = read(receipt)
        assert r['day'] == day and r['design_sha256'] == sha(DEP / 'full_design.json')
        assert (r['index']['path'].endswith(f'{day}.parquet')
                and r['points']['path'].endswith(f'{day}.bin'))
        # Historical chunk records include bytes; normalize only artifact schema.
        files = [art(ROOT / r[key]['path']) for key in ['index', 'points']]
        for key, a in zip(['index', 'points'], files):
            assert a['sha256'] == r[key]['sha256']
        chunks[day] = {'receipt': art(receipt), 'files': files}
    arrival = read(ARR / 'prepared.json')
    files = [arrival['query']] + [arrival['days'][d]['artifact'] for d in DAYS]
    for a in files:
        verify(a)
    identities = read(ROOT / 'data/derived/aircraft_rotation_20261010/identity.json')
    assert sha(IDENTITY_CACHE) == identities['identity_sha256']
    raw = [ROOT / p for p in plan['inputs']['training_raw_allowlist']]
    recorded = {Path(v['path']).name: v['sha256'] for v in identities['movement_sources']}
    for p in raw:
        assert sha(p) == recorded[p.name]
    sources = [art(__file__), art(AUDITOR), art(PLAN), art(OUT / 'source_review.json'),
               art(REPORT / 'arrival_boundary_transfer_independent_design_review.json'),
               art(REPORT / 'storage_contract.json'), art(REPORT / 'synthetic_preflight.json'),
               art(Path(geometry.__file__)), art(IDENTITY_CACHE)]
    write(OUT / 'identity.json', dict(source=art(__file__), auditor=art(AUDITOR),
          design=art(PLAN), source_review=art(OUT / 'source_review.json'), sources=sources,
          inherited_sources=plan['sources'], chunks=chunks, arrivals=files,
          raw=[art(p) for p in raw], days=DAYS, runtime=runtime(), no_numeric_read_before_freeze=True))
    print(json.dumps({'identity': art(OUT / 'identity.json')}), flush=True)


def guard(stage):
    ident = read(OUT / 'identity.json')
    assert ident['source'] == art(__file__) and ident['design'] == art(PLAN)
    assert ident['auditor'] == art(AUDITOR)
    assert ident['runtime'] == runtime()
    for a in ident['sources'] + ident['inherited_sources']:
        verify(a)
    allocation = read(OUT / 'allocation.json')
    assert allocation['authority'] == '/root'
    assert allocation['identity'] == art(OUT / 'identity.json')
    assert allocation['state'] == 'released_for_fixed_arrival_boundary_transfer_pilot'
    assert stage in allocation['stages'], stage
    assert not (OUT / 'closed.json').exists(), 'Pilot already closed'
    return ident


def hash_id(value):
    return hashlib.sha256(f'arrival_boundary_v1|{int(value)}'.encode()).hexdigest()


def haversine(lat1, lon1, lat2, lon2):
    a, b = np.radians(lat1), np.radians(lat2)
    da, dl = b - a, np.radians(lon2 - lon1)
    h = np.sin(da / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dl / 2) ** 2
    return 2 * 6371000. * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def point_features(raw, event_ms, stand_lat, stand_lon, airport, phase):
    """raw columns timestamp_ms, lat_microdeg, lon_microdeg, altitude, speed_deciknot.

    Returns motion features and separate point timestamps. Target clocks are not
    arguments. Pair adjacency refers to raw retained slots before filtering.
    """
    raw = np.asarray(raw, dtype=np.float64)
    assert raw.ndim == 2 and raw.shape[1] == 5 and len(raw) <= 128
    assert phase in ['DEP', 'ARR'] and airport in AIRPORTS
    assert np.isfinite(raw).all() and np.isfinite([stand_lat, stand_lon]).all()
    t = raw[:, 0]
    assert len(t) == 0 or np.all(np.diff(t) > 0)
    eligible = (raw[:, 3] == -123) & (raw[:, 4] >= 0)
    if eligible.sum() < 2:
        return np.empty((0, FEATURES)), np.empty(0, dtype=np.int64)
    speed = raw[:, 4] / 10
    lat, lon = raw[:, 1] / 1e6, raw[:, 2] / 1e6
    distance = geometry.metres(lat, lon, stand_lat, stand_lon)
    dt = np.r_[0., np.diff(t) / 1000]
    pair = np.r_[False, eligible[:-1] & eligible[1:] & (dt[1:] > 0) & (dt[1:] <= 120)]
    displacement = np.r_[0., haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])]
    sign = 1 if phase == 'DEP' else -1
    radial = np.r_[0., np.diff(distance)] * sign / np.maximum(dt, 1e-12) / 50
    values = [np.clip(speed / 300, 0, 1),
              np.clip(np.log1p(np.maximum(speed, 0)) / np.log1p(300), 0, 1),
              np.exp(-np.maximum(speed, 0) / 5),
              np.log1p(np.clip(distance, 0, 40000)) / np.log1p(40000),
              *[np.exp(-distance / scale) for scale in [50, 200, 1000]],
              pair.astype(float), np.log1p(np.clip(dt, 0, 120)) / np.log1p(120) * pair,
              np.clip(radial, -1, 1) * pair,
              np.clip(displacement / np.maximum(dt, 1e-12) / 50, 0, 1) * pair,
              np.clip(np.r_[0., np.abs(np.diff(speed))] / 50, 0, 1) * pair]
    x = np.column_stack(values)[eligible]
    airport_onehot = np.zeros((len(x), 10))
    airport_onehot[:, AIRPORTS.index(airport)] = 1
    x = np.column_stack([x, airport_onehot])
    assert x.shape[1] == FEATURES and np.isfinite(x).all()
    return x, t[eligible].astype(np.int64)


def boundary_targets(point_ms, block_ms, phase):
    sign = 1 if phase == 'DEP' else -1
    return expit(sign * (np.asarray(point_ms, float) - float(block_ms)) / 30000)


def other_anchor(anchors, own_id, start, end):
    if anchors is None:
        return False
    ids, times = anchors
    lo = np.searchsorted(times, start, side='left')
    hi = np.searchsorted(times, end, side='right')
    return bool(np.any(ids[lo:hi] != own_id))


def input_packets(ident, deadline):
    """Read only fixed-date input columns; write all features before label access."""
    c = connection()
    c.execute("SET TimeZone='UTC'")
    days_extended = sorted({(pd.Timestamp(d) + pd.Timedelta(days=k)).strftime('%Y-%m-%d')
                            for d in DAYS for k in [-1, 0, 1]})
    anchors = c.execute('''SELECT mvt_id,airport,hex,epoch_ms(t)::BIGINT time_ms
        FROM read_parquet(?) WHERE hex IS NOT NULL
        AND strftime(t,'%Y-%m-%d') IN (SELECT unnest(?))
        ORDER BY airport,hex,time_ms,mvt_id''', [str(IDENTITY_CACHE), days_extended]).df()
    anchor_map = {(str(a), int(h)): (g.mvt_id.to_numpy(), g.time_ms.to_numpy())
                  for (a, h), g in anchors.groupby(['airport', 'hex'], sort=False)}
    del anchors
    aq = c.execute('''SELECT mvt_id,airport,hex,landing_ms,stand_lat,stand_lon,day
        FROM read_parquet(?) ORDER BY mvt_id''', [str(verify(ident['arrivals'][0]))]).df()
    assert set(aq.day) == set(DAYS) and aq.mvt_id.is_unique
    packets, allmeta, coverage = [], [], []
    time_ranges = {'early': [], 'late': []}
    preparation_bounds = []
    arrival_metadata = read(ARR / 'prepared.json')
    for day in DAYS:
        assert time.monotonic() <= deadline, 'Fixed five-minute preparation budget exceeded'
        packet = ident['chunks'][day]
        for a in packet['files'] + [packet['receipt']]:
            verify(a)
        index = c.execute('''SELECT mvt_id,airport,hex,to_ms,stand_lat,stand_lon,
            start,count,day FROM read_parquet(?) ORDER BY mvt_id''',
            [str(ROOT / packet['files'][0]['path'])]).df()
        assert set(index.day) == {day} and index.mvt_id.is_unique
        binary = ROOT / packet['files'][1]['path']
        assert binary.stat().st_size % 24 == 0
        declared_points = int(arrival_metadata['days'][day]['points'])
        upper_flights = len(index) + int(aq.day.eq(day).sum())
        # Prospectively fixed conservative buffer allowance, checked before
        # materializing the ARR packet and feature arrays. RSS remains checked.
        estimated_peak = (peak_mb() + 256 +
            (declared_points * (8 * 7 * 8) + upper_flights * 128 * (FEATURES * 8 + 8) * 2 +
             binary.stat().st_size * 4) / 1024 ** 2)
        assert estimated_peak < 2048, f'Preparation memory bound exceeds cap: {estimated_peak}'
        preparation_bounds.append(dict(day=day, conservative_MB=estimated_peak,
                                        declared_arrival_points=declared_points, upper_flights=upper_flights))
        dp = np.memmap(binary, dtype='<i4', mode='r').reshape(-1, 6)
        apath = verify(ident['arrivals'][1 + DAYS.index(day)])
        ap = pd.read_parquet(apath, columns=['airport', 'hex', 'ts_ms', 'lat', 'lon', 'alt', 'gs'])
        assert len(ap) == declared_points
        ap = ap.sort_values(['airport', 'hex', 'ts_ms', 'lat', 'lon', 'alt', 'gs'], kind='stable')
        ap = ap.drop_duplicates(['airport', 'hex', 'ts_ms'])
        ag = {(str(a), int(h)): g[['ts_ms', 'lat', 'lon', 'alt', 'gs']].to_numpy(np.int64)
              for (a, h), g in ap.groupby(['airport', 'hex'], sort=False)}
        del ap
        for phase, queries in [('DEP', index), ('ARR', aq[aq.day.eq(day)])]:
            arrays, timearrays, rows, cursor = [], [], [], 0
            counts = dict(day=day, phase=phase, source_rows=len(queries),
                          missing_identity_or_map=0, other_movement=0, insufficient_points=0)
            for counter, q in enumerate(queries.itertuples(index=False)):
                if counter % 512 == 0:
                    assert time.monotonic() <= deadline, 'Fixed five-minute preparation budget exceeded'
                if pd.isna(q.hex) or not np.isfinite([q.stand_lat, q.stand_lon]).all():
                    counts['missing_identity_or_map'] += 1
                    continue
                event = int(q.to_ms if phase == 'DEP' else q.landing_ms)
                assert pd.Timestamp(event, unit='ms', tz='UTC').strftime('%Y-%m-%d') == day
                own = anchor_map.get((q.airport, int(q.hex)))
                assert own is not None
                own_times = own[1][own[0] == int(q.mvt_id)]
                assert len(own_times) == 1 and own_times[0] == event
                left, right = ((event - 7200000, event + 60000) if phase == 'DEP'
                               else (event, event + 5400000))
                if other_anchor(anchor_map.get((q.airport, int(q.hex))), int(q.mvt_id), left, right):
                    counts['other_movement'] += 1
                    continue
                if phase == 'DEP':
                    p = np.array(dp[int(q.start):int(q.start) + int(q.count)], dtype=np.int64)
                    raw = np.column_stack([event - p[:, 0], p[:, 1:5]])
                else:
                    p = ag.get((q.airport, int(q.hex)), np.empty((0, 5), dtype=np.int64))
                    lo, hi = np.searchsorted(p[:, 0], [left, right], side='left') if len(p) else (0, 0)
                    if len(p):
                        hi = np.searchsorted(p[:, 0], right, side='right')
                    raw = p[lo:hi]
                    if len(raw) > 128:
                        raw = raw[np.linspace(0, len(raw) - 1, 128).astype(np.int64)]
                assert not len(raw) or np.all((raw[:, 0] >= left) & (raw[:, 0] <= right))
                x, times = point_features(raw, event, q.stand_lat, q.stand_lon, q.airport, phase)
                if len(x) < 2:
                    counts['insufficient_points'] += 1
                    continue
                arrays.append(x); timearrays.append(times)
                rows.append(dict(mvt_id=int(q.mvt_id), phase=phase, day=day, airport=q.airport,
                                 hex=int(q.hex), event_ms=event, start=cursor, count=len(x),
                                 input_hash=hash_id(q.mvt_id), half='early' if day in DAYS[:6] else 'late'))
                cursor += len(x)
            folder = OUT / 'inputs' / day / phase
            folder.mkdir(parents=True, exist_ok=False)
            x = np.concatenate(arrays) if arrays else np.empty((0, FEATURES))
            times = np.concatenate(timearrays) if timearrays else np.empty(0, dtype=np.int64)
            if len(times):
                time_ranges['early' if day in DAYS[:6] else 'late'].append([int(times.min()), int(times.max())])
            frame = pd.DataFrame(rows, columns=['mvt_id', 'phase', 'day', 'airport', 'hex',
                             'event_ms', 'start', 'count', 'input_hash', 'half'])
            np.save(folder / 'features.npy', x, allow_pickle=False)
            np.save(folder / 'point_ms.npy', times, allow_pickle=False)
            frame.to_parquet(folder / 'flights.parquet', index=False)
            counts.update(eligible_flights=len(frame), eligible_points=len(x), peak_rss_MB=peak_mb())
            coverage.append(counts)
            packets.append(dict(day=day, phase=phase, rows=len(frame), points=len(x),
                      features=art(folder / 'features.npy'), times=art(folder / 'point_ms.npy'),
                      flights=art(folder / 'flights.parquet')))
            allmeta.append(frame)
            del arrays, timearrays, x, times
        del ag, dp
    c.close()
    meta = pd.concat(allmeta, ignore_index=True)
    assert meta.mvt_id.is_unique
    meta['train_selected'] = False
    for _, group in meta.groupby(['half', 'phase'], sort=True):
        picked = group.sort_values(['input_hash', 'mvt_id']).head(CAP).index
        meta.loc[picked, 'train_selected'] = True
    meta.to_parquet(OUT / 'input_flights.parquet', index=False)
    ranges = {k: None if not v else [min(a[0] for a in v), max(a[1] for a in v)]
              for k, v in time_ranges.items()}
    if all(v is not None for v in ranges.values()):
        assert ranges['early'][1] < ranges['late'][0], 'Cross-half observation overlap'
    write(OUT / 'inputs.json', dict(identity=art(OUT / 'identity.json'),
          packets=packets, flights=art(OUT / 'input_flights.parquet'), coverage=coverage,
          features_persisted_before_label_read=True, point_time_ranges=ranges,
          temporal_range_proves_no_reused_observation=True, preparation_bounds=preparation_bounds,
          peak_rss_MB=peak_mb()))
    return meta, packets


def prepare():
    ident = guard('prepare')
    assert not (OUT / 'prepare_started.json').exists(), 'Preserve partial preparation; no automatic restart'
    write(OUT / 'prepare_started.json', dict(identity=art(OUT / 'identity.json'),
          allocation=art(OUT / 'allocation.json'), utc=pd.Timestamp.now(tz='UTC').isoformat()))
    started = time.monotonic()
    deadline = started + 300
    meta, packets = input_packets(ident, deadline)
    c = connection(); c.execute("SET TimeZone='UTC'")
    rawpaths = [str(verify(a)) for a in ident['raw']]
    truth = c.execute('''SELECT MVT_ID_mvt::BIGINT mvt_id,PHASE_mvt phase,
        epoch_ms(MVT_TIME_UTC_mvt)::BIGINT event_ms,TAXITIME_SEC_mvt::DOUBLE taxi_sec,
        CASE WHEN PHASE_mvt='DEP' THEN ADEP_mvt ELSE ADES_mvt END airport,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') day
        FROM read_parquet(?) WHERE PHASE_mvt IN ('DEP','ARR')
        AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN (SELECT unnest(?))''', [rawpaths, DAYS]).df()
    assert truth.mvt_id.is_unique and set(truth.day) == set(DAYS)
    joined = meta.merge(truth, on=['mvt_id', 'phase', 'day', 'airport', 'event_ms'],
                        how='left', validate='one_to_one', indicator=True)
    assert joined._merge.eq('both').all()
    cached = pd.read_parquet(verify(ident['arrivals'][0]), columns=['mvt_id', 'inblock_ms'])
    arr_truth = truth[truth.phase.eq('ARR')].merge(cached, on='mvt_id', validate='one_to_one', how='left')
    expected = np.isfinite(arr_truth.taxi_sec) & arr_truth.taxi_sec.ge(0)
    assert np.array_equal(arr_truth.inblock_ms.notna().to_numpy(), expected.to_numpy())
    error = np.abs(arr_truth.loc[expected, 'inblock_ms'] -
                   (arr_truth.loc[expected, 'event_ms'] + 1000 * arr_truth.loc[expected, 'taxi_sec']))
    assert len(error) == 0 or error.max() <= 1.  # declared 1 millisecond serialization envelope
    by_id = cached.set_index('mvt_id').inblock_ms
    joined['block_ms'] = joined.event_ms - 1000 * joined.taxi_sec
    isarr = joined.phase.eq('ARR')
    joined.loc[isarr, 'block_ms'] = joined.loc[isarr, 'mvt_id'].map(by_id)
    joined['label_available'] = np.isfinite(joined.block_ms)
    labelpath = OUT / 'labels.parquet'
    joined.drop(columns='_merge').to_parquet(labelpath, index=False)
    labeled = joined.set_index('mvt_id')
    supervised = []
    for packet in packets:
        assert time.monotonic() <= deadline, 'Fixed five-minute preparation budget exceeded'
        frame = pd.read_parquet(verify(packet['flights']))
        times = np.load(verify(packet['times']), allow_pickle=False, mmap_mode='r')
        target = np.full(len(times), np.nan)
        for row in frame.itertuples():
            label = labeled.loc[row.mvt_id]
            if label.label_available:
                sl = slice(int(row.start), int(row.start + row.count))
                target[sl] = boundary_targets(times[sl], label.block_ms, row.phase)
        path = (ROOT / packet['features']['path']).parent / 'targets.npy'
        np.save(path, target, allow_pickle=False)
        supervised.append(dict(**packet, targets=art(path),
            finite_targets=int(np.isfinite(target).sum()),
            targets_below_001=int(np.sum(target < .01)),
            targets_above_099=int(np.sum(target > .99))))
    counts, passed = [], True
    for half in HALVES:
        airports = []
        for phase in ['DEP', 'ARR']:
            selected = joined[joined.half.eq(half) & joined.phase.eq(phase)]
            valid = selected[selected.train_selected & selected.label_available]
            counts.append(dict(half=half, phase=phase, eligible=len(selected),
                 sampled=int(selected.train_selected.sum()), finite_supervised_sample=len(valid),
                 finite_evaluation=int(selected.label_available.sum())))
            passed &= len(valid) >= 500
            airports.append(set(valid.airport))
        passed &= len(airports[0] & airports[1]) >= 2
    c.close()
    write(OUT / 'prepared.json', dict(identity=art(OUT / 'identity.json'), inputs=art(OUT / 'inputs.json'),
          labels=art(labelpath), packets=supervised, counts=counts, coverage_gate=bool(passed),
          arrival_raw_negative=int(arr_truth.taxi_sec.lt(0).sum()),
          arrival_raw_nonfinite=int((~np.isfinite(arr_truth.taxi_sec)).sum()),
          arrival_cache_null=int(arr_truth.inblock_ms.isna().sum()), peak_rss_MB=peak_mb(),
          elapsed_seconds=time.monotonic() - started))
    if not passed:
        write(OUT / 'closed.json', dict(state='inconclusive_insufficient_coverage',
                                      prepared=art(OUT / 'prepared.json')))
    print(json.dumps({'prepared': art(OUT / 'prepared.json'), 'coverage_gate': bool(passed)}), flush=True)


def load_prepared():
    p = read(OUT / 'prepared.json')
    assert p['identity'] == art(OUT / 'identity.json')
    verify(p['inputs']); verify(p['labels'])
    assert p['coverage_gate'] is True
    return p


def training_arrays(prepared, half, phase):
    labels = pd.read_parquet(verify(prepared['labels']))
    ids = set(labels.loc[labels.half.eq(half) & labels.phase.eq(phase) &
                        labels.train_selected & labels.label_available, 'mvt_id'])
    xx, yy, ww = [], [], []
    for packet in prepared['packets']:
        if packet['day'] not in HALVES[half] or packet['phase'] != phase:
            continue
        rows = pd.read_parquet(verify(packet['flights']))
        x = np.load(verify(packet['features']), allow_pickle=False, mmap_mode='r')
        y = np.load(verify(packet['targets']), allow_pickle=False, mmap_mode='r')
        for r in rows[rows.mvt_id.isin(ids)].itertuples():
            sl = slice(r.start, r.start + r.count)
            xx.append(np.array(x[sl])); yy.append(np.array(y[sl]))
            ww.append(np.full(r.count, 1 / r.count))
    x, y, w = np.concatenate(xx), np.concatenate(yy), np.concatenate(ww)
    assert len(ids) >= 500 and len(ids) <= CAP and len(x) <= CAP * 128
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert abs(w.sum() - len(ids)) < 1e-8
    peak_mb()
    return x, y, w, sorted(ids)


def objective(beta, x, y, w):
    logits = beta[0] + x @ beta[1:]
    loss = np.dot(w, np.logaddexp(0, logits) - y * logits) + .5 * np.dot(beta[1:], beta[1:])
    residual = w * (expit(logits) - y)
    gradient = np.r_[residual.sum(), x.T @ residual + beta[1:]]
    return float(loss), gradient


def fit():
    guard('fit'); prepared = load_prepared()
    preparation_audit = read(OUT / 'prepare_audit.json')
    assert preparation_audit['status'] == 'pass'
    assert preparation_audit['identity'] == art(OUT / 'identity.json')
    assert preparation_audit['prepared'] == art(OUT / 'prepared.json')
    audit_record = art(OUT / 'prepare_audit.json')
    assert not (OUT / 'fit_started.json').exists(), 'No automatic fit restart'
    write(OUT / 'fit_started.json', dict(identity=art(OUT / 'identity.json'),
          prepared=art(OUT / 'prepared.json'), prepare_audit=audit_record,
          allocation=art(OUT / 'allocation.json')))
    started = time.monotonic()
    deadline = started + 300
    models = {}
    for direction, (train, _) in DIRECTIONS.items():
        dx, dy, dw, dids = training_arrays(prepared, train, 'DEP')
        ax, ay, aw, aids = training_arrays(prepared, train, 'ARR')
        for arm in ['control', 'arrival']:
            folder = OUT / 'models' / direction / arm
            folder.mkdir(parents=True, exist_ok=False)
            if arm == 'control':
                x, y, w = dx, dy, dw
            else:
                x, y = np.concatenate([dx, ax]), np.concatenate([dy, ay])
                w = np.concatenate([.5 * dw, .5 * len(dids) / len(aids) * aw])
            evaluations = 0
            def fn(beta):
                nonlocal evaluations
                if time.monotonic() > deadline:
                    np.save(folder / 'budget_stop_beta.npy', beta, allow_pickle=False)
                    raise TimeoutError('Fixed five-minute fit budget exceeded; no retry')
                evaluations += 1
                return objective(beta, x, y, w)
            try:
                result = minimize(fn, np.zeros(FEATURES + 1), jac=True, method='L-BFGS-B',
                         options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
                # Preserve numerical state before checking convergence or scoring.
                np.save(folder / 'beta.npy', result.x, allow_pickle=False)
                value, gradient = objective(result.x, x, y, w)
                record = dict(identity=art(OUT / 'identity.json'), prepared=art(OUT / 'prepared.json'),
                    prepare_audit=audit_record,
                    direction=direction, arm=arm, train_half=train, beta=art(folder / 'beta.npy'),
                    DEP_ids_sha256=hashlib.sha256(np.asarray(dids, dtype='<i8').tobytes()).hexdigest(),
                    ARR_ids_sha256=hashlib.sha256(np.asarray(aids if arm == 'arrival' else [], dtype='<i8').tobytes()).hexdigest(),
                    DEP_flights=len(dids), ARR_flights=len(aids) if arm == 'arrival' else 0,
                    total_weight=float(w.sum()), success=bool(result.success), message=str(result.message),
                    iterations=int(result.nit), evaluations=evaluations, objective=value,
                    gradient_max=float(np.max(np.abs(gradient))), peak_rss_MB=peak_mb())
                write(folder / 'fit.json', record)
                assert result.success and np.isfinite(result.x).all(), record
                models[f'{direction}_{arm}'] = art(folder / 'fit.json')
            except BaseException as exc:
                write(OUT / 'closed.json', dict(state='fit_failed_no_retry', direction=direction,
                                              arm=arm, error=repr(exc)))
                raise
            if arm == 'arrival':
                del x, y, w
        del dx, dy, dw, ax, ay, aw
    write(OUT / 'fit.json', dict(identity=art(OUT / 'identity.json'), prepared=art(OUT / 'prepared.json'),
                               prepare_audit=audit_record, models=models, all_four_saved_before_scoring=True,
                               elapsed_seconds=time.monotonic() - started))
    print(json.dumps({'fit': art(OUT / 'fit.json')}), flush=True)


def paired_bootstrap(frame, seed=20261013):
    daily = frame.groupby('day', sort=True)[['control', 'arrival', 'motion']].agg(['sum', 'count'])
    assert len(daily) == 6
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(daily), size=(2000, len(daily)))
    totals = daily[('control', 'count')].to_numpy()[draws].sum(1)
    result = {}
    for name in ['arrival', 'motion']:
        delta = daily[(name, 'sum')].to_numpy() - daily[('control', 'sum')].to_numpy()
        result[name] = delta[draws].sum(1) / totals
    return result


def score():
    guard('score'); prepared = load_prepared()
    fitted = read(OUT / 'fit.json')
    assert fitted['identity'] == art(OUT / 'identity.json')
    assert set(fitted['models']) == {'A_control', 'A_arrival', 'B_control', 'B_arrival'}
    assert fitted['prepared'] == art(OUT / 'prepared.json')
    assert fitted['prepare_audit'] == art(OUT / 'prepare_audit.json')
    assert fitted['all_four_saved_before_scoring'] is True
    deadline = time.monotonic() + max(0., 300 - fitted['elapsed_seconds'])
    for a in fitted['models'].values():
        r = read(verify(a)); assert r['success'] is True; verify(r['beta'])
    write(OUT / 'score_started.json', dict(identity=art(OUT / 'identity.json'), fit=art(OUT / 'fit.json')))
    rows = []
    for direction, (_, held) in DIRECTIONS.items():
        betas = {arm: np.load(verify(read(verify(fitted['models'][f'{direction}_{arm}']))['beta']),
                                   allow_pickle=False) for arm in ['control', 'arrival']}
        betas['motion'] = betas['control'].copy()
        betas['motion'][1:13] = betas['arrival'][1:13]
        for packet in prepared['packets']:
            if packet['phase'] != 'DEP' or packet['day'] not in HALVES[held]:
                continue
            flights = pd.read_parquet(verify(packet['flights']))
            x = np.load(verify(packet['features']), allow_pickle=False, mmap_mode='r')
            y = np.load(verify(packet['targets']), allow_pickle=False, mmap_mode='r')
            for f in flights.itertuples():
                assert time.monotonic() <= deadline, 'Fixed five-minute fit+score budget exceeded'
                sl = slice(f.start, f.start + f.count)
                target = y[sl]
                if not np.isfinite(target).all():
                    assert np.isnan(target).all()
                    continue
                item = dict(direction=direction, mvt_id=f.mvt_id, day=f.day, airport=f.airport,
                            points=f.count)
                for arm, beta in betas.items():
                    logits = beta[0] + x[sl] @ beta[1:]
                    item[arm] = float(np.mean(np.logaddexp(0, logits) - target * logits))
                rows.append(item)
    frame = pd.DataFrame(rows)
    assert frame.mvt_id.is_unique and set(frame.day) == set(DAYS)
    frame.to_parquet(OUT / 'flight_losses.parquet', index=False)
    daily = frame.groupby(['direction', 'day'], sort=True)[['control', 'arrival', 'motion']].agg(['sum', 'count'])
    daily.columns = ['_'.join(c) for c in daily.columns]
    daily.reset_index().to_parquet(OUT / 'daily_losses.parquet', index=False)
    metrics, gate_primary, gate_motion = {}, True, True
    for direction in DIRECTIONS:
        f = frame[frame.direction.eq(direction)]
        means = {arm: float(f[arm].mean()) for arm in ['control', 'arrival', 'motion']}
        boot = paired_bootstrap(f)
        interval = {arm: np.quantile(v, [.025, .975]).tolist() for arm, v in boot.items()}
        np.savez(OUT / f'bootstrap_{direction}.npz', **boot)
        relative = (means['control'] - means['arrival']) / means['control']
        primary = relative >= .05 and interval['arrival'][1] < 0
        motion = means['motion'] < means['control']
        gate_primary &= primary; gate_motion &= motion
        metrics[direction] = dict(rows=len(f), means=means, relative_arrival_improvement=relative,
                                  ci95=interval, primary_pass=bool(primary), motion_pass=bool(motion))
    frame.groupby(['direction', 'airport'])[['control', 'arrival', 'motion']].agg(['mean', 'size']).to_csv(OUT / 'airport_losses.csv')
    result = dict(identity=art(OUT / 'identity.json'), prepared=art(OUT / 'prepared.json'),
        fit=art(OUT / 'fit.json'), predictions=art(OUT / 'flight_losses.parquet'),
        daily=art(OUT / 'daily_losses.parquet'), airports=art(OUT / 'airport_losses.csv'),
        bootstraps={d: art(OUT / f'bootstrap_{d}.npz') for d in DIRECTIONS}, metrics=metrics,
        gate=dict(primary=bool(gate_primary), motion=bool(gate_motion), passed=bool(gate_primary and gate_motion)),
        interpretation='Matched training-date boundary transfer only; no selected-stack residual headroom or taxi CV claim.',
        peak_rss_MB=peak_mb())
    write(OUT / 'result.json', result)
    print(json.dumps({'result': art(OUT / 'result.json'), 'metrics': metrics, 'gate': result['gate']}), flush=True)


def preflight():
    """Synthetic-only boundary, adjacency, objective, weighting and bootstrap tests."""
    rng = np.random.Generator(np.random.PCG64(20261013))
    raw = np.array([[0, 45000000, 7000000, -123, 0],
                    [30000, 45000100, 7000000, -123, 20],
                    [60000, 45000200, 7000000, 100, 30],
                    [90000, 45000300, 7000000, -123, 40],
                    [120000, 45000400, 7000000, -123, 50]], dtype=float)
    xd, t = point_features(raw, 0, 45., 7., 'EDDF', 'DEP')
    xa, ta = point_features(raw, 0, 45., 7., 'EDDF', 'ARR')
    assert xd.shape == (4, 22) and np.array_equal(t, ta)
    assert np.array_equal(xd[:, :9], xa[:, :9]) and np.array_equal(xd[:, 10:], xa[:, 10:])
    assert np.array_equal(xd[:, 9], -xa[:, 9])
    assert xd[2, 7] == 0 and xd[3, 7] == 1  # cannot bridge observed airborne slot
    for block in [-1e10, 60000., 1e10]:
        old = xd.copy()
        labels = boundary_targets(t, block, 'DEP')
        assert np.array_equal(old, point_features(raw, 0, 45., 7., 'EDDF', 'DEP')[0])
        np.testing.assert_allclose(labels + boundary_targets(t, block, 'ARR'), 1, atol=2e-16)
    assert boundary_targets([60000], 60000, 'DEP')[0] == .5
    assert other_anchor((np.array([1, 2]), np.array([0, 30000])), 1, 0, 40000)
    assert not other_anchor((np.array([1]), np.array([0])), 1, 0, 40000)
    x = rng.normal(size=(63, FEATURES)); y = expit(rng.normal(size=63))
    w = rng.uniform(.01, .2, size=63); beta = rng.normal(size=FEATURES + 1) * .1
    _, gradient = objective(beta, x, y, w)
    for j in range(len(beta)):
        offset = np.zeros_like(beta); offset[j] = 1e-6
        numerical = (objective(beta + offset, x, y, w)[0] - objective(beta - offset, x, y, w)[0]) / 2e-6
        np.testing.assert_allclose(gradient[j], numerical, atol=1e-8, rtol=1e-7)
    f = pd.DataFrame({'day': np.repeat(np.arange(6), np.arange(1, 7)),
                      'control': 2., 'arrival': 1.8, 'motion': 1.9})
    b = paired_bootstrap(f)
    np.testing.assert_allclose(b['arrival'], -.2, atol=1e-15)
    np.testing.assert_allclose(b['motion'], -.1, atol=1e-15)
    assert np.isclose(.5 * 5000 + .5 * 5000 / 2300 * 2300, 5000)
    shifted = raw.copy(); shifted[:, 0] += 123456789
    np.testing.assert_array_equal(xd, point_features(shifted, 123456789, 45., 7., 'EDDF', 'DEP')[0])
    write(REPORT / 'synthetic_preflight.json', dict(status='pass', source=art(__file__),
          design=art(PLAN), synthetic_only=True, gradient_coordinates=len(beta),
          label_poison_has_no_input_effect=True, opposite_phase_targets=True,
          no_airborne_slot_bridging=True, fixed_direction_only=True,
          absolute_time_shift_invariant=True, fixed_augmentation_total_mass=True,
          clustered_bootstrap_constant_delta=True, no_real_array_access=True))
    print(json.dumps({'fixture': art(REPORT / 'synthetic_preflight.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'prepare', 'fit', 'score', 'preflight'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
