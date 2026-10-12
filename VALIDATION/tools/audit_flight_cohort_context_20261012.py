"""Independent scalar-source and scoped-neighbor audit; no labels or model reads."""
import hashlib
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import flight_cohort_context as builder

OUT = ROOT / 'tmp/research_20261012/error_mechanisms/cohort_input_audit.json'
EXPECTED_FOLDS = {
    '2025-01': ['2025-%02d' % m for m in range(3, 12)],
    '2025-07': ['2025-01', '2025-02', '2025-03', '2025-09',
                '2025-10', '2025-11', '2025-12'],
}


def artifact(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)


def sample_rows(meta, values, months):
    """Fixed source-only sample: airport/month ends and missingness states."""
    chosen = {0, len(meta)-1}
    for airport in np.unique(meta[:, 2]):
        for month in sorted(set(months)):
            rows = np.flatnonzero((meta[:, 2] == airport) & (months == month))
            if not len(rows):
                continue
            order = rows[np.lexsort((meta[rows, 0], meta[rows, 1]))]
            chosen.update(order[[0, len(order)//2, -1]].tolist())
            for mask in [values[rows, 8] == 0, values[rows, 11] == 0,
                         values[rows, 5:8].sum(1) == 0]:
                subset = rows[mask]
                if len(subset):
                    chosen.add(int(subset[0]))
    return np.array(sorted(chosen), dtype=np.int64)


def exact_neighbors(meta, clock, query, reference, sorted_groups):
    key = (int(meta[query, 2]), tuple(reference))
    donor, times = sorted_groups[key]
    t = meta[query, 1]
    lo = np.searchsorted(times, t-1800000, side='left')
    hi = np.searchsorted(times, t+1800000, side='right')
    result = []
    for j in donor[lo:hi]:
        if meta[j, 0] == meta[query, 0]:
            continue
        if meta[query, 5] >= 0 and meta[j, 5] == meta[query, 5] and abs(meta[j, 1]-t) <= 120000:
            continue
        if (meta[query, 3] > 0 and meta[query, 4] > 0 and np.isfinite(clock[query])
                and meta[j, 3] == meta[query, 3] and meta[j, 4] == meta[query, 4]
                and meta[j, 1] == t and clock[j] == clock[query]):
            continue
        result.append(int(j))
    result.sort(key=lambda j: (abs(meta[j, 1]-t), meta[j, 1], meta[j, 0]))
    answer = np.full(16, -1, np.int32)
    answer[:min(16, len(result))] = result[:16]
    return answer


def audit():
    started = time.monotonic()
    identity = builder.guard('train', full=False)
    assert builder.FOLDS == EXPECTED_FOLDS
    folder = builder.OUT / 'train'
    # Verify the complete frozen bank and original narrow source files.
    for rec in identity['outputs'] + identity['sources']:
        assert artifact(ROOT / rec['path']) == rec, rec['path']
    meta = np.load(folder/'meta.npy', mmap_mode='r')
    value = np.load(folder/'values.npy', mmap_mode='r')
    clock = np.load(folder/'clock.npy', mmap_mode='r')
    assert meta.dtype == np.int64 and meta.shape == (identity['rows'], 6)
    assert value.dtype == np.float32 and value.shape == (len(meta), 15)
    assert clock.dtype == np.float64 and clock.shape == (len(meta),)
    assert np.all(meta[1:, 0] > meta[:-1, 0])
    months = pd.to_datetime(meta[:, 1], unit='ms', utc=True).strftime('%Y-%m').to_numpy()
    assert all(x.startswith('2025-') for x in set(months))
    selected = sample_rows(meta, value, months)
    reports = {}
    source_rows = set(selected.tolist())
    for held, train in EXPECTED_FOLDS.items():
        index = np.load(folder/f'neighbors_{held}.npy', mmap_mode='r')
        assert index.shape == (len(meta), 16) and index.dtype == np.int32
        allowed = np.isin(months, train)
        is_held = months == held
        eligible = allowed | is_held
        edges = 0
        for start in range(0, len(meta), 32768):
            stop = min(start+32768, len(meta))
            q = np.arange(start, stop)
            ix = index[start:stop]
            assert ix.min() >= -1 and ix.max() < len(meta)
            ok = ix >= 0
            assert not ok[~eligible[q]].any()
            assert not (ok[:, 1:] & ~ok[:, :-1]).any(), 'Non-prefix padding'
            row, slot = np.nonzero(ok)
            qi, di = q[row], ix[row, slot]
            edges += len(di)
            assert np.all(meta[qi, 0] != meta[di, 0])
            assert np.all(meta[qi, 2] == meta[di, 2])
            dt = np.abs(meta[qi, 1]-meta[di, 1])
            assert np.all(dt <= 1800000)
            assert np.all(allowed[di] | (is_held[qi] & is_held[di]))
            assert not ((meta[qi, 5] >= 0) & (meta[qi, 5] == meta[di, 5]) & (dt <= 120000)).any()
            signature = ((meta[qi, 3] > 0) & (meta[qi, 4] > 0) & np.isfinite(clock[qi])
                & (meta[qi, 3] == meta[di, 3]) & (meta[qi, 4] == meta[di, 4])
                & (meta[qi, 1] == meta[di, 1]) & (clock[qi] == clock[di]))
            assert not signature.any()
        groups = {}
        checks = selected[eligible[selected]]
        for refs in [train, train+[held]]:
            present = np.isin(months, refs)
            for airport in np.unique(meta[checks, 2]):
                rows = np.flatnonzero(present & (meta[:, 2] == airport))
                rows = rows[np.lexsort((meta[rows, 0], meta[rows, 1]))]
                groups[(int(airport), tuple(refs))] = rows, meta[rows, 1]
        for q in checks:
            refs = train+[held] if is_held[q] else train
            expected = exact_neighbors(meta, clock, q, refs, groups)
            np.testing.assert_array_equal(index[q], expected)
            source_rows.update(expected[expected >= 0].tolist())
        reports[held] = dict(all_selected_edges=edges, exact_window_queries=len(checks))
        del groups, index
    rows = np.array(sorted(source_rows))
    ids = pd.DataFrame({'mvt_id': meta[rows, 0]})
    sources = [ROOT/p for p in builder.SOURCES['train']]
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        con.register('bank_keys', pd.DataFrame({'mvt_id':meta[:,0], 'to_ms':meta[:,1], 'clock_ms':clock}))
        all_n, all_unique, all_bad = con.execute('''select count(*),count(distinct q.mvt_id),
            sum(case when b.mvt_id is null or q.to_ms is distinct from b.to_ms
                or q.clock_ms is distinct from b.clock_ms then 1 else 0 end)
            from read_parquet(?) q left join bank_keys b using(mvt_id)''', [str(sources[0])]).fetchone()
        assert all_n == all_unique == len(meta) and all_bad == 0
        con.unregister('bank_keys')
        con.register('audit_ids', ids)
        columns = ','.join('q.'+x for x in builder.SOURCE_COLUMNS)
        raw = con.execute(f'''select {columns},op.op_matched,op.op_ground_sec,
            op.op_own_stand_sec,ab.ab_matched,st.ab_stand_n,st.ab_stand_exit_sec
            from audit_ids i join read_parquet(?) q using(mvt_id)
            left join read_parquet(?) op using(mvt_id)
            left join read_parquet(?) ab using(mvt_id)
            left join read_parquet(?) st using(mvt_id) order by mvt_id''', list(map(str, sources))).df()
    np.testing.assert_array_equal(raw.mvt_id.to_numpy(), meta[rows, 0])
    vocab = identity['vocabulary']
    scalar_values = []
    for r in raw.itertuples(index=False):
        actual_present = not pd.isna(r.clock_ms)
        clocks = [(r.to_ms-r.clock_ms)/1000 if actual_present else np.nan,
                  r.op_ground_sec, r.op_own_stand_sec, r.ab_stand_exit_sec]
        op_known = not pd.isna(r.op_matched) and r.op_matched == 1
        ab_known = (not pd.isna(r.ab_matched) and r.ab_matched == 1
                    and not pd.isna(r.ab_stand_n) and r.ab_stand_n > 0)
        gates = [True, op_known, op_known, ab_known]
        valid = [bool(g) and not pd.isna(x) and np.isfinite(x) and 0 <= x <= 10800
                 for x, g in zip(clocks, gates)]
        numeric = [float(x) if ok else 0. for x, ok in zip(clocks, valid)]
        geometry = [r.stand_lat, r.stand_lon, r.origin_lat, r.origin_lon]
        known = all(not pd.isna(x) and np.isfinite(x) for x in geometry)
        x = 6371000*np.deg2rad(r.stand_lon-r.origin_lon)*np.cos(np.deg2rad(r.origin_lat)) if known else 0.
        y = 6371000*np.deg2rad(r.stand_lat-r.origin_lat) if known else 0.
        scalar_values.append(numeric+valid+[actual_present,x,y,known,
            np.log1p(min(r.raw_count,4096))/np.log(4097),
            r.raw_ground_count/max(r.raw_count,1),r.raw_count>0])
    expected = np.array(scalar_values, np.float32)
    np.testing.assert_array_equal(value[rows], expected)
    np.testing.assert_array_equal(clock[rows], raw.clock_ms.to_numpy(float, na_value=np.nan))
    np.testing.assert_array_equal(meta[rows, 1], raw.to_ms)
    np.testing.assert_array_equal(meta[rows, 5], raw.hex.fillna(-1).to_numpy(np.int64))
    for column, offset in [('airport',2),('stand',3),('runway',4)]:
        keys = raw[column].astype('string').fillna('').tolist()
        mapping = {name:i+1 for i,name in enumerate(vocab[column])}
        np.testing.assert_array_equal(meta[rows, offset], [mapping.get(k,0) for k in keys])
    result = dict(status='pass', scope='CV training bank only; no labels/models/raw points read',
        rows=len(meta), sample_queries=len(selected), source_rows=len(rows),
        all_source_ids_takeoff_original_clock_exact=True,
        independently_recomputed_scalar_values=int(expected.size), folds=reports,
        source=artifact(__file__), builder=artifact(builder.__file__),
        manifest=artifact(folder/'manifest.json'), seconds=time.monotonic()-started)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    audit()
