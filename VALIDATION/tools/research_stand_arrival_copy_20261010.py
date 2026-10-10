"""One-core, training-only falsifier for stand/month arrival recording state.

No departure held labels, full ordinary matrices, fits, raw positions or network
reads. Original movement-airline parsing is reused exactly. Two inputs frozen:
sum(arrival_copy - monthly_airline_prior)/(stand_count+50), stand_count.
"""
import ast
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp/research_20261010/error_mechanisms/stand_arrival_copy'
PACKETS = ROOT / 'VALIDATION/runs/v3_20261010_rome_own_arrival_jan_jul/inputs'
ROT = ROOT / 'data/derived/aircraft_rotation_20261010'
FOLDS = {'2025-01': [f'2025-{m:02d}' for m in range(3, 12)],
         '2025-07': ['2025-01', '2025-02', '2025-03', '2025-09', '2025-10', '2025-11', '2025-12']}
DESIGN = dict(group='airport LIRF x stand x UTC month', match_tolerance_sec=60,
    airline_prior='same UTC month original movement-airline share with K10 airport prior',
    feature='sum(arrival_copy - airline_prior)/(stand_arrival_count+50)',
    support='stand_arrival_count', stand_prior_weight=50,
    absent_known_stand='residual0,count0', missing_stand='both NaN',
    phases='arrival labels supplied as observable context; departure labels training-only',
    training_scope=FOLDS, held_departure_data_used=False,
    association='demean both target and residual within original airline x month; Pearson and slope',
    temporal='per month and first/second chronological half of each outer training window',
    no_fitting=True, no_threshold_search=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def airline_parser():
    source = ROOT / 'VALIDATION/rome_signals.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'airline_code')
    namespace = dict(pd=pd)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['airline_code']


def features(arrivals, queries):
    a = arrivals.copy()
    apt = a.groupby('month', observed=True)['copy'].mean()
    by = a.groupby(['month', 'airline'], observed=True)['copy'].agg(['sum', 'size'])
    key = pd.MultiIndex.from_frame(a[['month', 'airline']])
    sums = by['sum'].reindex(key).fillna(0).to_numpy()
    counts = by['size'].reindex(key).fillna(0).to_numpy()
    a['prior'] = (sums + 10*a.month.map(apt).to_numpy())/(counts+10)
    a['residual'] = a['copy']-a.prior
    valid = a.stand.notna() & a.airline.notna()
    grouped = a.loc[valid].groupby(['month', 'stand'], observed=True).residual.agg(['sum', 'size'])
    keys = pd.MultiIndex.from_frame(queries[['month', 'stand']])
    z = grouped.reindex(keys)
    count = z['size'].fillna(0).to_numpy(dtype=float, copy=True)
    delta = z['sum'].fillna(0).to_numpy(float)/(count+50)
    missing = queries.stand.isna().to_numpy()
    delta[missing], count[missing] = np.nan, np.nan
    return delta, count, dict(arrivals=len(a), arrival_copy_rate=float(a['copy'].mean()),
        arrival_stand_known=int(a.stand.notna().sum()), arrival_airline_known=int(a.airline.notna().sum()),
        eligible_stand_arrivals=int(valid.sum()), stand_month_cells=len(grouped))


def association(frame):
    valid = frame.delta.notna() & frame.airline.notna()
    f = frame.loc[valid, ['airline', 'month', 'target', 'delta']].copy()
    means = f.groupby(['airline', 'month'], observed=True)[['target', 'delta']].transform('mean')
    y, x = (f[['target', 'delta']]-means).to_numpy(float).T
    xx, yy, xy = float(x@x), float(y@y), float(x@y)
    return dict(rows=len(frame), feature_available=int(valid.sum()),
        stand_support_positive=int(frame.support.gt(0).sum()),
        stand_support_50plus=int(frame.support.ge(50).sum()),
        support_median=float(frame.support.median()) if frame.support.notna().any() else None,
        delta_rms=float(np.sqrt(np.nanmean(frame.delta.to_numpy(float)**2))) if frame.delta.notna().any() else None,
        partial_correlation=xy/np.sqrt(xx*yy) if xx>0 and yy>0 else None,
        within_airline_month_slope=xy/xx if xx>0 else None,
        residual_x_sse=xx, residual_y_sse=yy, crossproduct=xy,
        copy_rate=float(frame.target.mean()) if len(frame) else None)


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / 'design.json'
    if frozen.exists() and json.loads(frozen.read_text()) != DESIGN:
        raise ValueError('Stand falsifier design changed')
    write(frozen, DESIGN)
    parser = airline_parser()
    upstream = json.loads((ROT / 'identity.json').read_text())
    con = duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'})
    con.execute("SET TimeZone='UTC'")
    # Existing compact identity supplies clocks/stands; exact movement airline
    # requires only this four-column projection from locally cached originals.
    flights = con.execute("""SELECT MVT_ID_mvt mvt_id,FLIGHT_mvt
        FROM read_parquet(?) WHERE PHASE_mvt='ARR' AND ADES_mvt='LIRF'""",
        [str(ROOT / 'data/training_2025-*.parquet')]).df()
    arrivals = con.execute("""SELECT mvt_id,month,stand,
        abs(epoch(inblock)-epoch(arrival_schedule))<=60 AS copy
        FROM read_parquet(?) WHERE phase='ARR' AND airport='LIRF'
        AND inblock IS NOT NULL AND arrival_schedule IS NOT NULL""",
        [str(ROT / 'identity.parquet')]).df()
    queries = con.execute("""SELECT mvt_id,month,stand,no_id
        FROM read_parquet(?) WHERE phase='DEP' AND airport='LIRF'""",
        [str(ROT / 'identity.parquet')]).df().set_index('mvt_id')
    con.close()
    if not flights.mvt_id.is_unique or not arrivals.mvt_id.is_unique or not queries.index.is_unique:
        raise ValueError('Duplicate movement identity')
    flights['airline'] = parser(flights.FLIGHT_mvt).to_numpy()
    arrivals = arrivals.merge(flights[['mvt_id', 'airline']], on='mvt_id', validate='one_to_one', how='left')
    arrivals['copy'] = arrivals['copy'].astype(float)
    del flights
    gc.collect()
    report = dict(design=DESIGN, source_sha256=sha(Path(__file__)),
        scope='Only allowed-training departure labels; held departure packet fields not accessed',
        provenance=dict(rotation_manifest_sha256=sha(ROT/'identity.json'),
            rotation_identity_expected_sha256=upstream['identity_sha256'],
            movement_sources_expected_sha256=upstream['movement_sources'],
            airline_parser_sha256=sha(ROOT/'VALIDATION/rome_signals.py'),
            validation_scope='Upstream source hashes retained; no full-source rescan during bounded one-core audit'),
        total_projected_arrivals=len(arrivals), total_projected_departure_inputs=len(queries), folds={})
    for month, allowed in FOLDS.items():
        path = PACKETS / f'{month}.pkl.gz'
        record = json.loads(path.with_suffix('.json').read_text())
        if sha(path) != record['packet_sha256']:
            raise ValueError('Audited compact Rome packet changed')
        with gzip.open(path, 'rb') as handle:
            packet = pickle.load(handle)
        q = queries.loc[packet['train_ids']].copy()
        if set(q.month) != set(allowed) or len(q) != record['train_rows']:
            raise ValueError('Training membership changed')
        q['airline'] = packet['A'].airline.to_numpy()
        q['target'] = np.asarray(packet['target'], float)
        if int(q.target.sum()) != record['positives']:
            raise ValueError('Copy labels changed')
        q['delta'], q['support'], coverage = features(arrivals.loc[arrivals.month.isin(allowed)], q)
        result = dict(reference_months=allowed, context=coverage, packet_sha256=record['packet_sha256'],
            all=association(q), no_id=association(q.loc[q.no_id]), by_month={})
        for name, rows in q.groupby('month', observed=True):
            result['by_month'][name] = dict(all=association(rows), no_id=association(rows.loc[rows.no_id]))
        pivot = len(allowed)//2
        result['temporal_halves'] = {name:dict(months=months, all=association(q.loc[q.month.isin(months)]),
            no_id=association(q.loc[q.month.isin(months)&q.no_id]))
            for name, months in [('early', allowed[:pivot]), ('late', allowed[pivot:])]}
        report['folds'][month] = result
        del packet, q
        gc.collect()
        report['elapsed_seconds'] = time.monotonic()-started
        write(OUT / 'result.json', report)
    print(json.dumps({m:{k:v[k] for k in ['context','all','no_id','temporal_halves']} for m,v in report['folds'].items()},indent=2))
    print('elapsed_seconds', report['elapsed_seconds'])


if __name__ == '__main__':
    main()
