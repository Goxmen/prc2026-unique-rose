"""Step 1: observable, month-matched changes, without classifiers or loss scoring.

Run from the repository root with .venv/bin/python. Raw features and cached METAR
reports are used; departure labels and block times are never selected. Queue
counts use the same month's events on both sides, with the first 30 UTC minutes
marked unavailable. This compares data populations, not fold-specific history.
"""

from __future__ import annotations

import calendar
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import runway_queue
import weather

OUT = ROOT / 'VALIDATION/reports/train_ranking_step1'
CACHE = ROOT / 'tmp/train_ranking_step1'
MONTHS = [f'2025-{m:02d}' for m in range(1, 13)]
TARGETS = ['2026-01', '2026-07']
MISSING = '<missing>'
CATEGORIES = ['airport', 'dest', 'operator', 'airline', 'aircraft', 'market',
              'wake', 'stand', 'runway', 'hour', 'dow', 'wind_sector']
RAW_MISSING = {'FLIGHT_ID_mvt': 'no_flight_id', 'AOBT_3_flt': 'no_aobt3',
               'LOBT_flt': 'no_lobt', 'IOBT_flt': 'no_iobt', 'EOBT_1_flt': 'no_eobt1',
               'SCHED_TIME_UTC_mvt': 'no_schedule', 'STAND_mvt': 'no_stand',
               'RUNWAY_mvt': 'no_runway', 'AIRCRAFT_TYPE_mvt': 'no_aircraft',
               'AIRCRAFT_OPERATOR_flt': 'no_operator'}
WX_FLAGS = ['wx_thunder', 'wx_snow', 'wx_freezing', 'wx_fog', 'wx_rain',
            'wx_heavy', 'wx_deicing', 'wx_2h_thunder', 'wx_2h_snow_or_freezing', 'wx_2h_fog']
RATES = list(RAW_MISSING.values()) + ['late_over_4h', 'noid_late_over_4h',
         'aobt3_after_takeoff', 'wx_low_visibility'] + WX_FLAGS
GAPS = {'sched': 'SCHED_TIME_UTC_mvt', 'lobt': 'LOBT_flt', 'iobt': 'IOBT_flt',
        'eobt1': 'EOBT_1_flt', 'aobt3': 'AOBT_3_flt'}
QUEUES = ['q_apt_takeoffs_prev30', 'q_rwy_takeoffs_prev30', 'q_rwy_landings_prev30']
CONTINUOUS = [f'takeoff_minus_{key}_min' for key in GAPS] + [
    c for c in weather.WEATHER_FEATURES if c not in WX_FLAGS + ['wx_wind_dir']
] + QUEUES
RAW_COLUMNS = list(dict.fromkeys([
    'MVT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
    'FLIGHT_mvt', 'FLIGHT_ID_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt',
    'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'STAND_mvt', 'RUNWAY_mvt',
    *GAPS.values(), *RAW_MISSING,
]))
assert not {'TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt'} & set(RAW_COLUMNS)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_month(con, month, paths, reports):
    # UTC is set on the connection. Naive timestamps avoid timezone conversion
    # differences between the DuckDB and pandas installations.
    cols = ', '.join(f'CAST({c} AS TIMESTAMP) AS {c}' if c == 'MVT_TIME_UTC_mvt'
                     or c in GAPS.values() else c for c in RAW_COLUMNS)
    raw = con.execute(f"SELECT {cols} FROM read_parquet(?) WHERE PHASE_mvt='DEP' "
                      "AND strftime(MVT_TIME_UTC_mvt, '%Y-%m')=?", [[str(p) for p in paths], month]).df()
    arr = con.execute("SELECT ADES_mvt, RUNWAY_mvt, CAST(MVT_TIME_UTC_mvt AS TIMESTAMP) "
                      "AS MVT_TIME_UTC_mvt FROM read_parquet(?) WHERE PHASE_mvt='ARR' "
                      "AND strftime(MVT_TIME_UTC_mvt, '%Y-%m')=?", [[str(p) for p in paths], month]).df()
    assert len(raw) and raw.MVT_ID_mvt.notna().all() and raw.MVT_ID_mvt.is_unique
    assert raw.MVT_TIME_UTC_mvt.notna().all() and raw.ADEP_mvt.notna().all()
    frame = pd.DataFrame({'mvt_id': raw.MVT_ID_mvt, 'date': raw.MVT_TIME_UTC_mvt.dt.floor('D')})
    mapping = dict(airport='ADEP_mvt', dest='ADES_mvt', operator='AIRCRAFT_OPERATOR_flt',
                   aircraft='AIRCRAFT_TYPE_mvt', market='MARKET_SEGMENT_flt',
                   wake='WK_TBL_CAT_flt', stand='STAND_mvt', runway='RUNWAY_mvt')
    for key, column in mapping.items():
        frame[key] = raw[column]
    # Same airline-prefix convention as rome_signals.airline_code, without
    # importing its training stack into this descriptive tool.
    flight = raw.FLIGHT_mvt.astype('string').where(raw.FLIGHT_mvt.astype('string').str.len() >= 3)
    frame['airline'] = flight.str[:3].where(flight.str[:3].str.isalpha().fillna(False), flight.str[:2])
    frame['hour'] = raw.MVT_TIME_UTC_mvt.dt.hour
    frame['dow'] = raw.MVT_TIME_UTC_mvt.dt.dayofweek
    for column, name in RAW_MISSING.items():
        frame[name] = raw[column].isna().astype(float)
    for name, column in GAPS.items():
        frame[f'takeoff_minus_{name}_min'] = (raw.MVT_TIME_UTC_mvt - raw[column]).dt.total_seconds() / 60
    delay = frame.takeoff_minus_sched_min
    frame['late_over_4h'] = (delay > 240).astype(float).where(delay.notna())
    frame['noid_late_over_4h'] = ((delay > 240) & frame.no_flight_id.eq(1)).astype(float).where(delay.notna())
    gap = frame.takeoff_minus_aobt3_min
    frame['aobt3_after_takeoff'] = (gap < 0).astype(float).where(gap.notna())
    wx = weather.weather_features(raw, reports)
    direction = wx.wx_wind_dir
    frame['wind_sector'] = np.floor(((direction % 360) + 22.5) % 360 / 45).where(direction.notna())
    frame = pd.concat([frame, wx.drop(columns='wx_wind_dir')], axis=1)
    frame['wx_low_visibility'] = (frame.wx_visibility_m < 1000).astype(float).where(frame.wx_visibility_m.notna())
    queue = runway_queue.queue_features(raw, raw, arr, features=QUEUES)
    boundary = raw.MVT_TIME_UTC_mvt < pd.Timestamp(month + '-01') + pd.Timedelta(minutes=30)
    queue.loc[boundary, :] = np.nan
    frame = pd.concat([frame, queue], axis=1)
    for col in CATEGORIES:
        frame[col] = frame[col].astype('string')
        assert not frame[col].eq(MISSING).any()
    return frame


def category_counts(values):
    return {str(k): int(v) for k, v in values.fillna(MISSING).value_counts().items()}


def category_comparison(source, target):
    """TV includes missing; novelty excludes missing, with all target rows as base."""
    n0, n1 = sum(source.values()), sum(target.values())
    assert n0 > 0 and n1 > 0
    keys = sorted(set(source) | set(target))
    changes = [(k, 100 * source.get(k, 0) / n0, 100 * target.get(k, 0) / n1) for k in keys]
    tv = sum(abs(b-a) for _, a, b in changes) / 2
    new_count = sum(n for k, n in target.items() if k != MISSING and k not in source)
    return {'tv_pp': tv, 'new_share_pct': 100 * new_count/n1,
            'missing_pct': 100 * target.get(MISSING, 0)/n1, 'changes': changes}


def reference_summary(values, observed):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return {'n': 0, 'median': None, 'p90': None, 'max': None, 'fraction_smaller': None}
    return {'n': int(len(v)), 'median': float(np.median(v)), 'p90': float(np.quantile(v, .9)),
            'max': float(v.max()), 'fraction_smaller': float(np.mean(v < observed - 1e-12))}


def profiles(frame, month):
    scalar, cat, coverage, daily = [], {}, [], []
    groups = [('ALL', frame), *list(frame.groupby('airport', sort=True))]
    n_days = calendar.monthrange(int(month[:4]), int(month[5:]))[1]
    full_dates = pd.date_range(month + '-01', periods=n_days)
    for airport, f in groups:
        counts = f.groupby('date').size().reindex(full_dates, fill_value=0)
        coverage.append(dict(month=month, airport=airport, n=len(f), calendar_days=n_days,
                             observed_days=int((counts > 0).sum()), flights_per_day=len(f)/n_days,
                             airport_share_pct=100*len(f)/len(frame)))
        scalar.extend([
            dict(month=month, airport=airport, feature='departures_per_day', statistic='mean',
                 value=len(f)/n_days, n_valid=n_days, n_total=n_days, unit='flights/day'),
            dict(month=month, airport=airport, feature='airport_share', statistic='rate_pct',
                 value=100*len(f)/len(frame), n_valid=len(frame), n_total=len(frame), unit='percentage points'),
        ])
        for col in RATES + CONTINUOUS:
            a = f[col].to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(a)
            base = dict(month=month, airport=airport, feature=col, n_valid=int(valid.sum()), n_total=len(a))
            scalar.append(dict(**base, statistic='missing_pct', value=100*float((~valid).mean()), unit='percentage points'))
            if col in RATES:
                stats = {'rate_pct': 100*float(a[valid].mean()) if valid.any() else None}
                unit = 'percentage points'
            else:
                stats = dict(zip(['p50', 'p90', 'p99'], map(float, np.quantile(a[valid], [.5,.9,.99])))) if valid.any() else dict.fromkeys(['p50','p90','p99'])
                stats['mean'] = float(a[valid].mean()) if valid.any() else None
                unit = 'minutes' if col.startswith('takeoff_') or col == 'wx_age_min' else (
                    'C' if col.endswith('_c') else 'knots' if col.endswith('_kt') else
                    'metres' if col.endswith('_m') else 'feet' if col.endswith('_ft') else 'flights')
            scalar.extend(dict(**base, statistic=k, value=v, unit=unit) for k, v in stats.items())
        cat[airport] = {col: category_counts(f[col]) for col in CATEGORIES}
        # Daily counts/sums preserve denominators for a later day-blocked audit.
        for col in RATES + CONTINUOUS:
            d = f.groupby('date')[col].agg(['count', 'sum', 'mean'])
            for date, row in d.iterrows():
                daily.append(dict(month=month, airport=airport, date=str(date.date()), feature=col,
                                  n_valid=int(row['count']), total=float(row['sum']),
                                  mean=float(row['mean']) if row['count'] else None))
    return scalar, cat, coverage, daily


def compare_scalars(profiles_table):
    results = []
    for (airport, feature, statistic), group in profiles_table.groupby(['airport', 'feature', 'statistic'], sort=True):
        indexed = group.set_index('month')
        values = indexed.value.to_dict()
        adjacent = [abs(values[b]-values[a]) for a, b in zip(MONTHS[:-1], MONTHS[1:])]
        allpairs = [abs(values[b]-values[a]) for a, b in itertools.combinations(MONTHS, 2)]
        for target in TARGETS:
            source = '2025' + target[4:]
            a, b = values[source], values[target]
            if pd.isna(a) or pd.isna(b):
                continue
            delta = float(b-a)
            record = dict(airport=airport, feature=feature, statistic=statistic, source_month=source,
                          target_month=target, source_value=float(a), target_value=float(b), delta=delta,
                          source_n_valid=int(indexed.at[source, 'n_valid']), target_n_valid=int(indexed.at[target, 'n_valid']),
                          source_n_total=int(indexed.at[source, 'n_total']), target_n_total=int(indexed.at[target, 'n_total']),
                          unit=indexed.at[source, 'unit'])
            for name, seq in [('adjacent', adjacent), ('all_pairs', allpairs)]:
                record.update({f'{name}_{k}': v for k, v in reference_summary(seq, abs(delta)).items()})
            results.append(record)
    return pd.DataFrame(results)


def compare_categories(all_counts):
    results, details = [], []
    for airport in sorted(all_counts[MONTHS[0]]):
        for feature in CATEGORIES:
            counts = {month: all_counts[month][airport][feature] for month in MONTHS + TARGETS}
            adjacent = [category_comparison(counts[a], counts[b]) for a, b in zip(MONTHS[:-1], MONTHS[1:])]
            allpairs = [category_comparison(counts[a], counts[b]) for a, b in itertools.combinations(MONTHS, 2)]
            # Full training support is distinct from same-month turnover.
            full = {}
            for month in MONTHS:
                for k, n in counts[month].items():
                    full[k] = full.get(k, 0) + n
            for target in TARGETS:
                source = '2025' + target[4:]
                diff = category_comparison(counts[source], counts[target])
                record = dict(airport=airport, feature=feature, source_month=source, target_month=target,
                              source_n=sum(counts[source].values()), target_n=sum(counts[target].values()),
                              tv_pp=diff['tv_pp'], new_vs_matched_month_pct=diff['new_share_pct'],
                              new_vs_full_2025_pct=category_comparison(full, counts[target])['new_share_pct'],
                              target_missing_pct=diff['missing_pct'])
                for name, seq in [('adjacent', adjacent), ('all_pairs', allpairs)]:
                    for metric in ['tv_pp', 'new_share_pct']:
                        record.update({f'{name}_{metric}_{k}': v for k, v in reference_summary([d[metric] for d in seq], diff[metric]).items()})
                results.append(record)
                for category, a, b in diff['changes']:
                    # Scope stand/runway novelty by airport; ALL is a mix-only summary.
                    details.append(dict(airport=airport, feature=feature, source_month=source, target_month=target,
                                        category=category, source_count=counts[source].get(category, 0),
                                        target_count=counts[target].get(category, 0), source_pct=a, target_pct=b,
                                        delta_pp=b-a, absent_from_full_2025=category != MISSING and category not in full))
    return pd.DataFrame(results), pd.DataFrame(details)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute('SET threads=6')
    con.execute("SET TimeZone='UTC'")
    training_paths = sorted((ROOT/'data').glob('training_2025-*.parquet'))
    assert len(training_paths) == 12
    # Some boundary movements are stored in the next month's file. Partition by
    # the movement timestamp over all source files, never by the filename.
    paths = {m: training_paths for m in MONTHS}
    paths.update({m: [ROOT/'data/ranking.parquet'] for m in TARGETS})
    input_paths = set(p for group in paths.values() for p in group)
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(input_paths | {weather.METAR})}
    expected = con.execute("SELECT strftime(MVT_TIME_UTC_mvt, '%Y-%m') AS ym, count(*) n "
                           "FROM read_parquet(?) WHERE PHASE_mvt='DEP' GROUP BY 1",
                           [[str(p) for p in sorted(input_paths)]]).df().set_index('ym').n.to_dict()
    assert set(expected) == set(MONTHS + TARGETS)
    implementation = {'builder': inspect.getsource(build_month), 'raw_columns': RAW_COLUMNS,
                      'categories': CATEGORIES, 'missing': RAW_MISSING, 'gaps': GAPS, 'queues': QUEUES,
                      'weather': sha(Path(weather.__file__)), 'runway_queue': sha(Path(runway_queue.__file__))}
    signature = hashlib.sha256(json.dumps([input_hashes, implementation], sort_keys=True).encode()).hexdigest()
    manifest = CACHE/'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old['signature'] == signature, 'Feature cache inputs/code changed; move the old cache before rebuilding.'
    else:
        write_json(manifest, {'signature': signature, 'input_sha256': input_hashes, 'files': {}})
    cache_info = json.loads(manifest.read_text())
    reports = weather.load_reports()
    all_scalar, all_coverage, all_daily, counts = [], [], [], {}
    id_sets, airports = [], None
    for month, source_paths in paths.items():
        cached = CACHE/f'{month}.parquet'
        if cached.exists():
            assert sha(cached) == cache_info['files'][month], f'Cache hash mismatch: {month}'
            frame = pd.read_parquet(cached)
        else:
            frame = build_month(con, month, source_paths, reports)
            frame.to_parquet(cached, index=False)
            cache_info['files'][month] = sha(cached)
            write_json(manifest, cache_info)
        assert frame.mvt_id.is_unique and frame.mvt_id.notna().all()
        assert len(frame) == expected[month], f'Month population mismatch: {month}'
        id_sets.append(frame.mvt_id.to_numpy())
        current_airports = set(frame.airport)
        if airports is None:
            airports = current_airports
        assert current_airports == airports and len(airports) == 10
        scalar, cat, coverage, daily = profiles(frame, month)
        all_scalar.extend(scalar); all_coverage.extend(coverage); all_daily.extend(daily); counts[month] = cat
        print(f'{month}: {len(frame):,} departures; cumulative {time.time()-started:.1f}s', flush=True)
    ids = np.concatenate(id_sets)
    assert len(ids) == len(np.unique(ids)), 'Movement IDs overlap across populations'
    ranking_ids = np.concatenate(id_sets[-2:])
    template = con.execute('SELECT MVT_ID_mvt FROM read_parquet(?)', [str(ROOT/'data/submitting.parquet')]).df().iloc[:,0].to_numpy()
    assert len(template) == len(ranking_ids) and np.array_equal(np.sort(template), np.sort(ranking_ids))
    table = pd.DataFrame(all_scalar)
    scalar_changes = compare_scalars(table)
    category_changes, category_details = compare_categories(counts)
    outputs = {'profiles.csv': table, 'coverage.csv': pd.DataFrame(all_coverage),
               'scalar_changes.csv': scalar_changes, 'category_changes.csv': category_changes,
               'category_details.csv': category_details}
    for name, frame in outputs.items():
        frame.to_csv(OUT/name, index=False, float_format='%.10g')
    daily_path = CACHE/'daily_profiles.parquet'
    pd.DataFrame(all_daily).to_parquet(daily_path, index=False)
    write_json(OUT/'metadata.json', {
        'step': 1, 'purpose': 'Describe observable data differences, not prediction cost or conditional label changes',
        'input_sha256': {**input_hashes, 'data/submitting.parquet': sha(ROOT/'data/submitting.parquet')},
        'script_sha256': sha(Path(__file__)), 'feature_cache_signature': signature,
        'training_departures': int(sum(map(len, id_sets[:-2]))), 'ranking_departures': len(ranking_ids),
        'matched_pairs': [[f'2025{m[4:]}', m] for m in TARGETS],
        'adjacent_pairs_2025': list(zip(MONTHS[:-1], MONTHS[1:])),
        'all_pairs_2025': list(itertools.combinations(MONTHS, 2)),
        'category_metric': 'TV in percentage points = half the sum of absolute category-share changes; missing is an explicit category.',
        'novelty': 'Target share absent from the matched source month and full 2025 are separate. Missing keys are never novel. Use airport rows for stand/runway/destination support; ALL can hide airport-specific novelty.',
        'numeric_metrics': 'Flight-weighted mean, median, p90 and p99 on finite observations. Missing fraction and valid/total denominators retained.',
        'rate_metrics': 'Percentage among rows with observable inputs; counts and missing fractions retained. Rates and changes use percentage points.',
        'weather': 'Existing cached METAR, shared weather.py code. Weather-code flags follow the existing parser; absence of a code is treated as no phenomenon when a fresh report exists.',
        'traffic': 'Same-month events only in both years. First 30 minutes of each month masked for 30-minute windows. All departures retained. Daily volume divided by calendar days.',
        'scope': 'Within-airport and pooled summaries. Pooled changes also reflect airport mix. No fold/history-derived inputs, stand clocks, labels, propensity scores, kNN or model errors used.',
        'reference_limits': '11 adjacent changes and 66 overlapping pairs from one year are descriptive ranges, not independent replicates, p-values, or an estimate of normal year-to-year variation.',
        'output_sha256': {name: sha(OUT/name) for name in outputs},
        'daily_profile_cache': {'path': str(daily_path.relative_to(ROOT)), 'sha256': sha(daily_path)},
        'elapsed_seconds': time.time()-started,
    })
    print(f'DONE: {OUT} ({time.time()-started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
