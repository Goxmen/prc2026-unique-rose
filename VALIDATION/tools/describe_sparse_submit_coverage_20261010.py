"""Input-only sparse sequence coverage, with a compact Jan/Jul2025 reference.

Reads index metadata only, never points, labels, predictions or fitted models.
Mean raw cadence is span/(raw_count-1), not a maximum missing-observation gap.
No gate or model decision is selected from these descriptive distributions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/derived/sparse_track_submit_20261010'
TRAINING = ROOT / 'data/derived/sparse_track_sequence_20261010'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def record(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def quantile_changes(before, after):
    # No usable sequence in an airport/month produces SQL NULL for the whole
    # quantile array. Preserve this absent comparison instead of inventing zero.
    if before is None or after is None:
        return [None, None, None]
    if len(before) != 3 or len(after) != 3:
        raise ValueError('Unexpected quantile vector shape')
    return [y-x if x is not None and y is not None else None
            for x, y in zip(before, after)]


def aggregate(path, held_only=False):
    con = duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'})
    try:
        cursor = con.execute('''SELECT airport, substr(day,1,7) AS "month",
            count(*) AS "rows",
            sum("count"=0) zero_points, sum("count"=1) one_point,
            sum("count">=2) usable_points, sum(raw_count>128) subsampled,
            sum(hex IS NOT NULL) identified, sum(clock_ms IS NOT NULL) genuine_clock_known,
            sum(raw_ground_count>0) any_observed_ground,
            sum("count">=2 AND raw_ground_count>0) usable_with_ground,
            avg(CASE WHEN raw_count>0 THEN raw_ground_count::DOUBLE/raw_count END) mean_row_ground_fraction,
            sum(raw_ground_count)::DOUBLE/nullif(sum(raw_count),0) pooled_ground_fraction,
            quantile_cont(CASE WHEN "count">=2 THEN span_ms/1000.0 END,[0.1,0.5,0.9]) usable_span_sec_p10_p50_p90,
            quantile_cont(CASE WHEN raw_count>1 THEN span_ms/1000.0/(raw_count-1) END,[0.1,0.5,0.9]) mean_raw_cadence_sec_p10_p50_p90,
            quantile_cont(raw_count,[0.1,0.5,0.9]) raw_count_p10_p50_p90,
            sum(left_source_missing) left_source_missing,
            sum(right_source_missing) right_source_missing,
            sum(left_source_missing=1 OR right_source_missing=1) either_source_missing
            FROM read_parquet(?) WHERE (?=false OR substr(day,1,7) IN ('2025-01','2025-07'))
            GROUP BY airport,substr(day,1,7) ORDER BY airport,"month"''', [str(path), held_only])
        columns = [entry[0] for entry in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        con.close()


def prepare(out=OUT):
    started = time.monotonic()
    source = TRAINING / 'full_index.parquet'
    manifest = json.loads((TRAINING / 'full.json').read_text())
    if sha(source) != manifest['outputs'][source.name]:
        raise ValueError('Frozen training index changed')
    report = dict(source=record(source), manifest=record(TRAINING/'full.json'),
                  descriptor=record(Path(__file__)), rows=aggregate(source, held_only=True),
                  scope='All supplied January/July2025 departures; input metadata only',
                  seconds=time.monotonic()-started)
    path = out / 'coverage_reference_2025.json'
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != report[k] for k in ['source', 'manifest', 'descriptor', 'rows', 'scope']):
            raise ValueError('Frozen coverage reference changed')
        return old
    path.write_text(text)
    print(json.dumps(dict(groups=len(report['rows']), rows=sum(r['rows'] for r in report['rows']),
                          seconds=report['seconds']), indent=2), flush=True)
    return report


def describe(out=OUT):
    started = time.monotonic()
    reference = json.loads((out/'coverage_reference_2025.json').read_text())
    if reference['descriptor'] != record(Path(__file__)):
        raise ValueError('Coverage descriptor changed after reference export')
    report = dict(reference=record(out/'coverage_reference_2025.json'),
        descriptor=record(Path(__file__)), training=reference['rows'], ranking={}, deltas=[],
        scope='All supplied departures; no labels/predictions and no outcome-selected gating',
        caveats=['2026February/June have no same-season Jan/Jul2025 heldout comparison.',
                 'Observed ground is the retained raw alt=-123 code, not guaranteed stand occupancy.',
                 'Mean raw cadence is a flight average, not a maximum internal observation gap.',
                 'Missing source-file flags concern UTC window edges; present files may still lack observations.'])
    baseline = {(r['airport'], r['month'][5:]): r for r in reference['rows']}
    for view in ['old', 'final']:
        target = out/view
        manifest = json.loads((target/'full.json').read_text())
        index = target/'full_index.parquet'
        if sha(index) != manifest['outputs'][index.name]:
            raise ValueError('Production index changed')
        rows = aggregate(index)
        report['ranking'][view] = dict(manifest=record(target/'full.json'), index=record(index), rows=rows)
        for row in rows:
            before = baseline.get((row['airport'], row['month'][5:]))
            if before is None:
                continue
            difference = dict(view=view, airport=row['airport'], month=row['month'])
            for field in ['zero_points', 'one_point', 'usable_points', 'any_observed_ground',
                          'genuine_clock_known', 'either_source_missing']:
                difference[field+'_share_change'] = row[field]/row['rows']-before[field]/before['rows']
            for field in ['usable_span_sec_p10_p50_p90', 'mean_raw_cadence_sec_p10_p50_p90']:
                a, b = before[field], row[field]
                difference[field+'_change'] = quantile_changes(a, b)
            report['deltas'].append(difference)
    report['seconds'] = time.monotonic()-started
    (out/'coverage_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(groups={v:len(r['rows']) for v,r in report['ranking'].items()},
                          comparisons=len(report['deltas']), seconds=report['seconds']), indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'describe'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    (prepare if args.stage=='prepare' else describe)(args.out)
