"""Independent Python tuple-key census; no SQL macro or producer import."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261012_public_runway_key_census'
FEATURES = ROOT / 'VALIDATION/runs/v3_20261011_public_daily_taxi_features'


def read(path):
    return json.loads(path.read_text())


def art(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest())


def verify(a):
    p = ROOT / a['path']
    assert art(p) == {k: a[k] for k in ['path', 'sha256']}
    return p


def canonical(value):
    if value is None:
        return None
    x = value.strip().upper()
    if len(x) in [2, 3] and x[0] == '0' and x[1] in '123456789' and (len(x) == 2 or x[2] in 'LCR'):
        return x[1:]
    return x


def main():
    start = time.monotonic()
    target = OUT / 'independent_audit.json'
    assert not target.exists()
    design, result = read(OUT / 'design.json'), read(OUT / 'result.json')
    assert result['design'] == art(OUT / 'design.json')
    for a in design['sources']:
        verify(a)
    manifest, audit = read(FEATURES / 'manifest.json'), read(FEATURES / 'audit.json')
    assert audit['status'] == 'pass' and audit['manifest'] == art(FEATURES / 'manifest.json')
    public = verify(manifest['public'])
    source = pq.read_table(public, columns=['airport', 'day', 'runway'], use_threads=False).to_pydict()
    raw = Counter(zip(source['airport'], source['day'], source['runway']))
    assert len(raw) == sum(raw.values())
    normalized = Counter()
    for (airport, day, runway), n in raw.items():
        normalized[airport, day, canonical(runway)] += n
    collisions = sum(n > 1 for n in normalized.values())
    assert collisions == result['source_key_collisions']
    checks = {}
    for view, expected in [('2025', 2085047), ('ranking', 344841)]:
        path = verify(manifest['records'][view]['inputs'])
        assert art(path) in design['sources']
        keys, ids = Counter(), set()
        n = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,
                columns=['mvt_id', 'airport', 'local_day', 'runway'], use_threads=False):
            rows = batch.to_pydict()
            n += len(rows['mvt_id'])
            ids.update(rows['mvt_id'])
            keys.update(zip(rows['airport'], rows['local_day'], rows['runway']))
        assert n == len(ids) == expected
        del ids
        counts = Counter(rows=n)
        airports, recovered_airports, cases = Counter(), Counter(), Counter()
        for key, weight in keys.items():
            airport, day, runway = key
            available = runway is not None and key in raw
            canonical_count = normalized.get((airport, day, canonical(runway)), 0) if runway is not None else 0
            unambiguous = canonical_count == 1
            counts['original_matches'] += weight*available
            counts['canonical_matches'] += weight*unambiguous
            counts['recoverable_rows'] += weight*(not available and unambiguous)
            counts['ambiguous_rows'] += weight*(canonical_count > 1)
            airports[airport] += weight
            if not available and unambiguous:
                recovered_airports[airport] += weight
                cases[airport, runway, canonical(runway)] += weight
        for key in ['rows', 'original_matches', 'canonical_matches', 'recoverable_rows', 'ambiguous_rows']:
            assert counts[key] == result['views'][view][key], (view, key, counts[key])
        expected_airports = [dict(airport=a, rows=n, recoverable_rows=recovered_airports[a]) for a, n in sorted(airports.items())]
        assert expected_airports == result['views'][view]['by_airport']
        assert not cases and not result['views'][view]['recoverable_keys']
        checks[view] = dict(counts, distinct_input_keys=len(keys), IDs_unique=True)
    receipt = dict(status='pass', source=art(Path(__file__)), design=art(OUT / 'design.json'),
        result=art(OUT / 'result.json'), feature_manifest=art(FEATURES / 'manifest.json'),
        feature_audit=art(FEATURES / 'audit.json'), source_rows=sum(raw.values()),
        source_key_collisions=collisions, checks=checks, independent_python_tuple_keys=True,
        no_SQL_or_producer_function_reuse=True, no_labels_predictions_models_downloads=True,
        conclusion='Closed: zero formatting-only recoverable rows; no feature generation or model fit.',
        seconds=time.monotonic()-start)
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status='pass', source_rows=sum(raw.values()), checks=checks, seconds=receipt['seconds'])))


if __name__ == '__main__':
    main()
