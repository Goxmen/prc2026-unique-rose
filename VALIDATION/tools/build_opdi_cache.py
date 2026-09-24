"""Build the historical OPDI inputs from the verified local downloads; no network."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import opdi
from records import sha256, write_json


def main():
    started = time.monotonic()
    directory = opdi.OPDI_DIR
    download_path = directory / 'download_manifest.json'
    downloaded = json.loads(download_path.read_text())
    historical = json.loads((ROOT / 'VALIDATION/runs/v3_20260920T152734631417Z/result.json').read_text())
    if sha256(ROOT / 'VALIDATION/opdi.py') != historical['signature']['VALIDATION/opdi.py']:
        raise ValueError('Historical OPDI builder changed')
    if downloaded['status'] != 'complete':
        raise ValueError('Source downloads incomplete')
    missing = [name for name, info in downloaded['files'].items() if info['state'] == 'known_missing_404']
    if missing != ['flight_events_20251002_20251012.parquet']:
        raise ValueError(f'Unexpected missing files: {missing}')
    available = {name: info for name, info in downloaded['files'].items() if info['state'] in ('cached', 'downloaded')}
    if len(available) != 59:
        raise ValueError('Wrong source roster')
    identity = dict(download_manifest_sha256=sha256(download_path),
                    opdi_source_sha256=sha256(ROOT / 'VALIDATION/opdi.py'),
                    builder_sha256=sha256(Path(__file__)))
    manifest_path = directory / 'build_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Build inputs/code changed since checkpoint')
    else:
        manifest = dict(status='running', started=datetime.now(timezone.utc).isoformat(),
                        identity=identity, known_missing=missing, events={})
    for name, info in available.items():
        source = ROOT / info['path']
        if source.stat().st_size != info['bytes'] or sha256(source) != info['sha256']:
            raise ValueError(f'Changed downloaded source: {name}')
    print('All 59 source hashes verified', flush=True)
    event_dir = directory / 'events'
    event_dir.mkdir(exist_ok=True)
    expected_events = {name for name, info in available.items() if info['kind'] == 'flight_events'}
    if not expected_events:
        raise ValueError('No event sources in manifest')
    if set(p.name for p in event_dir.glob('*.parquet')) - expected_events:
        raise ValueError('Unexpected filtered event files')
    for number, name in enumerate(sorted(expected_events), 1):
        source = ROOT / available[name]['path']
        destination = event_dir / name
        if name in manifest['events']:
            if sha256(destination) != manifest['events'][name]['sha256']:
                raise ValueError(f'Changed filtered file: {name}')
            continue
        if destination.exists():
            raise ValueError(f'Unrecorded filtered output: {name}')
        partial = destination.with_suffix('.parquet.part')
        partial.unlink(missing_ok=True)
        before = time.monotonic()
        count = opdi.filter_events(source, partial)
        digest = sha256(partial)
        partial.replace(destination)
        manifest['events'][name] = dict(rows=count, sha256=digest, bytes=destination.stat().st_size,
                                       seconds=time.monotonic() - before)
        write_json(manifest_path, manifest)
        print(f'Filtered {number}/{len(expected_events)}: {name}: {count:,} rows', flush=True)
    output = opdi.FEATURES_PATH
    if 'features' in manifest:
        if sha256(output) != manifest['features']['sha256']:
            raise ValueError('Changed feature output')
        print('Feature cache already complete', flush=True)
        return
    if output.exists():
        raise ValueError('Unrecorded feature cache exists')
    # build_features' final count infers the format from the file extension.
    partial = output.with_name('opdi_features.partial.parquet')
    partial.unlink(missing_ok=True)
    con = duckdb.connect(config={'threads': 4, 'memory_limit': '6GB'})
    spill = ROOT / 'tmp/opdi_build_spill'
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{spill}'")
    print('Building departure features from the filtered local cache', flush=True)
    count = opdi.build_features(con, partial)
    unique, null_ids = con.execute(
        'select count(distinct mvt_id), count(*) filter (where mvt_id is null) from read_parquet(?)',
        [str(partial)]).fetchone()
    if count != 2429888 or unique != count or null_ids:
        raise ValueError(f'Unexpected departure population: {count}, {unique}, {null_ids}')
    coverage = con.execute('select adep, count(*) n, avg(op_matched) match_share, '
                          'count(op_ground_sec) ground_known from read_parquet(?) group by 1 order by 1',
                          [str(partial)]).df().to_dict('records')
    digest = sha256(partial)
    partial.replace(output)
    manifest.update(status='complete', completed=datetime.now(timezone.utc).isoformat(),
                    minutes=(time.monotonic() - started) / 60,
                    features=dict(path=str(output.relative_to(ROOT)), rows=count, sha256=digest,
                                  bytes=output.stat().st_size, coverage=coverage))
    write_json(manifest_path, manifest)
    report = dict(manifest, events_summary=dict(files=len(manifest['events']),
                  rows=sum(info['rows'] for info in manifest['events'].values())))
    write_json(ROOT / 'VALIDATION/reports/opdi_cache_build_20260924.json', report)
    print(f'Complete: {count:,} departures, SHA256 {digest}', flush=True)


if __name__ == '__main__':
    main()
