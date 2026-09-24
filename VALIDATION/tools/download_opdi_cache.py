"""Download and retain OPDI source parquets, with atomic files and an audit manifest.

Uses the historical fetch_opdi roster without changing frozen model code.
No feature generation, model training, uploads, or submission-table changes.
Run from the repo root: .venv/bin/python VALIDATION/tools/download_opdi_cache.py
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq

import fetch_opdi as original

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data/external/opdi'
MANIFEST = CACHE / 'download_manifest.json'
LOGS = ROOT / 'tmp/opdi_download_20260921'
KNOWN_MISSING = {'flight_events_20251002_20251012.parquet'}
REQUIRED = {
    'flight_list': {'id', 'flt_id', 'adep', 'first_seen', 'last_seen'},
    'flight_events': {'flight_id', 'type', 'event_time', 'latitude', 'longitude', 'altitude', 'info'},
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_manifest(record):
    temporary = MANIFEST.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(MANIFEST)


def validate(path, kind):
    parquet = pq.ParquetFile(path)
    missing = REQUIRED[kind] - set(parquet.schema_arrow.names)
    if missing or parquet.metadata.num_rows < 1:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)} or empty file')
    return dict(bytes=path.stat().st_size, rows=parquet.metadata.num_rows,
                row_groups=parquet.metadata.num_row_groups, sha256=digest(path),
                schema=str(parquet.schema_arrow))


def roster():
    return ([dict(kind='flight_list', name=f'flight_list_{month}.parquet') for month in original.LIST_MONTHS]
            + [dict(kind='flight_events', name=name) for name in original.event_files()])


def download(item):
    kind, name = item['kind'], item['name']
    url = f'{original.BASE}/{kind}/{name}'
    directory = CACHE / ('flight_list' if kind == 'flight_list' else 'raw')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    started = time.monotonic()
    result = dict(**item, url=url, path=str(path.relative_to(ROOT)))
    try:
        if path.exists():
            result.update(state='cached', **validate(path, kind))
            return result
        partial = path.with_suffix('.parquet.part')
        log = LOGS / f'{name}.log'
        # Final paths never contain partial files. C- resumes interrupted transfers.
        # The upstream server sometimes resets large HTTP/2 streams (curl 92).
        command = ['curl', '--http1.1', '--silent', '--show-error', '--fail', '--location',
            '--retry', '5', '--retry-delay', '2', '--connect-timeout', '20',
            '--max-time', '1800', '--speed-limit', '1024', '--speed-time', '120',
            '--continue-at', '-', '--output', str(partial), '--write-out', '%{http_code}', url]
        with log.open('a') as stream:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=stream, text=True)
        status = completed.stdout.strip()[-3:]
        if status == '416' or completed.returncode == 33:
            # This server can reject Range requests even for incomplete local
            # files. Curl may return zero for HTTP 416, so check HTTP explicitly.
            result['restart_without_range'] = True
            result['discarded_partial_bytes'] = partial.stat().st_size if partial.exists() else 0
            partial.unlink(missing_ok=True)
            start = command.index('--continue-at')
            del command[start:start + 2]
            with log.open('a') as stream:
                stream.write('Server rejected resume; restarting this partial file using HTTP/1.1.\n')
                stream.flush()
                completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=stream, text=True)
            status = completed.stdout.strip()[-3:]
        result.update(http_status=status, curl_returncode=completed.returncode)
        if completed.returncode or status not in ('200', '206'):
            if status == '404' and name in KNOWN_MISSING:
                result.update(state='known_missing_404', note='Also documented as unavailable by the original September 19 downloader.')
                return result
            raise RuntimeError(f'curl exit {completed.returncode}, HTTP {status}; see {log.relative_to(ROOT)}')
        result.update(validate(partial, kind))
        partial.replace(path)
        result['state'] = 'downloaded'
    except Exception as error:
        result.update(state='failed', error=f'{type(error).__name__}: {error}')
    finally:
        result['seconds'] = round(time.monotonic() - started, 3)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error('workers must be between 1 and 8')
    CACHE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    wanted = roster()
    record = dict(status='downloading', started=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(Path(__file__)), roster_source_sha256=digest(Path(original.__file__)),
        workers=args.workers, expected_files=len(wanted), known_missing_roster=sorted(KNOWN_MISSING), files={})
    write_manifest(record)
    print(f'Starting {len(wanted)} OPDI files with {args.workers} download workers', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(download, item) for item in wanted]
        for future in as_completed(pending):
            result = future.result()
            record['files'][result['name']] = result
            write_manifest(record)
            size = f"; {result['bytes'] / 1e6:.1f} MB" if 'bytes' in result else ''
            print(f"{len(record['files'])}/{len(wanted)} {result['state']}: {result['name']}{size}", flush=True)
    states = [x['state'] for x in record['files'].values()]
    record.update(status='failed' if 'failed' in states else 'complete',
        completed=datetime.now(timezone.utc).isoformat(),
        totals=dict(available=sum(s in ('cached', 'downloaded') for s in states),
                    known_missing=states.count('known_missing_404'), failed=states.count('failed'),
                    available_bytes=sum(x.get('bytes', 0) for x in record['files'].values())))
    write_manifest(record)
    print(json.dumps(record['totals']), flush=True)
    if record['status'] != 'complete':
        raise SystemExit('Some files failed; rerunning verifies completed files and resumes partial transfers.')


if __name__ == '__main__':
    main()
