"""Stream ADSB.lol daily heatmaps and keep only low traffic around the ten airports.

Each daily globe_history release on GitHub is a tar whose first ~0.6-0.9 GB holds
48 half-hour heatmaps: every aircraft worldwide, sampled every 10 s. This reads
only that prefix, keeps positions within BOX_KM of an airport that are on the
ground or at most MAX_ALT_FT, and closes the connection when the traces section
starts. Nothing raw is written to disk. One parquet per day plus a manifest record.

Heatmap record format (16 bytes, little-endian, checked on 2025-07-16):
  int32 hex, int32 lat, int32 lon, int32 packed (low int16 alt, high int16 gs)
  first n records          index of the file's n 10-s slices (record offsets), skipped
  hex == 0x0e7f7c9d       time marker, ms since epoch = lat << 32 | uint32(lon)
  |lat| > 90e6            callsign record: lat low 16 bits = squawk, lon + packed = 8 ASCII chars
  otherwise               position: lat/lon in 1e-6 deg; alt -123 = ground, else x25 ft;
                          gs x0.1 kt, -1 unknown; hex top byte carries address-type flags
Positions keep these raw encodings. Callsigns are kept only for aircraft that have a
kept position in the same half-hour.

Run from the repo root: .venv/bin/python VALIDATION/tools/download_adsblol_heatmap.py
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/external/adsblol_heatmap'
API = 'https://api.github.com/repos/adsblol/globe_history_{year}/releases?per_page=100&page={page}'
AGENT = 'PRC-2026-unique-rose research (github.com/Goxmen/PRC)'
MARKER = 0x0E7F7C9D
GROUND = -123
AIRPORTS = {  # aerodrome reference points
    'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861), 'EGLL': (51.4700, -0.4543),
    'EHAM': (52.3086, 4.7639), 'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626),
    'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389), 'LSZH': (47.4647, 8.5492),
    'LTFM': (41.2753, 28.7519),
}
CODES = sorted(AIRPORTS)


def wanted_days():
    """Held-out Jan/Jul first, then ranking months, then the training months of the
    July fit (Jan-Mar, Sep-Dec) so it can be scored before Apr-Jun and Aug arrive."""
    year = [date(2025, 1, 1) + timedelta(i) for i in range(365)]
    months = (1, 7, 2, 3, 9, 10, 11, 12, 4, 5, 6, 8)
    ranking = [date(2026, m, d) for m in (1, 7) for d in range(1, 32)]
    by_month = {m: [d for d in year if d.month == m] for m in months}
    return by_month[1] + by_month[7] + ranking + [d for m in months[2:] for d in by_month[m]]


def releases(year):
    cache = OUT / f'releases_{year}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    headers = {'User-Agent': AGENT, 'Accept': 'application/vnd.github+json'}
    try:
        token = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True,
                               env={'PATH': f'{Path.home()}/.local/bin:/usr/bin:/bin'}).stdout.strip()
        if token:
            headers['Authorization'] = f'Bearer {token}'
    except OSError:
        pass
    found, page = {}, 1
    while True:
        with urllib.request.urlopen(urllib.request.Request(API.format(year=year, page=page), headers=headers),
                                    timeout=60) as resp:
            batch = json.load(resp)
        for rel in batch:
            assets = sorted(rel['assets'], key=lambda a: a['name'])
            found[rel['tag_name']] = [dict(name=a['name'], size=a['size'], url=a['browser_download_url'])
                                      for a in assets]
        if len(batch) < 100 or page == 10:  # the API refuses pages past 1,000 releases (HTTP 422)
            break
        page += 1
    cache.write_text(json.dumps(found, indent=1, sort_keys=True) + '\n')
    return found


def source_for(day, catalogue, source='prod'):
    """The day's archive; 'staging' is ADSB.lol's parallel aggregator, used where prod lacks heatmaps."""
    found = [(sum(a['size'] for a in catalogue[t]), t) for t in
             (f'v{day:%Y.%m.%d}-planes-readsb-{source}-0', f'v{day:%Y.%m.%d}-planes-readsb-{source}-0tmp')
             if catalogue.get(t)]
    if not found:
        raise LookupError(f'{day}: no {source} release')
    # Where both exist, the larger is the whole day: 2025-06-11's -0 holds only its last five hours.
    tag = max(found)[1]
    return tag, catalogue[tag]


class HeatmapLater(Exception):
    """The traces section comes before the heatmaps in this day's tar."""


class Ranged:
    """The release's split tar parts read as one stream, starting at a byte offset."""

    def __init__(self, assets, start=0):
        self.parts, self.bytes, self.resp, base = [], 0, None, 0
        for asset in assets:
            if start < base + asset['size']:
                self.parts.append((asset['url'], max(0, start - base)))
            base += asset['size']

    def read(self, size=-1):
        while True:
            if self.resp is None:
                if not self.parts:
                    return b''
                url, offset = self.parts.pop(0)
                headers = {'User-Agent': AGENT, **({'Range': f'bytes={offset}-'} if offset else {})}
                self.resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
                if offset and self.resp.status != 206:
                    raise IOError(f'range request ignored: HTTP {self.resp.status}')
            chunk = self.resp.read(size)
            if chunk or size == 0:
                self.bytes += len(chunk)
                return chunk
            self.resp.close()
            self.resp = None

    def close(self):
        if self.resp is not None:
            self.resp.close()


def resolve(url):
    request = urllib.request.Request(url, headers={'User-Agent': AGENT, 'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=60) as resp:
        resp.read()
        return resp.geturl()


def fetch(assets, offset, length):
    base = 0
    for asset in assets:
        if offset < base + asset['size']:
            end = min(offset + length, base + asset['size']) - 1
            request = urllib.request.Request(asset['url'], headers={
                'User-Agent': AGENT, 'Range': f'bytes={offset - base}-{end - base}'})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request, timeout=60) as resp:
                        if resp.status != 206:
                            raise IOError(f'range request ignored: HTTP {resp.status}')
                        return resp.read()
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        base += asset['size']
    return b''


def tar_headers(buf, offset):
    """Valid ustar headers on 512-byte boundaries; split parts are multiples of 512 bytes."""
    for j in range(0, len(buf) - 511, 512):
        block = buf[j:j + 512]
        if block[257:262] == b'ustar':
            try:
                yield offset + j, tarfile.TarInfo.frombuf(block, 'utf-8', 'surrogateescape')
            except tarfile.HeaderError:
                continue


def top_dir(name):
    return name.lstrip('./').split('/')[0]


def locate_heatmap(assets, step=8 << 20, window=64 << 10):
    """Offset of the heatmap directory without reading the traces.

    Sample a small window every `step` bytes. Trace files average ~45 KB, so traces
    windows hold headers; the ~0.6-0.9 GB of 12-25 MB heatmap files form the longest
    run of empty windows. Walk member headers forward from the last non-heatmap
    header before such a run until the first heatmap member.
    """
    resolved = [dict(asset, url=resolve(asset['url'])) for asset in assets]
    total = sum(asset['size'] for asset in assets)
    starts = list(range(0, total, step))
    with ThreadPoolExecutor(8) as pool:
        windows = list(pool.map(lambda s: fetch(resolved, s, window), starts))
    found = [[(o, i) for o, i in tar_headers(buf, s) if top_dir(i.name) != 'heatmap']
             for s, buf in zip(starts, windows)]
    runs, begin = [], None
    for k, headers in enumerate(found + [[None]]):
        if not headers and begin is None:
            begin = k
        elif headers and begin is not None:
            runs.append((k - begin, begin, k))
            begin = None
    steps = 0
    for _, begin, end in sorted(runs, reverse=True):
        before = [h for hs in found[:begin] for h in hs]
        if not before:
            continue
        offset, info = before[-1]
        limit = starts[end] if end < len(starts) else total
        while offset < limit and steps < 1500:
            offset += 512 + -(-info.size // 512) * 512
            block = fetch(resolved, offset, 512)
            steps += 1
            if len(block) < 512 or block == bytes(512):
                break
            info = tarfile.TarInfo.frombuf(block, 'utf-8', 'surrogateescape')
            if top_dir(info.name) == 'heatmap':
                return offset, dict(windows=len(starts), window_bytes=len(starts) * window, header_steps=steps)
    raise LookupError(f'heatmap directory not found after {steps} header steps')


def boxes(box_km):
    out = []
    for code in CODES:
        lat, lon = AIRPORTS[code]
        out.append((int((lat - box_km / 111.32) * 1e6), int((lat + box_km / 111.32) * 1e6),
                    int((lon - box_km / (111.32 * np.cos(np.radians(lat)))) * 1e6),
                    int((lon + box_km / (111.32 * np.cos(np.radians(lat)))) * 1e6)))
    return out


def decode(raw, limits, max_alt_raw):
    rec = np.frombuffer(raw, dtype='<i4').reshape(-1, 4)
    is_marker = rec[:, 0] == MARKER
    slot = np.cumsum(is_marker) - 1
    times = (rec[is_marker, 1].astype(np.int64) << 32) | (rec[is_marker, 2].astype(np.int64) & 0xFFFFFFFF)
    is_call = ~is_marker & (np.abs(rec[:, 1]) > 90_000_000)
    is_pos = ~is_marker & ~is_call & (slot >= 0)
    pos, pos_slot = rec[is_pos], slot[is_pos]
    alt = (pos[:, 3] & 0xFFFF).astype(np.int16)
    airport = np.full(len(pos), -1, np.int8)
    for i, (lat0, lat1, lon0, lon1) in enumerate(limits):
        inside = (pos[:, 1] >= lat0) & (pos[:, 1] <= lat1) & (pos[:, 2] >= lon0) & (pos[:, 2] <= lon1)
        airport[inside] = i
    keep = (airport >= 0) & ((alt == GROUND) | (alt <= max_alt_raw))
    kept = dict(ts=times[pos_slot[keep]], airport=airport[keep], hex=pos[keep, 0], lat=pos[keep, 1],
                lon=pos[keep, 2], alt=alt[keep], gs=(pos[keep, 3] >> 16).astype(np.int16))
    call_rows = is_call & (slot >= 0)
    calls, call_ts = rec[call_rows], times[slot[call_rows]]
    near = np.isin(calls[:, 0], kept['hex'])
    calls, call_ts = calls[near], call_ts[near]
    text = calls[:, 2:].copy().view('S8').ravel()
    callsigns = dict(ts=call_ts, hex=calls[:, 0], squawk=(calls[:, 1] & 0xFFFF).astype(np.int32),
                     callsign=np.char.strip(np.char.decode(text, 'ascii', 'replace')))
    stats = dict(records=len(rec), markers=int(is_marker.sum()), positions=int(is_pos.sum()),
                 callsign_records=int(is_call.sum()), index_records=int((~is_marker & (slot < 0)).sum()))
    return kept, callsigns, stats


def atomic_parquet(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.parquet.tmp')
    pq.write_table(table, temporary, compression='zstd', compression_level=6)
    temporary.replace(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, rows=table.num_rows,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def process_day(day, tag, assets, box_km, max_alt_ft):
    started = time.time()
    limits, max_alt_raw = boxes(box_km), max_alt_ft // 25
    kept_parts, call_parts, files, totals, licence = [], [], [], {}, None

    def read_heatmaps(start, located, whole=False):
        """Read heatmap members from `start`; `whole` reads to the end, for scattered heatmaps."""
        nonlocal licence
        stream, seen = Ranged(assets, start), False
        try:
            with tarfile.open(fileobj=stream, mode='r|') as tar:
                for member in tar:
                    top = top_dir(member.name)
                    if top == 'heatmap':
                        seen = True
                        if not (member.isfile() and member.name.endswith('.bin.ttf')):
                            continue
                        payload = tar.extractfile(member).read()
                        raw = gzip.decompress(payload)
                        if len(raw) % 16:
                            raise ValueError(f'{member.name}: {len(raw)} bytes is not a multiple of 16')
                        kept, calls, stats = decode(raw, limits, max_alt_raw)
                        kept_parts.append(kept)
                        call_parts.append(calls)
                        for key, value in stats.items():
                            totals[key] = totals.get(key, 0) + value
                        files.append(dict(name=member.name, size=member.size,
                                          sha256=hashlib.sha256(payload).hexdigest()))
                    elif seen and not whole:
                        break
                    elif top == 'traces' and not located and not whole:
                        raise HeatmapLater
                    elif member.isfile() and 'LICENSE' in member.name.upper():
                        payload = tar.extractfile(member).read()
                        licence = dict(name=member.name, sha256=hashlib.sha256(payload).hexdigest())
                        (OUT / Path(member.name).name).write_bytes(payload)
        finally:
            stream.close()
        return stream.bytes

    try:
        streamed, layout, located = read_heatmaps(0, False), 'heatmap first', None
    except HeatmapLater:
        try:
            offset, located = locate_heatmap(assets)
            streamed = located['window_bytes'] + read_heatmaps(offset, True)
            layout, located['heatmap_offset'] = 'heatmap located after traces', offset
        except LookupError as error:
            streamed, layout, located = 0, 'heatmap not located', dict(error=str(error))
    if len({f['name'] for f in files}) < 48:
        # Some tars scatter heatmap files among the traces: read the whole archive once.
        for collected in (kept_parts, call_parts, files, totals):
            collected.clear()
        streamed += read_heatmaps(0, True, whole=True)
        layout += '; whole archive read for scattered heatmap files'
    if not kept_parts:
        raise ValueError(f'{day}: no heatmap files found')
    joined = {k: np.concatenate([p[k] for p in kept_parts]) for k in kept_parts[0]}
    if not len(joined['ts']):
        raise ValueError(f'{day}: no heatmap positions kept')
    order = np.lexsort((joined['hex'], joined['ts'], joined['airport']))
    joined = {k: v[order] for k, v in joined.items()}
    positions = pa.table({
        'ts': pa.array(joined['ts'], pa.timestamp('ms', tz='UTC')),
        'airport': pa.DictionaryArray.from_arrays(pa.array(joined['airport'], pa.int8()), pa.array(CODES)),
        'hex': joined['hex'], 'lat': joined['lat'], 'lon': joined['lon'],
        'alt': joined['alt'], 'gs': joined['gs'],
    })
    calls = pa.table({k: np.concatenate([p[k] for p in call_parts]) for k in call_parts[0]})
    calls = (calls.group_by(['hex', 'callsign', 'squawk'])
             .aggregate([('ts', 'min'), ('ts', 'max'), ('ts', 'count')])
             .rename_columns({'ts_min': 'first_ms', 'ts_max': 'last_ms', 'ts_count': 'records'})
             .select(['hex', 'callsign', 'squawk', 'first_ms', 'last_ms', 'records'])
             .sort_by([('hex', 'ascending'), ('first_ms', 'ascending')]))
    by_airport = pc.value_counts(positions['airport'].combine_chunks().dictionary_decode()).to_pylist()
    record = dict(
        day=str(day), tag=tag, assets=[a['name'] for a in assets], asset_bytes=sum(a['size'] for a in assets),
        streamed_bytes=streamed, layout=layout, located=located,
        heatmap_files=len(files), complete=len({f['name'] for f in files}) == 48, files=files, licence=licence, **totals,
        kept_positions=positions.num_rows, kept_ground=int(pc.sum(pc.equal(positions['alt'], GROUND)).as_py()),
        kept_by_airport={r['values']: r['counts'] for r in by_airport}, kept_callsigns=calls.num_rows,
        first_ts=str(datetime.fromtimestamp(joined['ts'].min() / 1000, timezone.utc)),
        last_ts=str(datetime.fromtimestamp(joined['ts'].max() / 1000, timezone.utc)),
        box_km=box_km, max_alt_ft=max_alt_ft,
        positions_file=atomic_parquet(positions, OUT / 'positions' / f'{day}.parquet'),
        callsigns_file=atomic_parquet(calls, OUT / 'callsigns' / f'{day}.parquet'),
        seconds=round(time.time() - started, 1),
        finished_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
    )
    manifest = OUT / 'manifest' / f'{day}.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.with_suffix('.json.tmp').write_text(json.dumps(record, indent=1) + '\n')
    manifest.with_suffix('.json.tmp').replace(manifest)
    return record


def with_retries(day, tag, assets, box_km, max_alt_ft, attempts=6):
    for attempt in range(attempts):
        try:
            return process_day(day, tag, assets, box_km, max_alt_ft)
        except Exception as error:  # network drops restart the whole day; it is only ~45 s
            if attempt == attempts - 1:
                raise
            print(f'{day} attempt {attempt + 1} failed: {error!r}', flush=True)
            time.sleep(min(300, 5 * 2 ** attempt))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--days', nargs='*', help='YYYY-MM-DD; default is all 2025 plus Jan/Jul 2026')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--box-km', type=float, default=20.0)
    ap.add_argument('--max-alt-ft', type=int, default=10_000)
    ap.add_argument('--source', choices=('prod', 'staging'), default='prod')
    ap.add_argument('--redo', action='store_true', help='refetch the listed days even if done')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = [date.fromisoformat(d) for d in args.days] if args.days else wanted_days()
    catalogue = {}
    for year in sorted({d.year for d in days}):
        catalogue.update(releases(year))
    todo = [d for d in days if args.redo or not (OUT / 'manifest' / f'{d}.json').exists()]
    print(f'{len(days)} days requested, {len(days) - len(todo)} already done, {len(todo)} to fetch', flush=True)
    started, streamed, failed = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(with_retries, d, *source_for(d, catalogue, args.source), args.box_km, args.max_alt_ft): d
                for d in todo}
        for n, job in enumerate(as_completed(jobs), 1):
            day = jobs[job]
            try:
                rec = job.result()
            except Exception as error:
                failed.append(str(day))
                print(f'{day} FAILED: {error!r}', flush=True)
                continue
            streamed += rec['streamed_bytes']
            rate = streamed / max(time.time() - started, 1)
            print(f'{n}/{len(todo)} {day} files={rec["heatmap_files"]} kept={rec["kept_positions"]:,} '
                  f'stream={rec["streamed_bytes"] / 1e6:.0f}MB {rec["seconds"]}s  '
                  f'total={streamed / 1e9:.1f}GB at {rate / 1e6:.1f}MB/s  '
                  f'eta={(len(todo) - n) * streamed / n / max(rate, 1) / 3600:.1f}h', flush=True)
    print(f'done in {(time.time() - started) / 3600:.2f} h, {streamed / 1e9:.1f} GB streamed, '
          f'failed: {failed or "none"}', flush=True)


if __name__ == '__main__':
    main()
