"""Download METAR reports for the ten challenge airports (external weather data).

Source: the Iowa Environmental Mesonet (IEM) ASOS/METAR archive,
https://mesonet.agron.iastate.edu/request/download.phtml, fed by the Unidata IDD
and NCEI ISD feeds. IEM states its products are public domain and free for any
lawful use; attribution to the Iowa Environmental Mesonet of Iowa State
University is appreciated.

Routine reports (type 3) and specials (type 4) are requested separately so each
row keeps its type. Requests run one at a time with a pause and retries.
Units are the service's: temperatures °F, wind knots, visibility statute miles,
cloud bases feet. Outside the US the archive carries no precipitation amounts.

Writes, under data/external/metar/ (gitignored):
    raw/<ICAO>_<type>.csv   service output as received
    metar.parquet           all stations, typed, UTC timestamps
    manifest.json           request URLs, retrieval times, sizes, SHA-256, row counts

Run: .venv/bin/python VALIDATION/tools/fetch_metar.py
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'external' / 'metar'
SERVICE = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
REPORT_TYPES = {3: 'routine', 4: 'special'}
FIELDS = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'gust', 'vsby', 'alti',
          'skyc1', 'skyc2', 'skyc3', 'skyc4', 'skyl1', 'skyl2', 'skyl3', 'skyl4',
          'wxcodes', 'metar']
NUMERIC = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'gust', 'vsby', 'alti',
           'skyl1', 'skyl2', 'skyl3', 'skyl4']


def request_url(station, report_type, start, end):
    params = [('station', station), *[('data', f) for f in FIELDS],
              ('sts', f'{start}T00:00Z'), ('ets', f'{end}T00:00Z'), ('tz', 'Etc/UTC'),
              ('format', 'onlycomma'), ('latlon', 'no'), ('elev', 'no'), ('missing', 'empty'),
              ('trace', 'empty'), ('direct', 'no'), ('report_type', report_type)]
    return f'{SERVICE}?{urllib.parse.urlencode(params)}'


def fetch(url, attempts=5):
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                body = response.read()
            if not body.startswith(b'station,valid'):
                raise ValueError(f'Unexpected response: {body[:200]!r}')
            return body
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f'  attempt {attempt} failed ({exc}); retrying', flush=True)
            time.sleep(15 * attempt)


def parse(body, report_type):
    frame = pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)
    frame = frame.replace('', pd.NA)
    frame['valid'] = pd.to_datetime(frame['valid'], utc=True)
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column])
    frame['report_type'] = REPORT_TYPES[report_type]
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2024-12-31', help='first UTC day (inclusive)')
    ap.add_argument('--end', default='2026-08-01', help='last UTC day (exclusive)')
    ap.add_argument('--pause', type=float, default=3.0, help='seconds between requests')
    args = ap.parse_args(argv)

    (OUT / 'raw').mkdir(parents=True, exist_ok=True)
    manifest = dict(source='Iowa Environmental Mesonet ASOS/METAR archive', service=SERVICE,
                    terms='https://mesonet.agron.iastate.edu/disclaimer.php',
                    units=dict(temperature='F', wind='knots', visibility='statute miles', cloud_base='feet'),
                    start=args.start, end=args.end, requests=[])
    frames = []
    for station in AIRPORTS:
        for report_type, name in REPORT_TYPES.items():
            url = request_url(station, report_type, args.start, args.end)
            print(f'{station} {name}', flush=True)
            body = fetch(url)
            path = OUT / 'raw' / f'{station}_{name}.csv'
            path.write_bytes(body)
            frame = parse(body, report_type)
            frames.append(frame)
            manifest['requests'].append(dict(
                station=station, report_type=name, url=url, file=str(path.relative_to(ROOT)),
                retrieved=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), rows=len(frame)))
            print(f'  {len(frame):,} reports', flush=True)
            time.sleep(args.pause)
    table = pd.concat(frames, ignore_index=True).sort_values(['station', 'valid', 'report_type'])
    table.to_parquet(OUT / 'metar.parquet', index=False)
    manifest['parquet_sha256'] = hashlib.sha256((OUT / 'metar.parquet').read_bytes()).hexdigest()
    manifest['rows'] = len(table)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(table):,} reports to {OUT / "metar.parquet"}', flush=True)


if __name__ == '__main__':
    main()
