"""Download OPDI v0.0.2 flight events and flight lists for 2025 and the two ranking months.

Each 10-day flight-events file (0.05-0.55 GB) is reduced to ground events near the
ten challenge airports (`opdi.filter_events`) and the raw file is deleted. Monthly
flight lists are kept whole. Files already present are skipped, so the command can
be re-run after an interruption.

Source: EUROCONTROL Open Performance Data Initiative (https://www.opdi.aero), built
from OpenSky Network ADS-B data. Declare it as an external dataset.

Run: .venv/bin/python VALIDATION/tools/fetch_opdi.py [--workers 8]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import opdi

BASE = 'https://www.eurocontrol.int/performance/data/download/OPDI/v002'
FIRST_EVENTS_FILE = date(2024, 12, 26)   # first 10-day file that reaches into 2025
EVENT_WINDOWS = [(date(2024, 12, 26), date(2026, 2, 9)), (date(2026, 6, 29), date(2026, 8, 8))]
LIST_MONTHS = [f'2025{m:02d}' for m in range(1, 13)] + ['202601', '202602', '202607']


def event_files():
    """10-day file names on OPDI's grid, except the short file that ends 2025."""
    names, start = [], FIRST_EVENTS_FILE
    while start < EVENT_WINDOWS[-1][1]:
        end = start + timedelta(days=10)
        if start == date(2025, 12, 31):
            end = date(2026, 1, 10)
        if any(lo <= start < hi for lo, hi in EVENT_WINDOWS):
            names.append(f'flight_events_{start:%Y%m%d}_{end:%Y%m%d}.parquet')
        start = end
    return names


def download(url, path):
    subprocess.run(['curl', '-sf', '--retry', '5', '-o', str(path), url], check=True)


def fetch_events(name, raw_dir, out_dir):
    out = out_dir / name
    if out.exists():
        return f'skip {name}'
    raw = raw_dir / name
    try:
        download(f'{BASE}/flight_events/{name}', raw)
    except subprocess.CalledProcessError:
        # 20251002_20251012 is listed on opdi.aero but returns 404 (checked 2026-09-19).
        raw.unlink(missing_ok=True)
        return f'MISSING on server: {name}'
    tmp = out.with_suffix('.tmp')
    rows = opdi.filter_events(raw, tmp)
    tmp.rename(out)
    raw.unlink()
    return f'{name}: {rows:,} ground events'


def fetch_list(month, out_dir):
    out = out_dir / f'flight_list_{month}.parquet'
    if out.exists():
        return f'skip {out.name}'
    download(f'{BASE}/flight_list/{out.name}', out)
    return out.name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args(argv)
    events, lists, raw = (opdi.OPDI_DIR / 'events', opdi.OPDI_DIR / 'flight_list', opdi.OPDI_DIR / 'raw')
    for folder in (events, lists, raw):
        folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for message in pool.map(lambda m: fetch_list(m, lists), LIST_MONTHS):
            print(message, flush=True)
        for message in pool.map(lambda n: fetch_events(n, raw, events), event_files()):
            print(message, flush=True)


if __name__ == '__main__':
    main()
