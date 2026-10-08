"""Download OPDI v0.0.2 inputs for the final test set (`final_ranking.parquet`, Jan/Feb/Jun/Jul 2026).

Same files, filter and server as `fetch_opdi.py` (whose 2025/Jan/Jul raw files were deleted after the
feature build), into a separate folder so the original caches stay untouched:
  events   10-day files from 2025-12-21 (Dec 31 departures are neighbours of Jan 1) to 2026-03-01,
           and 2026-05-30 (May 31 first-seen times) to 2026-08-08
  lists    monthly flight lists 2025-12, 2026-01/02/05/06/07
Each raw events file is reduced to ground events near the ten airports (`opdi.filter_events`)
and deleted. Re-running skips files already present.

Run: .venv/bin/python VALIDATION/tools/fetch_opdi_final.py [--workers 4]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_opdi as original

OUT = original.opdi.ROOT / 'data/external/opdi_final'
EVENT_WINDOWS = [(date(2025, 12, 21), date(2026, 3, 1)), (date(2026, 5, 30), date(2026, 8, 8))]
LIST_MONTHS = ['202512', '202601', '202602', '202605', '202606', '202607']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args(argv)
    original.EVENT_WINDOWS = EVENT_WINDOWS
    events, lists, raw = OUT / 'events', OUT / 'flight_list', OUT / 'raw'
    for folder in (events, lists, raw):
        folder.mkdir(parents=True, exist_ok=True)
    names = original.event_files()
    print(f'{len(names)} event files, {len(LIST_MONTHS)} flight lists', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(original.fetch_list, m, lists) for m in LIST_MONTHS]
        jobs += [pool.submit(original.fetch_events, n, raw, events) for n in names]
        for job in jobs:
            print(job.result(), flush=True)


if __name__ == '__main__':
    main()
