"""Download ADSB.lol historical traces for a set of aircraft-days.

One file is one aircraft on one UTC day, worldwide:
    https://globe.adsb.lol/globe_history/YYYY/MM/DD/traces/<last2hex>/trace_full_<icao24>.json

The body is gzip-compressed but served without a ``Content-Encoding`` header, so
it must be decompressed by hand. We keep it compressed on disk -- measured at a
mean 82 KB gzipped against 472 KB decompressed -- and decompress on read.

About 21% of requested aircraft-days have no file; those are recorded as misses
so a rerun does not ask again.

ADSB.lol data is ODbL 1.0 and its use must be declared alongside the OPDI entry
in SUBMISSIONS.md.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data/external/adsb/traces"
MISSES = REPO / "data/external/adsb/misses.json"
BASE = "https://globe.adsb.lol/globe_history"
AGENT = "PRC-DataChallenge-2026/1.0 (research; taxi-time prediction)"


def cache_path(icao24: str, day: str) -> Path:
    """day is YYYY-MM-DD."""
    return CACHE / day.replace("-", "/") / f"trace_full_{icao24}.json.gz"


def url_for(icao24: str, day: str) -> str:
    return f"{BASE}/{day.replace('-', '/')}/traces/{icao24[-2:]}/trace_full_{icao24}.json"


def fetch_one(icao24: str, day: str, retries: int = 3) -> str:
    """Return 'hit', 'miss' or 'cached'. Writes the gzipped body unchanged."""
    dest = cache_path(icao24, day)
    if dest.exists():
        return "cached"
    req = Request(url_for(icao24, day), headers={"User-Agent": AGENT})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=60) as resp:
                body = resp.read()
            if not body:
                return "miss"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.rename(dest)
            return "hit"
        except HTTPError as exc:
            if exc.code == 404:
                return "miss"
            if exc.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
                continue
            return "miss"
        except (URLError, TimeoutError, http.client.HTTPException, socket.error):
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
                continue
            return "miss"
    return "miss"


def fetch_many(pairs: list[tuple[str, str]], workers: int = 4) -> dict[str, int]:
    known = set()
    if MISSES.exists():
        known = {tuple(x) for x in json.loads(MISSES.read_text())}
    todo = [p for p in pairs if tuple(p) not in known]
    counts = {"hit": 0, "miss": 0, "cached": 0, "skipped": len(pairs) - len(todo)}
    new_misses = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (icao24, day), status in zip(todo, pool.map(lambda p: fetch_one(*p), todo)):
            counts[status] += 1
            if status == "miss":
                new_misses.append((icao24, day))

    if new_misses:
        MISSES.parent.mkdir(parents=True, exist_ok=True)
        MISSES.write_text(json.dumps(sorted(known | {tuple(m) for m in new_misses})))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolved", type=Path, required=True,
                    help="parquet from adsb_join.py")
    ap.add_argument("--days", nargs="*", help="limit to these YYYY-MM-DD dates")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.resolved, columns=["icao24", "dep_date"])
    df["dep_date"] = df.dep_date.astype(str)
    if args.days:
        df = df[df.dep_date.isin(args.days)]
    pairs = sorted(set(zip(df.icao24, df.dep_date)))
    print(f"{len(pairs):,} aircraft-days requested")

    start = time.time()
    counts = fetch_many(pairs, workers=args.workers)
    elapsed = time.time() - start
    print(f"{counts}  in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
