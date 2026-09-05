"""Build the data payloads for the leaderboard dashboard artifact.

Fetches the full (paginated) competition leaderboard and our team bucket
listing (uploads + per-file result JSONs via `mc`), aggregates them, and
writes JSON documents to tmp/dashboard/ ready to be pushed into the
artifact's database:

  meta.json         dashboard/meta        counts, timestamps, deadline
  leaderboard.json  dashboard/leaderboard per-team aggregates, ranked
  teams.json        dashboard/teams       registered-team roster (country, size)
  ours.json         dashboard/ours        our uploads with status/score
  feed-NNNN.json    feed/chunk-NNNN       all scored submissions, chunked
  manifest.json     (not a db doc) lists the doc ids written
  bootstrap.json    (not a db doc) combined snapshot to inline in the page

Run: .venv/bin/python pipeline/dashboard_data.py
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "dashboard"

TEAM = "unique-rose"
BUCKET = f"opensky/prc-2026-{TEAM}/"
API = (
    "https://datacomp.opensky-network.org/api/competitions/"
    "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
)
DEADLINE_UTC = "2026-10-11T21:59:59Z"  # 11 Oct 2026 23:59:59 CEST
CHUNK = 1000  # feed rows per db doc (db doc cap is 256 KiB)


def fetch_leaderboard():
    items, cursor = [], None
    while True:
        url = API + "/leaderboard" + (f"?cursor={cursor}" if cursor else "")
        with urlopen(url, timeout=30) as r:
            page = json.load(r)
        items += page.get("items", [])
        cursor = page.get("nextCursor")
        if not cursor:
            return items


def fetch_teams():
    """Registered-team roster (the leaderboard only shows teams that scored)."""
    teams, page = [], 1
    while True:
        with urlopen(API + f"/teams?page={page}&size=200", timeout=30) as r:
            data = json.load(r)
        teams += [
            {"name": t["teamName"], "country": t.get("country"),
             "type": t.get("type"), "participants": t.get("totalParticipants"),
             "registeredAt": t.get("registration_date"), "status": t.get("status")}
            for t in data.get("content", [])
        ]
        if page >= data.get("totalPages", 1) or not data.get("content"):
            return teams
        page += 1


def fetch_bucket():
    """Our uploads: parquet files merged with their *_result.json, via mc."""
    try:
        ls = subprocess.run(
            ["mc", "ls", "--json", BUCKET], capture_output=True, text=True, timeout=60
        )
        entries = [json.loads(line) for line in ls.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"warning: mc ls failed ({e}); dashboard will lack upload metadata")
        return []
    files = {e["key"]: e for e in entries if e.get("type") == "file"}
    ours = []
    for key, e in sorted(files.items()):
        if not key.endswith(".parquet"):
            continue
        item = {
            "file": key,
            "sizeBytes": e.get("size"),
            "uploadedAt": e.get("lastModified"),
            "status": "Pending",
            "score": None,
            "usedPairs": None,
        }
        if key + "_result.json" in files:
            for attempt in range(3):  # mc cat hits intermittent S3 timeouts
                cat = subprocess.run(
                    ["mc", "cat", BUCKET + key + "_result.json"],
                    capture_output=True, text=True, timeout=60,
                )
                if cat.returncode == 0:
                    result = json.loads(cat.stdout)
                    item["status"] = result.get("status", "Unknown")
                    item["score"] = result.get("score")
                    item["usedPairs"] = result.get("used_pairs")
                    break
            else:
                print(f"warning: could not read result for {key}: {cat.stderr.strip()}")
        ours.append(item)
    return ours


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    items = fetch_leaderboard()
    items.sort(key=lambda it: it["processedAt"], reverse=True)
    print(f"leaderboard: {len(items)} scored submissions")
    ours = fetch_bucket()
    print(f"bucket: {len(ours)} uploads")
    roster = fetch_teams()
    print(f"roster: {len(roster)} registered teams")

    teams = {}
    for it in items:
        t = teams.setdefault(
            it["teamName"],
            {"name": it["teamName"], "best": None, "bestFile": None, "count": 0,
             "firstAt": it["processedAt"], "lastAt": it["processedAt"]},
        )
        t["count"] += 1
        t["firstAt"] = min(t["firstAt"], it["processedAt"])
        t["lastAt"] = max(t["lastAt"], it["processedAt"])
        if t["best"] is None or it["score"] < t["best"]:
            t["best"], t["bestFile"] = it["score"], it["filename"]
    ranked = sorted(teams.values(), key=lambda t: t["best"])

    meta = {
        "updatedAt": now,
        "team": TEAM,
        "deadlineUtc": DEADLINE_UTC,
        "submissionCount": len(items),
        "teamCount": len(ranked),
        "registeredTeams": len(roster),
        "feedChunks": (len(items) + CHUNK - 1) // CHUNK,
    }
    leaderboard = {"updatedAt": now, "teams": ranked}
    ours_doc = {"updatedAt": now, "items": ours}
    teams_doc = {"updatedAt": now, "teams": roster}

    docs = {"dashboard/meta": meta,
            "dashboard/leaderboard": leaderboard,
            "dashboard/teams": teams_doc,
            "dashboard/ours": ours_doc}
    columns = ["team", "file", "score", "usedPairs", "processedAt"]
    for i in range(meta["feedChunks"]):
        rows = [[it["teamName"], it["filename"], it["score"],
                 it.get("usedPairs"), it["processedAt"]]
                for it in items[i * CHUNK:(i + 1) * CHUNK]]
        docs[f"feed/chunk-{i:04d}"] = {"i": i, "columns": columns, "rows": rows}

    for path, body in docs.items():
        f = OUT / (path.split("/")[-1] + ".json")
        if path.startswith("feed/"):
            f = OUT / (path.split("/")[-1].replace("chunk-", "feed-") + ".json")
        f.write_text(json.dumps(body))
        print(f"  {f.relative_to(ROOT)}  ->  {path}  ({f.stat().st_size:,} B)")

    (OUT / "manifest.json").write_text(json.dumps(
        {"updatedAt": now, "docs": sorted(docs)}, indent=2))
    bootstrap = {"meta": meta, "leaderboard": leaderboard, "teams": teams_doc,
                 "ours": ours_doc,
                 "feed": [docs[k] for k in sorted(docs) if k.startswith("feed/")]}
    (OUT / "bootstrap.json").write_text(json.dumps(bootstrap))
    print(f"bootstrap.json: {(OUT / 'bootstrap.json').stat().st_size:,} B")


if __name__ == "__main__":
    main()
