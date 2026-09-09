"""Fetch and summarize the live PRC leaderboard for team unique-rose.

This is the read-only command used by the local Codex leaderboard skill.  It
combines the public competition API with the team's MinIO result files, making
the distinction between the latest upload and the team's best score explicit.

Run:
    .venv/bin/python pipeline/leaderboard_status.py
    .venv/bin/python pipeline/leaderboard_status.py --json
    .venv/bin/python pipeline/leaderboard_status.py --cached
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone

from dashboard_data import OUT, TEAM, fetch_bucket, fetch_leaderboard, fetch_teams


VERSION_RE = re.compile(r"_v(\d+)\.parquet$")


def version_number(filename):
    match = VERSION_RE.search(filename or "")
    return int(match.group(1)) if match else None


def rank_teams(items):
    """Aggregate public submission rows using the competition's best-RMSE rule."""
    teams = {}
    for item in items:
        team = teams.setdefault(
            item["teamName"],
            {
                "name": item["teamName"],
                "best": None,
                "bestFile": None,
                "count": 0,
                "firstAt": item["processedAt"],
                "lastAt": item["processedAt"],
            },
        )
        team["count"] += 1
        team["firstAt"] = min(team["firstAt"], item["processedAt"])
        team["lastAt"] = max(team["lastAt"], item["processedAt"])
        if team["best"] is None or item["score"] < team["best"]:
            team["best"] = item["score"]
            team["bestFile"] = item["filename"]
    return sorted(teams.values(), key=lambda team: team["best"])


def load_live():
    items = fetch_leaderboard()
    return {
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshotSource": "live",
        "items": items,
        "uploads": fetch_bucket(strict=True),
        "registeredTeams": len(fetch_teams()),
    }


def load_cached():
    path = OUT / "bootstrap.json"
    with path.open() as handle:
        snapshot = json.load(handle)

    items = []
    for chunk in snapshot.get("feed", []):
        columns = chunk["columns"]
        for row in chunk["rows"]:
            value = dict(zip(columns, row))
            items.append(
                {
                    "teamName": value["team"],
                    "filename": value["file"],
                    "score": value["score"],
                    "usedPairs": value.get("usedPairs"),
                    "processedAt": value["processedAt"],
                }
            )
    return {
        "updatedAt": snapshot["meta"]["updatedAt"],
        "snapshotSource": "cache",
        "items": items,
        "uploads": snapshot.get("ours", {}).get("items", []),
        "registeredTeams": snapshot["meta"].get("registeredTeams"),
    }


def build_report(snapshot):
    ranked = rank_teams(snapshot["items"])
    our_rank = next(
        (index for index, team in enumerate(ranked, start=1) if team["name"] == TEAM),
        None,
    )
    our_team = ranked[our_rank - 1] if our_rank else None

    public_ours = {
        item["filename"]: item
        for item in snapshot["items"]
        if item["teamName"] == TEAM
    }
    uploads = []
    for upload in snapshot["uploads"]:
        item = dict(upload)
        public = public_ours.get(item["file"], {})
        item["version"] = version_number(item["file"])
        item["processedAt"] = public.get("processedAt")
        if item.get("score") is None and public.get("score") is not None:
            item["score"] = public["score"]
            item["usedPairs"] = public.get("usedPairs")
        uploads.append(item)

    uploads.sort(
        key=lambda item: (
            item.get("uploadedAt") or "",
            item["version"] if item["version"] is not None else -1,
        )
    )
    previous_score = None
    running_best = None
    for item in uploads:
        score = item.get("score")
        item["improvementVsPrevious"] = (
            previous_score - score
            if score is not None and previous_score is not None
            else None
        )
        item["scoreChangeVsPrevious"] = (
            score - previous_score
            if score is not None and previous_score is not None
            else None
        )
        if score is not None:
            previous_score = score
            running_best = score if running_best is None else min(running_best, score)
        item["bestSoFar"] = running_best

    latest = uploads[-1] if uploads else None
    leader = ranked[0] if ranked else None
    podium = ranked[2] if len(ranked) >= 3 else None
    nearby = []
    if our_rank:
        start = max(0, our_rank - 3)
        end = min(len(ranked), our_rank + 2)
        nearby = [dict(team, rank=index + 1) for index, team in enumerate(ranked[start:end], start)]

    return {
        "updatedAt": snapshot["updatedAt"],
        "snapshotSource": snapshot["snapshotSource"],
        "metric": "RMSE seconds (lower is better)",
        "team": {
            "name": TEAM,
            "rank": our_rank,
            "scoringTeamCount": len(ranked),
            "registeredTeamCount": snapshot["registeredTeams"],
            "bestScore": our_team["best"] if our_team else None,
            "bestFile": our_team["bestFile"] if our_team else None,
            "latest": latest,
            "scoredSubmissionCount": our_team["count"] if our_team else 0,
            "gapToLeader": (
                our_team["best"] - leader["best"] if our_team and leader else None
            ),
            "gapToPodium": (
                our_team["best"] - podium["best"] if our_team and podium else None
            ),
        },
        "history": uploads,
        "leader": dict(leader, rank=1) if leader else None,
        "podium": [dict(team, rank=index + 1) for index, team in enumerate(ranked[:3])],
        "nearby": nearby,
        "standings": [dict(team, rank=index + 1) for index, team in enumerate(ranked)],
        "publicSubmissionCount": len(snapshot["items"]),
    }


def format_score(score):
    return "pending" if score is None else f"{score:.4f}"


def format_text(report, top):
    team = report["team"]
    latest = team["latest"]
    lines = [
        f"Snapshot: {report['updatedAt']} ({report['snapshotSource']})",
        f"Metric: {report['metric']}",
    ]
    if team["bestScore"] is None:
        lines.append(f"{TEAM}: no scored submission")
    else:
        lines.append(
            f"{TEAM}: rank {team['rank']}/{team['scoringTeamCount']} scoring teams "
            f"({team['registeredTeamCount']} registered)"
        )
        lines.append(
            f"Best: {format_score(team['bestScore'])} via {team['bestFile']} | "
            f"gap to leader +{team['gapToLeader']:.4f} | "
            f"gap to #3 +{team['gapToPodium']:.4f}"
        )
    if latest:
        lines.append(
            f"Latest: {latest['file']} | {latest['status']} | "
            f"{format_score(latest.get('score'))} | uploaded {latest.get('uploadedAt', 'unknown')}"
        )

    lines.extend(["", "Our submission history:"])
    for item in report["history"]:
        delta = item.get("improvementVsPrevious")
        if delta is None:
            delta_text = "—"
        elif delta >= 0:
            delta_text = f"improved by {delta:.4f}"
        else:
            delta_text = f"worse by {-delta:.4f}"
        lines.append(
            f"  {item['file']}: {format_score(item.get('score'))} | "
            f"{item['status']} | vs previous {delta_text}"
        )

    lines.extend(["", f"Public leaderboard top {min(top, len(report['standings']))}:"])
    for team_row in report["standings"][:top]:
        lines.append(
            f"  {team_row['rank']:>3}. {team_row['name']}: "
            f"{team_row['best']:.4f} ({team_row['count']} scored)"
        )
    if report["nearby"] and team["rank"] > top:
        lines.extend(["", "Around unique-rose:"])
        for team_row in report["nearby"]:
            marker = " <- us" if team_row["name"] == TEAM else ""
            lines.append(
                f"  {team_row['rank']:>3}. {team_row['name']}: "
                f"{team_row['best']:.4f}{marker}"
            )
    lines.append(
        f"\nBoard totals: {report['publicSubmissionCount']} scored submissions from "
        f"{team['scoringTeamCount']} teams."
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete report as JSON")
    parser.add_argument("--top", type=int, default=10, help="number of leading teams in text output")
    parser.add_argument(
        "--all",
        action="store_true",
        help="include every team in JSON standings instead of only --top teams",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="use tmp/dashboard/bootstrap.json instead of contacting live sources",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be at least 1")
    try:
        snapshot = load_cached() if args.cached else load_live()
        report = build_report(snapshot)
    except Exception as exc:
        source = "cached snapshot" if args.cached else "live leaderboard sources"
        print(f"error: could not read {source}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        output = dict(report)
        if not args.all:
            output["standings"] = report["standings"][:args.top]
            output["standingsTruncated"] = len(report["standings"]) > args.top
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_text(report, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
