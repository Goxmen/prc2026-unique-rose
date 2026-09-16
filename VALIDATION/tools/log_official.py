"""Record official scores, keeping v3 and legacy score definitions separate."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import LOG, read_run, write_json
from protocol import VERSION


def calibration_key(row):
    if row.get("protocol") == VERSION:
        if not row.get("full_benchmark") or row.get("status") != "complete":
            return None
        cfg = row["config"]
        return (VERSION, row["score_version"], cfg["train_months"],
                cfg["early_stopping_rounds"])
    return ("legacy", "pooled_projection_s")


def local_score(row):
    if row.get("protocol") == VERSION:
        return row.get("report", {}).get("scores", {}).get("R_combined")
    return row.get("derived", {}).get("pooled_projection_s")


def report_rows(rows):
    groups = {}
    for row in rows:
        key = calibration_key(row)
        if key and row.get("official_rmse") is not None and local_score(row) is not None:
            groups.setdefault(key, []).append(row)
    if not groups:
        print("No eligible matched local/official results yet.")
    for key, paired in groups.items():
        print(f"\nSeparate score series: {key}")
        print(f"{'run_id':<30} {'local':>9} {'official':>9} {'offset':>9}")
        local, official = [], []
        for row in sorted(paired, key=lambda r: r["at"]):
            p, o = local_score(row), row["official_rmse"]
            local.append(p)
            official.append(o)
            print(f"{row['run_id']:<30} {p:>9.2f} {o:>9.2f} {o-p:>+9.2f}")
        offset = np.asarray(official) - local
        print(f"Observed offset mean {offset.mean():+.2f}; "
              f"sd {offset.std(ddof=1) if len(offset) > 1 else float('nan'):.2f}")
        if len(paired) >= 3 and np.std(local) and np.std(official):
            print(f"Observed correlation: {np.corrcoef(local, official)[0, 1]:.3f}")
        for a, b in zip(paired, paired[1:]):
            dl, do = local_score(b) - local_score(a), b["official_rmse"] - a["official_rmse"]
            if dl * do < 0:
                print(f"Direction disagreement: {a['run_id']} -> {b['run_id']}")
        print("These are observed comparisons, not a validated future-score forecast.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run ID to annotate")
    ap.add_argument("--official", type=float, help="official leaderboard RMSE")
    ap.add_argument("--submission", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--log", type=Path, default=LOG)
    args = ap.parse_args(argv)
    if args.run and (args.official is None or not np.isfinite(args.official) or args.official < 0):
        ap.error("--run requires a finite, nonnegative --official score")
    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()] if args.log.exists() else []
    if args.run:
        hits = [r for r in rows if r["run_id"] == args.run]
        if len(hits) != 1:
            raise ValueError(f"expected exactly one run {args.run} in {args.log}")
        hit = hits[0]
        hit.update(official_rmse=args.official, submission=args.submission)
        try:
            detail, directory = read_run(args.run, runs=args.log.parent)
        except FileNotFoundError:
            pass
        else:
            detail.update(official_rmse=args.official, submission=args.submission)
            write_json(directory / "result.json", detail)
        temp = args.log.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temp.replace(args.log)
        print(f"Recorded {args.official} s for {args.run}.")
    if args.report or args.run:
        report_rows(rows)


if __name__ == "__main__":
    main()
