"""Attach an official leaderboard score to a harness run, then report the
local-vs-official relationship across every run that has both.

This is the external consistency check, and it is the only honest way to answer
"can the leaderboard score be derived from the validation score?". One pair
tells you nothing. Several pairs tell you whether the offset is stable — and a
stable offset is what makes the harness a predictor instead of a thermometer.

  .venv/bin/python VALIDATION/log_official.py --run 20260907T101500Z --official 412.7
  .venv/bin/python VALIDATION/log_official.py --report
"""

import argparse
import json
from pathlib import Path

import numpy as np

LOG = Path(__file__).resolve().parent / "runs" / "experiments.jsonl"


def load():
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run_id to annotate")
    ap.add_argument("--official", type=float, help="the RMSE the leaderboard returned")
    ap.add_argument("--submission", default="", help="e.g. unique-rose_v3.parquet")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rows = load()
    if args.run:
        hit = [r for r in rows if r["run_id"] == args.run]
        if not hit:
            raise SystemExit(f"no run {args.run} in {LOG}")
        for r in hit:
            r["official_rmse"] = args.official
            r["submission"] = args.submission
        LOG.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        print(f"run {args.run}: official {args.official} s recorded")

    if args.report or args.run:
        paired = [r for r in rows if r.get("official_rmse") is not None
                  and r.get("derived", {}).get("pooled_projection_s")]
        if not paired:
            print("\nno paired runs yet — submit once and record the score to start the series.")
            return
        print(f"\n{'run_id':<20} {'label':<22} {'local':>9} {'official':>9} {'offset':>9}")
        loc, off = [], []
        for r in sorted(paired, key=lambda r: r["at"]):
            p, o = r["derived"]["pooled_projection_s"], r["official_rmse"]
            loc.append(p); off.append(o)
            print(f"{r['run_id']:<20} {r['label'][:22]:<22} {p:>9.2f} {o:>9.2f} {o - p:>+9.2f}")
        d = np.array(off) - np.array(loc)
        print(f"\noffset  mean {d.mean():+.2f} s   sd {d.std(ddof=1) if len(d) > 1 else float('nan'):.2f} s")
        if len(loc) >= 3:
            print(f"correlation local vs official: {np.corrcoef(loc, off)[0, 1]:+.3f}")
            print("  A stable offset with positive correlation means you can derive the")
            print("  leaderboard score: official ~ local + offset. A sign flip means the")
            print("  harness has stopped predicting the thing that pays — stop and diagnose.")
        else:
            print("  Need at least 3 paired points before the correlation means anything.")


if __name__ == "__main__":
    main()
