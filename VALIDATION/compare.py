"""Recompute v3 paired comparisons or block-length sensitivity without training."""

import argparse
import copy

from records import read_run, write_json
from reporting import report, print_report
from protocol import VERSION


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="v3 run ID or result.json")
    ap.add_argument("--baseline", help="v3 baseline ID or result.json")
    ap.add_argument("--block-days", type=int, default=1)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    args = ap.parse_args(argv)
    record, directory = read_run(args.run)
    if record.get("protocol") != VERSION or record.get("status") != "complete":
        raise ValueError("--run must identify a completed v3 run")
    record = copy.deepcopy(record)
    for field in ["block_days", "bootstrap_reps", "bootstrap_seed"]:
        record["config"][field] = getattr(args, field)
    baseline = read_run(args.baseline) if args.baseline else None
    result = report(record, directory, baseline)
    print_report(result)
    name = baseline[0]["run_id"] if baseline else "within-run"
    path = directory / f"comparison_{name}_block{args.block_days}.json"
    write_json(path, {"run_id": record["run_id"], "baseline_id": name,
                      "bootstrap": {k: record["config"][k] for k in
                                    ["block_days", "bootstrap_reps", "bootstrap_seed"]},
                      "report": result})
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
