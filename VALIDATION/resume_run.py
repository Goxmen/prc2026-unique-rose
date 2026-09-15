"""Resume an interrupted validation run with unchanged modelling and data.

Only CPU execution settings may change. Completed fits and folds are reused.
"""

import argparse
import glob
import json
import time
from datetime import datetime, timezone

from harness import load, plan, execute, connect, TRAIN_GLOB, RANKING
from modeling import PARAMS
from records import (MODEL_FILES, ROOT, read_run, sha256, data_identity,
                     source_snapshot, write_json, append_log)
from reporting import report, print_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--fit-workers", type=int, default=1)
    args = ap.parse_args()
    if args.fit_workers < 1 or args.threads < args.fit_workers:
        ap.error("Provide a positive worker count and sufficient thread budget")
    record, directory = read_run(args.run)
    if record["status"] == "complete":
        raise ValueError("Run is already complete")
    for name in MODEL_FILES:
        if record["source"]["files"].get(name) != sha256(ROOT / name):
            raise ValueError(f"Model source changed: {name}")
    if record["config"]["params"] != PARAMS:
        raise ValueError("Model parameters changed")
    if record["data_identity"] != data_identity([*glob.glob(TRAIN_GLOB), RANKING]):
        raise ValueError("Input data changed")
    progress = json.loads((directory / "progress.json").read_text())
    for info in progress["models"].values():
        for details in info["seeds"].values():
            for key in ["model", "training_loss"]:
                if not (directory / details[key]).is_file():
                    raise ValueError(f"Missing saved artifact: {details[key]}")
    for info in progress["folds"].values():
        for path in info["predictions"].values():
            if not (directory / path).is_file():
                raise ValueError(f"Missing saved predictions: {path}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    continuation = {"at": stamp, "previous_execution": {
        k: record["config"].get(k) for k in ["threads", "fit_workers"]},
        "execution": {"threads": args.threads, "fit_workers": args.fit_workers},
        "reused_fits": sum(len(m["seeds"]) for m in progress["models"].values()),
        "reused_folds": list(progress["folds"]),
        "source": source_snapshot(directory / "continuations" / stamp)}
    record.setdefault("continuations", []).append(continuation)
    record["config"].update(threads=args.threads, fit_workers=args.fit_workers)
    record["status"] = "running"
    record.pop("error", None)
    prior_minutes = record.get("minutes", 0)
    write_json(directory / "result.json", record)
    schedule = argparse.Namespace(**record["config"])
    schedule.seeds = len(record["config"]["seeds"])
    _, windows = plan(schedule)
    start = time.monotonic()
    try:
        con = connect()
        dep, arr = load(con, record["config"]["sample"])
        con.close()
        record["folds"], record["models"] = execute(
            dep, arr, windows, record["config"], directory, resume_progress=progress)
        del dep, arr
        record["report"] = report(record, directory)
        print_report(record["report"])
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["minutes"] = round(prior_minutes + (time.monotonic() - start) / 60, 3)
        write_json(directory / "result.json", record)
        append_log(record, directory.parent, replace_existing=True)
    print(f"Completed resumed run: {directory}")


if __name__ == "__main__":
    main()
