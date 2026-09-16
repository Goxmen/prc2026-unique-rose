"""Train seven full-year models and upload them directly to the leaderboard.

Run: .venv/bin/python VALIDATION/tools/sweep_taxi_thresholds.py
Preview IDs: add --dry-run. Resume: --resume VALIDATION/reports/sweeps/<id>.json.
Refresh scores: --status VALIDATION/reports/sweeps/<id>.json.
No holdout evaluation or early stopping is run. Defaults match submission v3.
"""

import argparse
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "VALIDATION"))
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, SUBMISSIONS, TEAM
from dashboard_data import BUCKET, mc_environment
from records import MODEL_FILES, data_identity, sha256, write_json
from submit import DEFAULTS

THRESHOLDS = (10_000, 16_000, 22_000, 32_000, 50_000, 70_000, 90_000)
SWEEPS = ROOT / "VALIDATION" / "reports" / "sweeps"
VERSION_RE = re.compile(rf"^{re.escape(TEAM)}_v(\d+)\.")
SOURCE_FILES = MODEL_FILES + ["VALIDATION/submit.py", "VALIDATION/harness.py",
                              "VALIDATION/protocol.py", "VALIDATION/records.py",
                              "VALIDATION/tools/sweep_taxi_thresholds.py", "requirements.txt"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mc(*args, binary=False):
    executable = shutil.which("mc") or str(Path.home() / ".local/bin/mc")
    return subprocess.run([executable, *map(str, args)], capture_output=True,
                          text=not binary, check=True, timeout=120,
                          env=mc_environment()).stdout


def remote_files():
    entries = [json.loads(line) for line in mc("ls", "--json", BUCKET).splitlines()
               if line.strip()]
    if any(entry.get("status") == "error" for entry in entries):
        raise RuntimeError("MinIO returned an error while listing submission IDs")
    return {entry["key"] for entry in entries if entry.get("type") == "file"}


def plan(remote):
    # Include partial local outputs and IDs reserved by interrupted sweeps.
    names = set(remote) | {path.name for path in SUBMISSIONS.glob(f"{TEAM}_v*")}
    for path in SWEEPS.glob("*.json"):
        names.update(item["file"] for item in json.loads(path.read_text())["submissions"])
    versions = [int(match.group(1)) for name in names
                if (match := VERSION_RE.match(name))]
    first = max(versions, default=0) + 1
    return [{"version": first + i, "file": f"{TEAM}_v{first + i}.parquet",
             "max_taxi_sec": threshold, "state": "planned", "score": None}
            for i, threshold in enumerate(THRESHOLDS)]


def inputs():
    files = sorted(DATA.glob("training_2025-*.parquet"))
    if len(files) != 12 or {p.name[9:16] for p in files} != {
        f"2025-{month:02}" for month in range(1, 13)
    }:
        raise ValueError("All twelve original 2025 training months are required")
    return files + [DATA / "ranking.parquet", DATA / "submitting.parquet"]


def source_identity():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def training_command(sweep, item):
    command = [sys.executable, str(ROOT / "VALIDATION/submit.py"),
               "--version", str(item["version"]),
               "--max-taxi-sec", str(item["max_taxi_sec"]),
               "--seed", str(sweep["seed"]), "--track-training-loss"]
    for name, value in sweep["config"].items():
        if name == "fpl_flag":
            if value:
                command.append("--fpl-flag")
        else:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def check_artifact(sweep, item):
    """Require a complete matching fit before accepting or uploading its output."""
    path = SUBMISSIONS / item["file"]
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    expected = dict(sweep["config"], max_taxi_sec=item["max_taxi_sec"])
    if (meta["config"] != expected or meta["seed"] != sweep["seed"]
            or meta["selected_rounds"] != expected["rounds"]
            or meta["inner_selection"] is not None or meta["harness_run_id"] is not None
            or meta["full_benchmark_validated"] or meta["freeze"] != "2025-12-31"
            or meta["model_source"] != {name: sweep["source"][name] for name in MODEL_FILES}
            or not path.with_suffix(".model.txt").is_file()):
        raise ValueError(f"Incomplete or mismatched training artifacts for {item['file']}")
    digest = sha256(path)
    if item.get("sha256") and item["sha256"] != digest:
        raise ValueError(f"Submission content changed: {item['file']}")
    item.update(sha256=digest, train_rows=meta["train_rows"])
    return path


def refresh_scores(sweep, remote):
    for item in sweep["submissions"]:
        key = item["file"] + "_result.json"
        # Only attach scores after confirming ownership of the uploaded file.
        if item["state"] != "uploaded" or key not in remote:
            continue
        result = json.loads(mc("cat", BUCKET + key))
        item.update(official_status=result.get("status", "Unknown"),
                    score=result.get("score"), used_pairs=result.get("used_pairs"))
    sweep["scores_checked_at"] = now()


def plot_training_curves(sweep, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = []
    for item in sweep["submissions"]:
        path = (SUBMISSIONS / item["file"]).with_suffix(".training_loss.json")
        if not path.is_file():
            continue
        curve = json.loads(path.read_text())
        if (curve["file"] != item["file"] or curve["max_taxi_sec"] != item["max_taxi_sec"]
                or curve["seed"] != sweep["seed"]
                or len(curve["rmse"]) != sweep["config"]["rounds"]):
            raise ValueError(f"Training curve does not match {item['file']}")
        curves.append(curve)
        item["training_rmse_final"] = curve["rmse"][-1]
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.17)
    for curve in curves:
        version = VERSION_RE.match(curve["file"]).group(1)
        ax.plot(range(1, len(curve["rmse"]) + 1), curve["rmse"], linewidth=1.6,
                label=f"v{version}: {curve['max_taxi_sec']:,} s (final {curve['rmse'][-1]:.2f})")
    ax.set(xlabel="Boosting round", ylabel="Training RMSE (seconds)")
    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Maximum training taxi-out time", fontsize=9)
    fig.suptitle(f"Taxi-out threshold sweep: training loss ({len(curves)}/7 models)",
                 x=0.09, ha="left", fontsize=17, weight="bold")
    fig.text(0.09, 0.905, f"All 12 training months · seed {sweep['seed']} · "
             f"{sweep['config']['rounds']} fixed rounds", color="#52514e")
    fig.text(0.09, 0.04,
             "Each curve measures its own filtered training rows. Compare leaderboard RMSE "
             "to choose the threshold.", fontsize=10, color="#52514e")
    chart = manifest.with_name(manifest.stem + "_training_loss.png")
    fig.savefig(chart, dpi=160)
    plt.close(fig)
    sweep["training_loss_chart"] = str(chart)
    print(f"Training loss chart: {chart}", flush=True)


def show(sweep):
    print("\nSubmission                 Max taxi (s)  Training rows  State / official status       RMSE")
    for item in sweep["submissions"]:
        score = "pending" if item.get("score") is None else f"{item['score']:.4f}"
        state = item["state"] + " / " + item.get("official_status", "—")
        rows = f"{item['train_rows']:,}" if "train_rows" in item else "—"
        print(f"{item['file']:<27} {item['max_taxi_sec']:>11,}  {rows:>13}  {state:<29} {score}")
    if sweep.get("scores_checked_at"):
        print(f"Scores checked: {sweep['scores_checked_at']}")


def execute(sweep, manifest):
    for item in tqdm(sweep["submissions"], desc="Threshold sweep", unit="model", dynamic_ncols=True):
        path = SUBMISSIONS / item["file"]
        try:
            remote = remote_files()
            if item["file"] in remote:
                # A prior upload may have succeeded just before an interruption.
                # Never adopt another person's submission merely because its ID matches.
                if not item.get("sha256"):
                    raise ValueError(f"Submission ID already used remotely: {item['file']}")
                digest = hashlib.sha256(mc("cat", BUCKET + item["file"], binary=True)).hexdigest()
                if digest != item["sha256"]:
                    raise ValueError(f"Remote submission ID collision: {item['file']}")
                item["state"] = "uploaded"
                item.pop("error", None)
                write_json(manifest, sweep)
                continue

            if not path.exists():
                item["state"] = "training"
                write_json(manifest, sweep)
                print(f"\nTraining {item['file']}: max_taxi_sec={item['max_taxi_sec']:,}", flush=True)
                subprocess.run(training_command(sweep, item), cwd=ROOT, check=True)
            path = check_artifact(sweep, item)
            item["state"] = "trained"
            item.pop("error", None)
            plot_training_curves(sweep, manifest)
            write_json(manifest, sweep)

            if item["file"] in remote_files():
                raise ValueError(f"Submission ID was taken during training: {item['file']}")
            # Attach the threshold to the remote object as well as the local manifest.
            attrs = (f"sweep-id={sweep['sweep_id']};max-taxi-sec={item['max_taxi_sec']};"
                     f"sha256={item['sha256']}")
            mc("cp", "--attr", attrs, path, BUCKET + item["file"])
            item.update(state="uploaded", uploaded_at=now())
            write_json(manifest, sweep)
            print(f"Uploaded {item['file']} (threshold {item['max_taxi_sec']:,} s)", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            item["error"] = str(error) or type(error).__name__
            write_json(manifest, sweep)
            print(f"\nResume with: {sys.executable} VALIDATION/tools/sweep_taxi_thresholds.py "
                  f"--resume {manifest}", flush=True)
            raise
    sweep["uploads_completed_at"] = now()
    plot_training_curves(sweep, manifest)
    write_json(manifest, sweep)
    refresh_scores(sweep, remote_files())
    write_json(manifest, sweep)
    show(sweep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="read-only ID preview; no training or uploads")
    mode.add_argument("--resume", type=Path, help="continue an existing sweep manifest")
    mode.add_argument("--status", type=Path, help="refresh a manifest's official scores without training")
    ap.add_argument("--threads", type=int, help="LightGBM threads for a new sweep (default 8)")
    args = ap.parse_args(argv)
    if args.threads is not None and (args.threads < 1 or args.resume or args.status):
        ap.error("--threads must be positive and is only used for new sweeps")
    if args.status:
        sweep = json.loads(args.status.read_text())
        refresh_scores(sweep, remote_files())
        write_json(args.status, sweep)
        show(sweep)
        return
    if args.dry_run:
        inputs()
        show({"submissions": plan(remote_files())})
        print("Preview only. IDs will be checked again when the sweep starts.")
        return

    # Linux and macOS: release automatically on exit, including interrupted runs.
    SUBMISSIONS.mkdir(exist_ok=True)
    with (SUBMISSIONS / ".taxi-threshold-sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            manifest = args.resume
            sweep = json.loads(manifest.read_text())
            if sweep["source"] != source_identity() or sweep["data"] != data_identity(inputs()):
                raise ValueError("Source or data changed since this sweep; start a new sweep")
        else:
            sweep_id = "taxi_tail_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            manifest = SWEEPS / f"{sweep_id}.json"
            config = {key: value for key, value in DEFAULTS.items() if key != "max_taxi_sec"}
            if args.threads is not None:
                config["threads"] = args.threads
            sweep = {"sweep_id": sweep_id, "created_at": now(), "validation": "skipped",
                     "training_months": "2025-01 through 2025-12", "seed": 42,
                     "config": config, "source": source_identity(), "data": data_identity(inputs()),
                     "submissions": plan(remote_files())}
            write_json(manifest, sweep)
        print(f"Sweep manifest: {manifest}", flush=True)
        show(sweep)
        execute(sweep, manifest)
        print(f"\nRefresh pending scores: {sys.executable} VALIDATION/tools/sweep_taxi_thresholds.py "
              f"--status {manifest}")


if __name__ == "__main__":
    main()
