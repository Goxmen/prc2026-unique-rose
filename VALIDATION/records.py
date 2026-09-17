"""Reproducible run records and protocol-separated experiment metadata."""

import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "VALIDATION" / "runs"
ARCHIVE_RUNS = ROOT / "VALIDATION" / "archive" / "runs"
LOG = RUNS / "experiments.jsonl"
MODEL_FILES = ["pipeline/common.py", "pipeline/recovery.py", "pipeline/recovery_relaxed.py",
               "pipeline/recovery_history.py", "pipeline/clock_features.py", "pipeline/timestamp_mixture.py",
               "VALIDATION/features_time.py", "VALIDATION/modeling.py",
               "VALIDATION/movement_data.py"]


def runtime_identity():
    import duckdb
    import lightgbm
    import numpy
    import pandas
    cpuinfo = Path('/proc/cpuinfo')
    cpu = platform.processor()
    if cpuinfo.exists():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    return {'cpu': cpu, 'platform': platform.platform(), 'python': platform.python_version(),
            'logical_cpus': os.cpu_count(),
            'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'versions': {m.__name__: m.__version__ for m in [duckdb, lightgbm, numpy, pandas]}}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append_log(record, directory=RUNS, replace_existing=False):
    log = Path(directory) / "experiments.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in record.items() if k not in {
        "folds", "models", "source", "model_source", "prediction_provenance", "report"}}
    summary["report"] = {"scores": record.get("report", {}).get("scores", {})}
    if replace_existing:
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
        rows = [row for row in rows if row["run_id"] != record["run_id"]] + [summary]
        temp = log.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(clean(row), allow_nan=False) + "\n" for row in rows))
        temp.replace(log)
        return
    with log.open("a") as fh:
        fh.write(json.dumps(clean(summary), allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_identity(paths):
    paths = sorted(Path(p) for p in paths)
    return [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths]


def source_snapshot(outdir):
    paths = sorted(set([ROOT / "requirements.txt", ROOT / ".gitignore"] +
                       list((ROOT / "pipeline").glob("*.py")) +
                       list((ROOT / "VALIDATION").glob("*.py")) +
                       list((ROOT / "VALIDATION" / "tools").glob("*.py")) +
                       list((ROOT / "VALIDATION" / "tests").glob("*.py")) +
                       list((ROOT / "VALIDATION").glob("*.md")) +
                       list((ROOT / "VALIDATION" / "docs").glob("*.md")) +
                       list((ROOT / "VALIDATION" / "reports").glob("*.md"))))
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        dest = Path(outdir) / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(relative)] = sha256(path)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", *[str(p.relative_to(ROOT)) for p in paths]],
                                    cwd=ROOT, text=True)
    (Path(outdir) / "source.patch").write_text(patch)
    return {"commit": head, "files": hashes, "patch": "source.patch",
            "untracked_sources_included": True}


def read_run(run, runs=RUNS):
    path = Path(run)
    if not path.is_file() and path.name != "result.json":
        path = path / "result.json" if path.is_dir() or len(path.parts) > 1 else Path(runs) / run / "result.json"
    if not path.is_file():
        # Old IDs and explicit pre-cleanup paths still resolve to their archived run.
        try:
            relative = path.resolve().relative_to(Path(runs).resolve())
        except ValueError:
            pass
        else:
            archived = Path(runs).resolve().parent / "archive" / "runs" / relative
            if archived.is_file():
                path = archived
    return json.loads(path.read_text()), path.parent
