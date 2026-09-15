"""Stage 0 CPU timings and recovery-off feature equivalence on Jan–Jun 2025.

Run with the project virtualenv. Fits are timing probes, never submissions or
model-selection results. Each isolated worker warms up then times a fresh fit.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import DATA, RAW_COLS, connect
from modeling import training_matrix, validation_matrix, fit_booster, predict
from movement_data import load_phase
from records import MODEL_FILES, data_identity, sha256, write_json


def worker(args):
    X = pd.read_parquet(args.cache / "X.parquet")
    y = np.load(args.cache / "y.npy")
    config = {"threads": args.threads}
    warm = fit_booster(X, y, 42, 20, config, f"warm-up {args.threads} threads")
    del warm
    started = time.time()
    history = {}
    if args.parallel_pair:
        def fit(seed):
            curve = {}
            return fit_booster(X, y, seed, args.rounds, config, f"shared matrix seed {seed}",
                               training_history=curve), curve
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(fit, [42, 142]))
        booster, history = results[0]
    else:
        booster = fit_booster(X, y, 42, args.rounds, config, f"timing {args.threads} threads",
                              training_history=history)
    ended = time.time()
    probe = predict(booster, X.iloc[::100])
    np.save(args.worker.with_suffix(".npy"), probe)
    write_json(args.worker, {
        "threads": args.threads, "rounds": args.rounds, "fit_seconds": ended - started,
        "workers": 2 if args.parallel_pair else 1,
        "fit_started_epoch": started, "fit_ended_epoch": ended,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "train_rmse": history["training"]["rmse"][-1],
        "prediction_sha256": hashlib.sha256(probe.tobytes()).hexdigest(),
    })


def prepare(cache):
    files = sorted(DATA.glob("training_2025-*.parquet"))
    if len(files) != 12:
        raise ValueError("Expected all 12 original training files")
    first_six = [str(path) for path in files[:6]]
    con = connect()
    dep = load_phase(con, first_six, "DEP")
    arr = load_phase(con, first_six, "ARR")
    old_dep = con.sql(f"SELECT MVT_ID_mvt mvt_id, TAXITIME_SEC_mvt taxi_sec, {RAW_COLS} "
                      "FROM read_parquet(?) WHERE PHASE_mvt='DEP'", params=[first_six]).df()
    old_arr = con.sql("SELECT TAXITIME_SEC_mvt taxi_sec, ADES_mvt, MVT_TIME_UTC_mvt "
                      "FROM read_parquet(?) WHERE PHASE_mvt='ARR'", params=[first_six]).df()
    pd.testing.assert_frame_equal(dep[old_dep.columns], old_dep)
    pd.testing.assert_frame_equal(arr[old_arr.columns], old_arr)
    config = {"max_taxi_sec": 200000, "lag_train_days": 0, "fpl_flag": False}
    X, y, cats = training_matrix(dep, arr, config)
    old_X, old_y, old_cats = training_matrix(old_dep, old_arr, config)
    pd.testing.assert_frame_equal(X, old_X)
    np.testing.assert_array_equal(y, old_y)
    for name in cats:
        pd.testing.assert_index_equal(cats[name], old_cats[name])
    del old_X, old_y
    val_dep = load_phase(con, str(files[6]), "DEP")
    val_arr = load_phase(con, str(files[6]), "ARR")
    V = validation_matrix(dep, arr, val_dep, val_arr, "2025-06-30", cats, config)
    old_V = validation_matrix(old_dep, old_arr, val_dep[old_dep.columns], val_arr[old_arr.columns],
                              "2025-06-30", old_cats, config)
    pd.testing.assert_frame_equal(V, old_V)
    cache.mkdir(parents=True, exist_ok=True)
    X.to_parquet(cache / "X.parquet", index=False)
    np.save(cache / "y.npy", y)
    con.close()
    return {"training_window": ["2025-01-01", "2025-06-30"], "validation_month": "2025-07",
            "training_rows": len(X), "validation_rows": len(V), "feature_names": X.columns.tolist(),
            "legacy_vs_expanded_loader": "exact equality: values, dtypes, row ordering, IDs",
            "training_matrix_equivalence": "exact", "validation_matrix_equivalence": "exact",
            "category_order_equivalence": "exact",
            "feature_matrix_sha256": sha256(cache / "X.parquet"),
            "data": data_identity(files[:7]),
            "source": {name: sha256(ROOT / name) for name in MODEL_FILES}}


def launch(cache, path, threads, rounds):
    log = path.with_suffix(".log").open("w")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(path),
                                "--cache", str(cache), "--threads", str(threads), "--rounds", str(rounds)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return process, log


def finish(process, log, path):
    code = process.wait()
    log.close()
    if code:
        raise RuntimeError(f"Timing worker failed; inspect {path.with_suffix('.log')}")
    result = json.loads(path.read_text())
    print(f"{path.stem}: {result['fit_seconds']:.2f} s, peak RSS {result['peak_rss_mib']:.0f} MiB", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=Path)
    ap.add_argument("--cache", type=Path, default=ROOT / "tmp/stage0/thread_benchmark")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--parallel-pair", action="store_true")
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = ROOT / "VALIDATION/benchmarks" / f"stage0_threads_{stamp}.json"
    print("Preparing full Jan–Jun matrices and checking legacy equivalence…", flush=True)
    report = prepare(args.cache)
    report.update(created_at=stamp, rounds=args.rounds, warmup_rounds=20, sequential=[], pairs=[])
    write_json(report_path, report)
    for i, threads in enumerate([8, 16, 16, 8, 8, 16], 1):
        path = args.cache / f"trial{i}_{threads}threads.json"
        result = finish(*launch(args.cache, path, threads, args.rounds), path)
        result["file"] = str(path)
        report["sequential"].append(result)
        write_json(report_path, report)
    # Independent processes avoid changing OpenMP settings inside a running fit.
    for repeat in range(2):
        jobs = []
        for n in range(2):
            path = args.cache / f"pair{repeat + 1}_worker{n + 1}.json"
            jobs.append((*launch(args.cache, path, 4, args.rounds), path))
        results = [finish(*job) for job in jobs]
        report["pairs"].append({"workers": 2, "threads_each": 4, "fits": results,
                                "fit_makespan_seconds": max(r["fit_ended_epoch"] for r in results)
                                - min(r["fit_started_epoch"] for r in results)})
        write_json(report_path, report)
    summary = {str(n): float(np.median([r["fit_seconds"] for r in report["sequential"]
                                      if r["threads"] == n])) for n in (8, 16)}
    reference = np.load(Path(report["sequential"][0]["file"]).with_suffix(".npy"))
    differences = [float(np.max(np.abs(np.load(Path(r["file"]).with_suffix(".npy")) - reference)))
                   for r in report["sequential"]]
    summary["max_probe_prediction_difference"] = max(differences)
    summary["two_4thread_fits_median_makespan"] = float(np.median([
        p["fit_makespan_seconds"] for p in report["pairs"]]))
    summary["two_sequential_8thread_fits_estimate"] = 2 * summary["8"]
    report["summary"] = summary
    write_json(report_path, report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
