"""Protocol v3: six reusable training windows, 16 A/B evaluations, three seeds.

Run: .venv/bin/python VALIDATION/harness.py --label baseline-v3
Diagnostics: --folds A,B,G7,F1,W1; S1 adds feature-only staleness.
Compare: --baseline <v3 run ID>. Inspect dates and fit counts with --dry-run.
"""

import argparse
import glob
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import RAW_COLS, TRAIN_GLOB, RANKING, connect
from modeling import (PARAMS, training_matrix, validation_matrix, choose_rounds, fit_booster,
                      predict, prediction_frame, period_rows, month_rows)
from protocol import (VERSION, SCORE_VERSION, MAX_TAXI_SEC, MAIN_NAMES, DEFAULT_FOLDS,
                      select_folds, window_key)
from records import RUNS, write_json, append_log, read_run, source_snapshot, data_identity
from reporting import report, print_report


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--seeds", type=int, default=3, help="count of seeds 42,142,242,... (default 3)")
    ap.add_argument("--rounds", type=int, default=800, help="fixed rounds or inner-tuning cap (default 800)")
    ap.add_argument("--sample", type=float, default=1.0, help="row fraction; sampled scores cannot rank ideas")
    ap.add_argument("--folds", default=DEFAULT_FOLDS,
                    help="IDs or A,B,main,all; G7/F1/W1/S1 optional; legacy names accepted")
    ap.add_argument("--train-months", type=int, default=6,
                    help="main window (default 6); 0 expanding; G7/F1/W1 retain 5/11/11")
    ap.add_argument("--lag-train-days", type=int, default=0, help="age training departure-history features only")
    ap.add_argument("--max-taxi-sec", type=int, default=MAX_TAXI_SEC, help="training label ceiling (default 200000)")
    ap.add_argument("--fpl-flag", action="store_true", help="add has_flight_plan")
    ap.add_argument("--smoke", action="store_true", help="5%% sample, 100 rounds, one seed; execution check only")
    ap.add_argument("--baseline", help="compatible v3 run ID or result.json for paired comparison")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--block-days", type=int, default=1, help="paired calendar-day block length (default 1)")
    ap.add_argument("--early-stopping-rounds", type=int, default=0,
                    help="0 fixes rounds; positive patience enables inner last-training-month selection")
    ap.add_argument("--threads", type=int, default=8, help="LightGBM threads (default 8; 0 automatic)")
    ap.add_argument("--dry-run", action="store_true", help="show schedule without loading data")
    ap.add_argument("--output-dir", type=Path, default=RUNS, help="parent directory for run artifacts")
    return ap


def parse_args(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.smoke:
        args.sample, args.rounds, args.seeds = 0.05, 100, 1
        args.label = "smoke:" + args.label
    for name in ["seeds", "rounds", "max_taxi_sec", "bootstrap_reps", "block_days"]:
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    for name in ["train_months", "lag_train_days", "early_stopping_rounds", "threads"]:
        if getattr(args, name) < 0:
            ap.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0 < args.sample <= 1:
        ap.error("--sample must be in (0, 1]")
    return args


def plan(args):
    selected, skipped = select_folds(args.folds, args.train_months)
    windows = OrderedDict()
    print("Fold   training window          validation  gap  cutoff      context")
    for fold, start in selected:
        windows.setdefault(window_key(fold, start), []).append(fold)
        print(f"{fold.name:<6} {str(start.date())}..{fold.train_end}  {fold.val}     "
              f"{fold.gap:>2}  {fold.freeze if fold.departure_history else 'none':<10}  training + {fold.val}")
    if skipped:
        print("Skipped (window starts before 2025): " + ", ".join(skipped))
    fits = len(windows) * args.seeds
    print(f"{len(selected)} evaluations x {args.seeds} seeds; {len(windows)} reused models per seed; {fits} outer fits")
    if args.early_stopping_rounds:
        print(f"Inner round selection adds {fits} fits inside the training windows.")
    if args.sample < 1:
        print("SAMPLED EXECUTION CHECK: scores cannot rank models.")
    return selected, windows


def load(con, sample):
    frac = f"USING SAMPLE {sample * 100:.4f} PERCENT (bernoulli, 7)" if sample < 1 else ""
    dep = con.sql(f"""SELECT MVT_ID_mvt mvt_id, TAXITIME_SEC_mvt taxi_sec, {RAW_COLS}
        FROM read_parquet('{TRAIN_GLOB}') WHERE PHASE_mvt = 'DEP' {frac}""").df()
    arr = con.sql(f"""SELECT TAXITIME_SEC_mvt taxi_sec, ADES_mvt, MVT_TIME_UTC_mvt
        FROM read_parquet('{TRAIN_GLOB}') WHERE PHASE_mvt = 'ARR' {frac}""").df()
    if dep.mvt_id.isna().any() or dep.mvt_id.duplicated().any():
        raise ValueError("missing or duplicate departure movement IDs")
    if not np.isfinite(dep.taxi_sec.to_numpy(float)).all():
        raise ValueError("missing/nonfinite departure targets; validation rows may not be silently removed")
    if dep.MVT_TIME_UTC_mvt.isna().any() or arr.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("missing movement timestamps")
    print(f"Loaded {len(dep):,} departures and {len(arr):,} arrivals.")
    return dep, arr


def ranking_weights(con):
    counts = dict(con.sql(f"""SELECT strftime(MVT_TIME_UTC_mvt, '%Y-%m'), count(*)
        FROM read_parquet('{RANKING}') WHERE PHASE_mvt = 'DEP' GROUP BY 1""").fetchall())
    jan, jul = counts.get("2026-01", 0), counts.get("2026-07", 0)
    if not jan or not jul:
        raise ValueError("ranking must contain both January and July 2026 departures")
    return {"jan_rows": jan, "jul_rows": jul, "jan_fraction": jan / (jan + jul)}


def execute(dep, arr, windows, config, outdir):
    """Fit once per window/seed; validation features never reach model fitting."""
    folds, models = {}, {}
    model_dir = Path(outdir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for number, ((start, end, history), evals) in enumerate(windows.items(), 1):
        model_id = f"M{number}"
        td, ta = period_rows(dep, start, end), period_rows(arr, start, end)
        X, y, cats = training_matrix(td, ta, config, history)
        boosters = {}
        models[model_id] = {"train_start": start, "train_end": end, "train_rows": len(y),
                            "departure_history": history, "features": X.columns.tolist(), "seeds": {}}
        for seed in config["seeds"]:
            t0 = time.monotonic()
            rounds, inner = choose_rounds(td, ta, seed, config, end, history)
            t1 = time.monotonic()
            booster = fit_booster(X, y, seed, rounds, config, f"{model_id}/{seed}")
            model_path = model_dir / f"{model_id}_seed{seed}.txt"
            booster.save_model(str(model_path))
            boosters[seed] = booster
            models[model_id]["seeds"][str(seed)] = {
                "rounds": rounds, "inner_selection": inner, "model": str(model_path.relative_to(outdir)),
                "inner_seconds": round(t1 - t0, 1), "fit_seconds": round(time.monotonic() - t1, 1)}
        del X
        for fold in evals:
            vd, va = month_rows(dep, fold.val), month_rows(arr, fold.val)
            if vd.empty:
                raise ValueError(f"no validation rows for {fold.name}/{fold.val}")
            V = validation_matrix(td, ta, vd, va, fold.freeze, cats, config, history)
            frame = prediction_frame(vd, V, start, end, fold.val)
            fold_record = {**asdict(fold), "train_start": start, "model_id": model_id,
                           "context_months": [str(p) for p in pd.period_range(start, end, freq="M")] + [fold.val],
                           "predictions": {}}
            for seed, booster in boosters.items():
                pred = predict(booster, V)
                if not np.isfinite(pred).all():
                    raise ValueError(f"nonfinite predictions in {fold.name}/{seed}")
                file = f"oof_{fold.name}_seed{seed}.parquet"
                frame.assign(fold=fold.name, seed=seed, pred=pred).to_parquet(Path(outdir) / file, index=False)
                fold_record["predictions"][str(seed)] = file
            folds[fold.name] = fold_record
            print(f"Scored {fold.name} ({fold.val}) on {len(frame):,} flights using {model_id}.")
            write_json(Path(outdir) / "progress.json", {"models": models, "folds": folds})
    return folds, models


def main(argv=None):
    args = parse_args(argv)
    selected, windows = plan(args)
    if args.dry_run:
        return
    baseline = read_run(args.baseline) if args.baseline else None
    if baseline and baseline[0].get("protocol") != VERSION:
        raise ValueError("--baseline must identify a v3 run; legacy scores are not comparable")
    config = {k: v for k, v in vars(args).items() if k not in {"output_dir", "baseline", "dry_run"}}
    config["seeds"] = [42 + 100 * i for i in range(args.seeds)]
    config["params"] = PARAMS
    run_id = datetime.now(timezone.utc).strftime("v3_%Y%m%dT%H%M%S%fZ")
    outdir = args.output_dir.resolve() / run_id
    outdir.mkdir(parents=True)
    record = {"run_id": run_id, "protocol": VERSION, "score_version": SCORE_VERSION,
              "label": args.label, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": config, "baseline_id": baseline[0]["run_id"] if baseline else None,
              "full_benchmark": args.sample == 1 and args.train_months == 6 and
                                set(MAIN_NAMES) <= {f.name for f, _ in selected},
              "official_rmse": None, "status": "running"}
    t0 = time.monotonic()
    try:
        record["source"] = source_snapshot(outdir)
        record["data_identity"] = data_identity([*glob.glob(TRAIN_GLOB), RANKING])
        con = connect()
        record["ranking_weights"] = ranking_weights(con)
        dep, arr = load(con, args.sample)
        con.close()
        record["folds"], record["models"] = execute(dep, arr, windows, config, outdir)
        del dep, arr
        record["report"] = report(record, outdir, baseline)
        print_report(record["report"])
        record["status"] = "complete"
    except Exception as exc:
        record["status"], record["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["minutes"] = round((time.monotonic() - t0) / 60, 3)
        write_json(outdir / "result.json", record)
        append_log(record, args.output_dir)
    print(f"\nWrote {outdir}/result.json; all fold/seed predictions and source snapshot saved.")


if __name__ == "__main__":
    main()
