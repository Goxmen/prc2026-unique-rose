"""Refit protocol v3 on all of 2025 and fill the official submission template.

--run-id loads a completed v3 experiment's model configuration and checks any
explicit overrides. Uploading remains a separate mc cp command.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import RANKING, SUBMISSIONS, TEAM, TEMPLATE, connect
from harness import load
from modeling import (PARAMS, training_matrix, validation_matrix, choose_rounds,
                      fit_booster, predict, month_rows)
from protocol import VERSION, SCORE_VERSION, MAX_TAXI_SEC
from records import read_run, write_json, sha256, MODEL_FILES
from movement_data import load_phase
from registered_rules import RULE_FILES, apply_rule
from timestamp_mixture import validate_recipe

FREEZE = "2025-12-31"
DEFAULTS = {"max_taxi_sec": MAX_TAXI_SEC, "rounds": 800, "fpl_flag": False,
            "lag_train_days": 0, "early_stopping_rounds": 0, "threads": 8,
            "prediction_threads": None, "recovery": "off", "prediction_rule": "off",
            "clock_features": False, "target_mode": "direct", "model_mode": "single"}


def resolve_config(args):
    run = read_run(args.run_id)[0] if args.run_id else None
    if run and (run.get("protocol") != VERSION or run.get("status") != "complete"):
        raise ValueError("--run-id must identify a completed v3 experiment")
    if run and run["config"]["sample"] != 1:
        raise ValueError("a sampled/smoke run cannot establish a submission recipe")
    config = {}
    for name, default in DEFAULTS.items():
        reference = run["config"].get(name, default) if run else default
        explicit = getattr(args, name, None)
        if run and explicit is not None and explicit != reference:
            raise ValueError(f"--{name.replace('_', '-')} differs from the linked validation run")
        config[name] = reference if explicit is None else explicit
    seed = args.seed if args.seed is not None else (run["config"]["seeds"][0] if run else 42)
    if run:
        if seed not in run["config"]["seeds"]:
            raise ValueError("submission seed was not evaluated in the linked run")
        model_source = run.get('model_source', run['source'])
        for name in MODEL_FILES:
            if model_source["files"].get(name) != sha256(ROOT / name):
                raise ValueError(f"{name} changed since validation; validate the current model code first")
        if run["config"]["params"] != PARAMS:
            raise ValueError("LightGBM parameters changed since validation")
        for name in RULE_FILES[config['prediction_rule']]:
            if run.get('rule_source', {}).get(name) != sha256(ROOT/name):
                raise ValueError('Prediction rule source changed since validation')
    if config["rounds"] < 1 or config["max_taxi_sec"] < 1:
        raise ValueError("rounds and training-label ceiling must be positive")
    if any(config[k] < 0 for k in ["lag_train_days", "early_stopping_rounds", "threads"]):
        raise ValueError("lag, early-stopping patience and threads must be nonnegative")
    if config['prediction_threads'] is not None and config['prediction_threads'] < 0:
        raise ValueError('prediction threads must be nonnegative')
    if config['target_mode'] == 'clock_residual' and not config['clock_features']:
        raise ValueError('clock residual mode requires clock features')
    validate_recipe(config)
    return config, seed, run


def fill_template(template, movement_ids, predictions):
    ids = pd.Index(movement_ids)
    wanted = pd.Index(template.column("MVT_ID_mvt").to_pandas())
    values = np.asarray(predictions, dtype=float)
    if ids.hasnans or not ids.is_unique or wanted.hasnans or not wanted.is_unique:
        raise ValueError("submission IDs must be nonmissing and unique")
    if len(ids) != len(wanted) or len(values) != len(ids) or set(ids) != set(wanted):
        raise ValueError("ranking and template IDs/row counts do not match")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite submission predictions")
    ordered = pd.Series(np.maximum(values, 0), index=ids).reindex(wanted).to_numpy()
    rounded = np.rint(ordered)
    if (rounded > np.iinfo(np.int32).max).any():
        raise ValueError("prediction exceeds the template's int32 range")
    field = template.schema.field("TAXITIME_SEC_mvt")
    out = template.set_column(template.schema.get_field_index(field.name), field,
                              pa.array(rounded.astype(np.int32), type=field.type))
    if not out.schema.equals(template.schema):
        raise ValueError("submission schema drifted from template")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", type=int, required=True, help="N in unique-rose_vN.parquet")
    ap.add_argument("--run-id", default="", help="completed v3 run ID or result.json; inherit its recipe")
    ap.add_argument("--max-taxi-sec", type=int, help="default 200000 or linked run's ceiling")
    ap.add_argument("--rounds", type=int, help="default 800 or linked run's fixed count/inner cap")
    ap.add_argument("--seed", type=int, help="default 42 or the linked run's first seed")
    ap.add_argument("--fpl-flag", action="store_true", default=None)
    ap.add_argument("--lag-train-days", type=int)
    ap.add_argument("--early-stopping-rounds", type=int, help="inner Jan-Nov fit / December selection")
    ap.add_argument("--threads", type=int)
    ap.add_argument('--prediction-threads', type=int)
    ap.add_argument("--recovery", choices=['off', 'exact', 'relaxed', 'history'])
    ap.add_argument('--prediction-rule', choices=list(RULE_FILES))
    ap.add_argument('--clock-features', action='store_true', default=None)
    ap.add_argument('--target-mode', choices=['direct', 'clock_residual'])
    ap.add_argument('--model-mode', choices=['single', 'timestamp_mixture'])
    ap.add_argument("--track-training-loss", action="store_true",
                    help="record training RMSE at each round and show it in the progress bar")
    args = ap.parse_args(argv)
    if args.version < 1:
        ap.error("--version must be positive")
    config, seed, run = resolve_config(args)
    dest = SUBMISSIONS / f"{TEAM}_v{args.version}.parquet"
    if dest.exists():
        raise ValueError(f"{dest} already exists; choose the next submission version")
    con = connect()
    dep, arr = load(con, 1.0)
    rank = load_phase(con, str(RANKING), "DEP")
    rank_arr = load_phase(con, str(RANKING), "ARR")
    con.close()
    if run:
        # Prevent a validated recipe from silently moving onto another dataset.
        for source in run["data_identity"]:
            path = ROOT / "data" / source["file"]
            if not path.is_file() or sha256(path) != source["sha256"]:
                raise ValueError(f"dataset changed since validation: {source['file']}")
    rounds, inner = choose_rounds(dep, arr, seed, config, FREEZE)
    prepared = []
    X, y, cats = training_matrix(dep, arr, config, prepared_out=prepared)
    history = {} if args.track_training_loss else None
    booster = fit_booster(X, y, seed, rounds, config, "refit all 2025", training_history=history)
    if history is not None:
        loss = {
            "file": dest.name, "max_taxi_sec": config["max_taxi_sec"],
            "seed": seed, "train_rows": len(y), "metric": "training_rmse_seconds",
            'model_mode': config['model_mode'], 'target_mode': config['target_mode'],
        }
        if config['model_mode'] == 'timestamp_mixture':
            loss.update(metric='component_training_losses_before_mixture_and_rule',
                        components=history, **booster.training_summary)
        else:
            loss['rmse'] = history['training']['rmse']
        write_json(dest.with_suffix('.training_loss.json'), loss)
    prediction = pd.Series(index=rank.index, dtype=float)
    # Each released month is an island; do not let one prediction batch supply
    # unavailable intermediate history to another.
    for month in sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime("%Y-%m").unique()):
        part = month_rows(rank, month)
        V = validation_matrix(dep, arr, part, month_rows(rank_arr, month), FREEZE, cats, config,
                              prepared_training=prepared[0] if prepared else None)
        model_prediction = predict(booster, V, threads=config['prediction_threads'], target_mode=config['target_mode'])
        prediction.loc[part.index] = apply_rule(model_prediction, part, config['prediction_rule'])[0]
    out = fill_template(pq.read_table(TEMPLATE), rank.mvt_id, prediction.to_numpy())
    SUBMISSIONS.mkdir(exist_ok=True)
    pq.write_table(out, dest)
    booster.save_model(str(dest.with_suffix(".model.txt")))
    write_json(dest.with_suffix(".meta.json"), {
        "protocol": VERSION, "score_version": SCORE_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness_run_id": run["run_id"] if run else None, "config": config, "seed": seed,
        "selected_rounds": rounds, "inner_selection": inner, "train_rows": len(y),
        "freeze": FREEZE, "features": X.columns.tolist(), "params": PARAMS,
        "model_source": {name: sha256(ROOT / name) for name in MODEL_FILES},
        "rule_source": {name: sha256(ROOT/name) for name in RULE_FILES[config['prediction_rule']]},
        "full_benchmark_validated": bool(run and run.get("full_benchmark")),
    })
    print(f"Wrote {dest}: {out.num_rows:,} rows; protocol {VERSION}, seed {seed}, {rounds} rounds.")
    print(f"Upload with: mc cp {dest} opensky/prc-2026-{TEAM}/")


if __name__ == "__main__":
    main()
