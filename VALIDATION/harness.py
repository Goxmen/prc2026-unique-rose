"""The standing validation harness for the PRC taxi-out task.

Folds are walk-forward and every one of them obeys the freeze rule: features
built from past departure taxi-outs are computed for validation rows at the
fold's TRAINING CUTOFF, never at the row's own date. That is the one thing the
2026 test months force on us, so it is the one thing the harness must copy.

Fold roster
-----------
  iterate    apr / aug / oct   spring, summer, autumn. The everyday loop for
                               ranking ideas. Deliberately spread across
                               seasons so feature decisions are not all taken
                               on one weather regime.
  rehearsal  dec_jan26         train Jan-Nov, validate Dec, freeze Nov 30.
                               The analogue of the January 2026 island: a
                               winter month one day past the training cutoff.
  rehearsal  jul_stale         train Jan-Jun, validate Jul, freeze Jan 31.
                               The analogue of the July 2026 island: a summer
                               month whose taxi-out aggregates are ~6 months old.
  control    jul_fresh         the SAME fold with the freeze at Jun 30.

Why the control exists: jul_stale trains on 6 months while the real July model
will train on 12, so its absolute RMSE is not comparable to dec_jan26's and must
not be read as a forecast. The trustworthy quantity is the paired difference
jul_stale - jul_fresh, in which the training-size handicap cancels. That
difference is the staleness penalty, and it is the headline number here.

Every fold runs under several seeds, and every fold score carries a bootstrap
band, because a difference smaller than the band is not a result.

  smoke test (~2 min):  .venv/bin/python VALIDATION/harness.py --smoke
  full run:             .venv/bin/python VALIDATION/harness.py --label baseline
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "VALIDATION"))
from common import (  # noqa: E402
    CATEGORICAL, MAX_TAXI_SEC, MIN_TAXI_SEC, RANKING, RAW_COLS, TRAIN_GLOB,
    build_features, connect, rmse,
)
from features_time import FPL_FEATURES, TIME_FEATURES, TimeFeatures, day_index, fpl_flags  # noqa: E402

RUNS = ROOT / "VALIDATION" / "runs"
LOG = RUNS / "experiments.jsonl"

# Reporting threshold for the tail split. Fixed, so the ex-tail column stays
# comparable across runs even when the training filter is moved.
TAIL_SEC = 7200

PARAMS = {
    "objective": "regression", "metric": "rmse", "learning_rate": 0.05,
    "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.9,
    "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
}


@dataclass(frozen=True)
class Fold:
    name: str
    role: str
    train_end: str   # last day of training, inclusive
    val: str         # validation month, "YYYY-MM"
    freeze: str      # day the stale-able windows stop at, for validation rows
    default: bool = True  # in the roster unless --folds names something else


FOLDS = [
    Fold("apr", "iterate", "2025-03-31", "2025-04", "2025-03-31"),
    Fold("aug", "iterate", "2025-07-31", "2025-08", "2025-07-31"),
    Fold("oct", "iterate", "2025-09-30", "2025-10", "2025-09-30"),
    Fold("dec_jan26", "rehearsal_jan", "2025-11-30", "2025-12", "2025-11-30"),
    Fold("jul_stale", "rehearsal_jul", "2025-06-30", "2025-07", "2025-01-31"),
    Fold("jul_fresh", "control_jul", "2025-06-30", "2025-07", "2025-06-30"),
    # Off by default. With --train-months these give a fixed-length iterate set
    # that reaches further into the year than apr/aug/oct can.
    Fold("sep", "iterate", "2025-08-31", "2025-09", "2025-08-31", default=False),
    Fold("nov", "iterate", "2025-10-31", "2025-11", "2025-10-31", default=False),
]


def train_start(fold: Fold, months: int) -> pd.Timestamp | None:
    """First training day. None = expanding window (everything up to train_end).

    A fixed window is what makes fold scores comparable *to each other*: with an
    expanding window April trains on 3 months and October on 9, so a difference
    between them is mostly training size, not season. The cost is reach — 2025
    starts in January, so an N-month window cannot validate before month N+1.
    """
    if months <= 0:
        return None
    end = pd.Timestamp(fold.train_end)
    return end.replace(day=1) - pd.DateOffset(months=months - 1)


def load(con, sample: float):
    """2025 departures (target + raw columns) and arrivals (taxi-in only)."""
    frac = f"USING SAMPLE {sample * 100:.4f} PERCENT (bernoulli, 7)" if sample < 1 else ""
    print("loading 2025 departures…")
    dep = con.sql(f"""
        SELECT TAXITIME_SEC_mvt AS taxi_sec, {RAW_COLS}
        FROM read_parquet('{TRAIN_GLOB}')
        WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL
        {frac}
    """).df()
    print("loading 2025 arrivals…")
    arr = con.sql(f"""
        SELECT TAXITIME_SEC_mvt AS taxi_sec, ADES_mvt, MVT_TIME_UTC_mvt
        FROM read_parquet('{TRAIN_GLOB}')
        WHERE PHASE_mvt = 'ARR' AND TAXITIME_SEC_mvt IS NOT NULL
        {frac}
    """).df()
    print(f"  departures {len(dep):,} | arrivals {len(arr):,}")
    return dep, arr


def test_weights(con):
    """Row counts of the two real test islands — the pooling weights."""
    w = con.sql(f"""
        SELECT strftime(MVT_TIME_UTC_mvt, '%Y-%m') AS month, count(*) AS n
        FROM read_parquet('{RANKING}') WHERE PHASE_mvt = 'DEP' GROUP BY 1
    """).df().set_index("month")["n"]
    return int(w.get("2026-01", 0)), int(w.get("2026-07", 0))


def matrix(df, tf, *, freeze_day, lag_days=0, fpl_flag=False):
    parts = [build_features(df), tf.transform(df, freeze_day=freeze_day, lag_days=lag_days)]
    if fpl_flag:
        parts.append(fpl_flags(df))
    return pd.concat(parts, axis=1)


def as_categorical(X_tr, X_va):
    """Categories from the training part only; unseen levels in validation go to
    NaN. This is exactly what predict.py does to the ranking rows."""
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for c in CATEGORICAL:
        cats = pd.Index(pd.unique(X_tr[c].dropna()))
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_va[c] = pd.Categorical(X_va[c].where(X_va[c].isin(cats)), categories=cats)
    return X_tr, X_va


def fit_predict(X_tr, y_tr, X_va, seed, rounds, desc):
    params = dict(PARAMS, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    bar = tqdm(total=rounds, desc=desc, unit="rnd", leave=False, position=1)
    booster = lgb.train(
        params, lgb.Dataset(X_tr, y_tr, categorical_feature=CATEGORICAL),
        num_boost_round=rounds,
        callbacks=[lambda env: bar.update(env.iteration + 1 - bar.n)],
    )
    bar.close()
    return booster.predict(X_va), booster


def boot_se(sq, rng, n_boot=300):
    """Bootstrap SE of an RMSE. Taxi-out errors are heavy-tailed, so the normal
    approximation understates this."""
    idx = rng.integers(0, len(sq), (n_boot, min(len(sq), 200_000)))
    return float(np.std(np.sqrt(sq[idx].mean(axis=1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="unlabelled", help="what this run is testing; goes in the log")
    ap.add_argument("--seeds", type=int, default=3, help="repeats per fold (spread tells you the noise floor)")
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of rows, for quick iterations")
    ap.add_argument("--folds", default="", help="comma-separated subset, e.g. jul_stale,jul_fresh")
    ap.add_argument("--train-months", type=int, default=0,
                    help="fixed rolling training window of N months instead of an expanding one, so "
                         "fold scores are comparable to each other. Folds whose window would start "
                         "before 2025-01-01 are skipped. 0 = expanding (default).")
    ap.add_argument("--lag-train-days", type=int, default=0,
                    help="lag the stale-able features of TRAINING rows by N days. Tests whether a "
                         "model should be trained on features as stale as the ones it will be served. "
                         "Only the taxi-out windows move; fresh features are left alone.")
    ap.add_argument("--max-taxi-sec", type=int, default=MAX_TAXI_SEC,
                    help=f"upper bound of the TRAINING target filter (default {MAX_TAXI_SEC}). The 2026 "
                         "ground truth keeps its ~24h date-rollover rows, and under squared loss the "
                         "optimal prediction is the conditional mean *including* that tail mass — so "
                         "filtering them out of training may be costing leaderboard score. Raise it to "
                         "test that. The ex-tail column stays pinned at 7200s either way.")
    ap.add_argument("--fpl-flag", action="store_true",
                    help="add has_flight_plan, an explicit flag for the rows where the corrupt "
                         "labels concentrate. Expect little effect at the default filter (those rows "
                         "are not in training to learn from) and more once --max-taxi-sec keeps them.")
    ap.add_argument("--smoke", action="store_true", help="5%% sample, 100 rounds, 1 seed — a wiring check")
    args = ap.parse_args()

    if args.smoke:
        args.sample, args.rounds, args.seeds = 0.05, 100, 1
        args.label = f"smoke:{args.label}"

    wanted = args.folds.split(",") if args.folds else None
    folds, skipped = [], []
    for f in FOLDS:
        keep_fold = (f.name in wanted) if wanted else f.default
        if not keep_fold:
            continue
        st = train_start(f, args.train_months)
        if st is not None and st < pd.Timestamp("2025-01-01"):
            skipped.append(f"{f.name} (needs training from {st.date()})")
            continue
        folds.append((f, st))
    if skipped:
        print(f"skipped, {args.train_months}-month window does not fit in 2025: {', '.join(skipped)}")
    if not folds:
        raise SystemExit("no runnable folds")
    seeds = [42 + 100 * i for i in range(args.seeds)]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = RUNS / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    con = connect()
    dep, arr = load(con, args.sample)
    n_jan, n_jul = test_weights(con)

    # Two different filters, deliberately. `keep` chooses which rows the model
    # trains on and is tunable. The aggregate tables always use the robust
    # [MIN, TAIL_SEC] range: a single 24 h rollover row inside a stand's 90-day
    # window moves that mean by ~170 s and poisons the feature for everyone.
    keep = dep["taxi_sec"].between(MIN_TAXI_SEC, args.max_taxi_sec).to_numpy()
    agg_dep = dep["taxi_sec"].between(MIN_TAXI_SEC, TAIL_SEC).to_numpy()
    agg_arr = arr["taxi_sec"].between(MIN_TAXI_SEC, TAIL_SEC).to_numpy()
    print(f"aggregate tables: {agg_dep.sum():,} departures + {agg_arr.sum():,} arrivals "
          f"(robust <= {TAIL_SEC}s)")
    print(f"training rows:    {keep.sum():,} departures (filter <= {args.max_taxi_sec}s)")
    tf = TimeFeatures(dep[agg_dep], arr[agg_arr])

    dep_day = day_index(dep["MVT_TIME_UTC_mvt"])
    ym = pd.to_datetime(dep["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m").to_numpy()
    d_of = lambda s: int((pd.Timestamp(s) - pd.Timestamp("2025-01-01")).days)  # noqa: E731

    print(f"\nrun {run_id} | {len(folds)} folds x {args.seeds} seeds x {args.rounds} rounds"
          f"{' | TRAIN FEATURES LAGGED %dd' % args.lag_train_days if args.lag_train_days else ''}")

    rng = np.random.default_rng(0)
    results, jobs = {}, [(f, st, s) for f, st in folds for s in seeds]
    outer = tqdm(jobs, desc="fits", unit="fit", position=0)
    t0 = time.time()

    for fold, st, seed in outer:
        outer.set_postfix_str(f"{fold.name} seed={seed}")
        tr = (dep_day <= d_of(fold.train_end)) & keep      # training rows: filtered
        if st is not None:
            tr &= dep_day >= d_of(st.strftime("%Y-%m-%d"))
        va = ym == fold.val                                # validation rows: never filtered
        if not tr.any() or not va.any():
            continue

        X_tr = matrix(dep[tr], tf, freeze_day=None, lag_days=args.lag_train_days, fpl_flag=args.fpl_flag)
        X_va = matrix(dep[va], tf, freeze_day=d_of(fold.freeze), fpl_flag=args.fpl_flag)
        X_tr, X_va = as_categorical(X_tr, X_va)
        y_tr, y_va = dep.loc[tr, "taxi_sec"].to_numpy(float), dep.loc[va, "taxi_sec"].to_numpy(float)

        pred, booster = fit_predict(X_tr, y_tr, X_va, seed, args.rounds, f"{fold.name}/{seed}")
        pred = np.clip(pred, 0, None)
        r = results.setdefault(fold.name, {
            "role": fold.role, "train_end": fold.train_end, "val": fold.val,
            "train_start": str(st.date()) if st is not None else "2025-01-01",
            "train_months": args.train_months or None,
            "freeze": fold.freeze, "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "rmse_by_seed": {}, "sq": None, "airport": None, "importance": None,
        })
        r["rmse_by_seed"][str(seed)] = round(rmse(y_va, pred), 3)
        if r["sq"] is None:  # slices and OOF from the first seed
            r["sq"] = (y_va - pred) ** 2
            # A few dozen rows carry most of the squared error: LIRF date-rollover
            # artifacts (~24 h "taxi-outs"). They are present in the 2026 ground
            # truth too, so they are not droppable — but a headline RMSE that
            # moves only because of them is not a modelling result, and you can
            # only see that with the tail split out.
            tail = y_va > TAIL_SEC
            r["tail_n"] = int(tail.sum())
            r["tail_pct_sse"] = round(100 * float(r["sq"][tail].sum() / r["sq"].sum()), 1) if tail.any() else 0.0
            r["rmse_ex_tail"] = round(float(np.sqrt(r["sq"][~tail].mean())), 2)
            r["airport"] = (
                pd.DataFrame({"airport": dep.loc[va, "ADEP_mvt"].to_numpy(), "y": y_va, "p": pred})
                .groupby("airport").apply(lambda g: round(rmse(g["y"], g["p"]), 1), include_groups=False)
                .sort_values().to_dict()
            )
            r["importance"] = dict(sorted(
                zip(booster.feature_name(), booster.feature_importance("gain").round().astype(int).tolist()),
                key=lambda kv: -kv[1])[:12])
            pd.DataFrame({"y": y_va, "pred": pred,
                          "airport": dep.loc[va, "ADEP_mvt"].to_numpy(),
                          "mvt_time": dep.loc[va, "MVT_TIME_UTC_mvt"].to_numpy()}
                         ).to_parquet(outdir / f"oof_{fold.name}.parquet", index=False)
    outer.close()

    # ---------------------------------------------------------------- report
    for name, r in results.items():
        v = np.array(list(r["rmse_by_seed"].values()), float)
        r["rmse_mean"] = round(float(v.mean()), 2)
        r["rmse_seed_sd"] = round(float(v.std(ddof=1)) if len(v) > 1 else float("nan"), 2)
        r["rmse_boot_se"] = round(boot_se(r["sq"], rng), 2)
        r["mse"] = float(np.mean(r["sq"]))
        r.pop("sq")

    print(f"\n{'=' * 84}\nfold results — RMSE in seconds, {args.seeds} seed(s)\n{'=' * 84}")
    print(f"{'fold':<12} {'train span':<16} {'-> val':<9} {'freeze':<7} {'n_train':>9} "
          f"{'RMSE':>8} {'ex-tail':>8} {'seed sd':>8} {'boot se':>8} {'tail n':>7} {'tail%SSE':>9}")
    for name, r in results.items():
        span = f"{r['train_start'][5:]}..{r['train_end'][5:]}"
        print(f"{name:<12} {span:<16} {r['val']:<9} {r['freeze'][5:]:<7} {r['n_train']:>9,} "
              f"{r['rmse_mean']:>8.2f} {r['rmse_ex_tail']:>8.2f} {r['rmse_seed_sd']:>8.2f} "
              f"{r['rmse_boot_se']:>8.2f} {r['tail_n']:>7,} {r['tail_pct_sse']:>8.1f}%")
    print(f"\n  ex-tail = RMSE over rows with a true taxi-out <= {TAIL_SEC}s. The gap between")
    print("  the two columns is how much of your score is decided by a handful of corrupt rows.")
    print("  Judge modelling ideas on ex-tail; judge the leaderboard on RMSE.")

    derived = {}
    if {"jul_stale", "jul_fresh"} <= results.keys():
        a, b = results["jul_stale"], results["jul_fresh"]
        common = sorted(set(a["rmse_by_seed"]) & set(b["rmse_by_seed"]))
        d = np.array([a["rmse_by_seed"][s] - b["rmse_by_seed"][s] for s in common], float)
        derived["staleness_penalty_s"] = round(float(d.mean()), 2)
        derived["staleness_penalty_sd"] = round(float(d.std(ddof=1)) if len(d) > 1 else float("nan"), 2)
        print(f"\nstaleness penalty (jul_stale - jul_fresh, paired by seed): "
              f"{derived['staleness_penalty_s']:+.2f} s  (sd {derived['staleness_penalty_sd']:.2f})")
        print("  Cost of the taxi-out aggregates being ~6 months old. Training size cancels,")
        print("  so unlike the raw jul_stale score this number is comparable across runs.")

    if "dec_jan26" in results and "jul_stale" in results and (n_jan + n_jul):
        wj, wl = n_jan / (n_jan + n_jul), n_jul / (n_jan + n_jul)
        pooled = float(np.sqrt(wj * results["dec_jan26"]["mse"] + wl * results["jul_stale"]["mse"]))
        derived.update(pooled_projection_s=round(pooled, 2), w_jan=round(wj, 4), w_jul=round(wl, 4))
        print(f"\npooled leaderboard projection: {pooled:.2f} s"
              f"   (Jan weight {wj:.3f} on {n_jan:,} rows, Jul weight {wl:.3f} on {n_jul:,} rows)")
        print("  Squared errors pooled the way the organizers pool them — one RMSE over both")
        print("  months — not an average of two fold scores.")
        print("  Known biases, do not read this as a forecast:")
        print("   - jul_stale trained on 6 months, the real July model gets 12   (pessimistic)")
        print("   - neither rehearsal ages the MODEL, only its features          (optimistic)")
        print("   - 2026 may simply differ from 2025 in ways one year cannot show (unknown sign)")
        print("  Its job is to be tracked against official scores over several submissions;")
        print("  a stable offset is usable information, a change of direction is an alarm.")

    for name in ("dec_jan26", "jul_stale"):
        if name in results:
            print(f"\nper-airport RMSE — {name}")
            for apt, v in results[name]["airport"].items():
                print(f"    {apt}: {v:>8.1f} s")

    if "dec_jan26" in results:
        print("\ntop feature gains — dec_jan26")
        for k, v in results["dec_jan26"]["importance"].items():
            print(f"    {k:<28} {v:>14,}")

    record = {
        "run_id": run_id, "label": args.label,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"seeds": seeds, "rounds": args.rounds, "sample": args.sample,
                   "lag_train_days": args.lag_train_days, "train_months": args.train_months,
                   "max_taxi_sec": args.max_taxi_sec,
                   "params": PARAMS,
                   "fpl_flag": args.fpl_flag,
                   "time_features": TIME_FEATURES + (FPL_FEATURES if args.fpl_flag else [])},
        "folds": results, "derived": derived,
        "official_rmse": None,  # fill with log_official.py after a submission
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (outdir / "result.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    RUNS.mkdir(exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    print(f"\nwrote {outdir.relative_to(ROOT)}/result.json (+ per-fold OOF parquet)")
    print(f"appended to {LOG.relative_to(ROOT)}  —  run_id {run_id}")
    print(f"took {record['minutes']} min")


if __name__ == "__main__":
    main()
