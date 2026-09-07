"""Train the harness-validated configuration on all of 2025 and fill a submission.

This is the harness's model, not train.py's: the time features from
features_time.py, and the training filter that keep-tail showed is worth ~71 s
of pooled projection. train.py predates both and is left alone.

The one thing that needs care here is the feature horizon. Departure taxi-out
labels stop at 2025-12-31 and never resume, so the STALE_ABLE windows are frozen
there — exactly as jul_stale rehearsed. But arrival taxi-ins and all movement
timestamps ARE present in ranking.parquet for both test months, so those tables
are built from 2025 *plus* the ranking rows and stay current. Getting that wrong
in either direction is the difference between the score you rehearsed and a
surprise.

Run: .venv/bin/python VALIDATION/submit.py --version 3
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "VALIDATION"))
from common import (  # noqa: E402
    CATEGORICAL, MIN_TAXI_SEC, RANKING, RAW_COLS, SUBMISSIONS, TEAM, TEMPLATE,
    TRAIN_GLOB, connect,
)
from features_time import TimeFeatures  # noqa: E402
from harness import PARAMS, TAIL_SEC, as_categorical, matrix  # noqa: E402

FREEZE = "2025-12-31"  # last day a departure taxi-out will ever be known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True, help="N in unique-rose_vN.parquet")
    ap.add_argument("--max-taxi-sec", type=int, default=200_000,
                    help="training filter ceiling; 200000 is the keep-tail setting")
    ap.add_argument("--rounds", type=int, default=800, help="match the harness so the score transfers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpl-flag", action="store_true", help="only if the harness showed it helps")
    ap.add_argument("--run-id", default="", help="harness run this reproduces; goes in the log")
    args = ap.parse_args()

    con = connect()
    print("loading 2025…")
    dep = con.sql(f"""SELECT TAXITIME_SEC_mvt AS taxi_sec, {RAW_COLS}
        FROM read_parquet('{TRAIN_GLOB}')
        WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL""").df()
    arr = con.sql(f"""SELECT TAXITIME_SEC_mvt AS taxi_sec, ADES_mvt, MVT_TIME_UTC_mvt
        FROM read_parquet('{TRAIN_GLOB}')
        WHERE PHASE_mvt = 'ARR' AND TAXITIME_SEC_mvt IS NOT NULL""").df()
    print("loading ranking…")
    rank = con.sql(f"""SELECT MVT_ID_mvt AS mvt_id, {RAW_COLS}
        FROM read_parquet('{RANKING}') WHERE PHASE_mvt = 'DEP'""").df()
    rank_arr = con.sql(f"""SELECT TAXITIME_SEC_mvt AS taxi_sec, ADES_mvt, MVT_TIME_UTC_mvt
        FROM read_parquet('{RANKING}')
        WHERE PHASE_mvt = 'ARR' AND TAXITIME_SEC_mvt IS NOT NULL""").df()
    print(f"  2025 dep {len(dep):,} | 2025 arr {len(arr):,} | "
          f"ranking dep {len(rank):,} | ranking arr {len(rank_arr):,}")

    # Robust range for the aggregate tables; the tunable filter is training-only.
    agg_dep = dep["taxi_sec"].between(MIN_TAXI_SEC, TAIL_SEC).to_numpy()
    arr_all = pd.concat([arr, rank_arr], ignore_index=True)
    agg_arr = arr_all["taxi_sec"].between(MIN_TAXI_SEC, TAIL_SEC).to_numpy()
    tf = TimeFeatures(
        dep[agg_dep],                       # taxi-out windows: 2025 only, frozen below
        arr_all[agg_arr],                   # taxi-in windows: current through Jul 2026
        dep_events=pd.concat([dep, rank], ignore_index=True),
        arr_events=arr_all,
    )
    freeze_day = int((pd.Timestamp(FREEZE) - pd.Timestamp("2025-01-01")).days)
    print(f"feature tables span {tf.n_days} days; taxi-out windows frozen at {FREEZE} (day {freeze_day})")

    keep = dep["taxi_sec"].between(MIN_TAXI_SEC, args.max_taxi_sec).to_numpy()
    print(f"training on {keep.sum():,} departures (filter <= {args.max_taxi_sec}s)")
    X_tr = matrix(dep[keep], tf, freeze_day=None, fpl_flag=args.fpl_flag)
    X_rk = matrix(rank, tf, freeze_day=freeze_day, fpl_flag=args.fpl_flag)
    X_tr, X_rk = as_categorical(X_tr, X_rk)
    y = dep.loc[keep, "taxi_sec"].to_numpy(float)

    params = dict(PARAMS, seed=args.seed, bagging_seed=args.seed + 1,
                  feature_fraction_seed=args.seed + 2)
    bar = tqdm(total=args.rounds, desc="fit 2025", unit="rnd")
    booster = lgb.train(params, lgb.Dataset(X_tr, y, categorical_feature=CATEGORICAL),
                        num_boost_round=args.rounds,
                        callbacks=[lambda e: bar.update(e.iteration + 1 - bar.n)])
    bar.close()

    chunk = 20_000
    pred = np.clip(np.concatenate([
        booster.predict(X_rk.iloc[i:i + chunk])
        for i in tqdm(range(0, len(X_rk), chunk), desc="predict", unit="chunk")
    ]), 0, None)
    rank["pred"] = pred

    template = pq.read_table(TEMPLATE)
    ids = template.column("MVT_ID_mvt").to_pandas()
    filled = ids.map(rank.set_index("mvt_id")["pred"])
    assert len(ids) == len(rank), f"template {len(ids)} rows vs ranking {len(rank)}"
    assert not filled.isna().any(), "some template MVT_IDs got no prediction"
    out = template.set_column(
        template.schema.get_field_index("TAXITIME_SEC_mvt"), "TAXITIME_SEC_mvt",
        pa.array(np.rint(filled).astype(np.int32), type=pa.int32()))
    assert out.schema.equals(template.schema), "schema drifted from the template"

    SUBMISSIONS.mkdir(exist_ok=True)
    dest = SUBMISSIONS / f"{TEAM}_v{args.version}.parquet"
    pq.write_table(out, dest)
    meta = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness_run_id": args.run_id, "max_taxi_sec": args.max_taxi_sec,
        "rounds": args.rounds, "seed": args.seed, "fpl_flag": args.fpl_flag,
        "freeze": FREEZE, "train_rows": int(keep.sum()), "params": params,
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    v = filled.to_numpy()
    is_jan = pd.to_datetime(rank.set_index("mvt_id").loc[ids, "MVT_TIME_UTC_mvt"]).dt.month.to_numpy() == 1
    print(f"\nwrote {dest.relative_to(ROOT)}  ({len(v):,} rows)")
    print(f"  overall  mean {v.mean():7.1f}s  median {np.median(v):7.1f}s  max {v.max():8.1f}s")
    print(f"  Jan 2026 mean {v[is_jan].mean():7.1f}s  ({is_jan.sum():,} rows)")
    print(f"  Jul 2026 mean {v[~is_jan].mean():7.1f}s  ({(~is_jan).sum():,} rows)")
    print(f"\nsubmit:  mc cp {dest} opensky/prc-2026-{TEAM}/")
    print(f"then:    .venv/bin/python VALIDATION/log_official.py --run {args.run_id or '<run_id>'} "
          f"--official <score> --submission {dest.name}")


if __name__ == "__main__":
    main()
