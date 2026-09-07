"""Phase 0 — the availability audit. Run this before trusting any fold score.

No training, no model: just the facts the harness depends on. Answers, in order:

  1. What is actually populated for DEP rows in ranking.parquet?
  2. How close is each off-block estimate to the true off-block? AOBT_3_flt is
     the flown-trajectory actual off-block and it survives in ranking, so
     MVT_TIME - AOBT_3 may already *be* the target. If that error is ~0 the
     model is reading the answer off the page and every fold number below is
     noise around a near-perfect feature. This is the single most important
     check in the file.
  3. Is MVT_ID_mvt monotone with time? (an ID-derived feature would fake-win)
  4. Duplicated movements from the movement/flight join.
  5. Target shape: how heavy is the right tail, and what does that do to the
     resolution of an RMSE comparison?
  6. The Jan/Jul row counts that weight the pooled leaderboard projection.

Run: .venv/bin/python VALIDATION/audit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import MAX_TAXI_SEC, MIN_TAXI_SEC, RANKING, TRAIN_GLOB, connect  # noqa: E402

OBT_COLS = ["AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"]


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    con = connect()
    pd.set_option("display.width", 200)

    rule("1. ranking.parquet — what is populated for DEP rows")
    cols = [c for c in con.sql(f"SELECT * FROM read_parquet('{RANKING}') LIMIT 0").df().columns
            if c != "PHASE_mvt"]  # it is the grouping key; counting it too collides on the alias
    exprs = ",\n".join(f"       count(\"{c}\") AS \"{c}\"" for c in cols)
    pop = con.sql(f"""
        SELECT PHASE_mvt, count(*) AS rows_,
{exprs}
        FROM read_parquet('{RANKING}') GROUP BY PHASE_mvt ORDER BY PHASE_mvt
    """).df().set_index("PHASE_mvt").T
    pop["DEP_pct"] = (100 * pop["DEP"] / pop.loc["rows_", "DEP"]).round(1)
    print(pop.to_string())
    blank = [c for c in cols if pop.loc[c, "DEP"] == 0]
    print(f"\nfully blank for DEP: {blank or 'none'}")
    print("-> anything blank here must never enter the feature set, in training either.")

    rule("2. off-block estimates vs the true off-block (training DEP rows)")
    diffs = con.sql(f"""
        SELECT {', '.join(f"date_diff('second', {c}, BLOCK_TIME_UTC_mvt) AS {c}" for c in OBT_COLS)}
        FROM read_parquet('{TRAIN_GLOB}') WHERE PHASE_mvt = 'DEP' AND BLOCK_TIME_UTC_mvt IS NOT NULL
    """).df()
    rows = []
    for c in OBT_COLS:
        e = diffs[c].dropna()
        rows.append({
            "estimate": c, "coverage_%": round(100 * len(e) / len(diffs), 1),
            "exact_0s_%": round(100 * float((e == 0).mean()), 1),
            "within_60s_%": round(100 * float((e.abs() <= 60).mean()), 1),
            "median_abs_s": round(float(e.abs().median()), 1),
            "p90_abs_s": round(float(e.abs().quantile(0.90)), 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    exact = rows[0]["exact_0s_%"]
    print(f"\n-> AOBT_3_flt equals the true off-block exactly on {exact}% of rows.")
    if exact > 50:
        print("   WARNING: mvt_minus_aobt3_sec is then essentially the target itself.")
        print("   It is legal (organizers left it populated) but it means the model is")
        print("   near-deterministic, the RMSE is dominated by the minority of rows where")
        print("   AOBT_3 is wrong, and the whole staleness question is close to irrelevant.")
        print("   Decide this before reading any fold score.")
    else:
        print("   Low enough that the estimate is a genuine feature, not the answer.")

    rule("3. is MVT_ID_mvt monotone with time?")
    mono = con.sql(f"""
        SELECT corr(MVT_ID_mvt, epoch(MVT_TIME_UTC_mvt)) AS pearson_id_vs_time,
               count(*) AS n
        FROM read_parquet('{TRAIN_GLOB}')
    """).df()
    print(mono.to_string(index=False))
    print("-> |r| near 1 means an ID-derived feature would win locally and die on the test.")

    rule("4. duplicate movements")
    dup = con.sql(f"""
        SELECT count(*) AS total_rows,
               count(DISTINCT MVT_ID_mvt) AS distinct_mvt_id,
               count(*) - count(DISTINCT MVT_ID_mvt) AS duplicate_rows
        FROM read_parquet('{TRAIN_GLOB}')
    """).df()
    print(dup.to_string(index=False))

    rule("5. target shape (DEP, 2025)")
    y = con.sql(f"""
        SELECT TAXITIME_SEC_mvt AS y FROM read_parquet('{TRAIN_GLOB}')
        WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL
    """).df()["y"].to_numpy(dtype=float)
    q = np.percentile(y, [1, 25, 50, 75, 90, 99, 99.9])
    print(f"n={len(y):,}  mean={y.mean():.1f}s  sd={y.std():.1f}s")
    print("  percentiles 1/25/50/75/90/99/99.9: " + " / ".join(f"{v:.0f}" for v in q))
    kept = (y >= MIN_TAXI_SEC) & (y <= MAX_TAXI_SEC)
    print(f"  outside the [{MIN_TAXI_SEC}, {MAX_TAXI_SEC}]s training filter: {(~kept).sum():,} rows")
    resid = y - y.mean()
    top1 = np.sort(resid**2)[-len(y) // 100:].sum() / (resid**2).sum()
    print(f"  share of total squared error contributed by the worst 1% of rows: {100*top1:.1f}%")
    print("-> that share is why fold scores need a bootstrap band before you compare them.")

    rule("6. pooled-projection weights and the RMSE resolution you can expect")
    w = con.sql(f"""
        SELECT strftime(MVT_TIME_UTC_mvt, '%Y-%m') AS month, count(*) AS dep_rows
        FROM read_parquet('{RANKING}') WHERE PHASE_mvt = 'DEP' GROUP BY 1 ORDER BY 1
    """).df()
    w["weight"] = (w["dep_rows"] / w["dep_rows"].sum()).round(4)
    print(w.to_string(index=False))
    n_test = int(w["dep_rows"].sum())

    rng = np.random.default_rng(0)
    sq = resid[rng.integers(0, len(resid), min(len(resid), 400_000))] ** 2
    boot = np.array([np.sqrt(rng.choice(sq, n_test, replace=True).mean()) for _ in range(200)])
    se, base = float(boot.std()), float(np.sqrt((resid**2).mean()))
    rel = 100 * se / base
    print(f"\nNo model exists at audit time, so this uses the residuals of the CONSTANT-MEAN")
    print(f"predictor (RMSE {base:.1f} s) only to show what the tail does to precision:")
    print(f"  bootstrap SE at n={n_test:,}:  {se:.2f} s  =  {rel:.2f}% of that RMSE")
    print(f"-> Do NOT read {se:.1f} s as your leaderboard resolution: a real model's errors are")
    print(f"   much smaller, and the band scales with them. As a rough rule the SE stays near")
    print(f"   {rel:.1f}% of whatever RMSE you score, so a model at R s has a band around")
    print(f"   +/-{1.96 * rel / 100:.3f} x R. The harness computes the true per-fold band from")
    print("   actual model residuals — use that one, and treat anything inside it as a tie.")

    print("\nAudit complete. Nothing was trained and nothing was written.")


if __name__ == "__main__":
    main()
