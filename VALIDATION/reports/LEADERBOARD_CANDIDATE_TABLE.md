# Selected candidates: CV and leaderboard scores

The table from [LEADERBOARD_CANDIDATES.md](../docs/LEADERBOARD_CANDIDATES.md),
with the confirmed official leaderboard (LB) scores and submission versions added.
All values are **RMSE in seconds; lower is better**. All five submissions succeeded.

| Order | Codename | Actual recipe | CV next-month | CV longer-gap | CV combined | Official LB | Submission |
|---|---|---|---:|---:|---:|---:|---|
| 1 | **B0-800** | Original baseline model; recovery off; no prediction rule | 366.7465 | 334.1785 | 348.9770 | 406.4281 | v12 |
| 2 | **Q0** | B0-800 plus the **six-hour Rome rule** | 300.5904 | 256.1971 | 276.7374 | 354.4303 | v13 |
| 3 | **Q1** | Baseline with **R3 flight-plan recovery**, plus the **six-hour Rome rule** | 296.9037 | 258.5505 | 276.1937 | 354.0004 | v14 |
| 4 | **Q0-4h** | B0-800 plus the same Rome rule at **four hours** | 296.2785 | 255.6200 | 274.3707 | 344.3114 | v15 |
| 5 | **Q1-4h** | Baseline with **R3 flight-plan recovery**, plus the Rome rule at **four hours** | 292.8785 | 257.4787 | 273.7215 | 344.4821 | v16 |

CV uses the standard three-seed loss average; combined CV is a provisional summary.
Final submissions use seed 42, 800 rounds and all twelve training months.

**Best official result in this batch: Q0-4h (v15), 344.3114 seconds.**

Official scores confirmed 2026-09-16T16:46:32+00:00 in the
[submission manifest](leaderboard_selected_20260916.json). See the
[full comparison report](LEADERBOARD_RESULTS.md) for LB–CV gaps, changes and uncertainty.
