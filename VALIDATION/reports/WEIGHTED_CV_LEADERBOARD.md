# Weighted CV versus actual leaderboard scores

The [expanded comparison](WEIGHTED_CV_LEADERBOARD_EXTENDED.md) adds five approaches and covers eleven submissions. The original six-case results below are preserved.

**Weighted CV meets your two criteria on these submissions.**

| Check | Current CV | Weighted CV |
|---|---:|---:|
| Average absolute CV–LB gap, six submissions | 66.12 s | 24.16 s |
| Correct improvement/regression calls | 4/5 | 4/5 |
| Correct calls when LB changes by at least 1 s | 3/3 | 3/3 |

| Submission | Current CV | Weighted CV | Actual LB |
|---|---:|---:|---:|
| v12 — B0-800 | 348.98 | 442.04 | 406.43 |
| v13 — Q0 | 276.74 | 326.34 | 354.43 |
| v14 — Q1 | 276.19 | 327.36 | 354.00 |
| v15 — Q0-4h | 274.37 | 322.85 | 344.31 |
| v16 — Q1-4h | 273.72 | 323.10 | 344.48 |
| v38 — artifact combined | 236.32 | 267.58 | 279.38 |

All scores are RMSE in seconds; lower is better.

| Change | Actual LB change | Current CV change | Weighted CV change |
|---|---:|---:|---:|
| v12 → v13 | -51.998 | -72.240 | -115.695 |
| v13 → v14 | -0.430 | -0.544 | +1.016 |
| v13 → v15 | -10.119 | -2.367 | -3.490 |
| v14 → v16 | -9.518 | -2.472 | -4.255 |
| v15 → v16 | +0.171 | -0.649 | +0.252 |

Secondary check: average error in the size of a change is 7.19 → 15.42 s. This is separate from your two deciding criteria.

Capping weights at p99 gives a 26.27 s average gap and 4/5 correct directions. Individual training-seed results are saved in the JSON.

**Scope:** these are six historical submissions, with five related rule/recovery comparisons. v38 contributes to score gaps; its exact scored parent v34 has no local prediction files. The overnight series v17–v21 and later v26–v30/v33–v34 also lack their original prediction files here. They were excluded rather than replaced with model replays. These results do not establish performance on those feature changes or future submissions.

**Checks:** original submission manifests and record hashes match; all six unweighted scores and the existing v38 weighted result reproduce. Flights and labels match across submissions. The same fixed weights are used for every model. No taxi-out refits, uploads, fold removals or default-score changes.

Reproduce: `.venv/bin/python VALIDATION/tools/compare_weighted_cv_leaderboard.py`. Weights and four discriminator models are retained in `tmp/weighted_cv_leaderboard_20260920/`. Full results: [weighted_cv_leaderboard.json](weighted_cv_leaderboard.json).
