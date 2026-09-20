# Weighted CV: five more approaches

**Weighted CV still meets your two criteria across all eleven submissions.**

| Check | Current CV | Weighted CV |
|---|---:|---:|
| Average absolute gap, all 11 submissions | 68.05 s | 23.42 s |
| Correct directions, all 10 comparisons | 7/10 | 7/10 |
| Average absolute gap, five additions only | 70.36 s | 22.53 s |
| Correct directions, five additions only | 3/5 | 3/5 |

The five additions each change a different part of v15:

| Submission / approach | Current CV | Weighted CV | Actual LB |
|---|---:|---:|---:|
| v17 — Remove month | 272.51 | 320.30 | 345.90 |
| v18 — Remove departure history | 274.63 | 323.50 | 342.50 |
| v19 — Add clock features | 273.13 | 321.24 | 338.11 |
| v20 — Use smaller trees | 274.56 | 323.70 | 345.31 |
| v21 — Exclude rule-handled rows from training | 267.58 | 312.82 | 342.40 |

| Change from v15 | Actual LB change | Current CV change | Weighted CV change |
|---|---:|---:|---:|
| v17 | +1.587 | -1.862 | -2.548 |
| v18 | -1.806 | +0.261 | +0.649 |
| v19 | -6.200 | -1.242 | -1.614 |
| v20 | +1.000 | +0.194 | +0.847 |
| v21 | -1.911 | -6.795 | -10.028 |

All values are seconds of RMSE. Negative changes mean improvement.

For LB changes of at least 1 s, overall direction accuracy is 5/7 → 5/7. v20 changes LB by 0.9999 s, so it is included in the main count but falls just below this threshold.

Error in predicting the size of a change is 5.21 → 9.66 s overall. This is separate from your two criteria.

**Checks:** original predictions were recovered from the desktop and verified against all 162 saved file hashes, including the overnight control. Every contributing fold/seed score and every original combined score reproduces. Flight IDs and labels match the earlier six cases. The weights, folds and scoring formula are unchanged. No model training or submissions ran.

**Limits:** this is a historical comparison. All five added approaches share v15 as their parent; they are not five independent datasets. The JSON also reports each seed, p99 clipping, and differences measured against the overnight control replay. Primary comparisons use the original scored v15, since the replay was not submitted.

Reproduce: `.venv/bin/python VALIDATION/tools/extend_weighted_cv_leaderboard.py`. [Full results](weighted_cv_leaderboard_extended.json); [original six-case comparison](WEIGHTED_CV_LEADERBOARD.md).
