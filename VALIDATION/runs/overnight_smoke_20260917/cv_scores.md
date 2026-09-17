# Overnight CV scores before new leaderboard outcomes

**EXECUTION CHECK ONLY; cannot select a validation metric.**

All sixteen chronological folds are retained. Seeds and repeated training origins average squared errors, not predictions. Seasonal holdouts are retrospective. The fixed blends combine month-balanced and eleven-month seasonal MSE; blend number is the percentage assigned to month-balanced MSE.

| Recipe | v3 combined | A | B long | Month balanced | Unique flight | 10m | 11m |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 358.0338 | 381.0649 | 338.6105 | 379.1125 | 381.3116 | 436.8944 | 437.1043 |
| no_month | 358.1892 | 381.3517 | 338.6489 | 379.5587 | 381.7889 | 437.9391 | 438.3267 |
| no_history | 358.8279 | 382.0020 | 339.2793 | 379.9337 | 382.1316 | 438.2033 | 438.3334 |
| clock_features | 357.2784 | 380.8375 | 337.3797 | 378.9353 | 381.1765 | 436.9267 | 436.9668 |
| small_trees | 358.0338 | 381.0649 | 338.6105 | 379.1125 | 381.3116 | 436.8944 | 437.1043 |
| rule_routing | 359.2135 | 382.3105 | 339.7351 | 380.3556 | 382.5532 | 438.5632 | 438.7908 |

The JSON retains every fold, monthly and per-seed score, artifact hash, and 2,000-draw paired interval (or the explicitly requested repetition count). No new official score is read by this command.
