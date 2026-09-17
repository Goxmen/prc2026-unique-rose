# Held-out leaderboard confirmation

Frozen selected metric: **R_A__mse**. Existing baseline: **R_combined**.

| Recipe | Official LB | Selected forecast | Existing v3 forecast | Selected error | Existing error |
|---|---:|---:|---:|---:|---:|
| small_trees | 345.3113 | 344.6264 | 344.5137 | -0.6849 | -0.7976 |
| rule_routing | 342.4009 | 336.0501 | 337.5251 | -6.3508 | -4.8758 |

| Metric | Held-out delta MAE | Raw level MAE | Frozen additive level MAE | Misorderings | Missed separations | Selection regret |
|---|---:|---:|---:|---:|---:|---:|
| R_A__mse | 3.5178 | 52.2227 | 3.5491 | 0 | 0 | 0.0000 |
| R_combined | 2.8367 | 72.7859 | 2.4352 | 0 | 0 | 0.0000 |

Only the frozen selected metric versus untouched R_combined is confirmatory. Other fixed readouts are descriptive. Two correlated candidates on one LB population cannot establish independent-year reliability; no metric is reselected here.

The observed regret here compares choosing among the unchanged control and the two held-out candidates. Every pairwise decision and every fixed readout is retained in JSON; these overlapping pairs are not independent trials.
