# Frozen CV selection before confirmation scores

Frozen at 2026-09-17T04:16:21.988319+00:00. Selected: **R_A__mse**.

Primary selection uses seven candidate-minus-Q0-4h changes: four historical alternatives and three discovery candidates. New candidates use the regenerated control; historical ones use historical Q0-4h. The control LB is 344.3114 seconds. The two confirmation LB outcomes were not supplied to this command.

Twenty fixed forecast policies combine the ten unchanged CV readouts with either RMSE-delta transport or MSE-delta transport (`__mse`). MSE transport computes `sqrt(anchor_LB² + candidate_CV² − control_CV²)` and rejects a negative radicand. Raw CV changes and transported forecast changes are stored separately. This amendment was specified before new leaderboard outcomes were revealed.

| Metric | Anchored delta MAE | Raw level MAE | Misorderings | Missed separations | Selection regret |
|---|---:|---:|---:|---:|---:|
| R_combined | 5.6301 | 69.9873 | 3 | 4 | 7.7871 |
| R_A | 4.8312 | 48.7661 | 1 | 7 | 6.3708 |
| R_B_long | 6.8735 | 88.0949 | 5 | 6 | 7.7871 |
| month_balanced | 5.3140 | 48.1115 | 2 | 6 | 6.3708 |
| unique_flight | 5.3591 | 46.1104 | 2 | 6 | 6.3708 |
| seasonal_10m | 8.2025 | 35.9838 | 4 | 3 | 6.3708 |
| seasonal_11m | 6.8221 | 34.6492 | 3 | 2 | 0.0000 |
| blend_25 | 6.3186 | 15.7020 | 3 | 2 | 0.0000 |
| blend_50 | 5.9849 | 8.0464 | 2 | 5 | 0.0000 |
| blend_75 | 5.6488 | 25.4584 | 1 | 7 | 0.0000 |
| R_combined__mse | 3.9445 | 69.9873 | 2 | 6 | 7.7871 |
| R_A__mse | 3.7343 | 48.7661 | 1 | 7 | 6.3708 |
| R_B_long__mse | 4.4857 | 88.0949 | 4 | 8 | 7.7871 |
| month_balanced__mse | 4.1739 | 48.1115 | 1 | 7 | 6.3708 |
| unique_flight__mse | 4.2630 | 46.1104 | 2 | 6 | 6.3708 |
| seasonal_10m__mse | 9.2102 | 35.9838 | 4 | 2 | 6.3708 |
| seasonal_11m__mse | 7.8379 | 34.6492 | 3 | 2 | 0.0000 |
| blend_25__mse | 6.6838 | 15.7020 | 3 | 2 | 0.0000 |
| blend_50__mse | 5.8725 | 8.0464 | 2 | 5 | 0.0000 |
| blend_75__mse | 5.0364 | 25.4584 | 1 | 7 | 0.0000 |

| Held-out recipe | Selected anchored LB forecast | Existing v3 anchored LB forecast |
|---|---:|---:|
| small_trees | 344.6264 | 344.5137 |
| rule_routing | 336.0501 | 337.5251 |

Leave-one-family-out sensitivity (diagnostic only; primary selection unchanged):

| Family held out | Winner using other families | Held-out family delta MAE | Existing v3 held-out delta MAE |
|---|---|---:|---:|
| historical_rule_recovery | blend_25__mse | 11.1745 | 7.2319 |
| no_month | R_combined__mse | 3.0625 | 3.4410 |
| no_history | R_A__mse | 2.0123 | 2.0762 |
| clock_features | R_A__mse | 3.9567 | 4.9663 |

Calibration selects a hypothesis. Two subsequent outcomes assess it without retuning; all scores share the same hidden LB population. Pair counts are dependent descriptions, not independent evidence.

Additive level calibration is stored separately and did not select the metric. Conditional CV bootstrap intervals do not quantify hidden-label LB uncertainty.
