# Ranking correction sensitivity

These are six CV-window coefficient sets applied separately to the existing scored v34 predictions. **This is not a full-year recipe or a prediction of hidden errors.** No model or submission file was created.

Departure labels and block timestamps are excluded by an explicit SQL projection. Handler signals use only each ranking month's own arrivals. Prediction changes are unrounded.

| Candidate | Eligible rows | Changed rows across windows | Mean absolute change on changed rows (s) | Largest change (s) |
|---|---:|---:|---:|---:|
| twins | 16–16 | 16–16 | 2253.5–2253.5 | 3330.0–3330.0 |
| clock12 | 1–1 | 1–1 | 2972.0–3279.8 | 2972.0–3279.8 |
| clock2 | 2–2 | 2–2 | 3109.5–3371.9 | 3247.0–3464.0 |
| day_shrunk | 5–5 | 5–5 | 3884.7–7711.0 | 5552.9–11042.3 |
| day_handler | 5–5 | 5–5 | 3884.7–7711.0 | 5552.9–11042.3 |
| rule_shrunk | 32–32 | 32–32 | 1081.8–2377.8 | 1830.3–4029.7 |
| combined_midgap | 54–54 | 54–54 | 1723.5–2829.5 | 5552.9–11042.3 |

**Clock12 transfer hypothesis:** 1 ranking movement(s) among 344,841, versus 1 unique changed CV movement(s) among 1,079,528. A higher affected frequency may produce a larger leaderboard gain than CV. The sample is too small to establish that.

Its headroom is small: at most 0.1927 s improvement across these coefficient sets **if** the hidden labels obey the empirical LOBT ±3660 s window and the endpoint assumptions are compatible with v34’s total squared error. This is a conditional bound.

**Day-pattern gains remain uncertain.** We cannot tell which ranking labels follow the schedule and which have a date shift.

**Conditional arithmetic only — not forecasts.** Assume the affected day-row labels equal the named clock. Keep all other squared errors fixed at the historical v34 total (280.4658 s; 344,841 rows). Assumptions whose affected-row control error exceeds that total are incompatible and have no score change. The table shows candidate-minus-v34 RMSE ranges across six coefficient sets.

| Day candidate | Labels = takeoff − schedule | Labels = 86400 + window late mean |
|---|---:|---:|
| day_shrunk | 0.490–1.930 s | -9.079–-5.015 s |
| day_handler | 0.490–1.930 s | -9.079–-5.015 s |

| Candidate | Example flight | Airport / stand | v34 (s) | Candidate range (s) |
|---|---|---|---:|---:|
| twins | WJA63 (212742519) | LFPG / C4A | 3021 | 6351.0–6351.0 |
| twins | RZO513 (212742817) | LFPG / Y05 | 3269 | 6599.0–6599.0 |
| clock12 | ITY340 (213147416) | LIRF / 204 | 83160 | 86132.0–86439.8 |
| clock2 | AFR19DE (212756714) | LFPG / F08 | 9960 | 13207.0–13424.0 |
| clock2 | ITY340 (213147416) | LIRF / 204 | 83160 | 86132.0–86439.8 |
| day_shrunk | WMTT3508 (202680942) | LIRF / 801 | 56520 | 62072.9–67562.3 |
| day_shrunk | SVII3006 (213154026) | LIRF / 106 | 60656 | 65465.4–70224.4 |
| day_handler | WMTT3508 (202680942) | LIRF / 801 | 56520 | 62072.9–67562.3 |
| day_handler | SVII3006 (213154026) | LIRF / 106 | 60656 | 65465.4–70224.4 |
| rule_shrunk | RYR4852 (213142746) | LIRF / 316 | 41395 | 37365.3–39564.7 |
| rule_shrunk | RYRR8830 (202681830) | LIRF / 318 | 40615 | 36663.3–38820.1 |
| combined_midgap | WMTT3508 (202680942) | LIRF / 801 | 56520 | 62072.9–67562.3 |
| combined_midgap | SVII3006 (213154026) | LIRF / 106 | 60656 | 65465.4–70224.4 |

The JSON companion retains every window summary, up to five examples per candidate, input hashes and coefficient fingerprints.
