# Seasonal validation against recorded leaderboard scores

Retrospective 2025 holdouts; the official scores below are fixed references, not new submissions.

Official results recorded 2026-09-16T16:46:32+00:00. Seeds: [42, 142, 242].
Official LB is pooled across both months; no official January-only or July-only scores are available.
January and July are scored separately; combined RMSE weights monthly mean squared errors by the 2026 ranking populations (January weight 0.44286787).
Standard scores average squared errors across seeds, not predictions. Seed 42 matches the submission seed.
Training features remain causal within retained months. Validation departure history is frozen at December 31, using training-month labels only. Future-in-calendar training is intentional; these scores do not measure forward-year forecasting.

Direction tolerance: 1.00 s, applied to both LB and CV deltas. Pointwise paired 95% intervals resample calendar days within each held-out month.
These five related submissions are the calibration set; later candidates must test whether alignment generalizes.

## Alignment with the same five LB references

| Strategy | Seed readout | Mean absolute score gap | Mean absolute delta gap | Agree / material LB pairs | Disagree | CV within tolerance |
|---|---|---:|---:|---:|---:|---:|
| existing-v3 | standard | 70.7304 | 7.1947 | 3 / 3 | 0 | 0 |
| existing-v3 | seed42 | 70.8367 | 7.4216 | 3 / 3 | 0 | 0 |
| 10m | standard | 36.2145 | 12.4461 | 3 / 3 | 0 | 0 |
| 10m | seed42 | 35.3824 | 11.9779 | 3 / 3 | 0 | 0 |
| 11m | standard | 35.7289 | 10.5048 | 3 / 3 | 0 | 0 |
| 11m | seed42 | 36.0751 | 10.0746 | 3 / 3 | 0 | 0 |

## 10m: standard

| Recipe | January RMSE | July RMSE | Combined RMSE | Official LB | LB − combined |
|---|---:|---:|---:|---:|---:|
| B0-800 | 404.6957 | 519.5482 | 472.1432 | 406.4281 | -65.7151 |
| Q0 | 361.2994 | 402.2779 | 384.6688 | 354.4303 | -30.2385 |
| Q1 | 360.8499 | 386.4428 | 375.3239 | 354.0004 | -21.3235 |
| Q0-4h | 360.7576 | 392.9176 | 379.0118 | 344.3114 | -34.7004 |
| Q1-4h | 360.5781 | 383.5962 | 373.5772 | 344.4821 | -29.0951 |

| Candidate − reference | ΔLB | ΔJanuary | ΔJuly | ΔCombined [95% interval] | Direction | Both months supported gains |
|---|---:|---:|---:|---|---|---|
| Q0 − B0-800 | -51.9978 | -43.3964 | -117.2703 | -87.4744 [-153.4004, -27.2704] | agree | True |
| Q1 − Q0 | -0.4299 | -0.4494 | -15.8351 | -9.3448 [-17.9887, -3.4191] | LB change within tolerance | False |
| Q0-4h − Q0 | -10.1189 | -0.5418 | -9.3603 | -5.6570 [-10.6229, -2.0111] | agree | False |
| Q1-4h − Q1 | -9.5183 | -0.2718 | -2.8466 | -1.7467 [-5.2728, +1.3361] | agree | False |
| Q1-4h − Q0-4h | +0.1707 | -0.1794 | -9.3214 | -5.4345 [-10.3003, -1.6933] | LB change within tolerance | False |

## 10m: seed42

| Recipe | January RMSE | July RMSE | Combined RMSE | Official LB | LB − combined |
|---|---:|---:|---:|---:|---:|
| B0-800 | 403.1719 | 519.0848 | 471.2810 | 406.4281 | -64.8529 |
| Q0 | 361.0456 | 400.1151 | 383.3041 | 354.4303 | -28.8738 |
| Q1 | 361.0352 | 386.0583 | 375.1823 | 354.0004 | -21.1819 |
| Q0-4h | 360.4074 | 390.4642 | 377.4484 | 344.3114 | -33.1370 |
| Q1-4h | 360.7872 | 383.0399 | 373.3486 | 344.4821 | -28.8665 |

| Candidate − reference | ΔLB | ΔJanuary | ΔJuly | ΔCombined [95% interval] | Direction | Both months supported gains |
|---|---:|---:|---:|---|---|---|
| Q0 − B0-800 | -51.9978 | -42.1263 | -118.9697 | -87.9769 [-154.2156, -27.4708] | agree | True |
| Q1 − Q0 | -0.4299 | -0.0105 | -14.0569 | -8.1218 [-17.0982, -2.0081] | LB change within tolerance | False |
| Q0-4h − Q0 | -10.1189 | -0.6382 | -9.6509 | -5.8557 [-10.8319, -2.1289] | agree | False |
| Q1-4h − Q1 | -9.5183 | -0.2479 | -3.0184 | -1.8337 [-5.5089, +1.4122] | agree | False |
| Q1-4h − Q0-4h | +0.1707 | +0.3798 | -7.4244 | -4.0999 [-9.0296, -0.3267] | LB change within tolerance | False |

## 11m: standard

| Recipe | January RMSE | July RMSE | Combined RMSE | Official LB | LB − combined |
|---|---:|---:|---:|---:|---:|
| B0-800 | 399.8259 | 516.8258 | 468.6279 | 406.4281 | -62.1998 |
| Q0 | 360.2256 | 406.2250 | 386.5293 | 354.4303 | -32.0990 |
| Q1 | 361.7698 | 387.0822 | 376.0824 | 354.0004 | -22.0820 |
| Q0-4h | 360.0421 | 391.0035 | 377.6050 | 344.3114 | -33.2936 |
| Q1-4h | 361.7212 | 382.5209 | 373.4523 | 344.4821 | -28.9702 |

| Candidate − reference | ΔLB | ΔJanuary | ΔJuly | ΔCombined [95% interval] | Direction | Both months supported gains |
|---|---:|---:|---:|---|---|---|
| Q0 − B0-800 | -51.9978 | -39.6003 | -110.6008 | -82.0987 [-144.7218, -24.9261] | agree | True |
| Q1 − Q0 | -0.4299 | +1.5442 | -19.1428 | -10.4469 [-19.7556, -3.8114] | LB change within tolerance | False |
| Q0-4h − Q0 | -10.1189 | -0.1835 | -15.2215 | -8.9243 [-15.5009, -3.5199] | agree | False |
| Q1-4h − Q1 | -9.5183 | -0.0486 | -4.5613 | -2.6301 [-6.9109, +0.6238] | agree | False |
| Q1-4h − Q0-4h | +0.1707 | +1.6791 | -8.4826 | -4.1527 [-8.0297, -0.9423] | LB change within tolerance | False |

## 11m: seed42

| Recipe | January RMSE | July RMSE | Combined RMSE | Official LB | LB − combined |
|---|---:|---:|---:|---:|---:|
| B0-800 | 398.2541 | 516.3770 | 467.7587 | 406.4281 | -61.3306 |
| Q0 | 359.5562 | 408.1334 | 387.3724 | 354.4303 | -32.9421 |
| Q1 | 361.5801 | 388.3723 | 376.7420 | 354.0004 | -22.7416 |
| Q0-4h | 359.4058 | 392.2306 | 378.0453 | 344.3114 | -33.7339 |
| Q1-4h | 361.5596 | 383.7927 | 374.1094 | 344.4821 | -29.6273 |

| Candidate − reference | ΔLB | ΔJanuary | ΔJuly | ΔCombined [95% interval] | Direction | Both months supported gains |
|---|---:|---:|---:|---|---|---|
| Q0 − B0-800 | -51.9978 | -38.6979 | -108.2436 | -80.3864 [-142.8208, -23.9268] | agree | True |
| Q1 − Q0 | -0.4299 | +2.0239 | -19.7611 | -10.6303 [-20.4092, -3.6184] | LB change within tolerance | False |
| Q0-4h − Q0 | -10.1189 | -0.1503 | -15.9028 | -9.3270 [-16.3493, -3.5891] | agree | False |
| Q1-4h − Q1 | -9.5183 | -0.0205 | -4.5796 | -2.6326 [-6.6534, +0.4742] | agree | False |
| Q1-4h − Q0-4h | +0.1707 | +2.1538 | -8.4379 | -3.9359 [-8.0106, -0.5720] | LB change within tolerance | False |

Full per-month intervals, seed variation, airport/tail/AOBT3 slices, and LB-minus-monthly-CV gaps are in `result.json`. Unrounded predictions, models and training-loss curves are saved beside it.
