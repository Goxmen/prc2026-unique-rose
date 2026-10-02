# Tuning the main model on Jan/Jul CV (row 47, v44)

The first hyperparameter search on this model. Only the ordinary regressor's LightGBM
settings change; inputs, Rome classifier, corrections and rules stay as in v43.

## Result

| | Jan/Jul CV, all flights | Flight-ID departures | Official LB |
|---|---:|---:|---:|
| v43 (row 46) | 318.0408 | 229.81 | 274.7660 |
| **v44 (row 47)** | **314.1959** | **224.46** | **269.5533** |
| Change | **−3.84 [−5.17, −2.93]** | −5.36 [−5.92, −4.86] | **−5.21** |

Three seeds (42/142/242), averaged in squared error within each fit as in the
[validation report](JAN_JUL_VALIDATION.md). Intervals are 95%, resampling whole days within
each fit. Both months improve: January −3.04 [−5.49, −1.75], July −4.65 [−5.83, −3.31].
The leaderboard moved 1.4× the Jan/Jul change.

| Setting | v43 | v44 |
|---|---:|---:|
| `num_leaves` | 255 | **127** |
| `min_data_in_leaf` | 200 | 200 |
| `lambda_l2` | 0 | **100** |
| `feature_fraction` | 0.9 | **0.7** |
| `bagging_fraction` | 0.8 | 0.8 |
| `path_smooth` | 0 | 0 |
| `cat_smooth` | 10 | **100** |
| `cat_l2` | 10 | 10 |
| `min_data_per_group` | 100 | 100 |
| `max_cat_threshold` | 32 | **16** |
| Learning rate × rounds | 0.05 × 800 | **0.02 × 5,000** |

In plain words: smaller trees, a strong brake on each leaf's value, heavier smoothing for
rare categories, coarser category splits, and smaller steps over more trees.

## How it was chosen

Run with `.venv/bin/python VALIDATION/tuning_seasonal.py search` (535 minutes on the Apple M3).

1. **Stage A.** v43's settings (A00) and 20 random settings from a fixed grid were fitted on both
   Jan/Jul splits at seed 42, learning rate 0.05, 2,000 rounds, scored every 200 rounds. A00
   at 800 rounds reproduced v43's saved Jan/Jul predictions exactly.
2. **Stage B.** The stage-A winner (A03) was refitted at learning rate 0.02 for 5,000 rounds (B00).
3. **Choice.** The lowest flight-ID RMSE over all stage A/B trials and rounds: B00 at 5,000.
   Flight-ID departures leave out most corrupted records, which otherwise dominate RMSE.
4. **Seed check.** v43 (800 rounds) and B00 refitted at seeds 142 and 242.

| Seed | v43 all | v44 all | Change | v43 ID | v44 ID | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 318.10 | 314.19 | −3.90 [−5.34, −2.96] | 229.89 | 224.43 | −5.46 |
| 142 | 318.31 | 314.20 | −4.11 [−5.21, −3.35] | 229.74 | 224.45 | −5.29 |
| 242 | 318.10 | 314.19 | −3.91 [−5.06, −3.20] | 229.66 | 224.49 | −5.18 |

All trials at seed 42; change against A00 at its best round:

| Trial | Leaves | Min per leaf | `lambda_l2` | Inputs/tree | Flights/tree | `path_smooth` | `cat_smooth` | `cat_l2` | `max_cat_threshold` | Best rounds | ID RMSE | Change | All RMSE | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A00 (v43) | 255 | 200 | 0 | 0.9 | 0.8 | 0 | 10 | 10 | 32 | 600 | 229.73 | +0.00 | 317.97 | +0.00 |
| A01 | 127 | 400 | 1 | 0.8 | 0.6 | 0 | 10 | 10 | 16 | 1,400 | 228.60 | -1.12 | 317.13 | -0.84 |
| A02 | 255 | 400 | 1 | 0.5 | 0.8 | 10 | 1 | 50 | 64 | 800 | 229.27 | -0.46 | 317.73 | -0.24 |
| A03 | 127 | 200 | 100 | 0.7 | 0.8 | 0 | 100 | 10 | 16 | 2,000 (last) | 224.95 | -4.78 | 314.45 | -3.52 |
| A04 | 127 | 200 | 0 | 0.6 | 0.8 | 100 | 100 | 50 | 64 | 1,800 | 226.93 | -2.80 | 316.54 | -1.43 |
| A05 | 511 | 1,600 | 1 | 0.7 | 0.9 | 10 | 1 | 10 | 16 | 2,000 (last) | 227.25 | -2.48 | 316.48 | -1.49 |
| A06 | 511 | 1,600 | 100 | 0.9 | 0.7 | 10 | 100 | 50 | 64 | 1,200 | 234.53 | +4.81 | 322.17 | +4.19 |
| A07 | 511 | 200 | 10 | 0.9 | 0.7 | 10 | 1 | 1 | 64 | 200 | 232.55 | +2.82 | 320.25 | +2.28 |
| A08 | 511 | 800 | 100 | 0.8 | 0.7 | 0 | 1 | 1 | 16 | 1,400 | 226.88 | -2.85 | 316.06 | -1.91 |
| A09 | 127 | 1,600 | 100 | 0.5 | 0.6 | 100 | 100 | 10 | 16 | 2,000 (last) | 228.78 | -0.95 | 317.63 | -0.35 |
| A10 | 127 | 400 | 100 | 0.7 | 0.6 | 100 | 50 | 10 | 16 | 2,000 (last) | 226.72 | -3.01 | 315.62 | -2.35 |
| A11 | 255 | 800 | 0 | 0.7 | 0.9 | 10 | 50 | 1 | 32 | 1,400 | 227.87 | -1.85 | 316.67 | -1.30 |
| A12 | 255 | 200 | 10 | 0.8 | 0.9 | 100 | 10 | 10 | 64 | 600 | 229.83 | +0.10 | 318.08 | +0.11 |
| A13 | 1,023 | 800 | 1 | 0.6 | 0.6 | 10 | 50 | 1 | 32 | 1,200 | 229.79 | +0.06 | 318.33 | +0.36 |
| A14 | 127 | 200 | 10 | 0.7 | 0.8 | 100 | 100 | 1 | 64 | 1,800 | 227.58 | -2.15 | 316.88 | -1.09 |
| A15 | 511 | 400 | 0 | 0.7 | 0.8 | 0 | 1 | 10 | 64 | 400 | 232.04 | +2.31 | 320.03 | +2.06 |
| A16 | 1,023 | 100 | 0 | 0.7 | 0.6 | 10 | 10 | 10 | 16 | 400 | 225.60 | -4.13 | 314.97 | -3.00 |
| A17 | 1,023 | 800 | 1 | 0.9 | 0.9 | 10 | 100 | 1 | 64 | 600 | 230.91 | +1.18 | 319.23 | +1.26 |
| A18 | 255 | 50 | 0 | 0.8 | 0.6 | 10 | 50 | 50 | 32 | 800 | 227.33 | -2.39 | 316.80 | -1.17 |
| A19 | 1,023 | 400 | 0 | 0.6 | 0.6 | 0 | 10 | 1 | 16 | 600 | 227.66 | -2.07 | 316.47 | -1.50 |
| A20 | 127 | 1,600 | 1 | 0.6 | 0.6 | 100 | 50 | 10 | 32 | 2,000 (last) | 230.94 | +1.21 | 319.25 | +1.28 |
| B00 (lr 0.02) | 127 | 200 | 100 | 0.7 | 0.8 | 0 | 100 | 10 | 16 | 5,000 (last) | 224.43 | -5.30 | 314.19 | -3.78 |

## Reading

- **v43 is on the loose side.** 13 of 20 random settings beat it. It peaks at 600 rounds
  and then worsens; the loosest losers, A07 and A15, peak at 200 and 400.
- **`max_cat_threshold` 16 is the clearest pattern.** All eight settings with 16 beat v43
  (average −2.67 s on flight-ID departures), against 2 of 4 with 32 and 3 of 8 with 64. A13 and A19 differ mainly in this dial (A19 is looser everywhere
  else) and are 2.1 s apart. The categorical inputs include `stand` (2,276 values),
  `airline` (1,019), `dest` (905) and `operator` (504); a lower limit stops one split from
  hand-picking dozens of rare stands or airlines. This is a pattern across random settings,
  not a one-dial test.
- **The change is not independent evidence.** B00 was picked from 22 tries on the same January
  and July data. The 16-fold confirmation was started, then cancelled on the user's request,
  and its partial output deleted, so the 16-fold columns of row 47 are empty.

## Submission

`.venv/bin/python VALIDATION/tuned_submit.py prepare|upload|result` refits seed 42 on all of
2025. Before the tuned fit it refits v43's classifier and 800-round regressor and requires both
to match v43's saved model files byte for byte. It also requires every rebuilt ranking input,
classifier probability and rule row to match v43's saved ranking frames. Uploaded 2026-10-01
19:52 UTC; official score 269.5533 on all 344,841 flights, rank 43/216.

Records: search trials and predictions in `VALIDATION/runs/v3_20261001_jan_jul/search/`,
summary in [jan_jul_validation/result.json](jan_jul_validation/result.json), submission
manifest in `VALIDATION/runs/v3_20261001_jan_jul/submission_v44/submission_v44.json`.
