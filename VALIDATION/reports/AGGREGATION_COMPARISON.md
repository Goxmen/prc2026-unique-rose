# Month-balanced aggregation candidate

The user requested a better overall aggregation without dropping folds. The candidate below is defined by coverage, not by fitting weights to the five LB scores. The default harness and historical scores are unchanged.

## Proposed overview: equal months, all available folds

1. Average squared errors across seeds for each fold.
2. Within each validation month, average those errors across available training origins.
3. Give each of July–December one-sixth of the total MSE; take the square root once.

All sixteen folds remain. December has five folds, but they share its 16.67% allocation. This is not an arithmetic average of fold RMSEs or an ensemble of model predictions.

| Month | Current combined weight | Proposed weight |
|---|---:|---:|
| 2025-07 | 7.82% | 16.67% |
| 2025-08 | 7.84% | 16.67% |
| 2025-09 | 7.55% | 16.67% |
| 2025-10 | 7.62% | 16.67% |
| 2025-11 | 34.23% | 16.67% |
| 2025-12 | 34.94% | 16.67% |

## Recorded-submission check

Three-seed loss averages; errors are in seconds. The one-second direction tolerance is unchanged. These five related submissions are calibration observations, not an independent proof that an aggregation is universally better.

| Design | Mean absolute LB gap | Mean change error | Recovery-change error | Material directions |
|---|---:|---:|---:|---:|
| current_combined | 70.73 | 7.19 | 0.47 | 3/3 |
| month_balanced | 49.41 | 7.13 | 3.47 | 3/3 |
| unique_flight_pooled | 47.42 | 7.21 | 3.72 | 3/3 |
| month_balanced_horizon_groups | 50.15 | 7.10 | 3.41 | 3/3 |

| Candidate | Current combined | Proposed month-balanced | Official LB |
|---|---:|---:|---:|
| B0-800 | 348.9770 | 368.4560 | 406.4281 |
| Q0 | 276.7374 | 300.6530 | 354.4303 |
| Q1 | 276.1937 | 296.7793 | 354.0004 |
| Q0-4h | 274.3707 | 297.0242 | 344.3114 |
| Q1-4h | 273.7215 | 293.6947 | 344.4821 |

## Horizon tradeoff and interpretation

Equal months correct the late-year dominance, but the available roster cannot also give every month the same forecast horizons: July/August have only A folds. Within-month equal origins give A weight 100%/100%/50%/33.33%/25%/20% from July to December. Across the whole score, A receives 54.72%, all B 45.28%, and k≥5 B3/B4/B7 only 10.83%. That last weight was 55.71% in the old combined score.

Therefore retain **R_A, R_B_long and matched-month A/B comparisons** alongside this candidate overview. A good overall number must not hide a supported long-gap regression. No January evaluation or long-gap summer evaluation is created by changing weights.

Two fixed sensitivities are shown, without selecting a winner from their LB fit: `unique_flight_pooled` gives each unique flight equal total weight after averaging its available origins, producing about 15–18% per month. `month_balanced_horizon_groups` first gives A and B equal shares inside months that have both; it removes dependence on the number of B origins but adds another convention. Their broad results are similar.

The proposed score is structurally more balanced as a general overview, while its observed change-size accuracy is essentially tied with the old score. Recovery-change calibration worsens, so separate horizon readouts remain necessary. No claim of statistically better LB prediction is made; original prediction files would be needed for new paired uncertainty calculations.

Verification: weights sum to one; each proposed month has weight 1/6; existing combined and unique-flight-pooled results reproduce the previous audit within 1e-9 for all five recipes and all four seed readouts. No new fits or protocol changes.

Reproduce: `.venv/bin/python VALIDATION/tools/compare_aggregations.py`.
Full weights, per-seed scores and controlled changes are in `aggregation_comparison.json`.
