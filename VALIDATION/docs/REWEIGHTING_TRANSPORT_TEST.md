# Reweighting transport test — frozen design

**Status: frozen 20 September 2026, before any result of this test was computed.**
Written so that the criteria cannot be chosen after seeing the answer. If any
part of this design changes once results are known, rerunning the same five
months is no longer independent confirmation and the rerun must say so.

## The question

An experimental reweighting moved the combined CV score from 236.32 to 269.40,
against an official 279.38. Nothing establishes that such an estimate transports:
it has never been checked where the answer is known. This test checks it on 2025
months whose labels we hold.

It is a separate investigation. The training-versus-ranking comparison does not
depend on its outcome.

## Design

**One fixed model.** All predictions come from `v3_2025-06`, trained on
January–June 2025. Leaving a month's errors out of a pooled score is not enough,
because later A-fold models trained on that month's labels. This model never
trained on any month it is scored against.

**Five target months**, each from that model's own folds:

| Fold | Validation month | Gap after cutoff |
|---|---|---|
| A1 | 2025-07 | 1 |
| B1 | 2025-09 | 3 |
| B2 | 2025-10 | 4 |
| B3 | 2025-11 | 5 |
| B4 | 2025-12 | 6 |

**Leave-one-month-out.** Each month in turn is the pretend ranking set. Its labels
are not read while estimating. The other four months are the source; the method
estimates the target month's RMSE from the source alone, and the estimate is then
compared with the month's actual RMSE.

**Estimator under test.** Cross-fitted density ratio, exactly as
`analyze_adversarial_shift.py` computes it: same feature set, same LightGBM
settings (300 rounds, 63 leaves, learning rate 0.05, seed 42), source and target
each split in half, each model trained on the complementary halves, and the
source/target class-count correction applied per half. Weighted RMSE over the
source rows is the estimate. No clipping in the primary readout; a p99-clipped
variant is reported beside it and is not the primary.

**Two baselines**, both computed on the same source rows:

1. **No weighting** — the plain pooled source RMSE.
2. **No-flight-ID only** — source rows reweighted so that the share of departures
   with no flight ID matches the target month's, and nothing else changes.

**Two things are tested, not one.**

- **Level.** Signed and absolute error of the estimated RMSE against the month's
  true RMSE, for every month separately. Reported for both `pred_combined` (the
  current recipe, the headline) and the control `pred_noid_cap`.
- **Difference.** For each saved candidate column against the control
  `pred_noid_cap`, the estimated candidate-minus-control change on the source
  against the true change on the target month. Predicting a score is not the same
  as ranking two models, and a method can do one without the other. All candidate
  columns come from the same run and the same training window, so nothing is
  refitted.

## Criteria, fixed in advance

The method **transports for level** if, across the five months, its mean absolute
error is smaller than both baselines' **and** its worst single month is smaller
than both baselines' worst single month.

The method **transports for differences** if it gets the sign of the true change
right on at least as many candidate-month pairs as the unweighted baseline, and
its mean absolute error on the changes is smaller.

Anything else is a failure to demonstrate transport. A mixed result is reported as
mixed, not as partial success.

## Reporting rules

- Every target month is reported. Averages never appear without the five months
  beside them.
- The real leaderboard score is not an input to this test and nothing is tuned
  against it.
- Intervals are descriptive and conditional on the fitted weights.

## Known limits of this test, stated in advance

- Five months, all from the second half of 2025. **January is not tested at all**,
  and January is 44.3% of the ranking set. A method that transports here is not
  thereby shown to transport to January 2026.
- One training window and one model. Transport may depend on both.
- The target months sit at different gaps after the cutoff (1 to 6 months).
  Estimating July from September–December transports backwards in gap, which the
  real application never does.
- The feature set is the raw movement and flight-plan set. It omits weather,
  traffic and history, so a failure here does not establish that reweighting
  cannot work with better features.
- Source and target here are months of the same year, which is a smaller shift
  than 2025 against 2026. Success is therefore a weak positive; failure is a
  strong negative.

## Outputs

`VALIDATION/tools/test_reweighting_transport.py` writes
`VALIDATION/reports/reweighting_transport.json` and records the SHA-256 of this
document as frozen at run time. Saved predictions and local diagnostics only: no
taxi-out refits, no submissions, no fold removals, no model promotions.
