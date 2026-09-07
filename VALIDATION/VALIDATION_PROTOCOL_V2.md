# PRC Taxi-Out — Validation Protocol v2

This is the validation protocol for the competition. Follow it exactly.
Train data: all of 2025. Test: one pooled RMSE over departures in Jan 2026 and Jul 2026.
Jan 2026 is the 1st month after our data ends; Jul 2026 is the 7th.

## Core rules

1. **Never validate with random splits.** All folds below are time-based.
2. **Freeze rule.** Any feature built from past departure taxi-out values (rolling
   means, operator/airport aggregates, etc.) must be computed, for validation rows,
   using only data up to the END of that fold's training window — never the days
   right before the flight. Reason: at test time our taxi-out data stops at
   Dec 31, 2025, so "recent" windows do not exist for Jul 2026. Features from data
   that exists at prediction time (weather, flight plan fields, arrival movements)
   are exempt and may be current.
3. **No fitting outside the fold.** Every fitted transform (imputation, encoders,
   scalers, target encoding) is fit on the fold's training window only.
4. Measured 2026-09-07: stale vs fresh features scored ~620 vs ~625 RMSE
   (negligible). Keep the freeze rule anyway; re-run that stale-vs-fresh check
   whenever a new feature family based on past taxi-outs is added.

## Fold design — two shapes, all 6-month training windows

- **Shape A (proxy for Jan 2026):** train 6 months, validate the following month.
- **Shape B (proxy for Jul 2026):** train 6 months, validate the k-th month after
  the training window, k = 3..6. Real July is k = 7, so the July readout comes
  from the far end of the gap curve (k ≥ 5, plus the G7 anchor below).

Efficiency: one experiment needs only 6 trained models (one per training window);
each model is scored on all its A and B validation months. Feature cutoff for a
validation month = last day of that model's training window.

## Fold table

| ID  | Train window (2025) | Validate  | k (months after train end) | Feature cutoff |
|-----|---------------------|-----------|----------------------------|----------------|
| A1  | Jan–Jun             | Jul 2025  | 1                          | Jun 30         |
| A2  | Feb–Jul             | Aug 2025  | 1                          | Jul 31         |
| A3  | Mar–Aug             | Sep 2025  | 1                          | Aug 31         |
| A4  | Apr–Sep             | Oct 2025  | 1                          | Sep 30         |
| A5  | May–Oct             | Nov 2025  | 1                          | Oct 31         |
| A6  | Jun–Nov             | Dec 2025  | 1                          | Nov 30         |
| B1  | Jan–Jun             | Sep 2025  | 3                          | Jun 30         |
| B2  | Jan–Jun             | Oct 2025  | 4                          | Jun 30         |
| B3  | Jan–Jun             | Nov 2025  | 5                          | Jun 30         |
| B4  | Jan–Jun             | Dec 2025  | 6                          | Jun 30         |
| B5  | Feb–Jul             | Oct 2025  | 3                          | Jul 31         |
| B6  | Feb–Jul             | Nov 2025  | 4                          | Jul 31         |
| B7  | Feb–Jul             | Dec 2025  | 5                          | Jul 31         |
| B8  | Mar–Aug             | Nov 2025  | 3                          | Aug 31         |
| B9  | Mar–Aug             | Dec 2025  | 4                          | Aug 31         |
| B10 | Apr–Sep             | Dec 2025  | 3                          | Sep 30         |
| G7  | Jan–May (5 months)  | Dec 2025  | 7                          | May 31         |
| F1  | Jan–Nov (11 months) | Dec 2025  | 1                          | Nov 30         |
| W1  | Feb–Dec (11 months) | Jan 2025  | trains on future           | —              |

Special folds:
- **G7** matches July's true gap (k=7) but with a shorter window — report it next
  to the B folds, marked, never mixed into averages.
- **F1** is the full-scale final gate: use it rarely, only to verify
  hyperparameters and early-stopping rounds at near-final training size before
  the final retrain on all 12 months.
- **W1** is a winter diagnostic only (the only January we have). It trains on the
  future, so: context/static/weather features only, and it never ranks
  experiments — it only answers "does the model handle winter conditions".

## Reporting — every experiment logs one line

- RMSE for every fold individually.
- Shape A: mean ± sd over A1–A6. This is the **Jan proxy**.
- Shape B: individual values grouped by k, plus the per-k averages and the overall
  average. The **Jul proxy** = pooled RMSE over the k ≥ 5 folds (B3, B4, B7, and
  G7 shown alongside). Do not use the all-k average as the July estimate — the
  k=3,4 folds only draw the curve.
- **Headline score** = pooled, not averaged:
  sqrt( (n_jan * JanProxy² + n_jul * JulProxy²) / (n_jan + n_jul) ),
  with n_jan, n_jul = departure row counts of each month in the ranking file.
- Slices: per-airport RMSE for the Jan proxy and Jul proxy; flag any airport that
  moves opposite to the headline.
- Early stopping is decided per fold on its own validation month, never fixed
  globally; final rounds are re-verified on F1.

## Decision rules

- Ideas are ranked by the headline score; a change must beat it by more than the
  noise floor (estimate once: bootstrap the squared errors inside validation
  months) or it is a tie.
- If Shape A and Shape B disagree on an idea, that is a finding, not noise —
  it means the idea helps near-future prediction but not far-future (or the
  reverse). Record it; consider per-month feature sets or models.
- The official leaderboard is ground truth. One submission per real milestone,
  logged next to the headline score. If local improvements stop moving the
  official score in the same direction, freeze feature work and investigate.