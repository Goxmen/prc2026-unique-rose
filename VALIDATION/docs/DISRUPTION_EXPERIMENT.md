# Snow duration and completed-arrival congestion

Frozen before fitting or examining candidate scores, 21 September 2026.

Test one package of 12 inputs in v38's ordinary regressor. All ten airports;
no airport-specific gate. Keep its classifier, corrections, rules, training-row
filters, 800 rounds, seeds 42/142/242 and all 16 folds unchanged. Reuse the
verified local v38 parent models. This is a comparison against v38, not v40:
v40's model and OPDI files are unavailable locally.

- Snow: fraction of half-hour checks with SN/SG during the last 6/24 hours,
  plus coverage fractions. Use the latest METAR at each check, at most 90
  minutes old. Require 10/12 or 40/48 known checks. No hail-only codes.
- Arrivals: counts, mean taxi-in (capped at 7,200 seconds for this input),
  shares over 30/60 minutes among arrivals completed in the previous 60/180
  minutes. Completion is landing plus a finite, nonnegative arrival label,
  strictly before takeoff. Empty windows have count zero and missing summaries.
- Training uses training-window arrivals. Evaluation uses that window plus
  the evaluation month, never intervening months. No departure labels feed
  the new features. Invalid arrival labels are omitted and counted.

Primary result: adopted, unclipped raw-feature Weighted CV; reuse the exact
saved weights. Average squared errors over seeds and repeated origins before
weighting. Report unweighted scores, each seed, each fold, airport/month and
missing-ID/snow slices. Compare paired differences using 2,000 bootstrap
samples of 1-day and 3-day blocks, shared across overlapping groups.

Supplement: predict January 2025 with the already fitted June–November
models for both arms. January departure labels are excluded from training.
This is a retrospective winter diagnostic, not chronological validation or
part of Weighted CV. It cannot reproduce January 2026's unprecedented snow.

Require a negative Weighted CV difference with both 95% bootstrap intervals
below zero and no seed regression to call the package promising. Otherwise
report the uncertainty or regression. These reused folds do not independently
confirm transfer to the leaderboard. Do not tune, submit or promote from this run.
Hash this document and the code in the run record; save all models/predictions.
