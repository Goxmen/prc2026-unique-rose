# Why the validation score levels differ

Investigation requested 2026-09-17. No models fitted, uploads made, validation
rows deleted or folds changed. References are the five official scores recorded
2026-09-16, original v3 summaries, completed seasonal predictions, and observable
2025/2026 movement fields. All timestamps/month boundaries use UTC.

**There is a concrete mismatch in the populations being evaluated.** Original
v3 emphasizes relatively easy late-2025 populations; seasonal evaluation includes
harder July and a January dominated by two exceptionally costly outcomes. This
explains the difference between the local validation levels. Hidden 2026 labels
prevent a complete attribution of either local-to-LB gap.

## 1. The current combined score gives July very little weight

Original v3's next-month readout evaluates July–December. Its long-gap readout
evaluates November/December. Applying the January/July ranking weights to those
two readouts gives **69.17% of combined MSE weight to November/December, 7.82%
to July, and none to January**. Actual ranking has 44.29% January and 55.71%
July rows. Matching forecast distance did not match the seasonal/error population.
The protocol already labels the combined number a provisional diagnostic rather
than expected leaderboard RMSE. See [the fold-weight audit](FOLD_ALIGNMENT.md).

For Q0-4h, the most successful submission in the five-candidate batch:

| Evaluation population | Overall RMSE | Ordinary-row RMSE (target ≤7,200 s) | Tail share of squared error |
|---|---:|---:|---:|
| Original July A1 | 412.67 | 353.86 | 26.52% |
| Original August A2 | 280.90 | 265.42 | 10.75% |
| Original September A3 | 277.72 | 254.73 | 15.89% |
| Original October A4 | 252.71 | 243.76 | 6.98% |
| Original November A5 | 266.24 | 242.85 | 16.81% |
| Original December A6 | 241.31 | 234.97 | 5.21% |
| Seasonal eleven-month January | 360.04 | 233.85 | 57.82% |
| Seasonal eleven-month July | 391.00 | 329.68 | 28.95% |

July's difficulty is broader than the Rome rule. Original July non-Rome RMSE
is 278.30 seconds versus 218.15–233.18 in October–December. The table also shows
that July remains harder on ordinary rows. These are population/model comparisons,
not an isolated causal estimate of seasonality: training windows also differ.

On **exactly the same July flights**, original → ten-month → eleven-month RMSE
is **412.67 → 392.92 → 391.00**. The seasonal model/history changes therefore
improve July, while the overall seasonal score rises from original combined
274.37 to 379.01/377.61. Worse fitting on the shared month does not explain that
increase. The dominant change is which losses enter the summary and their weights.

An exact MSE accounting decomposition replaces the January proxy R_A and July
proxy R_B_long with the two seasonal scores. Across all five recipes, replacing
the July component accounts for **69.7–87.2%** of the increase to ten-month CV;
for Q0-4h it is 72.6%. That component combines population, training and context
changes; it must not be described as a pure causal percentage for seasonality.

## 2. Risky observable populations also differ

Departure `AOBT_3_flt` is observable in both years. Missing values are associated
with exceptionally high local error; the original combined weighting gives this
population less representation than the real ranking set:

| Population | Effective fraction missing AOBT3 |
|---|---:|
| Original combined validation | 1.070% |
| Seasonal January/July validation | 1.564% |
| Actual 2026 ranking departures | 1.534% |

The ranking fraction is **43.3% higher** than original validation's weighted
fraction. This is direct evidence of an observable population mismatch, though
it does not establish equal conditional errors in 2025 and 2026 or explain a
specific number of LB seconds.

The distribution also shifts within seasons: January missing-AOBT3 frequency
rises from 0.934% in 2025 to 1.606% in 2026, while July falls from 2.065% to
1.477%. Matching calendar months alone does not eliminate year-to-year shifts.

## 3. Seasonal pessimism is extremely sensitive to two January outcomes

The following two LFPG departures have missing AOBT3 and are outside the Rome
rule. Their errors were identified after inspecting the eleven-month Q0-4h run:

| Movement | UTC date | Recorded target | Mean prediction across seeds |
|---|---|---:|---:|
| 182378789 | 2025-01-19 | 84,240 s | 880.82 s |
| 182377662 | 2025-01-23 | 58,206 s | 1,166.57 s |

Scores use mean squared errors across seeds, not the mean predictions in this
explanatory table. These two flights produce **51.20% of January squared error**
and **20.62% of combined seasonal MSE** in the eleven-month experiment. January's
ordinary-row RMSE is close to December's; its high overall RMSE is largely a
rare-tail issue. There are 29 January targets above 7,200 seconds in total.

| Q0-4h evaluation | Actual combined CV | Influence diagnostic omitting only these two rows |
|---|---:|---:|
| Ten-month | 379.01 | 337.94 |
| Eleven-month | 377.61 | 336.44 |
| Recorded official LB | 344.31 | — |

This is an **influence diagnostic, not a proposed exclusion**. Both flights
remain in all actual scores. It demonstrates sensitivity; it does not establish
that the hidden test lacks similar cases or that either recorded label is wrong.

With 2,000 calendar-day bootstrap draws, stratified by January/July, seed 0:

| Q0-4h setup | January 95% interval | July 95% interval | Combined 95% interval |
|---|---|---|---|
| Ten-month | [244.45, 486.60] | [328.65, 475.21] | [313.48, 453.25] |
| Eleven-month | [243.22, 486.39] | [326.86, 473.55] | [311.20, 451.72] |

The official 344.31 lies within both combined intervals. These intervals
condition on the fitted models and observed 2025 days; they are **not** a
future-year prediction interval or a test that the protocol is unbiased.

Simply correcting one observable frequency is insufficient. If seasonal
present/missing-AOBT3 conditional losses are assumed unchanged and only their
weights are replaced by 2026 proportions, combined Q0-4h estimates rise to
402.17/401.03 (10m/11m), farther from LB. The two January cases make that
conditional-loss assumption fragile. This sensitivity is not a new score or
recommended calibration, and its strata were not tuned to LB.

## 4. Other mismatches exist, but are not established explanations of the gap

- Original/seasonal/official metrics all use RMSE and score every supplied
  validation/ranking departure. Training filters and the original 27 inputs are
  held fixed. The code audit found no obvious hidden-target or block-time leak.
- Standard versus seed-42 v3 changes the average absolute LB gap by only 0.106 s.
  Nearest-integer submission rounding can change RMSE by at most 0.5 s. Neither
  explains a roughly 71-second gap.
- Original models train on six months; final models train on twelve. The net
  seasonal training/history change improves matched July, so attributing optimism
  to too little training is unsupported. The exact training-size effect remains
  confounded with training composition and history cutoffs.
- Seasonal January lacks December 2024 arrival context, whereas ranking January
  has December 2025. Seasonal July has June context, whereas ranking July lacks
  June. Seasonal first-week arrival coverage is 86.74%/100% for January/July.
  These asymmetries push in different directions. The two dominant January
  errors occur on days 19/23, outside the first-week startup gap.
- Historical feature-aging probes found 2.37–5.25-second penalties under older
  recipes. That is not evidence of a current 70-second effect, nor a guarantee
  that present feature-aging costs are equally small. See the existing protocol.

## What is workable now

1. Keep the chronological folds for comparing model changes, and keep the
   January/July results as separate population stress tests. Their absolute
   scores need not be forced into a single supposedly calibrated LB forecast.
2. Report ordinary/tail and raw AOBT3-availability losses, plus day/row influence,
   beside the unchanged all-row score. Distinguish broad gains from gains tied
   to a few events; bootstrap uncertainty cannot cover unknown recording regimes.
3. A targeted missing-context ablation can reuse saved seasonal models to
   quantify July's pre-month-context mismatch without fitting. It would address
   a known implementation-of-validation difference, not the whole year shift.
4. Any population reweighting or LB-level calibration needs confirmation on
   later, more varied candidates. A blanket +70-second offset or deleting hard
   rows/folds is not justified by this analysis.
5. A fully forecast-matched January/July rehearsal with a preceding full year
   needs more than the one labelled year supplied. Existing 2025 data cannot
   simultaneously match seasons, forward chronology and twelve-month training.

No protocol/model change is selected by this investigation. It explains the
local mechanisms and their limits; it cannot reveal the hidden 2026 error split.

Reproduce: `.venv/bin/python VALIDATION/tools/analyze_validation_optimism.py`.
Detailed numbers, verified prediction/source hashes and the MSE decomposition
are in `validation_optimism.json`; `validation_optimism_observable_counts.csv`
contains the aggregate, label-free population audit. No raw ranking departure
targets are used in that audit.
