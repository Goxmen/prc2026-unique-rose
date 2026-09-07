# PRC Taxi-Out — Validation Protocol v3

**Status: implemented.** Written 2026-09-07; implemented the same day in
[protocol.py](protocol.py) (fold roster), [modeling.py](modeling.py) (availability
rules, fitting, round selection), [reporting.py](reporting.py) (scores, slices,
paired bootstrap), [harness.py](harness.py) (runner), [compare.py](compare.py),
[submit.py](submit.py) and [log_official.py](log_official.py), with regression
tests in [tests/test_v3.py](tests/test_v3.py). [README.md](README.md) is the
operating manual. This revises [v2](VALIDATION_PROTOCOL_V2.md) using the evidence
in [FINDINGS.md](FINDINGS.md). The changes from v2 are highlighted below.

We have all of 2025 for training. The competition scores one pooled RMSE over
departures in January and July 2026. These are the first and seventh months
after the available departure labels end. The final model can train on twelve
months; the main local experiments use six-month windows.

## 1. What changes from v2

| Topic | v2 | v3 |
|---|---|---|
| Interpretation | A is the January proxy; long-gap B is the July proxy | A measures next-month performance; B tests longer gaps. Neither is a calibrated estimate of its named 2026 month. |
| Model selection | Early-stop separately on each outer validation month | Default to 800 fixed rounds. Any round selection uses an inner chronological split; one fitted model serves every horizon for its training window. |
| Shape A score | Mean fold RMSE enters the headline | Pool squared errors across flights. Mean and standard deviation of monthly RMSEs remain descriptive. |
| Repeated B rows | Pool B3, B4 and B7 together | Average losses across models predicting the same flight, then pool unique flights. December does not receive extra weight just because it appears twice. |
| Gap comparisons | Compare absolute scores at different gaps | Also compare A and B on the same validation month and flights. |
| Uncertainty | Estimate a noise floor once | Recompute paired, day-block bootstrap intervals for each candidate versus baseline; retain multiple seeds. |
| Extreme labels | Airport slices only | Also report extreme-label counts, error share, ordinary-row RMSE and LIRF/flight-plan slices. Never filter validation targets. |
| Feature availability | Freeze departure history; allow current context | Also mask unavailable gap months, enforce causal training features, and keep traffic counts independent of hidden departure targets. |
| Special folds | F1 is a final gate; W1 tests winter handling | F1 is a training-size diagnostic, not an untouched test. W1 is a retrospective diagnostic with limited interpretation. |
| Decisions | Rank by the combined headline; A/B disagreement is a finding | Assess A and long-gap B jointly. The combined score is provisional; disagreement needs uncertainty and slice checks. |

**Kept from v2:** chronological evaluation, six-month main training windows,
departure-history cutoffs, the A/B fold roster, separate G7/F1/W1 diagnostics,
and comparison with official milestone submissions.

## 2. What the existing findings establish

- The recorded `keep-tail` experiment scored **545.09 seconds with fresh
  departure-history features and 547.46 with stale features**: a 2.37-second
  penalty. Earlier experiments found penalties of 4.40 and 5.25 seconds.
- That comparison held the trained model fixed. It measures the effect of aging
  those particular features, not the performance of a model trained much earlier.
- Extreme targets accounted for approximately **62–79% of squared error** in
  the baseline rehearsal folds. Raising the training-label ceiling to 200,000
  seconds improved the local pooled score by 71.4 seconds, while ordinary-row
  accuracy deteriorated. Both effects must remain visible.
- One matched local/official result is insufficient to calibrate a leaderboard
  forecast. The existing 454.66 local / 405.70 official pair belongs to the old
  protocol and does not calibrate v3.

> **Changed from v2:** correct the fresh/stale ordering and distinguish feature
> aging from model aging. A small historical feature-aging effect does not make
> the B experiments redundant, or guarantee that new feature families behave alike.

## 3. Data and feature rules

1. **Use chronological folds.** Random train/validation splits are excluded from
   experiment ranking. W1 is an explicitly separate retrospective exception.
2. **Use the stated training window only for fitting.** Model labels, learned
   categories, imputation, scalers and target-derived aggregate histories come
   from that window. A global cache is allowed only if queries enforce both its
   start and end. Do not quietly use pre-window labels for target aggregates.
3. **Training target features must be causal.** For each training row, rolling
   departure-taxi statistics use earlier days within the training window;
   exclude the row's own target and all same-day/future targets. Apply the same
   exclusion principle to any new target encoding.
4. **Freeze validation departure history at the training cutoff.** The existing
   features are `op_taxi_30d`, `op_taxi_30d_n`, `rwy_taxi_30d`,
   `aphour_taxi_30d` and `stand_taxi_90d`. Their 30/90-day windows end at the
   cutoff, even when the validation month is much later. Neither departure
   `TAXITIME_SEC_mvt` nor departure `BLOCK_TIME_UTC_mvt` may enter validation
   feature construction.
5. **Current context must actually be available.** For each A/B evaluation,
   expose movement and arrival data from its training window and its validation
   month. Mask intervening months and all other evaluation months. Scoring a
   September B fold does not make September data available to its December B
   fold. Apply causal lookbacks within the exposed data for arrival taxi-in
   averages. Flight-plan fields and documented external observations may be
   used only with the availability they will have in the real submission.
6. **Keep context event counts independent of departure targets.** Include
   supplied departure movement records in traffic counts regardless of their
   taxi-out label. In particular, never use a validation departure's target to
   decide whether its event is counted. Arrival taxi-in filtering may use its
   available label, with the same rule in validation and submission.
7. **Handle missing history as missing history.** Do not fill gaps using hidden
   months or future observations. Missing rolling means stay missing. Use the
   same boundary handling in validation and submission, and report coverage
   during the first seven days of each validation month.

For example, **B4 exposes January–June and December**, not July–November.
December 1 therefore has no recent seven-day arrival history. This reproduces
the missing June arrival history at the start of the real July 2026 evaluation.
In contrast, A6 has November available immediately before December.

> **Changed from v2:** a cutoff alone is insufficient. Enforce the training
> window's lower bound, prevent training-row target leakage, and reproduce the
> missing context at the beginning of an evaluation month after a gap.

At submission time, the available data is 2025 plus the supplied January/July
ranking rows, with departure targets hidden. Any future feature whose lookback
reaches another supplied ranking month must document and validate that extra
availability explicitly; do not silently change the A/B comparison above.

## 4. Main folds

`k = 1` means the month immediately after training ends. All dates below are in
2025. Each training window contains six complete months, though seasonal traffic
means the number of training rows is not identical across windows.

| ID | Train | Validate | k | Departure-history cutoff | Reused model |
|---|---|---|---:|---|---|
| A1 | Jan–Jun | Jul | 1 | Jun 30 | M1 |
| A2 | Feb–Jul | Aug | 1 | Jul 31 | M2 |
| A3 | Mar–Aug | Sep | 1 | Aug 31 | M3 |
| A4 | Apr–Sep | Oct | 1 | Sep 30 | M4 |
| A5 | May–Oct | Nov | 1 | Oct 31 | M5 |
| A6 | Jun–Nov | Dec | 1 | Nov 30 | M6 |
| B1 | Jan–Jun | Sep | 3 | Jun 30 | M1 |
| B2 | Jan–Jun | Oct | 4 | Jun 30 | M1 |
| B3 | Jan–Jun | Nov | 5 | Jun 30 | M1 |
| B4 | Jan–Jun | Dec | 6 | Jun 30 | M1 |
| B5 | Feb–Jul | Oct | 3 | Jul 31 | M2 |
| B6 | Feb–Jul | Nov | 4 | Jul 31 | M2 |
| B7 | Feb–Jul | Dec | 5 | Jul 31 | M2 |
| B8 | Mar–Aug | Nov | 3 | Aug 31 | M3 |
| B9 | Mar–Aug | Dec | 4 | Aug 31 | M3 |
| B10 | Apr–Sep | Dec | 3 | Sep 30 | M4 |

For each seed, fit M1–M6 once and reuse their fitted transforms and model
parameters across all assigned validation months. Rebuild only the permitted
validation context for each month. A full A/B experiment requires **six fits per
seed**, or **18 fits with three seeds**, before optional tuning/diagnostics.

Also report B-minus-A RMSE on matched validation flights:

| Validation month | Compare these B folds against A |
|---|---|
| September | B1 versus A3 |
| October | B2, B5 versus A4 |
| November | B3, B6, B8 versus A5 |
| December | B4, B7, B9, B10 versus A6 |

These comparisons hold the evaluation month fixed. They measure sensitivity to
training recency and available history; they still change which months supplied
training data. An absolute curve across `k` also changes the mix of validation
months and must not be interpreted as a pure causal effect of model age.

> **Changed from v2:** describe A as **next-month validation** and B as
> **longer-gap validation**. The longest main gaps land in November/December,
> not summer July. Equal training duration does not remove this seasonal limit.

## 5. Training configuration and round selection

- Initial reference configuration: the existing harness's LightGBM parameters,
  **800 rounds**, and seeds **42, 142, 242**. Match seeds between candidates.
- Use the `keep-tail` training-label range **1–200,000 seconds** as the reference,
  based on the existing experiment. Other ceilings remain explicit experiments.
- Keep departure/arrival taxi aggregate inputs in **1–7,200 seconds**, using only
  labels allowed by section 3. This protects historical means without deleting
  extreme labels from the model's training target or the validation score.
- **Score every validation departure with its supplied target**, including
  extreme values. Missing labels are a data error to report, not rows to drop
  silently. Movement IDs are alignment keys, never model inputs.
- Preserve the existing prediction convention: clip negative predictions to
  zero and use unrounded predictions for local metrics. Round only when filling
  the official integer submission template, consistently for all candidates.
- All existing configurable choices remain explicit and recorded. In particular,
  `--fpl-flag` is an experiment, not silently enabled by this protocol.

Do not early-stop on an A or B month that supplies the reported outer score.
If round tuning is needed, fit on the first five months of a training window and
select rounds on its sixth month, with the same causal/availability rules.
Then refit on all six months using that count. **Every A/B horizon for that
window and seed uses the same refitted model and round count.** This adds inner
fits; report their cost and the selected counts separately.

When the round-selection policy changes, rerun the baseline under that policy.
Final twelve-month training uses the recorded policy too: either the fixed count
or an inner January–November fit / December selection followed by a full-year
refit. December used for selection supplies no independent final accuracy claim.

> **Changed from v2:** validation answers cannot select a separate best model for
> each outer horizon. The fixed-round policy preserves model reuse; inner tuning
> is available when training-size-dependent round selection needs testing.

## 6. Scores and weighting

For the default single-model submission strategy, average **squared errors across
seeds**, not predictions. This estimates performance across training randomness;
it does not create an ensemble. Also report each seed's RMSE and its spread.
If an ensemble is proposed, score its actual averaged predictions as a separate
candidate using the intended submission recipe.

For any evaluation group `S`, let `L[f,i]` be flight `i`'s squared error in fold
`f`, averaged across the selected seeds. All folds validating the same month
must score exactly the same departure IDs. Define:

```text
For each validation month m:
  L[S,m,i] = mean of L[f,i] over folds f in S that validate month m

R[S] = sqrt(sum over unique (m,i) of L[S,m,i]
            / number of unique validation flights in S)
```

Use this rule for airport/target slices as well, recalculating their counts.

| Readout | Definition | Use |
|---|---|---|
| `R_A` | `R[{A1,...,A6}]` | Main next-month comparison; ordinary pooling of all July–December validation flights. |
| `R_B(k)` | `R` over B folds at a particular `k` | Describe performance by gap, alongside the individual fold values. |
| `R_B_all` | `R[{B1,...,B10}]` | Secondary summary; not a July estimate. |
| `R_B_long` | `R[{B3,B4,B7}]` | Longer-gap readout at `k >= 5`; November and December only. |

For `R_B_long`, a November flight contributes its B3 loss. A December flight
contributes **the mean of its B4 and B7 losses**. Thus each flight has one total
unit of weight; December's weight reflects its row count, not its two models.
Do not average B4/B7 predictions: that would score an ensemble instead.

Log monthly RMSE mean and standard deviation descriptively. Their standard
deviation describes variation across months, not a confidence interval.

A provisional combined diagnostic may also be logged:

```text
w_jan = n_jan / (n_jan + n_jul)
w_jul = n_jul / (n_jan + n_jul)
R_combined = sqrt(w_jan * R_A^2 + w_jul * R_B_long^2)
```

Read `n_jan` and `n_jul` from ranking departure counts (currently 152,719 and
192,122). Keep these weights fixed across experiments. Label the result
**provisional combined diagnostic**, never expected leaderboard RMSE. The flight
populations differ from January/July, and November/December contribute to both
components. Do not use this number alone to select candidates.

> **Changed from v2:** pool losses consistently, make repeated-flight weights
> explicit, and separate an experiment diagnostic from a calibrated forecast.

## 7. Uncertainty and mandatory slices

For every candidate versus baseline comparison:

1. Align predictions by movement ID, fold and seed. Define each difference as
   **candidate score minus baseline score**; negative favors the candidate.
2. Run **2,000 paired bootstrap replicates**, using a recorded bootstrap seed.
   Within each validation month, sample UTC calendar days with replacement,
   retaining all flights and airports on each sampled day.
3. Use the same day draws for candidate and baseline, every seed, and every fold
   that evaluates those flights. Repeated December predictions must remain
   together. Recompute the section 6 weighting and score differences on each
   replicate; do not treat repeated rows or seeds as independent flights.
4. Report the difference and its 95% percentile interval for `R_A`, `R_B_long`,
   the combined diagnostic, and the matched-month aging comparisons. Report
   paired differences by seed separately as a check on training randomness.
5. These intervals condition on the fitted models and observed months. They do
   not establish robustness to a new year or every possible training sample.
   Check sensitivity to multi-day blocks for close decisions where disruptions
   persist across days; record any dependence on that choice.

Every full experiment also reports, by fold and for A/long-gap B:

- Overall RMSE and scored row count.
- RMSE on targets **at or below 7,200 seconds**.
- Number of targets **above 7,200 seconds**, their RMSE, and their share of total
  squared error under the same weighting. Empty slices report `N/A`.
- Per-airport RMSE and counts.
- LIRF separately, split by `AOBT_3_flt` present/missing. This is the current
  flight-plan-availability indicator; retain the precise definition in logs.
- First-seven-day context coverage, including unavailable arrival history.

> **Changed from v2:** uncertainty belongs to a particular paired comparison,
> not a permanent noise threshold. Extreme-label and coverage slices are part
> of the required evidence, because the recorded results show they can dominate
> an apparently temporal improvement.

## 8. Optional diagnostics

| ID | Train | Validate | Purpose and restriction |
|---|---|---|---|
| G7 | Jan–May 2025 | Dec 2025; cutoff May 31 | Actual seventh-month gap, but only five training months. Keep separate from A/B scores. |
| F1 | Jan–Nov 2025 | Dec 2025; cutoff Nov 30 | Near-final training-size check. Select rounds inside January–November if needed; never call this an untouched test. |
| W1 | Feb–Dec 2025 | Jan 2025 | Retrospective January diagnostic. Exclude departure-target-derived input features on both sides. No contribution to rankings, confidence claims about future January, or calibration. |

December already participates in A/B development. Running F1 infrequently does
not restore independence. A truly untouched December gate would require
excluding December from all earlier tuning and comparisons, which this protocol
does not claim to do. W1 can reveal failures on the available January; its
future-trained, reduced-feature model cannot verify the final deployed model.

Each special diagnostic needs an additional fit per seed. They are not included
in the six-model A/B budget.

Retain a separate **feature-only stale/fresh check** whenever a new family based
on past departure targets is introduced: hold the January–June trained model,
July validation rows, seed and current-context inputs fixed; compare departure
history cutoffs of January 31 and June 30. This reproduces the earlier question
without mixing in a change of trained model. It is diagnostic, not part of the
default A/B scores.

## 9. Decisions, records and official checks

- Use `R_A` for the main next-month comparison and inspect `R_B_long` before
  advancing a milestone. A full candidate comparison evaluates both shapes.
  A focused subset may screen an idea but cannot claim full-protocol performance.
- A paired interval entirely below zero supports improvement on that readout;
  entirely above zero supports regression. An interval crossing zero is
  inconclusive, not evidence of equivalence.
- Prefer candidates with supported improvement and no established regression in
  the other main readout. If A and B disagree, inspect paired intervals, matched
  months, seeds and extreme-label slices before attributing the difference to
  forecast horizon or designing separate models. Record unresolved tradeoffs.
- Keep the baseline, fold roster, score weights and slice definitions fixed
  during comparisons. Record unsuccessful experiments too. Repeated searching
  can overfit these validation months despite individual confidence intervals.
- Train on all twelve months in the final refit using the chosen training policy;
  do not silently switch feature logic, target filtering, rounds or seed/ensemble
  strategy between the validated recipe and submission.
- Log an official submission for each meaningful milestone. Compare the sign of
  local changes with official changes across multiple controlled milestones.
  If they repeatedly disagree, revisit the protocol before further feature work.
  No fixed local-to-official offset is justified by one pair, and v2 scores must
  not be mixed with v3 scores as if their definitions matched.

Write one experiment summary plus detailed fold/slice records. Save predictions
for every fold and seed with movement ID, validation date, airport, target and
the slice indicators needed above. Record the protocol version, baseline ID,
exact code state (including any uncommitted patch), dataset identity, training
windows, context masks, feature list, target filters, parameters, rounds, seeds,
bootstrap settings, scores, intervals and any official submission/score.

Flags remain available for training length, fold subsets, rounds, seeds, label
filters and optional features. A changed configuration must be recorded; a
different training length or roster is a variant rather than the default v3
benchmark. Sampled and smoke runs check execution only and cannot rank models,
because sampling changes traffic features and the rare-label distribution.

## References

- [Validation protocol v2](VALIDATION_PROTOCOL_V2.md): original proposal and fold IDs.
- [Recorded findings](FINDINGS.md): sections 2–5 cover extreme labels, paired
  uncertainty, feature staleness and the existing official comparison.
- [Current feature implementation](features_time.py): starting point for the
  availability rules; this document specifies changes rather than certifying
  that the current implementation already satisfies them.
- Selecting models and reporting their performance on the same validation data
  can bias evaluation; see Cawley and Talbot,
  [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).
