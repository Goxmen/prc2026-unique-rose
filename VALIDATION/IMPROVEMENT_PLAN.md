# Recovery and timestamp modelling: experiment plan

Written 2026-09-16. This is the working order for the next experiments.
Update the status and results here as each stage finishes.

## Resume here — Stage 2 complete; Stage 3 next

Work is on branch `stage2-rome-six-hour-rule`, based on Stage 1's open PR branch.
**Stage 2 is complete. Select Q0 (B0-800 + the fixed rule) as the conservative
parent for Stage 3; retain Q1 (R3 + rule) as an alternative.** Both rule
candidates improve against their parents. Q1 improves next-month RMSE over Q0
but has a supported longer-gap regression; its combined difference is uncertain.
Read [Stage 2](reports/STAGE2.md), [Stage 1](reports/STAGE1.md), and
[Stage 0](reports/STAGE0.md) for the round-count experiment.

| Experiment | Artifact / state |
|---|---|
| Original B0, 1,500 rounds | `runs/v3_20260915T225301657023Z/result.json`, frozen; its model/prediction files were absent on this laptop |
| B0-800, matched control | `runs/v3_20260916T095948645262Z/result.json`, complete; R_A 366.7465, R_B_long 334.1785 |
| R1, exact twins | `runs/v3_20260916T102029635587Z/result.json`, complete; **do not promote**, longer-gap regression |
| R2, plus relaxed twins | `runs/v3_20260916T104452470012Z/result.json`, complete; **do not promote**, longer-gap regression |
| R3, Stage 1 recovery candidate | `runs/v3_20260916T113256202842Z/result.json`, complete; R_A 360.5041, R_B_long 333.3043 |
| Q0, selected conservative parent | `runs/v3_20260916T123230629423Z/result.json`, complete; R_A 300.5904, R_B_long 256.1971 |
| Q1, retained tradeoff | `runs/v3_20260916T123433565511Z/result.json`, complete; R_A 296.9037, R_B_long 258.5505 |
| Verification | 45 tests; Stage 1 equivalence/audits retained; rule comparisons, changed-row CSVs and chart complete |

**Next task: Stage 3, T1 then T2, with Q0 explicitly named as parent.** Keep
recovery off and the fixed Rome rule when testing clock features and residual
prediction, so those changes are separable from recovery. No Stage 3 models
were run during Stage 2. Do not rerun Stage 1 audits, CPU tuning or notebook
caches unless relevant code/data changes. Full-year refits and uploads remain
deferred. Q0 retains B0-800's frozen model-source requirement for any later
full-year refit; the rule does not bypass that guard.

**Round count:** Stage 1 explicitly uses **800 rounds** for control and all
three candidates, following the Stage 0 checkpoint verdict. The historical
800-round predictions were unavailable here, so B0-800 was regenerated with a
new identity. Data hashes match Stage 0; its scores differ from the historical
checkpoint by less than 0.03 s. Do not relabel the original 1,500-round B0.

**Current machine:** i7-13650HX laptop under WSL, 20 exposed logical CPUs and
about 11.5 GiB RAM. Use **one fit worker, six training threads, 20 prediction
threads**, based on [local measurements](reports/STAGE1_CPU.md). Do not transfer
the old 7800X3D desktop setting automatically. Overlapping a full-data audit
with fitting caused swapping, so full-data audits and experiments also run
sequentially. Thread comparisons preserved predictions to numerical precision;
explicit prediction thread counts gave bit-for-bit equality.

The user authorizes local six-month training and validation. Full-year refits
and uploads remain deferred until configurations are selected. No automatic
submission schedule or quota retry is authorized by this plan.

Existing notebook caches were preserved and checked against their raw donors.
[The section/cell mapping and reuse limits](reports/STAGE1_DERIVED_DATA.md)
explain why their pooled learned history and omitted guards prevent direct use
in chronological validation. The pipeline reuses the notebook rules through a
shared fold-scoped interface; it does not rebuild the notebook caches.

Models, predictions, source snapshots and row-level audits remain gitignored.
New local control/candidate artifacts exist on this laptop; another clone must
copy or regenerate them. Track concise results, reports, charts and source.

## Objective and current decision

Improve prediction of the recorded taxi-out labels by recovering missing
flight-plan information and modelling the different timestamp patterns.
**Start with flight-plan recovery. Use local validation first; prepare a small
set of controlled submissions for tomorrow's submission session.**

The three modelling directions, in working order, are:

1. Recover missing flight-plan fields from corresponding arrivals and history.
2. Use the clocks more directly: explicit disagreement features and prediction
   of corrections to a timestamp estimate.
3. Model ordinary, schedule-substituted, and day-offset cases separately, then
   combine their predictions using estimated probabilities.

The simple **Rome six-hour rule** is a separate, early experiment between
recovery and the more involved timestamp models. It provides a useful control
for whether a large gain needs a new model at all.

### Fixed experimental settings

- Training labels: **1–200,000 seconds**, the existing keep-tail reference.
  Do not repair target values or add a new ceiling sweep in this series.
- Score **every validation departure**, including zero, negative, and extreme
  targets. The training filter does not apply to validation.
- Main benchmark: protocol v3 **A1–A6 and B1–B10**, six-month training windows,
  **800 fixed rounds for Stage 1**, seeds **42, 142, 242**, all rows in each
  six-month training window, no early stopping. Stage 0 tested the requested
  1,500 rounds; its held-out regression motivated this explicit return to 800
  with a separately registered matched control.
- Keep the current LightGBM parameters and the v3 feature set as the reference;
  `fpl_flag=False`. Any added model inputs must be explicit experimental changes.
- Full twelve-month refits are deferred until the experiments finish and
  configurations are selected for submission. Those later fits use seed **42**
  and the validated round count. A seed ensemble would be a separate experiment,
  evaluated as an ensemble locally first.
- Keep historical taxi aggregates on their existing **1–7,200-second** inputs.
- Preserve progress bars and per-round training-loss records for fits.
- No uploads during the local experiment session. No automatic quota-reset retry.

### CPU efficiency

**Current laptop:** use one fit worker, six training threads and 20 inference
threads; run full-data audits separately. See [the Stage 1 measurements](reports/STAGE1_CPU.md).

**Historical Stage 0 desktop:** the Ryzen 7 7800X3D exposed eight physical cores
and sixteen logical CPUs to WSL. Its selected sustained-run setting was
**`--fit-workers 1 --threads 8`**: sequential fits with eight threads each.
The first six fits also tested two simultaneous four-thread fits, with the
final unpaired seed using eight; their actual thread counts remain recorded.

- [x] Before long experiment runs, time eight versus sixteen threads on the
  same representative six-month feature matrix, seed, and round count. Include
  warm-up and repeat timings; record wall time and peak memory.
- [x] If throughput still warrants it, compare two independent four-thread fits
  with sequential eight-thread fits. Account for duplicated memory and shared
  CPU/cache resources; do not assume more workers improve total throughput.
- [x] Record the chosen execution settings and check prediction equivalence
  within numerical tolerance. Preserve reuse of each training matrix across
  seeds and each fitted model across assigned validation folds.
- Higher CPU utilization is not the selection criterion; use elapsed time per
  completed experiment. No full-year model is needed for these timings.

Measured on the full Jan–Jun training matrix, 300 rounds with 20 warm-up rounds:
median single-fit time **13.15 s at eight threads**, **22.86 s at sixteen**.
Two independent four-thread processes completed their fits in a median 19.73 s
versus an estimated 26.30 s sequentially. The shared-matrix implementation then
completed a two-seed batch in **20.38 s**, with **1,394 MiB** reported peak RSS.
Probe predictions differed from the eight-thread reference by at most
`9.1e-13` seconds. Full-experiment speedup is not inferred from fit timing alone.
See [the CPU benchmark](reports/benchmarks/stage0_threads_20260915T224428Z.json).

**Sustained-run correction:** on M2, the paired four-thread fits took 272.0 and
272.4 seconds each while the single eight-thread fit took 58.0 seconds. That
reversed the short-benchmark throughput advantage. The run was resumed with
sequential eight-thread fits, preserving the six completed models and seven
completed folds. The exact cause of the parallel slowdown is not established;
there was no swap use or cgroup CPU throttling. Prefer the sustained-run choice
for the recovery experiments.

## Starting evidence and its limits

- Existing official reference: **v3, RMSE 405.6987**, ceiling 200,000 seconds.
  v4 added a flight-plan flag and scored 405.7838. These are recorded results,
  not a new live leaderboard snapshot.
- Existing **800-round** v3 baseline run: `v3_20260907T202123182899Z`.
  Its main scores are `R_A=366.7673`, `R_B_long=334.2027`.
- Exploratory application of the six-hour rule to that run's stored predictions
  gave `R_A=300.6034` and `R_B_long=256.1843`, changing 44 and 8 unique flights,
  respectively. All three seeds and the protocol's repeated-flight weighting
  were used. These are reused development folds; paired uncertainty and a
  registered candidate record are still needed. They are not official scores.
- [Notebook §2b–§2c](../eda/feature_sweep.ipynb) documents recovery, but its
  whole-dataset derived files are not directly suitable for chronological
  validation. Its earlier valuation estimated a small score gain from recovery;
  completeness alone is not evidence of a leaderboard improvement.
- [Notebook map](../eda/NOTEBOOK_MAP.md), §17: reject recovered plans whose
  `ADEP_flt` disagrees with the departure's `ADEP_mvt`. This rejects three donors
  in the notebook's original full-data recovery. Treat those historical counts
  as audit references, not fold-specific coverage targets.

## Stage 0 — establish a comparable baseline

**Complete.** See [Stage 0 results](reports/STAGE0.md). Frozen B0 run:
`v3_20260915T225301657023Z`; `R_A=371.0708`, `R_B_long=335.2416`.
The same models' 800-round checkpoints scored better: the next-month difference
is a supported regression (+4.3035 s); the longer-gap difference is inconclusive.
Keep round-count changes explicit in subsequent comparisons.

- [x] Freeze source/data identities and register baseline `B0` with the settings
  above. Use the same code version and feature switches for baseline/candidate.
- [x] Prove the recovery-off path preserves feature values, row IDs, and ordering.
  Added columns for joining must not enter the baseline model accidentally.
- [x] Run a new **1,500-round B0** after the shared loader changes. The old
  800-round results remain historical references and cannot serve as the
  matched baseline for the new recovery experiments. Decreasing training loss
  motivates this round-count experiment; validation determines whether it helps.
- [x] Record `R_A`, `R_B_long`, per-fold/per-seed metrics, and the mandatory slices.

Command to reproduce the historical Stage 0 1,500-round baseline:

```bash
.venv/bin/python VALIDATION/harness.py \
  --label recovery-baseline --folds A,B --seeds 3 \
  --rounds 1500 --max-taxi-sec 200000 --threads 8 --fit-workers 1 \
  --recovery off --checkpoints 800
```

Recovery modes are implemented as `--recovery exact` (R1), `relaxed` (R2), and
`history` (R3); each later mode includes its parent layers. The rule switches
remain planned. Full experiment commands on the current laptop:

```bash
.venv/bin/python VALIDATION/harness.py --label R1-exact --folds A,B --seeds 3 \
  --rounds 800 --max-taxi-sec 200000 --threads 6 --prediction-threads 20 --fit-workers 1 \
  --recovery exact --baseline v3_20260916T095948645262Z
# After recording R1, run R2 with --recovery relaxed --baseline <R1-run>.
# After recording R2, run R3 with --recovery history --baseline <R2-run>.
# Also compare each completed candidate against the B0-800 control.
```

## Stage 1 — flight-plan recovery first

### Implementation boundary

Extract reusable recovery logic into a pipeline module and call it from both
validation and final submission. Extend the loader with the required movement
identifiers, flight number, route, aircraft type, and donor flight-plan fields.
Retain a full arrival table for matching separately from the narrow arrival
table used by historical taxi features.

Use the precedence **raw value → exact twin → relaxed twin → static history**.
Existing nonmissing raw values take precedence; original targets, movement IDs,
and movement timestamps remain intact. Return an audit table alongside the
model-facing fields: donor movement/flight ID, stage, confidence, and flags.

Preserve these distinct facts before filling anything:

- `raw_flight_id_missing`: original `FLIGHT_ID_mvt IS NULL`;
- `raw_plan_block_missing`: original `ADEP_flt IS NULL`;
- original `AOBT_3_flt` availability;
- availability after recovery, recovery stage, and static-imputation flags.

An ID can exist above an empty flight-plan block. Audit that case separately.
These provenance fields initially support diagnostics and the rule; if added as
model features, give the recovery-off control the identical flags too.

### Recovery variants

| ID | Change relative to parent | Parent |
|---|---|---|
| R1 | Unique exact arrival twin supplies missing flight-plan fields | B0 |
| R2 | Add the relaxed twin stages A/B/C from notebook §2c | R1 |
| R3 | Add reliable static-field history imputation | R2 |

Each variant gets its own configuration, validation result, and source identity.
Validate exact matching before progressing to relaxed matching; validate the
twins before progressing to history imputation. Compare the selected recovery
variant with B0 as well as with its immediate parent.

**R1 / R2 matching rules:**

- Use matching route, flight identifiers, aircraft compatibility, and the
  notebook's stage-specific landing-time windows. Require a unique candidate.
- Require donor `ADEP_flt = departure ADEP_mvt`; verify route consistency.
- Reject collisions where two departures claim the same donor or where the
  donor flight plan already belongs to another departure. Resolve stage
  precedence deterministically; ambiguous matches remain unfilled.
- Recovery eligibility and candidate selection must not read a departure's
  target or block timestamp, including whether its target looks plausible.
- R2's learned airline-prefix maps and route-duration tables are fitted inside
  each training window. Stage C remains separately identifiable in the audit.

**R3 static history:**

- Impute only stable categoricals such as operator, wake class, aircraft type,
  segment and flight type, using the notebook's per-field support/confidence
  rules. Do not invent flight IDs or off-block timestamps from history.
- Exclude the documented low-confidence flight-type assignments. Record
  confidence and support for every imputed field.
- Use earlier-day history for training rows; fit/freeze validation lookup tables
  using only the corresponding training window. Evaluate delayed/missing-plan
  slices separately from ordinary matched flights.

### Availability rules for every recovery variant

- **Training:** donor movements and fitted lookup tables stay within the
  training window. Do not repair a training row using an arrival from its
  validation month, even at a month boundary.
- **Validation:** donor pool is the training window plus that validation month.
  Exclude intervening months and other validation months. Hide departure targets
  and block timestamps before invoking recovery.
- A same-leg arrival later than takeoff may supply its visible flight-plan
  fields: this is a retrospective batch task and those arrivals are supplied.
  Keep that operation distinct from causal target-history features.
- **Final refit/prediction:** apply the same construction using 2025 for fitting
  and each ranking month separately for current context. Do not silently use
  January ranking data to supplement July's recovery/history.
- Rebuild fold-scoped recoveries. The notebook's all-year/all-ranking caches
  may be used for auditing but cannot bypass these boundaries.

### Recovery acceptance checks

- [x] On held-out known matches, hide the departure's plan, recover it, and
  compare with its true plan ID; exclude probes from learned reference tables.
- [x] Hide the true arrival twin and measure false recovery; test ambiguous
  routes, duplicate claims, incompatible origins, and month/gap boundaries.
- [x] Confirm hidden departure labels cannot affect recovery output and that
  evaluation months cannot alter the fitted training representation.
- [x] Report coverage and match errors by stage, airport, and raw missingness.
  Count actual recovered anchors separately from filled static categories.
- [x] Run the full paired benchmark for completed recovery variants. Record
  results on recovered rows, unrecovered rows, and the original Rome/no-plan
  slice. Increased coverage is not the promotion criterion.

All three variants are implemented and benchmarked. R1 and R2 are not promoted:
both regress against B0-800 on R_B_long. R3 is provisionally selected after
comparison with both R2 and B0-800; see the decision below.

## Stage 2 — register and test the Rome six-hour rule

Keep the original diagnostic definition fixed:

```text
gap = seconds(MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt)
if ADEP_mvt == "LIRF" and raw_flight_id_missing and gap >= 21600:
    prediction = gap
else:
    prediction = model_prediction
```

This changes predictions, not training labels. The trigger uses visible inputs
only. Use the **raw** missing-flight-ID flag even after recovery so the
definition and eligible rows do not change silently.

| ID | Candidate | Comparison |
|---|---|---|
| Q0 | B0 + six-hour rule | Q0 − B0 |
| Q1 | Selected recovery variant + six-hour rule | Q1 − recovery; Q1 − Q0 |

- [x] Reuse each parent's saved predictions; no extra fits are needed for the
  rule itself. Produce a separate candidate record and paired day-bootstrap CIs.
- [x] List changed IDs, original/rule predictions, per-row error changes, and
  counts by month. Identify damaging overrides as well as successful ones.
- [x] Keep six hours fixed for this test. A different trigger, including using
  raw plan-block missingness instead of raw ID missingness, is a new experiment.
- [x] Carry the exact same override into the linked submission recipe and hashes.

Complete: [Stage 2 report](reports/STAGE2.md). The shared rule implementation
is used by saved-prediction validation and linked submission, with separate
model/rule hashes. No new full benchmark fit or full-year refit was needed.

The rule is an empirical candidate: the schedule does not reproduce every
extreme label, and some day-offset cases follow a different pattern.

## Stage 3 — clock features, then residual prediction

Start each comparison from an explicitly named parent selected above.

- **T1: explicit clock disagreement.** Add `AOBT_3 - schedule`,
  `AOBT_3 - EOBT_1`, agreement/spread between available taxi estimates, and
  provenance-aware availability flags. The takeoff-minus-clock features already
  exist; do not present them as new inputs.
- **T2: residual prediction.** Define and freeze a timestamp baseline using
  available flight-plan clocks with a documented missing-clock fallback. Fit a
  correction to that baseline and add it back for prediction. Derive any clock
  calibration/reliability parameters inside the training window.
- [ ] Compare T1 with its parent, and T2 with the same features under direct
  target regression, so feature gains and residual formulation are separable.
- [ ] Score reconstructed taxi times, not residuals; include all extreme labels
  and retain the same final nonnegative-prediction convention.

## Stage 4 — model the different timestamp patterns

Test a probabilistic combination of ordinary taxi-out, schedule-substituted
timestamps, and roughly one-day-offset cases. Use training labels/block stamps
only to define training outcomes for these components. Predict their
probabilities from fields available in ranking; no target-dependent gates at
evaluation or submission time.

Compare this candidate with both its direct parent and the simpler Q0/Q1 rule.
Keep all recorded labels in the scoring population. Inspect false high
predictions on ordinary Rome flights, not only recovery of extreme targets.
Proceed after the recovery and rule results are recorded; avoid combining
untested components into one unexplained change.

## How we decide locally

Follow [protocol v3](docs/PROTOCOL.md):

- Report `R_A` and `R_B_long` together. `R_combined` is a provisional summary,
  not a leaderboard forecast. Average squared errors across seeds and repeated
  predictions of the same flight using the existing protocol weights.
- Use 2,000 paired UTC-day bootstrap replicates. Negative candidate-minus-parent
  differences indicate improvement. Report confidence intervals and seed spread.
- Prefer supported improvement without supported regression in the other main
  readout. If the readouts disagree, inspect months, airports, and the tail;
  record the tradeoff before advancing that candidate.
- Keep ordinary-row RMSE, >7,200-second error share, raw Rome/plan-availability
  slices, and recovery coverage visible. Add ≤12-second diagnostics without
  changing the overall scoring population.
- Smoke tests establish execution only. The existing development folds are
  reused data, so a large local gain remains a hypothesis for the leaderboard.

## Tomorrow: controlled leaderboard checks

Select the queue after local experiments finish. Only then fit selected
configurations on all twelve months with the exact validated feature logic,
filters, rule, rounds, and seed.
Freeze the queue before inspecting its new official scores.

Suggested priority: a reproducible B0 reference if needed, selected recovery
alone, Q0, Q1, then a supported timestamp candidate if ready. Reuse v3's official
reference only if the final reference predictions match the saved v3 artifact;
otherwise allocate a new baseline submission for the changed pipeline.

- Respect the observed **five-submission daily limit**, with **00:00 UTC** reset.
  Check live usage before uploads; local calendar dates are not the quota clock.
- Saved 70,000/90,000 threshold outputs (v10/v11) were rejected for the daily
  limit and remain a separate unfinished series. Completing them would consume
  two slots, leaving three for this plan; explicitly choose which queue has
  priority rather than submitting more than five candidates.
- Allocate fresh unused versions at execution time. Link **submission filename
  ↔ candidate ID ↔ validation run ↔ source/data hashes ↔ configuration**.
- Distinguish uploaded, pending, rejected, and successfully scored. A rejection
  is not a scored experiment. Reuse frozen prediction files when appropriate;
  a quota failure does not require retraining.
- For each controlled pair, compare the **direction and magnitude of the local
  RMSE change** with the official change. Track both local readouts and their
  uncertainty rather than fitting one offset from a single observation.
- Keep v3 and legacy validation score definitions separate. Extend the result
  record/logging path as needed for postprocessed candidates before uploading.
- Repeated local/LB direction disagreements trigger a validation review before
  further feature expansion. Agreement across a few selected candidates remains
  provisional evidence, not a calibrated general forecast.

## Experiment ledger

Fill rows immediately after each experiment, including failed/inconclusive ones.

| Candidate | Parent / scope | Status | Validation run | R_A / Δ | R_B_long / Δ | Submission / official RMSE | Decision |
|---|---|---|---|---|---|---|---|
| B0 | Recovery off, 1,500 rounds | **Complete** | `v3_20260915T225301657023Z` | 371.0708 | 335.2416 | Not submitted | Frozen reference; see Stage 0 results for the 800-round comparison |
| B0-800 | Recovery off, 800 rounds, laptop control | **Complete** | `v3_20260916T095948645262Z` | 366.7465 | 334.1785 | Not submitted | Matched local reference; same data identities as Stage 0 |
| R1 | B0-800 + exact twins | **Complete** | `v3_20260916T102029635587Z` | 365.6176 / −1.1289 | 335.7478 / +1.5693 | Not submitted | Do not promote: longer-gap regression; see tradeoff below |
| R2 | R1 + relaxed twins | **Complete** | `v3_20260916T104452470012Z` | 365.3698 / −0.2478 vs R1 | 335.3479 / −0.3999 vs R1 | Not submitted | Parent deltas inconclusive; still regresses against B0-800 on longer gaps |
| R3 | R2 + static history | **Complete** | `v3_20260916T113256202842Z` | 360.5041 / −4.8656 vs R2 | 333.3043 / −2.0435 vs R2 | Not submitted | Select provisionally for Stage 2; supported R_A gain versus control, R_B_long inconclusive |
| Q0 | B0-800 + six-hour rule | **Complete** | `v3_20260916T123230629423Z` | 300.5904 / −66.1561 | 256.1971 / −77.9814 | Not submitted | Supported gains on both readouts; conservative Stage 3 parent |
| Q1 | R3 + six-hour rule | **Complete** | `v3_20260916T123433565511Z` | 296.9037 / −63.6005 vs R3 | 258.5505 / −74.7538 vs R3 | Not submitted | Retain alternative: versus Q0, better next-month but worse longer-gap |
| T1 | Q0 + explicit clock disagreement | Planned | — | — | — | — | — |
| T2 | Q0 + residual prediction with the same added features | Planned | — | — | — | — | — |
| M1 | Probabilistic timestamp-pattern model | Planned | — | — | — | — | — |

Scope changes, newly discovered recording patterns, and postponed work belong
here before changing the experiment order. No HPO or another target-threshold
sweep is included in this plan.

### R1 decision before advancing to R2

R1's next-month difference is inconclusive: **−1.1289 s**, 95% interval
**[−3.6378, +0.9182]**. Longer-gap RMSE regresses by **+1.5693 s**,
**[+0.6539, +2.6897]**. All three seeds worsen on the long-gap readout.
The provisional combined change is +0.3145 s and inconclusive.

The next-month gain is concentrated in July (−3.9098 s), August (−1.3021) and
November (−2.6355); September, October and December worsen. Longer-gap ordinary
RMSE rises from 240.9446 to 242.1609 s, and tail RMSE from 16,982.0055 to
17,055.2772 s. The unrecovered-row change is **+1.0335 s [+.3748, +1.7581]**;
the raw Rome/no-plan slice worsens by **+61.6773 s [+4.8645, +181.5030]**.
This is broader than a handful of newly filled rows. Among airports, Rome's
long-gap RMSE rises 6.5545 s and Paris 1.8723 s.

**Keep B0-800 selected; do not promote exact recovery alone.** Continue R2 as
a separately measured extension with both R1 and B0-800 comparisons. Higher
matching coverage is not presumed to resolve the regression. Saved R1 audit,
predictions, paired results, slices and training-loss plots remain available.

### R2 decision before advancing to R3

R2 versus B0-800: R_A **−1.3767 s [−3.8862, +0.6296]**, inconclusive;
R_B_long **+1.1694 s [+0.0258, +2.5910]**, regression. All three long-gap seed
differences are positive. Against R1, R_A −0.2478 [−0.6966, +0.2143] and
R_B_long −0.3999 [−0.9668, +0.1692] are both inconclusive. The combined score
is effectively unchanged versus control (−0.0146 s).

Longer-gap ordinary flights regress +1.2481 s [+0.5235, +2.0449], including
matched non-delayed ordinary flights (+0.4082 [+0.0650, +0.7721]). Recovered
rows, the raw Rome/no-plan slice and delayed flights have inconclusive changes.
**Keep B0-800 selected; do not promote R2.** R3 proceeds as the separately
registered static-history extension, with both R2 and B0-800 comparisons.

### Stage 1 selection: R3, with B0-800 retained as control

R3 versus B0-800: R_A **−6.2423 s [−10.6670, −1.7190]**, improvement;
R_B_long **−0.8742 s [−8.4406, +6.8701]**, inconclusive. Against R2, the
next-month gain is also supported (−4.8656 [−8.3539, −0.9907]); the longer-gap
change remains inconclusive (−2.0435 [−8.8474, +5.1197]).

This meets the plan's main-readout criterion, but is not a uniform improvement.
Longer-gap ordinary flights regress +1.6140 s [+0.3200, +2.9784]; matched
non-delayed ordinary flights regress on both readouts. Next-month ordinary,
delayed and raw Rome/no-plan slices improve. Tail error remains large and its
change is uncertain. The [full Stage 1 report](reports/STAGE1.md) retains the
airport, seed and recovery-slice evidence rather than treating coverage as a gain.

Proceed to the fixed-rule comparisons **Q0 = B0-800 + rule** and
**Q1 = R3 + rule**, using saved predictions. Keep these local development
results distinct from official leaderboard evidence. No Stage 2 rule, full-year
refit or upload was executed during Stage 1.

### Stage 2 selection: Q0 default, Q1 retained as a tradeoff

Q0 − B0-800: R_A **−66.1561 s [−98.3525, −35.9451]**, R_B_long
**−77.9814 s [−143.3944, −10.0573]**. Q1 − R3: R_A **−63.6005 s
[−94.3112, −34.1482]**, R_B_long **−74.7538 s [−138.6469, −8.5358]**.
All four gains are supported under the registered paired bootstrap.

Q1 − Q0: R_A **−3.6867 s [−6.8853, −0.7738]**, improvement; R_B_long
**+2.3534 s [+0.6283, +4.1240]**, regression. All three seeds agree with
these opposing directions. Combined change −0.5437 [−2.3785, +1.2817] is
inconclusive. Q0 is the conservative default; Q1 does not dominate it.

Both rules affect the same 44 unique next-month flights: 43 improve and one
August flight worsens. All eight affected longer-gap flights improve. The bad
override is movement 195015201: recorded taxi time 1,078 s versus rule 23,039 s.
Keep it in evaluation; do not add a target-informed exception. Every changed
ID and error difference is saved. Their predictions are identical on triggered
rows; Q0/Q1 differences therefore come from other flights.

Stage 2 reused all fitted models, with zero new full benchmark fits. Its two
candidate runs took 1.567/1.599 minutes to register and score; recovery audits
and CPU measurements were not repeated. Continue with Stage 3 only when asked.
