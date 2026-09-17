# Recovery and timestamp modelling: experiment plan

Written 2026-09-16. This is the working order for the next experiments.
Update the status and results here as each stage finishes.

## Resume here — overnight submissions and held-out CV confirmation complete

The user-authorized 2026-09-17 overnight investigation is complete. Start with
the [final summary](reports/OVERNIGHT_SUMMARY_20260917.md) and
[held-out confirmation](runs/overnight_20260917/confirmation_report.md).

| Candidate | Submission | Official RMSE | Study role |
|---|---|---:|---|
| Remove calendar month | v17 | 345.8984 | Discovery |
| Remove departure-target history | v18 | 342.5049 | Discovery |
| Explicit clock features, direct target | v19 | **338.1113** | Discovery |
| Smaller trees, 63 leaves | v20 | 345.3113 | Confirmation |
| Exclude rule-handled rows from booster training | v21 | 342.4009 | Confirmation |

All five retain recovery off, 800 rounds and the same four-hour Rome rule.
Every validation row remains scored; routing changes only booster fitting.
**v19 is the best observed submission**, improving the unchanged Q0-4h/v15
study anchor by **6.2001 s**. Latest is v21. The live snapshot at
**2026-09-17 04:23:02 UTC** records rank **94/152** scoring teams, 416 registered;
see [saved status](reports/overnight_20260917/status_after.json).

CV selection compared 20 frozen forecast policies over 10 readouts, using the
earlier five submissions and three new discovery scores. The last two official
scores remained unread until the selection and forecasts were frozen.
**R_A__mse**, next-month MSE-difference transport, won discovery selection but
**failed confirmation**: held-out change MAE **3.5178 s**, versus **2.8367 s**
for existing R_combined. **Keep the default CV and all folds.** Modelling
improved, but these results do not establish a reliably better validation.
Further progress needs more varied future candidates with another untouched
confirmation batch; do not select a replacement from the now-revealed results.

Execution completed **153 new CV fits + 9 reused fits**, **360 prediction files**,
**6 full-year refits** (control plus five candidates), all five uploads and
**101 passing tests**. Preserve `runs/overnight_20260917/`, its source/data
signatures, submission manifest and immutable `cv_selection_frozen.json`.
The [original design](docs/OVERNIGHT_STUDY_20260917.md) and its amendment remain
frozen. Current machine: **7800X3D desktop; eight training/prediction threads,
one fit at a time**. Do not rerun completed work. All five authorized daily slots
were used; **additional submissions and automatic reset retries are not authorized**.
No notebook files were changed.

### Earlier validation investigations — historical context

An earlier user request concerned **aggregation**, independently of removing
folds. A proposed [month-balanced overview](reports/AGGREGATION_COMPARISON.md)
keeps all 16 folds and assigns each validation month 1/6 of the final MSE after
averaging its training origins. Its saved-run level/change errors are 49.41/7.13 s
versus 70.73/7.19 for the old combined number, with 3/3 material directions in
both. Recovery alignment worsens, and long-gap influence decreases, so R_A,
R_B_long and matched-month aging must remain separate checks. The default harness
and historical readouts are unchanged; this is a concrete proposed summary,
with fixed structural sensitivities, not a fitted LB calibration or selected
fold subset. See the report for all weights, seed checks and recipe scores.

The user clarified the decision criterion: identify demonstrably harmful folds,
not the subset that happens to match the five LB scores best. The session
conclusion is to **retain the current validation**: seasonal validation is not
clearly better overall, and no fold has a compelling removal case.

The subsequent [optimism diagnosis](reports/VALIDATION_OPTIMISM.md) is complete.
It finds a concrete evaluated-population mismatch (69.17% autumn weight, limited
July/no January coverage, and fewer raw missing-AOBT3 rows than ranking). Same-
July seasonal fitting improves scores, while two expensive January LFPG outcomes
make seasonal absolute levels highly unstable. No scoring bug or complete causal
explanation of the hidden 2026 errors was established. All folds/rows remain,
no refits/uploads ran, and no offset/reweighting is selected. A saved-model
context ablation is a possible separately scoped follow-up, not started work.

On 2026-09-17 the user requested two new validation splits for the five existing
submission recipes, comparing absolute scores and meaningful change directions
with their recorded LB scores: ten-month training excluding January/July, and
two eleven-month fits holding out each month separately. The runner, explicit
history/source policy, reports and commands are in
[SEASONAL_VALIDATION.md](docs/SEASONAL_VALIDATION.md). The user's full run completed
18 fits at three seeds; [results](runs/seasonal_jan_jul/comparison.md) show closer
absolute scores but worse change-size alignment than v3. No new strategy was
selected. Work is on `seasonal-validation`,
based on the completed leaderboard branch, with no notebook changes or uploads.
At that point, the seasonal comparison preceded the proposed training-routing
work. Routing has since been tested in the overnight study above.

The user then requested original-fold alignment and possible pruning. The
[fold analysis](reports/FOLD_ALIGNMENT.md) is complete, with all 16 fold scores,
seed readouts, leave-one-fold/month checks and explicit exploratory subsets.
Only nine folds contribute to the combined headline; November/December receive
69.17% of its effective squared-error weight. A-only improves mean score/change
errors to 50.05/6.46 s (combined: 70.73/7.19), but loses some of the combined
score's accurate cancellation of recovery effects. August+September tracks
changes at 2.65 s mean error; July–September matches levels at 17.52 s mean error
but overstates improvements. None is an independently confirmed replacement.
The original prediction parquets are missing on this machine; saved aggregates
reconstruct scores exactly, but cannot supply new paired day-bootstrap intervals.
No training, uploads, fold-definition changes or automatic pruning occurred.
The subsequent user clarification and final no-pruning decision are recorded
at the top of this resume section; the exploratory subsets were not selected.

### September 16 leaderboard checks — Q0-4h best in that batch

**All five selected submissions succeeded on 2026-09-16.** On branch
`leaderboard-five-candidates`, based on Stage 4: B0-800/v12 **406.4281**,
Q0/v13 **354.4303**, Q1/v14 **354.0004**, Q0-4h/v15 **344.3114**,
Q1-4h/v16 **344.4821** seconds. The live snapshot at 16:46:48 UTC puts the
team 96th of 152 scoring teams, with v15 best and v16 latest. The four-hour
baseline improves on the matched baseline by **62.1167 s (15.28%)**.
Rule gains transferred to LB; recovery is mixed across thresholds. The
[full report](reports/LEADERBOARD_RESULTS.md) records all three CV readouts,
seed-42 results, direction agreement, LB–CV gaps and paired changes/intervals.
The [execution note](docs/LEADERBOARD_EXECUTION.md) and
[manifest](reports/leaderboard_selected_20260916.json) preserve exact recipes
and artifacts. Two full-year fits took 2.067/4.461 minutes; four-hour recipes
were registered from saved predictions with no CV refits. All 56 tests passed.
That day's five-slot allowance was consumed. Q0 remains the six-hour control,
and Q0-4h remains the frozen reference for the subsequent overnight study.
Training routing was proposed here and has now been tested as v21; learned
rule corrections remain untested. No extra submission is authorized automatically.

### Completed modelling stages and selection history

**Stages 1–4 are complete. Do not promote M1; keep Q0 as the conservative
control.** Work is on `stage4-timestamp-mixture`, based on Stage 3. M1
`v3_20260916T144408082362Z` scores **544.8410/519.0849** (R_A/R_B_long), a
supported regression against T1, Q0 and Q1. Every seed regresses. The
[Stage 4 report](reports/STAGE4.md) records overconfident classification errors,
false high predictions, training curves and the [frozen recipe](docs/TIMESTAMP_MIXTURE.md).
The benchmark took **51.768 minutes**, with 36 boosters in 18 bundles and
48 prediction files; controls were reused. All 55 tests and the completed M1
linked-recipe checks pass. No recovery audit, CPU probe, notebook-cache
rebuild, full-year refit or upload ran.
**The first leaderboard queue is selected:** B0-800, Q0, Q1, Q0-4h and Q1-4h.
See the [frozen candidate list and plain-language recipes](docs/LEADERBOARD_CANDIDATES.md).
No refit or upload was performed during selection; the subsequently authorized
execution is complete above. Do not start another modelling sweep automatically.
The user also expanded the later leaderboard checks to explicitly measure
direction agreement and the absolute LB–CV gap, as specified below.

Stage 3 work is on branch `stage3-clock-features-residual`, based on Stage 2's open PR branch.
**Stage 3 is complete. Keep Q0 (B0-800 + fixed six-hour rule) as the conservative
control. T1's changes are inconclusive; do not promote T2.** T1 scores
298.9328/256.2445 (R_A/R_B_long), versus Q0's 300.5904/256.1971. T2 scores
419.6226/439.4680, a supported regression on both readouts. Its schedule
fallback produces extremely large errors on some ordinary flights.
The [Stage 3 report](reports/STAGE3.md) records comparisons, source-specific
errors and the [frozen design](docs/CLOCK_MODELS.md). T1/T2 used 18 additional
clock features, identical training rows, recovery off, 800 rounds and the fixed
six-hour rule. They took 19.470/19.292 minutes, with no full-year fits or uploads.
Q1 remains an alternative with better next-month but worse longer-gap results
than Q0. Read [Stage 2](reports/STAGE2.md), [Stage 1](reports/STAGE1.md), and
[Stage 0](reports/STAGE0.md) for the round-count experiment.

The user-authorized [4–8 hour follow-up sweep](reports/STAGE2_THRESHOLD_SWEEP.md)
is also complete on B0-800 and R3, using saved predictions in 1.621 minutes.
Four hours gives the best point estimates: 296.2785/255.6200 for B0-800 and
292.8785/257.4787 for R3 (R_A/R_B_long). Its next-month gains over six hours
are supported by pointwise intervals; longer-gap intervals touch zero.
Retain four hours as a promising sensitivity candidate. Q0/Q1 remain six-hour
controls; four-hour recipes were subsequently registered for the controlled LB check.

The [1–3 hour extension](reports/STAGE2_THRESHOLD_SWEEP_LOW.md) is complete
(1.631 minutes, no fits). Four hours remains best on next-month and combined
point estimates for both parents. One and two hours regress on those scores;
three versus four is inconclusive. R3's two-hour longer-gap advantage is only
0.0744 s and inconclusive. This adaptive extension preserves the original
sweep artifacts and six-hour controls; no threshold was promoted.

The complete [1–8 hour score table and verdict](reports/STAGE2_THRESHOLD_RESULTS.md)
are recorded together. The user's future sequence is also recorded below:
check selected candidates on the leaderboard, then, if gains transfer, discuss
excluding rule-handled rows from model training and later correcting the rule
with a model. The selected submissions are complete above; training exclusions
were proposals at this stage. Training routing was subsequently completed in
the overnight study; learned rule correction remains a proposal.

| Experiment | Artifact / state |
|---|---|
| Original B0, 1,500 rounds | `runs/v3_20260915T225301657023Z/result.json`, frozen; its model/prediction files were absent on this laptop |
| B0-800, matched control | `runs/v3_20260916T095948645262Z/result.json`, complete; R_A 366.7465, R_B_long 334.1785 |
| R1, exact twins | `runs/v3_20260916T102029635587Z/result.json`, complete; **do not promote**, longer-gap regression |
| R2, plus relaxed twins | `runs/v3_20260916T104452470012Z/result.json`, complete; **do not promote**, longer-gap regression |
| R3, Stage 1 recovery candidate | `runs/v3_20260916T113256202842Z/result.json`, complete; R_A 360.5041, R_B_long 333.3043 |
| Q0, selected conservative parent | `runs/v3_20260916T123230629423Z/result.json`, complete; R_A 300.5904, R_B_long 256.1971 |
| Q1, retained tradeoff | `runs/v3_20260916T123433565511Z/result.json`, complete; R_A 296.9037, R_B_long 258.5505 |
| Rome threshold sensitivity | 4, 5, 6, 7 and 8 hours on both parents; complete, no new fits; [scores and interpretation](reports/STAGE2_THRESHOLD_SWEEP.md) |
| Lower-threshold extension | 1, 2 and 3 hours on both parents, with 4/6-hour controls; complete, no new fits; [scores and interpretation](reports/STAGE2_THRESHOLD_SWEEP_LOW.md) |
| T1, clock features | `runs/v3_20260916T132038893775Z/result.json`, complete; R_A 298.9328, R_B_long 256.2445; both changes versus Q0 inconclusive |
| T2, residual target | `runs/v3_20260916T134231489815Z/result.json`, complete; R_A 419.6226, R_B_long 439.4680; do not promote |
| M1, timestamp mixture | `runs/v3_20260916T144408082362Z/result.json`, complete; R_A 544.8410, R_B_long 519.0849; do not promote; 51.768 minutes |
| Q0-4h, linked rule recipe | `runs/v3_20260916T164054429797Z/result.json`; original four-hour sweep reproduced; LB v15 344.3114 |
| Q1-4h, linked rule recipe | `runs/v3_20260916T164247756836Z/result.json`; original four-hour sweep reproduced; LB v16 344.4821 |
| Selected leaderboard checks | All five scored; [full comparison](reports/LEADERBOARD_RESULTS.md); v15 best observed; 56 tests pass |
| Verification | 55 tests; M1 linked source checks pass; Stage 4 paired comparisons, component curves and false-high diagnostics complete; prior audits retained |

**The planned modelling stages are complete.** The
user's future leaderboard-transfer check and subsequent training-routing/rule-
correction discussion are also recorded below. Do not automatically start
another threshold or residual-baseline sweep. Keep Q0 as the conservative
comparison parent and retain the separate four-hour sensitivity candidate.
Do not repeat recovery audits, CPU tuning or notebook caches without relevant
changes. The selected full-year refits and uploads are complete above.
Historical Q0/Q1/recovery recipes retain their frozen-source requirements;
current code now includes M1; older T1/T2 and Q0/Q1 wrappers require their
frozen source for refits.

**Round count:** Stage 1 explicitly uses **800 rounds** for control and all
three candidates, following the Stage 0 checkpoint verdict. The historical
800-round predictions were unavailable here, so B0-800 was regenerated with a
new identity. Data hashes match Stage 0; its scores differ from the historical
checkpoint by less than 0.03 s. Do not relabel the original 1,500-round B0.

**Historical Stage 1–4 machine:** i7-13650HX laptop under WSL, 20 exposed logical CPUs and
about 11.5 GiB RAM. Use **one fit worker, six training threads, 20 prediction
threads**, based on [local measurements](reports/STAGE1_CPU.md). Do not transfer
the old 7800X3D desktop setting automatically. Overlapping a full-data audit
with fitting caused swapping, so full-data audits and experiments also run
sequentially. Thread comparisons preserved predictions to numerical precision;
explicit prediction thread counts gave bit-for-bit equality.

The user authorized local six-month training and validation, then explicitly
authorized the selected five full-year refits/submission variants. No automatic
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

Subsequently, the user explicitly authorized a separate 4/5/6/7/8-hour rule
threshold sweep on both baseline and R3. This is complete; see the linked
[follow-up report](reports/STAGE2_THRESHOLD_SWEEP.md). The original fixed
six-hour experiment and its submission recipes retain their identities.

The rule is an empirical candidate: the schedule does not reproduce every
extreme label, and some day-offset cases follow a different pattern.

## Stage 3 — clock features, then residual prediction

Complete: [results and verdict](reports/STAGE3.md). T1 is inconclusive;
T2 has supported regressions. Q0 remains the control. The design below was
fixed before the experiments; the failed T2 fallback was not changed afterward.

Start each comparison from an explicitly named parent selected above.

- **T1: explicit clock disagreement.** Add `AOBT_3 - schedule`,
  `AOBT_3 - EOBT_1`, agreement/spread between available taxi estimates, and
  provenance-aware availability flags. The takeoff-minus-clock features already
  exist; do not present them as new inputs.
- **T2: residual prediction.** Define and freeze a timestamp baseline using
  available flight-plan clocks with a documented missing-clock fallback. Fit a
  correction to that baseline and add it back for prediction. Derive any clock
  calibration/reliability parameters inside the training window.
- [x] Compare T1 with its parent, and T2 with the same features under direct
  target regression, so feature gains and residual formulation are separable.
- [x] Score reconstructed taxi times, not residuals; include all extreme labels
  and retain the same final nonnegative-prediction convention.

## Stage 4 — model the different timestamp patterns

Complete 2026-09-16: [results and verdict](reports/STAGE4.md). **Do not promote
M1.** Against Q0, R_A worsens +244.2506 s [+199.5467, +287.7271] and R_B_long
worsens +262.8877 s [+183.5490, +340.7331]. Both T1 and Q1 comparisons also
show supported regressions. The classifier makes near-certain wrong choices
on some ordinary flights; the day class has only 5–9 examples per training
window. See the [frozen M1 recipe](docs/TIMESTAMP_MIXTURE.md); no retuning was
performed under this run ID. T1 is the direct feature-matched parent.

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

## Future follow-up — leaderboard transfer, training routing and rule corrections

User proposal recorded after the threshold sweeps, 2026-09-16:

1. Submit a selected set of candidates to check whether improvements on the
   validation folds translate into better leaderboard scores. Compare against
   a matching baseline and record local and official changes together, using
   the controlled submission procedure below. Explicitly check direction
   agreement and LB-minus-CV gaps; the user added these to scope on 2026-09-16.
   The first five candidates are now selected in
   [the frozen queue](docs/LEADERBOARD_CANDIDATES.md); all five subsequently
   scored successfully. Rule gains transferred; recovery remained mixed.
2. **If the improvement transfers**, discuss a separate experiment that removes
   rule-handled rows from the model's training population. At prediction time,
   route those cases to the rule and let the model handle the other cases.
   Define the routing condition explicitly from fields available at prediction
   time. Rule eligibility alone does not establish that a label is incorrect.
   Keep every validation row in the score so exclusion cannot manufacture an
   apparent gain by removing difficult evaluation cases. Compare with the same
   model/rule recipe trained on the original population to isolate this change.
3. Later, consider a model that learns corrections to the rule's predictions
   (`recorded taxi time − rule prediction`), then adds the predicted correction
   back. Compare it with the uncorrected rule. This overlaps the residual idea
   in Stage 3 T2, but the proposed specialization to rule-handled rows is a
   separate design decision for discussion.

This records future intent rather than starting those experiments. The existing
training filters and six-hour Q0/Q1 definitions are unchanged. Stage 3 subsequently
ran as a separate authorized experiment; Stage 4 is also complete. No automatic
upload or refit is scheduled by this note.

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

The user-authorized selection is complete: **B0-800, Q0, Q1, Q0-4h, Q1-4h**,
in that order. The [frozen queue](docs/LEADERBOARD_CANDIDATES.md) records exact
recipes, source runs, CV results and controlled pairs. It was selected before
viewing new official scores. The four-hour recipes were subsequently registered
from saved predictions, reproducing the sweep without CV retraining.
Final preparation used two underlying twelve-month models (baseline and R3),
800 rounds, seed 42 and the exact validated logic, reusing predictions for rule
variants. All five authorized uploads succeeded; see the execution summary above.

For every successfully scored candidate, produce a linked comparison table:

- CV `R_A`, `R_B_long` and provisional `R_combined`, official LB RMSE, and
  **LB minus each CV score**, all in seconds. Positive means LB is worse.
- Against the matching reference, report **candidate-minus-reference changes**
  on LB and on each CV readout, in seconds and percent. Negative is better.
- Mark whether LB and CV move in the **same direction**, disagree, or show no
  change. Preserve CV confidence intervals: an inconclusive CV difference is
  still inconclusive even if its point estimate agrees with LB.
- Report **delta-LB minus delta-CV** to show how much the improvement magnitude
  transfers, and changes in the LB–CV gap. Use a paired-delta scatter plot if
  enough comparable submissions exist; correlation is descriptive only and
  should not be inferred from one pair or treated as reliable with few trials.
- Keep the standard three-seed CV loss average and also give seed-42 CV as
  the secondary match for the planned single-seed full-year refit. Record
  six-month CV versus twelve-month final training, evaluation-period shifts,
  and integer submission rounding so raw LB–CV gaps are not misrepresented
  as directly comparable holdout errors or pure overfitting estimates.

Use only confirmed official scores and exact recipe/artifact links. A rejected
or pending upload has no LB score and cannot establish alignment. The combined
CV score remains a provisional diagnostic, not a calibrated LB forecast.

The selected queue replaces the earlier provisional priorities. Standalone R3
and T1 are outside this batch; T2/M1 are excluded. Reuse v3's official reference
only if the final reference predictions match the saved v3 artifact; otherwise
allocate a new baseline submission for the changed pipeline. Preserve historical
model-source guards rather than silently using current Stage 3/4 wrappers.

- Respect the observed **five-submission daily limit**, with **00:00 UTC** reset.
  Check live usage before uploads; local calendar dates are not the quota clock.
- Saved 70,000/90,000 threshold outputs (v10/v11) were rejected for the daily
  limit and remain a separate unfinished series. Completing them would consume
  two slots. They remain outside the selected queue, which now has priority;
  check the actual remaining live quota before choosing upload timing.
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
| T1 | Q0 + explicit clock disagreement | Complete | `v3_20260916T132038893775Z` | 298.9328 / −1.6576 | 256.2445 / +0.0474 | Not submitted | Inconclusive; retain alternative, no promotion over Q0 |
| T2 | Q0 + residual prediction with the same added features | Complete | `v3_20260916T134231489815Z` | 419.6226 / +120.6899 vs T1 | 439.4680 / +183.2235 vs T1 | Not submitted | Do not promote; supported regression versus T1 and Q0, schedule fallback dominates |
| M1 | Probabilistic timestamp-pattern model | Complete | `v3_20260916T144408082362Z` | 544.8410 / +245.9083 vs T1 | 519.0849 / +262.8404 vs T1 | Not submitted | Do not promote; supported regressions versus T1, Q0 and Q1 |
| Q0-4h | B0-800 + four-hour Rome rule | Sweep complete; selected for LB; linked recipe pending | Saved B0-800 predictions; threshold sweep | 296.2785 / −4.3119 vs Q0 | 255.6200 / −0.5771 vs Q0 | Not submitted | Leading threshold candidate; six-hour Q0 remains control |
| Q1-4h | R3 + four-hour Rome rule | Sweep complete; selected for LB; linked recipe pending | Saved R3 predictions; threshold sweep | 292.8785 / −4.0252 vs Q1 | 257.4787 / −1.0718 vs Q1 | Not submitted | Leading threshold candidate with recovery; not a universal promotion |

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
