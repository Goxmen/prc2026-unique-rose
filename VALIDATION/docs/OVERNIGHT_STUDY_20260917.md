# Five new model interventions and prospective CV checks

The user explicitly authorized training, validation and **five new submissions**
on 2026-09-17, followed by CV investigation and a final Markdown summary. This
authorization supersedes the old handoff's deferral for this batch only. Live
preflight at 01:41 UTC showed zero of the five UTC-day slots used. The baseline
official reference remains Q0-4h/v15, RMSE **344.3114**; latest old submission is
v16, RMSE 344.4821. Final status must distinguish best from latest.

## Question and fixed experimental scope

The previous five scores mainly tested two parent models and variants of one
rule. The purpose now is broader evidence about whether local validation predicts
useful model changes. No guarantee of a breakthrough is assumed. The final report
must say whether the evidence supports a better CV and/or more submissions are
needed. Do not use per-row LB probes, tune predictions to hidden labels, or submit
extra candidates beyond the five authorized files.

All five retain recovery off, direct taxi-time regression, the **same four-hour
Rome rule**, 800 rounds, original LightGBM learning/sampling settings, training
labels 1–200,000 seconds and all validation labels. Historical taxi averages
retain their 1–7,200-second source population. Submission uses seed 42 and all
twelve 2025 months. Every local experiment uses seeds 42/142/242. The desktop
uses one fit at a time and eight threads. No seed search or hyperparameter sweep.

| Order | Recipe | Change from Q0-4h | Evidence role |
|---|---|---|---|
| 1 | `no_month` | Drop only the calendar `month` input | Discovery |
| 2 | `no_history` | Drop the five departure-target-history inputs | Discovery |
| 3 | `clock_features` | Add Stage 3's 18 observable clock inputs; keep direct target | Discovery |
| 4 | `small_trees` | Change only `num_leaves`, 255 → 63 | Confirmation |
| 5 | `rule_routing` | Exclude original-raw four-hour-rule-eligible rows from booster fitting only | Confirmation |

`no_history` drops `op_taxi_30d`, `op_taxi_30d_n`, `rwy_taxi_30d`,
`stand_taxi_90d`, `aphour_taxi_30d`; arrival/context features remain. Routing
computes eligibility from observable original fields, builds feature/history
context before the routing mask, preserves every evaluation row, and applies
the unchanged rule to final predictions. No target-dependent evaluation routing.

The additional local `control` reproduces Q0-4h through the same shared-matrix
implementation. Its missing original v3 artifacts must be regenerated to support
paired analysis. Its completed seasonal baseline artifacts should be imported
with provenance checks, not needlessly refitted. There are no control uploads.
Refitting a local full-year control allows comparison with the existing v15
artifact to measure host/replay drift, without spending a submission slot.

All implementations share fold-scoped feature construction. A common 45-column
matrix is built, then explicit recipe allowlists choose 22/26/27/45 model inputs.
The 27-column control must exactly match the historical feature construction;
the adapter's synthetic check verifies bitwise control predictions as well.
No historical model-source guard is disabled. New models have their own complete
source/data/runtime/recipe identity and must use it unchanged at final refit.

## Validation populations

- Original v3: six six-month windows, A1–A6 and B1–B10.
- Seasonal ten-month: exclude January/July and evaluate each independently.
- Seasonal eleven-month: separate fits holding out January and July.

All are scored on full populations, with saved per-flight predictions, targets,
original plan flags, model/rule predictions, training-loss curves and model hashes.
The seasonal December-history and context limitations remain explicitly those
documented in `SEASONAL_VALIDATION.md`; no new split is selected after seeing LB.

The full schedule contains 162 model/seed/window combinations including control.
Nine already-completed seasonal control fits can be imported, leaving 153 new
local fits. Full-year refits follow completed validation and have seed 42 only.
Run heavy jobs sequentially. Small synthetic/sampled checks establish execution,
never candidate ranking. No obviously broken implementation is to be uploaded.

## Prospective separation

Freeze this selection before new official scores. Train and validate all five,
prepare and upload all five immutable artifacts, then:

1. Fetch only `no_month`, `no_history`, `clock_features` official results.
2. Use the old five and these three as the calibration/discovery observations.
3. Freeze the chosen CV adjustment, exact source/input hashes, rationale, and
   predicted changes/absolute scores for **both confirmation recipes**.
4. Only then fetch `small_trees` and `rule_routing` official results.
5. Evaluate the frozen predictions without retuning. Any post-confirmation
   analysis is labelled exploratory and cannot claim these results as untouched.

Do not run a global live leaderboard/history check between upload and freezing
the CV selection: it would expose confirmation scores. Bucket object listings
for quota/version checks are fine; they contain no score payload. If confirmation
results are accidentally exposed, record the breach rather than claiming a holdout.

## Prespecified CV candidates and decision measures

Keep the following interpretable readouts available: R_A, R_B_long, original
R_combined, all-fold month-balanced, all-fold unique-flight pooled, seasonal
ten-month combined, and seasonal eleven-month combined. Also permit three fixed
MSE blends of month-balanced and seasonal-eleven-month scores, with month-balanced
weights 0.25, 0.50 and 0.75. No free 16-fold weight search, subset search, or
selection of arbitrary hard-row exclusions against the observed LB outcomes.

Use squared-error averaging across seeds and repeated origins, not averaged
predictions. Report three-seed and seed-42 readouts. Use paired day-bootstrap
comparisons from saved predictions; distinguish fitted-model/2025-day uncertainty
from year-to-year uncertainty. All targets remain in the actual score.

The primary calibration criterion is mean absolute error in candidate-minus-
Q0-4h changes. For old recipes, subtract their old Q0-4h local reference; for new
recipes, subtract the regenerated control under the same CV. This cancels small
host/replay differences. The corresponding anchored absolute forecast is
`344.3114 + local candidate-minus-control change`. A separate median additive
offset can be frozen for absolute level forecasting; it must not be confused
with better change/ranking prediction. Prefer the unchanged current readout
when it is within one second of the best calibration change-error criterion.

Assess more than selected direction pairs: material ordering errors, a local
near-tie that misses a material LB separation, change-size error, and the LB
cost of choosing the local winner. Use the predeclared one-second tolerance;
all pair comparisons are correlated and are not independent evidence units.
The Q1 versus Q0-4h cross-recipe miss must remain visible in historical comparisons.

Two confirmation results can falsify an apparently promising adjustment but
cannot establish universal reliability. Meaningful progress requires improving
confirmation change prediction without introducing material selection mistakes;
report the actual sizes and uncertainty rather than forcing a success label.
If evidence is mixed or small, conclude that additional varied submissions are
needed. All data still belong to the same competition evaluation population.

## Execution and provenance

New standalone modules are `overnight_models.py`, `overnight_run.py`,
`overnight_analysis.py`, and `overnight_submit.py`. Their substantive unit checks
and a short end-to-end smoke run must pass before full work. The production run
lives in `VALIDATION/runs/overnight_20260917/`; models/predictions/source copies
remain ignored, concise metadata/reports are retained. Resume verifies completed
artifacts and reuses them. Training source is frozen before the full run.

Allocate fresh unused versions from live bucket and local reservations. Preserve
template IDs/order/schema, finite nonnegative predictions and integer rounding.
Before each upload check live UTC quota; existing remote versions must match the
frozen SHA-256 or the action stops. Verify uploaded bytes by reading them back.
Pending/rejected uploads have no score. Exactly five new uploads are authorized;
there is no automatic extra batch after a quota reset.

The final report will be `VALIDATION/reports/OVERNIGHT_SUMMARY_20260917.md`, with
each recipe, all CV/LB scores, discovery/confirmation chronology, chosen CV and
held-out assessment, computation/reuse details, and a clear conclusion.
