# Selected leaderboard candidates

Selected 2026-09-16, after Stages 1–4 and before viewing any new official
scores for these candidates. This freezes the first five-candidate queue.
Selection is complete; no full-year refit or upload has been performed here.

## Codenames and recipes

All five use the original 27 model features, direct taxi-time regression,
800 rounds and the existing 1–200,000-second training-label filter. Final
models use seed 42 and all twelve months of 2025. History features retain
their existing 1–7,200-second inputs. There are no T1 clock additions, residual
targets, mixture models, seed ensembles or new training-row exclusions.

| Order | Codename | Actual recipe | CV next-month | CV longer-gap | CV combined |
|---|---|---|---:|---:|---:|
| 1 | **B0-800** | Original baseline model; recovery off; no prediction rule | 366.7465 | 334.1785 | 348.9770 |
| 2 | **Q0** | B0-800 plus the **six-hour Rome rule** | 300.5904 | 256.1971 | 276.7374 |
| 3 | **Q1** | Baseline with **R3 flight-plan recovery**, plus the **six-hour Rome rule** | 296.9037 | 258.5505 | 276.1937 |
| 4 | **Q0-4h** | B0-800 plus the same Rome rule at **four hours** | 296.2785 | 255.6200 | 274.3707 |
| 5 | **Q1-4h** | Baseline with **R3 flight-plan recovery**, plus the Rome rule at **four hours** | 292.8785 | 257.4787 | 273.7215 |

CV values are RMSE in seconds, lower is better, with the standard three-seed
loss average. They are not official scores or seed-42-only results. The
combined score is a provisional summary. Four hours is a candidate selected
from an adaptive sweep on reused folds, not an independently confirmed optimum.
Q0 remains the conservative control; selecting candidates for LB testing does
not promote either four-hour variant or Q1 as a universal improvement.

**Rome rule:** if `ADEP_mvt == 'LIRF'`, the original `FLIGHT_ID_mvt` is missing,
and `MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` is at least the threshold, predict
that gap instead of the model's output. Thresholds are inclusive: six hours
is 21,600 seconds; four hours is 14,400 seconds. Else keep the model prediction.
The original missing-ID condition remains unchanged after recovery.

**R3 recovery:** fill eligible missing flight-plan fields from uniquely matched
arrivals using exact and relaxed matching, then fill reliable stable categories
from history. Existing raw values take precedence. It does not repair taxi-time
labels or invent off-block timestamps from history. The model feature names
remain the same; recovered field values can change their inputs.

## What each comparison answers

| Candidate minus reference | Question |
|---|---|
| Q0 − B0-800 | Does the six-hour rule's baseline CV gain transfer to LB? |
| Q1 − Q0 | Does flight-plan recovery help when the six-hour rule is fixed? |
| Q0-4h − Q0 | Does lowering the baseline rule threshold from six to four hours help? |
| Q1-4h − Q1 | Does lowering the recovered model's rule threshold help? |
| Q1-4h − Q0-4h | Does recovery help when the four-hour rule is fixed? |

Record official and CV score levels, changes in seconds and percent, direction
agreement, LB-minus-CV gaps and delta-LB-minus-delta-CV for these explicit pairs.
Include seed-42 CV as a secondary match for the single-seed final models and
retain the standard three-seed readouts and their uncertainty. Follow the
[expanded comparison scope](../IMPROVEMENT_PLAN.md#tomorrow-controlled-leaderboard-checks).

Standalone R3 is outside this first queue, so the pure recovery effect without
a rule is not isolated on LB here. T1 stays an inconclusive reserve; R1, R2,
T2 and M1 are excluded because of their validation regressions. Other rule
thresholds and the older v10/v11 label-ceiling series are outside this queue.

## Validated sources and preparation still to do

| Candidate | Validation source |
|---|---|
| B0-800 | `v3_20260916T095948645262Z` |
| Q0 | `v3_20260916T123230629423Z` |
| Q1 | `v3_20260916T123433565511Z`; recovered model parent `v3_20260916T113256202842Z` |
| Q0-4h | B0-800 saved predictions, four-hour row in the [threshold results](../reports/STAGE2_THRESHOLD_RESULTS.md) |
| Q1-4h | R3 saved predictions, four-hour row in the same threshold results |

`B0-800` specifically means the matched 800-round control, not historical
1,500-round B0. `Q0-4h` and `Q1-4h` are newly selected queue labels for existing
validated sweep results. They do not yet have registered linked submission
recipes; Q0 and Q1 themselves keep their original six-hour identities.

Before executing the queue:

1. Register the four-hour candidates from saved predictions and connect their
   exact threshold logic to submission recipes. Preserve the frozen baseline
   and R3 model source; do not bypass current source-mismatch guards or silently
   refit with Stage 3/4 feature/model changes. No full CV retraining is needed
   just to register the already-evaluated four-hour postprocessing.
2. Produce only **two underlying full-year models**: recovery off and R3.
   Reuse each parent's predictions to make its rule variants, preserving the
   same raw eligibility, ranking-month isolation, rounding and template order.
   Reuse an existing full-year model/artifact only after verifying its exact
   recipe and provenance; do not assume the historical official v3 is identical.
3. Freeze prediction artifacts and link codename, validation source, final
   recipe/source/data hashes and fresh submission version. Assign versions
   only when preparing execution; no `vN` identifiers are reserved here.
4. Check live submission status and available quota before uploads. This
   selected queue takes priority over the separate v10/v11 series. No automatic
   quota-reset retry or extra candidate is added by this selection.

After scored results arrive, preserve this queue and record successes, failures
and inconclusive transfers. Discuss rule-row training exclusion only after
assessing that evidence; specialist rule correction remains later work.
