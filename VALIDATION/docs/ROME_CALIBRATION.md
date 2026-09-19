# Rome probability calibration — frozen September 19 design

User request: run the calibrated binary and calibrated three-class experiments
and record their validation scores in `SUBMISSIONS.md`. These are two local
validation candidates, with no full-year fit or submission. The user's latest
instruction is to implement and supply the command; the user starts the run.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python VALIDATION/rome_calibration_run.py --output-dir VALIDATION/runs/v3_20260919_rome_calibration
```

Add `--resume` to that same command after an interruption. `--dry-run` prints
the configuration and workload without training or writing a run. Completion
writes `report.md`, `scorecard.json`, diagnostics and comparisons in the run
directory, then appends exactly two validation-only rows to `SUBMISSIONS.md`.
Partial runs and execution-only smoke checks cannot update the table.

## Common control and scope

Use `v3_20260919_artifact_control`, the saved row-34 replay, as the matched
control (253.0698 / 228.9793 / 239.9468 s). Keep every original evaluation row,
all 16 folds, seeds 42/142/242 and the original protocol weights. Reuse its
ordinary predictions, 800-round ordinary models and training-window late means.
Outside Rome, predictions must remain exactly equal to this control. Preserve
the flight-plan window and no-ID cap after Rome prediction. Row 37's saved
artifact-combined result is an additional reference, not an assumed submission
identity: row numbers and remote upload names are separate records.

Both headline candidates remove the four-hour hard Rome override. They blend
ordinary/late taxi and schedule-gap estimates, so both can address the rule's
false positives. Save the uncalibrated binary mixture without that override,
the calibrated binary mixture with the original override, and the uncalibrated
three-class mixture as diagnostics. These require no additional model fits.
Compare binary calibration to the rule-free binary diagnostic, and three-class
to binary with the same rule treatment, as well as both to the row-34 control.

## Features, classifier outcomes and components

Reuse the selected stand-clock classifier's 54 inputs: the original 27,
18 clock inputs, airline/handler-share/handler-count/flight-number-format, and
five stand-clock inputs. Both classifiers see Rome training rows only.

Binary outcome: schedule-recorded iff |target − schedule gap| <= 60 s.
Reuse the existing full-window binary classifiers; fit equivalent classifiers
only for new inner splits. Three-class outcomes are schedule (same definition,
takes precedence), day (otherwise target − 86,400 in [1, 7,200]), and
ordinary/other (the remaining rows, including unmatched tails). Do not pretend
these labels prove a physical cause. All classifier training labels use the
existing 1–200,000 s training filter; evaluation has no target filter.

The three-class classifier uses the same 31-leaf, minimum-leaf-100,
300-round, learning-rate-0.05, feature/bagging settings as the current binary
classifier, changing only the objective to three-class log loss. No class
weights or resampling. This differs substantially from the older failed M1's
all-airport, 255-leaf, 800-round classifier and target >2h schedule definition.

Ordinary component: reuse the saved ordinary prediction, with the existing
training-window late estimate at no-ID Rome gaps strictly between 1 and 4h.
Schedule component: takeoff minus schedule, with ordinary fallback for a
missing gap; floor the final mixture at zero, matching the current binary
blend. Day component: 86,400 + ordinary prediction clipped to
[60, 7,200] seconds (one day plus a bounded ordinary duration).

The day probability is restricted to the *existing* observable day-correction
scope: Rome, no original flight ID, gap >=14h and exactly one Rome-local date
crossed. Outside it, renormalize the ordinary/schedule probabilities. Inside
it, shrink day probability toward zero by n_day/(n_day+20), where n_day is
the number of eligible day-labelled training flights in this outer window.
This fixed conservative constraint prevents a handful of day examples from
adding day-sized predictions to the general population. It cannot discover
day artifacts outside this scope. There is no independent day regressor.

## Calibration without outer-validation leakage

For each six-month outer window, build three expanding inner splits: months
1–3 -> month 4, months 1–4 -> month 5, months 1–5 -> month 6. Rebuild history,
category vocabularies and observable context at each inner cutoff. Inner
validation movements/arrivals are available only in their own month; their
departure labels never enter features or classifier fitting. Calibration uses
only retained-label Rome rows in these inner held-out months. Save IDs,
probabilities, outcomes, split metadata and hashes. Never use outer OOF labels
to estimate calibration parameters.

Per seed and classifier, fit separately for original-ID present/missing.
Calibrated logits are `a * log(p) + class_bias`, with ordinary bias fixed at
zero. The binary form is Platt scaling; three-class has a shared inverse
temperature and two class biases. Minimize summed log loss plus
`20/2 * ||(a-1, biases)||²`; a in [0.25, 4], biases in [-8, 8]. Clip input
probabilities at 1e-8. A stratum below 20 inner observations uses identity.
Calibration is probability-based; taxi RMSE and deployment after day shrinkage
are evaluated separately. Do not tune these settings after outer scoring.

## Workload, validation and records

18 inner origins × three seeds × two classifiers = 108 new inner boosters;
six outer origins × three seeds = 18 new three-class boosters. Reuse 18 outer
binary classifiers and every ordinary model: 126 new classifiers total.
One fit at a time, six training threads, 20 prediction threads on this WSL host.
Completed models, inner predictions, calibrators and outer predictions are
hashed and resumable; never repeat a completed fit.

Report R_A, R_B_long and combined, per-seed/per-fold scores, paired 2,000-day
bootstrap intervals (one-day primary; three-day sensitivity for the two
headline candidates), probability calibration, ordinary false-high predictions
and the rare day cohort's errors. Keep all flights in all published scores.
These are reused development folds, not independent confirmation. Add exactly
two candidate rows to the submission table, with official LB blank and
`Not submitted`; retain diagnostic ablations in the report/run files.
