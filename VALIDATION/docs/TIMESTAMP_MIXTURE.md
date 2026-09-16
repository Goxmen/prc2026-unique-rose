# Stage 4 — frozen timestamp-mixture design

Candidate **M1** changes the target formulation of T1. T1 is its direct parent
(identical 45 input features); Q0 remains the conservative control, with Q1
also compared. Recovery stays off and the fixed six-hour Rome rule runs last.
Use 800 rounds, seeds 42/142/242, all six training windows and 16 folds, and
the measured laptop settings: one worker, six fit threads, 20 prediction threads.
No additional threshold sweep or global clock-residual baseline is introduced.

## Training outcomes, fixed before the full run

Apply the existing training filter, 1–200,000 seconds. Define one class per
retained training row, using its target and observable schedule gap:

1. **Schedule:** target >7,200 seconds, finite takeoff-minus-schedule gap,
   and absolute target-minus-gap ≤60 seconds.
2. **One extra day:** not schedule, and target minus 86,400 is in [1, 7,200].
3. **Ordinary/other:** all remaining retained training rows. This is a
   catch-all, including unmatched extreme labels; the name is not a claim
   that all these records are physically ordinary.

Schedule takes precedence when definitions overlap. The 60-second tolerance
is a fixed modelling choice, not a tuned threshold; the day component covers
one positive day only. These are operational label patterns, not proven causes.
Neither hidden block timestamps nor validation targets enter model inputs.

Fit a three-class LightGBM probability model on **all** retained training rows.
Use the existing tree/learning/sampling parameters, multiclass log loss, and
three classes; no class weights, resampling or probability recalibration.
Fit a separate direct taxi-time regressor on ordinary/other training rows,
with the existing regression parameters. Record each component's population
and per-round training loss. This is two boosters per window/seed, 36 total.
An absent rare class is allowed; an empty ordinary component is an error.

## Prediction, with observable inputs only

Let `ordinary = max(ordinary_regressor(X), 0)`, `schedule = max(schedule_gap, 0)`
(fall back to ordinary if missing), and `day = ordinary + 86,400`.
The classifier supplies probabilities `p_ordinary`, `p_schedule`, `p_day`.

```text
mixture = p_ordinary * ordinary + p_schedule * schedule + p_day * day
final = existing_six_hour_Rome_rule(max(mixture, 0), original_raw_fields)
```

The schedule gap is weighted by a learned probability instead of always added
as in T2. The day component shares the ordinary regressor; it has no separately
fitted correction. Save probabilities, component predictions and pre-rule
mixture predictions. No hard argmax gate, target-informed evaluation routing,
new cap, or removal of evaluation rows is permitted.

Use the same model bundle, feature logic and prediction order for validation,
resume and linked submission. Fixed-round checkpoints apply the requested
round to both boosters. Inner early stopping is unsupported for this recipe.
The existing training-history features keep their 1–7,200-second population.
This component training split is distinct from the future proposal to exclude
rows by the observable Rome-rule trigger; rule eligibility never removes rows
from the classifier's training population.

## Evaluation and decision

Report the existing paired 2,000 UTC-day bootstrap comparisons versus T1, Q0
and Q1; keep both R_A and R_B_long, the provisional combined score, monthly,
airport, seed, ordinary, tail and ≤12-second readouts. Reuse saved references.
Additionally inspect ordinary Rome flights with model/final predictions above
7,200 seconds, with raw missing-ID and rule-trigger slices. Compare counts,
false-high squared-error contribution and the worst affected IDs to Q0/T1.
Use the protocol's equal-flight loss weighting across seeds/repeated origins.

Do not tune this recipe after inspecting its full-run result. Record a failure
as a failure; any revised definition is a separate experiment. Full-year refits
and uploads remain deferred until submission candidates are selected.

```bash
.venv/bin/python VALIDATION/harness.py --label M1-timestamp-mixture-i7 \
  --clock-features --model-mode timestamp_mixture --prediction-rule rome_6h \
  --rounds 800 --threads 6 --prediction-threads 20 --fit-workers 1 \
  --baseline v3_20260916T132038893775Z
```
