# Clock features and residual models — frozen Stage 3 design

Stage 3 is complete: [T1 is inconclusive and T2 regresses](../reports/STAGE3.md).
The schedule fallback below remains part of the recorded failed T2 recipe.

Stage 3 uses Q0 (`v3_20260916T123230629423Z`) as the control. Keep recovery off,
800 rounds, seeds 42/142/242, all six training windows and 16 validation folds,
training labels 1–200,000 seconds, all validation labels and the fixed six-hour
Rome rule. No training rows are removed based on rule eligibility.

T1 adds 18 explicit features, implemented in `pipeline/clock_features.py`:

- Four signed timestamp differences: AOBT3−schedule, AOBT3−EOBT1,
  EOBT1−IOBT, IOBT−LOBT.
- Availability of each of the five clocks, plus the original missing-ID,
  missing-plan-block and missing-AOBT3 flags (retained even after recovery).
- Count, span, population standard deviation and median of the available
  takeoff-minus-clock estimates. Missing clocks are ignored; all-missing
  spread/median stays missing and count is zero.
- A deterministic baseline and its source code. Choose the first finite gap
  in **AOBT3 → EOBT1 → IOBT → LOBT → schedule**, or zero if all are missing.
  Clip this baseline to zero from below. Source codes 0–4 follow that order;
  code 5 denotes the zero fallback. No label-derived calibration is fitted.

The existing takeoff-minus-clock inputs remain present. These additions are
explicit differences, summaries and provenance, not newly discovered clocks.
The baseline is included in both T1 and T2 so their feature matrices match.

T1 fits taxi time directly. T2 fits `taxi_time − clock_baseline_sec` on the
same rows/features and reconstructs `baseline + predicted_correction`.
Negative corrections are valid: clipping happens only after reconstruction.
Then the unchanged Rome rule runs on original raw movement fields. Saved
predictions include the baseline/source, reconstructed model prediction and
rule audit columns. Training RMSE measures the unclipped reconstruction before
the rule; final reported scores measure full taxi predictions after the rule.

Compare T1 with Q0, T2 with T1 (target formulation only), and T2 with Q0.
Use the existing paired day-bootstrap reporting, monthly/airport/ordinary/tail
slices and seed variation. This global residual experiment is distinct from
the user's future idea of fitting a specialist correction on rule-handled rows.

Validation, resume and linked submission use the same feature/target logic.
`--target-mode clock_residual` requires `--clock-features`. Model hashes include
the clock module; rule hashes remain separate. Existing historical models need
their frozen source for later full-year refits. No such refit or upload is part
of this stage.

```bash
# T1, compare against the frozen Q0 predictions.
.venv/bin/python VALIDATION/harness.py --label T1-clock-direct-i7 \
  --clock-features --target-mode direct --prediction-rule rome_6h \
  --rounds 800 --threads 6 --prediction-threads 20 --fit-workers 1 \
  --baseline v3_20260916T123230629423Z

# T2, same features, compare against T1.
.venv/bin/python VALIDATION/harness.py --label T2-clock-residual-i7 \
  --clock-features --target-mode clock_residual --prediction-rule rome_6h \
  --rounds 800 --threads 6 --prediction-threads 20 --fit-workers 1 \
  --baseline <T1-run-id>
```
