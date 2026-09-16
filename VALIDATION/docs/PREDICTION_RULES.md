# Fixed prediction rules

`pipeline/prediction_rules.py` contains the single shared implementation used
by saved-prediction validation and `submit.py`. `rome_6h` replaces predictions
only when the original movement has airport `LIRF`, a missing `FLIGHT_ID_mvt`,
and takeoff minus scheduled movement time at least 21,600 seconds. The new
prediction is that gap. Neither recovered plan availability nor the target
participates in the trigger; six hours is fixed, not a tunable command option.

The separate diagnostic `VALIDATION/tools/sweep_rule_thresholds.py` evaluates
the user-requested threshold grids on saved B0-800 and R3 predictions. The
default remains 4, 5, 6, 7 and 8 hours; `--hours` specifies an extension and
always retains the six-hour control. Use a separate output prefix to preserve
earlier experiment artifacts, for example:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python VALIDATION/tools/sweep_rule_thresholds.py \
  --hours 1 2 3 4 6 --output VALIDATION/reports/stage2_threshold_sweep_low
```

It reads each parent once, preserves per-flight/seed/origin loss weighting,
and checks the six-hour scores against Q0/Q1. Its report includes both main
scores, the provisional combined score, monthly results, helped/harmed counts,
changed IDs and paired day-bootstrap intervals. The sweep does not change the
fixed implementation or linked submission recipes. Its development-fold
comparisons are sensitivity evidence, not independent threshold confirmation.

## Register and compare without fitting

```bash
.venv/bin/python VALIDATION/postprocess.py \
  --parent <B0-800-run> --label Q0-rome-six-hour
.venv/bin/python VALIDATION/postprocess.py \
  --parent <R3-run> --label Q1-recovery-rome-six-hour
.venv/bin/python VALIDATION/compare.py --run <Q1-run> --baseline <Q0-run>
.venv/bin/python VALIDATION/tools/summarize_rule.py --q0 <Q0-run> --q1 <Q1-run>
```

Each candidate has its own run ID, configuration, source snapshot, original
model source, rule hash, data identities and paired parent comparison. Model
files and recovery audits refer back to the parent; no models are copied or
fitted. Each parent prediction file's hash and its resulting candidate hash
are recorded. Predictions remain local and ignored; retain their parents when
copying or archiving these dependent artifacts.

`changed_rows.csv` lists every triggered fold/seed prediction, including
movement ID, label, original prediction, rule prediction, absolute errors and
squared-error changes. The Stage 2 summary averages squared errors over seeds
and repeated training origins for each flight, then reports unique helped and
harmed flights by month. A mean prediction is descriptive only; it is not used
to compute an ensemble score. Triggered cases are retained even if an original
prediction already equals the rule value.

## Linked submission recipe

`submit.py --run-id <Q-run>` inherits `prediction_rule=rome_6h` and applies the
same function to raw ranking movements after prediction and before template
rounding. Explicit rule overrides, changed rule source and changed model source
are rejected. Rule and model hashes are separate: postprocessing does not
relabel current model code as the code that fitted an older parent.

Q1 inherits R3's model source, which matches the current implementation. Q0
inherits B0-800's frozen model source. The older control predates shared-wrapper
changes, so its previously documented source guard still blocks a full-year
refit using the current wrapper. Preserve the frozen validated model code when
preparing that refit; the rule feature does not bypass or weaken the guard.
No full-year refit or upload is needed to complete Stage 2.
