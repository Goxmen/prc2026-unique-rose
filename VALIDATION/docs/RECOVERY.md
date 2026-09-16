# Fold-scoped flight-plan recovery

The validation harness and final refit share `pipeline/recovery.py` through
`modeling.training_matrix` / `validation_matrix`. The notebook caches are audit
references only; see [their inventory](../reports/STAGE1_DERIVED_DATA.md).

| CLI mode | Candidate | Fields supplied |
|---|---|---|
| `off` | B0 | Raw inputs, unchanged |
| `exact` | R1 | Missing plan fields from a unique exact arrival twin |
| `relaxed` | R2 | R1, then cross-field identifier A, normalized identifier B, route-duration C |
| `history` | R3 | R2, then supported static categories from departure history |

Recovery does not change the model feature list, movement flight ID, movement
ID, target, movement timestamps, or nonmissing raw values. Filled flight-plan
fields can change existing direct and historical features. Raw missingness flags
and recovery provenance are saved for diagnostics; they are not new features.
The existing aircraft feature still uses `AIRCRAFT_TYPE_mvt`; filled
`AIRCRAFT_TYPE_flt` is retained and audited but does not become an extra feature
or a movement-type fallback in these experiments. Plan route and identifier
fields likewise support matching/auditing without replacing movement fields.

## Availability and ownership

`public()` physically selects matching and plan fields, excluding departure
targets and block timestamps. The matcher rejects frames containing those hidden
columns. Training recovery sees only its own window. Each evaluation sees that
window and its own month. Its learned airline, duration and static references
come from raw training rows only. The training representation is recovered once
and reused across the window's evaluations; evaluation cannot rewrite it.

Final fitting uses 2025; January and July ranking contexts are processed
separately. Source hashes include all three recovery modules so a validated
recipe cannot silently submit changed matching/history code.

An exact match requires equal movement route and flight number, landing from
0 through 8 hours after takeoff, and compatible movement aircraft types (one
missing type is allowed). It counts candidates before applying the aircraft and
plan consistency guards, matching the conservative notebook exact stage.
Donor plan origin/destination must equal the departure movement route. Existing
departure flight IDs must agree. Plans owned by other available departures are
excluded; competing claims on either the donor movement or donor flight ID are
rejected. Movement-side flight IDs remain raw; recovered IDs live in the audit.

Relaxed stages require equal, nonmissing movement aircraft types. A/B use the
notebook's strict 15-minute/8-hour bounds. C uses a training route median with
at least 20 distinct known departure pairs and strict bounds of 0.8× median
through 1.3× median + 15 minutes. Prefix mapping requires at least 30 supporting
observations and 90% dominance. Ties are deterministic. Exact-stage ownership
is reserved before the relaxed waterfall, whose competing claims are rejected
across stages.

## Static history

Only original matched training departures supply static references. Training
queries exclude the same UTC day and all future days; evaluation freezes the
whole training-window lookup. The per-field cascade follows notebook §2c:
flight number + origin, prefix + origin, prefix, and own aircraft type where
specified. Support counts and modal agreement gate every assignment. Low-
confidence prefix flight-type assignments are excluded. No timestamp or flight
ID is invented. Per-field support, agreement, level and low-confidence flags
are serialized into the audit's `static_details` column.

Static-field accuracy also gets a separate delayed-flight cohort: visible
takeoff minus scheduled movement time at least three hours. This fixed
diagnostic follows the notebook's delay discussion and does not change model
inputs or predictions. Score summaries compare it with raw missing-plan slices
and matched, non-delayed departures whose recorded targets are at most 7,200 s.

## Artifacts and checks

Each completed training window saves `recovery_M<n>_training.parquet`; each
evaluation saves `recovery_<fold>.parquet`. Audit rows cover departures with at
least one missing plan field, including unrecovered cases. `result.json` contains
coverage summaries by airport, original missing-ID/block flags, stage and reason.
Recovered anchors and static-imputed rows are counted separately. Prediction
files keep the original plan-availability slice and add raw missingness flags,
recovery stage and anchor recovery.

In a multi-stage audit, `candidate_count` and an unrecovered row's original
rejection reason describe the exact stage. A successful relaxed stage replaces
the stage/confidence/reason; a detected relaxed collision has its own reason.

```bash
.venv/bin/python -m unittest discover -s VALIDATION/tests
.venv/bin/python VALIDATION/tools/audit_recovery.py --mode exact \
  --output VALIDATION/reports/stage1_exact_audit.json
.venv/bin/python VALIDATION/tools/audit_recovery.py --mode relaxed \
  --output VALIDATION/reports/stage1_relaxed_audit.json
.venv/bin/python VALIDATION/tools/audit_static_history.py \
  --output VALIDATION/reports/stage1_static_audit.json
.venv/bin/python VALIDATION/tools/summarize_recovery.py \
  --run <candidate> --baseline <control>
```

Audits mask 5,000 known plans per evaluation month, then separately hide their
true arrival plans. They report ID errors, coverage, airport and stage. Static
accuracy is measured on withheld training examples with earlier-day history,
and on July–December using only January–June references. Full benchmarks retain
all validation labels, three seeds, 2,000 paired day-bootstrap draws and the
protocol's weighting of repeated flights.

On the current i7-13650HX laptop, use sequential six-thread fits and explicitly
set `--prediction-threads 20` for separately measured inference. Full-data
audits also run sequentially with experiments: concurrent loading can exceed
the WSL instance's approximately 11.5 GiB RAM and cause swapping. The CPU probe
is machine-specific and does not establish a universal thread default.
