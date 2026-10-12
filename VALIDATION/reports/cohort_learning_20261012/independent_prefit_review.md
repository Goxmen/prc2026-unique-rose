# Independent prefit review

The full-day context design contains new input information beyond the closed
local-context experiment. An explicit timestamped fixture preserves all twelve
public features, the query's inputs, its old ±30-minute context and physical
marginals while changing joint information among distant same-day donors.
This establishes an information distinction, not predictive value. UTC takeoff
grouping follows the existing feature cohort; identity with the public source's
calendar/population remains unproven.

The separate bank audit passes every original ID/date/rule and all 35,395,989
scaled values for 2,082,117 donors in 3,650 groups. Three independent SQL joins
reconstruct every excluded edge. There are 2,082,117 actual exclusion edges,
all self edges; synthetic checks additionally exercise aircraft/signature
aliases and the inclusive 120-second boundary. Every donor belongs to exactly
the query's airport and UTC date, so different allowed UTC-month sets cannot
mix. No departure targets or model predictions enter this audit.

Independent active-context arithmetic agrees with explicit per-query donor
gathering within 1.87e−8 standardized context units. Empty/all-excluded contexts
are exactly zero, unrelated cohorts have exactly zero gradients, and included
donors have active gradients. Sum-minus-exclusion arithmetic leaves float32
gradient cancellation at most 1.58e−9 (3.33e−8 relative to the maximum query
gradient). An initial bitwise-zero audit expectation was corrected to an
explicit 64×fp32-epsilon relative bound; the model was unchanged. These values
are numerical checks, not scores or practical gain claims.

Static trainer/scorer review found no blocking logic error. The original
training update function is reused unchanged; administrative AST edits select
only member zero. Original initialization/dropout states are preserved around
full held-row control replay. The retained two members, all downstream rules
and weights remain fixed. The score adds the new-minus-old serialized native
component once to exact v80, followed by one nonnegative clip and rounding.
The saved-model-before-inference sequence preserves completed fits on resume.

Before fitting, require the final frozen source identities, refreshed CPU
constructor/output/gradient/RNG checks and same-laptop CUDA capacity preflight.
The preflight must reserve retained tabular/target arrays, RaggedTracks
geometry/quality and the full donor bank, and exercise the actual 512-row
training and inference width. No architecture or batch resize follows a failed
capacity gate. Full checkpoint, input and score audits remain required after
the fits. Any standalone member-MSE comparison is secondary to the frozen
exact-v80 Jan/Jul comparison and is not a new selection criterion.

The first laptop CUDA preflight stopped before capacity probing or fitting:
bitwise backward equality failed under the retained default CuDNN recipe.
A separate synthetic diagnostic repeated both original/original and
original/candidate comparisons four times in eval and training modes. Default
original/original gradients differ by up to 1.87e−9 and the first AdamW update
by 1.36e−6; original/candidate values are 2.80e−9 and 4.07e−6. Initial outputs
and random states stay exact. Scoped deterministic CuDNN makes all gradients,
original-parameter first updates and original-path post-step outputs bitwise
equal in all sixteen comparisons. No tolerance was relaxed. The approved
runner-only change scopes deterministic kernels to the parity assertions and
asserts restored flags before capacity/training. This does not claim that
default-backend training is bitwise deterministic. The original failed
sources/identities/log remain archived under `prefit_cuda_parity_v1/`.
Full per-parameter diagnostic: `cuda_repeated_control_diagnostic.json`, SHA256
`cf48dff162b05d884be6f7441e3f8a3a967659333279c2b353c224380453d0ba`.

Receipts: `independent_pilot_review.json`, `independent_bank_audit.json`,
`independent_context_model_audit.json`. Auditors live in `VALIDATION/tools/` as
`audit_daily_cohort_{pilot,bank,model}_20261012.py`. No fit or upload was run by
this review.
