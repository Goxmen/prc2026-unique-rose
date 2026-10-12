# Full-day learned peer context: design for review

This is a new input representation, not a revival of the closed local-context
screen. Its donors use the airport/UTC takeoff-day grouping of the existing
public features. Exact public-source cohort/calendar identity is not established.
The earlier screen used 16 flights within 30 minutes.
Existing public features use fixed clock/reference sums; existing reconciliation
uses fixed prior distributions and calibrated likelihoods. Neither supplies a
learned, complete-day joint distribution of peers' plan clocks and physical
witnesses to the ordinary network.

The input-only pilot covers all 2,085,047 2025 departures. Distant paired donors
exist for 80.15%/89.17% of Jan/Jul rows and 72.55%/83.99% of rows without any own
physical witness. Median daily population is 590.5, maximum 852. A fixed collision
fixture preserves the 12 public features, the query's old local context, its own
inputs, and marginal physical values while changing the joint donor information.
These findings establish information and computational feasibility, **not a
predictive gain or numerical headroom estimate**. Cohort correlation could also
transfer sensor bias or cause negative transfer. No fits or candidate exist.

## Proposed single comparison

- Keep selected seed-42 TabM, all existing 168 ordinary inputs, raw-track CNN,
  ordinary training membership, labels, clipping, optimizer, 2048-row logical
  batches/512-row microbatches, random permutations, 24-epoch schedule, and
  21/17 fixed stops. No group-batched optimizer trajectory is required.
- Encode each donor with a small deterministic 17→32→32 SiLU MLP, without
  dropout. Inputs are the existing 15 audited local-context scalar values plus
  scheduled takeoff gap and its finite-value indicator. Use fixed physical
  scales (clock/3600, stand offset/2000, gap/7200, explicit validity masks),
  not statistics fitted on all dates. Preserve the original training-only
  TabM preprocessing exactly.
- Pool learned means and second moments, plus log donor count. A zero-initialized
  65→original-joined-width linear projection adds this context to the original
  joined vector immediately before its existing TabM ensemble view. Query
  clocks, original public inputs and physical embedding then interact with
  context through the unchanged nonlinear backbone. The original weights,
  initial outputs, gradients and dropout RNG must replay exactly. Construct
  additional modules within a forked RNG context; omit any new dropout.
- The donor bank includes all input departures irrespective of ordinary/copy
  routing, but excludes explicit VFR flights, matching the existing public
  departure population. Unknown flight rules remain included. No donor labels,
  predictions, raw point tensors or derived target flags enter this bank.
- Exclude the query and its aliases before pooling: same movement ID; known
  same aircraft within 120 seconds; exact known takeoff/AOBT/stand/runway tuple.
  Build deterministic alias lists once from inputs. Subtract their latent sums
  and squared sums per query. An empty donor context produces an exactly zero
  added branch, including after its projection bias becomes nonzero.
- Training donors belong only to the exact allowed UTC months for that fold;
  held donors belong only to their own held UTC month. Same-airport UTC-day
  pooling cannot cross these month sets. Audit this property by IDs and dates,
  not by trusting an allowed-month flag. Source values reflect the retrospective
  competition inputs; no online-causal claim is made. Old and final timestamp
  views require separate banks, as in the existing local-context implementation.
- Within each original random microbatch, recompute the current encoder for
  each unique required day. Do not cache parameter-dependent training encodings
  across optimizer steps. Padding and stable donor order permit deterministic
  reductions without scatter-add. Inference can encode each cohort once under
  the final frozen model and reuse its summaries.

At worst one 512-row microbatch touches 512 cohorts × 852 donors = 436,224
small tuples. Padded donor inputs occupy about 30 MB; each 32-wide activation
about 56 MB. The complete padded scalar bank is about 212 MB. Expected extra
memory is below 1 GB, relative to the original laptop preflight's 3.70 GB peak;
this is an estimate, not a measured guarantee. A full-width CUDA preflight must
reserve original inputs and checkpoint required batch/gradient execution before
allocating either fit. Reject infeasible execution rather than changing the
architecture or batch size after fitting begins.

## Matched comparison and gate

First reproduce saved original seed-42 predictions from the same laptop and
verify constructor/output/update/RNG parity with the new branch disabled.
Original controls are reusable only if those checks, original input hashes and
the trainer's unchanged update schedule all pass; otherwise obtain a genuinely
matched control before comparing outcomes. Freeze code, identities, inference
transport and gate before training. The primary comparison replaces only seed42
in the selected three-TabM pool, transports its serialized native component
delta once onto exact v80 and preserves every downstream rule. This follows the
existing independently audited component transport and avoids refitting or
retuning the leaderboard blend.

Evaluate the full original Jan/Jul population with a paired-day interval and
both months separately; report the matched selected-pool component and exact-v80 effects.
Require both months to improve and the paired interval's upper endpoint below
zero before considering the predeclared other two seeds. This is repeatedly
reused development evidence, not independent confirmation. A failed first
comparison closes this design; no donor-window/width/pooling/loss search follows.
No production or upload is authorized by the pilot itself.

Independent audits must cover all source columns/IDs/dates, input-only alias
rules, permutation invariance, label poisoning, query/alias exclusion, empty
context, complete gradients, trained checkpoint reloads, full raw-truth/score
arithmetic and the exact unchanged v80 control. The loss remains ordinary MSE.

## Why not a public aggregate loss now

The training cohort is fully supervised, so an extra sum penalty does not add
independent labels; it changes the loss covariance and model's allocation of
error between cohort means and within-cohort contrasts. Published cohorts differ
from competition cohorts. Moreover, the ordinary network excludes routed/copy
rows present in public totals. Subtracting those excluded rows' **true** labels
during training cannot be reproduced at inference and would create a target
shortcut. Hard equality to a source total is therefore unjustified. A principled
soft likelihood would require cross-fitted predictions for every routed
component, training-only source discrepancy calibration and the nonlinear
truncation/count model, substantially overlapping the existing reconciliation.
The proposed input encoder avoids these unresolved loss assumptions.

Evidence: `pilot_identity.json`, `pilot_result.json`, and the ignored complete
cohort table `pilot_cohorts.parquet`. The pilot's SQL audit checks all group
counts and 512 independently enumerated temporal neighborhoods.
