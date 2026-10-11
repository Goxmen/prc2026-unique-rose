# Research toward 215 seconds — 11 October 2026

## Fixed recurrent track screen running, 19:03 Amsterdam

A single forward GRU reads the existing width 32 physical convolution states.
Its last observed state enters the unchanged mean/max representation through
one zero-initialized 32-to-64 projection. This adds 8,384 parameters, retains all
168 inputs and the original MSE/AdamW/OneCycle update, and fixes seed 42 and
21/17 epochs. It tests longer sequence order; there is no new source, wider
hyperparameter search, epoch selection or production fit. The implementation
uses [PyTorch's GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).

CPU and both desktop/laptop GPU preflights preserve original weights, outputs,
RNG, padding and missing-track fallback. Independent gate equations agree
within 1.11e-16; reversing valid sequence order changes the summary. Initial
track-gradient differences stay within the retained float-reduction tolerance;
other gradients are exact. The input audit checks all 2,085,047 index bounds
and 2,096 sampled raw chronological sequences against scalar tensor encoding.
The full-dimension 2,048-row synthetic benchmark uses 1.818 GiB; actual training
starts at 50.27 ms per batch and 4.754 GiB peak allocated memory.

Runner `nn_public_recurrent_jan_jul.py` reuses the exact retained training and
model/score audit bodies. Queue 1046322 runs both fits sequentially. A separate
fixed `recurrent_selected_stack.py` comparison was declared before any new
held-prediction file existed: matched seed 42 and one-of-three replacement,
through every v77 boundary, daily shift and quality gate. Queue 1047695 waits
for the first queue's completed audit, then runs this comparison and its
independent SQL/raw-truth/final-correction audit. No new candidate or upload.

The retained fit runner rewrote its standard preflight receipt with identical
checks plus two training-provenance flags. The complete automatic receipt is
preserved as `fit_preflight.json`; the queue-bound `preflight.json` was restored
byte for byte. Both hashes and the recovery are recorded. Model code, inputs,
checks, fitting and schedule are unchanged; all queue source hashes pass.

## Internal track-pool share screen closed, 18:55 Amsterdam

Changing only the public physical pool's neural share from 0.50 to 0.75 worsens
exact-v77 Jan/Jul CV 291.549710 →291.718041 (**+0.168330**), paired-day interval
[+0.061670,+0.318593]. The total neural share remains 0.71. All original v77
constraint/rounding stages reproduce exactly; independent SQL checks all
344,419 labels, dates, controls and scores. Retain
`v3_20261011_physical_pool_share/`; no search in the opposite direction,
production, ledger row or upload follows this result.

A fresh read-only public-code review found two newly published solutions:
[gracious-tornado](https://github.com/il-miscusi/prc-2026-gracious-tornado)
and [fearless-barn](https://github.com/e-seet/fearless-barn). Their README
self-reported scores, 233.1717 and 269.9888 respectively, do not establish a
better solution than our live best. Their described tree/network blends,
public aggregate corrections and ADS-B adjustments overlap retained work.
No external code was copied or executed, and no new signal was established.

Live 19:00 Amsterdam still reports v74 **221.6881**, rank 12/242 on 344,841
flights. The target 215 remains active. The unchanged v75/v76/v77 upload
watchers remain scheduled for 02:00/02:10/02:20 Amsterdam on 12 October.

## Current scored directions leave no material blend-limit gain, 18:46 Amsterdam

The fixed convex follow-up minimizes the worst-case affine MSE allowed by
four-decimal official score rounding, with coefficients summing to one. It
replaces the arbitrary coefficient box and ridge with the actual uncertainty
cost of each absolute coefficient. It uses only the 48 already scored vectors;
v75–v77 are explicitly excluded until scored. No new prediction direction,
hidden label, model fit or candidate is created.

On this same basis, the original bounded recipe projects 221.543213 and the
robust unbounded recipe projects 221.537157: only **0.006056 seconds** apart.
After measured output rounding, the latter's conservative score interval is
[221.247527, 221.826786], which does not establish an improvement over v74.
Jan/Jul CV is 293.540630; the historical reconstruction caveat remains.
There is no selected candidate, version reservation or upload. The current
coefficient limits do not explain the 6.6881-second target gap.

The synthetic known-label case requires a coefficient beyond the old box and
checks the score/serialization bounds. Independent SQL reconstructs all
344,841 ranking rows and 344,419 CV rows, official scores, raw truth and both
affine score identities. Maximum raw reconstruction difference is 1.75e-10
seconds. Retain `robust_scored_geometry.py`, its separate auditor and run
`v3_20261011_robust_scored_geometry/`. The existing next-window evaluations and
upload watchers remain unchanged. Live 18:38 Amsterdam still reports v74 at
**221.6881**, rank 12/242 on 344,841 flights; the 215 goal remains active.

## Continuous Rome targets closed after regression, 18:42 Amsterdam

All six laptop fits are complete and every prediction reloads exactly on the
desktop. Through the exact v77 stack, the continuous coefficient screen scores
291.708557 versus the same-laptop Logloss control's 291.537477: **+0.171080**,
interval [+0.026738,+0.305197]. It also loses 0.158847 against unchanged v77;
both months worsen. Binary RMSE alone loses 0.106427 against Logloss, and the
continuous target adds a further inconclusive 0.064653. No seed expansion,
production fit, model tuning, candidate or upload follows.

Independent SQL verifies all 344,419 labels, dates, gates, unchanged v77
stages and scores. The source packets, all six models, target/input audit,
cross-machine reload receipts and final score audit are retained. Both hosts
are idle after these completed experiments.

## Continuous Rome mixture screen on the laptop, 18:36 Amsterdam

`rome_continuous_mixture.py` tests a different training target for the existing
Rome mixture. On original training no-ID rows with schedule gaps strictly
between one and four hours, the target is `(y-other)/(gap-other)`, where `other`
is the retained training-only late mean capped by `max(gap-3600,60)`. This is a
continuous mixture coefficient, not a calibrated copy probability. It remains
unclipped during RMSE training; inference clips it to [0,1] and changes only the
same input-defined gate. Other training rows retain their original binary label.
Same-laptop Logloss and binary-RMSE controls separate the target and objective
changes. [CatBoost documents the RMSE objective](https://catboost.ai/docs/en/concepts/loss-functions-regression).

All80 inputs, original126,507/86,723 training rows, seed42 and1182/1401 rounds
stay fixed. Continuous targets affect1,085/502 training rows; the held gate
contains48 January and282 July rows. Original inputs/membership match and an
independent SQL calculation reproduces every target and gate exactly. Only the
two129.75MB prepared packets and the portable runner were transferred; both
hashes pass before fitting. The first preflight reached the still-incomplete
transfer and stopped before any fit; the completed transfer was then verified.
RTX5060 GPU checks support finite negative and above-one regression targets.
The first three January fits are retained; July follows sequentially.

The frozen scorer replaces just one of the three public Cat members inside
the gate, keeps R20's half unchanged, and recomputes all v77 corrections. It
must replay v77 exactly and independently reload all six models on the desktop.
No gate, round, weight, production fit or upload is selected by this screen.
Run `v3_20261011_rome_continuous_mixture/`; laptop worker10609.

## Time-weighted pooling closed; remaining error cost described, 18:33 Amsterdam

The fixed pooling screen worsens matched CV by0.115265 seconds, interval
[-0.109047,+0.362242]. One-of-three replacement worsens by0.028603,
interval[-0.045310,+0.110144]. Both months worsen. All full original/new reloads,
the separate1,094-row model check and all344,419 SQL truth/control/score checks
pass. Retain both models; no expansion, production or upload. Result SHA-256:
`56d558e62a29be7a807fbc02c1eb7b2a0588e596dbb4fa968a71630474bd6958`.

`tools/describe_remaining_error_cost.py` describes the actual v77 CV errors.
Rome accounts for44.64% of July SSE; its no-ID one-to-four-hour group accounts
for11.35% across282 rows. January remains dominated by extreme Paris labels:
its worst row alone carries42.81% of that month's SSE. Raw truth, calendar,
airport/track identity, exhaustive group totals and independent SQL scores all
agree. These are held-label diagnostics and optimistic perfect-prediction
ceilings, not attainable gains or 2026 forecasts. No new deployment rule follows.
Existing frozen-track Rome and short-window arrival-copy screens were reviewed;
their earlier inconclusive/negative results are retained without repeating fits.

## Gaussian likelihood closed after regression, 18:14 Amsterdam

The fixed observation-variance objective worsens matched seed42 CV
292.834296 →297.492245 (**+4.657949**), interval[+3.135877,+7.799266]. Its
one-of-three replacement worsens292.449363 →293.096278 (**+0.646915**),
interval[+0.359397,+1.155369]. January worsens1.150310 and July0.143519 in
the pool. Do not expand or tune this loss direction; no production or upload.

Both original/new full-held mean predictions and new variances reload exactly.
Independent model/input checks cover1,094 rows, with maximum mean difference
0.000306 seconds and maximum log-variance difference1.79e-6. All344,419
truth/calendar/control rows and SQL scores pass. Result SHA-256:
`7e600f4f7f6b32b9237bc8b90afe3c93dee56aa319937f951f4c6124921424f9`.
Retain both checkpoints and variance predictions; no fit is repeated.

Time-weighted pooling worker1016455 is now training under queue1002895 after
passing desktop CPU/CUDA checks. Its real first20-step benchmark is40.32ms per
batch with4.626GiB peak allocated GPU memory. The laptop full-shape candidate
workflow test has finished and its returned receipt/source hashes pass.
Live18:13 Amsterdam still gives v74 **221.6881**, rank12/242 on344,841 flights.
The215 target remains active; v75–v77 still have no official scores.


## Scored-basis packaging ready for actual results, 18:11 Amsterdam

`scored_basis_candidate.py` consumes the frozen51-vector diagnosis only after
all required actual official results exist. It verifies live scores and exact
source hashes before creating a version. Selection requires a conservative
upper RMSE bound at least0.25 seconds below the actual current best, or a bound
that reaches215; it skips an already-met goal. This is explicit leaderboard
selection. Jan/Jul CV is reported even when it regresses, with the inherited
historical reconstruction caveat. No new fit or weight search is performed.

`tools/audit_scored_basis_candidate.py` independently reconstructs the blend in
SQL as v64 plus weighted differences. The score identity, four-decimal score
uncertainty and measured clipping/rounding RMS give a conservative bound without
assuming hidden-label values. It checks every ranking row, every CV label/day/
month against raw truth, all weight bounds, schema and unchanged template values.
It produces the final review required by the existing guarded uploader.

Small synthetic known-label checks pass for reordered/signed weights, active
clipping, score rounding, malformed coefficients and corrupted output. A
separate laptop integration check runs the **exact production source bytes**
inside a temporary root with344,841 synthetic ranking rows,344,419 synthetic
CV rows and51 vectors. The full build and separate SQL audit pass. Missing
scores stop before any output; corrupt files fail the audit; existing candidates
cannot be overwritten. No real API is called and the temporary data is deleted.
Only the receipt is retained under `target215/scored_candidate_workflow_check.json`.
Its source hashes and remote/local receipt hash agree.

The initial integration run caught floating reduction order flipping549 exact
half-integer predictions. The builder now preserves the diagnosis's already
evaluated integer vectors. It independently checks unrounded arithmetic and
nearest-integer cells, and the SQL bound uses the actual serialized output.
This preserves the intended file rather than silently creating a numerical
variant. No real submission was affected.

**No actual v78, new version reservation, projected live score or blend upload
exists.** The actual v75–v77 results are still required. Gaussian training
continues on the desktop, followed by time-weighted pooling; the laptop test
ran independently with one heavy job per host.


## Early-public projection closed after full CV, 17:55 Amsterdam

Both fixed fits and all audits are complete. Matched seed42 CV changes
292.834296 →292.655692 (**−0.178604**), interval[−0.460545,+0.140037]. Replacing
only seed42 in the selected three-member pool changes292.449363 →292.375435
(**−0.073928**), interval[−0.169447,+0.032660]. January worsens0.079412 and July
improves0.227268 in the pool. Both comparisons are inconclusive, so there is no
seed expansion, full-year fit, candidate or upload.

All original and new full-held predictions reload exactly. The separate model
check covers1,094 rows with maximum0.000718 seconds difference, and independent
SQL checks all344,419 truth/calendar/control rows and every reported score.
Keep both completed models. Result SHA-256:
`b92bb102cc56233f6a8ea322581e58c4f19e71ae28c013f013a23bac227d88bd`.

Gaussian worker1004500 has started under queue992975. Its CPU/CUDA preflight
passes and the initial real training benchmark uses4.626GiB allocated GPU
memory. Time-weighted pooling1002895 follows it. All frozen queue source hashes
pass. Live17:54 Amsterdam still shows v74 **221.6881**, rank12/242,344,841 rows.
V75–V77 remain prepared with live upload watchers; the215 goal is not complete.


## Time-weighted track-pooling screen prepared, 17:51 Amsterdam

The retained physical encoder pools equally across observations. A new fixed
screen adds a centered time-weighted pool: each valid adjacent retained interval
contributes half its duration to each endpoint, capped at60 seconds per interval.
These are sampling weights, not a reconstruction of unobserved dwell time.
The weighted mean minus the ordinary mean enters a zero-initialized32→64 linear
projection alongside the original mean/max representation. Only2,048 weights
are added; the existing168 inputs, original parameters, output, RNG,21/17 epochs,
seed42 and optimizer recipe remain unchanged. Missing/single-point fallback and
padding checks pass. No interval cap or weight scan is planned.

`nn_public_duration_pool_jan_jul.py` binds the existing audit/training/scoring
body with identical bytecode and records its source hash in the design. It
retains pre-fit original-model reloads, checkpoint-before-diagnostics recovery,
complete new prediction reloads and independent model/SQL audits. Comparisons
are the matched seed42 model and its one-of-three replacement, as for the other
fixed neural screens. There is no production selection or upload.

`tools/audit_duration_pooling_inputs.py` checks2,107 cached flights against an
independent integer-timestamp loop. Maximum pooling-weight difference is
0.0002003 seconds, within the0.0017309 bound from the existing float32 time
encoding; padding stays zero and all weights are finite/nonnegative. Of1,698
usable sampled tracks,1,648 differ from equal observation weights; median L1
weight distance is0.4159. This establishes a nontrivial transform, not a gain.
No departure label is read in this check.

CPU checks and a separate RTX5060 laptop CUDA preflight pass. All66 imported
local source files match across hosts; only the new8KB model module and source
manifest were transferred. The return JSON hash agrees with the remote file.
The initial GPU gradient check found rounding across the track graph, maximum
1.40e-9 and2.40e-7 of tensor scale. Only track gradients permit eight dtype
epsilons times tensor maximum; tabular/public gradients, outputs, weights and
RNG remain exact. The pre-training revision is preserved; no fit was repeated.

Identity SHA-256:
`36f8ad24b8f58e2bf1c026735110acef6ef631bc9fb74163900c6412e75f44f7`.
Queue1002895 waits for Gaussian992975 as a resource barrier; it requires its own
completed input audit, so an unrelated Gaussian failure does not invalidate this
screen. The desktop still runs one heavy job. Early-public completed January
and is training July; its full CV score is not yet available.


## V77 prepared and queued; production complete, 17:34 Amsterdam

The laptop-trained daily main tree passes the desktop original-source and
all-344,841-prediction reload audit. The combined selected component also passes
its tree arithmetic, daily shifts, constraints, serialization and 344,419-row
CV audit. It scores **291.549710**, versus v76's **292.449408**; both months
improve. The main-tree increment over the full-CatBoost/quality parent is only
0.039768 seconds and receives no separate submission row.

V77 now contains the exact audited component bytes, SHA-256
`e8381979532aded6d795f84963c842bf31592713dbdd58c275f9006276364ae1`.
It changes 331,381 rows versus v76, RMS 24.571827 seconds. Its first packager
attempt failed before creating output because one ancestor stores source hashes
as a mapping instead of a list. Both formats now receive the same hash checks;
corruption and malformed records are rejected. No fit or prediction was changed.
The original terminal queues and recovery receipt are retained under
`target215/provenance_format_recovery_1733/`.

`tools/audit_daily_refinements_candidate.py` independently reconstructs the
three-tree mean and both constraint/rounding boundaries, and checks every CV
label/calendar against raw historical truth. It reuses completed model and daily
source audits. This is a separate root-authored implementation, not a second
agent review. All checks pass. The manifest is frozen at SHA-256
`dbf7ed321006ecb07d09814775c040b771d1b0dda811de50d2e4ac4603e2776b`.
Watcher990172 queues v77 for **12 October 00:20:05 UTC / 02:20 Amsterdam**, after
v75/v76. All three files remain unscored. Live17:33 Amsterdam still gives v74
**221.6881**, rank12/242; the215 target remains unmet.

The 51-vector basis is prepared and audited; all old50 vectors match exactly.
Evaluator992976 waits for the actual v77 result, then requires successful
v75/v76/v77 official scores, remote hashes and a fresh public score check.
No projected score exists yet and no automatic blend upload is scheduled.

## CUDA preflight corrected before neural training

Early-public stopped before any fit because its check demanded bit-exact CUDA
convolution gradients. A controlled repeat using **two identical networks**
shows the same difference: maximum9.31e-10, about1.99e-7 of the tensor maximum.
The Gaussian comparison showed the same convolution-only effect. All other
weights, outputs, gradients and RNG checks were exact. The diagnostic is in
`target215/cuda_gradient_diagnostic.json`.

Only CUDA `track.blocks.*.convs.*.weight` gradient comparisons now permit
`8 * dtype epsilon * tensor maximum`; CPU and all other checks remain exact.
Both complete CPU/CUDA preflights pass. Original identities and stopped queues
are archived under `target215/cuda_preflight_recovery/`; no model checkpoint,
fit or training result existed. The original architecture/optimizer/epochs and
training function remain unchanged. Queues992974 →992975 restart early-public
then Gaussian. The laptop is idle after its two completed production fits.

## Broad offset opportunity described, not selected

`tools/describe_remaining_offset_bias.py` independently matches all344,419
selected CV rows to raw truth/calendar/IDs and describes four fixed, separate
constant-offset families. Held labels choose the best shared January/July offset,
so this is an optimistic descriptive diagnostic, **not honest validation**.
The best mean-monthly-RMSE changes are −0.000377 for all rows, −0.080933 for
non-Rome missing-ID rows, −0.076104 for Rome missing-ID rows and −0.066954 for
Rome identified flights with a1–4h schedule gap. Analytic derivatives, expanded
SSE and direct row calculations agree.

These results bound only the stated unrestricted, unrounded constant families
on the retained CV months; they do not bound 2026 gains or nonlinear/clipped
corrections. No deployment offset or candidate was created. Broad offset probes
are deprioritized; preserve upload slots for the selected models and scored
blend analysis. The inverse-reference screen's final model/SQL audit is now
complete and passes; its negligible result remains closed without production.


## Stand-reference inverse screen closed on CV, 17:08 Amsterdam

Both176-input main-tree fits are complete. Adding the two inverse-reference
inputs changes rounded-v76 CV292.449363 →292.446844 (**−0.002519**), with
paired-day interval[−0.013419,+0.008099]. Inside the retained airport/t5/monthly
recipe,292.202850 →292.202417 (**−0.000433**), interval[−0.010950,+0.009665].
January changes−0.007117/−0.004698 respectively, while July changes
+0.002079/+0.003831. These are inconclusive, tiny effects,
not evidence of improvement. No production expansion or submission row follows.

The new inputs were used: stand/contrast split counts are8,991/11,670 in January
and9,239/10,839 in July. The earlier public-reference reconstruction pilot's
lower reference error therefore did not translate into meaningful taxi-out CV.
Retain both models and the source pilot; do not repeat them. January/July fit
and prediction receipts record19.57/15.60minutes, and every original and new
held prediction reloads exactly. The final independent all-row model/SQL audit
is still running; its completed status will be recorded separately.

## Daily main production returned; selected candidate takes priority, 16:58 Amsterdam

The laptop has finished its selected184-input main LightGBM and all344,841
ranking predictions. The full-year fit took17.66minutes; every prediction
reloads exactly on the laptop. Collector935144 finished at14:57:19UTC after
verifying the model, prediction, input-audit and receipt hashes. Independent
return checks also confirm all template IDs and finite nonnegative values.

Model SHA-256:
`2977801222c1d77ceeeff579f6c923a61269e13805577d0999a7cfb346c51d94`.
Prediction SHA-256:
`3a78718d460b94fa66de588cfcff1f25aee7138bef4f1827e278bc88e50c1b44`.
The model is141,771,527bytes and the prediction table3,620,658bytes. The full
original-source desktop inference audit remains required; the return check is
not a substitute for it. No second training fit is needed.

Only idle desktop waiters were reordered, after checking every process handle,
empty child list and frozen source hash. The running inverse-reference test
943878 is untouched. The new chain is daily-main finish971695 → v77
packaging971696 → early-public neural971697 → Gaussian971698 → geometry
preparation971699. Selected production now completes before the optional neural
experiments. Previous queue files and idle process starts are retained in
`target215/selected_production_priority_1658/`. No active fit was stopped and
upload watchers649782/792755 remain unchanged. The laptop has no remaining
training job; both authorized production fits are retained on the desktop.

## Daily-candidate score geometry prepared for the next window, 16:51 Amsterdam

`daily_models_geometry.py` will append the selected v77 vector to the retained,
audited 50-column ranking and Jan/Jul basis. It reuses every existing vector;
no model refit or reconstructed leaderboard score is introduced. Preparation
requires the completed package audit and immutable v77 file/CV hashes. Its
all-row SQL audit also checks every old column against the saved original basis.

Evaluation reuses the frozen solver, expanded-MSE identity, official-score
rounding interval and exact output-serialization bound. It requires successful
**actual v75, v76 and v77 scores**, exact remote file hashes and a fresh check of
all scored rows against the public leaderboard. Weights retain the same fixed
[−3, 4] bounds, sum-to-one constraint and ridge1. Jan/Jul CV is reported as reused
development evidence, separately from this explicit leaderboard-based selection.
The result stays a diagnosis; a chosen blend would need its own final file audit.
No score is projected or imputed before those official results exist.

The synthetic hidden-target/rounding/expanded-identity checks pass, and the
parent audit/evaluation functions are reused with identical bytecode. Queue965724
prepares and audits the 51-column matrices after Gaussian959021 releases the
desktop. Its required result is the selected v77 package audit, so an unrelated
Gaussian failure does not invalidate the selected candidate. No upload is queued
by this diagnostic. Existing v75/v76 upload and evaluation watchers are unchanged.

The inverse-reference January fit is complete, with 10,000 rounds and exact
original/new held prediction reloads on153,706 rows. Its elapsed fit/inference
receipt is19.57minutes; July and the combined score/audit are still outstanding.
The laptop's daily main fit remains complete at17.66minutes, with ranking checks
still running; do not infer a stall or refit from the absence of a final receipt.

## Fixed observation-variance screen prepared, 16:43 Amsterdam

The public physical network uses one squared-error objective for both reliable
clock observations and ambiguous recording cases. A new fixed screen learns a
per-row observation variance while retaining exactly the same mean architecture,
168 inputs, training population and clipped target. The loss is twice Gaussian
negative log likelihood, `exp(-s) * (mean - target)^2 + s`, with standardized
log variance bounded to [−6, 4]. Its new linear head sees the existing hidden,
track, quality and public inputs and starts at zero. At initialization the loss
and original mean gradients are exactly those of MSE. Inference still uses only
the predicted mean. This is a hypothesis about training behavior, not a claimed
improvement or a change to the leaderboard metric.

`nn_public_gaussian.py` and `nn_public_gaussian_jan_jul.py` preserve the existing
seed42, 21/17 epochs, batches, AdamW and OneCycle schedule. An AST comparison
permits only the loss statement and its logged metric name to differ from the
frozen trainer. CPU checks pass for original weights, nonzero mean paths,
initial outputs/loss/gradients, initialization/dropout randomness, padding,
variance-only changes leaving mean inference unchanged, and independent
analytic gradients at nonzero variance. CUDA checks run before any fit. Each
completed fit saves a recovery checkpoint before diagnostics; every held mean
and variance reloads, with separate sampled input/model and all-row SQL audits.
Selection uses matched Jan/Jul RMSE and the one-of-three replacement in v76.
There is no variance/weight/stopping scan and no selected production fit.

Identity SHA-256:
`5ba0cd80630bf8f81ee4cd085444bf59d2eaa83e7ac942580df8a979f017dfd8`.
Queue959021 waits behind candidate packaging959020, which follows the existing
daily-main finish943880. The earlier idle Gaussian waiter957091 was replaced
before any child or fit so selected production is packaged first. All five
current queue source bindings pass; one heavy job per host remains enforced.

`daily_refinements_candidate.py` packages the selected combined production file
without fitting or recalibration. Its pending v77 build preserves the component
bytes, copies Jan/Jul predictions, binds immutable sources and the v76 snapshot,
and independently checks all344,841 ranking values and344,419 CV rows in SQL.
No v77 file, submission row or upload is claimed before those stages complete.
V75/V76 upload watchers remain unchanged.

The laptop daily-main model checkpoint is saved after17.66minutes onGLAPTOP;
SHA-256 `2977801222c1d77ceeeff579f6c923a61269e13805577d0999a7cfb346c51d94`.
Ranking inference/reload checks are still running, so the collector has not yet
accepted a complete return. Live16:41Amsterdam still gives v74 at221.6881,
rank12/242 on344,841 rows, leaving6.6881seconds to the active target.

## Rome daily features closed; returned CatBoost fully audited, 16:28 Amsterdam

All six Rome daily-taxi classifier fits and audits are complete. V76 CV changes
292.449363 → 292.568401 (**+0.119038**), with January +0.068160 and July
+0.169915. The paired-day interval is [−0.011181, +0.248215]. In the selected
daily-quality recipe, CV changes 291.666009 → 291.789902 (**+0.123893**),
interval [−0.010112, +0.259859]. Both months worsen. All original 80 and new
10 inputs, six saved-model probability reloads, non-Rome controls and
344,419-row independent SQL checks pass. Retain the models and close this
direction without a production fit or submission entry.

The laptop-trained full174 CatBoost now passes the complete desktop production
audit: all 344,841 ranking predictions reload exactly and the original 134
input values/categories agree. The combined CatBoost/airport/t5/monthly/
daily-quality file is built and audited too, with every parent stage replayed
exactly and all-row SQL tree means, gates, rounding and CV checks passing.
Its CV is 291.589478; candidate SHA-256:
`469a5606aab314e3811c5a424438617ce85b84f9b5b85cff70543c6c2cb7bfec`.
It remains an unversioned component pending the selected daily main tree.

Desktop queue 943878 now runs the stand-reference inverse main-tree test.
Laptop worker 5443 continues the full-year daily main fit; collector 935144
waits for its verified result. Each host runs one heavy job. The live check
at 16:22 Amsterdam still gives v74 at 221.6881, rank 12/242; v75/v76 have no
official results yet. No goal completion or leaderboard transfer is claimed.

## Daily neural direction closed; production returns, 16:20 Amsterdam

Adding the ten daily-taxi inputs to the public neural model does not improve
CV. Matched seed42 changes292.834296→293.071374 (**+0.237078**), with
January+0.062326/July+0.411830. Replacing only seed42 in the selected three-member
pool changes292.449363→292.524454 (**+0.075091**), worsening both months.
The respective paired-day intervals[−0.004097,+0.476950] and
[−0.002799,+0.162001] include zero. Both full held sets reload exactly; the
independent sampled model/input and all344,419-row SQL audits pass. Retain the
completed models and close expansion; these are not production candidates.

The selected airport/t5/monthly/daily-quality production file is now prepared
and independently audited. It changes323,988 rows versus v76, with prediction
RMS22.3961seconds, and excludes54,792 rows from the daily correction through
the existing quality rule. Jan/Jul CV remains291.666009. File SHA-256:
`1d974a195bdc548adb61ce625eef1bddda243ccb726cae067b428141dc86ffae`.
It remains an unversioned component pending the selected tree replacements.

The first production audit stopped at its tree-mean check. Its SQL evaluated
FLOAT main+FLOAT airport before adding DOUBLE CatBoost, introducing a maximum
0.000325521-second rounding difference. The production matrix already uses
double throughout. `tools/audit_public_daily_taxi_production.py` casts every
term before addition and gets **exactly zero** discrepancy on all344,841 rows;
an independent NumPy mean agrees too. All source-quality, parent-control,
daily-shift, bound, schema and344,419-row CV checks pass. The original runner,
manifest, calibration and every prediction remain unchanged. Error and queue
snapshots are preserved in `audit_precision_recovery/`; no preparation or fit
was repeated. The original failed audit is not relabeled as a successful run.

Only idle dependent watchers and the CatBoost return collector were replaced.
Current desktop chain: Rome943857 →Cat inference943861 →combined943877
→inverse943878 →early-public943879 →daily-main finish943880. The Rome
zero-change replay passes all three original scoring stages; its six fits
are now running. Existing upload watchers and laptop training were untouched.

Laptop CatBoost finished10,000 rounds on2,051,327 rows in28.07minutes. Its
328,530,384-byte checkpoint returned and passed model/receipt/input-audit hashes
at14:19:41UTC; model SHA-256
`ed0dacdfb76cf362c0830a7bb9bfb29d04fa6c6c6eb108c9899faf61ea668294`.
Collector943887 resumed desktop waiter943861 only after verification; it will
reuse the model for ranking predictions after Rome. Laptop queue2979 has
sequentially started daily-main worker5443 (184inputs,10,000rounds,four threads).
No official score changed during this work.

## Daily main tree survives the selected combination, 16:08 Amsterdam

The main-tree screen's independent audit passes: all344,419 original and new
held predictions replay exactly, and SQL verifies truth, calendars, controls
and scores. The ten new inputs improve rounded-v76 CV by0.085240 seconds.

`daily_main_refinements_cv.py` then reuses all fits and calibrations in four
fixed combinations. It reconstructs the two serialized boundaries, recomputes
the unchanged daily source correction, and applies the selected quality gate.

| Fixed combination | Jan/Jul CV |
| --- | ---: |
| Airport/t5/monthly/daily-quality parent | 291.666009 |
| Parent with daily main tree | 291.624416 |
| Parent with full174 CatBoost | 291.589478 |
| Parent with both tree changes | **291.549710** |

The incremental daily-main change on the selected CatBoost parent is
**−0.039768**, paired-day interval[−0.081566,−0.012448]; January−0.026548,
July−0.052989. The effect before the daily-quality correction is−0.073526,
so adding standalone improvements would overstate the retained gain.
All parent intermediate predictions, shifts and final values replay exactly;
all-row SQL and separate shift/gate/bound checks pass. This is a small selected
component on reused development months, with no separate submission entry or
claim of leaderboard transfer. Its combined CV is0.899653 below rounded v76.

`daily_main_submit.py` prepares the fixed184-input LB1/10,000/seed42 production
fit. The desktop's complete source guard passes, including exact equality of
the cached old ranking matrix/IDs against the original ranking builder. The
laptop verified every portable input/source hash; only28,176,015 further bytes
needed transfer. Queue2979 waits for the live CatBoost supervisor1600 before
starting the LightGBM, preserving one heavy job per host.

Collector935144 (`remote_component_return.py`) verifies the completed remote
queue, model, predictions and input-audit receipt. Tests cover a valid return,
corrupted model transfer, a missing live process without completion, and later
prediction corruption; all four pass. Desktop queue935145 follows early-public
queue917136, waits for the verified return, then runs the original-source
all-row prediction audit and `daily_main_refinements_submit.py prepare/audit`.
The final component builder reuses the same full2025 calibration, checks every
parent stage and preserves the submission schema. Those production stages are
queued; no model or combined file is claimed complete yet. V75/V76 upload
watchers remain unchanged. Live score check at15:51Amsterdam remains v74,
221.6881, rank12/242 on344,841 flights.

## Laptop training restored, 15:50 Amsterdam

The user reopened laptop WSL and authorized distributing the current work.
SSH succeeds, the checkout fast-forwarded to main5b7ac28, and the5060 GPU was
idle. Its original2,552,204,749-byte input cache has the same SHA-256 as the
desktop. Only62,464,715 bytes needed transfer; no raw data was downloaded.

`catboost_full_inputs_portable.py` exports a manifest after the complete original
production guard, then verifies154 files on the laptop. It reuses the unchanged
2,051,327 ordinary training rows, all uncapped labels, original134 categorical
conversions, all156 original inputs, and the same18 appended inputs and K07
10,000-round seed42 settings. The all-row input audit passes. Training started
at13:48UTC onGLAPTOP, supervisor1600/worker1601, with about6.4GB GPU memory;
no score or completed fit is claimed yet.

The unstarted desktop waiter897098 was stopped before the remote launch,
with its process start and predecessor retained in `portable_worker/migration.json`.
No active fit was interrupted. `catboost_portable_dispatch.py collect` (925705)
waits for a completed remote receipt, copies back only model/receipt/audit/log,
checks their hashes and identity, then resumes that same waiter. The frozen
desktop fit runner will reuse the model and perform ranking predictions and
the original full-row audit. Downstream scheduling and upload watchers remain
intact; one heavy job runs per host.

The daily-taxi main tree has now completed both months: rounded-v76 CV
292.449363→292.364123 (−0.085240), with month changes−0.065708/−0.104773.
Inside airport/t5/full-monthly refinements the change is−0.076536. These are
small gains on reused development data; the independent all-row model/SQL
audit is still running. Combination with the selected daily-quality correction
and new CatBoost must be checked before expanding production.

## Early access to public neural inputs, 15:30 Amsterdam

The selected public neural model supplies its twelve daily departure inputs
only to a32-unit residual head after the main tabular network. A fixed new
architecture also projects those same twelve transformed values into the first
512-unit hidden layer, before its activation. This adds6,144 weights, all
initialized to zero; the existing public head, physical track branch, raw
inputs, transforms, population, loss, optimizer, schedule and21/17 stopping
epochs remain unchanged. There is no additional dataset, weight sweep or
continuation. The hypothesis is that the public information can interact with
ordinary inputs throughout the hidden layers.

`nn_public_early.py` passes CPU checks: every original parameter, initial
prediction and gradient matches exactly in training and evaluation modes;
initialization/dropout random state matches; existing nonzero public/track
heads are preserved; the new weights receive finite nonzero gradients and
change predictions when activated; padded track values have no effect. CUDA
checks run before fitting. The training update reuses the original bytecode.
Before each new fit, the complete saved late-head control must replay on the
unchanged held inputs. Each new checkpoint saves before diagnostics and must
replay every held prediction. A separate model/input audit and all-row SQL
score audit follow.

The primary comparison is matched seed42 versus seed42 inside rounded v76.
The secondary comparison replaces only seed42 in its three-member public
physical pool, retaining1042/2042. Both are reused Jan/Jul development checks.
Runner`nn_public_early_jan_jul.py`, run`v3_20261011_nn_public_early/`.
Queue917136 follows reference-inverse queue909597; one heavy desktop job
continues at a time. No fit, CV gain, production or upload is claimed yet.
Its first idle watcher914722 was stopped before any fit to move the partial
checkpoint guard ahead of training. An injected partial checkpoint now rejects
before model construction; the former source/identity/queue are retained in
`pre_fit_recovery_guard/`. No model or active training was repeated.

This follows an inventory check of the older learned-clock proposal. Its core
mechanism is already covered by the completed clock-bypass experiment, which
improved its matched laptop single but did not establish a gain over the
selected pooled model. That check is not repeated or reclassified as a new
candidate. The current daily-taxi main tree has completed January's10,000
rounds; all153,706 original and new held predictions reload exactly. July and
the final two-month score remain pending; no early single-month selection.

## Full-input CatBoost passes narrowly; stand-reference inverse pilot, 15:18 Amsterdam

The ordinary CatBoost screen is complete. Its fixed134→174 input expansion
adds the22 retained combo,6 callsign and12 public departure inputs, keeping
K07/10,000 rounds, seed42, original populations and uncapped labels. The
matched direct CV changes291.7234→291.5770 (−0.1464). Transfer to v76 changes
292.4494→292.3274 (−0.1220, paired-day interval[−0.2170,−0.0328]); both months
improve. Both models save/reload all344,419 held predictions exactly, and the
independent SQL score audit passes. Historical control CatBoost models used
the laptop GPU, while this expansion used the desktop; retain that caveat.
Run`v3_20261011_catboost_full_inputs/`.

The actual selected daily-quality combination is separately replayed, including
its direct and rounded controls and recalculated fixed daily shifts. It changes
**291.6660→291.5895 (−0.0765)**, interval[−0.1531,−0.0035]. January changes
only−0.0076 and July−0.1455. The interval barely excludes zero and all evidence
reuses development data. Retain this as a small component of the combined
recipe, not a distinct submission or an independent confirmation. The cumulative
CV difference versus rounded v76 is−0.8599. Run
`v3_20261011_catboost_refinements_cv/`; all-row source-shift, constraint and SQL
audits pass. One full-year174-input CatBoost is queued under897098 after Rome's
daily-taxi screen. Queue907102 then assembles/audits its combined component,
reusing the daily calibration without fitting it again. No version or upload.

Uncertainty-weighted daily allocation **fails**. One fixed screen allocates the
same daily shift in proportion to retained seven-member prediction disagreement,
normalizing each cell's average individual variance to200² and retaining15²
common variance. All cell totals are preserved before bounds and rounding.
CV291.6660→292.3252 (**+0.6591**), interval[+0.1688,+1.2994]; January+1.2617,
July+0.0566. Independent SQL member variance, cell conservation, controls and
scores pass. Run`v3_20261011_daily_taxi_uncertainty/`. Close this direction;
no production and no new submission row.

A new public-only pilot reconstructs stand/runway reference contrasts from
changes in daily stand composition. The [official methodology](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf)
defines stand/runway reference times with monthly rolling-year recalculation.
The published dashboard supplies runway/day reference means. A fixed ridge
model combines stand fractions with separate month intercepts; it does not
access departure labels. Held public calendar days divisible by4 give source
RMSE **13.9872→10.3799 seconds in2025** and **13.3828→10.0593 in2026**,
across264/296 source cells. This is reconstruction of public totals, **not a
taxi-out CV score or leaderboard gain**. Source cohorts differ and individual
stand coefficients are underidentified; month intercepts cannot capture all
stand-specific changes.

`tools/probe_public_reference_inverse.py` and its separate auditor retain55
groups in`v3_20261011_public_reference_inverse_pilot/`. Independent raw metadata,
UTC/local calendars, stand counts, public targets and scores pass; all full
coefficient reconstructions agree within3.5e−13. The first auditor attempt
used a nonexistent abbreviated training filename; correcting that path was
the only fix and the completed pilot was not repeated.

`public_reference_inverse_features.py` extends the same fixed construction to
all2025 inputs using month pairs six months apart. Jan/Jul2025 and2026 reuse
the completed pilot coefficients. Two inputs are estimated stand reference and
its contrast against that month's published runway mean. Missing values stay
missing:232,291 of2,085,047 training rows and25,560 of344,841 ranking rows.
An independent all-row audit passes4,859,776 feature entries and8,305 public
source-cell reconstructions, including metadata, local calendar, missing joins
and row permutation. A fixed174→176-input main-LightGBM screen is queued under
909597 after component builder907102; it must preserve/replay every
original input and model before fitting, with originalLB1/10,000/seed42 settings.
No taxi CV result or selection exists for these inputs yet.

All ten public airport production models are complete; all344,841 predictions
reload exactly and the separate production audit passes. The daily-taxi main
LightGBM is now the only heavy desktop job. The neural and Rome daily-taxi
screens follow in sequence. V75/V76 upload watchers remain unchanged.
Live check at15:18:04 Amsterdam still reports v74 best/latest221.6881,
rank12/242,344,841 scored flights. Queue/source/file integrity passes at15:21,
with exactly one heavy desktop job. The goal remains active.

## Rome daily-taxi inputs queued; count/sum inference remains exploratory, 14:42 Amsterdam

`rome_daily_taxi_jan_jul.py` adds the ten audited runway/day and airport/day
taxi inputs to the selected Rome CatBoost pool: 80→90 columns, seeds42/142/242,
the original1182/1401 rounds, identical126,507/86,723 training rows and
11,317/15,211 Rome held rows. Preparation checks original features, targets,
IDs, masks and permitted months exactly. The equal Cat3/original-R20 mixture
and all other models remain fixed. Primary scoring uses production-rounded
v76; secondary scoring substitutes into the selected refinements and recomputes
their unchanged daily-quality correction. All three controls must reproduce.
Before fitting, a zero-change replay exercises this complete scoring path.
Every old classifier must replay before its replacement fits; new models save
before diagnostics, then all held probabilities reload. A separate model,
input and SQL score audit follows. Queue877491 runs after production-component
queue870127, preserving one heavy desktop job. No production fit or upload is
selected from this new test. Run`v3_20261011_rome_daily_taxi/`.

A separate source diagnostic found Rome's public count typically close to the
challenge's stand-present/0–2h cohort: median coverage0.9935, median omission
2flights, residual MAD2.91flights across608 runway-days. The count residual
RMSE is19.72flights, however, and four source counts exceed the comparable
challenge count. These are approximate measurements, not exact subset bounds.
The published taxi model has date/airport/phase/runway dimensions; its schema
does not expose stand, hour or individual-flight rows. No further source
download ran. Diagnostic cache:`tmp/research_20261011/daily_taxi_count/`.

Two fixed, inexpensive posterior screens tested that underexplored count signal.
They replace uniform daily shifts only for ambiguous Rome recording modes,
starting from the refined prediction before its daily correction. Public counts
and sums are truncated at the source's2h cutoff. A training-only Huber fit
estimates missing counts/sums and their covariance on complete local days;
the source likelihood is variance-matched bivariate t5. The first version uses
Gaussian ordinary/copy modes; the one follow-up reuses the retained training
quantiles, copy point and5% broad support with a mean-preserving tilt. It does
not refit source calibration or select a new spread. Both preserve the selected
quality-corrected predictions outside the ambiguous Rome scope and reapply
the original final bounds and rounding.

| Screen | Jan/Jul CV | Change vs selected291.6660 | January / July change |
| --- | ---: | ---: | ---: |
| Count and sum, narrow modes | 291.3708 | −0.2952, inconclusive | −0.1324 / −0.4580 |
| Retained broad supports | 291.2781 | −0.3879, inconclusive | −0.1502 / −0.6257 |

Paired-day intervals are[−1.2413,+0.6391] and[−1.2947,+0.5050], respectively.
These repeatedly reused development results are **not selected**. The initial
screen changes2,416 rows: July copied-clock cases save168.2million SSE, while
ordinary cases lose142.5million SSE. A few large errors dominate. The broader
follow-up addresses the narrow ordinary assumption but does not establish a
reliable gain. Do not add either to`SUBMISSIONS.md`, retune gates on their
winning/losing dates, or claim the two screens independently confirm each other.

Runs`v3_20261011_taxi_count_sum/` and`v3_20261011_taxi_count_sum_support/` retain
all parameters, predictions and source-cell diagnostics. Both audits pass:
independent normal-moment quadrature, Student-density/discrete-state checks,
permutation checks, exact calibration reuse, all-row SQL labels/controls/scores,
permitted-month calibration counts/sums and final-constraint replay. No heavy
taxi model was fitted for these screens. The desktop's174-input CatBoost has
completed January and is training July. Laptop WSL remains offline; a separate
SSH attempt to its online Windows host also timed out, so no remote job ran.

## Source-quality follow-up passes both months, 14:16 Amsterdam

The daily-mean failure diagnosis locates January regressions mainly at EDDF,
EDDM and LSZH. The original correction shifts their eligible means downward
more than their actual prediction bias. This is consistent with source-cohort
mismatch, including the documented exclusion of deicing after off-block; it
is not proof that deicing explains each flight. The initial correction's
January LFPG aggregate still improves despite high invalid share.

A single follow-up, `public_daily_taxi_quality.py`, reuses the **existing monthly
invalid-share<5% rule** for daily corrections. It preserves every coefficient,
noise estimate, original shift and the120-second cap; missing quality leaves
the parent unchanged. No new threshold or airport choice. It was declared
after inspecting the original winter losses, so this remains reused-development
selection, not independent confirmation.

On production-rounded v76 the result is **291.7905 (−0.6589)**,
interval[−0.9015,−0.4491]. On the selected airport/t5/full-monthly refinement,
**292.2029→291.6660 (−0.5368)**, interval[−0.7420,−0.3577].
**January improves0.0825 and July0.9912** on the refined parent. Relative to
its unrestricted daily correction the refinement improves0.1796,
interval[−0.3357,−0.0054]; the same gate difference on v76 is inconclusive.
The cumulative change from production-rounded v76 is−0.7834. Independent SQL
checks all344,419 raw-source quality shares, gate decisions, parent/daily
predictions, dates, final choices and scores. Keep the original result too.

The diagnostic under`v3_20261011_public_daily_taxi/diagnosis/` also computes
perfect held-label runway-cell means: CV290.8270 before reapplying plan bounds.
This is deliberately label-informed headroom diagnosis, **not a usable model,
validation gain or rigorous bound after postprocessing**. It is not selected.

The passing quality-gated correction is selected for preparation with the
previously passing combined refinement. `public_daily_taxi_submit.py` assembles
the saved public main/physical/Rome parts, replaces only the airport trees,
reuses the audited t5/full-monthly ranking stage and refits only the small
source-mismatch calibration on full2025. It must reproduce v76's direct and
serialized outputs and the aggregate control before making changes. A separate
SQL/tree/source-gate/CV/schema audit follows, with exact daily-shift and final
constraint replay. The prepared template-compatible file has **no version and
no upload** until reviewed against the other pending candidates. Queue870127
runs after the neural screen863590; it requires airport production's audit too.
This preserves the sequential desktop queue. The full-year taxi models are
never repeated by this builder.

## Daily-taxi neural test queued; broader source check closed, 14:08 Amsterdam

`nn_public_daily_taxi.py` adds the same ten raw daily-taxi inputs to the
selected public physical network. It widens the existing public head's first
layer268→278, preserving every old weight/bias and initializing only the ten
new input columns to zero. Original tabular/track/public initialization,
initial outputs and initialization/dropout RNG match on CPU. The old public
head still works and new columns receive finite nonzero gradients after the
output head becomes active. CUDA checks run at the start of its queued fit.
The old twelve features and their transformed values reproduce exactly for
all2,085,047 training rows; source feature SQL audit remains the prerequisite.

The fixed screen reuses original training code, seed42 and21/17epochs. Primary
comparison is old/new seed42 inside the production-rounded public stack.
A second fixed comparison replaces only seed42 of the selected three-member
pool and preserves1042/2042, so both comparisons have equal member counts.
The actual selected control must reproduce exactly. Each completed model is
saved before diagnostics, then every held prediction reloads exactly; a separate
raw-track/encoder sample audit and all-row SQL score audit follow. No weight,
architecture-width, stopping or seed search. No full-year fit is selected yet.
Run`v3_20261011_nn_public_daily_taxi/`, runner
`nn_public_daily_taxi_jan_jul.py`; queue863590 follows daily main-tree859814.
Do not edit the frozen source/identity or start it alongside the active CatBoost.

A read-only check of the [CODA all-causes dashboard](https://ansperformance.eu/capacity/tot_dly/)
found monthly network-wide summaries, not daily airport/operator aggregates.
Its public metadata is cached at`tmp/research_20261011/coda_public_source/`;
no data queries or model experiments followed. The public taxi methodology
again confirms local takeoff dates, the2h exclusion, after-off-block deicing
exclusion and stand/runway reference eligibility. No exact cohort identity is
inferred from that documentation. Laptop SSH still times out; no remote job ran.

## Daily runway source, production parity and completed averaging, 14:00 Amsterdam

A new public source provides **daily taxi-out totals by runway** for all ten
airports. The [EUROCONTROL taxi-out dashboard](https://ansperformance.eu/efficiency/taxiout/)
embeds an anonymously published Power BI report. Its ordinary public query API
exposes date, airport, runway, flight count, total taxi minutes, reference minutes
and additional minutes. The local cache contains16,169 distinct cells from
January2025 through July2026. An independent, uncompressed query grouped only
by airport reproduces every count exactly and all three minute totals to within
1.1e-8. This is source extraction evidence, not competition-cohort identity.

The source population is different. A24-view diagnostic of all2,085,047 training
departures finds local takeoff dates match better than UTC or off-block dates.
Among10,111 common2025 runway-day cells, the basic stand-present/0–2h cohort
matches the source count in794 cells and both count/sum in only21. Exclusions
and reference eligibility remain unobserved. Do not treat the public mean as an
exact competition-label constraint. The [published terms](https://ansperformance.eu/about/disclaimer/)
require attribution and limit reuse; no general open-data license is claimed.
Raw responses are retained unmodified and not committed. Fetchers:
`tools/bootstrap_public_daily_taxi_source.py` and `tools/fetch_public_daily_taxi.py`.
Do not redownload the completed cache. Extraction and population audits are in
`research_network_20261010/public_daily_taxi_{source_audit,population}.json`.

The fixed first correction, `public_daily_taxi.py`, fits airport intercepts and
count-deficit coefficients using only the outer fold's permitted training labels.
It uses Huber20, a training residual noise estimate and existing daily prior
scales to shrink the mean correction. Source/competition counts must be within
0.5–1.2, both at least20; the inherited120-second raw cap stays fixed. Only rows
with stand/runway and incoming predictions up to2h receive the correction.
The source calendar uses local takeoff days; partial training days are excluded.
All plan bounds and production rounding are reapplied.

The correction improves production-rounded v76 CV292.4494→291.9389
(−0.5105, interval[−0.8762,−0.2112]). On the selected airport/t5/full-monthly
refinement it changes292.2029→291.8456 (−0.3572,
interval[−0.6706,−0.0882]). **January worsens0.2768; July improves0.9912**
on that refined parent. Retain the result without claiming a winter solution.
Independent SQL checks every held label/date/parent/score, all10,776 calibration
cells and training membership; a separate scalar implementation verifies all
4,010 inference-group updates. No production file or new submission row exists.
Run: `v3_20261011_public_daily_taxi/`.

The ten source inputs are also prepared for a conditional model: runway/day
and airport/day taxi, reference and additional means/counts, runway count share
and runway-minus-airport mean. Missing source cells remain missing. All ten
columns for2,085,047 training and344,841 ranking rows pass independent SQL
arithmetic/calendar checks (24,298,880 values);2,000 permuted rows per view
reconstruct exactly. No departure label is read by the feature builder.
`public_daily_taxi_features.py` retains both caches and its manifest/audit.
A fixed main-LightGBM comparison adds these ten inputs to the existing174,
retaining LB1/10000/seed42 and original rows. Both original and new entire held
predictions reload; actual production-rounded v76 and refined controls must
reproduce exactly. Queue859814 waits for airport production847448, then runs
preflight→fit→score→audit. Source and identity are frozen; no fit or score yet.

Production parity for the earlier joint refinement is now explicit:
`public_refinements_submission_cv.py` rounds both v74 and final boundaries in
both arms, giving292.4493629→292.2028500 (−0.2465129),
interval[−0.3874474,−0.1492070]. January/July both improve. The original
unrounded comparison remains retained separately. Independent SQL verifies
all344,419 labels, controls, serialization boundaries and scores.
`public_refinements_aggregate_submit.py` prepares/audits the t5/full-monthly
production component on all344,841 rows, exactly reproducing v72/v73/v74/v76
controls and the existing affine weight. It changes292,259 serialized rows
with RMS12.5961 seconds; this is prediction change, not measured score gain.
Airport production substitution and final candidate assembly still remain.

The parameter-average experiment initially finished January's training but
failed before saving because a diagnostic subtracted Boolean PLE buffers.
The failed trajectory, original source, identities and queue states are archived
under `v3_20261011_nn_weight_average/failed_attempt1/`; no recoverable checkpoint
existed. The fix compares nonfloating buffers separately and atomically saves
last weights plus shadow averaged parameters before diagnostics. Tests with real
January/July Boolean buffers and a synthetic optimizer/recovery trajectory pass.
Only that failed fit was repeated. Both new checkpoints now reload completely,
the separate model audit and all344,419 SQL scores pass.

Averaging is closed: same-trajectory direct+0.5492,
interval[+0.2584,+0.8360]; transferred+0.5437,
interval[+0.2463,+0.8308]. It does not improve the selected model. Historical
control differences are0.1027 transferred, inconclusive, so historical runs were
not substituted for the actual same-trajectory control. All models remain saved.
The recovered EMA queue847446 is complete. CatBoost queue847447 now runs,
then airport production847448, then daily-taxi main-tree859814. All remain
sequential. V75/V76 upload watchers649782/792755 and geometry queue826430 are
unchanged. The laptop's WSL remains unavailable and the three research agents
have exhausted their account allowance. No new agent jobs were created.

## Airport gain, failed callsign expansion and combined refinements, 13:25 Amsterdam

Public inputs in the airport LightGBMs pass the fixed screen: direct
CV291.7234→291.5826 (−0.1408), interval[−0.2095,−0.0970]. In the selected
v76 recipe, CV292.4494→292.3459 (−0.1035), interval[−0.1573,−0.0660].
January improves0.0920 and July0.1151. All twenty saved models reload every
held prediction; independent SQL verifies344,419 labels, dates, controls and
scores. The existing settings and original156 inputs were preserved, with
only the twelve public inputs added. This small gain is retained as a component,
without a separate submission entry.

The public callsign three-member pool does **not** confirm the single-seed
result. The matched direct change is−0.1016, interval[−0.2888,+0.1088], with
January slightly worse. Its transferred effect is+0.0770,
interval[−0.0882,+0.2966], with both months worse. Comparison against the
selected original fifteen-member pool is+0.3149, but has unequal pool sizes.
All old/new predictions reload exactly and the SQL audit passes. Keep all six
fits; do not expand to fifteen or prepare a production version.

A single fixed combination of public airport trees, the previously retained
t5 daily posterior and full monthly multiplier improves v76
CV292.4494→292.2032 (−0.2462), interval[−0.3874,−0.1484]. January improves
0.2274 and July0.2650. It also improves over airport-only by0.1427 and over
the two aggregate changes alone by0.0896; both intervals exclude zero.
`public_refinements_cv.py` exactly reproduces all four individual controls,
retains the original stage ordering and plan constraints, and passes the
independent344,419-row SQL arithmetic/score audit. This combination was chosen
after observing its components on the same reused development months; the
interval is not independent confirmation. No version or upload is reserved.

`public_airport_submit.py` now prepares the ten full-year airport models for
that combined candidate. Its guard binds the passing individual/joint results,
all twenty CV recipes, the original ordinary membership and the168-column
feature order. It retains original labels and P04/5000/seed42 settings. Every
ranking prediction must reload twice, with a separate full-row audit after
fitting. QueuePID840168 waits for the CatBoost CV audit, preserving one heavy
job per desktop. No production fit has started yet.

The ordinary CatBoost was found to have only134 inputs: K07 never received the
twenty-two combined inputs, six callsign clocks or twelve public statistics.
`catboost_full_inputs.py` supplies the same174 inputs used by the main model,
keeping K07/10000/seed42, original rows and uncapped labels. It verifies the
old134 values and categorical representations exactly. The current computed
winner is K07/c10000; an older winners snapshot has a stale null checkpoint.
Both input preflights pass. Historical K07 controls ran on the laptop GPU,
so hardware/nondeterminism remains a comparison caveat. QueuePID834461 waits
for the fixed parameter-average screenPID824858, which is currently training.
CatBoost has no new fit result yet.

The previously scored affine basis already projects only221.5321; its CV
interval includes no gain. A descriptive v73/v74 two-point calculation also
offers only about0.096 seconds before serialization and constraints. That
direction includes both monthly and plan changes, so it is not a measurement
of the monthly multiplier alone. No repeat upload or new candidate follows
from either diagnosis. New50-vector evaluation still waits for actual v75/v76
scores. Both upload watchers remain unchanged. Laptop WSL is still offline;
the three research agents are unavailable because their account allowance is
exhausted. Their completed research is retained; root continues the local queue.

## Width result, callsign result and next fixed screens, 12:57 Amsterdam

The width64 physical encoder does **not** improve the recipe: matched single
CV292.4340→292.4192 (−0.0148), interval[−0.2605,+0.2773]. January worsens0.1041
and July improves0.1337. The transferred change is+0.0140, also inconclusive.
A separate comparison with the existing three-member pool regresses0.4000,
but that comparison has unequal pool sizes. Both checkpoints are retained;
1,094 sampled model reloads pass (maximum0.0040s), and independent SQL checks
all344,419 truths, calendars and complete scores. No width expansion or upload.

The public-input callsign single has a useful primary signal: matched
CV292.6074→292.2710 (−0.3364), interval[−0.6174,−0.0581]. January gains0.1775
and July0.4952. In the transferred stack it gains0.1588, interval
[−0.4478,+0.1336], so the selected-stack effect remains inconclusive. The new
single still trails the selected original callsign15 pool by0.6984; that is a
pool-size comparison, not evidence against the inputs. Every old/new held
prediction reloads exactly, original162-column preprocessing matches, and all
344,419 SQL truths, selected controls and scores pass. Retain the completed
`v3_20261011_nn_calls_public/` run without a submission entry.

A fixed matched-three-member check now reuses both seed42 fits and adds only
1042/2042. Original models for all three seeds are already retained. Each
additional control must reload on its entire held month before a new fit;
all new predictions reload too. Inputs, encoders, optimizer,21/17epochs and
blend stay fixed. `nn_calls_public_pool3.py` passed its source/parent/seed guards;
queuePID824857 waits for the active airport-tree queuePID816876. These are
reused development months, not independent confirmation. No production yet.

Parameter averaging is untested in the retained experiment inventory. The
new fixed decay0.999 screen keeps the original public physical model and
optimizer trajectory, while collecting an exponential average of its parameters
after each update. It saves both the last iterate and averaged result. Their
same-trajectory comparison is primary; the previous seed42 checkpoint and
three-member pool are separate references. There is no new stopping, blend or
decay search. The average starts at the original initialization and includes
every update; model buffers stay unchanged. A twelve-step CPU oracle verifies
identical original weights, gradients, optimizer moments, dropout RNG and
learning rates, and independently checks the recurrence within1.49e−7.
All held predictions for both final states must reload. Original21/17epochs
and all other settings remain fixed. `nn_weight_average_jan_jul.py` is queued
asPID824858 after callsign3. Its earlier idle watcher was safely rebound before
any fit started; no active training was interrupted.

The50-vector geometry dataset is complete and independently audited across
all344,841 ranking and344,419 CV rows. Evaluation still requires the actual
successful v75/v76 scores; none are imputed. QueuePID826430 waits for the v76
upload watcher to exit, then runs that guarded diagnosis without uploading. Live leaderboard at12:46 Amsterdam
remains v74 best/latest221.6881, rank12/242. V75/v76 upload watchers still wait
for12 October02:00/02:10 Amsterdam. No failed or inconclusive screen enters
the submission ledger.

## Public inputs in the airport models

The ten airport LightGBMs are another component that still lacks the twelve
public daily inputs. A fixed screen adds those inputs to their original 156
columns, retaining all ordinary training rows and labels, P04 settings, seed42
and 5,000 rounds. It changes no airport-specific settings or ensemble weights.
The already audited source features are reused. This is distinct from the
failed per-airport tailoring search: it supplies new information to the
existing models.

`public_airport_trees.py` passed syntax, source, selected-parameter and artifact
checks. Queue PID816876 waits for the public callsign experiment to finish.
The twenty fits run sequentially with four CPU threads. Each saved airport
model must reproduce every held prediction immediately and again in the final
audit; individual completed fits are resumable. The scoring path must exactly
reconstruct v76 before substituting the new airport component. Independent SQL
checks all labels, dates, control predictions and scores. Historical and new
thread counts can differ, so tiny changes are not reliable feature evidence.
No fit, production selection or submission has occurred for this direction.

## Public callsign network and preparation for the next official scores

V76 adds public daily information to the physical networks, main LightGBM and
Rome classifier, but its fifteen callsign networks still have only the original
156 inputs plus six callsign clocks. The next fixed screen supplies the same
twelve public transforms to that remaining neural component. It keeps every
original162 input, the exact saved encoder/bins, original seed42 initialization,
2048-row batches, optimizer,24-epoch schedule and fixed21/17 stop epochs. A
zero-initialized32-wide public head reads the original256-wide hidden state plus
the12 scaled inputs. No new data download, clock feature or stopping search.

`nn_calls_public.py` passes CPU checks for original weights, initial output,
initialization/dropout RNG and finite nonzero new-head gradients. The isolated
training factory reuses the original function bytecode. Before each fit, the
saved original model must reload **every** held-month prediction exactly; the
saved new checkpoint must likewise reproduce every new prediction. Regenerated
preprocessing must equal the original162-column state, including types/order.
Public transforms additionally receive scalar arithmetic checks. The screen
compares one new callsign network against the matching retained seed42 network
inside the selected public physical/main/Rome stack. Callsign15 is a separate,
different-size reference, not the primary control. Independent SQL checks all
344,419 labels, calendar fields, selected controls and complete recipe scores.

No fit has started for this direction. Queue PID811666 runs fit→score→audit after
the current width64 work and the short geometry preparation. Its frozen sources
and identity are in `v3_20261011_nn_calls_public/` and
`target215/calls_public_queue.json`. No production candidate is selected.

`public_models_geometry.py` prepares the retained45-vector scored basis plus
v70/v72/v74/v75/v76, using actual ranking files and matching Jan/Jul vectors.
Mutable upload manifests are captured as immutable snapshots. Preparation and
all-row independent SQL audits are queued after width64 (PID808050); no score is
imputed for v75 or v76. Once both are officially successful, `evaluate` checks
all basis scores live and uses the existing weight bounds[−3,4], sum-one
constraint and ridge1. It independently verifies the expanded squared-error
identity and reports a conservative bound including official score rounding,
zero-clipping and integer serialization. A synthetic hidden-target oracle and
serialization/score-rounding checks already pass. It produces a diagnosis only,
without reserving a version or uploading. The inherited v3–v9 CV reconstruction
caveat is retained; leaderboard-derived weights are not independent validation.

## Completed aggregate tests; wider track encoder started

The exact selected-stack t5 replacement improves v76 CV292.4494→292.3760
(−0.0734), interval[−0.1520,−0.0232]. January improves0.0942 and July0.0527.
The isolated daily component gains0.2632; the fixed ensemble attenuates it.
All original v72/v73/v74/v76 stages replay, and all344,419 SQL labels, calendars,
weights, serialization and scores pass. Retain `v3_20261011_daily_heavy_tail_stack/`;
the small effect does not receive its own candidate or upload.

Public-only airline/month adaptation is inconclusive on v76:−0.0507,
interval[−0.3466,+0.2002]. January improves0.1913 while July worsens0.0900.
Direct row-loop aggregates, source totals/counts, shared airline shifts and all
SQL scores pass; no production expansion. Restoring excluded long Rome training
rows was already tested by the error-mechanisms agent on October10 and failed
(+0.6193); it was reviewed, not repeated.

A new numerical screen interpolates each possible positive delay between
adjacent minute nodes, preserving its expected delay and physical taxi-state
mass. It retains Gaussian ordinary weights, all source/noise parameters, the
actual v71 prior, original v73 blend and all later corrections. Checks include
physical mass/first moments, bounded grid variance, row permutation, exact old
control replay and an independent24,600-state enumeration (max1.14e−10s).
Actual v76 CV changes **+0.0011**, interval[−0.0032,+0.0058]: no useful gain.
The daily component alone regresses0.0156. Full control/SQL audits pass; retain
`v3_20261011_daily_grid_interpolation/` without promotion.

A second fixed screen applies the existing monthly correction fully instead
of at half strength. Training-only coefficients, eligibility rules,120-second
raw-shift cap, previous predictions and later constraints stay fixed. CV
292.4494→292.3556 (−0.0938), interval[−0.1615,−0.0326]; January gains0.0704,
July0.1171. Independent SQL verifies every multiplier/zero-floor operation and
score; original controls replay. Retain `v3_20261011_monthly_strength/` as a small
research effect; no distinct submission. All intervals reuse development data.

The next larger model test doubles the physical track encoder width32→64.
The successful physical transforms and public head stay; the same three temporal
blocks, dilations[1,4,16], mean/max pooling,128-wide fusion hidden layer,
preprocessing, optimizer,2048-row batches and21/17 epochs are retained. Only
the track channel count and corresponding fusion input dimension increase.
Fresh branch initialization is isolated from the original RNG. CPU/CUDA checks
verify identical tabular/public weights, initial outputs and dropout RNG,
missing/single-track gates, masked-padding invariance and nonzero gradients.
A three-step2048-row synthetic GPU check peaks at1.72GB, leaving room for the
original encoded data. This capacity check is not an accuracy result.

`nn_track_wide_jan_jul.py` runs fit→model audit→score→score audit sequentially,
queue PID803581, child803582. January training started at12:22 Amsterdam;
first20 real steps take59.81ms each, allocated peak4.625GiB. Compare one new network against the retained public seed42 on
the same machine; the selected three-member pool is a different-size diagnostic.
No width, epoch or blend sweep is planned, and no production candidate is selected.
V75/v76 remain queued for12 October02:00/02:10 Amsterdam. Latest live check
**11 October12:12 Amsterdam: v74 best/latest221.6881, rank12/242**,344,841 flights.

## V76 prepared and two completed screens, 12:05 Amsterdam

**V76 is built, independently audited and queued for12 October02:10 Amsterdam.**
Its joint public main/neural change has Jan/Jul CV292.4494 versus v75's292.9415,
a0.4921-second gain, interval[−0.6646,−0.3504]. Both months and comparisons with
both individual changes improve. The production pool, main tree, original
controls, final constraints, IDs and schema reproduce across all344,841 rows.
Independent SQL reproduces all344,419 CV truths and scores. File SHA-256
`17d66918de307849c7afbb6522f257d6c2f9eef3e3ffde478a8995ab84d1b425`;
all96 frozen dependencies verified before queueing. Upload watcher PID792755
binds manifest hash `aa99213f914267e41b37fc3bb5e46e4ac411712821524e8918dc3fdd37e1a4ac`.
V75's earlier02:00 queue is unchanged. No official result for either file yet.

Raw-track withholding **fails**: matched single CV292.4340→292.7875
(+0.3534), interval[+0.1124,+0.6478]. Transferred CV292.9740→293.3088
(+0.3348), interval[+0.1050,+0.5994]. Both months worsen. Numerical model
reloads pass1,094 sampled rows (maximum0.001031s); SQL independently checks
all344,419 truths, calendars and scores. Retain both21/17-epoch models in
`v3_20261011_nn_track_dropout/`; no production expansion or submission.

The frequency-weighted monthly reference bridge changes v75 CV
292.9415→292.8897 (−0.0519), interval[−0.0795,−0.0330]; both months improve.
Every reference mass and all344,419 SQL scores pass. This is a small effect
on reused development data, using approximate reference membership. Retain
`v3_20261011_monthly_reference_bridge/`; no distinct candidate or upload.

The neural Rome classifier also fails to improve: selected CV292.4494→
293.3022 (+0.8528), interval[−0.0762,+2.1373]. January improves0.0799 but
July worsens1.7855. All saved probabilities reload, and independent training
membership, train-only encoders, stopping and all344,419 SQL scores pass.
Retain `v3_20261011_rome_neural/`; no production fit or upload. The desktop
continues with public-only airline/month adaptation and exact-stack t5.
Earlier queue statuses below are historical. The subagent network remains
unavailable due to its account usage limit; its prior research is retained.

## Heavy-tail result and exact selected-stack follow-up

The fixed t5 ordinary-aggregate prior improves the matched aligned Gaussian
screen **292.1302→291.5711 (−0.5591)**, interval **[−0.9126,−0.3181]**.
January improves0.5358 and July0.5823. Against the original all-departure
takeoff-day posterior, the gain is0.4338 with interval[−1.5036,+0.6652]; versus
its IFR variant it is0.4084, interval[−1.4676,+0.6952]. Those wider comparisons
remain inconclusive. All321,707 scoped Gaussian predictions replay within1e−8;
all344,419 labels, calendars, fallbacks and scores pass independent SQL checks.
Retain `v3_20261011_daily_heavy_tail/`; no production candidate is selected.

The descriptive decomposition improves11 of20 airport-month groups. Istanbul
accounts for69.59 million less squared error across both months, and January
Rome for25.02 million, versus the aligned Gaussian control. The two previously
identified January Paris extreme labels worsen by2.50 million squared seconds,
so they do not drive the gain. July Rome is almost unchanged versus that
control, and remains45.34 million squared seconds worse than the original
takeoff-day method. These reused-data diagnostics select no airport gate or
label exclusion; they reinforce the need for the direct original-date test.
Full groups: `v3_20261011_daily_heavy_tail/effect_decomposition.json`.

The aligned Gaussian parent was itself unselected, so the0.5591 figure does not
establish an improvement to today's recipe. The follow-up
`daily_heavy_tail_stack.py` replaces only the Gaussian aggregate shape inside
the actual selected stack. It uses v72's original serialized v71 prior,
takeoff-day/all-departure grouping and original training-only noise parameters.
It keeps the stored v73 weight0.4032872515123992 and integer serialization,
recomputes v74's monthly shifts with the same fitted coefficients, restores
its plan bounds, and applies the selected v76 public-model change and bounds.
No date, mixture-strength, source-noise or weight search is performed.

Every original v72 Gaussian output and all later control stages must replay.
The independent SQL audit checks the replacement weight, round-to-even
serialization, labels, calendars and full selected-recipe scores. Numerical
preflight retains the8,036-state enumeration and verifies SQL half-integer
rounding. The fixed screen/audit is queued after public monthly-airline
adaptation (PID782369); no score, full-year inference or submission yet.

## New fixed screen: share public evidence across an airline's month

The existing daily posterior conditions flights separately within each day.
The new `rome_public_monthly_shift.py` instead estimates one regularized logit
shift per Rome airline, shared across the held month's interior dates. It
starts from the selected joint stack's public Cat3/R20 probabilities and uses
the existing movement-flight-number airline parser. This is public aggregate
adaptation, not calibration against held individual departure labels.

The fixed prior standard deviation is0.5 logits. A t3 likelihood compares
public daily totals with expected positive delays under the original stack's
conditional predictions at probability zero and one. Variance is fixed at the
incoming probabilities: independent recording-mode variance, the existing150s
ordinary and15s common-day scales, and retained permitted-training source/count
mismatch estimates. No likelihood, prior-scale, airline gate or blend search.
An independent conditional-mode mean can differ from the clipped blended
prediction; the screen records that discrepancy as a limitation.

The source dates use takeoff minus the frozen incoming prediction; calendar
edges and explicit VFR are excluded. Inference reads a strict whitelist and
never sees `y` or the recorded off-block date. The inferred probabilities pass
through all existing prediction corrections and the same v74 transfer/bounds.
The selected joint controls must replay exactly and non-Rome predictions stay
identical. There are no new taxi-model fits.

Numerical checks pass: independent daily sums, analytic versus finite-difference
gradient (maximum error6.1e−12), exact identity for matching source totals or an
empty scope, row/source permutation, shared airline shifts, expected update
direction and poisoned-label isolation. The screen and independent row-loop/
SQL audits are queued after the neural Rome screen (PID777889). No Jan/Jul gain,
production candidate or upload is claimed. Retain the source, frozen identity
and logs under `v3_20261011_rome_public_monthly_shift/`.

## Current queue: selected v76 and a new Rome classifier

**Production completed11:39 Amsterdam.** All three15-epoch networks are saved,
with exact all344,841-row checkpoint reloads per member. The separate audit
reconstructs inputs and predictions on615 rows per network (1,845 total), with
maximum difference0.001371 seconds at0.01 tolerance;7,380 public-transform values
and32 scalar track reconstructions also pass. Retain all models and predictions.
The desktop has advanced to the fixed heavy-tail screen. V76's build remains
queued after the existing validation jobs; there is still no v76 file.

At11:28 Amsterdam, the selected joint public main/neural recipe has a reserved
version **v76**, CV **292.4494**. Its build and independent audit are queued after
the existing monthly-reference comparison; full-year neural production remains
in progress. No v76 submission file or official score exists yet. This queue has
no upload stage, and the already authorized v75 upload watcher is unchanged.

A new independent classifier family is queued after that build: a PLE neural
binary classifier for Rome's recorded schedule-copy mechanism. The primary
comparison adds a fixed one-third neural probability to the existing public
Cat3/R20 mean on the selected joint stack. One seed, unweighted binary
cross-entropy, one architecture, no blend search. Pure neural probabilities are
a diagnostic only. This differs from earlier auxiliary heads in the ordinary
taxi network and from failed cost-weighted Rome trees.

The75 inputs are the existing80 minus five departure-label history columns;
their exclusion keeps the inner stopping month out of target-history inputs.
Consequently this is a family-plus-input-removal comparison, not a pure family
ablation. Encoders and numeric bins are fitted separately on inner and full
training rows. The final epoch is the lowest inner-month log loss among48
epochs; held-out January/July labels never select the stopping epoch. All
scoring includes the complete taxi correction stack and the v74 transfer.

CPU checks cover unknown/missing categories, finite extreme inputs, gradients,
state reloads, row permutation and isolation of ordinary-model settings. Input
checks match all126,507 January-fold and86,723 July-fold training IDs and binary
targets to raw rows; held Rome populations are11,317 and15,211. Full fits will
compare predictions from the in-memory trained network against its disk
checkpoint before releasing memory. No fitting or validation gain is claimed
yet. Sources: `rome_neural_model.py`, `rome_neural_jan_jul.py`; retained identity
and preflight in `v3_20261011_rome_neural/`, queue in `v3_20261011_target215/`.

The active target is **215.00 seconds or lower**, replacing the completed
225.00 target. Starting best/latest is **v71, 224.3602**, rank 15/242 on 344,841
original January/July flights (live 06:29 Amsterdam). The gap is 9.3602 seconds.
Autonomous uploads and both computers remain authorized, one heavy job per
machine. The three research agents report an account usage limit and were not
restarted. Laptop WSL is currently offline; work is running on the desktop.

## Public daily pre-departure totals

Newly published [gracious-tornado documentation](https://github.com/il-miscusi/prc-2026-gracious-tornado)
at commit `a0344440b497c38bc80390a61b3e7890904b3634` reports a material gain from
conditioning flight predictions on EUROCONTROL daily pre-departure totals.
The earlier network had identified the source but had not downloaded or tested
it. The portal's displayed coverage ends in June, but the actual 2026 file
covers every January/July airport-day needed here. All four downloaded source
hashes match the public repository's documented copies.

At airport-day level the proposed measurement is
`60 * DLY_ALL_PRE_2 = sum(max(0, takeoff - schedule - taxi_out)) + error`.
An independent DuckDB calculation exactly reproduces all **3,650** training
day aggregates over **2,085,047** departures. On equal-count days the median
absolute discrepancy is 52 seconds at Rome, 56 at Munich and 58 at Heathrow.
There are also large discrepancies: Rome's 99th percentile exceeds 13,000
seconds despite equal counts. Counts alone do not guarantee the same flights.
The likelihood therefore needs uncertainty for population mismatch and tails.

Source: [EUROCONTROL performance data](https://ansperformance.eu/data/),
`all_pre_departure_delays_{2025,2026}.csv` from its public download service.
Credit: EUROCONTROL. Its [terms](https://ansperformance.eu/about/disclaimer/)
permit attributed non-commercial reuse and restrict modification; they are
not an open licence. Files are retained unmodified locally and not redistributed.
The other team's reported organiser ruling names the monthly taxi-out file,
not the daily file; no daily-file ruling has been independently established.
The user authorized continuing without treating competition guidelines as an
approval gate. This does not constitute a claim of organiser clearance.

## Completed screens

All fitted measurement-error parameters and held-taxi distributions use only
the Jan/Jul fold's allowed training months. No target or block timestamp is
loaded for ranking inference. The original predictions, fold definitions and
historical model artifacts are retained unchanged.

| Method | Parent | Jan/Jul CV | Change |
|---|---|---:|---:|
| Gaussian positive-part moments | v71 | 296.7114 | +0.3954 |
| Discrete long-flight posterior | v71 | 295.4354 | −0.8806 |
| Same posterior, alternate starting model | v70 | 292.0049 | −0.8628 |

The Gaussian screen failed mainly at Rome. It uses rectified-normal expected
contributions and a linear Gaussian update; its positive-part correction can
raise the estimated daily total and push too much correction onto Rome's long
flights. Retain it as failed evidence, without a submission-ledger entry.

The second model uses discrete long-flight supports, a robust t3 measurement
likelihood and exact forward/backward messages on a 60-second grid. Short
flights use a linear aggregate with fixed 150-second individual and 15-second
common-day residual scales. Rome's long-flight prior includes ordinary and
schedule-copy modes; its prior mean matches the incoming prediction. Support
spread and source-mismatch terms are fitted on allowed training rows. No
held-out airport gate or scale search was performed. This is an independent
implementation of the credited idea, with different prior/likelihood choices;
no public source code was copied.

Against v71, January changes **−0.5034**, July **−1.2579**; 19 of 20
airport-month groups improve. The paired-day interval **[−2.5864, +0.9180]**
includes no gain. Against v70 both months gain about 0.86 seconds and the
interval is **[−2.4143, +0.5859]**. These are reused development months and
correlated comparisons, not independent confirmation. The v71 variant is
selected for an exploratory leaderboard test on top of the best scored file.

Numerical checks pass: five quadrature comparisons of Gaussian moments,
seven exact prior-mean checks, no-information identity, permutation invariance,
and explicit enumeration of **8,036** latent states against the message
algorithm (maximum error **7.3e−12 seconds**).

Runners: `daily_delay_reconcile.py`, `daily_delay_posterior.py`,
`daily_delay_submit.py`, `tools/audit_daily_delay.py`. Retained runs:
`v3_20261011_daily_delay/`, `v3_20261011_daily_delay_posterior/`,
`v3_20261011_daily_delay_submit/`. Source receipts and complete raw CSVs are
cached under `data/external/eurocontrol_performance_20261011/`; do not refetch.
The monthly taxi-out files are cached but have not entered a candidate.


## Official results and continuing work

**v72: 226.5719**, 2.2117 worse than v71. Full-strength daily conditioning did
not transfer. The subsequent scored blend **v73: 223.1850** improves v71 by
**1.1752 seconds**, rank **13/242** at 06:53 Amsterdam. V73 uses weight 0.4033
on v72 with the other retained weights recomputed. The independent squared-error
identity projects 223.1851; the conservative clipping/rounding bound is 223.4742.
Jan/Jul CV improves to 295.2491 (−1.0669, interval [−2.0688, −0.1323]). This
is leaderboard-conditioned selection, not independent confirmation. Both remote
file hashes match. Four October11 UTC uploads are used; one remains.

The target remains **8.1850 seconds away**. A new neural screen adds a
training-only per-observation off-block-boundary head to the physical track
encoder. Ordinary training labels identify whether each cached observation
falls before or after recorded off-block, smoothed over 30 seconds. Fixed
auxiliary BCE weight 0.1; the main target, original156 inputs, optimizer, batch
order and original21/17 epochs remain. This differs from the failed tabular
recorder-mode supervision: it teaches the track representation a dense temporal
boundary and never replaces the Rome classifier. Initial physical model weights,
RNG and outputs match exactly; CPU/CUDA checks of masking, target orientation
and gradient flow pass. Two sequential desktop fits are in progress under
`v3_20261011_nn_track_boundary_jan_jul/`. No model gain is claimed yet.

## Monthly means and recovered flight plans: v74

**v74 officially scores 221.6881**, improving v73 by **1.4969 seconds** and
v71 by **2.6721**. Rank **12/242** at 07:12 Amsterdam; **6.6881 seconds** remain.
All five UTC October11 slots (v70–v74) are consumed. Continue preparing and
validating candidates; the next upload window begins October12 00:00 UTC.

The credited public repository also led to two underused sources of information.
The monthly taxi-out file gives an average for eligible departures. Its
[official method](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf)
uses airport-local months and excludes missing stand/runway/off-block values,
taxi times above two hours, helicopters, post-off-block deicing and combinations
without a reference time. `TF - VALID` is an invalid share, not a measured
deicing share. We cannot reconstruct all exclusions. A Huber model fits the
difference between our eligible training average and the published average,
using airport intercepts, invalid and no-reference shares. Fit only permitted
training months, apply half the estimated correction with a 60-second cap,
and require invalid share below 5% and at least 100 selected flights. True
training taxi versus predicted test taxi for the two-hour cut is a limitation.
On v73 this alone changes CV **295.2491 → 294.9433 (−0.3058)**, interval
[−0.4168, −0.2314], both months improving. No separate upload.

For LIRF/LTFM no-plan departures, unique arrival records can recover the flight
plan that failed the original departure join. Matching uses exact flight number
and route, compatible aircraft, a consistent plan route, a landing 10 minutes
to 20 hours later, and exactly one candidate. It reads no departure target or
block time. All **175/175** accepted 2025 matches have taxi outside recovered
LOBT ±3606 seconds. The ranking set has 49 accepted matches. Fold context is
restricted to training plus held input months. The airport gate comes from
published work using the same 2025 data; this remains reused development evidence.

The first prototype incorrectly treated a negative lower edge as zero. An audit
found four CV outputs still inside the excluded interval. The corrected version
uses the upper edge when the lower edge is negative. It passes synthetic boundary
checks and gives **−1.1041 CV**, interval [−2.7325, −0.0891] against v73. Nearest
projection onto a nonconvex support does not guarantee improved squared error;
this rule is justified by empirical matching and validation, not such a theorem.
Retain the failed prototype separately; it was never submitted.

V74 composes monthly calibration, restoration of original matched-plan convex
bounds after the signed blend, then the corrected recovered-plan rule. CV is
**293.7173 (−1.5318)**, interval **[−3.2008, −0.5140]**, January −0.3089 and July
−2.7546. Integer CV is 293.7157. It changes 273,161 rounded predictions, including
70 matched-bound and 20 recovered-window changes. An independent pandas join
reproduces all 224 accepted training/ranking donor matches; all 344,841 production
rows and intermediate stages replay exactly. Remote SHA-256 matches the audited
file, `55bd7ba5c8aa214b676ca5ec11ca0e882aab3bb93b0a31e9d724aac72f3989c5`.

Runners: `monthly_taxi_reconcile.py`, `recovered_plan_exclusion_fixed.py`,
`public_constraints_candidate.py`, `tools/audit_public_constraints.py`.
Retain `v3_20261011_monthly_taxi/`, `v3_20261011_recovered_exclusion_fixed/`
and `v3_20261011_public_constraints/submission_v74/`. Do not rebuild or upload v74.

## Boundary supervision failed; airport specialization running

The two completed boundary-auxiliary fits give **293.5174** versus the matched
physical single model's **293.3477**, **+0.1697**, interval [−0.0551, +0.4056].
January worsens 0.1867 and July 0.1527. No production fit or upload is selected;
retain both models and predictions without adding a submission-ledger row.

The next fixed screen starts from saved physical seed42 fold networks and runs
two continuation epochs at learning rate 2e−5, batch 2048. One arm trains globally;
the other makes a full copy for each airport and updates each only on its own
rows. Inputs, preprocessing, labels, loss and starting weights match. This differs
from the previously failed shared-trunk airport residual heads. The global arm
controls for extra training. CPU/CUDA preflight verifies initial predictions,
expert dispatch and unknown-airport fallback. No held labels select epochs or
learning rate. Both arms run sequentially on the desktop under
`v3_20261011_nn_airport_finetune/`; no gain is claimed yet.

The airport screen is complete: global continuation **293.3600**, separate
experts **293.3138**, original physical single **293.3477**. The primary gain
is only **0.0462 seconds**, interval [−0.1044, +0.0063], and is inconclusive.
No production fit or upload. Both arms and all models are retained.

Extending recovered-plan matching to all ten airports, with a gate of at least
50 permitted training matches and zero interval violations, adds only one changed
CV row and three ranking rows. CV moves by **−0.00007 seconds**; no promotion.
The broader audit finds real violations at EDDF, EHAM, LEBL, LFPG and LSZH, so
extending the hard rule indiscriminately is unsupported. Recombining separately
scored v74 with the existing basis projects **221.5321**, only 0.1560 better;
CV 293.4460 has interval [−0.8026, +0.0708] versus v74. Keep this evaluation in
run records, without an upload or candidate-ledger entry.

## Correct the daily source population; learn how to use its signal

The daily file counts IFR departures, while the first posterior included all
movements. Excluding explicit VFR flights reduces median absolute aggregate
error at EHAM **2,547.2 → 156.0 seconds**, and LSZH **360.8 → 74.4**. EHAM
count agreement rises from 13 to 303 of 365 days; LSZH from 77 to 349.
Unknown movement flight rules remain included: removing them worsens EDDF
agreement materially. This is a source-identity diagnostic, not model evidence.
The original arithmetic audit was exact for its stated all-departure sum; its
population assumption was too broad.

Changing only this population in the existing daily posterior gives CV
**295.4177**, **−0.0177 versus the old posterior**, interval [−0.0450, −0.0034].
Both months improve slightly; it is not a material standalone candidate.
Retain `daily_delay_ifr.py`, `v3_20261011_daily_ifr/` and the source diagnostic
under `v3_20261011_daily_ifr_audit/`. Do not claim this closes the target gap.

The next screen gives the Rome R20 classifier twelve input-only daily aggregate
features: public count mismatch and mean delay, long-flight counts/sums,
residual totals against fixed 900/1800-second taxi references and the raw
AOBT_3 clock, and the amount left for this flight after those peer estimates.
It uses the corrected IFR population. No departure target or block timestamp
enters a feature. The fixed references require no label fitting; daily groups
do not cross held months. Original68 classifier inputs, rows, labels, R20
settings, 1,700 rounds and seed42 remain unchanged. This tests learned use of
the public measurement rather than another hand-set posterior scale. Primary
comparison uses the selected v70 ordinary models and Cat3/R20 half mixture;
a fixed secondary delta transfer to v74 is reported separately. Synthetic
population, arithmetic, row-order and target-isolation checks pass. Runner
`rome_public_delay_jan_jul.py`; features in `public_delay_features.py`.

## Public daily inputs: classifier selection and small neural effect

The Rome LightGBM screen completed with **−0.4732 seconds** in the selected v70
stack (January −0.2323, July −0.7141), interval [−0.9261, −0.0384]. Its fixed
delta transfer to v74 gives **293.1715**, **−0.5459**, interval [−1.0194, −0.0853].
Both saved LightGBMs reload with exact probabilities on every held Rome row.

The same features were then given to the original three CatBoost seeds with
unchanged settings and original training-selected rounds (1182/1401). No new
stopping search. All six models reload with **exact** held probabilities.

| Changed classifier components | CV in v70 stack | Change | Fixed v74 transfer CV | Change |
|---|---:|---:|---:|---:|
| Original components | 292.8675 | — | 293.7173 | — |
| R20 only | 292.3944 | −0.4732 | 293.1715 | −0.5459 |
| CatBoost pool only | 292.2010 | −0.6666 | 292.9415 | −0.7758 |
| Both | 292.2383 | −0.6293 | 292.9019 | −0.8155 |

**Select CatBoost only.** Its v74 transfer improves both months; interval
[−1.3893, −0.1546]. Changing both components has a wider interval
[−1.7749, +0.1962] and does not beat CatBoost alone in the primary matched
stack. This is reused development selection, not independent confirmation.
Large Rome long-flight errors improve and worsen; the selected change is not
a per-flight exception. Three full-year CatBoost refits use the original2017
rounds and160,577 rows/33,205 positives. The other full-year models are reused.
Runner `rome_public_delay_submit.py`; v75 is fully prepared and independently
audited, with no upload yet. All three production models reload exactly on
26,899 Rome rows, and all344,841 final predictions reproduce. All317,942
non-Rome rows stay unchanged. Ranking features pass an independent SQL check
of4,138,092 values. The audited file is queued for12 October02:00 Amsterdam,
with fresh source/schema/quota checks before upload; queue state is in
`v3_20261011_target215/queue_v75.json`. Its manifest is immutable while queued.

Independent SQL reproduced **25,020,564 feature values on all2,085,047** 2025
departures, maximum floating-point difference 4.7e−10. The feature builder reads
only movement/schedule/AOBT_3 clocks, flight rule and the public aggregate.

A short neural follow-up adds a zero-initialized residual head using the same
12 values plus the original hidden representation, starting from saved physical
models. Two fixed epochs (original weights2e−5, new head1e−3) give293.2865:
**−0.0613** against the original physical single model, interval
[−0.1595, +0.0097]. Against the matched two-epoch global continuation it gains
0.0735, interval [−0.1228, −0.0497], but the original-model comparison remains
inconclusive and the effect is small. No expansion or production. Both models
remain under `v3_20261011_nn_public_delay/public/`.

A main-LightGBM screen is running in `public_delay_lgb_jan_jul.py`: preserve
all162 current inputs, LB1 settings,10,000 rounds andseed42, append the same12
public values. It started at07:53 Amsterdam after the production classifier
fits, on the desktop with four threads. Saved main controls are from the laptop; any small result
must retain that numerical-comparison caveat. Laptop WSL still timed out at
07:45 Amsterdam; no laptop job was started.

Reusing all six retained TabM predictions (three seeds per fold), a fixed
25% neural share in the selected v70 stack scores293.1503 versus292.8675:
**+0.2828**, interval[+0.1178,+0.5059], both months worse. The fixed v74
transfer is also worse by0.1123, with an interval including zero. This resolves
the old unused-pool lead without new fits. No weight search, production model,
upload or candidate-ledger row follows. Runner `tabm_retained_blend.py`, run
`v3_20261011_tabm_retained_blend/`.

## Fixed follow-ups queued after the main LightGBM

The LightGBM fits are complete (18.5/14.7 minutes). Its primary CV improves
**292.8675 → 292.6367 (−0.2308)**, interval[−0.3177,−0.1734]; the fixed v74
transfer improves **293.7173 → 293.5470 (−0.1703)**, interval[−0.2298,−0.1229].
Both months improve. Saved controls ran on the laptop and candidates on the
desktop, a numerical-comparison caveat. Saved-model reload and its combination
with the selected Rome change are queued; no production fit or ledger row yet.
XGBoost started after both fitting and scoring finished. Every heavy job is
sequential on the desktop; laptop SSH timed out again around08:18 Amsterdam.

1. **XGBoost diversity:** the existing 162 inputs, original training rows and
   labels, 4,000 fixed depth-8 rounds, seed42. Add it as an equal fourth tree
   member inside the unchanged0.29 tree share. No held-label stopping or weight
   search. The native categorical path preserves the existing shared category
   vocabularies and missing values. The initial numeric-code assumption was
   corrected before fitting: retained categories are strings. CPU/CUDA checks
   cover missing/unseen categories, shuffled category order and serialization;
   real cached64-row train/held frames pass all162 columns and13 categories.
   reloading onto the same device fixes a4.8e−7 difference from default CPU
   reload of a CUDA model. XGBoost3.4.1 is pinned;3.4.2 shown in the live docs
   was unavailable from the package index. See the official
   [GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html) and
   [categorical](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)
   documentation. Runner `xgboost_diversity_jan_jul.py`.
2. **Full public neural training:** the earlier two-epoch continuation was
   too limited to establish whether the public inputs help joint learning.
   This fixed follow-up trains the same zero-initialized public residual head
   from the original physical model initialization for the original21/17 epochs.
   Original156 preprocessing and the exact original sparse optimizer bytecode
   remain; the12 fixed-scaled public channels bypass that encoder. CPU checks
   verify original weights/RNG, dropout, a nonzero physical path, padding and
   gradient flow. Comparison is with the saved same-desktop physical single
   models at the same epochs. Runners `nn_public_scratch.py` and
   `nn_public_scratch_jan_jul.py`.

These are unscored experiments, not selected candidates. Queue/source bindings
and status live in `v3_20261011_target215/research_queue.json`. Its cancellation
file or an inactive objective stops further jobs and any active child.

**XGBoost is complete and rejected.** The fixed fourth-tree blend scores
293.1108 versus292.8675, **+0.2433**, interval[+0.1736,+0.3524]. The fixed v74
transfer also worsens **+0.1631**, interval[+0.1025,+0.2557]. Both months worsen.
All held predictions from both saved models reload exactly on CUDA. No weight
search, production model or upload follows. The full public neural run started
after its scoring completed; models and results remain in
`v3_20261011_xgboost_diversity/`.

## Cross-fitted recording-mode peer budgets

The selected public inputs subtract fixed ordinary taxi references for every
other flight. A new fixed screen instead discounts those contributions by a
recording-mode probability: a schedule-copy row contributes approximately zero
pre-departure delay. Ten features include that probability, the remaining daily
budget at900/1800-second references, its per-flight fraction, and Bernoulli
uncertainty in the total. This remains a learned feature treatment of the public
measurement; no hidden labels enter it.

For each outer fold, three interleaved groups of its permitted training months
produce month-excluded training probabilities. A fourth model, fitted on all
permitted months, predicts the outer holdout. All use fixed R20 settings and
1,700 rounds. The five departure-taxi history columns are removed from these
inner models so they cannot carry excluded-month departure labels; public daily
inputs are also absent. The remaining63 inputs are clocks, supplied arrivals,
categories and open track observations. The final three CatBoosts retain v75's
parameters, training rows and rounds, adding the ten features to its80 inputs.

Feature arithmetic, VFR exclusion, unknown-rule inclusion, padding-independent
row order and target isolation pass hand-built checks. Actual input coverage is
complete except the already rule-covered109/46 training flights: these use the
existing zero-delay copy-mode fallback. There are no other missing peer
probabilities in either training fold. Month membership checks pass. These are
preflight results, not evidence of predictive gain. Runners
`rome_public_peer_features.py` and `rome_public_peers_jan_jul.py` are queued
after the neural run and main-LightGBM audit via `peer_research_queue.json`.

**Completed: no production selection.** The cross-fitted peer inputs improve
the primary stack 292.2010 → 291.9126 (−0.2884), but its paired interval
[−0.7870, +0.1841] includes a loss. The v75 transfer changes 292.9415 →
292.8009 (−0.1407), interval [−0.5469, +0.2545], also inconclusive. Retain
all eight inner models and six CatBoost fits; do not expand or upload this arm.
The queued all-model reload audit still runs.

The separate SQL auditor reconstructs every inner training/prediction ID and
checks all **2,399,130** peer-feature values across the two outer populations.
Maximum floating-point discrepancy is **1.63e−9**. All inner fitting months
exclude their predicted months; the five departure-label history columns and
public inputs are absent. See `tools/audit_rome_peer_features.py` and the run's
`feature_audit.json`. These are implementation checks, not independent evidence
of predictive gain.

## Independent public daily punctuality inputs

EUROCONTROL's [daily airport punctuality dashboard](https://www.eurocontrol.int/Economics/DailyPunctuality-Airports.html)
directly links a public JSON containing departure/arrival on-time fractions,
mean schedule delays and the operated-schedule fraction. The retained unmodified
file covers every day of 2025 and all four ranking months at all ten airports.
Its SHA-256 is `c239cd2fc7ae8b274fad56ab100d7bf4401e07bd238e4f7e8d91f4048318e4f1`.
Credit: EUROCONTROL; the general attributed non-commercial terms discussed
above apply. No organiser clearance or open licence is asserted.

This source uses airline schedules and differs from the supplied movement
population. Across 2025 days, departure-mean correlations range 0.885–0.991,
but absolute mean differences range 1.39–5.23 minutes; Rome's median signed
difference is +5.03 minutes. Departure-punctuality correlations range 0.940–0.991.
These are descriptive source checks, not validation gains. The source will
enter learned features, not an exact aggregate constraint.

The fixed Rome screen adds nine inputs to v75's 80-column CatBoost recipe:
five source values and four discrepancies from input-only 900/1800-second
taxi references or AOBT_3 clocks. The original rows, three seeds, rounds and
R20 mixture remain fixed. An independent raw-JSON parser and DuckDB calculation
check all **18,765,423** feature values exactly, including VFR exclusion and
unknown-rule inclusion. Hand arithmetic, row-order invariance, source coverage
and target-isolation checks pass. The run is queued after the peer-budget
experiment; no fit or score is claimed yet.

Runners: `public_punctuality_features.py`, `tools/audit_public_punctuality.py`,
`rome_punctuality_jan_jul.py`. Runs: `v3_20261011_public_punctuality/`,
`v3_20261011_rome_punctuality/`; queue `punctuality_research_queue.json`.

**Rome result: no gain, no production.** The primary score changes
292.2010 → 292.2328 (+0.0318), interval [−0.1417, +0.2245]. The v75 transfer
changes 292.9415 → 292.9544 (+0.0128), interval [−0.1416, +0.1793]. Both
comparisons are inconclusive. All six saved models reproduce every held
probability exactly; independent SQL reproduces the scores. The peer models'
six probability reloads also pass. Retain both completed Rome screens without
more fits or uploads.

This tests the recording classifier only. A separate fixed ordinary-regressor
screen is now queued: append the same nine inputs to the passing 174-column
main LightGBM, retain LB1/10,000/seed42, and compare against its actual
public-main/public-Cat baseline, CV 292.7707. Both controls and new fits use the
desktop. All original 174 inputs and training rows remain unchanged. Runner
`punctuality_lgb_jan_jul.py`, queue `punctuality_main_queue.json`, after the
structural-posterior screen. No score or production selection is claimed yet.

A separate v75 residual grouping reconfirms that two previously investigated
January LFPG labels dominate one small group's CV error. No defensible new
recording rule was found in the earlier investigation; no clipping, removal or
reweighting follows. The descriptive grouping is retained in
`v3_20261011_target215/v75_error_groups.json`; it is not ranking-error attribution.

## Full public neural training passes the matched screen

Training the public residual head jointly for the original 21/17 epochs gives
**293.0964** versus the same-desktop physical seed-42 control's **293.3477**:
**−0.2513 seconds**, paired-day interval **[−0.5010, −0.0369]**. January gains
0.2065 and July 0.2961 seconds. The older v64 sparse pool is also shown in the
run, but differs in architecture and pool size; it is not the matched control.
This fixed full-training comparison follows the inconclusive two-epoch screen.
The development months have been reused extensively.

The declared follow-up retains the completed seed 42 and adds seeds 1042/2042
at the original 21/17 epochs. It compares equal-size three-member physical
pools inside the prepared v75 recipe and restores the existing final bounds.
Weights, stopping times, other models and training rows remain fixed. CPU
initialization checks reproduce all original weights and RNG state at both new
seeds; the compiled training AST differs only in its seed argument and model
constructor. No full-year public neural fit is selected yet.

`expansion_research_queue.json` follows the punctuality screen. It reloads both
Rome experiments, audits the single public network, runs and audits the public
three-member pool, and then fits the already passing public main LightGBM once
on all 2025 ordinary rows if its combined CV audit passes. Each job is sequential;
the shared cancellation file stops this queue too. Runners:
`nn_public_pool3.py`, `tools/audit_public_neural.py`,
`tools/audit_public_rome_cv.py`, `public_delay_lgb_submit.py`.

The main-LightGBM reload audit initially omitted the production helper's
existing zero floor. Its 44 January disagreements were precisely negative raw
outputs versus stored zeros. The auditor now applies the same floor before
its exact comparison; completed model fits and prediction files are unchanged.
The stopped queues were rebound to the corrected auditor and resumed; their
failure records are retained. This was an audit implementation error, not a
change to model validation results.

The corrected main audit is **complete and passes**: all 344,419 predictions
reproduce exactly, including the 44 January and 19 July zero-floored values.
Independent SQL reproduces the CV arithmetic. The explicit combination with
the selected public Rome CatBoost pool scores **292.7707**, versus v75's
**292.9415**: another **−0.1709 seconds**, interval **[−0.2306, −0.1230]**.
January improves 0.1438 seconds and July 0.1979. The primary underlying stack
also improves 292.2010 → 291.9697 (−0.2313). This computes the Rome/ordinary
interaction directly rather than summing separately projected changes.
The full-year main fit is queued once; no production prediction or official
score for this combination exists yet.

The eventual main-tree combination builder and independent final auditor are
prepared as `public_main_candidate.py` and `tools/audit_public_main_candidate.py`.
They must reproduce v70's tree mean and v75's entire file before changing the
main tree. A snapshot of v75's reviewed manifest keeps its later upload-status
changes separate from immutable prediction provenance. These tools have not
been run; no new version has been reserved or assembled.

## Queued follow-up: preserve the recording-mode distribution

The retained daily posterior infers Rome's copy probability from the incoming
mean and a single airport-level ordinary taxi mean. Its v71 input was already
a signed ensemble, so this inversion need not recover the actual classifier
probability or the per-flight ordinary mean. This is a concrete limitation of
the failed full-strength posterior, distinct from adding daily totals as inputs.

After the earlier queued screens, a fixed follow-up constructs the two
conditional means directly from v70's saved ordinary prediction and its original
Rome classifier, then condition that distribution on the public total. The
zero/one classifier endpoints of the existing correction function provide a
check that the mixture reconstructs the original prediction. The prior must
exclude the new public-data models to avoid treating the same daily measurement
as independent evidence twice. The new screen retains the incoming serialized
v70 means, all likelihood/noise/spread settings, the original all-departure
population and every non-Rome posterior prediction. It changes only Rome's
long-flight prior support. Equal correction endpoints at the schedule gap give
a fixed-copy point mass; otherwise identifiable copy modes use the saved
per-flight ordinary endpoint. Nonidentifiable cases retain the original support.

The existing discrete message algorithm is reused unchanged. Numerical checks
preserve prior means within 8.64e−12 seconds, reproduce the fallback exactly and
keep a fixed-copy state unchanged even under conflicting public totals. These
are preflight checks, not a validation result. `daily_structural_prior.py` is
queued after the current expansion work via `structural_research_queue.json`;
it needs no new model fits. No production or v74/v75 transfer is selected.

## New source-identity finding: the daily total follows off-block dates

The existing daily method groups flights by UTC takeoff date. Large source
discrepancies often cancel on adjacent days: Rome's daily residual has lag-one
correlation −0.481. Replacing the training grouping with the recorded off-block
date sharply improves the source identity. This is an empirical finding; the
[official column description](https://ansperformance.eu/reference/dataset/all-pre-departure-delay/)
only calls `FLT_DATE` the flight date and does not specify that boundary.

| Airport | Equal-count days, takeoff → off-block | 95th-percentile absolute total error, seconds |
|---|---:|---:|
| Rome | 244 → 358 | 7,230.6 → 141.6 |
| Barcelona | 253 → 359 | 13,170.6 → 179.4 |
| Madrid | 140 → 354 | 15,620.6 → 233.8 |
| Amsterdam | 303 → 321 | 8,245.0 → 603.7 |
| Istanbul | 70 → 334 | 25,988.4 → 1,510.0 |

These are 2025 source-reconciliation diagnostics with explicit VFR departures
excluded, not prediction scores. Paris still has substantial population mismatch.
Every one of the 2,085,047 training off-block timestamps equals takeoff minus
the supplied taxi label. An independent SQL calculation using the raw block
timestamps reproduces all 7,301 airport-date aggregates across the two groupings.
Counts and computed truth sums match exactly; independent CSV parsing agrees
on public totals within 1e−8 seconds.

The true off-block timestamp is unavailable at ranking inference. A first
input-only approximation subtracts the retained v70 prediction from takeoff.
On the reused Jan/Jul development rows this assigns the wrong date on only
24 of 344,419 flights; Rome's source-error 95th percentile is then 137.7 seconds.
This uses labels for the diagnostic comparison after assignment, not for
inference. It does not establish a taxi-prediction improvement. Independent
SQL reproduces every assigned date and inference-scope flag.

A fixed one-pass posterior screen is now queued before ordinary punctuality.
It compares against a matched v70 IFR-population posterior, retaining the
original message algorithm/settings. It assigns dates once from the original
v70 prior, with no public-input models in that prior. All nuisance and support
distribution estimates use allowed training months only; the off-block fit
excludes first/last calendar days to reduce partial source-day coverage. Thus
the comparison changes both grouping and its associated calibration population.
At inference, first/last/missing held off-block dates retain the matched control.
Only explicitly whitelisted input columns enter date assignment and conditioning.

Hard date assignment does not marginalize boundary uncertainty, and the
posterior may cross its initially assigned date; the result will report those
cases. No weight/parameter search or production is selected. Numerical checks
cover midnight, month edges, target isolation and row order. The waiting
punctuality queue was stopped and rebound, without touching its model sources
or starting any fit, so all heavy jobs remain sequential.

Runners: `daily_offblock_alignment.py`, `tools/audit_offblock_alignment.py`.
Run: `v3_20261011_daily_offblock_alignment/`; queue:
`v3_20261011_target215/offblock_alignment_queue.json`. Source audit, calendar
preflight and the descriptive prior-date diagnostic are retained there.

## Public neural production preparation

`nn_public_submit.py` is prepared for a passing three-member result. Its
preparation guard requires both the primary and v75-transfer comparisons to
pass, plus the queued neural reload audit. It reuses the verified full-year
156-column packet and original encoder, then appends the same twelve public
transforms as CV. The three seeds retain the original15-epoch full-year schedule;
training bytecode changes only the model constructor. CPU checks reproduce
original weights/RNG for all three seeds and verify masking and gradient flow.

No full-year public neural fit has started or been selected. The planned arm
keeps all new members on the desktop; the original full-year physical pool was
fitted on the laptop, while both CV pools use the desktop. Each future fit is
saved before inference, and every old-ranking prediction must reproduce on
reload. `tools/audit_public_neural_production.py` additionally checks a sample
using the retained encoder, raw tracks and independently recomputed public
transforms. These are prepared checks, not completed production audits.

A fresh read of the credited public
[daily-method source](https://github.com/il-miscusi/prc-2026-gracious-tornado/blob/main/research/final/levers/daily.py)
also highlights a remaining difference: its ordinary-day shift uses a fitted
heavy-tailed prior, while ours has fixed Gaussian scales. That is a possible
follow-up after the pending date-alignment and structural-prior results; no
new experiment or claim of gain follows from this source comparison alone.

**The three-seed public neural pool subsequently passed.** In the actual public
Rome recipe, matched-size primary CV improves **292.2010 → 291.9242 (−0.2767)**,
interval **[−0.4264, −0.1444]**. The v75 transfer improves **292.9415 → 292.5880
(−0.3536)**, interval **[−0.5065, −0.2263]**. January gains 0.1984 and July 0.5087
seconds in that transfer. These remain reused development months, not an
official-score result. All six model reload checks pass on 3,282 sampled rows,
maximum discrepancy 0.00147 seconds. A separate SQL audit checks all 344,419
labels, dates and retained controls exactly and reproduces the monthly scores.
No repeated fits were needed.

Full-year production is now selected and queued after ordinary punctuality:
`public_neural_production_queue.json` (now PID721659), waiting on PID707786.
It prepares the frozen packet, fits all three members, checks every ranking
reload and then runs the separate input/encoder/track auditor. It shares the
research cancellation file and has no upload stage. The training guard now
also binds the independent score audit and revalidates the completed CV
manifest; its updated CPU preflight passes. The earlier preflight record is
retained as history. No new submission version or final blend is reserved yet;
combining the public main tree and new neural pool still needs an explicit
joint CV calculation. The public main-tree full-year fit is running now.

That direct calculation is now the first job in the neural production queue.
`public_stack_cv.py` uses the retained six neural models and two main-tree
models; no refitting, blend-weight search or new classifier is involved.
It requires exact all-row replays of v75 and both single-change candidates,
then applies the same full correction stack to their combination. A separate
SQL pass checks all labels, calendars, controls and monthly scores. The
provenance guard has passed; scores remain pending. The original waiting
queue (PID717805) was stopped without starting a child and archived before
rebinding its source hashes to PID721659. No upload queue was changed.

`public_neural_candidate.py` and `tools/audit_public_neural_candidate.py` are
prepared for the retained production pool, either alone or with the public
main tree. The combined mode requires improvement over both individual
transferred recipes in the joint CV; neither mode refits models. They replay
the original neural pool, v70 and v75, then independently check all final
predictions, constraints, template IDs and SQL scores. Compilation and CLI
import checks pass. Full execution awaits production and joint CV; no file,
version or upload has been created by these builders yet.

## Source precision check

The [official daily-data schema](https://ansperformance.eu/reference/dataset/all-pre-departure-delay/)
describes the delay field in minutes; it does not guarantee an exact match to
this challenge's flight-level sum. A follow-up on the already audited 2025
off-block aggregates finds that equal counts still leave material residuals.
At Rome, 358 equal-count days have median signed error −41.5 seconds, robust
standard deviation 58.6 seconds and 95th-percentile absolute error 132.9 seconds.
Across airports the robust scale divided by square root of the count is roughly
2.7–3.4 seconds. This is descriptive evidence, not a fitted inference rule or
proof of the source's noise mechanism. Retain uncertainty; CSV precision does
not justify an exact or rounding-only constraint. Full statistics and the
audited input hash are in `v3_20261011_daily_offblock_alignment/source_noise_diagnostic.json`.

## Main production complete; conditional Rome support rejected

The public main LightGBM completed its original 10,000 rounds on 2,051,327
ordinary rows in 20.16 minutes. The saved model reproduces every one of its
344,841 ranking predictions exactly after reload. Retain the model, input
identity and prediction records under `v3_20261011_public_delay_lgb_submit/`;
no extra fit or upload is needed. A combined final file still awaits joint
CV and the new neural production pool.

The fixed structural-prior screen finishes at **292.5676**, versus **292.0049**
for the original daily posterior: **+0.5626 seconds**. January changes −0.0611
and July +1.1863; paired-day interval [−0.0344, +1.3290]. This does not support
expansion. The prior-mean checks and exact serialized parent replay pass;
all non-Rome predictions stay unchanged. There are 2,839 conditional-copy
supports, four fixed-copy supports and 1,323 original fallbacks. Preserve
the endpoints, predictions and result under `v3_20261011_daily_structural_prior/`.
No submission-ledger entry, full-year expansion or upload follows this result.
The off-block-date screen has started next in the sequential queue.

A short recovery-source check does not justify another expansion: the cached
OPDI flight lists contain aircraft/callsign/route identities and first/last
observations, but no LOBT or other flight-plan off-block timestamp. The retained
LIRF/LTFM recovery pairs have 12 ambiguous departures in training and only one
in ranking; the existing rule already accepts 49 of the 50 ranking departures
with any candidate donor. Resolving ambiguity would therefore cover only one
additional test departure and would not address the broad missing-plan
population. No new fetch, match relaxation, model or flight-specific correction
was made.

## Date alignment does not improve the fixed posterior

The corrected-date screen finishes at **292.1302**, versus **291.9795** for its
matched IFR/takeoff-date control: **+0.1506 seconds**, interval
**[−0.8961, +1.3272]**. January changes −0.0469 and July +0.3482. Relative to
the historical all-departure posterior it worsens by 0.1252. No production
candidate is selected. The independent audit passes all source sums, all
344,419 assigned dates/scope/fallbacks, and monthly SQL scores. Only 24 incoming
dates disagree with the recorded dates; five posterior outputs cross their
assigned dates. Source alignment is real, but it has not demonstrated a model
gain under this likelihood and prior.

Residual analysis on these same development rows locates the regression in
long flights. July Rome long flights add **45.72 million squared seconds**;
July Istanbul long flights add 15.35 million, while Istanbul ordinary flights
remove 17.42 million. Wrong assigned dates account for only 0.017 million of
July's 34.30-million total increase. These are diagnostics for generating
hypotheses, not independent selection evidence or reasons for per-flight edits.

One fixed follow-up, `daily_heavy_tail.py`, tests **t5 ordinary-aggregate weights**
on the unchanged 41-node grid. Its scale is solved deterministically to match
the Gaussian grid's variance, using no outcomes. All source parameters, priors,
dates and fallbacks stay fixed. The hypothesis is that rare aggregate ordinary
errors need more probability than the Gaussian allows, so the daily constraint
may otherwise push ambiguous long flights into the wrong mode. No airport gate,
prior scale search or additional taxi model is introduced. Each scoped group
must replay the Gaussian control before scoring the new weights.

Numerical preflight passes equal variance, centering, Gaussian replay,
no-long/no-ordinary edges, permutation invariance and independent enumeration
of **8,036 states** (maximum difference **4.55e−11 seconds**). A separate SQL
stage checks labels, calendars, controls and scores. No CV result exists yet.
`heavy_tail_queue.json` (PID737069) waits for the selected neural production
queue (PID721659) and its audit, then runs the screen/audit sequentially.
It shares the research cancellation file and contains no upload stage.
Ordinary punctuality is the current desktop fit (PID734641).

## Fixed raw-track withholding screen

The next distinct neural experiment keeps the public physical network and
withholds its entire raw-track branch on a fixed25% of training rows. All
tabular/public inputs remain available; inference uses every observed track.
This tests whether the residual track branch over-specializes to its training
availability. It does not simulate the loss of all ADS-B-derived inputs or
specifically ground-only coverage. The original parameterization, encoder,
21/17 epochs, seed42, optimizer and batch order are retained. There is no
probability, epoch or inference-gate search.

`nn_track_dropout.py` uses a private per-device generator to preserve the
original dropout RNG stream. CPU checks pass identical initialization and
parameter keys, bit-exact inference, an independent masked-training replay,
global RNG preservation, finite gradients, zero track gradients when every
track is withheld, an active public head and padding invariance. CUDA checks
and both Jan/Jul fits remain pending. `nn_track_dropout_jan_jul.py` reuses the
original training bytecode in an isolated namespace and compares against the
audited public single. The selected three-member pool is a separate-size
diagnostic reference. Reload and independent SQL audits are queued too.

Run `v3_20261011_nn_track_dropout/` and its frozen identity are retained.
`track_dropout_queue.json` (PID746093) waits for the heavy-tail queue
(PID737069) and its audit. It shares the research cancellation file, runs
one heavy job at a time, and has no production or upload stage.

A live check at **10:47:40 Amsterdam on11 October** still reports best/latest
v74 **221.6881**, rank12/242; the target gap remains6.6881 seconds. January's
ordinary-punctuality fit has finished and July is running. V75 remains queued
for the next UTC upload day.

## Public-source follow-up

Read-only checks of recent public READMEs found no new compelling method to
add ahead of the queued screens. [Unbelievable-quartz](https://github.com/ToukoUrsin/prc2026-taxi-out)
uses direct/residual tree blending and reports389.12 seconds;
[quirky-honey](https://github.com/dhsdb010/OPDI) and
[gentle-octopus](https://github.com/joris-decombe/taxiout) describe clock mixtures,
ADS-B correction and missing-plan handling already explored here. Their
reported best scores are269.33 and271.9 seconds, respectively; these are
their published claims, not a new live-score check. The recent
[zestful-fountain](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout) README
still centers on following-clock features and longer LightGBM training,
which our retained feature/HPO work already addresses. No competitor source
code was copied or executed in this follow-up. Local read-only snapshots
are under `tmp/research_20261011/public_sources/`.

## Monthly reference-population follow-up

The [official methodology](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf)
excludes flights without a usable stand/runway reference, alongside other
filters. A December2025 descriptive check applies its reference-count rule
to the supplied year's labels. Compared with the basic eligible mean, this
reduces the discrepancy from the published mean at Rome from13.31 to−0.93
seconds, Barcelona4.71 to−0.22 and Istanbul14.88 to−2.04. Counts still differ,
and winter gaps remain large at airports where deicing exclusions cannot be
reproduced. This is source-population diagnosis, not model validation.
Results and input hashes are in
`research_network_20261010/monthly_population_probe_20261011/`; runner
`tools/probe_monthly_reference_population.py`. It uses no hidden labels.

One fixed experiment, `monthly_reference_bridge.py`, approximates reference
membership using stand/runway frequency in the allowed training and held-input
months. Within each airport-month it weights the most frequent cells until
their mass matches the source's covered fraction; tied frequencies receive
equal fractional weights. It calibrates the weighted predicted mean against
the source. The original half-strength shift,120-second raw cap,5% invalid
share gate and training-only composition regression stay unchanged. There
is no coverage threshold or blend-weight search. The reference mask remains
an approximation; no exact source identity is claimed.

Preflight passes16 independent allocation-oracle cases, requested mass,
fractional ties, row-order invariance, a hand-computed weighted mean,
full-coverage identity and label isolation of input frequencies. The screen
must replay v74 exactly, then score both the replacement monthly stage and
its transfer on prepared v75, followed by independent SQL label/control/score
checks. No CV score or production selection exists yet. Run:
`v3_20261011_monthly_reference_bridge/`. Queue
`monthly_reference_queue.json` (PID755899) waits for track-withholding
PID746093 and its audit, uses the same cancellation file, and has no upload.

## Combined public models pass; ordinary punctuality does not

The fixed joint calculation passes with transferred CV **292.4494**, versus
v75 **292.9415** (−0.4921, paired-day interval **[−0.6646,−0.3504]**).
It also beats the public main tree alone by0.3213 and the public neural pool
alone by0.1386; both intervals exclude zero and both months improve in every
comparison. The combined January/July scores are326.1741/258.7247. Primary
untransferred CV is291.7234. Exact all-row replays of all three controls and
the independent SQL audit pass. This remains reused development evidence,
not a new official score.

Run `v3_20261011_public_stack_cv/` retains the result and audit. Full-year
public neural production has started on the desktop under PID759761,
queuePID721659. After its three members and production audit finish, the
combined candidate can be built with `public_neural_candidate.py --with-main`.
No final file or new version is reserved yet; the selected main-tree model
is already complete and will be reused.

Ordinary punctuality finishes at **291.9869**, versus its public-main/Cat
control291.9697: **+0.0172**. The transfer changes292.7707 →292.7843,
**+0.0137**, interval[−0.0021,+0.0272]. Both months worsen. No production
expansion is selected. All344,419 saved predictions reload exactly during
the fits; `tools/audit_punctuality_main.py` separately verifies their receipts,
hashes, all truth/calendar/control values and SQL scores.

That new auditor initially failed because DuckDB requires explicit `AS` for
the alias `value`. This is corrected. The same syntax appeared in the three
waiting experiment auditors, so their queues were stopped before any child
started, their old identities/queue records retained with `_before_sql_alias`,
and their guards rerun. No numerical/training function changed and no model
was repeated. Current downstream PIDs: **761632 heavy-tail →761633 track
withholding →761634 monthly reference**. The active neural production and
V75 upload queues were not changed.
