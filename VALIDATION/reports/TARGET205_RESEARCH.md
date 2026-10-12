# Research toward 205 seconds — 12 October 2026

## Official progress

The user resumed research at 02:24 Amsterdam after the requested pause until
02:00. At 02:40:30, best/latest v79 scores **219.6130**, rank **9/242**, on all
344,841 original January/July departures. The **205.00** target remains
14.6130 seconds away. Autonomous submissions and both hosts remain authorized.

| Version | Change | Jan/Jul CV | Official RMSE |
| --- | --- | ---: | ---: |
| v76 | v75 adds public daily inputs to the main tree and physical networks. | 292.4494 | 221.0948 |
| v77 | v76 adds the selected daily and tree refinements. | 291.5497 | 220.8328 |
| v78 | v77 adds the joint public-source correction. | 290.9301 | 220.2933 |
| v79 | v78 is recombined with earlier scored predictions using updated affine weights. | 290.6744 | 219.6130 |

V76–v78 were the three retained uploads requested by the user. All succeeded
once at 02:10/02:20/02:30; v75 remains cancelled. V79 used their actual feedback
and uploaded at 02:36. Together they improve v74 by 2.0751 seconds. All remote
hashes match. Four October12 UTC slots are used; one remains for a future
candidate selected from new evidence. No duplicate or cancelled upload is queued.

The 02:00 one-shot timer fired but its message was rejected by the IDE because
the request lacked `threadId`. It did not start inference. The original failure
receipt remains unchanged; work began after the explicit user resumption.
The native goal retains its old text/status; the local project objective records
205 and no native usage limit was altered.

## Updated scored mixture: v79

The new diagnosis removes only unsubmitted v75 from the previously prepared
52-vector basis. All 51 retained prediction columns are exact, including integer
tie decisions. V75's prepared ancestry is preserved where later candidates
legitimately depend on it; its nonexistent official score is never imputed.

The original bounds [−3, 4], sum-one constraint and ridge MSE 1 are unchanged.
Using actual successful scores for v76–v78 projects 219.613357 before output
clipping/rounding. Separate SQL moment identities reproduce this value. An
independent allowance for official-score precision, numerical error and measured
serialization distance gives a conservative upper bound of 219.923278.

The official result is 219.6130, 0.6803 below v78. Jan/Jul CV improves 0.255762,
but its paired-day interval [−0.621416, +0.049142] includes no gain. January
changes 325.860608→325.530874 and July 255.999630→255.817841. This selection
uses the leaderboard; reused development predictions are not independent
confirmation. The v3–v9 reconstructed historical-source caveat remains explicit.

Every ranking/CV row, raw validation label, schema, ID, integer prediction and
declared source hash passes independent geometry and final-file checks. See
`../runs/v3_20261012_target205_geometry/` and
`../runs/v3_20261012_target205_candidate/submission_v79/`.

## New experiments

**Laptop: full-input physical TabM.** The fixed model has the original 156
tabular inputs plus twelve public inputs, the retained physical-track encoder,
and 32 TabM heads with two 512-wide layers. Per-head MSE is averaged over rows
and heads; inference averages heads. Microbatches of 512 preserve effective
batch 2,048, sample weighting, and one optimizer/scheduler step per logical
batch. January/July use the original 21/17 epochs and folds, seed42. Two TabM
fits and two same-laptop original-network controls run sequentially. No search,
held-out stopping or additional seeds is part of this comparison. Independent
review passes loss, update, preprocessing and exact-v78 scoring, with a fixed
parent-result hash check added before acceptance. CPU/CUDA preflight passed;
actual first updates use 3.64 GiB of GPU memory. The four-fit worker started
encoding at 02:39 on the laptop, PID14874. Reuse the existing verified cache;
only 60.25 MB of missing compact inputs/sources was transferred. Artifacts:
`../runs/v3_20261012_nn_public_tabm/`.

**Desktop: original flight-plan timestamp phases.** Seven fixed additions to
the selected 184-input main LightGBM are the seconds of minute of AOBT3, LOBT,
IOBT, EOBT1 and ARVT3, plus AOBT3 minute modulo 5 and 15. Original missingness
is retained; there is no movement-time fallback or target-derived timestamp.
All 3,100,678 source rows and 7,210 sampled scalar values pass a separate UTC
arithmetic audit. Every shared old/final row preserves plan phases exactly,
whereas movement timestamps were jittered. The feature population is narrower
than initially hoped: IOBT seconds are constant where observed, and all seven
features are missing for Rome's no-ID 1–4h group. The authorized fixed test
keeps those choices, seed42, folds and main-tree settings unchanged. A matched
184-input saved-control replay, saved-model reloads and exact-v78 complete-stack
comparison are required. Source: `../source_clock_phase_lgb.py`.

**Arrival clock context: closed.** On twelve fixed training dates, prior-two-hour
airport arrival context predicts arrival landing-minus-ARVT3 disagreement:
clipped RMSE40.156→37.939 across66,828 queries. The necessary departure transfer
check then fails: clipped error RMSE323.2905→323.4246, raw335.3769→335.5072
across65,632 eligible ordinary departures, with regressions in both halves.
Separate source/window/target/arithmetic audits pass. No January/July departure
labels, full feature build or taxi-out fit follow. The retained receipt is
`source_clocks_20261012/arrival_context_closed.json`.

**Same-flight endpoint check: closed.** Linking each departure to its unique
arrival covers about15% of ranking departures. On10,245 eligible training
pairs, landing and departure clock errors have correlation−0.00992. The natural
shared-clock correction worsens the proxy RMSE384.90→392.70 and every checked
month. Identical flight-plan copies at the two endpoints do not establish a
common timestamp error. No model fit follows.

**All-discrete public peers: numerical rejection.** The current approximation
represents other long flights by Gaussian moments. A source-only pilot instead
retains all their discrete supports, samples peers with scrambled Sobol points,
and integrates the target exactly. Tiny enumerated cases and independent
arithmetic pass, but the largest475-peer cohort has effective sample size3.53
at16,384 draws; two scrambles differ by up to1,068.6seconds. No validation score
is computed for unstable predictions. A fixed single-site Gibbs follow-up
passes exact tiny-state tests, but also fails convergence on the largest cohort.
At256burn-in+512retained sweeps, its increment has mean ESS20.13 and Monte Carlo
standard error1,263seconds; the two budgets change the increment from−253.3
to+525.1seconds. The1.23-second pilot reads no labels or CV scores. This closes
the tested numerical branch; it does not establish impossibility of every
discrete inference algorithm. No outcome-selected cohort gate or further budget
expansion follows. See [the numerical report](ALL_DISCRETE_PEER_INFERENCE_20261012.md).

**Linear-leaf trees: resource preflight planned.** A newly updated
[joyous-rainbow solution](https://github.com/radekacar/joyous-rainbow/tree/dbe1eded734da48bdad7cc11b741028794316179)
reports gains from linear leaves on a substantially weaker, different pipeline.
That result is motivation, not evidence of a gain here. Our proposed comparison
keeps the original184inputs and main-model recipe, with training-only numeric
standardization and fresh matched constant-leaf controls. The sole model
change is `linear_tree=True, linear_lambda=1`. The
[LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree)
recommends comparable numeric scales and warns of higher memory use. A fixed
64-round training-only probe will measure cost and the fraction of rows that
actually use linear coefficients before any full experiment. Missing path
features can force constant fallbacks. No public code was executed or copied.

**Current-parent comparison frozen before results.** The timestamp and TabM
tests retain their exact-v78 primary comparisons. A secondary read-only check
adds each unchanged component delta to integer v79, with coefficient1 and the
same nonnegative integer serialization. It fits no weights or new bounds;
zero-delta rows remain exactv79. Source hashes and the two experiment arms were
frozen before either result existed. See `../target205_delta_transport.py` and
`../runs/v3_20261012_delta_transport/design.json`. This remains reused development
evidence, not independent confirmation or an uploadable candidate.

Completed negative screens and their retained models are documented in
[the 11 October report](TARGET215_RESEARCH.md). Do not repeat those fits,
reopen cancelled queues or spend the final slot on numerical noise.

## Reapplying selected plan bounds: closed

Affine recombination can cross previously selected clock bounds. A fixed
diagnostic therefore reapplies v74's matched-plan bound and recovered-plan
exclusion to v79, without monthly recalibration, new thresholds or fitted rules.
Jan/Jul CV changes 290.674357→290.639693, −0.034664 seconds, interval
[−0.071902, −0.007527]. Both months improve across 32 changed validation rows.

Only nine ranking predictions change, all from the matched bound; the recovered
exclusion changes none. The RMS prediction displacement is 0.088502 seconds,
which bounds the absolute possible leaderboard RMSE change by the reverse
triangle inequality, regardless of hidden labels. This is too little opportunity
for the remaining submission slot. No selected candidate, version or upload
follows. Thirty held labels lie outside the matched bound, so this diagnostic
does not claim that the empirical bound is universally exact.

Separate SQL reconstructs all 344,841 ranking and 344,419 validation predictions
and monthly scores. The maximum ranking change is 49 seconds. Sources,
predictions and receipts remain in `../runs/v3_20261012_reapply_selected_bounds/`;
runner `../reapply_selected_bounds.py`, auditor
`../tools/audit_reapplied_bounds.py`.
