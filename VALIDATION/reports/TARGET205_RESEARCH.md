# Research toward 205 seconds — 12 October 2026

## Official progress

The user resumed research at 02:24 Amsterdam after the requested pause until
02:00. At the12:27:42 live check, best/latest v80 scores **218.8525**, rank **9/242**, on all
344,841 original January/July departures. The **205.00** target remains
13.8525 seconds away. Autonomous submissions and both hosts remain authorized.

| Version | Change | Jan/Jul CV | Official RMSE |
| --- | --- | ---: | ---: |
| v76 | v75 adds public daily inputs to the main tree and physical networks. | 292.4494 | 221.0948 |
| v77 | v76 adds the selected daily and tree refinements. | 291.5497 | 220.8328 |
| v78 | v77 adds the joint public-source correction. | 290.9301 | 220.2933 |
| v79 | v78 is recombined with earlier scored predictions using updated affine weights. | 290.6744 | 219.6130 |
| v80 | v79 uses the selected TabM and runway update. | 289.7628 | 218.8525 |

V76–v78 were the three retained uploads requested by the user. All succeeded
once at 02:10/02:20/02:30; v75 remains cancelled. V79 used their actual feedback
and uploaded at 02:36. Together they improve v74 by 2.0751 seconds. V80 uploaded at12:23:58 and scored218.8525, gaining0.7605 more. All remote
hashes match. All five October12 UTC slots are used. No duplicate or cancelled upload is queued.

The 02:00 one-shot timer fired but its message was rejected by the IDE because
the request lacked `threadId`. It did not start inference. The original failure
receipt remains unchanged; work began after the explicit user resumption.
The native goal retains its old text/status; the local project objective records
205 and no native usage limit was altered.

## V80 official breakthrough,12:27 Amsterdam

All three full-year TabM members and the robust same-date runway component
passed the production/reload/independent SQL audits. V80 uploaded once at
12:23:58 and scored **218.8525**, improving v79 by **0.7605 seconds**. The total
October12 gain versus v74 is2.8356. Latest and best agree, rank9/242,344841rows.
The final slot is consumed; no v81 or automatic upload is assigned. The two
additional production fits and seven-step completion worker are finished.
Retain every model/prediction and do not repeat them. The separate final route
now proceeds with saved-model inference; its four-month feature audit is done.

## Actual v80 feedback: fixed geometry closed

Appending the genuinely new v80 vector to the retained51-vector basis projects
218.619014, a0.233486-second gain. The unchanged quarter-second selection gate
fails even before serialization; the independent conservative upper bound is
219.022742. CV289.702019 changes−0.060759, interval[−0.298048,+0.196076].
No v79 rounding direction or cancelled v75 score was added. All52 official
scores, full vector joins, expanded SQL score identities and344419 CV rows pass.
No new candidate number, fit, queue or upload follows. See
`../runs/v3_20261012_tabm_feedback_geometry/result.json`.

## Annealed particle inference: source-only screen closed

The new method anneals the unchanged t5 likelihood with adaptive0.8N-ESS
steps, systematic resampling and two exact tempered Gibbs sweeps per step.
Unlike product prior sampling, its particle states retain induced peer dependence.
The same frozen7/69/475-peer cohorts use four512/2048-particle replicates,
maximum128 temperatures and a240-second total cap; no departure labels are read.
Finite-state detailed balance, deterministic-sweep invariance, explicit compiled
transition/Rao–Blackwell parity and incremental importance identities pass.

The pilot finishes in84.81seconds. Five of six cohort/view comparisons pass.
The largest joint estimate is2171.9910 with replicate SE35.9666; the largest
departure-only estimate is2087.4464 with SE39.3191, narrowly above its fixed
39.0939 threshold. Thus the numerical gate **fails**. Better numerical behavior
than the earlier sampler is not a predictive improvement. No threshold change,
extra particles, broader cohort screen, CV score, production or candidate follows.
An independent audit checks61,440 saved particle states, original source/kernel
identities, replicate/gate arithmetic and direct SciPy t5 likelihoods (maximum
error1.07e-14). All artifacts are retained in `../runs/v3_20261012_joint_public_smc_pilot/`.

## Longer TabM training: closed after matched comparison

The fixed24-epoch comparison completed and passed the independent model/SQL audit.
With the selected runway combination, its CV is289.731327 versus289.732376 for
the same-trajectory21/17 control: **−0.001050 seconds**, interval[−0.023749,+0.019831].
January improves0.010182 and July worsens0.008083. Against the actual retained
three-TabM/runway pool, the change is−0.031451, interval[−0.072228,+0.011807],
also inconclusive. The analogous raw v78 comparisons are inconclusive too.
The fixed gate fails. No epoch search, extra seeds, production fit, candidate
or upload follows; both trajectories and intermediate weights are retained.
See `../runs/v3_20261012_nn_public_tabm_duration/closed.json`.

## Fixed TabM duration screen and research state, 11:33 Amsterdam

The selected three-member production pool is running on the laptop. The desktop
runs one fixed alternative: seed42 completes the original24-epoch OneCycle
schedule in both folds. Checkpoints at21/17epochs provide same-trajectory controls;
held predictions are computed only after training ends. Architecture, loss,
optimizer, inputs, other members and all correction weights stay unchanged.
Exact training-AST restoration, CPU update/RNG parity, desktop/laptop input
fingerprints and GPU capacity checks pass. This is not a stopping-epoch search.
Both the actual selected pool and matched-duration comparisons must improve
in both months with paired-day upper endpoints below zero. Results are pending.

The persistent index is `../research_registry.json`; `../research_state.json`
records the two active jobs and the last official score. Run
`tools/research_state.py status --remote` for a compact refresh. A finite worker
will return verified laptop artifacts, build and independently audit the selected
combination after the desktop becomes free. It never fits or uploads.

The updated descriptive error table confirms the known concentration: the largest
10 CV residuals contribute44.25% of equal-month squared error; the two previously
investigated January LFPG rows still dominate. This is not evidence of the same
errors in2026, and no new recording rule, clipping or weighting follows. Full
input-group and airport tables are in `current_error_concentration_20261012/`.

A public-repository refresh found no selected new technique. It did establish
that the [official final route](https://prc-data-challenge-2026.netlify.app/ranking.html#final-submission)
requires `unique-rose_final.parquet`, separate from versioned leaderboard uploads.
The earlier v63 rejection does not test that separately named route. Preserve
old/final input namespaces and prepare an audited four-month final file before
the deadline, after the active leaderboard production work. No final upload or
new final-route acceptance is claimed. See `public_refresh_20261012.json`.

The separate final public-input table is now prepared. Its 670,790 rows retain
all twelve selected formulas; every old-view value replays exactly. Independent
SQL checks all 8,049,480 final feature values, maximum absolute error 4.66e-10.
No label values, training or upload are used. `final_public_tabm_inference.py`
is prepared for inference with saved weights/encoders on the original laptop;
its execution will wait until the active production and artifact return finish.
This does not yet provide a complete final model stack.

## Three-TabM breakthrough and cross-date closure, 10:50 Amsterdam

The four extra-seed CV fits completed on the laptop; the original seed42 is
unchanged. The compact return contains17 missing files/38,146,554bytes, with
remote and local SHA-256 agreement. All six model records, original controls,
344,419 rows and SQL scores pass the independent audit. The equal three-TabM
pool improves the one-TabM pool by0.370856 on v78 and0.464664 on v79.
The frozen three-TabM/runway combination scores **289.762778**, a **0.911579**
improvement versus v79, interval[−1.322560,−0.646076]. Versus the already audited
one-TabM/runway combination, it improves0.466063, interval[−0.688582,−0.300130].
Adding the retained runway correction to three TabM alone improves0.156312,
interval[−0.310605,−0.036095]. These are fixed comparisons on reused development
data; the original PLE execution difference remains a caveat, not an independent
confirmation. No official gain or v80 upload is claimed.

The conditional production gates therefore pass. Full-year seeds1042/2042
started10:36 on the laptop,15epochs each; completed seed42 is retained exactly.
Before starting,137 checkout collisions were backed up with verified hashes;
136 scientific files were byte-identical to incoming main, while an obsolete
machine-specific queue differed and remains in the backup. The laptop safely
fast-forwarded to856c2af, and only missing production receipts were transferred.
No model/cache was overwritten. Do not pull or synchronize sources during fits.
The compact return collector and three-member stack/combination auditors are
prepared; the previously audited one-member file remains a fallback.

A distinct cross-date source extension is complete and closed. Unlike the
selected same-date correction, it adds all eligible other-date taxi cells
sharing flights with the unchanged assigned-day departure cohort. A separate
15-second source-error loading is used for each new date; shared-flight moments
supply cross-date dependence. No date is reassigned. The old same-date block
replays exactly, then a separate capped t5-score Schur update conditions the
added block. This remains a hybrid moment approximation with uncalibrated
cross-date error independence. Twelve preselected source-only cohorts pass
explicit-flight covariance reconstruction and direct block-solve checks.

All2336 existing CV targets retain original eligibility;2277 have added cells.
The selected old floating posterior replays within2.91e−11seconds and integers
replay exactly. The new correction versus same-date changes v78 **+0.006944**,
interval[−0.062216,+0.085938], and v79 **+0.008620**, interval[−0.064950,+0.096534].
January worsens in both comparisons. The fixed gate fails; the three-TabM
combination is not run. An independent direct-conditioning/SciPy-density audit
checks166 posteriors, maximum error1.64e−11, plus all344,419 SQL rows and scores.
No production, version or upload follows. Sources and receipts:
`../joint_public_cross_date_cv.py`, `../runs/v3_20261012_joint_public_cross_date/`.

## Completed production and follow-ups, 12 October morning

The one-member TabM full-year fit completed15 epochs at05:30 and reloaded all
344,841 ranking predictions exactly. After verified artifact return, its stack
and the robust other-runway stack were built sequentially on the desktop.
The fixed combined file changes313,110 predictions and passes independent
integer SQL, protected-column, identity, hash and Jan/Jul arithmetic checks.
Its CV remains290.228841 (−0.445516 versusv79); no official gain is claimed.
No version/upload is assigned while the additional-seed comparison runs.
Four reviewed/frozen TabM CV fits for1042/2042 started on the laptop at07:44;
seed42 is reused exactly. Two further production fits depend on audited fixed-comparison gains.

Before the first extra seed completed, `tabm_pool3_other_runway_combo_cv.py`
froze a coefficient-one combination using original v78 integer component
deltas and a single clip on v79. Its primary comparison is three-TabM/runway
versus the already audited one-TabM/runway combination. No score exists yet.
`nn_public_tabm_pool3_submit.py` prepares two full-year continuations only if
the independent pool audit and both three-versus-one comparisons pass. Its
isolated member namespaces retain the original fit/helper bytecode and the
reviewed seed specialization; no further production fit has begun.

The two source-only pilots are complete and closed. The movement-ID arrival
clock obtains no valid bracketing anchors for any of339 intended no-ID1–4h
Rome rows. Independent scalar/SQL audits reproduce all available clocks and
coverage. No threshold/calendar expansion or model follows.

The exact target/site pair-Gibbs kernel passes finite-state detailed balance,
full-sweep invariance, compiled-versus-Python checks and24 real-data residual
reconstructions (maximum error1.16e−11). It improves the target's mixing, but
largest-cohort long-budget increment MCSE is111.7143seconds, exceeding the
fixed25-second limit. No extra sweeps, chains, CV or production follows.
Closure reports are in `all_discrete_peer_20261012/`.

The Rome TabM screen is complete and closed as inconclusive. Four inner/full
pairs preserve original R20 rows,75 history-free inputs and identical quality
fields. The physical arm changes v78−0.159950 (interval[−0.523327,+0.186250])
and v79−0.199983 (interval[−0.556818,+0.132167]); the tabular arm changes
v78−0.084600 and v79−0.133567, also inconclusive. Physical versus tabular is
−0.075350, interval[−0.241696,+0.074012]. All8 models, full saved-model reloads,
original labels/memberships, inner stopping and344,419 scored rows pass the
separate audit. No extra seeds, production or upload follows. The preflight's
pilot diagnostic was corrected to the original>1h definition before training;
the separate auditor's path-helper adaptation was fixed before any receipt.
See `rome_tabm_tracks_20261012/{audit,closed}.json`.

A new arrival-duration lead first tested stand drift on March/September/
October/November training months. For1369 stands, a slope fitted on the other
three months predicts the held month's airport-centered median change from
arrival changes. Training contrasts exclude the held departure month entirely.
Every month worsens (weighted cell MSE+77.47,+35.19,+18.07,+12.58); three
fitted slopes are negative. Independent SQL window medians and scalar isolated
references reproduce the result. This closes the proposed direct drift
transfer, without a multi-task fit or any Jan/Jul departure-label use.

A separate static-context screen is complete: same-month stand medians and
counts of other departures' supplied takeoff-minus-AOBT_3 clocks and supplied
arrival taxi-in. It encodes typical stand conditions rather than transferring
month-to-month arrival changes. It can populate stand context when a stand
category is absent from the departure model's training. Related within-frame
stand features appear in [gracious-tornado's fs8 description](https://github.com/il-miscusi/prc-2026-gracious-tornado/blob/main/docs/METHOD.md);
our implementation and model test are independent, and no gain is inherited.

The four inputs use actual UTC months and separate original training/ranking
views. Valid clocks/durations are1..7200s; null/blank stand gets no group. The
query's clock is removed using exact odd/even order statistics, including ties,
and counts use log1p. No departure target or block timestamp enters features.
All2,085,047 training and344,841 ranking departures are retained;1200 raw-source
scalar reconstructions and synthetic edge cases pass. The main-tree screen
keeps original184 columns, labels, rows, settings and10,000 rounds; exactv78
scoring and coefficient-one v79 transport were fixed before fits. Both models
and344,419 rows pass independent audits. The result is effectively flat:
v78−0.000022 (interval[−0.017399,+0.015150]), v79+0.003370
(interval[−0.013154,+0.018274]). No production, combination or upload follows.
Sources: `../stand_month_context.py`, `../stand_month_context_lgb.py`.

The updated51-score robust affine minimax check uses actualv76–v78 scores,
excluding cancelledv75. It offers only0.00587018seconds projected gain;
its serialized upper bound219.91644 is worse than the actual incumbent.
Independent SQL/geometry checks pass; no candidate/version/upload follows.

A distinct fixed read-only comparison is complete: distribute the existing
score-difference constraints using nonuniform row uncertainty. The reference
is v78, before score-based recombination. Variance across seven recent recipes
sets a bounded diagonal weight; its normalization uses ranking predictions
only. A robust convex solve retains the original score precision and ridge1.
The same frozen coefficients and ranking normalization then apply to Jan/Jul.
The primary checks compare with v79 and an otherwise identical equal-variance
solve. This creates directions outside the affine span, so its leaderboard
RMSE cannot be derived from the available scores. CV changes290.674357→
290.630277 (−0.044080), interval[−0.291878,+0.212033]; January is worse
by0.131390. The equal-variance comparison is likewise inconclusive. The
predeclared gate fails, so no combination, candidate or upload follows.
An initial coordinate solver failed its KKT tolerance before opening validation
labels; retained evidence records its0.00022986 residual. Active-set linear
refinement solves the unchanged objective. A separate Fenchel-dual optimizer
confirms both optima, and all-row predictions and SQL scores pass. Source:
`../heteroscedastic_score_conditioning.py`. This closes the one tested
disagreement-variance formula; it does not test every covariance prior.

The final linear-leaf result is inconclusive: matched standardized control
−0.051480, actualv78−0.032067 and actualv79−0.023679. All four completed
models and all344,419 scored rows pass independent saved-model/provenance/
SQL audits. Retain the models without production expansion.

The fractional punctuality pilot uses52,681 departures across122 permitted
training dates. Leave-month-out median-bias calibration gives fraction
RMSE0.023215; monthly RMSE ranges0.014312–0.034676. Typical declared prior
SD on the separate Jan/Jul input dates is0.007937/0.010141, so source error is
1.41–4.37 times as large. The arrival-source witness correlation is−0.118906.
All source fractions, allowed departure/arrival daily summaries, calibration
and correlations pass independent SQL/arithmetic checks. This closes the
proposed source likelihood without posterior scoring, calendar/threshold
search or CV. It does not prove every punctuality method impossible.

Flight-ID ordering passes its input-only chronology study: median fixed-date/
airport Spearman with IOBT isabout0.99996 in each of the four sampled months.
An independent audit rebuilds all264 group/clock comparisons. This does not
establish a creation timestamp. A predeclared nearest-peer interpolation
follow-up on64,892 fixed-date departures finds mostly redundant clock values,
but39–152 rows per month differ by over300seconds. No missing IOBT is recovered
in that sample. A fixed model screen adds only interpolated-peer-minus-own-IOBT
and signed bracketing IOBT span to the selected184-input main tree.

Both peer IDs must be strictly outside the query's own ID and lie within its
actual UTC month. The first implementation's coverage check found two
training departures stored in the next file, three seconds before the UTC
month boundary. The original cache and failed preflight are retained; grouping
now spans every original training file before selecting UTC months. No model
had been fitted. Two synthetic SQL tests and1,400 independent scalar peer/
interpolation checks pass. Original labels, rows, tree recipe and final stack
are unchanged; the separatev79 transport is fixed before scores. The completed
screen is closed: v78+0.00265234 (interval[−0.014806,+0.017233]),
v79+0.00741884 (interval[−0.008703,+0.021534]). Both10,000-round models,
all held reloads and independent SQL scores pass; no production or upload.

The three subagents stopped at the account usage limit. Root continues using
their completed reviews and artifacts. The user requested a brief local note;
`LOCAL_PROGRESS.md` is gitignored and never committed. GitHub `Goxmen/PRC`
was verified PRIVATE, and the current user instruction supersedes automatic
publication. Historical work descriptions below retain their original timing.

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

**Full-input physical TabM: positive fixed comparison, production preparation.** The fixed model has the original 156
tabular inputs plus twelve public inputs, the retained physical-track encoder,
and 32 TabM heads with two 512-wide layers. Per-head MSE is averaged over rows
and heads; inference averages heads. Microbatches of 512 preserve effective
batch 2,048, sample weighting, and one optimizer/scheduler step per logical
batch. January/July use the original 21/17 epochs and folds, seed42. Two TabM
fits and two same-laptop original-network controls run sequentially. No search,
held-out stopping or additional seeds is part of this comparison. Independent
review passes loss, update, preprocessing and exact-v78 scoring, with a fixed
parent-result hash check added before acceptance. CPU/CUDA preflight passed;
actual first updates use 3.64 GiB of GPU memory. The four-fit worker ran on the
laptop from 02:39 to 04:24, with 98.4 minutes of epoch training. Only 60.25 MB
of missing compact inputs/sources was transferred; the existing cache was
reused. All four saved models reproduce every held prediction. Replacing one
of the three physical members improves v78 CV 290.930119→290.677255,
−0.252864 seconds, interval [−0.367777, −0.173798]. The fixed v79 transport
improves 290.674357→290.383754, −0.290603 seconds, interval
[−0.421282, −0.206950]; January improves 0.172101 and July 0.409105. The
same-laptop single-family comparison improves 1.131927 seconds. Historical
desktop versus matched laptop control differs by +0.259518 seconds as a single
member, so that execution difference is kept separate from the model comparison.

All 344,419 scored rows pass the independent SQL, identity, truth, calendar,
parent replay and transport audit. The laptop training-input hashes pass.
The desktop training guard fails because a frozen laptop-specific cache path
is absent locally; it was not patched or claimed to pass. Returned artifacts
and semantic fingerprints are verified. These are single-seed results on
reused development data, not a demonstrated official gain. One seed-42
full-year replacement started on the laptop at04:48 Amsterdam, PID17027,
retaining the other two physical members. It uses15epochs on the original
24-epoch schedule. Independent source review, exact CPU update/RNG parity and
full-width GPU preflight pass; peak GPU allocation is4.80GB. All157 cached
dependencies are verified; only10.94MB of missing files was transferred.
The runner checks old-view template and all three parent prediction identities
before training, saves completed weights before inference and requires exact
all-row checkpoint reload. No production upload has occurred. See
`PUBLIC_TABM_RESULT_20261012.json` and `../runs/v3_20261012_nn_public_tabm/`.

**Original flight-plan timestamp phases: closed.** Seven fixed additions to
the selected 184-input main LightGBM are the seconds of minute of AOBT3, LOBT,
IOBT, EOBT1 and ARVT3, plus AOBT3 minute modulo 5 and 15. Original missingness
is retained; there is no movement-time fallback or target-derived timestamp.
All 3,100,678 source rows and 7,210 sampled scalar values pass a separate UTC
arithmetic audit. Every shared old/final row preserves plan phases exactly,
whereas movement timestamps were jittered. The feature population is narrower
than initially hoped: IOBT seconds are constant where observed, and all seven
features are missing for Rome's no-ID 1–4h group. The authorized fixed test
keeps those choices, seed42, folds and main-tree settings unchanged. Both fits
are complete. The184-input control replays, both candidate model reloads and
all344,419 final arithmetic/truth/score rows pass their independent checks.
Exact-v78 CV changes290.930119→290.942507, +0.012388seconds, interval
[−0.004784,+0.027121], with both months worse. The predeclared v79 transport
also worsens290.674357→290.691960, +0.017603seconds, interval
[+0.000630,+0.032125]. No seed expansion, production or submission follows.
Source: `../source_clock_phase_lgb.py`; retained closure
`source_clocks_20261012/phase_closed.json`.

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

A distinct damped expectation-propagation proposal also fails its frozen tiny
numerical test. It updates peer cavities while preserving coupled count, taxi
sum and delay supports and the Student-t likelihood. Six fixed two-/three-peer
cases use exact enumeration and the current prior-moment method as baselines.
The one-thread test takes 23.76 seconds and 5.60 MiB of additional memory:
185 of 288 configurations complete, 103 encounter invalid cavities, and none
meet the accuracy criteria. Numerical convergence occurs in 177 configurations
and is insufficient: the cutoff example can remain biased by about 1,420 seconds.
The large-source cases also have unresolved initial quadrature errors, so their
large errors cannot be attributed solely to the EP approximation. Independent
enumeration and saved-error arithmetic pass. This closes the tested design,
with no real-cohort inference, held score, tuning or upload. Retained evidence:
`all_discrete_peer_20261012/ep_toy_closure.json`.

**Other-runway observations: positive robust comparison, production preparation.** This adds
information rather than changing the discrete inference solver. The selected
joint message sees only the query's own runway taxi count/sum and its assigned
day's airport departure-delay total. Other runway totals can constrain peers
inside that same delay total. Earlier calendar and discrete-peer tests do not
add these observations.

A frozen input-only pilot selects six existing cohorts per month before reading
the added-source residuals. It runs in 3.07 seconds with 51.4 MiB additional
memory. Nine of twelve cohorts have an eligible extra runway, covering 155 of
166 sampled targets. Median target-weighted delay-variance contraction is
6.33% in January and 22.36% in July under the proposed Gaussian conditioning.
These are sampled model-information diagnostics, not measured prediction gains.
Every original three-coordinate covariance block, target subtraction and
no-extra/disconnected/positive-definite check passes.

The plain Gaussian proposal is rejected before CV: two added-source squared
Mahalanobis residuals are 313.47 and 283.82, and one induced standardized mean
shift has norm 9.28. Retaining the original Student likelihood afterwards would
not repair an already excessive shift. A single robust alternative is being
prepared, with weight `min(1, (5+k)/(3+q))`, where `k` is the added dimension and
`q` its original predictive Mahalanobis residual. This is the variance-matched
t5 score weight with a conservative cap, used for a Gaussian Schur update;
it is explicitly an approximation, not exact Student conditioning. The earlier
unexecuted `5+q` denominator was corrected during mathematical review before
freezing or any posterior calculation. No weight search is planned.

The weight can suppress useful evidence when the prior is wrong; one anomalous
cell also weakens the whole added set. Exact disconnected recovery applies to
an entirely zero-cross-covariance added set, not to appending disconnected
blocks alongside informative ones. Cross-runway reporting-error correlations
remain uncalibrated. The selected supports, source scopes, original t5 message,
half-delta transfer and final corrections remain fixed.

The frozen comparison completes inference in 24.91 seconds. Exact-v78 CV
improves 290.930119→290.793473, −0.136645 seconds, interval
[−0.281705, −0.020365]. The fixed v79 transport improves
290.674357→290.520194, −0.154163 seconds, interval [−0.306432, −0.035022],
with both months better. Extra evidence covers 1,600 targets and changes 1,179
serialized predictions. The cached integer v78 control replays exactly;
floating old-posterior replay differs by at most 2.91e−11 seconds, within the
1e−8 tolerance fixed before inference. Deltas use the saved old posterior.

All-row score/source checks and a separate six-target block-solve audit pass.
The latter independently assembles shared-flight covariance and support
likelihoods, agreeing within 1.14e−11 seconds. Production preparation reuses
the existing 26,899-row Rome ranking packet and full-year calibration, with
no downloads or taxi-model fits. The production runner and its independent
auditor pass review, awaiting a clean host for execution.

The separately frozen additive combination uses exactly
`rint(max(0,v79+(TabM_v78-v78)+(other_runway_v78-v78)))`.
All344,419 rows pass source, truth, calendar, formula and SQL score checks.
CV improves290.674357→290.228841, −0.445516 seconds, interval
[−0.655109,−0.288599]; January improves0.242175 and July0.648857.
It also improves over TabM alone by0.154913 and over the runway change alone
by0.291353. This is a post-selection combination on reused development data,
not independent confirmation. No production file or upload exists yet.
Evidence is under `all_discrete_peer_20261012/other_runway_*`.

**Linear-leaf trees: resource preflight passed; full comparison authorized.** A newly updated
[joyous-rainbow solution](https://github.com/radekacar/joyous-rainbow/tree/dbe1eded734da48bdad7cc11b741028794316179)
reports gains from linear leaves on a substantially weaker, different pipeline.
That result is motivation, not evidence of a gain here. Our proposed comparison
keeps the original184inputs and main-model recipe, with training-only numeric
standardization and fresh matched constant-leaf controls. The sole model
change is `linear_tree=True, linear_lambda=1`. The
[LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree)
recommends comparable numeric scales and warns of higher memory use. A fixed
64-round training-only sample probe takes24.49seconds including input assembly,
scaling and hash checks, with cumulative peak memory7.154GiB. On65,536 rows,
constant/linear arms take1.082/1.157seconds; late-tree costs differ by1.16×.
Linear coefficients apply without missing-feature fallback on62.806% of sampled
row/tree pairs, and outputs are finite. On all1,596,201 January training rows,
the64-round constant/linear probes take18.23/20.83seconds, with late-tree means
0.2043/0.2334seconds and cumulative peak memory7.159GiB. Outputs remain finite.
The first8,192 training rows have54.13% linear-active row/tree pairs; this is
not a representative full-year or test-set active-rate estimate.

The fixed four-fit comparison has passed independent review and is frozen: each month gets
a fresh standardized constant-leaf control and its linear-leaf counterpart,
with 10,000 rounds and the original settings. Expected total runtime is roughly
1.3–2.5 hours before allowing for full inference, with substantial uncertainty
from extrapolating early-tree costs. No held score was read in either probe.
Both full input/scaler views pass. January's constant control completed on the
desktop, taking 16.93 minutes for training and 3.41 minutes for full inference
and exact saved-model reload. Its 10,000-round model and all 153,706 held
predictions are retained. A reviewed operational option stops after each
requested number of newly saved model/prediction records, preserving fit order
and resume behavior. The desktop was released for the completed TabM audit
and the short other-runway comparison before the remaining three arms,
which resumed at04:35 Amsterdam under PID1369670.
Previous frozen designs and source
versions remain under `v3_20261012_linear_leaf_main/prefit_v1/`; all scientific
choices are unchanged and were refrozen before any fit.
No public code was executed or copied. See
`source_clocks_20261012/linear_leaf_cost.json` and
`source_clocks_20261012/linear_leaf_design.json`.

## Follow-ups prepared during the two active jobs

**Additional TabM members: frozen, awaiting the laptop.** Retain seed42 and
add only1042/2042, with unchanged21/17 epoch stops,24-epoch schedule,
preprocessing and update recipe. Compare the fixed equal three-TabM pool with
both the one-TabM/two-PLE pool and the original three-PLE pool, requiring exact
existing replays. A separate coefficient-one v79 transport is frozen before
outcomes. Source review and CPU update/RNG tests pass. Four CV fits are
authorized after the current production member completes and its artifacts
return with verified hashes. All new fits must use the same laptop/GPU/Torch
as seed42. Original controls remain desktop artifacts, an explicit execution
caveat. No new full-year fits or seed/weight selection are authorized yet.

**Public punctuality: exact-count interpretation is unsupported.** A fixed
JSON-only probe reads eight Rome records, days10/20 in January/July2025/2026.
Every fraction admits multiple denominators between1 and3000:5–69 per record.
An unverified integer-minute mean-delay assumption leaves no compatible
value for10July2025. An independent exact-rational audit reproduces all24,000
checks. No count is selected; no movement rows, departure labels or CV scores
are read. The fraction may still be a noisy observation of the proportion
with `gap-y<=900`, distinct from taxi count/sum and positive delay sum.
A source-noise pilot is being prepared for common permitted training months
March/September/October/November2025, with whole-month exclusions for scalar
bias calibration. It will compare noise with existing input-only prior OTP
uncertainty, retaining population/schedule/calendar mismatch and covariance
with existing source errors. Rational fractions do not establish binomial
precision. See `punctuality_counts_20261012/`.

**Relative flight-plan ID order: unverified input hypothesis.** The earlier
movement-ID clock probe failed. A new fixed-date pilot examines flight-plan
ID order against original IOBT/EOBT clocks, including missingness, duplicates,
format and shared endpoint consistency. These numeric IDs are not documented
as creation timestamps and are not equated with IFPLID. No absolute-ID
feature, labels or model fit is involved. Execution awaits a clean host.

**Tempered discrete inference: deferred without implementation.** Replica
exchange could address the single-site chain's mode mixing, but does not
establish adequate posterior precision. The retained error scale is itself
uncertain because mixing failed. Any future proposal needs whole-score Monte
Carlo error bounds and independent-chain stability; a10-second per-flight
tolerance is not required in principle. No implementation, pilot, held score
or parameter search follows. See
`all_discrete_peer_20261012/replica_exchange_assessment.json`.

**Current-parent comparison frozen before results.** The timestamp and TabM
tests retain their exact-v78 primary comparisons. A secondary read-only check
adds each unchanged component delta to integer v79, with coefficient1 and the
same nonnegative integer serialization. It fits no weights or new bounds;
zero-delta rows remain exactv79. Source hashes and the two experiment arms were
frozen before either result existed. See `../target205_delta_transport.py` and
`../runs/v3_20261012_delta_transport/design.json`. This remains reused development
evidence, not independent confirmation or an uploadable candidate.

**Cross-flight context: closed after an inconclusive comparison.**
The proposed encoder keeps each nearby flight's observed plan clock paired with
that same flight's physical witnesses and relative time/runway/stand geometry.
Existing peer-clock and sensor features summarize those sources separately;
earlier attention models attended to the query aircraft's own points. The
potential novelty is the joint donor tuple, not another missingness count.

A four-date, input-only pilot takes 6.57 seconds on one thread and reads no
labels, raw point tensors or model matrices. With 16 peers, paired observations
cover 56.3%/59.4% of January/July 2025 queries and 71.3%/68.5% in 2026. Among queries
with fewer than two own track points, coverage is only 9.9%/8.6% and 8.7%/5.2%:
airport-wide observation gaps remain. Between 85% and 95% of the ±30-minute
neighborhoods have more than 16 peers; their median size is 30–37. These four dates are descriptive
coverage evidence, not representative error estimates or validation gains.

The fixed comparison retains the original 168-input physical network,
training rows, seed 42 and 21/17 epochs, adding a compact context branch. Its
final head combines 64 pooled context values with the existing 256-dimensional
query representation, so interpretation can depend on the query's known regime.
This choice was made before freezing or reading held scores. The original
public-input branch remains unchanged. Independent checks establish permitted
sources, query/alias exclusion, training-only scaling, complete-tuple permutation
invariance, empty-context fallback and original initialization/RNG parity. A
fixture with identical marginal summaries but different donor pairings
distinguishes the intended new information.

The full bank contains 2,085,047 flights. An independent audit checks all source
identities and clocks, all 48,924,633 allowed neighbor links, 1,037 reconstructed
neighbor lists and 125,190 scalar source values. CUDA checks preserve the
original initial outputs, gradients and random state, including native batch
updates. The full-bank preflight peaks at 3.93 GiB, with 30–32 ms per update.
The two fixed fits ran from 03:40 to 03:58, taking 17 min 52 s and peaking at
8.22 GiB CPU memory. All held predictions reload exactly. Replacing one of the
three pool members changes v78 CV 290.930119→290.897368, −0.032751 seconds,
interval [−0.105121,+0.035596]. The matched single model changes −0.064089,
interval [−0.286420,+0.137226], with July worse by 0.03756 seconds. The fixed
v79 transport changes 290.674357→290.653242, −0.021115 seconds, interval
[−0.092287,+0.047115]. All 344,419 rows pass independent identity, truth,
calendar, control, serialization and score checks. These small differences
are inconclusive. No extra seeds, variants, production fits or upload follow.
Retain both models and predictions. See `COHORT_CONTEXT_RESULT_20261012.json`
and `cohort_context_20261012/`.

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


## Archived final-set components recovered,13:33 Amsterdam

v38 and v17–v21 have retained original boosters, verified by original hashes.
All344,841 old integer predictions reproduce exactly; all670,790 final rows
are now inferred. V17–v21 also pass full final saved-model reloads.
The overnight control is not an exact v12/v13/v15 substitute:2698/2698/2697
old integer rows differ, RMS0.252197/0.252197/0.251915seconds, maximum25.
Those mappings stop before final inference.

Retained component arrays reproduce v54/v57/v58/v64/v66/v68/v70 exactly on
the old set and provide final-view blends. The older base neural arrays came
from October8 refits; their recorded old-input float RMS difference was
2.39918e-6seconds. This is retained numerical provenance, not a new fit or gain.
V45/v46 old controls also reproduce exactly; final inference/reloads are running.
A separate SQL audit checks source hashes, every original integer, final ID
coverage, serialization and all four monthly populations. No complete affine
port or final upload is claimed. Missing v26–v44 originals were not found on
either PC or in git history; backup availability was requested.

The fixed deeper-TabM screen continues on the laptop. Its completion worker
waits for the desktop legacy audit, returns only missing artifacts, then runs
the frozen exact-v80 comparison and independent architecture/model/SQL/day
bootstrap audit. It records the result and gate in the registry/report; it
starts no fits or uploads. State: `../runs/v3_20261012_nn_public_tabm_depth3/completion/execution.json`.


## Fifteen archived final components audited

Completed at 2026-10-12T11:39:56.070518+00:00. Retained versions: v17, v18, v19, v20, v21, v38, v45, v46, v54, v57, v58, v64, v66, v68, v70.
Independent SQL checks cover15,234,465 old/final values, original-file equality, ID membership, integer serialization and four-month populations. V12/v13/v15 control substitutions remain excluded. The15 archived October8 base-network arrays also reconstruct their final mean exactly; their recorded old-input RMS difference of2.39918e-6seconds is independently reproduced and explicitly retained as refit provenance. No new fit, complete affine port, final candidate or upload follows. Receipt: `../runs/v3_20261012_final_legacy_audit/audit.json`.


## Runway-dependent public-source uncertainty: closed

The existing measurement model already scales taxi covariance by cohort size
and departure noise by count mismatch. A distinct fixed screen partially pools
each runway's taxi covariance with the existing global covariance, using100
pseudo-cells, while preserving Huber bias, three-MAD filtering and t5 likelihood.
Blocked inner-month source validation improves mean negative log likelihood
by0.385245/0.225097 in the two allowed training windows; both day intervals
exclude zero. This source-level improvement is not a taxi prediction gain.

Exact-v80 CV289.762778 →289.747967 changes−0.014811seconds, interval
[−0.035175,+0.009368]. January−0.008844 and July−0.020779 are both tiny.
The frozen gate fails; no tuning, production, new candidate or upload follows.
The existing native source component receives the fixed half posterior change,
original bounds and integer serialization; its integer delta is transported
once onto v80. No taxi model was refitted. Source covariance estimation and
replay of the original calibration coefficients are the only fitting operations.

Independent checks cover all810 source cells, inner-month/DST exclusions,
SQL covariance/shrinkage, SciPy source densities, every344,419 CV row, exact
v80/native controls and a separately reconstructed paired-day interval.
Covariance lookup is the only AST change in the two inherited inference
functions; original controls replay within2.91e-11seconds, and shuffled rows
with poisoned extraneous labels reproduce predictions. Preserve both runs:
`../runs/v3_20261012_public_runway_covariance_pilot/` and
`../runs/v3_20261012_public_runway_covariance_cv/`.
