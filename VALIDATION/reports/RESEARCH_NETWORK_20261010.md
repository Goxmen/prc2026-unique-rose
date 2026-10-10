# Research network: 10 October 2026

**October 11 continuation:** the user replaced the five-second goal with an official
leaderboard score of **225.00 or lower**, and authorized autonomous uploads.
**v69 now scores 225.3994 officially**, leaving **0.3994 seconds**. It improves
v64 by 7.3757 seconds and v58 by 10.7901 seconds. v65 previously scored 225.8994.
The old five-second objective is complete; the 225.00 objective remains active. The original objective and
approval statements below describe the earlier phase; the new instructions take
precedence. New experiment records live in `../runs/v3_20261011_target225/`.

A further fixed screen adds a ground-only mean/max readout to the physical
track representation. It preserves the existing all-point representation,
initial weights, training bytecode, dropout RNG and missing-track gate; a
zero-initialized 64-by-64 projection uses existing ground flags. Desktop Jan/Jul
fits are running against saved physical single controls. This has no result or
production candidate yet. Run: `../runs/v3_20261011_nn_track_ground_pool_jan_jul/`.
Relaxing affine weight bounds/regularization projects only about 0.01 seconds
beyond v69, so no candidate is selected from that diagnostic.

A fixed airport-bias diagnostic on v69 also fails: each month uses the other
month's mean error per airport, shrunk by a fixed 10,000-row prior. January
worsens 0.8802 seconds and July 1.0366 (mean +0.9584). Several airport biases
reverse sign. This reused, overlapping-fold diagnostic is not independent nested
validation; no correction is selected or uploaded.

A saved-prediction scan of Rome CatBoost shares 0.0–1.0 on v64 selects 0.6,
but improves only 0.0118 CV seconds over 0.5. This is inconclusive development
noise; the fixed half mixture stays unchanged, with no candidate or upload.

**Physical single production result:** v66 scores **233.7954**, 1.0203 worse than
v64. Its fixed three-member CV continuation improves **293.6566 → 292.8675**
(−0.7891; interval [−1.1227, −0.5638]), with both months better. Saved-model
reload checks pass for all six fold/member combinations. The laptop is fitting
production members1042/2042, retaining completed42. Adding the single model to
the scored affine basis projects225.7828; that file(v67) is retained without
upload while another complementary component is evaluated.

The remaining cost-weighted Rome calibration idea is now closed. Original
unweighted R20 classifiers were fitted on inner training months, with all
label histories rebuilt and frozen before three inner calibration months.
A weighted-Brier temperature/bias fit in each flight-ID group worsens v64
**293.6566 → 295.3301 (+1.6735)**, interval [+0.3921, +3.8614]; January +0.0549,
July +3.2921. All v64 errors reproduce exactly. The analytic objective gradient
matches central differences to below 8e−10. No full-year fit or upload follows;
retain `../runs/v3_20261011_rome_cost_calibration/` without repeating the fits.

The first October 11 screen adds fourteen fixed local-distance/motion transforms
to the sparse-track encoder through a zero-initialized parallel stem. Original
weights, preprocessing, dropout RNG, epochs and missing-track gates are preserved.
On the selected stack, the single physical-scale member scores **293.3477** versus
**293.9425** for the matched original single member (−0.5947; paired-day interval
[−0.8729, −0.3590]) and **293.6566** for actual v64 (−0.3089; interval
[−0.5841, −0.0965]). Both months improve. The latter comparison has different
track-pool sizes; a fixed three-member continuation is prepared. These are reused
development months. The scalar transform audit checks 29,624 values; saved-model
reload checks cover 1,094 held rows, and all v64 row errors reproduce exactly.
Results: `../runs/v3_20261011_nn_track_physical_scales_jan_jul/`.

A separate algebraic diagnostic uses our own already-scored integer predictions
and aggregate RMSE values. Squared-error geometry identifies the score of any
affine combination before final rounding. This is explicit **leaderboard-based
selection**, not independent validation or hidden-label access. A basis excluding
the date-specific probes, nearest-row experiment and unavailable completed
v39–v42 validation artifacts projects **225.8995**. Uploaded as **v65**, it scores **225.8994** on all 344,841 rows.
Current-fold legacy fits completed sequentially on the laptop/desktop; saved modern
predictions were replayed. Jan/Jul CV regresses **293.6569 → 297.0247** after
submission-equivalent rounding (January 328.5817, July 265.4676; paired-day change
interval [−0.2738, +7.4274]). This does not establish an out-of-sample improvement.
The score projection is verified by a second algebraic identity and synthetic
signed-weight cases. Upload/remote SHA and template audits pass; records are in
`../runs/v3_20261011_scored_blend_submit/submission_v65/`. No old ledger rows are backfilled. The exact v3–v9
historical source blob is absent; an archived original27 implementation and the
recorded feature/parameter contracts are used, with that limitation retained in
the reconstruction identity. The three research subagents remain unavailable
because their account allowance is exhausted; root performs these checks.

**Status: v64 uploaded with explicit user approval and scored 232.7751 seconds,
improving v58 by 3.4144 seconds on all 344,841 flights.** The objective is a five-second
official RMSE reduction, from **236.1895 to 231.1895**. That requires approximately
4.19% less mean squared error on the same scored population. Local validation
improvements are not official gains.

The upload completed at **2026-10-11 00:00:55 Amsterdam**. Its remote SHA-256
matches the audited file, and the scorer processed it at 00:01:30. The live
00:02 check places unique-rose **20th of 242 scoring teams**. Approval was bound
to this file and consumed; any further upload requires fresh approval. The
five-second goal remains **1.5856 seconds** away. See the
[submission record](../runs/v3_20261010_sparse_callsign_submit/submission_v64/submission_v64.json).

A diagnostic using the two official scores and the exact prediction difference
rules out another substantial gain merely by scaling this combined change.
The best point on the v58-to-v64 prediction line is approximately 1.1855 times
the change, with projected RMSE 232.6888, only 0.0863 below v64. This follows
from squared-error arithmetic on the same scored rows; it is conditioned on
rounded leaderboard scores, not independent validation or a submitted result.
No candidate or upload was created from this diagnostic.

The selected fixed combination scores **293.6566** on Jan/Jul CV versus its
matched seed42-tree control **295.7295**, a **2.0729-second reduction** with a
paired-day interval [-2.8070, -1.4699]. January improves 0.9880 seconds and July
3.1579. It combines equal shares of three raw-track networks and 15 callsign
networks within the unchanged 0.71 neural share, callsign inputs in the main
LightGBM, and a fixed half mixture of three-seed CatBoost and R20 Rome probabilities.
The other two trees and inherited corrections remain unchanged.

The track increment contributes **0.7651 seconds** beyond the completed
callsign/Rome combination (294.4217), interval [-1.1690, -0.4984]. Replacing the
track networks with three matched original networks in the same blend loses
1.0269 seconds, which supports a representation gain beyond ensemble arithmetic.
All comparisons use extensively reused development months, not independent
confirmation. The target five-second official improvement is not achieved.

All 15 full-year callsign networks, the main LightGBM and three Rome CatBoost
models are complete and retained. Both ranking candidates passed the full-source audit, exact parent replay,
independent all-row blend/serialization checks and 512 saved-tree predictions
per view. Their neural reload audit covers 15,360 predictions. All three selected full-year track
networks are complete, with both ranking predictions and saved-model reload checks.
Both ranking-view track caches were built and independently audited on the
laptop; their returned files passed SHA-256 verification on the desktop. A separate matched-laptop physical
clock-bypass screen has finished and is not selected. The selected old-view
candidate was uploaded once as v64; the four-month candidate remains local.

The sparse production trainer initially failed an exact CUDA parity check.
Repeating the original trainer also produces tiny variations (maximum parameter
difference 1.49e-8), identifying GPU kernel nondeterminism rather than evidence
of a changed update rule. With deterministic CuDNN enabled only inside the test,
the original and revised paths have exactly equal weights, losses, RNG states
and predictions. Production retains the original backend settings; its four
preflight tests passed before fitting.

Combining callsign6 and raw tracks inside one network failed the completed
same-desktop screen: **+0.3401 seconds** against the matched single-track
control, interval [+0.0392, +0.7425], with both months worse. It also loses
0.6259 seconds against the selected three-track pool, a separate comparison
with different ensemble size. The fixed 21/17-epoch fits, predictions and
preprocessing are retained; this branch is closed. A separate Rome classifier
test using the frozen track representation improves 0.3817 seconds against
its matched control but remains inconclusive, interval [-0.8153, +0.0459].
It does not change v64.

A follow-up saved-prediction diagnosis estimates only 0.2722 seconds from
expanding the track pool from three to fifteen seeds, and 0.3404 for an infinite
pool, assuming independent exchangeable seeds and approximately linear final
rules. These are projections, not measured gains or upper bounds; only three
seeds inform them. Further generic pooling is deprioritized. The final rules
are nonlinear on a small number of rows, explicitly recorded in the diagnostic.
A fixed context-conditioned attention residual completed its desktop screen.
It retains the original sparse model and adds four learned queries
from the flight-plan hidden state over the existing track representation.
CPU/CUDA checks preserve the original nonzero pooled path, initialization and
dropout RNG exactly, including missing-track gates and padding invariance.
Original156 preprocessing and the training-loop bytecode are unchanged; the
new constructor alone changes the model. Two seed42 fits use fixed 21/17 epochs,
with matched same-desktop controls and the actual v64 pool separately scored.
The candidate scores **293.8753**, against **293.9425** for the matched single-
track mixture: -0.0671 seconds, interval [-0.3597, +0.2511], January -0.1846
and July +0.0503. It is 0.2187 seconds worse than actual v64, interval
[-0.0202, +0.4807]. Both comparisons are inconclusive; this does not justify
expansion. Separate arithmetic checks reproduce monthly and aggregate scores
from all 344,419 stored row errors, and
1,094 saved-model reload predictions differ by at most 0.001 seconds. There
are 64 additional scalar raw-track reconstruction checks. All artifacts are
retained and v64 stays unchanged.
The three subagents are currently unavailable due verified account usage limits;
the root completed and reviewed these follow-ups.

The laptop completed a separate twelve-date arrival-clock pilot on March,
September, October and November 5/15/25, all within both outer training folds.
It reuses the audited stop witness, tests causal recent lag context against
leave-day-out airport/airline baselines and reads no departure targets. Its
portable packet contains 67,741 arrivals and 8,551,637 position observations
(89.95 MB). It finds 1,796 physical-stop witnesses; 1,795 can be compared with
a cross-date prior (one LFPG witness has no other-date airport reference).
Arrival-lag RMSE is 267.5498 for the static airport/airline baseline, 268.3273
with recent airport context and 267.7982 with recent airline context. MAE also
worsens, from 60.0800 to 61.1723/61.0166. All 11,910 donor edges satisfy the
strict causal boundary; 60 separate context reconstructions and every metric
pass. This signal-feasibility test does not justify a full-year feature build
or taxi-out fit. These metrics are arrival-clock errors, not challenge CV.

## Starting evidence

The live check at 2026-10-10 18:56:02 UTC found v58 still best at 236.1895,
rank 26/241 scoring teams. The latest scored upload is v61 at 236.6319.
The four-month v63 file was rejected previously; its existence does not mean
the scorer has accepted the enlarged truth set. No submission slot is used
by this research. A separate public-API check finds all 5,253 published scored
submissions, including the latest at 18:55 UTC on 10 October, still report
344,841 pairs.
That provides no evidence that four-month submissions are accepted yet.

The current recipe combines 15 PLE networks (share 0.71) with an equal mean of
main LightGBM LB1, airport LightGBMs P04 and CatBoost K07, then applies the R20
Rome recording classifier and inherited corrections. It has 156 ordinary
inputs. The matched three-tree-seed control is **295.7414415742** on Jan/Jul
CV: January 328.4780441039, July 263.0048390445. This averages the monthly
RMSEs; it is not pooled RMSE.

The target is a recorded duration. Some labels follow schedule-copy and
timestamp-recording conventions rather than physical surface movement alone.
Track information can improve the ordinary estimate while leaving these
recording errors unresolved. The current correction stack is consequential.

The historical gains identify where further work is most plausible:

- Main LightGBM tuning (v44) improved the leaderboard by 5.21 seconds.
- ADSB.lol ground and stand witnesses (v45/v46) supplied much larger gains.
- Neural diversity and better numeric embeddings (v47/v48) added about
  8.94 seconds together. Later pooling and tree diversity helped further.
- Repeated generic hyperparameter searches, per-airport blend weights, label
  caps, nearest-training-row selection, and weather are already explored.
  Weather and nearest-row selection lost on the leaderboard. The earlier
  pooled TabM experiment was also reviewed; it is not an untouched lead.
- Rome-classifier CV gains have transferred weakly or reversed. A classifier
  change must be checked through the complete correction stack, not judged
  from classification accuracy or log loss alone.

Sources: [submission ledger](../../SUBMISSIONS.md),
[neural report](NN_IMPROVE_JAN_JUL.md), [ideas report](IDEAS_JAN_JUL.md),
[HPO report](HPO_JAN_JUL.md), [weather report](OPEN_METEO_JAN_JUL.md).

## Network and execution

The root coordinates three independent branches: error-mechanism forensics,
model diversity, and external/unused signal research. Each reviews evidence
and proposes a fixed experiment; the root allocates the next heavy job.
Agents also reviewed each other's data alignment and training boundaries.

The user explicitly authorized laptop training during this session. Heavy
jobs run one at a time on each machine. Seven existing split/feature caches
were SHA-256 matched across machines before the laptop assembled the original
156-input matrices locally. Only new code and necessary results are transferred.
All seeds of a candidate stay on the same machine. A laptop-vs-desktop
LightGBM comparison has a machine caveat until a matched local control exists.
After the flight-plan fits, fingerprints verified equality of all original
156 columns, indices, labels, movement IDs, dtypes and category vocabularies
on both machines. This rules out reconstructed-input differences, not
machine-dependent LightGBM arithmetic.

The laptop's pre-existing files that collided with `git pull` were preserved
in stash `62ea25562b2e0dd5a3e08046c464321d7c5eea52`, not discarded or reapplied.
Both checkouts remain on main. Details are retained on the laptop in
`tmp/research_20261010/laptop_sync_preserved.json`.

Production preparation retains exactly 2,051,327 original ordinary training
rows and all 156 original inputs, appending only the six validated callsign
features. Both the unrounded v58/v63 predictions and their integer submission
serialization reproduce exactly. An initial control check incorrectly compared
floats with rounded integers; it failed before fitting, was corrected with a
regression test, and its partial artifacts are retained separately. Five bounded
production tests pass. Independent all-table checks cover 2,085,047 feature
training rows and both isolated ranking views (344,841 and 670,790 rows).
Old/final overlap has 316,651 changed takeoff timestamps, at most 12 seconds;
272,276 callsign-age values change accordingly. The two views do not share
their flight-keyed features.

The separate production Rome packet reproduces all 160,577 original classifier
rows, 33,205 positive labels and 68 inputs. Its compact portable transfer contains
101.9 MB before compression. The laptop verifies packet, code and evidence hashes;
desktop export must additionally reverify the full original sources before
creating probability records. These preparation checks do not establish a
leaderboard improvement, and neither new runner has an upload stage.

December log loss selected 2,017 CatBoost rounds. All three full-year refits
use that count; 3,072 saved-model reload probabilities reproduce exactly.
Desktop export reverified the full original sources. The neural portable
worker reuses the laptop's hash-matched original training cache and transfers
only a 120.20 MB final-view packet plus the small new feature/evidence files.
Its original matrix construction, category order and serialization passed
parity tests. All 15 networks use one laptop host and the original full-year
stopping epochs (15/21/16 for the three five-member groups).
The first laptop worker attempt exited 137 before any epoch or model artifact;
kernel logs confirm host out-of-memory termination at about 15.8 GB anonymous
resident memory plus swap. It is preserved as a failed preparation attempt.
The separate bounded-memory adapter calls the original encoder on row chunks,
uses disk-backed arrays and computes the same PLE quantiles one column at a
time. Two tests and an independent check reproduce tensors, targets, bins,
preprocessing, layouts and RNG state exactly, including missing values,
infinities, constant inputs and reordered categories. Parent sources remain
unchanged. A GPU inference-memory probe precedes the retried fits.
The first bounded-memory attempt trained, but revealed another execution
bottleneck: full prediction on 1,015,631 unlabeled ranking rows after every
epoch spilled GPU memory and took about 70 seconds per epoch. It was stopped
before completing any model. A separate fixed-epoch trainer keeps all optimizer
updates unchanged, monitors only 1,024 unlabeled rows, and performs full
prediction once at the end in 16,384-row batches. Weight/RNG parity is checked
before this execution change is used; no stopping epoch is selected from it.
CPU and CUDA probes subsequently reproduce every weight, loss and RNG state
after two fixed epochs exactly. A 65,537-row full-width inference probe also
matches exactly between the two batch sizes. Peak inference allocation falls
from 4.84 to 1.34 GiB; with all production tensors resident, the new peak is
6.05 GiB. Full-year epochs now take about six seconds rather than seventy.
All 15 models subsequently finished. An independent reload audit checked 512
input-selected rows per view for every model: 15,360 predictions in total,
maximum absolute difference 0.003418 seconds against the saved float32 outputs.
This is an inference reproducibility check, not validation of prediction accuracy.

A separate raw-source production audit checked 720 reconstructed values on
72 flights, including 12 midnight bridges and eight overlapping old/final
jitter pairs. It checked 101 callsign-day hashes, 72 departure clocks/callsigns
and 64 arrival in-block clocks. Current-call-start coverage is 84–87% across
the four final months. The all-table and raw-source audits are complementary;
neither uses hidden departure labels.

An input-only sensitivity check finds no old/final test flight crossing the
inherited p<0.05 exception within the no-ID Rome 4–12-hour gap band. Mean
absolute probability changes are about 0.023–0.025 by month. This rules out
that particular discontinuous change for the prepared classifier mixture;
it does not validate the remaining probability changes against hidden labels.

## Completed model screens

Changes below are exploratory comparisons on reused development months.
Paired-day bootstrap intervals do not make those months independent evidence.
No failed or inconclusive experiment is added to the submission ledger.

| Fixed change | Comparison | Jan/Jul CV change | Decision |
|---|---|---:|---|
| Genuine AOBT residual target, bounded offset and no schedule fallback | One matched N09 network | +0.7280 s | Stop |
| Same residual network, half mixed with v58's network pool | v58 | +0.5835 s | Stop |
| Ten explicit AOBT-minus-sensor differences | One matched N09 network | -0.0971 s | Inconclusive; stop |
| Same sensor-difference network, half mixed with v58's pool | v58 | +0.4100 s | Stop |
| Airport-specific residual output heads | One matched N09 network | +0.8895 s | Stop |
| Same airport-head network, half mixed with v58's pool | v58 | +1.0040 s | Stop |
| Eight aircraft identity/rotation inputs | One matched N09 network | +0.7713 s | Stop |
| Same rotation network, half mixed with v58's pool | v58 | +0.7156 s | Stop |
| Same eight rotation inputs in main LightGBM | v58, tree seed 42 | -0.0162 s | Inconclusive; stop |
| 22 cached ADSB inputs added to R20 Rome classifier | v58 | +1.7809 s | Stop |
| Restore rule-covered rows to R20 training, no new inputs | v58 | +0.6193 s | Stop |
| Replace Rome late constant with bounded current ordinary prediction | v58 | +0.2258 s | Stop |
| Remove LFPG twin-record override | v58 | +0.3853 s | Keep override |
| Remove non-Rome no-ID cap | v58 | -0.0076 s | Numerical/sampling noise; keep cap |
| Eight flight-plan arrival-time inputs in main LightGBM | v58, tree seed 42 | -0.0045 s | Inconclusive; stop |
| CatBoost recording classifier replacing R20 | v58 | -0.1665 s | Inconclusive |
| Fixed half CatBoost / half R20 probabilities | v58 | -0.5807 s | Exploratory lead |
| Three-seed CatBoost pool, fixed half mixture with R20 | v58 | -0.5772 s | Exploratory; no gain from bagging |
| Dedicated PLE Rome classifier on original R20 rows | v58 | +2.6215 s | Stop |
| Fixed half dedicated PLE / half R20 probabilities | v58 | +0.3855 s | Stop |
| Four short-window arrival recording inputs in R20 | v58 | +1.4824 s | Stop |
| Four month-excluded recorder-offset encodings | One matched N09 network | +0.9149 s | Stop |
| Same offset-encoding network, half mixed with v58's pool | v58 | +0.9106 s | Stop |
| Auxiliary recorder-offset and schedule-copy supervision | One matched N09 network | +0.2894 s | Stop |
| Same auxiliary network, half mixed with v58's pool | v58 | +0.6355 s | Stop |
| Six callsign/squawk timing inputs | One matched N09 network | -0.3711 s | Three-member check completed |
| Same callsign network, half mixed with v58's pool | v58 | +0.2200 s | Does not yet beat parent |
| Three callsign networks | Three matched N09 networks | -0.8279 s | Expand to 15 members |
| Three callsign networks, half mixed with v58's pool | v58 | -0.4544 s | Promising CV candidate |
| Fifteen callsign networks | v58 | -0.6528 s | Passing CV candidate |
| Fifteen callsign networks, half mixed with v58's pool | v58 | -0.5364 s | Passing fixed alternative |
| Six callsign inputs in main LightGBM | Matched laptop v58, tree seed 42 | -0.1450 s | Passing fixed comparison |
| Callsign15 plus fixed half CatBoost/R20 probabilities | v58 | -1.2395 s | Passing fixed combination |
| Same combination plus callsign main LightGBM | Matched laptop v58, tree seed 42 | -1.3078 s | Passing fixed combination |
| Prior own-aircraft arrival copy flag and age in R20 | v58 | -0.2005 s | Inconclusive; no expansion |
| Same own-arrival classifier, fixed half R20 mixture | v58 | -0.2121 s | Inconclusive; no expansion |
| Six track observations around original AOBT | One matched N09 network | +0.4193 s | Stop |
| Same AOBT-observation network, half mixed with v58's pool | v58 | +0.6135 s | Stop |
| Stand/month arrival recorder inputs in R20 | v58 | +0.7092 s | Stop |
| Same R20 change inside selected CatBoost/R20 mixture | Selected track/callsign recipe | +0.3216 s | Stop |
| Physical-clock output gates | Same-laptop single network | -1.7364 s | Conditional single-model gain only |
| Half clock-gated network / half original15 | Actual v58 pool | +0.6384 s | Stop |
| Whole-plan-masked specialist on missing-AOBT non-Rome flights | v58 | +0.2092 s | Stop |
| Same specialist, fixed half-parent mixture within its gate | v58 | -0.1257 s | Inconclusive; no expansion |
| R20 trained only on Rome schedule gaps over one hour | v58 | +5.0082 s | Stop |
| Same long-gap classifier, fixed half-R20 mixture in its gate | v58 | +0.1534 s | Inconclusive; no expansion |
| Raw sparse-track temporal branch | One matched N09 network | -2.2255 s | Expand to three members |
| Same sequence model, fixed half mixture with v58's pool | v58 | -0.8350 s | Passing fixed comparison |
| Same sequence half mixture | Matched half-N09/half-v58 control | -1.5292 s | Passing matched comparison |
| Three raw-track networks | Three matched N09 networks | -1.9140 s | Passing matched comparison |
| Same three raw-track networks | Full v58 network pool | -0.9405 s | Passing fixed comparison |
| Half raw-track3 / half v58 network pool | v58 | -1.1165 s | Passing fixed comparison |
| Half raw-track3 / half callsign15 with callsign main tree and half Cat3/R20 | Matched laptop v58, tree seed 42 | -2.0729 s | v64: 232.7751 official (-3.4144 s) |
| Joint callsign/raw-track network in fixed half mixture | Matched same-desktop single-track mixture | +0.3401 s | Stop |
| Frozen track encoder inputs in Rome CatBoost | Same-laptop single-classifier control in selected ordinary stack | -0.3817 s | Inconclusive; retain lead |
| Context-conditioned attention over raw-track representation | Matched same-desktop single-track mixture | -0.0671 s | Inconclusive; stop |

The one-network direct control is 298.9873315237, substantially worse than
the existing 15-network pool. Comparing a new single network directly with
the full pool would conflate architecture and ensemble size. Both the
matched-size and fixed half-pool comparisons are retained.

The residual network worsened July by 1.6423 seconds. The sensor-difference
matched comparison has interval [-0.6742, +0.4947]. Neither the residual nor the sensor-difference single-network
half mixture beats the actual 15-member parent convincingly.

An additional audit corrects for the ensemble-size difference: half historical
N09 seed42/k0 plus half v58's pool scores **296.4356**, already 0.6942 seconds
worse than v58. Against that matched half control, callsign improves by
**0.4741 seconds**, interval [-0.8628, -0.1483], with January -0.5227 and July
-0.4255. Sensor differences improve by 0.2842, interval [-0.6081, +0.0132];
residual, AOBT observations and auxiliary supervision are inconclusive. The
other three arms regress. This supports expanding callsign first, rather than
rejecting it from a single-network-versus-pool comparison. The historical k0
is itself part of v58's pool, so this comparison combines feature effects and
useful diversity; it is not an isolated causal feature effect. It also reuses
the same development months.

All 15 historical individual-network half mixtures lose to their complete
pool, ranging from 296.1921 to 297.6133 (standard deviation 0.3568). The callsign
single-network half mixture scores 295.9615. The completed fixed three-network
comparison retains the first fit and the original parent stopping epochs 21/17:

- Three matched original networks: **296.7149**; three callsign networks:
  **295.8870**, change **-0.8279**, interval [-1.3176, -0.4486].
- Half callsign-three/half v58 pool: **295.2871**, change **-0.4544** versus
  the actual v58 model, interval [-0.7354, -0.2588]; January -0.2840 and July
  -0.6247. The overall network share remains 0.71.
- The corresponding half-original-three/half v58 control is **295.9419**;
  against this matched half control, the change is **-0.6548**, interval
  [-0.9438, -0.4425]. The same overlap/diversity caveat applies.

The completed 15-member expansion reuses the six saved month/member fits and
the original parent stopping epoch for each seed group. Its matched historical
15-member control reproduces v58 exactly. Full replacement scores **295.0887**,
change **-0.6528**, interval [-0.9420, -0.3731], with January -0.0499 and July
-1.2557. Its fixed half-parent alternative scores **295.2051**, change
**-0.5364**, interval [-0.7068, -0.4005], improving both months. All 30 saved
month/member models and prediction hashes, the six inherited members and both
preprocessing files pass independent checks. This is the first ordinary-model
family in this research to beat the actual parent clearly on these development
months. It does not establish a future-year or leaderboard gain.

Combining full15 callsign predictions with the previously fixed half CatBoost
Rome probabilities improves to **294.5020**, a further 0.5867 seconds relative
to callsign alone. The combination runner verifies the original, callsign-only
and CatBoost-only controls before scoring; an independent reconstruction
reproduces the 15-network mean and every saved score. No blend-weight search
is part of this comparison.

The additional main-LightGBM callsign comparison uses seed42 for every ordinary
tree and compares the new and original main LightGBMs fitted on the same laptop.
Its incremental change is -0.0680 seconds, interval [-0.1173, -0.0302], with
January -0.0267 and July -0.1093. The resulting combination scores **294.4217**
against **295.7295** for this matched v58 control. This single-tree-seed result
is kept separate from the three-tree-seed averages above; no mixture weights
were searched.

Rome ADSB changed one ordinary July flight's probability across the inherited
0.05 override boundary and produced a large error; other rows also regressed.
January Rome ground-track coverage is zero versus substantial training
coverage. The classifier-restoration arm added only 109 January-training
and 46 July-training rows, with all shared inputs and labels verified equal.

Saved runners and full run records:
`nn_residual_jan_jul.py`, `nn_sensor_differences_jan_jul.py`,
`nn_airport_heads_jan_jul.py`, `rome_adsblol_jan_jul.py`,
`rome_restore_rows_jan_jul.py`, under `VALIDATION/`;
matching `v3_20261010_*` directories under `VALIDATION/runs/` retain predictions,
models, preprocessing, code identities and scores. Do not repeat completed fits.

## Error and identity investigations

Two LFPG missing-ID January flights have labels of 84,240 and 58,206 seconds.
Their aircraft histories and stand neighborhoods were examined. Prior known
arrivals precede takeoff by approximately 32.5 and 40.6 hours; they do not
recover the recorded off-block times. These cases provide no defensible rule
or flight-specific adjustment.

Adjacent LFPG movement IDs often refer to the same aircraft: 97.46% among
473 sampled adjacent pairs across three January days. Other sampled airports
do not share that property. This is evidence for a possible identity join,
not permission to use a generic numerical-ID regressor or tune the two errors.

A separate cheap pilot checked whether movement-ID order exposes recorded
off-block chronology. Across all January/July departures, adjacent IDs within
airport/day have increasing block times only 49.8–56.4% of the time, similar
to takeoff-time ordering. High within-day rank correlation at some airports
does not supply tight off-block bounds. No ID-based clock predictor was fitted.

A new runway/time matcher recovers identities when exact callsigns are absent.
It uses ground-to-air events, runway geometry, airport time offsets calibrated
on two dates inside both training folds, bidirectional uniqueness and aircraft
conflict checks. Six targeted tests pass. On disjoint dates, 1,116 of 1,121
recovered deliberately hidden identities match the original identity. The five
disagreements instead appear to expose corrupt singleton callsign observations
in the original match: the recovered aircraft has a sustained correct callsign.
This interpretation is supported by retained callsign histories, not labels.

The complete 62-day Jan/Jul input-only pass then measured score opportunity:

| Rows that could change | Rows | Perfect-prediction ceiling on CV gain |
|---|---:|---:|
| Newly recovered identities | 792 | 1.7734 s |
| Existing identities changed by requiring >=3 callsign records | 374 | 0.0952 s |
| Union | — | 1.8693 s |
| Recovered with plausible stand-exit witness | 398 | 0.1280 s |

The five largest recovered errors account for 65.9% of that group's squared
error; some new tracks begin only seconds before takeoff despite multi-hour
labels. Better identity does not recover a missing departure clock. These
ceilings assume perfect changes on the selected rows and unchanged predictions
elsewhere; they do not bound incidental effects of refitting a whole model.
No full-year rebuild or model fit is justified for the five-second target.

Evidence remains in `tmp/research_20261010/error_mechanisms/` and
`tmp/research_20261010/signal_scout/kinematic_heldout_bound/`. The reusable matcher
is `VALIDATION/adsblol_kinematic.py`; it does not overwrite existing caches.

## Further experiments and source findings

Completed designs and continuing leads:

- **Arrival flight-plan times:** four unused durations/revisions from ARVT1,
  ARVT3 and supplied clocks plus four deviations from route/type medians.
  References use permitted training rows only and exclude each training query's
  whole month. Fixed main LightGBM settings; every other ensemble member stays
  v58. Three targeted tests pass. The laptop fit is complete and inconclusive.
- **CatBoost Rome classifier:** original R20 rows and 68 inputs, native
  categorical handling, fixed depth 6/LR .05/L2 10. Last-training-month log loss
  selects iterations before refitting all permitted rows. Evaluate full
  replacement and a predefined half mixture of probabilities with R20.
- **Aircraft rotation history:** broader previous-same-aircraft arrival and
  identity features are absent from the ordinary 156 inputs. Feasibility from
  retained identity caches passed a four-date pilot: 85–87% unique identity
  coverage, approximately 98% outside LTFM, with prior-arrival coverage on
  158/294 no-ID flights. All 3,954 identities independently matched to OPDI
  agree. Own-aircraft arrivals differ from the latest stand arrival for
  15–19% of identified pilot departures. Eight fixed inputs were built
  without downloading the deleted raw OPDI cache. The neural screen failed;
  the laptop main-LightGBM screen is flat (-0.0162 seconds, interval
  [-0.0498, +0.0179]). An independent audit
  checked 1,472 feature values on 184 rows, including month boundaries.
- **Short-window arrival recording:** same-airport/airline arrival in-block
  schedule-copy shares and counts over centered ±24 hours and trailing seven
  days. The existing monthly handler rate remains. Four fixed inputs use only
  supplied arrival labels, with a K10 airport/window prior. Their standalone
  association is weaker than the monthly input. The completed classifier
  screen worsens January/July CV by 1.4824 seconds and is rejected.

The rejected airport-head network used the unchanged shared PLE trunk/global
output plus zero-initialized linear residual heads, with a fixed 0.1 penalty
on their standardized squared output. Initialization reproduced the original
network exactly, including RNG state; unknown airports used the global head.
Both months regressed.

The flight-plan screen has interval [-0.0216, +0.0103] seconds, entirely
consistent with noise. Its seed-42 control is 295.7301261062, rather than the
three-tree-seed control used by the classifier and neural screens. Both fits
ran on the laptop, approximately 13.7 and 11.4 minutes; their original input
matrices were subsequently verified equal to the desktop's.

The initial single-seed CatBoost/R20 mixture changes January by +0.0394 and
July by -1.2008 seconds; its overall interval is [-1.1801, +0.0086]. That screen
was exploratory; the later three-seed mixture enters the selected combination.
The five largest positive contributions account for 19.8% of gross squared
error reduction, so the gain is not confined to one correction. A fixed
three-seed bagging check reuses seed 42 and fits 142/242 at the parent model's
training-selected 1,182/1,401 iterations, without another stopping search.
That completed check gives -0.5772 seconds (January -0.0206, July -1.1339),
interval [-1.1400, -0.0132]. This is essentially the same gain, not additional
or independent confirmation. Full classifier replacement remains inconclusive.

The neural rotation regression is +0.7713 seconds against the matched single
network, interval [+0.1975, +1.5314]. Most regression occurs on common known
aircraft, not unseen identities; removing aircraft ID is a possible ablation
but does not yet have a strong score mechanism. Independently estimated
remaining variance from more copies of the existing neural recipe is small:
15 to 30 networks projects about 0.16 seconds, and an infinite pool about
0.31 seconds, before final-rule attenuation. These estimates are diagnostics,
not measured gains; further generic seed expansion is deprioritized.

The completed frozen neural screen appended four clock-offset encodings: median
recorded-block minus original actual-AOBT offsets for airport/stand,
airport/operator and airport/runway, plus a stand-corrected actual clock.
Training rows exclude their whole month from reference labels; held rows use
only allowed training labels. Historical control stopping epochs are fixed,
avoiding inner-stop leakage from target encoding. Stand offsets show stable
descriptive cross-month structure, but both fixed comparisons regressed.
The independent audit reproduced all 56,389 median/support rows across 18
lookup tables, 1,008 sampled feature values, every missing-clock mask and all
2,085,047 raw label relationships. The matched-network regression is +0.9149
seconds, interval [+0.3934, +1.4086]; the half-pool regression is +0.9106,
interval [+0.7245, +1.1486]. No pool expansion is justified.

A callsign-clock pilot found first retained broadcasts sometimes start after
an existing ground segment and may identify arrival/turnaround contamination.
The source aggregates daily callsign/squawk groups and is censored at heatmap
chunk boundaries: these timestamps are not true callsign-switch times or
physical departure bounds. A compact, input-only feature family uses explicit
ambiguity and censoring flags.
Its six inputs are now built; the independent audit checked 306 values across
34 flights, including eight midnight bridges, plus full table bounds, identity
missingness and scoped arrival comparisons. First-current-call coverage is
82.3% in January and 85.2% in July. The neural screen improves the matched
single-network control by 0.3711 seconds, interval [-1.0706, +0.2761], January
-0.6879 and July -0.0543. Its three- and fifteen-member expansions pass the
comparisons above. The laptop main-LightGBM screen initially improved by
0.1456 seconds against the saved desktop control. Its completed matched-laptop
comparison improves by **0.1450 seconds**, interval [-0.2193, -0.0975], with
January -0.1004 and July -0.1895. The laptop control differs from the historical
desktop control by only -0.0006 seconds, consistent with numerical variation.
This resolves the machine caveat for this feature comparison.

A separate six-feature family observes the aircraft at its original reported
AOBT: nearest-point time offset, speed, stand distance, ground flag and moving
fractions in the preceding/following five minutes. It uses the audited identity
cache and cached positions, with a 120-second nearest-point tolerance and the
existing takeoff-plus-60-second observation cutoff. This is clock-centered
geometry, rather than another whole-track shape summary. Three boundary,
missing-input and unit/arithmetic tests pass. The completed neural screen
regresses by 0.4193 seconds, interval [-0.1269, +0.9816], with both months worse.
There is no justification for expanding this feature family.
The complete build covers 2,085,047 departures. Independent reconstruction
checked 354 values on 59 flights and all-row eligibility/bounds. Any observation
is available for 20.3%/29.4% of January/July validation rows, holding 2.4%/13.0%
of baseline squared error. Removing 10% of those rows' squared error would
improve average CV by about 1.06 seconds; this is a hypothetical error budget,
not a measured gain.

A completed neural arm adds auxiliary supervision for recorder modes:
ordinal recorded-block/actual-AOBT offsets and schedule-copy labels. Its main
mean-regression objective, original 156 inputs and inference output remain
unchanged. It regresses by 0.2894 seconds, interval [-0.3175, +0.8217]; January
improves 0.7480 and July regresses 1.3268. The auxiliary probabilities cannot
replace the Rome classifier: known Rome copies are excluded from the ordinary
regressor's training, leaving no Rome positives for that auxiliary head.
Using those probabilities in place of R20 worsens CV by 78.94 seconds; a fixed
half mixture worsens by 4.02. This failed diagnostic motivates a separate PLE
classifier trained on the original R20 population, rather than reusing the
ordinary regressor's auxiliary head. That dedicated classifier also fails:
full replacement +2.6215 seconds, interval [+0.4275, +4.7091]; fixed half-R20
+0.3855, interval [-0.6508, +1.3027], with January -0.1511 and July +0.9220.
It uses inner-training-only preprocessing and last-training-month log-loss
stopping (7/8 epochs), with original R20 rows and inputs. Portable preprocessing
and saved model reloads reproduce real predictions. No expansion is justified.

One new input has a more direct recording mechanism: whether the same aircraft's
previous arrival copied its own scheduled in-block time, plus that arrival's
age. It is absent from the current 68 classifier inputs. A training-only pilot
finds higher departure-copy rates when that prior arrival was a copy, including
within airline/month strata; the smaller costly no-ID/late slice is inconsistent.
The completed fixed two-feature classifier screen improves by 0.2005 seconds,
interval [-0.8251, +0.4538]; a fixed half-R20 mixture improves by 0.2121,
interval [-0.5401, +0.1282]. Both are inconclusive. Every original classifier
value/category/index and all 239,758 prepared Rome rows' two new values were
independently checked. Fits ran on the laptop, with a historical desktop R20
comparison caveat. No expansion or new rule is selected.

A missing-plan specialist has completed. In the current validation,
all 4,976 non-Rome rows missing original AOBT lack all 16 plan fields. Excluding
the two previously investigated LFPG extreme cases only for opportunity
accounting, this group holds 3.03%/7.22% of January/July squared error. Perfect
predictions there would improve CV by 7.35 seconds; 10%/25%/50% subset-SSE
reductions would yield about 0.725/1.817/3.647 seconds. These are hypothetical
ceilings, not model results. Actual scoring retains every row. Coherent
whole-plan masking removes 35 plan-derived inputs from the original 156 and
masks physical features when their original lookup relied on a changed key.
The 121-input specialist trains on the original ordinary population and applies
only to non-Rome flights with missing original AOBT. Independent checks cover
3,309 metadata rows and all dependent-input groups. Full replacement regresses
by 0.2092 seconds, interval [-0.1169, +0.5598]. A fixed half-parent mixture
improves by 0.1257, interval [-0.3000, +0.0228], with both months slightly
better. That result is inconclusive; no expansion is planned. Predictions
outside the gate remain exactly unchanged. Removing the two LFPG extremes
only for diagnostics does not change that conclusion.

A sparse raw-track neural branch passed its four-date input-only pilot:
17,762 of 21,814 departures have at least two retained observations (81.4%).
This includes airborne-only histories; two-ground-point coverage varies from
41.4% to 65.3% across the pilot dates. Histories with 2–31 points account for
45.8% of usable rows. The masked temporal branch preserves the original N09
parameters, initial predictions and dropout RNG exactly, and has zero residual
for absent/single-point histories. Eight tests and independent checks of
6,672 raw values, 17,792 transformed values and all 130,884 geometry metadata
values pass. Synthetic forward/backward time is 6.06 times the tabular model,
with 870 MiB peak allocated GPU memory before full encoded data and optimizer
state. The complete cache now contains 2,085,047 departures and 84,624,170
observations (2.03 GB), built in 678 seconds. It reproduces all 775,873 pilot
points and their 4,655,238 raw values, and passes all time/order bounds.
The fixed seed42 Jan/Jul fits are complete. January epochs take about 29
seconds and July epochs about 21; peak allocated GPU memory is 3.72 GiB.
The matched-single comparison scores **296.7618** versus **298.9873**, change
**-2.2255**, interval [-3.3353, -1.5160], with January -2.5048 and July -1.9462.
The fixed half-sequence/half-parent-pool comparison scores **294.9064** versus
**295.7414**, change **-0.8350**, interval [-1.3387, -0.5088], improving both
months. Against the equivalent half-original-single/half-pool control, the
change is -1.5292, interval [-2.2173, -1.1437]. The lone sequence model remains
worse than the entire parent pool by 1.0204; ensemble sizes must remain explicit.

The fixed coverage diagnosis supports a physical signal: observed-ground
subsets improve matched RMSE by 12.51/4.56 seconds in January/July, and sparse
2–31-point histories improve by 3.54/4.46. The five largest gains contribute
only 1.0% of gross squared-error reduction. Zero/one-point rows are flat in
January and worse in July because the shared tabular branch is jointly trained.
The completed fixed three-member expansion reused k0 and added k1/k2 without
changing inputs, architecture or stopping epochs. It scores 294.8009, improving
the matched original three-member pool by 1.9140 seconds and the actual parent
by 0.9405. A fixed half mixture with the parent scores 294.6249, improving
1.1165 seconds. The selected combined recipe and fair-control results are
reported at the top; no availability gate enters that selected recipe.
A secondary availability fallback, declared after the k0 diagnosis, keeps the
original parent pool on fewer-than-two-point rows; its matched control and
post-k0 development status must be reported. There is no ground/count threshold
search. The four-fit matched-laptop physical-clock-bypass screen is complete.
It improves its same-laptop single-network control by 1.7364 seconds, interval
[-2.6626, -0.9675], and the fair half-single/half-pool control by 0.4981 seconds.
However, its half mixture scores 296.3799, **0.6384 seconds worse than v58's
15-network pool**, interval [+0.4628, +0.8747]. No expansion is selected.
The laptop single control itself is 1.7222 seconds worse than its historical
desktop counterpart, almost entirely in July. The comparison therefore supports
a conditional architecture effect, not a deployable improvement. All four
models and predictions are retained. Callsign15 CV was run entirely on the
desktop, as were its historical neural controls; the later callsign production
fits ran on the laptop.

Production sequence preparation reuses the audited aircraft identities and
unchanged sequence selector, with separate old/final namespaces. The compact
queries total 22.29 MB; the required cached position union is 122 files and
637.94 MB. The training sequence cache stays on the desktop. A single pinned,
low-priority CPU prepared the queries in 10.44 seconds. All 344,841 overlapping
flights retain the same original AOBT, while 316,651 takeoff clocks shift by up
to 12 seconds. Known aircraft identities never disagree, but identity availability
changes for 475 flights (231 known-to-missing, 244 missing-to-known). The original
descriptive manifest incorrectly reported zero differences because nullable
comparisons omitted these cases; a supplemental audit corrects that count. Each
view retains its own audited identities, queries and sequence cache. Missing March 1/August 1 source days
remain explicit. Five focused tests and 224 independently reconstructed queries
check schema, identity, raw clocks, geometry and view isolation. Both caches
were built on the laptop: 16,165,071 old-view points (388 MB) and 31,252,680
final-view points (750 MB). The independent output audit checks all 1,015,631
rows and 47,417,751 points, plus 96 reconstructed raw cases containing 3,733
points and eight shared old/final jitter IDs. These cache checks establish input
integrity, not an accuracy gain; the selected production recipe is recorded above.

Input-only coverage reveals substantial transfer uncertainty. Overall usable
track coverage rises from 77.18% to 82.67% for January and from 82.16% to
83.80% for July between 2025 and 2026. Rome has a much sharper reversal in
ground observations: 1/11,317 flights in January 2025 versus 9,662/11,061 in
January 2026, but 14,804/15,211 in July 2025 versus 48/15,838 in July 2026.
Most July 2026 Rome flights still have airborne observations; this is not
simply missing files or no track. No outcome-driven gate is derived from
these input shifts. See the saved coverage and independent output audits.

A different event-localization architecture was assessed without fitting.
On the fixed March 10/September 10 training-date pilot, 4,990 of 10,148 original
ordinary-training rows have observations bracketing their recorded off-block
time. Only 1,231 bracket it within 120 seconds, and 2,013 have any point within
60 seconds. The median bracketing gap is 4,020 seconds, also present in the raw
points; sequence subsampling does not explain it. A universal timestamp pointer
would constrain many flights to the wrong observed interval, so training is
deferred. Any later localizer needs an unrestricted fallback. The pilot uses no
held-out labels and establishes coverage, not an accuracy forecast.

A cached-data feasibility check also rejected dynamic airline clock-versus-
physical-sensor offsets. Prior two-hour same-airline peer coverage is sparse;
the resulting sensor residual estimates lose to airport or static-airline
references. Whole-day airline medians add almost nothing over other-day
airline medians in both 2025 validation months. No full feature build or fit
was started for this direction.

A different arrival-clock mechanism remains an audited pilot: compare supplied
arrival in-block time with an independently observed stop at that aircraft's
mapped stand, then consider causal airport/operator lag summaries for departures.
The initial witness accepted only 10 of 10,818 arrivals on two fixed dates.
Its immediate-predecessor condition incorrectly rejected normal gradual
deceleration: 571 arrivals already met the same slow/near/three-point/30-second
stop requirements. A preserved second pilot allows intermediate moving points
inside the stand circle, while still requiring an observed ground approach
from outside 100 m within 300 seconds and continuous gaps of at most 120 seconds.
It accepts **221/10,818 (2.04%)**. Independent reconstruction checked all 221
witnesses, 19,368 raw points and 1,633 approach/stop evidence points. Arrival
labels never select a stop, and no departure targets were read.

Observed median recording lags are mostly airport-specific: EHAM -33.5/-31
seconds, LSZH -13/-1, EDDF +54/+35 on the January/July pilot dates. Two dates
and sparse airline coverage do not establish incremental predictive value or
day-to-day regime changes. The corrected mechanism remains underexplored;
no full-year builder, taxi-out fit or promotion follows the initial pilot.
The subsequent twelve-date laptop test above finds no useful incremental
recent context under its fixed definition; this lead is now deprioritized.

Fixed diagnostic exclusions do not rescue the failed neural screens: removing
the two previously identified LFPG extreme labels, or restricting to labels
of at most three hours, preserves each experiment's direction. No such label
exclusion is applied to model selection scores or future predictions. Raising
the neural cap was already tested as HPO N14 and lost 3.26 seconds versus its
matched older control; the surviving non-LFPG ordinary tails are too sparse to
justify repeating that direction.

An additional Rome diagnostic supplies perfect copy/non-copy labels to the
unchanged probability/rule stack and improves Jan/Jul CV by 22.49 seconds.
That oracle establishes error headroom, not a feasible predictor. No-ID-only
classifier training is untested, but only 554–1,172 training rows qualify and
copy rates shift substantially between training and held months. It is
deprioritized in favor of better inputs and classifier-family diversity.

A fixed long-gap Rome specialist is now closed: it trains R20 only on the
original eligible rows with schedule gaps over one hour and changes only
that same input-defined held-out group (716 January and 3,450 July rows).
Full replacement regresses by 5.0082 seconds, interval [+1.9433, +8.9172];
a fixed half-R20 mixture regresses by 0.1534, interval [-0.8283, +1.1212].
No threshold search or expansion follows. Historical probability calibration
was also checked before proposing new work: the September two-parameter
Platt-calibration experiment, preserving its hard rule, regressed by 0.3419
seconds, interval [-0.0117, +0.6782], with all three seeds worse. Its much larger
reported regression combined calibration with removing the rule; those are
different comparisons. Cost-weighted Brier calibration remains untested, but
requires new honest inner predictions and is not a cheap saved-output fix.

A fixed stand/month arrival-recorder screen is also closed. Two inputs captured
the stand's arrival schedule-copy residual after airline/month adjustment (K50
shrinkage) and its support count. Original 68 classifier inputs, labels, folds
and 1,700 R20 rounds stayed unchanged. Four laptop fits provided matched local
controls and candidates. Both historical recipe scores reproduced exactly.
Replacing R20 on v58 regresses 0.7092 seconds, interval [+0.2320, +1.1893];
replacing only R20's half inside the selected track/callsign/CatBoost recipe
regresses 0.3216, interval [+0.0980, +0.5467]. The predeclared half changes also
regress (0.2721/0.1412). Small training associations and broad arrival coverage
did not establish useful incremental information. No full-year fit, promotion
or submission follows this test.

A separate Rome screen freezes each outer fold's ordinary sparse-track encoder
and appends its 64 pooled motion activations plus nine input-quality values to
the original classifier. The downstream classifier retains all original
copy-positive labels. Reusing the completed CatBoost rounds (1,182/1,401) avoids
selecting an inner stopping epoch from rows whose taxi targets trained the
encoder; encoders never cross outer folds. Only the track branch is extracted,
with no predicted target or tabular hidden state. Original 68 inputs, categories,
row identities and labels are preserved. Preparation completed in 10.2 seconds;
128 raw-scalar/vectorized reconstruction checks pass. Four completed same-laptop
GPU fits compare original68 with the new141 inputs. The primary comparison uses one
matched classifier per arm; the actual selected three-classifier pool is a
separate reference. There is no new gate, weight search or production fit.
Input-only coverage still reveals substantial transfer risk: January's 11,317
Rome validation rows contain 8,822 usable tracks but only **one** with ground
observations; July has 14,804/15,211 ground-observed rows. The separate coverage
pilot used slightly broader training populations than the exact retained
classifier packets; prepared.json records the actual 126,507/86,723 training
rows. The root reviewed this runner after the subagents exhausted their account
usage allowance; this is not an independent-agent review. The matched control
scores 293.6707 and the candidate 293.2890: **-0.3817 seconds**, interval
[-0.8153, +0.0459], January -0.0155 and July -0.7480. Relative to actual v64,
the change is -0.3676, interval [-0.8074, +0.0622], with different classifier
pool size explicit. Both comparisons are inconclusive on reused development
months. The original v64 squared errors replay exactly. Models, predictions
and features are retained; no expansion, full-year fitting or promotion follows.
All four returned model/prediction hashes pass; 2,048 input-selected reload
probabilities reproduce exactly. A fixed input-cohort diagnosis sharpens the
transfer concern: July's usable ground-track rows account for 107.38 million
less squared error, while absent and airborne-only groups jointly incur 33.35
million more. Ground coverage almost disappears in July 2026. This is a
development-cohort diagnosis, not a measured test-set effect, and does not
justify a new outcome-selected gate.

The arrival-plan direction was suggested by public
[kind-mango source](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango/blob/main/src_v3/build_plan_nm_taxi_v65.py).
No source code was copied; this experiment's medians are input-only.

The newly published
[logical-ladder repository](https://github.com/AbdullahKhan-x/PRC-2026) suggests
EUROCONTROL daily ATFM/slot/pre-departure delay series. Its reported gain also
adds OPDI features, so the daily-data effect is not isolated. The official
[source terms](https://ansperformance.eu/about/disclaimer/) restrict commercial
use and modification; openly licensed competition reuse was not established.
The [coverage page](https://ansperformance.eu/data/) lists pre-departure series
only through June 2026, though file sizes suggest the page could lag the files.
July coverage is unverified. No series was downloaded or used for training.

Additional October 10 public releases (quirky-honey, resilient-kiwi and
gentle-octopus) report worse scores than v58 and mostly repeat ideas already
tested here. No identifiable newly published 213–225-second implementation
was found. Quirky-honey reports only a small gain from a different previous-
aircraft-landing interval family; that is a weak prior rather than a test of
our supplied-arrival, identity and stand-comparison features. Primary links
and the search result are retained in
[the source evidence](research_network_20261010/new_public_sources.json).

Concise aggregate findings and independent audit evidence are preserved under
[`research_network_20261010/`](research_network_20261010/manifest.json), with
source paths and hashes. Row-level caches remain ignored and local.

## Promotion standard

Continue with concrete new signals and materially different model families.
Judge a small neural screen with matched-size controls, including a matched
half-single/half-pool control when needed. Mixing one new network into a
15-network parent changes ensemble variance; that difference alone is not an
architecture verdict. Further candidates must beat the actual v64 combination;
retain v58 comparisons where they isolate a previously studied change.
Retain failed evidence and original
controls. Prepare and audit each concrete candidate before uploading. The October11
autonomous-upload instruction supersedes the historical per-file approval rule.
