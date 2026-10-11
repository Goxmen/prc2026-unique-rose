# Research toward 215 seconds — 11 October 2026

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
