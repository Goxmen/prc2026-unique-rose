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
