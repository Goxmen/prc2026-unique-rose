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
