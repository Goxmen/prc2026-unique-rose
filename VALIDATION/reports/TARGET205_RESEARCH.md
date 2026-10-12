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

**Exploratory source check:** airport arrival landing time versus the same
arrival's ARVT3 plan endpoint may expose current source-clock disagreement.
This differs from arrival taxi-in/delay and the failed ADS-B physical-stop lag.
Coverage, time stability and split-half predictability must pass a source-only
feasibility check before any departure-label association or fit. No gain is
established. The error agent is separately checking unexplored recording and
flight-linkage mechanisms while the two hosts run their fixed jobs.

Completed negative screens and their retained models are documented in
[the 11 October report](TARGET215_RESEARCH.md). Do not repeat those fits,
reopen cancelled queues or spend the final slot on numerical noise.
