# Submitted models

**Current validation: Weighted CV.** The column is between CV combined and Official LB below.
Best verified Weighted CV: **v42 at 263.5929 s**
([validation report](VALIDATION/reports/OPDI_DISRUPTION_EXPERIMENT.md)). The six uploads before
it — v34, v37, v38, v39, v40 and v41 — are scored on it too
([report](VALIDATION/reports/WEIGHTED_CV_RECENT.md)). Row 46 (stand coordinates, v43)
scores **263.1476**, −0.47 s against a matched rebuild of v42's recipe.

**Best and latest submission: v43, 274.7660 s**, improving v42 by **0.1416 s**.
Live check **2026-09-25 12:17 UTC**: rank **38/191** scoring teams; all **344,841** flights scored.
[Submission record](VALIDATION/runs/v3_20260925_stand_coords/submission_v43/submission_v43.json).

Rows 22–28 are our Rome substitution mixture and hybrids. Rows 22–25 have
validation scores only and have not been submitted. Rows 26–30 were uploaded as
v26–v30 and row 33 as v33, so each file number matches its row; v22–v25 and
v31–v32 were never used. Row 29
adds Rome recording signals to row 28. Rows 30–32 add stand clock and dirty-day
inputs to row 29; rows 31–32 have validation scores only. Row 33 applies the
flight-plan window to row 30's predictions after the model (no refit). Rows 34–36
each make one change to row 33's predictions; row 34 was uploaded as v34, and rows
35–36 have validation scores only. The selected Artifact-combined (former row 44)
is row 37 below and was submitted as **v38**; filename v37 was already used by a
separate model. Our
Day-shift hedge is row 40 below and was uploaded as v37. Rows 41 and 42 add inputs
from open ADS-B ground tracks: row 41 on row 34's recipe, with a validation score
only, and row 42 on row 37's, the recipe behind v38, uploaded as v39. Row 43 is
row 42 refitted with the flights the four-hour Rome rule overwrites removed from
the model fit, uploaded as v40. Row 44 adds four taxi occupancy inputs to row 43 and
was uploaded as v41; its supported CV gain reversed on the leaderboard. Row 45 adds
12 snow-duration and completed-arrival inputs to row 43 and was uploaded as **v42**,
Row 46 adds three stand-position inputs to row 45 and was uploaded as **v43**, which is the
best submission.

## Codenames and recipes

Unless a row says otherwise, the models from v3 onwards are LightGBM fits on all
twelve months of 2025 with seed 42, the original 27 model inputs (raw movement
and flight-plan columns, plus departure-taxi history, arrival taxi-in history and
traffic counts), 800 rounds and a training-label filter of 1–200,000 seconds.

| Order | Codename | Actual recipe | CV next-month | CV longer-gap | CV combined | Weighted CV | Official LB | Submission |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | **Airport-mean** | No model: 2025 mean taxi-out per airport (labels 1–7,200 s); global mean fallback | — | — | — | - | 661.5737 | v1 |
| 2 | **LGBM-raw** | LightGBM, raw columns only (`pipeline/train.py`); 3,000 rounds; labels **1–7,200 s** | — | — | — | - | 533.3759 | v2 |
| 3 | **Keep-tail** | 27 inputs; 800 rounds; labels widened to **1–200,000 s** | — | — | — | - | 405.6987 | v3 |
| 4 | **FPL-flag** | Keep-tail + `has_flight_plan` input | 366.5209† | 333.9163† | 348.7321† | - | 405.7838 | v4 |
| 5 | **Ceiling-10k** | Keep-tail, labels capped at **10,000 s** | — | — | — | - | 518.3068 | v5 |
| 6 | **Ceiling-16k** | Keep-tail, labels capped at **16,000 s** | — | — | — | - | 498.3336 | v6 |
| 7 | **Ceiling-22k** | Keep-tail, labels capped at **22,000 s** | — | — | — | - | 486.0517 | v7 |
| 8 | **Ceiling-32k** | Keep-tail, labels capped at **32,000 s** | — | — | — | - | 472.7133 | v8 |
| 9 | **Ceiling-50k** | Keep-tail, labels capped at **50,000 s** | — | — | — | - | 433.6797 | v9 |
| 10 | **Ceiling-70k** | Keep-tail, labels capped at **70,000 s** | — | — | — | - | Rejected (daily limit) | v10 |
| 11 | **Ceiling-90k** | Keep-tail, labels capped at **90,000 s** | — | — | — | - | Rejected (daily limit) | v11 |
| 12 | **B0-800** | Baseline; no recovery, no rule | 366.7465 | 334.1785 | 348.9770 | 442.0361 | 406.4281 | v12 |
| 13 | **Q0** | B0-800 + **6 h Rome rule** | 300.5904 | 256.1971 | 276.7374 | 326.3408 | 354.4303 | v13 |
| 14 | **Q1** | B0-800 + **R3 recovery** + 6 h Rome rule | 296.9037 | 258.5505 | 276.1937 | 327.3573 | 354.0004 | v14 |
| 15 | **Q0-4h** | B0-800 + **4 h Rome rule** | 296.2785 | 255.6200 | 274.3707 | 322.8504 | 344.3114 | v15 |
| 16 | **Q1-4h** | B0-800 + **R3 recovery** + 4 h Rome rule | 292.8785 | 257.4787 | 273.7215 | 323.1023 | 344.4821 | v16 |
| 17 | **no_month** | Q0-4h − **`month`** input | 296.4686 | 251.8408 | 272.5082 | 320.3023 | 345.8984 | v17 |
| 18 | **no_history** | Q0-4h − **5 departure-taxi history inputs** | 296.5108 | 255.9093 | 274.6319 | 323.4994 | 342.5049 | v18 |
| 19 | **clock_features** | Q0-4h + **18 clock inputs** (still predicts taxi time) | 293.6614 | 255.6326 | 273.1284 | 321.2360 | 338.1113 | v19 |
| 20 | **small_trees** | Q0-4h with **63 leaves** instead of 255 | 296.6377 | 255.6624 | 274.5645 | 323.6976 | 345.3113 | v20 |
| 21 | **rule_routing** | Q0-4h; **rule-eligible flights dropped from training**, rule still predicts them | 286.6291 | 251.4024 | 267.5759 | 312.8226 | 342.4009 | v21 |
| 22 | **Rome-mix-base** | B0-800; Rome prediction = **p × gap + (1 − p) × B0-800**; no rule | 311.4125 | 283.5840 | 296.2310 | - | — | Not submitted |
| 23 | **Rome-mix-ordinary** | Row 22 with the (1 − p) part from the **regressor** | 317.6906 | 294.5641 | 305.0225 | - | — | Not submitted |
| 24 | **Rome-hybrid** | Row 23 + **6 h Rome rule** override; regressor at all airports | 280.6325 | 250.5677 | 264.3047 | - | — | Not submitted |
| 25 | **Rome-hybrid-4h** | Row 24 with the **4 h** rule | 280.8670 | 250.4076 | 264.3304 | - | — | Not submitted |
| 26 | **Rome-hybrid-4h-clock** | Row 25 + **18 clock inputs** (regressor and classifier) | 271.9287 | 246.7919 | 258.2262 | - | 333.8712 | v26 |
| 27 | **Rome-hybrid-4h-clock-weather** | Row 26 + **21 METAR weather inputs** (regressor only) | 270.3396 | 239.4157 | 253.5766 | - | 329.1609 | v27 |
| 28 | **Rome-hybrid-4h-clock-weather-queue** | Row 27 + **11 runway queue inputs** (regressor only) | 265.3809 | 234.4385 | 248.6174 | - | 322.6253 | v28 |
| 29 | **Rome-signals** | Row 28 + **airline**, and for the classifier **handler share** and **flight-number format**; **late-row estimate** for no-ID Rome rows at 1-4 h | 260.3360 | 233.2610 | 245.6201 | - | 314.4667 | v29 |
| 30 | **Stand-clock** | Row 29 + **5 stand clock inputs** (regressor and classifier) | 256.1836 | 229.7016 | 241.7877 | - | 309.9872 | v30 |
| 31 | **Dirty-day** | Row 29 + **6 dirty-day inputs** (regressor and classifier) | 261.3248 | 232.7730 | 245.8271 | - | — | Not submitted |
| 32 | **Stand-clock-dirty-day** | Row 29 + **both input sets** (regressor and classifier) | 257.6550 | 229.3705 | 242.3045 | - | — | Not submitted |
| 33 | **Plan-window** | Row 30's predictions clipped to **takeoff - LOBT +/- 61 min** for flights with a flight ID (post-processing) | 253.9467 | 229.2985 | 240.5262 | - | 282.4525 | v33 |
| 34 | **No-ID-cap** | Row 33 + **cap on no-ID predictions** outside Rome when the flight took off late (post-processing) | **253.0535** | **228.9805** | **239.9398** | 273.2162 | 280.4658 | v34 |
| 35 | **Twin-records** | Row 33; where LFPG **logs one takeoff twice**, the flight-ID record is set to takeoff - LOBT - 5.5 min (post-processing) | 253.6312 | 228.9143 | 240.1746 | - | — | Not submitted |
| 36 | **Seed-average** | Row 33's predictions **averaged over seeds 42, 142 and 242** (a submission needs two more full-year refits) | 252.6779 | 227.9686 | 239.2266 | - | — | Not submitted |
| 37 | **Artifact-combined** | Row 34 + **four recording-artifact corrections** ([recipe](VALIDATION/reports/RECORDING_ARTIFACT_EXPERIMENTS.md)) | 249.9460 | 224.8911 | 236.3151 | 267.5814 | **279.3805** | v38 |
| 38 | **Rome-binary-cal** | Row 34; **calibrated Rome probabilities** replace the hard 4 h override ([run](VALIDATION/runs/v3_20260919_rome_calibration/report.md)) | 296.7058 | 296.4860 | 296.5834 | - | — | Not submitted |
| 39 | **Rome-three-cal** | Row 38 with a **three-class classifier** (ordinary/schedule/day) ([run](VALIDATION/runs/v3_20260919_rome_calibration/report.md)) | 294.2137 | 282.8235 | 287.9234 | - | — | Not submitted |
| 40 | **Dayshift-hedge** | Row 34; Rome rule rows **14 h+ late on a later local day** get the probability-weighted mean of their two possible labels (post-processing) | 251.6273 | 221.3351 | 235.2323 | 265.9048 | 280.1743 | v37 |
| 41 | **OPDI-tracks** | Row 30 + **14 OPDI inputs** from open ADS-B ground tracks, then rows 33 and 34. Superseded by row 42 | 250.2107 | 226.7397 | 237.4207 | - | — | Not submitted |
| 42 | **OPDI-combined** | Row 41 + row 37's corrections, i.e. **v38's recipe with the OPDI inputs** | **247.0477** | **222.6081** | **233.7470** | 264.4218 | **277.3629** | v39 |
| 43 | **Rule-routed-training** | Row 42 with the **127 rule-covered departures dropped from the model fit**; every evaluation row still scored | **245.9817** | **222.5802** | **233.2338** | **264.3397** | **275.4146** | v40 |
| 44 | **Taxi-occupancy** | Row 43 + **4 taxi occupancy inputs**: other aircraft between their own `AOBT_3` and takeoff, at this flight's off-block and takeoff. CV gain did not transfer | 245.7880 | 222.2004 | 232.9414 | 263.8852 | 275.9592 | v41 |
| 45 | **Snow-arrivals** | Row 43 + **12 snow-duration and completed-arrival inputs** (ordinary regressor only) ([report](VALIDATION/reports/V42_SUBMISSION.md)) | **245.5661** | **221.8747** | **232.6647** | **263.5929** | **274.9076** | v42 |
| 46 | **Stand-coordinates** | Row 45 + **3 stand-position inputs** from public airport maps (ordinary regressor only) ([report](VALIDATION/reports/STAND_COORDINATES_EXPERIMENT.md)) | **245.1722** | **221.2893** | **232.1696** | **263.1476** | **274.7660** | v43 |

All values are **RMSE in seconds; lower is better**. **Weighted CV is the new
validation headline**, adopted on 20 September 2026. It uses the existing v3
folds and three-seed losses, with fixed weights that make the validation flights
better match ranking. The older CV columns remain for comparison. A `-` in
Weighted CV means it has not been computed and verified for that recipe.
See the [method](VALIDATION/docs/WEIGHTED_CV.md), the
[11-submission comparison](VALIDATION/reports/WEIGHTED_CV_LEADERBOARD_EXTENDED.md) and the
[five earlier uploads](VALIDATION/reports/WEIGHTED_CV_RECENT.md).

- **Weighted CV for the six uploads before v42** (v34, v37, v38, v39, v40 and v41) comes from
  `VALIDATION/tools/weighted_cv_recent_submissions.py`, which scores the fold predictions behind
  each uploaded file with the adopted weights. The local weight cache was rebuilt from the frozen
  specification: its four discriminator models are byte-identical to the recorded ones, with the
  same cross-fitting split, held-out calibration and effective sample size, so the ratios are the
  same numbers. Every original v3 score reproduces from the same predictions before weighting.
  Across these six, the average absolute gap to the official score is **11.57 s**, against
  42.89 s for CV combined. Direction agreement on the five consecutive steps is 3/5 either way,
  and the average error in the size of a step is 2.70 s weighted against 1.82 s, so the weighting
  fixes the level of the score without predicting how far a change moves the leaderboard. v38's
  entry is the previously verified value, carried over because its fold predictions are not on
  this machine.
  **v41 is a clean miss for both.** Its step from v40 is −0.29 unweighted and −0.45 weighted,
  while the leaderboard moved **+0.54**: the weighting made the predicted gain *larger*, not
  smaller, so it does not protect against a change whose CV gain fades with model age. The
  ageing readouts did catch it (8 of 10 positive, see row 44), which is the signal to lean on
  when weighted CV and combined CV agree on a small gain.
- **v1–v3 and v5–v11 have no protocol-v3 CV.** v1–v3 predate the protocol; v3 was
  checked only with the older, non-comparable harness. Its configuration was
  later validated as `baseline-v3` (366.7673 / 334.2027 / 348.9996) and
  resubmitted as B0-800 (v12), which scored 406.4281 against v3's 405.6987.
  The label-ceiling series (v5–v11) skipped validation.
- **† v4:** CV from the matching `fpl-flag` run `v3_20260909T161759260709Z`, which
  finished 35 minutes before the upload. No formal run-to-file link was recorded.
- **v12–v16:** CV from their linked validation runs; see
  [LEADERBOARD_CANDIDATES.md](VALIDATION/docs/LEADERBOARD_CANDIDATES.md).
- **Rows 22–24:** CV from run `v3_20260916T214848990330Z` on macOS: same protocol,
  800 rounds, three seeds, all 16 folds. Its baseline and six-hour-rule columns
  scored 366.7403 / 334.2015 / 348.9864 and 300.5811 / 256.1744 / 276.7212,
  a small replay difference from the B0-800 and Q0 rows. Details in
  [M1_MIXTURE.md](VALIDATION/reports/M1_MIXTURE.md), where they are called M1a,
  M1b and H2. The code and run are not committed yet and are not wired
  into the current harness.
- **Rome-hybrid was defined after seeing these validation results**, so its CV is
  optimistic. It is computed from the run's saved predictions, not a registered
  run column. It uses the six-hour rule and none of the v15–v21 changes.
- **Rows 25–26:** CV from run `v3_20260917T124103912051Z` on macOS, which also
  refitted Q0-4h, clock_features and rule_routing on the same folds and seeds.
  Those refits scored 296.2612 / 255.6012 / 274.3526, 293.6612 / 255.6325 / 273.1282
  and 286.6731 / 251.3930 / 267.5919, within 0.05 s of rows 15, 19 and 21.
  Paired day-bootstrap differences in combined CV (negative favours the hybrid,
  95% interval): Rome-hybrid-4h minus rule_routing −3.26 [−8.44, −0.10];
  Rome-hybrid-4h-clock minus clock_features −14.90 [−22.66, −9.27], and minus
  rule_routing −9.37 [−19.08, −3.79]. Next-month and longer-gap intervals are in
  the run's `comparison_*.json` files. The rule override was chosen after
  inspecting six-hour results on these folds, and CV gains have not always
  transferred to LB.
- **Row 27:** CV from run `v3_20260917T155106873031Z` on macOS (82.8 minutes),
  which refitted Rome-hybrid-4h-clock on the same folds and seeds as its control.
  That control reproduced row 26 exactly: all 8,353,590 saved predictions match.
  Paired day-bootstrap differences, weather minus control (95% interval):
  next-month −1.59 [−1.96, −1.23], longer-gap −7.38 [−11.06, −4.35], combined
  −4.65 [−6.69, −3.02]; all three seeds agree. Every one of the 16 folds improves,
  most in November and December. The weather inputs were fixed before this run.
  None of the folds contains January.
- **Row 28:** CV from run `v3_20260917T172934966809Z` on macOS (70.4 minutes).
  Row 27 is the control and was not refitted: its boosters were reloaded from the
  run above and reproduced its saved predictions exactly on every fold and seed,
  which also proves this run builds the same inputs. Paired day-bootstrap
  differences, queue minus row 27 (95% interval): next-month −4.96 [−5.59, −4.44],
  longer-gap −4.98 [−6.05, −4.04], combined −4.96 [−5.77, −4.29]. All three seeds
  agree, all 16 folds improve, and no airport opposes the headline in either score.
  The ageing differences (`B*_vs_A*`) all include zero, so the gain does not fade
  as the model gets older. A past-only variant, without the four inputs that use
  movements after takeoff, scored 267.4362 / 236.3997 / 250.6194; adding those four
  is worth a further −2.06 [−2.35, −1.78] next-month and −1.96 [−2.39, −1.58]
  longer-gap. The inputs and the rule for choosing between the two variants were
  fixed before this run. None of the folds contains January.
- **v26–v28 (rows 26–28):** uploaded 2026-09-18 by
  `VALIDATION/rome_hybrid_submit.py`, v28 at 09:53 UTC and v26 and v27 at 10:04.
  Each is one seed-42, 800-round refit of the ordinary regressor and the Rome
  classifier on all of 2025. Training inputs see all 2025 movements. Each ranking
  month (January and July 2026) sees all of 2025 plus its own month only. v26 has
  no weather or queue inputs. The script refuses to run if any source or data file
  differs from the validation run: v28 is checked against row 28's run, and v26
  and v27 against row 27's run. That run's control is row 26's recipe and matched
  all of row 26's saved predictions. Row 26's own run predates the `extra`
  argument of `rome_hybrid.py`, so it no longer passes the check. In every file
  the four-hour Rome rule overrides 43 of the 344,841 ranking rows, and every
  official score used all 344,841 rows.
  Official versus CV-combined steps (negative is better):

  | Step | Official LB | CV combined |
  |---|---:|---:|
  | v26 minus v19 (Rome hybrid on clock_features) | −4.24 | −14.90 |
  | v27 minus v26 (weather inputs) | −4.71 | −4.65 |
  | v28 minus v27 (runway queue inputs) | −6.54 | −4.96 |
  | v29 minus v28 (Rome recording signals) | −8.16 | −3.00 |
  | v30 minus v29 (stand clock inputs) | −4.48 | −3.83 |
  | v33 minus v30 (flight-plan window) | −27.53 | −1.26 |
  | v34 minus v33 (no-ID cap) | −1.99 | −0.59 |
  | v37 minus v34 (day-shift hedge) | −0.29 | −4.71 |
  | v39 minus v38 (OPDI ground-track inputs) | −2.02 | −2.56 |
  | v40 minus v39 (rule-covered rows out of training) | −1.95 | −0.51 |
  | v41 minus v40 (taxi occupancy inputs) | **+0.54** | −0.29 |
  | v42 minus v40 (snow and completed-arrival inputs) | **−0.51** | −0.57 |
  | v43 minus v42 (stand-position inputs) | **−0.14** | −0.51 |

  The weather and queue gains carried over to the leaderboard in full. Only about
  a third of the Rome hybrid's CV gain did. v41 is the first step in this table whose
  official score moved **against** a supported CV gain. Provenance, including file and model
  hashes, is in `submission_v<N>.json` in each run directory.
- **Rows 34–36:** CV from run `v3_20260919T104157403241Z`, derived from row 33's
  saved predictions without any refit by `VALIDATION/record_fixes_run.py`; its
  `pred_window` column reproduces row 33 exactly, and each candidate makes one change
  to it. Paired day-bootstrap differences against row 33 (95% interval), next-month /
  longer-gap / combined:

  | Row | Next-month | Longer-gap | Combined | Fold/seed rows moved |
  |---|---:|---:|---:|---|
  | 34 No-ID-cap | −0.89 [−1.59, −0.34] | −0.32 [−0.79, +0.01] | −0.59 [−0.98, −0.24] | 1,648 (1,569 closer, 79 farther) |
  | 35 Twin-records | −0.32 [−0.48, −0.17] | −0.38 [−0.81, −0.02] | −0.35 [−0.63, −0.11] | 414 (336 closer, 78 farther) |
  | 36 Seed-average | −1.27 [−1.32, −1.21] | −1.33 [−1.44, −1.21] | −1.30 [−1.37, −1.22] | all |

  All three seeds agree on every headline and no airport gets worse under any row.
  Row 34's longer-gap change is inconclusive; its gain is mostly LSZH and LFPG, where
  single day-long 2025 labels had inflated predictions at stands 103 and D06/C03.
  Rows 34 and 35 read their thresholds and offset off the 2025 labels, so their CV
  is somewhat optimistic. In the ranking set row 34 lowers 100 no-ID predictions
  (largest: NJE278K at LSZH stand 103, 13,089 s → at most 2,415 s) and row 35 moves
  the flight-ID record of 16 twin pairs, all at LFPG (e.g. UAL914, 2,166 s → 5,391 s).
  None of the folds contains January.
- **v34 (row 34):** prepared 2026-09-19 by `VALIDATION/record_fixes_submit.py 34
  prepare` from v33's saved ranking predictions, which the script checks still
  reproduce the scored v33 file byte for byte, and uploaded 2026-09-19 at 10:52 UTC,
  the second upload that day. Against v33's file, 100 ranking rows are lowered and
  none raised (EHAM 22, LFPG 18, EGLL 17, EDDF 14, LSZH 11, EDDM 7, LEBL 4, LTFM 4,
  LEMD 3); the 43 Rome rule rows do not move. The largest changes are four no-ID
  business jets at LSZH stand 103: NJE278K 13,089 → 2,415 s, DWW017 9,741 → 2,448 s,
  QGA01A 9,722 → 4,369 s and ECA2JP 10,226 → 5,011 s. Before upload, applying the
  validation ratio of error saved per squared move to the ranking moves predicted
  about −1.8. Official score 280.4658 on all 344,841 rows: **−1.99 against v33**,
  against a CV change of −0.89 next-month and −0.59 combined, and rank 39 of 166
  (from 40). The ranking set has more no-ID departures than validation (1.53% vs
  1.08%) and more poisoned-stand rows, so the gain was larger than in CV.
- **Row 40:** CV from run `v3_20260919T112057144323Z`, derived from row 34's saved
  predictions (`pred_noid_cap` of run `v3_20260919T104157403241Z`) without any refit
  by `VALIDATION/dayshift_hedge_run.py`. Each fold reads its rates and wrong-date taxi
  from its own training window only. Paired day-bootstrap differences against
  row 34 (95% interval):

  | Run | Next-month | Longer-gap | Combined | Moved |
  |---|---:|---:|---:|---|
  | 37 Dayshift-hedge (`v3_20260919T112057144323Z`) | −1.43 [−8.83, +5.54] | −7.65 [−18.83, +0.00] | −4.71 [−13.08, +1.90] | 13 flights (81 fold/seed rows) |
  | First run, superseded (`v3_20260919T111742090180Z`) | +0.15 [−8.12, +8.72] | −7.26 [−17.86, +0.00] | −3.76 [−12.09, +2.98] | 13 flights (81 fold/seed rows) |

  All readouts are inconclusive; the seeds are identical (only rule rows move) and
  only LIRF changes. In validation the wrong-date flights (ITYTY630, ITYTY680,
  BAWW561D, VOE6BB) gain more than the typed-in ones (QFAQFA6 twice, ITYTY674,
  WMTT5232) lose. In the first run the format rate was not shrunk, so in early
  windows a clean-number group of one or two wrong-date rows gave p = 1.0, a full
  switch instead of an in-between guess. The shrinkage was added after seeing that
  run, so this CV is optimistic. With so few flights, this row is a bet, not a
  measured gain. In the ranking set it moves five rows (p = the lean toward the
  wrong date):

  | Flight | p | v34 prediction | v37 prediction |
  |---|---:|---:|---:|
  | WMTT3508 | 0.32 | 56,520 s (15.7 h) | 66,333 s (18.4 h) |
  | SVII3006 | 0.63 | 60,656 s (16.8 h) | 77,660 s (21.6 h) |
  | ITYTY672 | 0.71 | 61,322 s (17.0 h) | 79,928 s (22.2 h) |
  | NOSOS431 | 0.57 | 111,654 s (31.0 h) | 97,948 s (27.2 h) |
  | QFAA7006 | 0.32 | 87,484 s (24.3 h) | 87,483 s (both labels ≈ 24.3 h) |

  Before upload, the expected leaderboard change was about −4 to −5, with outcomes
  from about −10 (all four wrong-date) to about +5 (all four typed in).
- **v37 (row 40):** prepared 2026-09-19 by `VALIDATION/dayshift_hedge_submit.py
  prepare` from v34's saved ranking predictions, which the script checks still
  reproduce the scored v34 file byte for byte, and uploaded 2026-09-19 at 12:52 UTC,
  the third upload that day. Only the five rows above move; every other row matches
  v34. Official score 280.1743 on all 344,841 rows: **−0.29 against v34**, inside
  the expected range, and rank 39 of 166 (unchanged). The score was not used to
  infer which way any single flight went.
- **Row 41:** CV from run `v3_20260919T211914721668Z` (60.5 minutes, macOS), built by
  `VALIDATION/opdi_run.py`. Row 30 is the control and was not refitted: its boosters were
  reloaded from run `v3_20260918T151436598224Z` and reproduced row 30 exactly on every
  fold and seed. The run also saves each column after row 33's window and row 34's cap;
  its row-34 column scores 253.05 / 228.98 / 239.94, reproducing row 34. Paired
  day-bootstrap differences against row 34 (95% interval):

  | Candidate | Next-month | Longer-gap | Combined |
  |---|---:|---:|---:|
  | Row 41, both parts | −2.84 [−3.21, −2.49] | −2.24 [−2.70, −1.77] | −2.52 [−2.87, −2.17] |
  | Regressor only, classifier unchanged | — | — | −2.51 [−2.84, −2.18] |
  | Before the window and cap (row 30 + OPDI vs row 30) | −2.75 [−3.12, −2.38] | −2.25 [−2.73, −1.76] | −2.48 [−2.84, −2.11] |

  All three seeds agree on every headline and no airport is worse by more than 1 s. The
  gain sits entirely in the regressor. It is concentrated where the ground tracks exist:
  at LSZH RMSE falls 183.6 → 139.9 and at EDDF 177.5 → 170.6 (seed 42, next-month folds),
  together 82% of the gain, and 95% of the gain is on the 14% of flights OPDI saw moving.
  Every validation month improves. The inputs were fixed before this run. None of the
  folds contains January.
- **Row 42:** CV from run `v3_20260919T223006736822Z`, derived from row 41's run without any
  refit by `VALIDATION/opdi_combined_run.py`. It applies row 37's three corrections to both
  of that run's row-34 columns, using the parameters fitted in run
  `v3_20260919_artifact_experiments`; those read only each training window's labels and
  observable columns, never the model, and this run checks that its window late means match.
  In all six windows the handler groups fall back to the pooled day weight, so the two
  arrival-handler signals cannot change a prediction. The derived row-37 column scores
  249.9317 / 224.8927 / 236.3092 and its change against row 34, −3.63 [−7.83, −0.44], matches
  the v38 study's −3.6317, so the derivation reproduces row 37. Paired day-bootstrap
  differences, row 42 minus row 37 (95% interval): next-month **−2.88 [−3.24, −2.53]**,
  longer-gap **−2.28 [−2.76, −1.80]**, combined **−2.56 [−2.91, −2.21]**; all three seeds
  agree and no airport is worse by more than 1 s. The OPDI inputs and the v38 corrections
  change different flights (498 fold/seed rows move under the corrections, all at Rome and
  LFPG, where OPDI sees no ground tracks), so the two gains add. Against the leaderboard's
  current best, v38 at 279.3805, the expected change is about **−2**, from Zurich and
  Frankfurt in July 2026.
- **v39 (row 42)** prepared and uploaded 2026-09-19 at 23:05 UTC by `VALIDATION/opdi_submit.py`:
  one seed-42, 800-round refit of both boosters on all of 2025 with the OPDI inputs
  (1.6 minutes), the late mean 1,079.5 s and the correction parameters fitted on the same
  year (day blend 0.324989, as in v38). Each ranking month sees all of 2025 plus its own
  month only. The corrections move the same kinds of rows as v38 (16 twin records, 5 day
  blends, one 12-hour clock anchor) and the four-hour Rome rule overrides the same 43 rows.
  OPDI ground tracks reach 15.0% of ranking departures. Against v38's file, 339,423 rows
  differ by 33.5 s on average (142 by more than 10 minutes): the model itself was refitted,
  so every prediction moves a little, as it did between v34 and v38. The manifest is
  `submission_v39.json` in the run directory. Official score **277.3629** on all 344,841
  rows: **−2.02 against v38**, against a CV change of −2.88 next-month and −2.56 combined,
  and rank **33 of 166** (from 37). The estimate before upload was about −2. The gain came
  from ordinary flights at the two airports with ground coverage, so it transferred at
  roughly the ratio CV predicted. **The team is still to confirm the use of open
  ground-track data with the organisers**; the basis so far is the organiser's Discord
  answer of 2026-09-18, recorded second-hand in another team's public repository.
- **Row 43:** CV from run `v3_20260920T152734631417Z` (three seeds, all 16 folds, 25.5 minutes
  on resume), built by `VALIDATION/opdi_run.py --exclude-rule-rows`, then row 37's corrections
  applied without refit by `VALIDATION/opdi_combined_run.py --source-run` as run
  `v3_20260920T170600876190Z`. Row 42 is the control and was not refitted: its arm reproduces
  row 42's entry exactly (247.0477 / 222.6081 / 233.7470), and the source run's per-window
  `late_mean` parameters are identical to it, confirming only the model changed. Paired
  day-bootstrap differences against row 42 (95% interval): next-month **−1.0660
  [−1.5774, −0.6090]**, longer-gap −0.0279 [−0.5458, +0.4887] (inconclusive), combined
  **−0.5132 [−0.9152, −0.1167]**; all three seeds agree on both headlines. The 441
  rule-triggered evaluation rows are **byte-identical** between the two arms, since the rule
  overwrites them either way, so removing them from training costs nothing there; the gain
  lands on ordinary flights (RMSE 185.01 → 183.55 on the 8.04 M fold/seed rows under 30
  minutes). Screened first on row 41's stack at −0.4482 combined. A blunt version that instead
  dropped **every** label above 7,200 s regresses by +6.48 combined: it gains −4.65 s on
  ordinary rows but loses far more on the 574 large-label rows no rule owns. Details in
  [TRAINING_LABEL_FILTER.md](VALIDATION/reports/TRAINING_LABEL_FILTER.md). None of the folds
  contains January.
- **v40 (row 43):** prepared and uploaded 2026-09-20 at 18:15 UTC by
  `VALIDATION/opdi_submit.py --exclude-rule-rows`, the first upload of that UTC day. One
  seed-42, 800-round refit of both boosters on all of 2025 excluding exactly **127**
  rule-covered departures (fitting on 2,084,532 rows, 1.46 minutes); the late mean 1,079.5 s
  and day blend 0.324989 are unchanged from v39, because the correction parameters are fitted
  before the rows are dropped. Each ranking month sees all of 2025 plus its own month only.
  The four-hour rule overrides the same 43 ranking rows. Against v39's file, 338,789 rows
  differ by 29.0 s on average (median 17 s), but the **mean signed change is −0.014 s**: the
  model was refitted, so every prediction moves a little, without bias. Official score
  **275.4146** on all 344,841 rows: **−1.95 against v39**, against a CV change of −1.07
  next-month and −0.51 combined, and rank **31 of 167** (from 33). The leaderboard moved 3.8×
  CV's prediction, but v40's LB-minus-CV offset of +42.18 sits inside the established
  +42.8 ± 1.5 band, so the direction agreed while the magnitude is not separable from transfer
  noise on a single upload. The manifest is `submission_v40.json` in the run directory.
- **Row 44:** CV from run `v3_20260924T202324352147Z` (three seeds, all 16 folds, 41.5 minutes
  on macOS), built by `VALIDATION/occupancy_run.py`, then row 37's corrections applied without
  refit by `VALIDATION/occupancy_combined_run.py --source-run` as run `v3_20260924T211259763470Z`.
  Row 43 is the control and was not refitted: its boosters were reloaded from row 43's run and
  reproduced it exactly on every fold and seed, and the corrected arm reproduces row 43's entry
  to four decimals (245.9817 / 222.5802 / 233.2338). Paired day-bootstrap differences against
  row 43 (95% interval): combined **−0.2924 [−0.4641, −0.1088]**, longer-gap **−0.3798
  [−0.6022, −0.1665]**, next-month −0.1937 [−0.4439, +0.1039] (inconclusive). No airport opposes
  the combined headline. **Two warnings were recorded before the upload and both proved right.**
  The seeds disagree on next-month (+0.071 / −0.549 / −0.103), and 8 of the 10 ageing differences
  are positive, one excluding zero (B8_vs_A5 +0.54 [+0.01, +1.08]) — the opposite of row 28's
  queue inputs, whose ageing differences all included zero and whose gain transferred in full.
  Before the upload the expected change was about −0.3, with the fading caveat stated. Screened
  first by a cross-fitted, month-held-out residual stack on A folds at −0.2533 s, which is what
  the run then measured; the univariate gate the inputs failed (they rank taxi length at +0.221
  within airport against `q_apt_others_gap_prev60`'s +0.227) did not predict the CV outcome.
  None of the folds contains January.
- **v41 (row 44):** prepared and uploaded 2026-09-24 at 21:19 UTC by
  `VALIDATION/occupancy_submit.py --exclude-rule-rows`, the first upload of that UTC day. One
  seed-42, 800-round refit of both boosters on all of 2025 with the occupancy inputs, excluding
  the same **127** rule-covered departures (fitting on 2,084,532 rows, 110 prepared inputs); the
  late mean 1,079.5 s and day blend 0.324989 are unchanged from v39 and v40, and the four-hour
  rule overrides the same 43 ranking rows. Each ranking month sees all of 2025 plus its own month
  only. Against v40's file, 339,136 rows differ by 28.2 s on average (largest 1,257 s, an order of
  magnitude tighter than v40-against-v39's 10,062 s), so no single row drives the result. Official
  score **275.9592** on all 344,841 rows: **+0.5446 against v40**, against a CV change of −0.29
  combined and −0.38 longer-gap. **The supported CV gain reversed on the leaderboard.** This is
  the first such reversal in the ladder above, and it is consistent with the ageing readouts: the
  ranking months sit up to seven months past the training cutoff, where the fade the B-folds
  measured is largest, while CV's next-month score — the one closest to the test months — was
  already inconclusive. v40 stayed the best submission until v42; rank unchanged at 36 of 189. The
  manifest is `submission_v41.json` in the run directory.
- **Row 46:** CV from run `v3_20260925_stand_coords/main` on macOS (Apple M3, three seeds,
  all 16 folds, 75.0 minutes). The three inputs are the stand's east/north offset from the
  airport's runway-end centre and its straight-line distance to the end of the take-off runway,
  from the X-Plane Gateway and OpenStreetMap stand maps (99.2% of training departures located).
  The v42 artifacts are on the WSL desktop, so the control was rebuilt here with v42's
  byte-identical code and the original OPDI cache. It scores 245.5594 / 221.9008 / 232.6754 /
  263.6158, within 0.03 s of row 45. Paired day-bootstrap differences against that control
  (95% interval, 1-day blocks; 3-day blocks agree): Weighted CV **−0.4682 [−0.649, −0.268]**,
  weighted next-month −0.282 [−0.487, −0.056], weighted longer-gap −0.619 [−0.881, −0.356],
  original combined −0.506 [−0.607, −0.402]. All three seeds improve on every one of these
  (Weighted CV −0.34 / −0.65 / −0.41). January retrospective +0.19 [−0.33, +0.53], inconclusive.
  **Ageing does not repeat row 44's warning:** 7 of 10 ageing differences are negative, three
  of them excluding zero. One is positive and excludes zero: B10_vs_A6 +0.77 [+0.33, +1.29].
  The gain is concentrated at LFPG (−1.47), LEMD (−0.70), LTFM (−0.65) and EHAM (−0.58);
  EDDF, EDDM and LSZH move +0.19 to +0.21 (untested per airport). A separate hidden-stand test
  removes 10% of stands from each window's training and blanks their name and 90-day history
  at prediction, as for a stand first used in 2026. There (seed 42), the inputs cut RMSE on
  those flights by **2.01 s [−2.28, −1.74]**, at every airport. Weighted CV comes from
  this Mac's copy of the adopted weights: its four discriminators are byte-identical to the
  record, and its effective sample size matches to 1e-16. Design, frozen before the fits:
  [STAND_COORDINATES_EXPERIMENT.md](VALIDATION/docs/STAND_COORDINATES_EXPERIMENT.md).
- **v43 (row 46):** prepared and uploaded 2026-09-25 at 12:14 UTC on macOS by
  `VALIDATION/stand_coords_submit.py`, the first upload of that UTC day. One seed-42, 800-round
  refit of both boosters on all of 2025 with the stand inputs (ordinary 91.3 s; 3.3 minutes in
  all), excluding the same **127** rule-covered departures (2,084,532 rows). Late mean
  1,079.5 s and correction parameters are unchanged; the four-hour rule overrides the same 43
  ranking rows, whose predictions are identical to v42's. 98.99% of ranking departures have a
  stand position. Against v42's file, 338,708 rows differ by 27.7 s on average (median 18 s,
  largest 690 s, mean signed −0.26 s). Official score **274.7660** on all 344,841 rows:
  **−0.1416 against v42**, against a CV change of −0.47 weighted and −0.51 combined, so it
  transferred at about 0.3×. The file was built with the original OPDI cache used by row 46's
  CV, while v42 was built with the rebuilt cache on the WSL desktop. So the step also contains
  that cache and platform difference; on CV the two controls differ by only 0.02 s. The
  manifest is `submission_v43.json` in the run directory. External data still to declare: X-Plane
  Scenery Gateway airport data (GPL v2 or later) and OpenStreetMap (ODbL).
- **Row 29:** CV from run `v3_20260918T120331999764Z` on macOS (45.2 minutes).
  Row 28 is the control and was not refitted: its boosters were reloaded from
  row 28's run and reproduced its saved predictions exactly on every fold and seed.
  Every candidate in the run changes one thing. Paired day-bootstrap differences in
  combined CV against row 28 (95% interval; all three seeds agree on every row):
  late-row estimate −1.05 [−2.01, −0.25]; classifier + airline −1.23 [−1.79, −0.69];
  classifier + arrival handler share −1.15 [−1.58, −0.71]; classifier +
  flight-number format −0.11 [−0.27, +0.03]; regressor + airline −0.47
  [−0.72, −0.23]; classifier with all three inputs −1.37 [−1.94, −0.79].
  Row 29 is all of these together: −5.04 [−7.18, −3.06] next-month,
  −1.18 [−2.54, +0.00] longer-gap and −3.00 [−4.24, −1.83] combined. A tested
  day-shift correction for 14 h+ rule rows scored −4.59 [−13.88, +2.97]. Its
  result depends on about 11 flights, so it is not part of row 29. Row 29's
  column is derived exactly from the run's `all` column, because the day-shift
  changes only rule rows, which otherwise keep the rule's gap prediction. Every
  change was chosen after exploring row 28's saved predictions on these folds, so
  this CV is optimistic. None of the folds contains January. Details in
  [ROME_SIGNALS.md](VALIDATION/reports/ROME_SIGNALS.md).
- **v29 (row 29):** uploaded 2026-09-18 at 14:42 UTC by
  `VALIDATION/rome_signals_submit.py`, the fourth upload that day. One seed-42,
  800-round refit on all of 2025 of the ordinary regressor with the airline input
  and the Rome classifier with all three signals; the late mean is 1,079.5 s from
  654 training rows, and the day-shift is off. Each ranking month sees all of 2025
  plus its own month only, so its handler shares come from that month's own
  arrivals. The script refuses to run if any source or data file differs from run
  `v3_20260918T120331999764Z`. Against v28's file: the four-hour rule overrides the
  same 43 ranking rows with identical predictions; the late estimate applies to 305
  rows and lowers them by 599 s on average; the median change elsewhere is 17 s.
  Official score 314.4667 on all 344,841 rows: **−8.16 against v28**, against a CV
  change of −5.04 next-month and −3.00 combined. The leaderboard gain is larger than
  CV's, in line with the ranking set holding more of the targeted rows (0.088% of
  rows against 0.056% in CV).
- **Rows 30–32:** CV from run `v3_20260918T151436598224Z` on macOS (102.9 minutes).
  Row 29 is the control and was not refitted: its boosters were reloaded from row
  29's run and reproduced row 29 exactly on every fold and seed, with the same late
  mean in every window. Each row adds its inputs to both the regressor and the
  classifier; the late-row estimate stays on and the day-shift stays off. Paired
  day-bootstrap differences against row 29 (95% interval; all three seeds agree on
  every non-inconclusive readout):

  | Row | Next-month | Longer-gap | Combined |
  |---|---:|---:|---:|
  | 30 Stand-clock | −4.15 [−4.68, −3.66] | −3.56 [−4.19, −2.98] | −3.83 [−4.29, −3.39] |
  | 31 Dirty-day | +0.99 [+0.10, +2.03] | −0.49 [−1.15, +0.11] | +0.21 [−0.36, +0.81] |
  | 32 Stand-clock-dirty-day | −2.68 [−4.20, −0.87] | −3.89 [−4.72, −3.15] | −3.32 [−4.21, −2.32] |

  Row 30 improves every airport on both readouts (next-month: EGLL −7.78, LTFM
  −6.73, LFPG −5.87, LIRF −3.90) and is better on all 16 folds. Most of its gain is
  on ordinary flights with a flight ID at the nine other airports (−2.30% of row
  29's squared error), not on corrupted rows. Row 31 is a supported next-month
  regression. In row 32, LEMD (+1.80) and LIRF (+1.09) oppose the next-month
  headline. Two more columns split row 32's inputs by component: regressor only
  257.6287 / 229.3032 / 242.2566 (−3.36 combined), classifier only 260.3337 /
  233.2898 / 245.6343 (+0.01, inconclusive), so the gain sits in the regressor.
  Row 30's ageing differences are mixed (smaller gains against November, larger
  against December), with no consistent fade. The inputs and the five candidates
  were fixed before this run; choosing between rows 30 and 32 uses these results.
  None of the folds contains January.
- **v30 (row 30):** uploaded 2026-09-18 at 17:22 UTC by
  `VALIDATION/stand_day_submit.py`, the fifth upload that day. One seed-42,
  800-round refit on all of 2025 of row 29's ordinary regressor and Rome classifier,
  each with the five stand clock inputs added; the late mean is 1,079.5 s from 654
  training rows, as for v29, and the day-shift is off. Each ranking month sees all
  of 2025 plus its own month only, so its stand clock reads that month's own
  arrivals; it is present for 99.99% of ranking departures. The script refuses to
  run if any source or data file differs from run `v3_20260918T151436598224Z`.
  Against v29's file: the four-hour rule overrides the same 43 ranking rows with
  identical predictions, the late estimate applies to the same 305 rows, and the
  median absolute change is 21 s (mean +0.2 s). Official score 309.9872 on all
  344,841 rows: **−4.48 against v29**, against a CV change of −4.15 next-month and
  −3.83 combined. Like the weather and queue inputs, this gain on ordinary flights
  carried over to the leaderboard in full.
- **Row 33:** CV from run `v3_20260918T181924599658Z`, derived from row 30's saved
  predictions without any refit by `VALIDATION/plan_window_run.py`; its
  `pred_stand` column reproduces row 30 exactly. The script first checks the rule
  on every 2025 departure: all 2,062,577 with a flight ID and an LOBT have their
  label inside the window (largest distance from takeoff − LOBT: 60.1 minutes).
  The window moves 1,193 of 8,353,590 fold/seed predictions, and every moved
  prediction got closer to its label. Paired day-bootstrap differences against
  row 30 (95% interval; all three seeds agree): −2.24 [−5.13, −0.57] next-month,
  −0.40 [−0.79, −0.12] longer-gap and −1.26 [−2.71, −0.43] combined; no airport
  opposes either headline. The rule was read off the 2025 labels and held on
  every one of them. None of the folds contains January.
- **v33 (row 33):** prepared 2026-09-18 by `VALIDATION/plan_window_submit.py
  prepare` from v30's saved ranking predictions, which the script checks still
  reproduce the scored v30 file byte for byte, and uploaded 2026-09-19 at 00:12 UTC,
  the first upload that day. Against v30's file, 58 ranking rows move (52 raised, 6
  lowered; EHAM 25, LFPG 12, LIRF 6, five other airports 15) and the 43 Rome rule
  rows do not. One row carries most of the expected change: ITY340 (LIRF → LFMN,
  scheduled 2026-07-16 19:35 UTC, LOBT 21:44 UTC, takeoff recorded 2026-07-17
  21:51 UTC) moves from 12,930 s to 83,160 s, the lowest label its window allows.
  Before upload, that row alone was expected to lower the official score from
  309.99 to roughly 281–286 if its label lay inside the window, as every 2025
  label did. Official score 282.4525 on all 344,841 rows: **−27.53 against v30**,
  against a CV change of −2.24 next-month and −1.26 combined, and rank 40 of 163
  (from 79). Almost all of the leaderboard gain is ITY340. The CV folds hold one
  similar row (AEZ146R, July 2025, label 24.2 h), but there row 30 already
  predicted 18.1 h, so the window added far less.
- **v17–v21:** CV from `VALIDATION/runs/overnight_20260917/`, whose regenerated
  Q0-4h control scored 296.2717 / 255.6100 / 274.3622 (a small replay
  difference from v15's row). See the
  [overnight summary](VALIDATION/reports/OVERNIGHT_SUMMARY_20260917.md).

**Schedule-recorded rows:** Rome training departures whose taxi-time label equals
takeoff minus scheduled time within 60 seconds, meaning off-block was logged at
the scheduled time. The classifier is trained on these; prediction uses only
fields available in the ranking data.
From row 22 on, the **classifier** is this Rome classifier, p is its probability
that a row is schedule-recorded, and gap is takeoff minus schedule. From row 23
on, the **regressor** is the model trained without schedule-recorded rows.

**Rome rule:** if `ADEP_mvt == 'LIRF'`, the original `FLIGHT_ID_mvt` is missing,
and `MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` is at least the threshold, predict
that gap instead of the model's output. Thresholds are inclusive: six hours
is 21,600 seconds; four hours is 14,400 seconds. Otherwise keep the model prediction.

**R3 recovery:** fill eligible missing flight-plan fields from uniquely matched
arrivals using exact and relaxed matching, then fill reliable stable categories
from history. Existing raw values take precedence. It does not repair taxi-time
labels or invent off-block timestamps.

**Removed history inputs (v18):** `op_taxi_30d`, `op_taxi_30d_n`, `rwy_taxi_30d`,
`stand_taxi_90d`, `aphour_taxi_30d`. Arrival and traffic inputs remain.

**Clock inputs (v19):** four signed gaps between off-block clocks, availability
flags for each clock and for missing flight ID/plan/AOBT3, the count, span, spread
and median of the takeoff-minus-clock estimates, and a fixed clock baseline with
its source. See [CLOCK_MODELS.md](VALIDATION/docs/CLOCK_MODELS.md).

**Weather inputs (row 27):** from the airport's latest METAR report at or before
takeoff, if at most 90 minutes old: temperature, dew-point spread, wind speed,
gust and direction, visibility, cloud ceiling, flags for thunder, snow,
freezing, fog, rain, heavy precipitation and de-icing conditions, and the
report's age; plus, over the two hours before that report, lowest visibility
and ceiling, highest gust, and whether thunder, snow or freezing, or fog occurred.
Reports come from the Iowa Environmental Mesonet archive (public domain;
`VALIDATION/tools/fetch_metar.py`); code in `VALIDATION/weather.py`.

**Runway queue inputs (row 28):** the traffic around each flight's own takeoff,
from movement timestamps that `ranking.parquet` still carries. Before takeoff:
seconds since the previous takeoff on the same runway, takeoffs on that runway
and at the airport in the previous 30 minutes, the mean takeoff-minus-AOBT3 of
departures that left that runway and that airport in the previous 60 minutes,
airport takeoffs between this flight's own AOBT3 and its takeoff, and landings on
that runway in the previous 30 minutes. After takeoff: seconds until the next
takeoff on the same runway, takeoffs on it in the next 30 minutes, and the same
takeoff-minus-AOBT3 means over 30 minutes either side. Every window excludes the
takeoff instant, so a flight never counts itself; runways are scoped by airport;
departure block times and taxi-out labels are never read. Code in
`VALIDATION/runway_queue.py`; run by `VALIDATION/runway_queue_run.py`.

**Taxi occupancy inputs (row 44):** how many *other* aircraft are on the tarmac at a
given instant, from the same supplied movement timestamps the queue inputs use. A
departure occupies the tarmac over [`AOBT_3`, takeoff), counted only when both
timestamps are present and the duration is within 0–7,200 s — the same filter
`runway_queue.py` applies to its takeoff-minus-AOBT3 means. Four inputs: the count at
this flight's own off-block and again at its takeoff, each at airport and runway scope.
A flight never counts itself, runways are scoped by airport, and departure block times
and taxi-out labels are never read. The motivation is that the existing queue inputs
measure *flow*, which turns around when an airport jams because fewer aircraft get off
the ground: ordered by actual taxi length, `q_apt_takeoffs_prev30` rises to the 0.559
within-airport percentile at 25–30 minutes and falls back to 0.466 beyond an hour.
Occupancy measures *stock* instead, but it is capped by apron throughput (at most 43
aircraft airport-wide) and so saturates too, peaking at 0.619 and falling to 0.556,
while the duration input `q_apt_others_gap_prev60` keeps climbing (0.428 → 0.692 →
0.663). Code in `VALIDATION/taxi_occupancy.py`; run by `VALIDATION/occupancy_run.py`,
corrections by `VALIDATION/occupancy_combined_run.py`, file prepared by
`VALIDATION/occupancy_submit.py`.

**Rome signals (row 29):** the airline is read from `FLIGHT_mvt`: the first three
characters if they are letters, otherwise the first two. Rows without a flight
ID lack `operator`, so this is the only airline these rows carry. The arrival
handler share is the share of that airline's arrivals at the airport in the same
UTC month whose in-block (landing plus taxi-in) equals the schedule within 60
seconds, shrunk toward the airport's monthly share, plus the number of those
arrivals. The flight-number format flags Rome's mangled numbers (ICAO prefix plus
the last five callsign characters, e.g. `ITYTY630`). The late-row estimate
reflects how Rome loses flight IDs: a departure whose real off-block was recorded
loses its flight ID only when it left more than an hour after schedule. For those
rows, the part of the blend that is not the gap becomes the mean taxi of such
late departures in the training window (1,009–1,120 s), capped at the gap minus
one hour. Code in `VALIDATION/rome_signals.py`; run by
`VALIDATION/rome_signals_run.py`.

**Stand clock inputs (rows 30 and 32):** from arrivals at the same airport and
stand, with in-block = landing plus taxi-in: seconds from the last in-block before
takeoff to takeoff, the same for the arrival before it, whether that last arrival
has the same aircraft type and the same airline as the departure, and its in-block
minus the departure's schedule. If the last arrival was another aircraft, this
one had already left, so the taxi was at least that long; if it was this
aircraft, the taxi was at most that long.

**Dirty-day inputs (rows 31 and 32):** for the airport's UTC day of takeoff, the
share of other departures without a flight ID and without an aircraft type, the
share of arrivals whose in-block equals their schedule within 60 seconds, the mean
takeoff-minus-schedule of other departures with a flight ID (each clipped to
0–3 h) and the share of them more than 45 minutes late; plus the share of
departures without a flight ID within 90 minutes either side of takeoff. A flight
never counts itself. Departure block times and taxi-out labels are never read by
either input set. Code in `VALIDATION/stand_day.py`; run by
`VALIDATION/stand_day_run.py`.

**Flight-plan window (row 33):** a departure's flight ID links it to a flight plan
only when its recorded off-block is within an hour of the plan's last off-block
time (`LOBT_flt`); over all 2025 departures with a flight ID the largest distance is
60.1 minutes, at every airport. The label is takeoff minus recorded off-block, so it
lies within takeoff − LOBT ± 61 minutes. A prediction outside that window moves to
its nearest edge. Rows without a flight ID or an LOBT keep their prediction, so
every Rome rule row does. No label is read. Code in `VALIDATION/plan_window.py`;
validated by `VALIDATION/plan_window_run.py`; file prepared by
`VALIDATION/plan_window_submit.py`.

**No-ID cap (row 34):** a departure without a flight ID was recorded leaving its
stand more than an hour away from its flight plan (see row 33), and it carries no
plan, so the schedule stands in for it. Outside Rome, 99.0% of 2025 no-ID
departures that took off more than 75 minutes after schedule have a label of at
most (takeoff − schedule) − 1 hour; LTFM is the exception at 92.4%. So, outside
LIRF, a no-ID prediction is lowered to at most (takeoff − schedule) − 1 hour when
the flight took off more than 75 minutes after schedule (not at LTFM), and to at
most takeoff − schedule when more than 30 minutes after. Predictions are never
raised; Rome keeps its own no-ID model. Code in `VALIDATION/record_fixes.py`.

**Twin records (row 35):** LFPG sometimes logs one takeoff twice: two departures
with the same airport and flight number whose takeoffs are at most 3 minutes
apart, usually from different stands (a tow). The first stand's off-block is near
the plan, so that record keeps the flight ID and a long label; the real pushback
came more than an hour after LOBT, so the other record loses its ID and has an
ordinary label. In 2025 there are 110 such clusters (107 at LFPG). In the 102
two-record clusters with exactly one flight ID, that record's label is close to
takeoff − LOBT (median 5.5 minutes below it) and is the longer of the two in 97% of
pairs. That record's prediction becomes takeoff − LOBT − 330 seconds; the other
record keeps its prediction. Code in `VALIDATION/record_fixes.py`.

**Seed average (row 36):** each flight's prediction is the mean of row 33's
predictions from seeds 42, 142 and 242 in the same fold. Nothing else changes.
Validation already holds all three seeds; a ranking file needs two more full-year
refits of row 30's recipe (seeds 142 and 242), then v33's window.

Rows 34–36 are validated by `VALIDATION/record_fixes_run.py`.

<!-- rome-calibration:v3_20260919_rome_calibration -->
The two Rome-calibration rows above are local validation experiments from `v3_20260919_rome_calibration`. Their probability calibrators use inner chronological predictions inside each training window. Both remove Rome’s hard four-hour override; the linked report separates that change from calibration. All 16 folds and three seeds remain. No full-year fit or leaderboard upload was performed.

**Day-shift hedge (row 40):** a LIRF departure without a flight ID that took off at
least 14 hours after schedule, on a later local (Europe/Rome) day, gets one of two
labels in 2025: takeoff − schedule when the schedule was typed in as the off-block,
or days × 86,400 s + taxi when the real clock time was typed with the schedule's
date. Example: Jet2 EXSXS3AD and EXSXS1GT, both about 16 hours late on the night of
24–25 January 2025 from stands 612 and 613, got one of each (56,633 s and 87,177 s).
Nothing visible separates the two, so the prediction becomes their
probability-weighted mean. The probability of the wrong date is the share among
labelled training rows of this kind, shrunk three times with weight 2: overall
toward 0.5, flight-number format (mangled or clean) toward overall, airline toward
format. The taxi is the median of the wrong-date rows (1,080 s on all of 2025). On
all of 2025, 11 of 19 such rows have the wrong date (mangled 9 of 14, clean 2 of 5;
WMT 0 of 2, ITY 3 of 4, NOS 1 of 2). Code in `VALIDATION/dayshift_hedge.py`;
validated by `VALIDATION/dayshift_hedge_run.py`; file prepared by
`VALIDATION/dayshift_hedge_submit.py`.

**Rule-covered training rows (row 43):** the four-hour Rome rule replaces the
model's prediction whenever `ADEP_mvt` is LIRF, the original `FLIGHT_ID_mvt` is
missing and takeoff minus schedule is at least 14,400 s. Over all of 2025 that is
**127 departures**, median label 20,887 s. Their labels are recording artifacts the
model cannot learn — every input describes an ordinary flight — and the rule
overwrites them at prediction time, so fitting them only distorts the rest. They are
dropped from the model fit only: every evaluation row is still scored, and the
categories, context, history and correction parameters still see them. The mask comes
from the rule itself on observable inputs, reading no label and no prediction. Code in
`VALIDATION/opdi_run.py` (`rule_rows`, `--exclude-rule-rows`); file prepared by
`VALIDATION/opdi_submit.py --exclude-rule-rows`.

**OPDI inputs (row 41):** [OPDI](https://www.opdi.aero) is EUROCONTROL's Open
Performance Data Initiative, which turns OpenSky Network ADS-B positions into per-flight
events: first seen, leaving a parking position, entering a taxiway, apron, de-icing pad
or runway, take-off. A departure is matched to an OPDI flight at the same airport with
the same callsign whose track spans its takeoff. Ten inputs describe the flight itself,
in seconds before takeoff: first seen on the ground, leaving its own stand (the stand in
`STAND_mvt`), leaving any parking position or apron, entering a taxiway, entering the
runway, time spent on de-icing pads, and the earliest of these ground sightings; plus
whether a flight matched at all. Four more describe the airport's other departures
around it: their mean earliest ground sighting in the hour before and within half an
hour either side, how many of them there are, and the share that used a de-icing pad. No
label and no block time is read; the events are open data and must be declared.

Ground coverage decides the value and is very uneven: in July 2025 OPDI saw 96% of
Zurich departures moving and 55% of them leaving their own stand, but 0% at Heathrow,
Paris, Rome and Istanbul. In the ranking months July 2026 matches July 2025 (LSZH 97%,
EDDF 65%), while January 2026 is thinner (LSZH 61%, EDDF 44%) and OPDI's file for
2–9 January 2026 holds only 1 January. The file for 2–11 October 2025 is listed on the
OPDI site but returns 404, so those days have no ground events either. Coverage also
drifts within 2025 (EDDF 6–11% before September, 37–55% after; LEMD 85% in November
alone), which the aging readouts pick up: models three or more months old gain about
2.7 s less on November.

Fetched by `VALIDATION/tools/fetch_opdi.py`, built into one table per departure by
`VALIDATION/opdi.py`, validated by `VALIDATION/opdi_run.py`, tests in
`VALIDATION/tests/test_opdi.py`. The organiser confirmed on Discord (2026-09-18) that
off-block times extracted from open trajectory data are allowed; the team is checking
this directly with the organisers before any upload.
