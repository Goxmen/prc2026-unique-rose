# Submitted models

Every file uploaded to `prc-2026-unique-rose`, from the live leaderboard check
at 2026-09-19 10:54 UTC (`pipeline/leaderboard_status.py`).
There are 28 uploads: 26 scored and 2 rejected by the five-per-day limit.
**Best and latest: v34, 280.4658 s.** Rank 39 of 166 scoring teams.

Rows 22–28 are our Rome substitution mixture and hybrids. Rows 22–25 have
validation scores only and have not been submitted. Rows 26–30 were uploaded as
v26–v30 and row 33 as v33, so each file number matches its row; v22–v25 and
v31–v32 were never used. Row 29
adds Rome recording signals to row 28. Rows 30–32 add stand clock and dirty-day
inputs to row 29; rows 31–32 have validation scores only. Row 33 applies the
flight-plan window to row 30's predictions after the model (no refit). Rows 34–36
each make one change to row 33's predictions; row 34 was uploaded as v34, and rows
35–36 have validation scores only.

## Codenames and recipes

Unless a row says otherwise, the models from v3 onwards are LightGBM fits on all
twelve months of 2025 with seed 42, the original 27 model inputs (raw movement
and flight-plan columns, plus departure-taxi history, arrival taxi-in history and
traffic counts), 800 rounds and a training-label filter of 1–200,000 seconds.

| Order | Codename | Actual recipe | CV next-month | CV longer-gap | CV combined | Official LB | Submission |
|---|---|---|---:|---:|---:|---:|---|
| 1 | **Airport-mean** | No model: 2025 mean taxi-out per airport (labels 1–7,200 s); global mean fallback | — | — | — | 661.5737 | v1 |
| 2 | **LGBM-raw** | LightGBM, raw columns only (`pipeline/train.py`); 3,000 rounds; labels **1–7,200 s** | — | — | — | 533.3759 | v2 |
| 3 | **Keep-tail** | 27 inputs; 800 rounds; labels widened to **1–200,000 s** | — | — | — | 405.6987 | v3 |
| 4 | **FPL-flag** | Keep-tail + `has_flight_plan` input | 366.5209† | 333.9163† | 348.7321† | 405.7838 | v4 |
| 5 | **Ceiling-10k** | Keep-tail, labels capped at **10,000 s** | — | — | — | 518.3068 | v5 |
| 6 | **Ceiling-16k** | Keep-tail, labels capped at **16,000 s** | — | — | — | 498.3336 | v6 |
| 7 | **Ceiling-22k** | Keep-tail, labels capped at **22,000 s** | — | — | — | 486.0517 | v7 |
| 8 | **Ceiling-32k** | Keep-tail, labels capped at **32,000 s** | — | — | — | 472.7133 | v8 |
| 9 | **Ceiling-50k** | Keep-tail, labels capped at **50,000 s** | — | — | — | 433.6797 | v9 |
| 10 | **Ceiling-70k** | Keep-tail, labels capped at **70,000 s** | — | — | — | Rejected (daily limit) | v10 |
| 11 | **Ceiling-90k** | Keep-tail, labels capped at **90,000 s** | — | — | — | Rejected (daily limit) | v11 |
| 12 | **B0-800** | Baseline; no recovery, no rule | 366.7465 | 334.1785 | 348.9770 | 406.4281 | v12 |
| 13 | **Q0** | B0-800 + **6 h Rome rule** | 300.5904 | 256.1971 | 276.7374 | 354.4303 | v13 |
| 14 | **Q1** | B0-800 + **R3 recovery** + 6 h Rome rule | 296.9037 | 258.5505 | 276.1937 | 354.0004 | v14 |
| 15 | **Q0-4h** | B0-800 + **4 h Rome rule** | 296.2785 | 255.6200 | 274.3707 | 344.3114 | v15 |
| 16 | **Q1-4h** | B0-800 + **R3 recovery** + 4 h Rome rule | 292.8785 | 257.4787 | 273.7215 | 344.4821 | v16 |
| 17 | **no_month** | Q0-4h − **`month`** input | 296.4686 | 251.8408 | 272.5082 | 345.8984 | v17 |
| 18 | **no_history** | Q0-4h − **5 departure-taxi history inputs** | 296.5108 | 255.9093 | 274.6319 | 342.5049 | v18 |
| 19 | **clock_features** | Q0-4h + **18 clock inputs** (still predicts taxi time) | 293.6614 | 255.6326 | 273.1284 | 338.1113 | v19 |
| 20 | **small_trees** | Q0-4h with **63 leaves** instead of 255 | 296.6377 | 255.6624 | 274.5645 | 345.3113 | v20 |
| 21 | **rule_routing** | Q0-4h; **rule-eligible flights dropped from training**, rule still predicts them | 286.6291 | 251.4024 | 267.5759 | 342.4009 | v21 |
| 22 | **Rome-mix-base** | B0-800; Rome prediction = **p × gap + (1 − p) × B0-800**; no rule | 311.4125 | 283.5840 | 296.2310 | — | Not submitted |
| 23 | **Rome-mix-ordinary** | Row 22 with the (1 − p) part from the **regressor** | 317.6906 | 294.5641 | 305.0225 | — | Not submitted |
| 24 | **Rome-hybrid** | Row 23 + **6 h Rome rule** override; regressor at all airports | 280.6325 | 250.5677 | 264.3047 | — | Not submitted |
| 25 | **Rome-hybrid-4h** | Row 24 with the **4 h** rule | 280.8670 | 250.4076 | 264.3304 | — | Not submitted |
| 26 | **Rome-hybrid-4h-clock** | Row 25 + **18 clock inputs** (regressor and classifier) | 271.9287 | 246.7919 | 258.2262 | 333.8712 | v26 |
| 27 | **Rome-hybrid-4h-clock-weather** | Row 26 + **21 METAR weather inputs** (regressor only) | 270.3396 | 239.4157 | 253.5766 | 329.1609 | v27 |
| 28 | **Rome-hybrid-4h-clock-weather-queue** | Row 27 + **11 runway queue inputs** (regressor only) | 265.3809 | 234.4385 | 248.6174 | 322.6253 | v28 |
| 29 | **Rome-signals** | Row 28 + **airline** (regressor and classifier); classifier + **handler share**, **flight-number format**; **late-row estimate** for no-ID Rome rows at 1–4 h | 260.3360 | 233.2610 | 245.6201 | 314.4667 | v29 |
| 30 | **Stand-clock** | Row 29 + **5 stand clock inputs** (regressor and classifier) | 256.1836 | 229.7016 | 241.7877 | 309.9872 | v30 |
| 31 | **Dirty-day** | Row 29 + **6 dirty-day inputs** (regressor and classifier) | 261.3248 | 232.7730 | 245.8271 | — | Not submitted |
| 32 | **Stand-clock-dirty-day** | Row 29 + **both input sets** (regressor and classifier) | 257.6550 | 229.3705 | 242.3045 | — | Not submitted |
| 33 | **Plan-window** | Row 30's predictions, each one for a flight with a flight ID kept within **takeoff − LOBT ± 61 min** (post-processing, no refit) | 253.9467 | 229.2985 | 240.5262 | 282.4525 | v33 |
| 34 | **No-ID-cap** | Row 33's predictions; for flights **without a flight ID** outside Rome, at most **delay − 1 h** when more than 75 min late (not LTFM) and at most the **delay** when more than 30 min late (post-processing, no refit) | **253.0535** | **228.9805** | **239.9398** | **280.4658** | v34 |
| 35 | **Twin-records** | Row 33's predictions; where one takeoff is **logged twice** (same airport and flight number, takeoffs ≤ 3 min apart, one record with a flight ID), that record set to **takeoff − LOBT − 5.5 min** (post-processing, no refit) | 253.6312 | 228.9143 | 240.1746 | — | Not submitted |
| 36 | **Seed-average** | Row 33's predictions **averaged over seeds 42, 142 and 242** (a submission needs two more full-year refits) | 252.6779 | 227.9686 | 239.2266 | — | Not submitted |
| 37 | **Artifact-combined** | CV only: row 34's local replay + twin-record fix + handler day blend + confidence exception + 12 h plan-clock anchor ([recipe and results](VALIDATION/reports/RECORDING_ARTIFACT_EXPERIMENTS.md)) | 249.9460 | 224.8911 | 236.3151 | — | Not submitted |

All values are **RMSE in seconds; lower is better**. CV is validation protocol v3:
three-seed loss average, next-month = `R_A`, longer-gap = `R_B_long`, and a
provisional combined summary. CV and official scores are not on the same level.

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

  The weather and queue gains carried over to the leaderboard in full. Only about
  a third of the Rome hybrid's CV gain did. Provenance, including file and model
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
