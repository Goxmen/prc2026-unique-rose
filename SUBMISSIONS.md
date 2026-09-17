# Submitted models

Every file uploaded to `prc-2026-unique-rose`, from the live leaderboard check
at 2026-09-17 12:12 UTC (`pipeline/leaderboard_status.py --json`).
There are 21 uploads: 19 scored and 2 rejected by the five-per-day limit.
**Best: v19, 338.1113 s. Latest: v21, 342.4009 s.** Rank 96 of 152 scoring teams.

Rows 22–26 are our Rome substitution mixture and hybrids. They have validation
scores only and have not been submitted.

## Codenames and recipes

Unless a row says otherwise, the models from v3 onwards are LightGBM fits on all
twelve months of 2025 with seed 42, the original 27 model inputs (raw movement
and flight-plan columns, plus departure-taxi history, arrival taxi-in history and
traffic counts), 800 rounds and a training-label filter of 1–200,000 seconds.

| Order | Codename | Actual recipe | CV next-month | CV longer-gap | CV combined | Official LB | Submission |
|---|---|---|---:|---:|---:|---:|---|
| 1 | **Airport-mean** | No model: mean 2025 taxi-out per airport (labels 1–7,200 s), global mean as fallback | — | — | — | 661.5737 | v1 |
| 2 | **LGBM-raw** | LightGBM on raw columns, no history features (`pipeline/train.py`); 3,000 rounds; labels **1–7,200 s** | — | — | — | 533.3759 | v2 |
| 3 | **Keep-tail** | The original 27 inputs; 800 rounds; training labels widened to **1–200,000 s**, so extreme labels stay in training | — | — | — | 405.6987 | v3 |
| 4 | **FPL-flag** | Keep-tail plus an explicit `has_flight_plan` input | 366.5209† | 333.9163† | 348.7321† | 405.7838 | v4 |
| 5 | **Ceiling-10k** | Keep-tail with training labels capped at **10,000 s**; no validation run | — | — | — | 518.3068 | v5 |
| 6 | **Ceiling-16k** | Keep-tail with training labels capped at **16,000 s**; no validation run | — | — | — | 498.3336 | v6 |
| 7 | **Ceiling-22k** | Keep-tail with training labels capped at **22,000 s**; no validation run | — | — | — | 486.0517 | v7 |
| 8 | **Ceiling-32k** | Keep-tail with training labels capped at **32,000 s**; no validation run | — | — | — | 472.7133 | v8 |
| 9 | **Ceiling-50k** | Keep-tail with training labels capped at **50,000 s**; no validation run | — | — | — | 433.6797 | v9 |
| 10 | **Ceiling-70k** | Keep-tail with training labels capped at **70,000 s**; no validation run | — | — | — | Rejected (daily limit) | v10 |
| 11 | **Ceiling-90k** | Keep-tail with training labels capped at **90,000 s**; no validation run | — | — | — | Rejected (daily limit) | v11 |
| 12 | **B0-800** | Original baseline model; recovery off; no prediction rule | 366.7465 | 334.1785 | 348.9770 | 406.4281 | v12 |
| 13 | **Q0** | B0-800 plus the **six-hour Rome rule** | 300.5904 | 256.1971 | 276.7374 | 354.4303 | v13 |
| 14 | **Q1** | Baseline with **R3 flight-plan recovery**, plus the **six-hour Rome rule** | 296.9037 | 258.5505 | 276.1937 | 354.0004 | v14 |
| 15 | **Q0-4h** | B0-800 plus the same Rome rule at **four hours** | 296.2785 | 255.6200 | 274.3707 | 344.3114 | v15 |
| 16 | **Q1-4h** | Baseline with **R3 flight-plan recovery**, plus the Rome rule at **four hours** | 292.8785 | 257.4787 | 273.7215 | 344.4821 | v16 |
| 17 | **no_month** | Q0-4h without the calendar **`month`** input | 296.4686 | 251.8408 | 272.5082 | 345.8984 | v17 |
| 18 | **no_history** | Q0-4h without the **five departure-taxi history inputs** | 296.5108 | 255.9093 | 274.6319 | 342.5049 | v18 |
| 19 | **clock_features** | Q0-4h plus **18 explicit clock inputs**; still predicts taxi time directly | 293.6614 | 255.6326 | 273.1284 | **338.1113** | v19 |
| 20 | **small_trees** | Q0-4h with **63 leaves** per tree instead of 255 | 296.6377 | 255.6624 | 274.5645 | 345.3113 | v20 |
| 21 | **rule_routing** | Q0-4h, but **four-hour-rule-eligible flights are left out of model training**; the rule still sets their predictions | 286.6291 | 251.4024 | 267.5759 | 342.4009 | v21 |
| 22 | **Rome-mix-base** | B0-800, but each Rome prediction becomes **p × (takeoff − schedule) + (1 − p) × B0-800's prediction**, where p is a Rome-only classifier's probability that off-block was recorded at the scheduled time; no Rome rule | 311.4125 | 283.5840 | 296.2310 | — | Not submitted |
| 23 | **Rome-mix-ordinary** | Same Rome blend, but the (1 − p) part comes from a model **trained without Rome's schedule-recorded rows** | 317.6906 | 294.5641 | 305.0225 | — | Not submitted |
| 24 | **Rome-hybrid** | Rome-mix-ordinary, but the **six-hour Rome rule** overrides the blend on its flights; other airports also use the model trained without those rows | 280.6325 | 250.5677 | 264.3047 | — | Not submitted |
| 25 | **Rome-hybrid-4h** | Rome-hybrid with the **four-hour Rome rule** instead of six hours | 280.8670 | 250.4076 | 264.3304 | — | Not submitted |
| 26 | **Rome-hybrid-4h-clock** | Rome-hybrid-4h plus the **18 clock inputs** in both the model trained without schedule-recorded rows and the classifier | 271.9287 | 246.7919 | 258.2262 | — | Not submitted |

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
- **v17–v21:** CV from `VALIDATION/runs/overnight_20260917/`, whose regenerated
  Q0-4h control scored 296.2717 / 255.6100 / 274.3622 (a small replay
  difference from v15's row). See the
  [overnight summary](VALIDATION/reports/OVERNIGHT_SUMMARY_20260917.md).

**Schedule-recorded rows:** Rome training departures whose taxi-time label equals
takeoff minus scheduled time within 60 seconds, meaning off-block was logged at
the scheduled time. The classifier is trained on these; prediction uses only
fields available in the ranking data.

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
