# Stages 2 and 3: exact inputs and worked examples

Stage 2 overrides selected predictions with a rule. Stage 3 adds 18 clock
features and tests direct prediction (T1) against predicting a correction (T2).
The examples below are illustrative unless explicitly marked as real flights.
All time differences supplied to the models are **seconds**.

## Stage 2: four raw fields, one replacement rule

| Raw field | Use | Illustrative value |
|---|---|---|
| `ADEP_mvt` | Require departure from Rome Fiumicino | `LIRF` |
| `FLIGHT_ID_mvt` | Require the original flight-plan ID to be missing | null |
| `MVT_TIME_UTC_mvt` | Takeoff time | 13:00 |
| `SCHED_TIME_UTC_mvt` | Recorded scheduled departure time | 06:00 |

The gap is seven hours: 25,200 seconds. With the fixed six-hour threshold
(21,600 seconds), replace the model's prediction with **25,200 seconds**.
The trigger does not use the recorded taxi-time target. The raw missing-ID
condition stays the same even when R3 recovers some flight-plan fields.

Q0 applies this to baseline predictions; Q1 applies it to R3 predictions.
Stage 2 adds no model inputs and does not retrain either model. Subsequent
sweeps tested thresholds **1, 2, 3, 4, 5, 6, 7, 8 hours**, or **3,600, 7,200,
10,800, 14,400, 18,000, 21,600, 25,200, 28,800 seconds**. Four hours was the
leading overall CV sensitivity candidate; Q0/Q1 remain six-hour controls.

**Real damaging example:** Rome movement `195015201` had a schedule gap of
**23,039 seconds** (about 6.4 hours), so the six-hour rule predicted that value.
Its recorded taxi time was **1,078 seconds** (about 18 minutes). The rule is
useful overall but does not establish that every eligible label follows it.

## The 27 inputs already present before Stage 3

These stay present in Q0/Q1 and T1/T2. Q1 can have recovered field values;
T1/T2 keep recovery off. The ID and airport rule fields above are separate
from this model matrix; flight IDs themselves are not model features.

| Model feature | Meaning | Example |
|---|---|---|
| `airport` | Departure airport, from `ADEP_mvt` | `LIRF` |
| `dest` | Destination, from `ADES_mvt` | `EGLL` |
| `aircraft_type` | Movement aircraft type | `A320` |
| `runway` | Airport plus runway | `LIRF:25` |
| `stand` | Airport plus stand | `LIRF:101` |
| `flight_rule` | Movement flight rule | `I` |
| `market_segment` | Flight-plan market segment | `Mainline` |
| `flight_type` | Flight-plan flight type | `S` |
| `wake_cat` | Wake turbulence category | `M` |
| `operator` | Anonymized flight-plan operator | an operator category |
| `hour` | UTC takeoff hour | 10 |
| `dow` | Takeoff weekday, Monday=0 | 2 for Wednesday |
| `month` | Takeoff month | 7 for July |
| `mvt_minus_sched_sec` | Takeoff minus schedule | 1,800 s below |
| `mvt_minus_lobt_sec` | Takeoff minus LOBT | 2,400 s below |
| `mvt_minus_iobt_sec` | Takeoff minus IOBT | 2,100 s below |
| `mvt_minus_eobt1_sec` | Takeoff minus EOBT1 | 1,500 s below |
| `mvt_minus_aobt3_sec` | Takeoff minus AOBT3 | 1,200 s below |
| `op_taxi_30d` | Operator's mean taxi-out over the available 30-day history | 950 s |
| `op_taxi_30d_n` | Number of departures supporting that operator mean | 400 |
| `rwy_taxi_30d` | Airport/runway mean taxi-out over 30 days | 1,000 s |
| `stand_taxi_90d` | Airport/stand mean taxi-out over 90 days | 900 s |
| `aphour_taxi_30d` | Airport/takeoff-hour mean taxi-out over 30 days | 980 s |
| `arr_taxiin_7d` | Airport's mean arrival taxi-in over the prior seven days | 500 s |
| `dep_actual_30m_before` | Other departures in the prior 30 minutes, using available off-block clocks | 12 |
| `arr_actual_30m_before` | Landings in the prior 30 minutes | 8 |
| `dep_sched_60m_around` | Scheduled departures within 30 minutes either side | 25 |

Traffic windows are anchored on the row's available off-block reference.
Departure taxi histories use earlier days in training and freeze at the training
cutoff for validation. They cannot use validation departure targets. Their
historical-label population remains 1–7,200 seconds. Available arrival taxi-in
history and traffic context follow the existing fold availability rules.

## Stage 3: one example, all 18 new features

Takeoff is **10:30**. The stored clocks are:

| Raw field | Meaning | Illustrative time | Takeoff minus clock |
|---|---|---:|---:|
| `SCHED_TIME_UTC_mvt` | Recorded scheduled departure time | 10:00 | 1,800 s |
| `LOBT_flt` | Last known off-block time | 09:50 | 2,400 s |
| `IOBT_flt` | Initial off-block time | 09:55 | 2,100 s |
| `EOBT_1_flt` | Estimated off-block time from the flight-plan trajectory | 10:05 | 1,500 s |
| `AOBT_3_flt` | Actual off-block time from the flown trajectory | 10:10 | 1,200 s |

Off-block means leaving the stand/gate. These stored clocks can disagree with
one another and with the airport movement record used to construct the target.
The raw schedule field is documented as scheduled departure time; this example
does not establish more precise semantics for every airport's schedule feed.

Assume the original ID, plan block and AOBT3 are all present. Stage 3 adds:

| New feature | Calculation or meaning | Example value |
|---|---|---:|
| `clock_aobt3_minus_schedule_sec` | 10:10 − 10:00 | 600 s |
| `clock_aobt3_minus_eobt1_sec` | 10:10 − 10:05 | 300 s |
| `clock_eobt1_minus_iobt_sec` | 10:05 − 09:55 | 600 s |
| `clock_iobt_minus_lobt_sec` | 09:55 − 09:50 | 300 s |
| `clock_schedule_available` | Schedule present? | 1 |
| `clock_lobt_available` | LOBT present? | 1 |
| `clock_iobt_available` | IOBT present? | 1 |
| `clock_eobt1_available` | EOBT1 present? | 1 |
| `clock_aobt3_available` | AOBT3 present? | 1 |
| `clock_raw_flight_id_missing` | Original `FLIGHT_ID_mvt` missing? | 0 |
| `clock_raw_plan_block_missing` | Original `ADEP_flt` missing? | 0 |
| `clock_raw_aobt3_missing` | Original `AOBT_3_flt` missing? | 0 |
| `clock_available_count` | Count available clocks | 5 |
| `clock_span_sec` | Largest gap minus smallest gap | 2,400 − 1,200 = 1,200 s |
| `clock_std_sec` | Population standard deviation of the five gaps | 424.26 s |
| `clock_median_sec` | Middle of the five gaps | 1,800 s |
| `clock_baseline_sec` | Takeoff minus preferred available clock, clipped below at zero | 1,200 s |
| `clock_baseline_source` | Code for the chosen clock | 0 |

The four clock differences are signed; a negative value is retained. Summary
features ignore missing clocks. With no clocks, count is zero and span/std/median
are missing. Flags use 1 for yes and 0 for no.

Baseline priority/source codes are **AOBT3=0 → EOBT1=1 → IOBT=2 → LOBT=3 →
schedule=4**. Use the first available finite gap. If every clock is missing,
the baseline is zero and source is **5**. For example, with only the schedule
present here, the baseline is 1,800 seconds and its source is 4.

## T1 versus T2, with numbers

Both have **45 features: the original 27 plus these 18**. Suppose the example's
recorded taxi time is 900 seconds (15 minutes), while its clock baseline is
1,200 seconds (20 minutes).

- **T1:** train directly on the answer **900**.
- **T2:** train on **900 − 1,200 = −300**. If it predicts that correction,
  reconstruct **1,200 + (−300) = 900** seconds.

Negative corrections are allowed. Clip the reconstructed prediction below at
zero, then apply the same fixed six-hour Rome rule. Neither model receives the
validation target or hidden departure block timestamp as an input.

Both use recovery off, **800 rounds**, seeds **42/142/242**, training labels
**1–200,000 seconds**, and every validation label. `fpl_flag` remains false;
the new clock/provenance flags are the explicit feature additions above.

**Real T2 failure:** Amsterdam movement `200741002` used the schedule fallback
with a **151,639-second baseline (42.1 hours)**. Its target was **549 seconds
(9.15 minutes)**. Its mean next-month T2 prediction was **132,665 seconds
(36.9 hours)**, compared with **2,663 seconds** from T1. These are means across
seed predictions for explanation; official CV scoring averages squared errors,
not predictions. The learned correction did not cancel the huge baseline.

Results: [Stage 2](../reports/STAGE2.md), [threshold sweep](../reports/STAGE2_THRESHOLD_RESULTS.md),
[Stage 3](../reports/STAGE3.md). Implementations: [clock features](../../pipeline/clock_features.py),
[rule](../../pipeline/prediction_rules.py), [model features](../../pipeline/common.py),
[history features](../features_time.py).
