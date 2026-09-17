# M1 — Rome schedule-substitution mixture

Completed 2026-09-17 on the macOS M3 (43.2 minutes). Run out of plan order, at
the user's request, before recovery R1. Flight-plan recovery is still off.

## What was tested

At LIRF about 20% of departures record off-block at the scheduled time, so the
label is takeoff minus schedule (delay plus taxi). Because
`TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` exactly for every
2025 departure, training labels identify these rows without loading block
times: LIRF and `|(takeoff − schedule) − label| ≤ 60 s`.

`harness.py --mixture lirf-schedule` ([mixture.py](../mixture.py)) fits per
window and seed, alongside the unchanged baseline regressor:

- a LIRF-only substitution classifier (LightGBM binary, 31 leaves, 300 rounds,
  baseline features plus raw flight-ID and plan-block missingness);
- an *ordinary* regressor: baseline recipe without the substituted rows.

Saved columns: `pred` (baseline), `pred_rule6h` (Q0), `pred_mix_base` =
p·delay + (1−p)·baseline, `pred_mix_ordinary` = p·delay + (1−p)·ordinary,
`pred_ordinary`, `p_substituted`, `gap_sec`. Mixtures apply only at LIRF.

- **Run:** [`v3_20260916T214848990330Z`](../runs/v3_20260916T214848990330Z/result.json)
- **Recipe:** 800 fixed rounds; seeds 42, 142, 242; folds A1–A6 and B1–B10;
  labels 1–200,000 s; recovery off. All candidates are columns of the same fits,
  so every comparison below is paired on identical models and flights.
- **Reproduction:** `pred` scores R_A 366.7403 and R_B_long 334.2015, matching
  B0's saved 800-round checkpoint (366.7673, 334.2028) from the WSL machine.

## Results

Differences are candidate minus reference; 2,000 paired UTC-day bootstrap
replicates; negative favours the candidate.

| Candidate | R_A | R_B_long | Δ R_A vs baseline | Δ R_B_long vs baseline |
|---|---:|---:|---|---|
| Baseline (`pred`) | 366.74 | 334.20 | — | — |
| Q0 six-hour rule | 300.58 | 256.17 | −66.16 [−98.25, −35.92] | −78.03 [−143.57, −10.05] |
| M1a mixture on baseline | 311.41 | 283.58 | −55.33 [−77.96, −33.41] | −50.62 [−87.07, −10.47] |
| M1b mixture on ordinary | 317.69 | 294.56 | −49.05 [−67.79, −30.77] | −39.64 [−68.51, −7.96] |

**Both mixtures improve on the baseline but lose to the simpler rule.** M1b minus
Q0: R_A +17.11 [−2.30, +37.76] (inconclusive), R_B_long +38.39 [+1.63, +76.49]
(regression). No airport opposed any headline change.

### Why: one band

R_A squared error at LIRF, seed-averaged, by `AOBT_3_flt` availability and delay
(takeoff minus schedule):

| Slice | Baseline | Q0 rule | M1b |
|---|---:|---:|---:|
| No plan, ≥ 6 h | 5.23e10 | **4.61e9** | 2.70e10 |
| No plan, 2–6 h | 1.34e10 | 1.34e10 | **7.11e9** |
| Plan, ≥ 6 h | 6.25e9 | 6.25e9 | **3.59e9** |
| All other LIRF slices | 1.86e10 | 1.86e10 | **1.67e10** |

The mixture wins everywhere except long-delayed flights without a plan, where it
gives back about half the rule's gain. Two causes:

- **Calibration is poor exactly there.** The classifier is well calibrated
  overall (Brier 0.110 against 0.160 for the base rate; every probability bin
  within 0.03 of its observed rate), but in that band it predicts 0.555 against
  an observed 0.795. The six A months hold only 44 such flights.
- **The non-substituted rows there are not ordinary.** In 2025 they are mostly
  day-late takeoffs (labels near 87,000 s), which the (1 − p)·ordinary term
  prices as a normal taxi. Predicting the delay is closer for both groups.

The ordinary regressor does what step 2 intended: rows at or below 7,200 s
improve from 273.16 to 260.00 (R_A) and 240.93 to 238.02 (R_B_long). Used at
every airport, `pred_ordinary` also improves non-LIRF RMSE, 234.08 → 230.67
(R_A) and 227.83 → 226.09 (R_B_long).

## Exploratory hybrids — post hoc, not registered

Defined after seeing the band table, from saved columns, with the same bootstrap:

- **H1:** M1b, but the Q0 rule region (LIRF, raw flight ID missing, delay ≥ 6 h)
  predicts the delay.
- **H2:** H1 at LIRF, `pred_ordinary` at every other airport.

| Candidate | R_A | R_B_long | Δ R_A vs Q0 | Δ R_B_long vs Q0 | Rows ≤ 7,200 s (A / B_long) |
|---|---:|---:|---|---|---|
| H1 | 283.23 | 252.03 | −17.35 [−26.04, −10.30] | −4.14 [−6.56, −2.10] | 260.6 / 238.0 |
| H2 | 280.63 | 250.57 | −19.95 [−28.70, −12.86] | −5.61 [−8.07, −3.38] | 257.7 / 236.5 |

H2 minus H1: R_A −2.60 [−3.31, −2.00], R_B_long −1.47 [−2.01, −0.92]. The same
override on M1a scored R_A 292.57 and R_B_long 253.89, worse than H1.

These intervals are optimistic: the combination was chosen on these validation
months. H2 is a strong hypothesis, not a validated result.

## Next

1. Register H2 as a candidate: compute its prediction in `mixture.py`, rerun, and
   add `submit.py` support for the mixture recipe.
2. Submit Q0 and H2 as a controlled leaderboard pair.
3. Replace the ≥ 6 h override with a third, day-late component, rather than tuning
   the threshold on these folds.
