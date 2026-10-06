# Other teams' ideas on v48: Jan/Jul CV

Six ideas from other teams' public write-ups, plus three of our own, tested on v48's recipe
(2026-10-03). Nothing was submitted.

**Control:** v48's recipe at Jan/Jul CV **299.9790**. This is v46's LightGBM blended 0.4 with
the five `ple` networks per seed at 0.6, followed by the Rome classifier, corrections and rules.

**Every arm** is scored in that same blend at weight 0.6, at seeds 42/142/242. Squared errors
are averaged over seeds within each month, and 95% intervals resample whole days.

**Pass rule** (fixed before any arm was scored): the change in the blend has an interval below
zero, and all three seeds improve.

## Single changes

| Idea | Change to the gradient-boosting (GBM) part | Blend change | 95% interval | Seeds | GBM alone |
|---|---|---:|---|---:|---:|
| 1 | + arrival taxi-in at the stand / on the runway (4 inputs) | −0.17 | [−0.24, −0.12] | 3/3 | −0.34 |
| 2 | + other departures' AOBT_3 clocks around takeoff (11 inputs) | −0.09 | [−0.18, −0.03] | 3/3 | −0.19 |
| 5 | + runway sets in use, takeoff bursts, runway streak (7 inputs) | −0.13 | [−0.24, −0.06] | 3/3 | −0.05 |
| 4 | + OpenStreetMap taxi-route length and turning (3 inputs) | −0.07 | [−0.16, −0.01] | 3/3 | −0.15 |
| 1+2+5 | all three groups in one model (22 inputs) | −0.31 | [−0.50, −0.20] | 3/3 | −0.42 |
| 3b | 0.5 × per-airport LightGBMs + 0.5 × v46 | **−0.42** | [−0.72, −0.20] | 3/3 | −1.59 |
| 3a | 0.5 × CatBoost + 0.5 × v46 | **−0.65** | [−0.87, −0.47] | 3/3 | **−2.60** |
| 6 | second-stage model trained on the other month's errors | **+3.13** | [+2.60, +3.93] | 0/3 | |
| ours | blend weight per airport, chosen on the other month | +0.11 | [+0.03, +0.21] | | |
| ours | 15 pooled `ple` networks instead of 5 per seed | **−0.43** | [−0.60, −0.36] | | |
| ours | 5 `ple` networks + TabM, half each | −0.10 | [−0.31, +0.15] | | |

Seven pass. CatBoost alone (301.16) beats v46's LightGBM alone (303.75), and it improves every
airport in both months. The second-stage model and the per-airport weights do not carry over
from one season to the other. The other teams' write-ups reported 2.01 s (idea 1), 1.42 s
(idea 2) and 2.87 s (idea 3a) on the leaderboard. Ideas 1 and 2 give a small fraction of that
here, because we already had related inputs. Each new input group holds 1–5% of LightGBM's
split gain.

## Combined candidates

Fixed before any candidate was scored. The network part is 0.6. The GBM part is an
equal-weight mean of its members:

| Candidate | Network part | GBM members | Jan/Jul CV | Change vs v48 | 95% interval | Seeds |
|---|---|---|---:|---:|---|---:|
| v48 | 5 `ple` per seed | v46 | 299.9790 | | | |
| C1 | 15 `ple` pooled | v46 | 299.5479 | −0.43 | [−0.60, −0.36] | 3/3 |
| C2 | 15 `ple` pooled | 1+2+5 global, 1+2+5 per airport | 298.8027 | −1.18 | [−1.73, −0.87] | 3/3 |
| C3 | 15 `ple` pooled | C2's + CatBoost | 298.5345 | −1.44 | [−1.98, −1.16] | 3/3 |
| **C4** | **15 `ple` refitted with the 1+2+5 inputs** | **C3's** | **298.2138** | **−1.77** | **[−2.40, −1.42]** | **3/3** |

- **The parts stack up.** Network alone improves 301.87 → 300.68 with 15 networks pooled, and
  → 299.89 with the new inputs. GBM part alone improves 303.75 → 301.67 with the features and
  per-airport models, and → 300.10 with CatBoost.
- **Weight 0.6 still fits.** Choosing the network share on the other month gives 0.6 / 0.45 for
  C4, at 298.26.
- **This is not independent evidence.** The candidates combine the arms that passed on these
  same two months, which v44–v48 were also chosen on. Treat the gains as upper estimates.

## Leaderboard

The user chose C1 and C4. Both were fitted on all of 2025 and uploaded on 2026-10-03 by
`VALIDATION/ideas_submit.py`, one fitting process at a time.

| Upload | Candidate | Jan/Jul CV change vs v48 | Official LB | Change vs v48 (239.7731) |
|---|---|---:|---:|---:|
| v50 | C1 | −0.43 | 239.0108 | −0.76 |
| **v51** | **C4** | **−1.77** | **238.2770** | **−1.50** |

C1 transferred at about 1.8× its CV gain. C4's further step from C1 transferred at about 0.55×.

## What a submission would need (before upload)

These are full-year fits that have not been run.

| Candidate | Fits | Estimated time |
|---|---|---|
| C1 | 15 `ple` networks | ~15 min GPU |
| C2 | C1 + 1+2+5 LightGBM (global and 10 per-airport) | + ~30 min CPU |
| C4 | C2 + CatBoost (December-chosen iterations) + 15 networks on the new inputs | + ~1–2 h GPU |

The ranking inputs for 1+2+5 already exist: `extra_features.parquet` covers every ranking
departure.

## Run

| Arm | Fit minutes (all seeds and months) |
|---|---:|
| arrtaxi, peers, rwycfg, route, combo | 75, 77, 105, 92, 105 |
| airport, combo_airport | 57, 63 |
| CatBoost | 302 (seeds 42/142 on the desktop RTX 3080, 242 on the laptop RTX 5060) |
| networks with the 1+2+5 inputs | ~35 |

LightGBM settings are v44's (learning rate 0.02, 5,000 rounds; 2,500 per airport).

CatBoost: GPU, depth 8, learning rate 0.05, iteration count chosen on the last training month
(it reached the 10,000 cap nearly every time). For seeds 142/242 only the GPU memory share
changed (memory only, same model).

**Feature checks:** a brute-force audit of 300 random departures (2,700 values) matched the
cached inputs exactly (`VALIDATION/tools/audit_extra_features.py`).

**Code**
- Inputs: `VALIDATION/extra_features.py`, `VALIDATION/taxi_route.py`.
- Runners: `VALIDATION/ideas_jan_jul.py`, `VALIDATION/catboost_jan_jul.py`,
  `VALIDATION/nn_features_jan_jul.py`.
- Scorers: `VALIDATION/tools/score_ideas_jan_jul.py`,
  `VALIDATION/tools/score_candidates_jan_jul.py`, `VALIDATION/tools/stack_jan_jul.py`,
  `VALIDATION/tools/blend_by_airport.py`, `VALIDATION/tools/nn_ensemble_jan_jul.py`.

**Results** are in `VALIDATION/runs/v3_20261003_ideas_jan_jul/` (scores, fit records,
predictions, networks).

**New external data:** OpenStreetMap taxiway geometry (ODbL), cached in
`data/external/osm_taxi/`. Declare it if idea 4 is ever submitted.
