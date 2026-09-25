# Stand coordinates on v42: frozen design

Written 2026-09-25, before any fit. Idea 34, step 2. Step 1 (the positions) is in
[the stand report](../reports/STAND_COORDINATES.md).

## Question

Do stand positions help the model, and in particular on stands it has not seen?
Stand names already tell the model about every well-used stand. So the main
cross-validation can show little, and a separate test hides stands on purpose.

## Arms

| Arm | Ordinary regressor inputs | Everything else |
|---|---|---|
| control | v42: v40's inputs + the 12 disruption inputs (`opdi_disruption_run.EXTRA`) | v40 recipe: 800 rounds, recovery off, four-hour Rome rule, 127-row training exclusion, v38 corrections, shared Rome classifier |
| candidate | control + `sc_x_m`, `sc_y_m`, `sc_rwy_end_m` (`stand_features.py`) | identical; the classifier is shared and never sees the new inputs |

The control is rebuilt on this machine (Apple M3). It uses the local original
OPDI cache (`c1b6e337…`), not the rebuilt cache behind gunes's v42 CV (`6df44098…`).
Gains are therefore measured against this matched control. Its level is compared
with gunes's v42 CV (Weighted 263.5929) only as a reproduction check.

## Main experiment

The six v3 windows × seeds 42/142/242 give 54 fits: control, candidate and
classifier per window and seed. The runs cover all 16 original folds, plus the
January 2025 retrospective from the June–November models, as in the v42
experiment. Scored with the adopted Weighted CV (unchanged weights), next-month
R_A, longer-gap R_B_long and original combined. Paired 1-/3-day block
bootstrap, 2,000 draws, seed 0. Per-seed changes are reported.

## Hidden-stand test

For each window, seed 42 only, a stand is hidden when
`sha256("stand-coords-hidden|<window>|<airport>:<stand>")` falls in the lowest
10% of its range. The draw uses every named stand, located or not.

- Training: rows at hidden stands are removed before both ordinary fits
  (12 fits in all). Every other step is unchanged.
- Validation: at hidden-stand rows, `stand` and `stand_taxi_90d` are set to
  missing in both arms. That is exactly how a stand first used in 2026 reaches
  the model: an unseen name with no history.
- Prediction: the main run's seed-42 classifier and the same correction stack.
- Score: RMSE on hidden-stand rows pooled over the 16 folds, overall, for
  located stands and per airport, with paired 1-/3-day block bootstrap
  intervals. Rows at visible stands are scored as a sanity check only.

Nothing in this test enters Weighted CV. It measures the case the idea is for.

## Reading the results

- **Weighted CV** decides as usual. A gain is supported when both paired
  intervals are below zero and all three seeds improve. A change inside the
  noise is inconclusive, not an improvement.
- **The hidden-stand test** is supported when both of its intervals are below zero.
- A supported hidden-stand gain with an inconclusive Weighted CV means the inputs
  help only on unseen stands. Such stands carry about 1,500 ranking departures,
  mostly EDDM 103–108 and EDDF, so any leaderboard effect would be small.
- No full-year fit, upload or `SUBMISSIONS.md` row is part of this design. A
  submission needs the user's separate approval. The ledger gets a row only for
  a candidate with supported Weighted CV evidence.

## Not tested here

Taxi-route length along the taxiway graph; stand-to-runway geometry beyond the
straight line; other seeds for the hidden-stand test.
