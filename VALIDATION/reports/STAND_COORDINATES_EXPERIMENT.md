# Stand coordinates on v42 (idea 34, step 2)

**Three map inputs improve every validation, with 1- and 3-day intervals below
zero and all three seeds agreeing. Weighted CV improves 263.6158 → 263.1476
(−0.47 s).** On stands the model has never seen, they cut the error by 2.0 s.
This is ledger row 46, **uploaded as v43: 274.7660 on the leaderboard, 0.14 s better
than v42**. That is about 0.3× the CV gain, below the −0.3 s expected.

The inputs are the stand's east/north offset from the airport's runway-end centre
(`sc_x_m`, `sc_y_m`), and its straight-line distance to the end of the runway the
flight takes off from (`sc_rwy_end_m`). They come from the X-Plane Gateway and
OpenStreetMap stand maps ([step 1](STAND_COORDINATES.md)). Only the ordinary
regressor receives them, as in v42.

| Validation (lower is better) | v42 control | + stand inputs | Change [95% interval] |
|---|---:|---:|---:|
| **Weighted CV** | 263.6158 | 263.1476 | **−0.468 [−0.649, −0.268]** |
| Weighted next-month | 266.2992 | 266.0173 | −0.282 [−0.487, −0.056] |
| Weighted longer-gap | 261.4630 | 260.8440 | −0.619 [−0.881, −0.356] |
| Original combined | 232.6754 | 232.1696 | −0.506 [−0.607, −0.402] |
| Original next-month | 245.5594 | 245.1722 | −0.387 [−0.478, −0.290] |
| Original longer-gap | 221.9008 | 221.2893 | −0.612 [−0.780, −0.439] |
| January 2025 retrospective | 344.8972 | 345.0879 | +0.191 [−0.334, +0.532] |

These are 1-day block intervals; 3-day blocks give the same decisions. The
Weighted CV change by seed is −0.34 / −0.65 / −0.41 s. January uses the
June–November models and sits outside Weighted CV; its change is inconclusive.

**The control reproduces v42.** It was rebuilt on this Mac with v42's
byte-identical experiment code. The run checks this against gunes's recorded
source hashes and stops if any differ. Its Weighted CV is 263.6158, against row
45's 263.5929 (+0.023 s). The small difference comes from the original versus
rebuilt OPDI cache and a different CPU. Gains are measured against this matched control.

## Stands the model has never seen

For each training window, 10% of stand names were hidden: their departures were
removed from training, and their name and 90-day stand history blanked at
prediction. That is how a stand first used in 2026 reaches the model, such as
Munich 103–108. Seed 42 only; 1,147 different stands and 287,337 flight
evaluations across the 16 folds.

| Hidden-stand flights | v42 control | + stand inputs | Change [95% interval] |
|---|---:|---:|---:|
| All | 262.08 | 260.07 | **−2.01 [−2.28, −1.74]** |
| Stand has a map position | 261.89 | 259.89 | −2.00 [−2.28, −1.72] |
| Visible stands (sanity check) | 234.33 | 234.07 | −0.26 [−0.40, −0.11] |

Every airport improves on hidden stands: LEMD −4.69, LFPG −4.34, LTFM −2.90,
EHAM −2.28, EGLL −2.27, LEBL −2.18, LSZH −0.80, EDDM −0.57, EDDF −0.20,
LIRF −0.20 s. The 1,306 hidden flights without a map position also change
(−4.4 s): the two models differ everywhere, so not all of the gain is the
positions themselves.

## Where it comes from, and caveats

- **The main gain is at large airports with runways far apart:** LFPG −1.47,
  LEMD −0.70, LTFM −0.65, EHAM −0.58, EGLL −0.40 s. The distance to the take-off
  runway gives the model the stand-by-runway effect directly. Without it, the
  model has to assemble that effect from two long lists of names.
- **EDDF, EDDM and LSZH are +0.19 to +0.21 s in the main CV** (not tested per
  airport). EDDM and EDDF also gain least on hidden stands. These two airports hold
  most of the ranking's new stands, so the extra benefit there is modest.
- **Ageing does not repeat v41's warning.** In 7 of 10 comparisons, the gain is
  *larger* when an older model predicts the same month; three of these exclude zero.
  One comparison fades clearly: B10_vs_A6 (September model on December), +0.77
  [+0.33, +1.29]. v41 had 8 of 10 positive before its leaderboard reversal.
- **The hidden-stand test uses one seed and one draw of stands per window.**
- **Expected leaderboard effect: about −0.3 s.** This assumes the gain transfers like
  v42, whose −0.75 s Weighted CV became −0.51 s on the leaderboard. v41 shows a
  small CV gain can also reverse. Rows at new ranking stands (EDDM 103–108,
  EDDF J5/J10; about 1,400 flights) add little on top.

## Files and reproduction

```sh
.venv/bin/python VALIDATION/stand_coords_run.py --mode main      # 54 fits, 75 min on an M3
.venv/bin/python VALIDATION/stand_coords_run.py --mode hidden    # 12 fits, 28 min
.venv/bin/python VALIDATION/tools/score_stand_coords.py --mode main
.venv/bin/python VALIDATION/tools/score_stand_coords.py --mode hidden
.venv/bin/python -m unittest VALIDATION/tests/test_stand_features.py VALIDATION/tests/test_stand_geometry.py
```

- Inputs: `VALIDATION/stand_features.py`. Design, frozen before the fits:
  [STAND_COORDINATES_EXPERIMENT.md](../docs/STAND_COORDINATES_EXPERIMENT.md). 28 tests pass.
- Scores, intervals, ageing and per-airport figures:
  `stand_coords_experiment/main.json` and `hidden.json` in this folder.
- Models and predictions (gitignored): `VALIDATION/runs/v3_20260925_stand_coords/`.
  Logs: `tmp/stand_coords_20260925/`. Do not repeat the fits.
- Weighted CV uses this Mac's copy of the adopted weights. Its four discriminators
  are byte-identical to the record, and its effective sample size matches to
  1e-16; only the parquet bytes differ.

## v43

Uploaded 2026-09-25 12:14 UTC after the user's approval, by
`VALIDATION/stand_coords_submit.py` (`prepare`, `upload`, `result`). It is a
seed-42 full-year refit of the two boosters with these inputs, v42's recipe
otherwise. The 43 Rome rule rows are unchanged; the other rows move by 27.7 s on
average. Official **274.7660** on all 344,841 rows, −0.1416 against v42.
The step also contains the OPDI cache and platform difference (original cache on
the Mac versus the rebuilt one behind v42), which moved CV by only 0.02 s.
Manifest: `VALIDATION/runs/v3_20260925_stand_coords/submission_v43/submission_v43.json`.
