# Training-label filter: cutting contaminated labels out of the fit

Run `v3_20260920T122629684296Z` (label `maxtaxi-7200`), 21.804 minutes, 16 folds,
seed 42, on the 8-core macOS host. Control is the seed-42 arm of the completed OPDI
run `v3_20260919T211914721668Z`. Only `max_taxi_sec` differs: **7,200 s versus
200,000 s**. No uploads. This is the training-routing experiment listed as proposed
and unstarted in `AGENTS.md`.

## Design

`modeling.training_matrix` filters training rows by `config["max_taxi_sec"]`;
`modeling.validation_matrix` applies no label filter, so **every evaluation
departure is scored in both arms**. `opdi_run.py`'s `check_control` verified per
fold that the evaluation rows are identical to the control's. A `--max-taxi-sec`
flag was added to `VALIDATION/opdi_run.py`, defaulting to the existing value.

Scored column is `pred_opdi`. The `pred` column reuses the control's boosters
(`reuse_control`) and is identical by construction, so it cannot be compared;
the 12 models this run fitted are the `_opdi` ordinary/classifier pairs.

## Result: supported regression

Seed 42, column `pred_opdi`, 2,784,530 pooled rows.

| metric | control | experiment | change | 95% CI | decision |
|---|---:|---:|---:|---|---|
| R_A | 253.0960 | 263.8747 | **+10.7787** | [+3.59, +22.05] | regression |
| R_B_long | 227.4171 | 230.0519 | **+2.6348** | [+0.30, +5.03] | regression |
| R_B_all | 239.6133 | 242.2353 | +2.6220 | [+0.75, +4.49] | regression |
| R_combined | 239.1299 | 245.6062 | **+6.4763** | [+2.46, +12.30] | regression |

**Do not promote.** Pooled RMSE 241.6827 -> 248.2379 (+6.5551).

## But the premise was right

The hypothesis was that contaminated labels distort the fit for ordinary flights.
That is correct, and the effect is large:

| label band | rows | control RMSE | experiment RMSE | Δ SE, share of total |
|---|---:|---:|---:|---:|
| < 30 m | 2,680,782 | 189.0 | **182.3** | **-4.07%** |
| 30-60 m | 98,659 | 577.8 | **571.6** | **-0.43%** |
| 1-2 h | 4,041 | 1969.1 | 2061.3 | +0.92% |
| > 2 h (removed from training) | 705 | 5075.2 | 6835.6 | **+9.09%** |

On the **2,783,825 ordinary rows (99.97% of the data)** the cut is worth
**-4.65 s** (227.8204 -> 223.1690), and the residual bias improves from -3.10 to
-2.26 s. The 705 removed rows then cost more than that gain back.

The measured motivation held up too: those 705 rows are 0.025% of the data but
carry **39.04%** of the target's squared deviation, and each row above 14 h has the
gradient weight of ~93 ordinary flights.

## Why it lost, and the obvious refinement

Split the 705 removed rows by whether a rule overwrites their prediction anyway:

| | rows | control RMSE | experiment RMSE | Δ SE |
|---|---:|---:|---:|---:|
| rule-covered (Rome 4 h) | 131 | 8981.5 | 8981.5 | **0.000** |
| not rule-covered | 574 | 3636.9 | 6243.4 | +1.478e10 |

**The entire penalty comes from 574 rows the rules do not own.** For the 131 rows
the Rome rule overwrites, removing them from training costs *exactly zero* — the
prediction is replaced by arithmetic either way.

So the filter as run is too blunt. The correct form of the experiment is the one
`AGENTS.md` actually describes: exclude **only rule-handled rows** from training,
not every large label. That is free on the 131 rows by construction, and those
rows include the most extreme labels (all 29 rows above 50,000 s are LIRF no-ID
with the rule triggering), which do the most gradient damage per row.

Expected shape: a fraction of the -4.65 s ordinary-row gain at zero tail cost.
Untested; it needs its own run and its own control.

## Status

Seed 42 only — a quick read, not a protocol score. Nothing promoted, nothing
uploaded, no fold removed. The `--max-taxi-sec` flag defaults to the previous
behaviour, so no existing recipe changes.

---

# Part 2: excluding only rule-covered rows (the refinement)

Run `v3_20260920T152734631417Z` (label `rulerows-excluded`), **three seeds
(42/142/242), 16 folds**, 25.521 minutes on resume after the first attempt was
killed by a session teardown at 12/36 models. Control is the same OPDI run
`v3_20260919T211914721668Z` at the same three seeds. `max_taxi_sec` stays at
200,000; the only change is that training drops the departures the prediction
rule overwrites.

## Implementation

Part 1's filter was implemented inside `modeling.training_matrix`. That was wrong
twice over: it changes the hash of frozen shared sources, which the row-30
integrity guard in `artifact_control_run.historical_recipe()` correctly rejects,
and `overnight_run.prepare_training` recomputes the kept-row set independently
and asserts it matches, so it would have raised at runtime. Both files were
reverted. The routing now lives only in `VALIDATION/opdi_run.py`, which drops the
rows from `X`, `y` and `raw_kept` together after the correction parameters are
fitted. Context, categories, history and those parameters still see every row.

The mask comes from the frozen rule itself (`registered_rules.apply_rule` on a
zero vector, reading only its audit), so no label and no prediction enters the
decision. Over all of 2025 it selects **127 departures** — every one LIRF with no
flight ID and a schedule gap of 4 h or more, median label 20,887 s.

## Result: supported improvement

Column `pred34_opdi`, the recipe row 41 reports. The control reproduces row 41's
ledger entry exactly (250.2107 / 226.7397 / 237.4207), so these are ledger-comparable.

| metric | control (row 41) | rule rows excluded | change | 95% CI | decision |
|---|---:|---:|---:|---|---|
| R_A | 250.2107 | **249.2832** | **-0.9275** | [-1.3857, -0.5424] | improvement |
| R_B_long | 226.7397 | 226.7101 | -0.0296 | [-0.5472, +0.4806] | inconclusive |
| R_combined | 237.4207 | **236.9724** | **-0.4482** | [-0.8305, -0.0796] | improvement |

All three seeds agree on R_A (-1.311 / -0.686 / -0.785) and R_combined
(-0.594 / -0.224 / -0.527). Longer-gap is flat.

## The mechanism is exactly as designed

Pooled over three seeds (8,353,590 rows), on the raw `pred_opdi` column:

| population | rows | control | experiment | change |
|---|---:|---:|---:|---:|
| rule-triggered (excluded from the fit) | 441 | 9969.2 | 9969.2 | **identical** |
| label < 30 m | 8,042,298 | 185.01 | **183.55** | **-1.459** |
| label 30-60 m | 295,977 | 577.04 | **576.10** | -0.932 |
| label 1-2 h | 12,123 | 1968.35 | 1967.74 | -0.612 |
| label > 2 h, not rule-covered | 1,722 | 3680.18 | 3836.33 | +156.150 |

The excluded rows are **byte-identical** between arms, confirming the premise:
the rule overwrites them, so their labels were only ever distorting the fit. The
gain is real but partly given back by 1,722 large-label rows no rule owns — the
same 574-row population (x3 seeds) that sank Part 1.

## Reading it honestly

-0.4482 s combined is small. It is supported, unanimous across seeds, and free by
construction on the rows it removes, but it is the same order as row 35
(twin records, -0.35) rather than a step change. It does **not** close any part of
the ~37 s gap to the leader.

It is measured on **row 41's stack, not row 42's**. Row 42 (OPDI-combined, the
current best at CV 233.7470) adds row 37's three corrections on top. Training
routing is orthogonal to those corrections in principle, but that combination is
untested; the gain may not survive or may not add.

No upload, no full-year refit, no fold removed.

---

# Part 3: the same routing on row 42's stack (the current best)

Run `v3_20260920T170600876190Z` (`opdi-combined-rulerouted`), a **derivation, no
fits**, 16 folds x 3 seeds. `VALIDATION/opdi_combined_run.py` gained a
`--source-run` flag; it applies row 37's three corrections to the rule-routed
models of `v3_20260920T152734631417Z` exactly as row 42 applies them to row 41.

Validity checks that passed before deriving: the source run's per-window
`late_mean` correction parameters are **identical** to the control's in all six
windows (confirming the row filter ran after `fit_parameters`, so only the model
changed), and the control arm reproduces row 42's ledger entry exactly.

| metric | row 42 (v39) | rule-routed | change | 95% CI | decision |
|---|---:|---:|---:|---|---|
| R_A | 247.0477 | **245.9817** | **-1.0660** | [-1.5774, -0.6090] | improvement |
| R_B_long | 222.6081 | 222.5802 | -0.0279 | [-0.5458, +0.4887] | inconclusive |
| R_combined | 233.7470 | **233.2338** | **-0.5132** | [-0.9152, -0.1167] | improvement |

All three seeds agree on R_A (-1.327 / -0.697 / -1.174) and R_combined
(-0.597 / -0.230 / -0.713). The gain survives the correction stack and is
marginally larger than on row 41 (-0.5132 vs -0.4482 combined).

**233.2338 is the best CV combined recorded.** Under the transfer relation in
`prc-cv-lb-transfer` (LB = CV_combined + 42.8 +/- 1.5) that projects to roughly
**276.0** against v39's actual 277.3629 — a predicted gain of about half a second,
well inside the +/-1.5 s spread of that relation. Treat the projection as a screen,
not a forecast.

No upload and no full-year refit were performed; neither is authorized here.

---

# Part 4: submitted as v40 — official score 275.4146

Prepared with `VALIDATION/opdi_submit.py prepare --exclude-rule-rows --version 40
--run v3_20260920T152734631417Z --combined-run v3_20260920T170600876190Z`
(the flags are additive; `EXCLUDE_RULE_ROWS` defaults to False, so v39's recipe
still reproduces byte-for-byte). The full-year refit excluded **exactly 127**
rule-covered departures, matching the CV run, and fitted on 2,084,532 rows in
1.46 minutes. Late mean 1079.5 s, unchanged.

File checks before upload: 344,841 rows, `MVT_ID_mvt` order identical to
`data/submitting.parquet`, all values finite and non-negative, 43 Rome-rule rows.
Against v39: 338,789 rows changed, mean absolute change 28.955 s, median 17.0 s,
**mean signed change -0.014 s** — an unbiased perturbation, as dropping 127
training rows should be.

| | CV combined | official LB | LB - CV |
|---|---:|---:|---:|
| v39 (row 42) | 233.7470 | 277.3629 | +43.616 |
| **v40 (row 43)** | **233.2338** | **275.4146** | **+42.181** |
| change | **-0.5132** | **-1.9483** | |

Live snapshot 2026-09-20T18:16:32Z: **rank 31 of 167** scoring teams (444
registered), best and latest both v40, gap to leader 35.838 s, gap to podium
33.050 s. One of five daily slots used (v39 landed 2026-09-19 23:05 UTC).

**The leaderboard moved 3.8x more than CV predicted** (-1.9483 against -0.5132).
Do not read that as the routing being worth four times its CV estimate. A single
upload cannot resolve a half-second effect, the transfer relation carries a
+/-1.5 s spread, and v40's LB-minus-CV offset of +42.181 sits inside the
established band (42.8 +/- 1.5) — so the honest reading is that the routing helped,
the direction agreed, and the magnitude is not separable from transfer noise on one
observation. It is a real gain of uncertain size, not a confirmed 1.95 s.
