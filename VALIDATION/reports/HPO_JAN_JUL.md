# Hyperparameter search for v51's parts on Jan/Jul CV

Asked 2026-10-05: tune every part of v51 (C4) whose settings were never searched. Run
2026-10-05 13:10 → 2026-10-06 01:46 on the desktop (RTX 3080) and, for CatBoost, the laptop
(RTX 5060). Runner `VALIDATION/hpo_jan_jul.py`; run `VALIDATION/runs/v3_20261005_hpo_jan_jul/`.

**How a trial is scored.** The trial's part replaces the same part of C4, and the blend is scored
as v51 is built: 0.6 × networks + 0.4 × mean(main LightGBM, per-airport LightGBMs, CatBoost), then
the Rome mix, corrections and rules. Search trials use seed 42; each part's winner was refitted at
seeds 142 and 242 (networks: 15, as C4) and scored at three seeds with the paired day bootstrap.
The trial lists and the rule for one combined trial per part were fixed before the search.

**Optimism.** Winners, and their rounds, were picked on the same two months they are scored on,
which v44–v51 were also chosen on. Rome's search-time gain (−2.32) moved the leaderboard by −0.51
(v53, about 0.22×).

## Result (Jan/Jul CV, three seeds; C4 = v51's recipe, 298.2138)

| Part | Winner (change from v51's settings) | Change vs C4 | 95% interval | Seeds |
|---|---|---:|---|---:|
| Rome classifier | 127 leaves, 60% inputs per tree, ≥50 rows per leaf, learning rate 0.02, 1,700 rounds | **−2.32** | [−4.96, −0.42] | 3/3 |
| Networks | batch 2,048 instead of 4,096 | −0.05 | [−0.21, +0.12] | 3/3 |
| CatBoost | learning rate 0.1, 70% row sampling (Bernoulli), 10,000 rounds | **−0.15** | [−0.23, −0.10] | 3/3 |
| Main LightGBM | `max_cat_threshold` 8, 10,000 rounds at learning rate 0.02 | **−0.12** | [−0.15, −0.09] | 3/3 |
| Per-airport LightGBMs | 63 leaves, ≥50 rows per leaf, `lambda_l2` 10, 5,000 rounds | **−0.18** | [−0.27, −0.10] | 3/3 |
| Three tree models together | | **−0.43** | [−0.60, −0.32] | 3/3 |
| Rome + three tree models | | **−2.77** | [−5.44, −0.86] | 3/3 |
| All five (C5) | | **−2.85** | [−5.51, −0.95] | 3/3 |

On top of the Rome change, the three tree models add **−0.45 [−0.61, −0.33]**. Nearly all of
Rome's gain is July (267.86 → 263.29); the tree models help both months a little. Rome's seeds
share one classifier, so its seed agreement adds no evidence.

## What the search says, part by part

- **Rome classifier:** larger trees with smaller steps sort schedule-recorded flights better.
  It is the only large change. Already uploaded alone as v53 (237.7683, −0.51 vs v51).
- **Networks:** v51's settings were already about right. The best single changes at seed 42
  (batch 2,048 −0.29, dropout 0.2 −0.27) made it worse together (+0.35) and shrink to −0.05 with
  15 networks at three seeds: noise. Higher learning rates (+8 to +9), a 6 h label cap (+3.3)
  and a wider first layer (+1.0) clearly hurt.
- **CatBoost:** hitting the 10,000-round cap was not costing much (16,000 rounds: −0.04). Row
  sampling helped most. Deeper trees (10) and depth 6 did not help. Adding the 22 combo inputs
  to the winner was worse than without them (−0.10 vs −0.18 at seed 42).
- **Main LightGBM:** v44's settings were close to best; single changes stayed within ±0.07.
  Coarser category splits (`max_cat_threshold` 8) and more rounds help a little; still improving
  at 10,000 rounds.
- **Per-airport LightGBMs:** smaller trees with a light brake help a little; more rounds
  (5,000 instead of 2,500) help in every setting.

## Checks

- Trials with v51's own settings reproduce v51's models: Rome at 300 rounds exactly, per-airport at
  2,500 rounds (−0.00001), main LightGBM at 5,000 rounds (−0.004).
- The laptop's inputs (splits, movement data, ADS-B and extra-feature tables, C4's saved
  predictions) were hash-checked identical to the desktop's; CatBoost's search and its seed
  refits ran entirely on the laptop. The per-network trial N11 (96 PLE bins) overflowed the
  3080's memory and ran ~14× slower; its result is valid.

## Files

`rome.json`, `nn.json`, `catboost.json`, `lgb.json`, `airport.json` (every trial, every
checkpoint), `winners.json`, `report.json`, `predictions/<part>/<trial>_s<seed>.parquet`,
`logs/` (laptop logs in `logs/laptop/`). Rome-only upload: `VALIDATION/rome_hpo_submit.py`,
record in `VALIDATION/runs/v3_20261005_rome_hpo_submit/`.

No full-year refit of the tree-model winners has run.
