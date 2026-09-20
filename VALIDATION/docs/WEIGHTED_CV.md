# Weighted CV — current validation headline

Adopted by the user on 20 September 2026. Use **Weighted CV** for new model
comparisons, while retaining the original v3 scores as diagnostics. Existing
run records and frozen study designs keep their original scores.

The method is the **unclipped, raw-feature weighting** tested on eleven scored
submissions. It reduced the average absolute CV–LB gap from 68.05 to 23.42 s;
direction accuracy stayed at 7/10. Predicting the size of changes got worse.
See the [comparison](../reports/WEIGHTED_CV_LEADERBOARD_EXTENDED.md).

Each group gets weights learned from its own flights: July–December for `R_A`,
November–December for `R_B_long`, both against the ranking inputs. Two LightGBM
fits per group hold out half of each population. Every source flight receives
a weight from a model that did not train on it:

`weight = p / (1 - p) × source_training_count / ranking_training_count`

Keep the existing folds and average squared errors over seeds and repeated
training origins for the same flight. Within each group, compute
`weighted_MSE = sum(weight × squared_error) / sum(weight)`. The headline is:

`Weighted CV = sqrt((152719 / 344841) × weighted_MSE_A + (192122 / 344841) × weighted_MSE_B_long)`

Reuse the same weights for every candidate. Their hashes are in
[`weighted_cv_leaderboard_extended.json`](../reports/weighted_cv_leaderboard_extended.json);
the local cache is `tmp/weighted_cv_leaderboard_20260920/`. The exact feature list,
seed and settings are in `tools/analyze_adversarial_shift.py`. Do not substitute
the older 269.40 estimate, clipped weights or the rich-feature discriminator.

The standard harness still emits the original v3 diagnostics; compute Weighted
CV from its saved predictions before presenting a new validation comparison.
The saved-score examples are `tools/compare_weighted_cv_leaderboard.py` and
`tools/extend_weighted_cv_leaderboard.py`. They reproduce the eleven verified
entries in [SUBMISSIONS.md](../../SUBMISSIONS.md); other entries remain `-`.

This changes validation scoring only. It does not authorize refits or uploads.
