# Weighted CV for the five most recent submissions

The five uploads after the weighted validation was adopted are scored on it here: v34, v37, v38, v39 and v40, each from the fold predictions behind the uploaded file. The adopted weights, folds and formula are unchanged.

| Upload | Row | Recipe | CV combined | Weighted CV | Official LB | Weighted gap |
|---|---:|---|---:|---:|---:|---:|
| v34 | 34 | No-ID-cap | 239.9398 | 273.2162 | 280.4658 | -7.25 |
| v37 | 40 | Dayshift-hedge | 235.2323 | 265.9048 | 280.1743 | -14.27 |
| v38 | 37 | Artifact-combined | 236.3151 | 267.5814 | 279.3805 | -11.80 |
| v39 | 42 | OPDI-combined | 233.7470 | 264.4218 | 277.3629 | -12.94 |
| v40 | 43 | Rule-routed-training | 233.2338 | 264.3397 | 275.4146 | -11.07 |

All values are RMSE in seconds; lower is better. The weighted gap is Weighted CV minus the official score.

| Check | Current CV | Weighted CV |
|---|---:|---:|
| Average absolute gap to the official score | 42.87 s | 11.47 s |
| Largest absolute gap | 44.94 s | 14.27 s |

| Change | Actual LB change | Current CV change | Weighted CV change | Direction |
|---|---:|---:|---:|---|
| v34 -> v37 | -0.2915 | -4.7075 | -7.3114 | agrees |
| v37 -> v38 | -0.7938 | +1.0828 | +1.6767 | **disagrees** |
| v38 -> v39 | -2.0176 | -2.5680 | -3.1596 | agrees |
| v39 -> v40 | -1.9483 | -0.5132 | -0.0821 | agrees |

Direction agreement on consecutive uploads: 3/4 current CV, 3/4 weighted. Average error in the size of a step is 2.07 s current CV and 3.12 s weighted, so the weighting fixes the level of the score without predicting how far a change will move the leaderboard.

**Checks:** the weights are the adopted ones. The local cache was rebuilt from the frozen specification and its four discriminator model files are byte-identical to the ones recorded with the method, with the same cross-fitting split, held-out calibration and effective sample size, so the ratios are the same numbers; only the parquet container bytes differ, which a different pyarrow build explains and the identical calibration and effective sample size rule out as a difference in values. Every original v3 score reproduces from the same predictions before weighting. Flight identities and labels in both groups match the adopted comparison. Each submission manifest records an accepted upload scored on all 344,841 rows.

**Limits:** v38 is carried over from [the adopted comparison](weighted_cv_leaderboard.json) because fold predictions for v3_20260919_artifact_experiments are not on this machine; it was not recomputed here. These five uploads are consecutive refinements of one model line, not independent evidence, and their official scores were already known, so this is a retrospective reading, not a validation of the weighting. Weighted CV was tuned on neither these rows nor their scores, but v38 did inform the weighting work.

Reproduce: `.venv/bin/python VALIDATION/tools/weighted_cv_recent_submissions.py`. [Full results](weighted_cv_recent_submissions.json); [method](../docs/WEIGHTED_CV.md).
