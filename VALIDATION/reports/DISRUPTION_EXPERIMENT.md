# Snow and arrival-congestion experiment

**Small CV gain; the winter problem remains.** One fixed package of 12 inputs added to v38. It passes the frozen CV criterion, but is not selected for promotion.

| Score (seconds; lower is better) | v38 | New features | Change |
|---|---:|---:|---:|
| Weighted CV | 267.5814 | 266.8465 | -0.7349 |
| Original combined CV | 236.3151 | 235.6700 | -0.6450 |
| Weighted next-month | 272.0013 | 271.4967 | -0.5046 |
| Weighted longer-gap | 264.0153 | 263.0914 | -0.9239 |
| January retrospective check | 348.1483 | 348.6760 | +0.5278 |

Weighted CV change, 95% paired intervals: 1-day blocks [-1.22, -0.33] s; 3-day blocks [-1.17, -0.33] s.
Seed changes: 42: -0.65 s, 142: -0.73 s, 242: -0.83 s.

January worsens overall (+0.53 s; 3-day interval +0.21 to +0.94 s). At EHAM it worsens by 2.03 s, including +7.23 s on 293 missing-ID departures. The 27 EHAM prolonged-snow departures improve by 3.99 s, but that tiny slice cannot establish a winter fix. This tests both feature groups together; it does not identify which group caused the changes.

The model uses past snow duration and taxi times of arrivals completed before takeoff. The classifier, rules, corrections, weights and all 16 folds stayed fixed. All 48 saved v38 prediction files reproduced within 1e-8 s; 18 ordinary fits completed.

Nine targeted tests and an [independent audit](disruption_experiment/independent_audit.json) passed: 4,080 feature-value checks across 340 sampled fold/flight rows. No implementation fault was found in these checks. The ten largest improvements account for only 1.35% of the gross weighted-MSE decrease.

January uses June–November models and excludes January departure labels from training. It is a retrospective diagnostic, outside Weighted CV. Its snow is much milder than January 2026. These reused folds cannot establish leaderboard transfer. This compares against v38, not v40.

[Frozen design](../docs/DISRUPTION_EXPERIMENT.md) · [Scores and intervals](disruption_experiment/results.json) · [Airport/month/snow slices](disruption_experiment/slices.csv) · [Largest individual effects](disruption_experiment/influential_rows.csv)

No upload, full-year fit, model promotion or submission-ledger entry.
