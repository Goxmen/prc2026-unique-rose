# Reports

- [Weighted CV: eleven scored submissions](WEIGHTED_CV_LEADERBOARD_EXTENDED.md): current validation headline, adopted on 20 September; smaller CV–LB gaps with unchanged direction accuracy. [Method](../docs/WEIGHTED_CV.md).
- [Initial six-submission comparison](WEIGHTED_CV_LEADERBOARD.md) and [adversarial follow-up review](ADVERSARIAL_FOLLOWUP_REVIEW.md): earlier evidence and remaining limits. Their pre-adoption recommendations are historical.
- [Residual atlas for officially scored recipes](SCORED_RESIDUALS.md): matched-flight error patterns, airport/clock/queue slices, coverage gaps, interactive exploration and an eight-page PDF; saved predictions only.
- [September 19 recording-artifact experiments](RECORDING_ARTIFACT_EXPERIMENTS.md): error patterns, nine local corrections, matched-control scores and test-set hypotheses; no submissions.
- [September 17 overnight study: five successful submissions and held-out CV verdict](OVERNIGHT_SUMMARY_20260917.md).
- [Held-out confirmation details](../runs/overnight_20260917/confirmation_report.md): selected R_A__mse change MAE 3.5178 s versus current 2.8367 s; retain the default and all folds.
- [September 17 live status snapshot](overnight_20260917/status_after.json): 04:23:02 UTC, best v19 338.1113 s, latest v21 342.4009 s, rank 94/152 scoring teams. All five authorized slots were used; no extra uploads are authorized.
- [Month-balanced aggregation proposal, unchanged default](AGGREGATION_COMPARISON.md).
- [Validation optimism: population weights and extreme-error influence](VALIDATION_OPTIMISM.md).
- [Original-fold alignment and removal checks; no folds dropped](FOLD_ALIGNMENT.md).
- [January/July seasonal validation comparison](../runs/seasonal_jan_jul/comparison.md).
- [September 16 five-candidate leaderboard results](LEADERBOARD_RESULTS.md).
- [Stage 0: baseline and round-count verdict](STAGE0.md).
- [Stage 1: recovery experiments](STAGE1.md).
- [Stage 2: fixed Rome six-hour rule, gains and damaging overrides](STAGE2.md).
- [Rome rule: consolidated 1–8 hour scores and verdict](STAGE2_THRESHOLD_RESULTS.md).
- [Stage 2 follow-up: 4–8 hour rule sweep on baseline and R3](STAGE2_THRESHOLD_SWEEP.md).
- [Stage 2 extension: 1–3 hours, compared with four hours](STAGE2_THRESHOLD_SWEEP_LOW.md).
- [Stage 3: clock features, residual results and schedule-fallback failure](STAGE3.md).
- [Stage 4: probability mixture, false high predictions and Q0 retained](STAGE4.md).
- [Stage 1 laptop CPU and memory settings](STAGE1_CPU.md).
- [Notebook-derived data: section mapping and reuse limits](STAGE1_DERIVED_DATA.md).
- `benchmarks/`: measurements supporting the CPU execution settings.
- `sweeps/`: threshold-to-submission-ID mappings, recorded scores and loss charts.

These are saved snapshots, not live leaderboard data. For a fresh score check,
run `.venv/bin/python pipeline/leaderboard_status.py --json` from the repository
root. Detailed current baseline artifacts remain in `../runs/`.
