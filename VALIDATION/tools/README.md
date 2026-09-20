# Occasional tools

The everyday commands remain at the top level of `VALIDATION/`. Run these
helpers from the repository root:

**Current validation headline: [Weighted CV](../docs/WEIGHTED_CV.md), adopted on
20 September 2026.** `compare_weighted_cv_leaderboard.py` reproduces the initial
six scored cases; `extend_weighted_cv_leaderboard.py` adds v17–v21;
`weighted_cv_recent_submissions.py` scores the five most recent uploads (v34,
v37, v38, v39, v40). All reuse saved taxi-out predictions and the same weights.
The initial tool can regenerate the four discriminator fits if its weight cache
is missing, and the recent-submission tool then checks the rebuild against the
recorded discriminators before scoring; none of them uploads.
Earlier recommendations below are historical.

The [September 17 overnight study](../reports/OVERNIGHT_SUMMARY_20260917.md)
is complete: v17–v21 all succeeded, v19 is best at 338.1113 s, and the proposed
CV alternative failed held-out confirmation. Keep the current default/folds;
further varied confirmation is needed. All five authorized slots are consumed.
Its top-level modules are listed here for discovery; do not rerun completed fits
or uploads, and do not treat this index as authorization for another batch.

| Study module (under `VALIDATION/`) | Use | Trains / uploads? |
|---|---|---|
| `rome_calibration_run.py --output-dir VALIDATION/runs/v3_20260919_rome_calibration` | Two Rome calibration experiments; inner chronological calibration, resumable classifiers, full original validation and automatic validation-only `SUBMISSIONS.md` rows; [design and command](../docs/ROME_CALIBRATION.md) | Local classifiers only; no uploads |
| `overnight_run.py` | Resumable original/seasonal validation and full-year refits; preserves completed artifacts | Local fits only |
| `overnight_models.py` | Fixed control and five feature/capacity/routing interventions | Imported fitting adapter |
| `overnight_analysis.py` | Saved CV scorecards, immutable selection and held-out comparison | Neither |
| `overnight_submit.py` | Prepare artifacts, explicitly upload the fixed queue, reveal discovery/confirmation scores separately | Only explicit `upload` writes remotely |

| Tool | Use | Trains / uploads? |
|---|---|---|
| `plot_scored_residuals.py --leaderboard <status.json>` | Compare scored recipes using saved OOF; export residual plots, interactive HTML, PDF and slice/worst-flight tables, with replay and missing-artifact labels | Neither |
| `compare_aggregations.py` | Compare fixed month/flight-balanced overview scores using all original folds and saved results | Neither |
| `analyze_validation_optimism.py` | Diagnose population weights, January error concentration and observable 2026 shifts using saved predictions | Neither |
| `analyze_fold_alignment.py` | Compare original folds, seed stability and exploratory reductions against the five recorded LB scores; export tables and charts | Neither |
| `audit.py` | Check dataset availability, duplicates and target shape | Neither |
| `summarize_baseline.py --run <id>` | Plot training loss and compare saved round checkpoints | Neither |
| `benchmark_cpu.py --output <json>` | Measure local thread throughput and prediction equivalence | Local probe fits only |
| `inventory_derived.py` | Match existing notebook caches to raw donors without rebuilding them | Neither |
| `verify_recovery_off.py --baseline <id> --output <json>` | Compare full matrices with frozen control source | Neither |
| `audit_recovery.py --mode exact\|relaxed --output <json>` | Mask known plans/true twins and measure matching errors | Neither |
| `audit_static_history.py --output <json>` | Measure static-field accuracy with causal/frozen history | Neither |
| `summarize_recovery.py --run <id> --baseline <id>` | Paired recovery slices and training-loss plots | Neither |
| `summarize_stage1.py --control <id> --r1 <id> --r2 <id> --r3 <id>` | Compare Stage 1 results with paired confidence intervals | Neither |
| `summarize_rule.py --q0 <id> --q1 <id>` | Summarize rule comparisons, changed IDs and damaging overrides | Neither |
| `summarize_clock_models.py --t1 <id> --t2 <id>` | Summarize Stage 3 comparisons and training curves from completed reports | Neither |
| `inspect_clock_residual_errors.py --t1 <id> --t2 <id>` | Diagnose residual errors by saved clock source and list damaging predictions | Neither |
| `summarize_timestamp_mixture.py --run <id>` | Compare Stage 4 with T1/Q0/Q1; plot component losses and count false high predictions | Neither |
| `sweep_rule_thresholds.py [--hours 1 2 3 4 6 --output <prefix>]` | Score Rome thresholds on saved B0-800/R3 predictions; default 4–8 h, six-hour control always included | Neither |
| `log_official.py --run <id> --official <score> --submission <file>` | Attach an already-observed official score to a run | Neither |
| `sweep_taxi_thresholds.py --dry-run` | Preview threshold-to-submission IDs against local files and the bucket | Neither |
| `sweep_taxi_thresholds.py --status <manifest>` | Refresh a saved sweep's statuses and scores | Neither |
| `sweep_taxi_thresholds.py` | Run the seven-threshold full-year sweep | **Trains and uploads** |

Example:

```bash
.venv/bin/python VALIDATION/tools/summarize_baseline.py --run v3_20260915T225301657023Z
.venv/bin/python VALIDATION/tools/log_official.py --report
```

Sweep manifests now live in `VALIDATION/reports/sweeps/`. Their saved filenames
still identify the original leaderboard uploads. See the
[operating manual](../docs/OPERATIONS.md) for resume and submission details.
