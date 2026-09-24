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
| `stress_holdout_run.py` | Optional February–May retrospective predictions from the saved June–November v39/v40 models, selected by observable stress similarity | Neither; saved-model inference only |
| `opdi_disruption_run.py --variant v39\|v40 [--resume]` | Matched reconstructed OPDI controls and the same 12 disruption inputs; [design](../docs/OPDI_DISRUPTION_EXPERIMENT.md) | 54 local boosters per recipe; no uploads |
| `disruption_run.py [--resume]` | Fixed snow-duration/completed-arrival feature experiment versus local v38; [design](../docs/DISRUPTION_EXPERIMENT.md) | 18 local ordinary fits; no uploads |
| `rome_calibration_run.py --output-dir VALIDATION/runs/v3_20260919_rome_calibration` | Two Rome calibration experiments; inner chronological calibration, resumable classifiers, full original validation and automatic validation-only `SUBMISSIONS.md` rows; [design and command](../docs/ROME_CALIBRATION.md) | Local classifiers only; no uploads |
| `overnight_run.py` | Resumable original/seasonal validation and full-year refits; preserves completed artifacts | Local fits only |
| `overnight_models.py` | Fixed control and five feature/capacity/routing interventions | Imported fitting adapter |
| `overnight_analysis.py` | Saved CV scorecards, immutable selection and held-out comparison | Neither |
| `overnight_submit.py` | Prepare artifacts, explicitly upload the fixed queue, reveal discovery/confirmation scores separately | Only explicit `upload` writes remotely |

| Tool | Use | Trains / uploads? |
|---|---|---|
| `audit_stress_analogues.py` | Check every accepted match against literal limits and verify uncapped arrival diagnostics in SQL | Neither |
| `prepare_stress_analogues.py` | Strict same-airport snow/congestion matching; records unsupported test cases without reading departure outcomes | Neither |
| `score_stress_analogues.py` | Paired errors and input-selected examples on honestly held-out historical analogues | Neither |
| `build_opdi_cache.py` | Filter verified local raw OPDI files and build the historical departure inputs; no network | Neither |
| `score_opdi_disruption.py --variant v39\|v40` | Adopted Weighted CV, paired intervals, historical-control comparison and January diagnostics | Neither |
| `audit_opdi_disruption.py` | Check fit membership, feature alignment, correction stacking and independent SQL scores after both runs finish | Neither |
| `download_opdi_cache.py [--workers 4]` | Download and retain the original OPDI source-file roster; validate Parquet files, record hashes, and handle rejected resume requests without trusting partial files | Neither |
| `score_disruption_experiment.py` | Score the completed disruption experiment with adopted weights, paired day intervals and January diagnostics | Neither |
| `audit_disruption_experiment.py` | Independently check saved arrival features in SQL and snow history directly against METAR reports | Neither |
| `investigate_arrival_shift.py` | Step 4: local arrival weather/traffic matching, recording checks and same-day departure context | Neither |
| `check_arrival_snow_severity.py` | Exploratory follow-up separating brief from prolonged reported snow; support coverage stays explicit | Neither |
| `audit_arrival_shift.py` | Independently check raw labels, weather timestamps, traffic counts, snow duration and conditional arithmetic | Neither |
| `analyze_train_ranking_cost.py` | Step 3: connect saved OOF losses to similarity and flagged groups; entirely local, keeps January's older model separate | Neither |
| `audit_train_ranking_cost.py` | Independently check step-3 cost slices, neighbour estimates and stand support | Neither |
| `export_shift_cost_inputs.py` | Export compact seasonal OOF losses on the desktop for a one-time 13 MB transfer; preserves original hashes and score checks | Neither |
| `check_train_ranking_similarity.py` | Step 2: blocked propensity classifiers, calibration, held-out group importance and exact kNN; resumable | Diagnostic classifiers only; no uploads |
| `check_similarity_weather_scale.py` | Correct the step-2 kNN fallback scale for nearly constant weather fields; retain original outputs | Neither |
| `summarize_train_ranking_similarity.py` | Assemble current distances, propensities and neighbour IDs; run after the two tools above | Neither |
| `audit_train_ranking_similarity.py` | Independently verify date separation, IDs and original/corrected nearest-neighbour arithmetic | Neither |
| `describe_train_ranking_shift.py` | Month-matched, airport-specific training/ranking differences with all 2025 month-pair references; [step 1 findings](../reports/TRAIN_RANKING_STEP1.md) | Neither |
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
