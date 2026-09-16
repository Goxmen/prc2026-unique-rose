# Occasional tools

The everyday commands remain at the top level of `VALIDATION/`. Run these
helpers from the repository root:

| Tool | Use | Trains / uploads? |
|---|---|---|
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
