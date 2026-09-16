# Occasional tools

The everyday commands remain at the top level of `VALIDATION/`. Run these
helpers from the repository root:

| Tool | Use | Trains / uploads? |
|---|---|---|
| `audit.py` | Check dataset availability, duplicates and target shape | Neither |
| `summarize_baseline.py --run <id>` | Plot training loss and compare saved round checkpoints | Neither |
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
