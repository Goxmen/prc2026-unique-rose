# Validation

Start with the [improvement plan](IMPROVEMENT_PLAN.md). Stage 4 is running with
the [frozen timestamp-mixture recipe](docs/TIMESTAMP_MIXTURE.md), comparing M1
with T1, Q0 and Q1. The [exact feature list and examples](docs/STAGE2_STAGE3_EXAMPLES.md)
explain the preceding stages in plain language. See
[Stage 3 results](reports/STAGE3.md). **Stage 3 is complete: T1 is inconclusive,
T2 regresses substantially, and Q0 remains the conservative control.** The
[frozen clock-model design](docs/CLOCK_MODELS.md) and report explain the
schedule-fallback failure. Q1 remains a tradeoff. The shared pipeline supports exact twins,
relaxed twins and static history; the report records their paired comparisons.

The [Stage 0 report](reports/STAGE0.md) records the baseline, loss charts, and
800-versus-1,500-round comparison. The evidence favors 800 rounds; the saved
1,500-round B0 remains a distinct reference. Keep the chosen round count and
comparison explicit when continuing.

## Where things live

| Location | Purpose |
|---|---|
| [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) | Current decisions, next step, experiment ledger and handoff |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Current v3 folds, availability rules and scoring |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Detailed commands, configuration and result formats |
| [reports/](reports/README.md) | Stage summaries, CPU measurements and submission-sweep records |
| `runs/` | Control/candidate artifacts and the central `experiments.jsonl` ledger |
| [docs/RECOVERY.md](docs/RECOVERY.md) | Recovery modes, availability boundaries, audits and notebook reuse |
| [docs/PREDICTION_RULES.md](docs/PREDICTION_RULES.md) | Fixed rule, saved-prediction comparisons and linked submission recipes |
| [docs/CLOCK_MODELS.md](docs/CLOCK_MODELS.md) | Stage 3 clock features, baseline fallback and residual reconstruction |
| [docs/TIMESTAMP_MIXTURE.md](docs/TIMESTAMP_MIXTURE.md) | Stage 4 outcome definitions, component training and probability mixture |
| [docs/STAGE2_STAGE3_EXAMPLES.md](docs/STAGE2_STAGE3_EXAMPLES.md) | Exact feature names and numerical examples for Stages 2–3 |
| [tools/](tools/README.md) | Occasional audits, charts and submission utilities |
| [archive/](archive/README.md) | Historical findings and five earlier experiments |
| `tests/` | Regression checks for the model and workflow |

## Main commands

Run these from the repository root with `.venv/bin/python`:

```bash
# Inspect the validation schedule without training.
.venv/bin/python VALIDATION/harness.py --dry-run

# Train/evaluate a configuration; use the round count agreed in the plan.
.venv/bin/python VALIDATION/harness.py --label <experiment> --rounds <rounds> \
  --threads 6 --prediction-threads 20 --fit-workers 1

# Compare existing results without training.
.venv/bin/python VALIDATION/compare.py --run <candidate> --baseline <reference>

# Register the fixed Rome rule on saved predictions, without fitting.
.venv/bin/python VALIDATION/postprocess.py --parent <run> --label <rule-experiment>

# Continue an interrupted run, preserving completed fits and folds.
.venv/bin/python VALIDATION/resume_run.py --run <id> --threads 6 --fit-workers 1

# Full-year refit for a selected, validated recipe, when submissions resume.
.venv/bin/python VALIDATION/submit.py --version <N> --run-id <id>
```

`submit.py` writes a submission locally; upload is a separate step. The optional
[threshold-sweep tool](tools/README.md) trains and uploads automatically.
Full-year refits and uploads remain deferred during the improvement experiments.
The thread settings above are measured for the current i7-13650HX laptop;
benchmark another machine before carrying them over. See [the CPU report](reports/STAGE1_CPU.md).

## Saved results

The Stage 1 recovery-off control is
[`v3_20260916T095948645262Z`](runs/v3_20260916T095948645262Z/result.json),
at 800 rounds. The original Stage 0 1,500-round run remains a distinct record:
[`v3_20260915T225301657023Z`](runs/v3_20260915T225301657023Z/result.json).
Older run IDs still work in the readers and comparison tools; they resolve
through `archive/runs/`. The experiment ledger retains the complete history,
with legacy and v3 score definitions kept separate.

Concise results and charts are tracked. Models, prediction parquets and source
snapshots stay local and gitignored. Archiving a run moves its entire directory
without changing its contents.

The shared implementation modules (`modeling.py`, `features_time.py`,
`movement_data.py`, `protocol.py`, `records.py`, `reporting.py`) remain beside the
commands. Recovery modules live in `pipeline/recovery*.py` and participate in
the model source hashes checked by linked submission recipes.

## Checks

```bash
.venv/bin/python -m unittest discover -s VALIDATION/tests
```

The superseded v2 protocol, original simulation/HTML, and one-off CPU benchmark
script were removed from the working tree. Their last versions remain in Git;
see [the archive index](archive/README.md).
