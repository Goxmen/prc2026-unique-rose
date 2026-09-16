# Validation

Start with the [improvement plan](IMPROVEMENT_PLAN.md). **Stage 0 is complete;
exact-twin flight-plan recovery (R1) is next.**

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
| `runs/` | Current B0 artifacts and the central `experiments.jsonl` ledger |
| [tools/](tools/README.md) | Occasional audits, charts and submission utilities |
| [archive/](archive/README.md) | Historical findings and five earlier experiments |
| `tests/` | Regression checks for the model and workflow |

## Main commands

Run these from the repository root with `.venv/bin/python`:

```bash
# Inspect the validation schedule without training.
.venv/bin/python VALIDATION/harness.py --dry-run

# Train/evaluate a configuration; use the round count agreed in the plan.
.venv/bin/python VALIDATION/harness.py --label <experiment> --rounds <rounds> --threads 8

# Compare existing results without training.
.venv/bin/python VALIDATION/compare.py --run <candidate> --baseline <reference>

# Continue an interrupted run, preserving completed fits and folds.
.venv/bin/python VALIDATION/resume_run.py --run <id> --threads 8 --fit-workers 1

# Full-year refit for a selected, validated recipe, when submissions resume.
.venv/bin/python VALIDATION/submit.py --version <N> --run-id <id>
```

`submit.py` writes a submission locally; upload is a separate step. The optional
[threshold-sweep tool](tools/README.md) trains and uploads automatically.
Full-year refits and uploads remain deferred during the improvement experiments.

## Saved results

The current baseline is
[`v3_20260915T225301657023Z`](runs/v3_20260915T225301657023Z/result.json).
Older run IDs still work in the readers and comparison tools; they resolve
through `archive/runs/`. The experiment ledger retains the complete history,
with legacy and v3 score definitions kept separate.

Concise results and charts are tracked. Models, prediction parquets and source
snapshots stay local and gitignored. Archiving a run moves its entire directory
without changing its contents.

The six shared implementation modules (`modeling.py`, `features_time.py`,
`movement_data.py`, `protocol.py`, `records.py`, `reporting.py`) remain beside the
commands. Core model paths and contents are stable so saved baseline source
checks continue to work.

## Checks

```bash
.venv/bin/python -m unittest discover -s VALIDATION/tests
```

The superseded v2 protocol, original simulation/HTML, and one-off CPU benchmark
script were removed from the working tree. Their last versions remain in Git;
see [the archive index](archive/README.md).
