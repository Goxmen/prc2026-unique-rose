# Validation

The [September 19 recording-artifact study](reports/RECORDING_ARTIFACT_EXPERIMENTS.md)
is an archived exploratory study based on row 34 in [SUBMISSIONS.md](../SUBMISSIONS.md).
Its entries were removed from the submission list at the user's request. No
experiment was promoted or independently confirmed; duplicate predictions,
inactive changes and inconclusive differences remain in the study report only.
The [design](docs/RECORDING_ARTIFACT_EXPERIMENTS.md) records the recipes and
training-window calibration. No additional submission is authorized.

The paragraphs below describe earlier studies and their dated score snapshots.

The [overnight study is complete](reports/OVERNIGHT_SUMMARY_20260917.md): all five
new submissions succeeded. **Explicit clock features/v19 is best at 338.1113 s**;
latest is routing/v21 at 342.4009 s. The September 17 04:23:02 UTC snapshot
records rank 94/152 scoring teams. The selected CV alternative, **R_A__mse**,
failed its two-candidate confirmation (change MAE 3.5178 s versus the current
score's 2.8367 s). **Keep the default validation and all folds.** More varied
future confirmation is needed. All five authorized slots are used; completed
fits/uploads must not be repeated and no additional uploads are authorized.

The study completed 153 new CV fits, reused nine, saved 360 prediction files,
and ran six full-year refits. All 101 tests passed. Its run directory is
[`runs/overnight_20260917/`](runs/overnight_20260917/), including the sealed
[confirmation report](runs/overnight_20260917/confirmation_report.md).
The investigations below are earlier context; their historical recipes and
results remain unchanged.

The [month-balanced aggregation proposal](reports/AGGREGATION_COMPARISON.md)
keeps all 16 folds while giving July–December equal total weight. It has been
computed from saved runs; the default harness is unchanged.

The [optimism diagnosis](reports/VALIDATION_OPTIMISM.md) explains the original
versus seasonal score gap through population weights and rare-error influence.
Current decision: retain the existing validation; no fold removal or seasonal
replacement has enough evidence. No row exclusions or score correction were made.

The requested [original-fold alignment analysis](reports/FOLD_ALIGNMENT.md) is
complete: it compares all folds and possible reductions against the five recorded
LB references. No folds have been removed. Reproduce it without fitting using
`tools/analyze_fold_alignment.py`.

The [January/July seasonal comparison](docs/SEASONAL_VALIDATION.md) is also complete:
[results](runs/seasonal_jan_jul/comparison.md) show closer absolute scores, but
worse alignment in change sizes than v3. No replacement protocol is selected.

Start with the [improvement plan](IMPROVEMENT_PLAN.md) and
[Stage 4 results](reports/STAGE4.md). **Stages 1–4 are complete. M1 regresses;
keep Q0 as the conservative control.** The [frozen mixture recipe](docs/TIMESTAMP_MIXTURE.md)
and report record the failed classifier and false high predictions.
The [exact feature list and examples](docs/STAGE2_STAGE3_EXAMPLES.md)
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
| [reports/OVERNIGHT_SUMMARY_20260917.md](reports/OVERNIGHT_SUMMARY_20260917.md) | Five new submissions, held-out CV verdict, current best and remaining evidence gap |
| [docs/OVERNIGHT_STUDY_20260917.md](docs/OVERNIGHT_STUDY_20260917.md) | Frozen overnight candidate design and discovery/confirmation split |
| `overnight_run.py`, `overnight_models.py` | Resumable study fits/refits and fixed model interventions; completed run is retained |
| `overnight_analysis.py`, `overnight_submit.py` | CV scorecards, immutable selection, conditional uploads and separately revealed official scores |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Current v3 folds, availability rules and scoring |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Detailed commands, configuration and result formats |
| [reports/](reports/README.md) | Stage summaries, CPU measurements and submission-sweep records |
| `runs/` | Control/candidate artifacts and the central `experiments.jsonl` ledger |
| [docs/RECOVERY.md](docs/RECOVERY.md) | Recovery modes, availability boundaries, audits and notebook reuse |
| [docs/PREDICTION_RULES.md](docs/PREDICTION_RULES.md) | Fixed rule, saved-prediction comparisons and linked submission recipes |
| [docs/CLOCK_MODELS.md](docs/CLOCK_MODELS.md) | Stage 3 clock features, baseline fallback and residual reconstruction |
| [docs/TIMESTAMP_MIXTURE.md](docs/TIMESTAMP_MIXTURE.md) | Stage 4 outcome definitions, component training and probability mixture |
| [docs/STAGE2_STAGE3_EXAMPLES.md](docs/STAGE2_STAGE3_EXAMPLES.md) | Exact feature names and numerical examples for Stages 2–3 |
| [docs/LEADERBOARD_CANDIDATES.md](docs/LEADERBOARD_CANDIDATES.md) | Frozen five-candidate leaderboard queue, recipes, comparisons and preparation |
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
  --threads 8 --prediction-threads 8 --fit-workers 1

# Compare existing results without training.
.venv/bin/python VALIDATION/compare.py --run <candidate> --baseline <reference>

# Register the fixed Rome rule on saved predictions, without fitting.
.venv/bin/python VALIDATION/postprocess.py --parent <run> --label <rule-experiment>

# Continue an interrupted run, preserving completed fits and folds.
.venv/bin/python VALIDATION/resume_run.py --run <id> --threads 8 --fit-workers 1

# Full-year refit for a selected, validated recipe, when submissions resume.
.venv/bin/python VALIDATION/submit.py --version <N> --run-id <id>
```

`submit.py` writes a submission locally; upload is a separate step. The optional
[threshold-sweep tool](tools/README.md) trains and uploads automatically.
The September 16 and September 17 authorized refits/uploads are complete;
additional uploads require a new user instruction. The settings above apply
to the current 7800X3D desktop. The [Stage 1 CPU report](reports/STAGE1_CPU.md)
records the separate historical laptop settings; do not mix the two machines.

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
