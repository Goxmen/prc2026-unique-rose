# Validation operating manual

See the [current plan](../IMPROVEMENT_PLAN.md), [Stage 0 results](../reports/STAGE0.md),
and [protocol v3](PROTOCOL.md). The top-level [README](../README.md) is the entry point.

## What a run does

Six reusable models, sixteen evaluations, three seeds. Each model `M1`–`M6` is
fitted on a six-month window of 2025 and reused for every validation month
assigned to it. `A` folds validate the month right after training (`k = 1`);
`B` folds validate three to six months later with the same model.

| model | train | A fold (next month) | B folds (longer gap) |
|---|---|---|---|
| M1 | Jan–Jun | A1 → Jul | B1 → Sep, B2 → Oct, B3 → Nov, B4 → Dec |
| M2 | Feb–Jul | A2 → Aug | B5 → Oct, B6 → Nov, B7 → Dec |
| M3 | Mar–Aug | A3 → Sep | B8 → Nov, B9 → Dec |
| M4 | Apr–Sep | A4 → Oct | B10 → Dec |
| M5 | May–Oct | A5 → Nov | |
| M6 | Jun–Nov | A6 → Dec | |

Optional diagnostics, each costing one more fit per seed: `G7` (Jan–May → Dec,
a real seven-month gap on five training months), `F1` (Jan–Nov → Dec, the
near-final training size), `W1` (Feb–Dec → Jan, retrospective, without any
departure-history features), and `S1` (M1 scored on July with departure history
frozen at January 31, the feature-only staleness check against `A1`).

The legacy names `jul_fresh, aug, sep, oct, nov, dec_jan26, jul_stale` map onto
`A1`–`A6` and `S1`; `apr` still exists but needs `--train-months 3` or less.

## The availability rules, and where each one lives

| rule (protocol §3) | implementation |
|---|---|
| Labels, categories and target aggregates come only from the training window | `harness.execute` slices the window with `modeling.period_rows` before anything is built; `modeling.context` never sees other months |
| Training rolling features are causal | `TimeFeatures.transform(freeze_day=None)` ends every window the day before the row |
| Validation departure history is frozen at the cutoff | `validation_matrix` passes the fold's `freeze` as `freeze_day` for `op_taxi_30d`, `op_taxi_30d_n`, `rwy_taxi_30d`, `aphour_taxi_30d`, `stand_taxi_90d` |
| Current context = training window + the validation month, nothing in between | `validation_matrix` is given the training frames plus that month's rows only, so B4 sees Jan–Jun and Dec; December 1 has no arrival history |
| Traffic counts ignore the departure label | `modeling.observable` strips `taxi_sec` and `BLOCK_TIME_UTC_mvt` before event tables are built; arrival taxi-ins are filtered to 1–7,200 s on both sides |
| Missing history stays missing | no fill; `arrival_history_available` and `arrival_history_days_available` are saved per prediction and summarised for days 1–7 |

`tests/test_v3.py` pins each of these with a small synthetic dataset:

```bash
.venv/bin/python -m unittest VALIDATION/tests/test_v3.py
```

## Training configuration

Stage 0 fitted a **1,500-round B0**, then found that its 800-round checkpoints
performed better on next-month validation. See [the plan](../IMPROVEMENT_PLAN.md)
for the explicit round-count/reference choice when continuing. The CLI default
remains 800; pass `--rounds 1500` only when that is the intended experiment.

On the current i7-13650HX laptop use `--fit-workers 1 --threads 6
--prediction-threads 20`; see [the measurements](../reports/STAGE1_CPU.md).
Run full-data audits sequentially with experiments to stay within WSL's memory
allocation. Stage 0's eight-thread recommendation applies to its 7800X3D desktop.
The optional `--fit-workers 2 --threads 8` mode shares an eight-thread
budget across two fits (four each), with eight for an unpaired final seed.
Models are fitted once per window and seed and reused across their assigned
folds. Every fit records its actual thread count and per-round training loss.

`resume_run.py --run <id> --threads 6 --fit-workers 1` continues an interrupted
run using completed models and folds. It requires unchanged model source and
data, records execution-setting changes and continuation source snapshots, and
keeps one final entry per run in the experiment log.

`--checkpoints 800` also saves the 800-tree predictions from each longer fit in
the same prediction parquet. After completion, `tools/summarize_baseline.py --run <id>`
plots the training curves and makes the paired 1,500-versus-800-round comparison.
Checkpoint scoring is diagnostic; it does not select rounds or early-stop.

Reference: the LightGBM parameters in `modeling.PARAMS`, **800 fixed rounds**,
seeds **42, 142, 242**, training labels **1–200,000 s** (`--max-taxi-sec`),
aggregate inputs 1–7,200 s, negative predictions clipped to zero, unrounded
predictions for local scores. Validation rows are never filtered.

`--early-stopping-rounds N` switches to inner selection: fit on the first five
months of the window, pick the round count on the sixth, refit on all six, and
use that one model for every horizon of the window. The selected counts and the
extra fit time are recorded per model and seed under `models` in `result.json`.
Changing this policy makes the run a different series; rerun the baseline
under the same policy before comparing.

## Scores

Everything is pooled per flight (protocol §6). A flight's loss is its squared
error averaged over seeds; a December flight that is predicted by both B4 and
B7 contributes the mean of those two losses, once.

| readout | folds | meaning |
|---|---|---|
| `R_A` | A1–A6 | next-month performance, all July–December flights |
| `R_B_long` | B3, B4, B7 | longer gap (`k >= 5`), November and December only |
| `R_B_k3` … `R_B_k6`, `R_B_all` | B folds by gap / all | descriptive |
| `R_combined` | | `sqrt(w_jan R_A² + w_jul R_B_long²)` with the ranking row counts as weights; a **provisional diagnostic**, never a leaderboard forecast |
| `B1_vs_A3` … `B10_vs_A6`, `S1_vs_A1` | | longer-gap minus next-month RMSE on the *same* flights |

A `_partial` suffix means the run did not contain every fold of that group;
`full_benchmark` in the record is true only for an unsampled, six-month,
A-plus-B run.

Every run also reports, per fold and for `R_A` / `R_B_long`: RMSE on targets
at or below 7,200 s; count, RMSE and squared-error share of targets above it;
per-airport RMSE; LIRF split by `AOBT_3_flt` present/missing; and the arrival
history coverage of days 1–7. Empty slices print `N/A`.

## Comparing two runs

Pass `--baseline <run id>` to a new run, or recompute afterwards:

```bash
.venv/bin/python VALIDATION/compare.py --run <candidate> --baseline <baseline>
.venv/bin/python VALIDATION/compare.py --run <candidate> --baseline <baseline> --block-days 3
```

Differences are **candidate minus baseline**, so negative favours the
candidate. Each carries a 95% interval from 2,000 paired bootstrap replicates
that resample UTC calendar days within every validation month, using the same
day draws for both runs, every seed and every fold. `--block-days` resamples
multi-day blocks for close calls where disruptions persist. The report marks
each interval `improvement`, `regression` or `inconclusive`; an inconclusive
interval is not evidence of equivalence. Per-seed differences and per-airport
differences that oppose the headline are printed alongside. The baseline must be
a v3 run with the same seeds, folds, data files and round policy, or the
comparison refuses.

Decide on `R_A` and inspect `R_B_long` before advancing a milestone. If they
disagree, look at the paired intervals, the matched-month aging rows, the
seeds and the tail slices before blaming the horizon.

## Running it

```bash
# execution check only, a few minutes: 5% sample, 100 rounds, one seed.
# Sampled scores cannot rank anything (the congestion counts shrink 20x).
.venv/bin/python VALIDATION/harness.py --smoke

# print the schedule and fit count without loading data
.venv/bin/python VALIDATION/harness.py --dry-run --folds all

# the default v3 benchmark: 16 evaluations, 6 models x 3 seeds = 18 fits
.venv/bin/python VALIDATION/harness.py --label baseline-v3

# a candidate, paired against that baseline
.venv/bin/python VALIDATION/harness.py --label fpl-flag --fpl-flag --baseline <baseline id>

# everything, including G7/F1/W1/S1 (27 fits per seed set)
.venv/bin/python VALIDATION/harness.py --label full --folds all
```

Other flags: `--seeds`, `--rounds`, `--train-months` (a different length is a
variant, not the v3 benchmark), `--lag-train-days`, `--max-taxi-sec`,
`--early-stopping-rounds`, `--bootstrap-reps`, `--bootstrap-seed`,
`--block-days`, `--threads`, `--prediction-threads`, `--output-dir`.
`--recovery off|exact|relaxed|history` selects the shared fold-scoped recovery
path; see [the design and audit commands](RECOVERY.md). All flags are recorded.

Each run writes `runs/v3_<timestamp>/`:

- `result.json` — configuration, dataset hashes, git commit, per-model rounds
  and fit times, fold definitions and context months, the full report with
  every slice and interval, and (later) the official score. Tracked in git.
- `oof_<fold>_seed<seed>.parquet` — movement ID, time, airport, target, flight
  plan flag, coverage indicators and prediction for every fold and seed.
- `models/` — the fitted boosters; `source/` and `source.patch` — the exact
  code, including uncommitted changes. These stay local (gitignored).
- one summary line appended to `runs/experiments.jsonl`.

Earlier experiments live in `archive/runs/` and still resolve by run ID.
The ledger remains in `runs/experiments.jsonl`. Legacy scores use a different
definition and are never mixed with v3 scores.

## Submitting

`submit.py` refits the validated recipe on all twelve months and fills the
template, scoring January and July as separate islands with departure history
frozen at 2025-12-31 and only that month's arrivals and movements as context.

```bash
.venv/bin/python VALIDATION/submit.py --version 4 --run-id <v3 run id>
mc cp submissions/unique-rose_v4.parquet opensky/prc-2026-unique-rose/
mc cat opensky/prc-2026-unique-rose/unique-rose_v4.parquet_result.json   # the score
```

With `--run-id` the recipe (rounds, label ceiling, flags, first seed) is taken
from that run; any explicit flag that disagrees with it is an error, as is a
change to the model code or the data files since the run. Without `--run-id`
the defaults apply and the metadata says so. A `.meta.json` and the model are
written next to the submission.

### Full-year threshold sweep with direct submission

`sweep_taxi_thresholds.py` trains and uploads one model for each ceiling:
**10,000, 16,000, 22,000, 32,000, 50,000, 70,000, and 90,000 seconds**.
Every fit uses all twelve training months, 800 fixed rounds, seed 42, and the
same features as submission v3 (`fpl_flag=False`). It skips local validation
and early stopping; ordinary template/schema/ID checks still run. The ceiling
filters training target rows; historical aggregates retain their existing
1–7,200-second input filter.

```bash
# Preview the seven version IDs, checking both local files and the live bucket.
.venv/bin/python VALIDATION/tools/sweep_taxi_thresholds.py --dry-run

# Train all seven models sequentially and upload each as soon as it is ready.
.venv/bin/python VALIDATION/tools/sweep_taxi_thresholds.py

# Continue using the manifest path printed when the sweep started.
.venv/bin/python VALIDATION/tools/sweep_taxi_thresholds.py --resume VALIDATION/reports/sweeps/<id>.json

# Refresh scores later; scoring may still be pending when the uploads finish.
.venv/bin/python VALIDATION/tools/sweep_taxi_thresholds.py --status VALIDATION/reports/sweeps/<id>.json
```

The script automatically assigns the next seven unused `unique-rose_vN.parquet`
IDs. `VALIDATION/reports/sweeps/<id>.json` records the threshold-to-filename mapping,
settings, source/data hashes, training row counts, upload progress, and official
scores. Keep this small manifest in git to share the mapping. Models, prediction
parquets, and per-submission metadata remain under gitignored `submissions/`.
The uploaded objects also carry `sweep-id`, `max-taxi-sec`, and `sha256` metadata.

The terminal shows an overall **seven-model progress bar** plus the current
model's **boosting-round progress and training RMSE**. Each model writes
`submissions/unique-rose_vN.training_loss.json` with RMSE for all 800 rounds.
`VALIDATION/reports/sweeps/<id>_training_loss.png` overlays the curves and updates after
each model finishes. These are losses on each model's filtered training rows;
use official scores to compare thresholds, since the training sets differ.

On resume, completed uploads are verified by content and skipped; completed fits
are reused. Source/data changes and ID collisions stop the sweep. If interrupted
while writing a model's output files, incomplete artifacts cause an explicit
error; move that version's partial local files aside before resuming. Check the
printed manifest rather than assuming IDs from an earlier preview are reserved.

## Closing the loop with the leaderboard

```bash
.venv/bin/python VALIDATION/tools/log_official.py --run <v3 run id> --official 401.2 \
    --submission unique-rose_v4.parquet
.venv/bin/python VALIDATION/tools/log_official.py --report
```

The report groups matched local/official pairs by score definition, so legacy
pooled projections and v3 `R_combined` values are listed as separate series.
It prints the offsets, their spread, the correlation once three pairs exist,
and any pair whose local and official changes point in opposite directions.
Repeated sign disagreement means the protocol needs revisiting before more
feature work.

## Known limits

- Neither A nor B is a calibrated estimate of its named 2026 month. The longest
  gaps land in November and December, not in July.
- December takes part in development, so `F1` is a training-size check, not an
  untouched test. `W1` trains on future months with fewer features and cannot
  verify the deployed model.
- Bootstrap intervals condition on these fitted models and these months. They
  say nothing about a different year.
- Repeated searching over the same six validation months can overfit them
  despite the intervals. Record failed experiments too.
