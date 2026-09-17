# Retrospective January/July validation

Requested 2026-09-17. Compare two validation strategies on the **same five
already-scored recipes**: B0-800/v12, Q0/v13, Q1/v14, Q0-4h/v15 and Q1-4h/v16.
Their recorded official scores come from `reports/leaderboard_selected_20260916.json`.
This runs local validation only; it does not refit submission models or upload.

| Strategy | Training months in 2025 | Held-out months | Fits per seed |
|---|---|---|---:|
| `10m` | February–June and August–December | January and July, scored separately using the same model | 2 |
| `11m` | February–December; separately, all months except July | January; separately, July | 4 |

The counts include two model parents: baseline/recovery-off and R3/history.
Rule variants reuse the corresponding parent's predictions. With both strategies
and seeds 42/142/242 this is **18 fits**, each at 800 rounds. The original 27
features, LightGBM parameters, 1–200,000-second training-label filter, 1–7,200-
second history inputs, clipping, original-ID rule eligibility, and 4/6-hour
thresholds are retained. Every held-out departure is scored, including extreme,
zero and negative targets. Validation predictions are not rounded.

## Availability and retrospective interpretation

The model, categories, recovery references and labelled departure histories see
**only the listed training months**. Training histories/static fills remain
causal within those retained months. Training recovery cannot see held-out
arrivals. Validation exposes those training months and its own held-out month;
the other held-out month in `10m` is unavailable to it. In `11m`, the other target
month is legitimately a training month for that fit.

For validation, departure-target history windows end at **December 31, 2025**,
using retained training labels only. R3's learned references are likewise frozen
from training. This deliberately permits future-in-calendar *training* data for
January/July, as the requested retrospective fits do. It preserves all 27 inputs;
it is a different experiment from W1's reduced-feature diagnostic. It does not
simulate a forecast made before January/July 2025 or remove 2025-to-2026 drift.
No held-out departure target or block timestamp can affect features or recovery.
Arrival histories and traffic counts use the existing local clock windows;
excluded-month gaps are not filled. July has June context in these strategies,
unlike July 2026, another limit to record when interpreting alignment.

The runner restores the hash-verified Stage 1 `modeling.py` from the archived
snapshot or Git and verifies its unchanged dependencies. R3 uses its original
implementation. Baseline uses that wrapper's recovery-off path, whose exact
equivalence to B0-800 on all original matrices is recorded in
`reports/stage1_off_equivalence.json`. This is explicitly recorded as an
equivalent implementation, not byte-identical B0 source. Existing submission
source guards are untouched. Dataset, library, rule, source and reference hashes
are checked and saved. Models, predictions and source copies stay local.

## Run

From the repository root, on this 7800X3D desktop (eight threads, sequential fits):

```bash
.venv/bin/python VALIDATION/seasonal.py --dry-run

.venv/bin/python VALIDATION/seasonal.py \
  --strategy both --seeds 3 --threads 8 --prediction-threads 8 \
  --output-dir VALIDATION/runs/seasonal_jan_jul
```

Use `--strategy 10m` or `--strategy 11m` and a separate output directory to run
just one strategy. `--seeds 1` uses seed 42 only: six fits for both strategies.
The measured laptop settings were six fit threads and twenty prediction threads;
do not transfer desktop settings to that machine automatically.

To continue an interrupted run, repeat the **same command with `--resume`**.
Completed models and prediction files are checked and reused. Configuration,
source, data, library or reference changes refuse resume. Per-round training
losses are saved with every model. Runs are locked against concurrent use.

The short execution check is separate from the real run:

```bash
.venv/bin/python VALIDATION/seasonal.py --smoke --output-dir tmp/seasonal-smoke
```

It uses 1% of rows, five rounds and seed 42. Its counts/features/scores cannot
rank models or validation strategies.

## Outputs and selection evidence

`comparison.md` and `result.json` under the output directory contain:

- January and July RMSE separately, and a combined score weighted by the actual
  January/July **2026** ranking row counts (combine squared errors, not RMSEs).
- Standard three-seed loss averages and seed-42-only results. These are not
  ensembles. Per-seed, airport, ordinary/tail, Rome and all-airport AOBT3
  availability diagnostics are retained in JSON.
- Official LB minus each CV score, and the same five controlled candidate-minus-
  reference comparisons as the original submission report.
- Paired 95% intervals from 2,000 calendar-day bootstrap draws stratified by
  month, plus whether both months show supported gains. January and July remain
  visible even when their combined changes cancel.
- Absolute score-gap and delta-gap summaries for each strategy and the existing
  v3 protocol, on the exact same LB references and seed readouts.
- Direction agreement with a **predeclared 1-second tolerance**. An LB change
  within tolerance does not count as a directional success/failure; a small CV
  change against a material LB change is marked separately. Magnitude errors
  remain visible for all pairs. `--negligible-sec` records a different choice.

These five related submissions are the calibration set, with only two distinct
model parents. Closeness on them does not validate a protocol independently.
Freeze a chosen strategy before checking subsequent candidates. Neither a
constant offset fitted to LB nor automatic strategy promotion is performed.
Only pooled official LB scores are available; LB-minus-January/July gaps are
descriptive comparisons with that pooled reference, not official monthly errors.

Verification: all **62 validation tests pass**. The sampled execution completed
six fits and eight prediction files across both strategies. An interrupted
prediction step was replayed bit-identically with all six saved models unchanged;
resuming the completed run rebuilt reports without loading data or fitting.
Score weighting and LB gaps were independently checked against the generated
results. The user's full run subsequently completed all 18 fits and 24 prediction
files; see [the full comparison](../runs/seasonal_jan_jul/comparison.md).

Ten/eleven-month mean absolute score gaps are 36.21/35.73 seconds, versus v3's
70.73, but mean change errors are 12.45/10.50 versus 7.19. All three match the
three material LB change directions. Seasonal scores are pessimistic, whereas
v3 is optimistic. No replacement was selected. The subsequent
[original-fold analysis](../reports/FOLD_ALIGNMENT.md) examines whether a reduced
summary offers a useful tradeoff. Do not repeat the completed seasonal fits.
