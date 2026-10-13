# Four-month model selection

Prepared on 13 October 2026 at the user's request and subsequently launched by
the user on the desktop. **Results and final selection are pending.** The user
will return the scores and then choose the final model. The earlier research
loop remains paused. This runner never uploads or selects a final model.

## Validation

All folds use 2025 labels. Each holds out one complete month and excludes the
adjacent calendar months from training, wrapping December around January as
requested. Every fold trains on nine months.

| Held out | Also excluded | Training months |
| --- | --- | --- |
| January | December, February | March–November |
| February | January, March | April–December |
| June | May, July | January–April, August–December |
| July | June, August | January–May, September–December |

The old July fit trained on January–March and September–December, excluding
April and May too. The user explicitly chose to correct that: July now includes
April and May. Existing historical CV artifacts are preserved. This suite builds
fresh inputs and fits for all four folds; no old fitted weights, encoders,
corrections or January/July prediction caches are borrowed.

Label-based histories, corrections, encoders and quantile bins use the fold's
training rows only. Training event context excludes held/buffer months; held
context can use the supplied inputs and arrival observations in the held month,
but never its departure taxi labels or movement block times. External public
daily aggregates and input-only track observations remain available as at test
time. Fold-specific peer features are rebuilt instead of reading globally
prepared peer summaries. No validation label chooses stopping epochs.

Primary score: **mean of the four monthly, all-row RMSEs**, in seconds. The
report also gives pooled all-row RMSE, January/July mean and February/June mean.
Predictions use production-style nonnegative clipping and integer rounding after
the Rome mixture, rules and corrections. Every held departure is scored.
Paired day-bootstrap comparisons against C10 are saved in `scores.json`.

These remain reused development data, including hyperparameter choices made
during earlier work. The four-fold score is not an untouched test estimate;
small differences should be considered alongside the monthly changes and
bootstrap intervals (which are exploratory and not multiplicity-adjusted).

## Ten candidates

`Trees` below is the equal mean of C01–C03's ordinary regression outputs.
For C05–C10 the ordinary blend is **71% neural / 29% Trees**, preserving the
previously selected overall neural share. Mixtures happen before shared
postprocessing. All ten use the same freshly fitted Rome R20/public CatBoost
probability mixture and training-only correction recipe.

| ID | Ordinary model recipe |
| --- | --- |
| C01 | Main LightGBM with 184 inputs, including callsign, public delay and daily taxi features |
| C02 | Per-airport LightGBM with 168 inputs, including public delay features |
| C03 | CatBoost with 174 inputs, including callsign and public delay features |
| C04 | Trees: equal mean of main LightGBM, per-airport LightGBM and CatBoost |
| C05 | Callsign PLE pool + Trees |
| C06 | Physical-track PLE pool + Trees |
| C07 | Public physical-track TabM pool + Trees |
| C08 | Equal callsign/physical PLE pools + Trees |
| C09 | Equal callsign PLE, physical PLE and public TabM pools + Trees |
| C10 | Equal callsign PLE/public TabM pools + Trees; fixed comparison reference |

These are **new, reproducible direct recipes, not v80/v83 replays**. The historical
versions also relied on many earlier affine components and public posterior
corrections with no generic February/June refitter. Reusing their cached deltas
would give invalid new-fold scores. This suite does not include those layers.
It compares strong, independently refittable model families and blends. A chosen
winner will need its corresponding full-year production fit after review; the
accepted final submission is untouched.

Fixed main LightGBM: LB1, 10,000 rounds. Airport LightGBM: P04, 5,000 rounds per
airport. Ordinary CatBoost: K07, 10,000 rounds. Each uses seed42. Rome R20 uses
1,700 rounds; public Rome CatBoost averages seeds42/142/242 with fixed
1,182 rounds for January/February and 1,401 for June/July, inherited from the
earlier seasonal settings. No new hyperparameter search is performed.

Each neural family averages seeds42/1042/2042, with 21 epochs from the original
24-epoch schedule. This consistent prospectively chosen schedule replaces the
historical month-specific stops. Callsign uses three members rather than the
historical fifteen. All use effective batch2048 with microbatches512; the
callsign dropout execution therefore differs from its historical implementation.
These differences are explicit rather than claims of identical old models.

## Run manually

From the desktop repository:

```bash
cd /home/gunes/PRC
.venv/bin/python -u VALIDATION/final_month_selection.py run --threads 4
```

This is an overnight-sized GPU workload: 36 neural fits plus shared ordinary
and Rome tree fits. Historical laptop timings put the twelve TabM fits alone
near nine hours; desktop time and fresh-fold preprocessing may differ. There is
no promised completion time. Only one component runs at a time, on this host.
The command does not contact the laptop or start other paused experiments.

Logs and artifacts are under
`VALIDATION/runs/v3_20261013_final_month_selection/`.
Each stage prints its log path; the final compact table is printed and written
to `scores.txt`, with `scores.csv` and the full provenance/uncertainty report in
`scores.json`. Paste the table back into the conversation.

```bash
cat VALIDATION/runs/v3_20261013_final_month_selection/scores.txt
```

The same run command verifies and reuses completed components. Neural fits
resume from completed epoch checkpoints. A tree fit interrupted before its model
was saved is retained as incomplete and fails explicitly rather than silently
refitting; its log/marker must be reviewed. Do not delete completed artifacts.
The runner locks its output directory against concurrent starts and refuses
mixed source/input identities. Keep the code and cached input sources unchanged
while it is running.

`--folds 2025-01 2025-02` can restrict a manual batch; omitted folds must still
be completed in the same run directory before a selection table is produced.
This is not an automatic two-machine scheduler. `report` regenerates the table
from all four completed fold outputs without training.

## Setup checks

At preparation, only static review, metadata checks and synthetic unit tests
were performed; no real fitting or scoring was launched by the setup task.
The user subsequently started the full run. Tests cover embargo membership, hidden
target isolation, feature donor scope, train-only encoding, neural shapes/loss
semantics, candidate arithmetic, pooled/monthly scoring and corrupted cache
rejection. The running job's logs and receipts are the authority for completed
stages; a complete four-fold result is required before choosing a candidate.
