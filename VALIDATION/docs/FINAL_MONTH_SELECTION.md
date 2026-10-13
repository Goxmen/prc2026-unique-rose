# Four-month model selection

The original ten-recipe run was launched by the user on 13 October. At the
user's subsequent request, it was reduced to **three finalists** and distributed
across desktop and laptop. Results and final selection remain pending. The earlier
research loop remains paused; validation never uploads or selects a final model.

The active controller is `tools/final_selection/reduced_three.py`. The original
runner and its 456 frozen Python sources remain byte-for-byte unchanged, so
completed January components retain their original provenance. The old scheduler
and its unselected physical-PLE worker were stopped; the latter's last atomic
checkpoint is retained. Do not restart the old ten-recipe command.

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

## Three active finalists

`Trees` is the equal mean of main LightGBM184, per-airport LightGBM168, and
CatBoost174. Every finalist keeps the frozen 71% neural / 29% Trees share and
identical freshly fitted Rome mixture, rules, and training-only corrections.

| ID | Ordinary model recipe |
| --- | --- |
| C05 | Callsign PLE pool (three seeds) + Trees |
| C07 | Public physical-track TabM (seed42) + Trees |
| C10 | Equal callsign PLE pool / public TabM neural blend + Trees |

C10 remains the fixed comparison reference. These retain the callsign model and
the public TabM family that improved the historical physical-PLE replacement.
The single-family alternatives test whether combining the two is beneficial on
the revised months. Physical PLE and standalone tree candidates are excluded.
No new four-month scores were consulted to choose this shortlist.

Callsign uses seeds42/1042/2042; TabM uses seed42 only, consistently in every fold.
This preserves all three completed January callsign fits and reduces neural work
from36 to16 fits (12 callsign,4 TabM). Original C07/C10 used three TabM seeds;
these budget-reduced versions are explicitly distinct, potentially noisier models.
The three-member callsign recipe C05 is unchanged. The discarded original
candidate definitions remain available in the unmodified original source.

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

Both neural families use21 epochs from the original24-epoch schedule.
The active per-family seed counts are specified above. This consistent prospectively chosen schedule replaces the
historical month-specific stops. Callsign uses three members rather than the
historical fifteen. All use effective batch2048 with microbatches512; the
callsign dropout execution therefore differs from its historical implementation.
These differences are explicit rather than claims of identical old models.

## Distributed execution and resumption

The replacement jobs are stopped at the user's request for manual launch.
January's ten completed selected components and February's completed input
preparation are retained. No new model or submission is selected.

Run this single command from the desktop repository:

```bash
.venv/bin/python -u tools/final_selection/run_distributed.py
```

It verifies matching frozen identities, runs January/July on the desktop and
February/June on the laptop, returns completed laptop artifacts, and prints the
three-candidate score table when all four months are complete. Keep the terminal
and both machines online. Only one heavy worker runs per host. The agent does
not need to remain active or monitor training.

The underlying per-host commands are `tools/final_selection/reduced_three.py run
--threads 4 --folds 2025-01 2025-07` (desktop) and the same with `2025-02 2025-06`
(laptop). Do not launch these alongside the distributed command. Original run
locks prevent duplicate controllers. Neural checkpoints may resume only on
their original host and execution settings.

The reduced protocol, frozen controller hash, compact evaluation packets, and
results live under `VALIDATION/runs/v3_20261013_final_month_selection/reduced_three/`.
Original component models and input receipts remain in the parent run's `folds/`.
The collector returns completed laptop models, checkpoints, receipts and compact
scoring packets. The4GB full input pickles and large encoded arrays stay on their
producing host. Full input hashes are verified there; the desktop report verifies
the transferred packets and their provenance without claiming to rehash absent
remote inputs. Every prediction is fixed before any four-month scores are computed.

```bash
.venv/bin/python tools/final_selection/reduced_three.py status
cat VALIDATION/runs/v3_20261013_final_month_selection/reduced_three/scores.txt
```

The final table has only C05, C07 and C10. All four folds are required; no partial
fold result selects a candidate. `scores.csv` and `scores.json` retain detailed
metrics/provenance and paired day-bootstrap comparisons. Logs are saved to disk;
the user requested that the agent not inspect them.

## Setup checks

At preparation, only static review, metadata checks and synthetic unit tests
were performed; no real fitting or scoring was launched by the setup task.
The user subsequently started the full run. Tests cover embargo membership, hidden
target isolation, feature donor scope, train-only encoding, neural shapes/loss
semantics, candidate arithmetic, pooled/monthly scoring and corrupted cache
rejection. The running job's logs and receipts are the authority for completed
stages; a complete four-fold result is required before choosing a candidate.
