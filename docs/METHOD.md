# Method and model identity

This repository predicts the **recorded taxi-out duration in seconds** for
departures at the ten challenge airports. The supplied records have noisy and
occasionally substituted clocks; predicting this recorded value is different
from estimating physical ground movement alone. All submitted predictions are
aligned to the organizer's movement IDs and serialized to the template schema.

## Three model identities to keep separate

| Identity | Evidence and status on 13 October 2026 |
|---|---|
| v80 | Best original Jan/Jul leaderboard result, **218.8525 seconds** on 344,841 departures. Its exact full ensemble was not recovered for every final row. |
| v83-derived accepted final | Distilled portable ensemble; original leaderboard result **220.9278**, historical Jan/Jul CV **289.607608**. `unique-rose_final.parquet` was accepted on all **670,790** final departures, with no final numeric score published. |
| C01–C10 four-month validation | Ten newly frozen direct recipes, scored on January, February, June and July 2025. User-launched selection experiment; these are neither exact v80/v83 reconstructions nor an automatically selected replacement submission. |

Authoritative final identity, score and receipt references are in the release
manifest and `VALIDATION/runs/v3_20261013_v80_teacher_distillation/`. The accepted
file's SHA-256 is
`f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f`.

## Feature construction

The shared inputs combine supplied aircraft, operator, airport, stand, runway,
calendar and flight-plan clocks with traffic/history summaries. Label-dependent
histories and correction parameters are fitted on each fold's training rows.
Known arrival records provide context. METAR describes local conditions; OSM
and X-Plane maps provide stand/runway geometry; OPDI and ADSB.lol provide ground
event observations. Own-aircraft callsign/squawk changes and sparse positions
give additional views of observed motion. Missing observations and ambiguous
identity matches are represented explicitly. Sources, time conventions and
licenses are in [DATA_SOURCES.md](DATA_SOURCES.md).

Public daily pre-departure-delay and runway taxi totals are joined by their
published airport/date/runway scopes. Taxi aggregates use airport-local takeoff
dates; other sources retain their documented calendars. Public totals can have
different flight populations, exclusions and rounding from the challenge.
The historical final stack calibrates their mismatch using permitted training
data and uses robust uncertainty rather than treating the totals as exact
constraints on individual targets.

The held departure target and block-time aliases are masked before the new
four-fold feature APIs are called. Encoders, numerical bins and label-dependent
corrections are fitted from the allowed training months. The input-only caches
can be reused, but peer/arrival context is rebuilt for the permitted fold
population. These checks do not imply online, real-time deployment: the
competition supplies complete month-level movement context.

## Base learners and Rome mixture

Ordinary taxi duration is modeled with global and per-airport LightGBM,
CatBoost, and neural regression. The neural families use categorical embeddings,
training-fitted numerical transformations and piecewise-linear embeddings.
Physical-track models add a masked track encoder and observation-quality
inputs. The public TabM model uses 32 ensemble heads, each supervised by squared
error; inference averages their outputs. Three retained seeds reduce variation.
Architecture primitives come from PyTorch, `rtdl_num_embeddings` and `tabm`;
our domain feature construction, physical encoder integration, mixtures,
corrections and evaluation code are described in [THIRD_PARTY.md](THIRD_PARTY.md).

Rome requires a separate treatment for records whose taxi duration follows a
schedule-copy mode. Its ordinary regressor excludes the substituted training
targets. Classifiers estimate the recording mode and mix ordinary versus
clock-derived predictions. Frozen prediction rules and learned corrections are
then applied in their recorded order. This avoids forcing one ordinary
regression loss to explain two recording mechanisms.

## Accepted final: portable stack and teacher distillation

The accepted model uses the portable v82 public-source ensemble as anchor. That
anchor carries the retained monthly/daily corrections and joint/other-runway
increments. Some increments are explicitly transported from their historical
v70 prior as additive terms; they are not new posterior fits from the anchor.
`portable_public_source_20261013.py` and its identity/design receipts specify
the exact order and provenance.

The v83 student approximates the complete original v80 prediction using 28
historical components that have verified old and final views. In the order
stored in `production/model.json`, these are versions
`1, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 30, 33, 34, 37, 38, 45, 46,
54, 56, 57, 58, 60, 61, 64, 70`.

For anchor `z`, component prediction `p_j`, and teacher `t`, define
`X_j = p_j - z`. A fixed ridge coefficient vector solves

```text
(X.T @ X / n + I) beta = X.T @ (t - z) / n
prediction_before_constraints = z + X @ beta
```

Ridge strength is 1, without an intercept or learned scaling. The production
solve uses v80 **predictions**, not hidden evaluation labels, on the 344,841
original ranking rows. One saved coefficient vector is applied to all final
rows. The recorded constraint/rule/rounding function is then applied, and the
template is filled by ID. Replaying must use the saved coefficient vector and
component bytes; refitting the solve is a separate experiment.

Student CV uses four blocked solves within the historical January/July outer
predictions: predict days 1–15 from days 17 onward, and days 16 onward from days
1–14. Each direction has a one-day gap. Predictions precede the separate target
scoring stage. The student improves its v82 anchor by 1.231818 seconds in mean
Jan/Jul CV; its 0.155170-second improvement over v80 is inconclusive, with
January worse. Transferring this approximation to February/June was not
independently validated when the accepted final was selected.

## Current four-month model-selection protocol

The new folds have exactly one excluded adjacent month on each side, interpreted
cyclically at January. Every fold has nine training months:

| Held month | Training months in 2025 | Excluded months |
|---|---|---|
| January | March–November | December, January, February |
| February | April–December | January, February, March |
| June | January–April and August–December | May, June, July |
| July | January–May and September–December | June, July, August |

The older July fold also excluded April and May. Its historical scores must not
be mixed with the new July scores. Both use 2025 development data rather than
the hidden 2026 final labels.

The ten recipes compare individual global/per-airport/CatBoost trees, their
equal blend, and fixed combinations with callsign PLE, physical-track PLE and
public physical-track TabM. Neural family means use seeds 42/1042/2042; blended
recipes allocate 0.71 total neural and 0.29 total tree weight. All recipes share
fresh fold-fitted Rome/rule/correction stages. Exact C01–C10 weights, features,
rounds, checkpoints and commands are documented in
[FINAL_MONTH_SELECTION.md](../VALIDATION/docs/FINAL_MONTH_SELECTION.md).

The primary metric is the unweighted mean of four all-row monthly RMSEs. Pooled
RMSE and each month's RMSE are also reported. All ten prediction arrays are
assembled before scoring; selection is manual after the completed results.
Bootstrap comparisons against fixed C10 are diagnostic, not independent
confirmation after repeated development-data use.

## Limitations and reproducibility

Official leaderboard RMSE, historical two-month CV and the new four-month CV
measure different populations and cannot be substituted for one another. Model
selection used repeated development evaluation and public leaderboard feedback;
the final score is not known. Exact retained-artifact replay and clean retraining
are different reproducibility claims: missing historical models/raw snapshots
and hardware-dependent fitting mean a fresh training run is not promised to
produce byte-identical historical predictions. The reproduction guide records
what can be replayed, what needs acquisition, and what remains unverified.
