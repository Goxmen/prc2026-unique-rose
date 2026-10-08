# Open-Meteo weather in v58: Jan/Jul comparison

All five v58 parts are tested with 11 additional ERA5 weather inputs. The user
requested this extension after the main-LightGBM screen was inconclusive. The
all-part amendment supersedes that screen's stopping gate. All fits and scoring
are complete. Adding weather everywhere worsens CV by **1.0963 seconds**; v58
remains the best observed submission. The user subsequently requested full-year
fits and uploads of both Rome-only and all-part variants, recorded below.

## Results

Mean January/July RMSE in seconds; lower is better. Each single-part row changes
only that part in v58. Intervals resample paired whole days within each month.

| Weather added to | January | July | Jan/Jul CV | Change | 95% interval |
|---|---:|---:|---:|---:|---|
| Matched v58 control | 328.4317 | 263.0286 | 295.7301 | — | — |
| Main LightGBM | 328.4272 | 263.0493 | 295.7382 | +0.0081 | [−0.0201, +0.0324] |
| Per-airport LightGBMs | 328.4070 | 263.0617 | 295.7344 | +0.0042 | [−0.0223, +0.0270] |
| CatBoost | 328.4859 | 262.9874 | 295.7367 | +0.0066 | [−0.0498, +0.0731] |
| 15 networks | 330.4043 | 264.0719 | 297.2381 | +1.5080 | [+1.0494, +2.3619] |
| Rome classifier | 328.3132 | 262.2282 | 295.2707 | −0.4594 | [−1.1327, +0.1421] |
| **All five parts** | **330.3375** | **263.3153** | **296.8264** | **+1.0963** | **[+0.2838, +2.1992]** |

The three tree-regressor changes are inconclusive and effectively flat. The
network loss dominates the all-part result, which worsens in both months.
Rome improves in both months, mostly July, but its interval includes no gain.
It remains an exploratory lead rather than a selected candidate. This classifier
handles the Rome recording rule; its gain does not by itself show better direct
taxi-duration prediction. Earlier Rome CV gains also transferred poorly to the
leaderboard (see the HPO and Rome-cost reports).

## Comparison

Each part retains v58's HPO settings, original feature set and training population.
Only the 11 weather columns are added. The network share stays 0.71, the three
tree regressors retain equal shares, and all final corrections and rules stay
unchanged. CatBoost keeps its original 134 inputs before adding weather; it does
not receive the 22 inputs used by the other ordinary regressors. Rome retains
the classifier population, including substituted-label positives.

The two held-out months contain 153,706 January and 190,713 July departures.
Ordinary training populations contain 1,596,201 and 1,140,233 departures. Tree
and classifier seed 42 is used for this comparison. Each month has a pool of
15 new networks: seed groups 42/142/242, five members each. Network epochs are
chosen on the last training month, then each member is refitted on the complete
fold training population. Held-out January/July labels do not choose epochs.

The matched seed-42 control scores **295.7301**. Reusing all three historical
tree seeds reproduces v58's recorded **295.7414**; that different seed aggregate
is not the control for this experiment. Individual-part readouts replace just
that part inside the final ensemble. They are not standalone-model scores and
their effects need not add to the all-part change.

## Network diagnostic

Replacing only the networks raises final-ensemble CV from 295.7301 to 297.2381
(+1.5080): January +1.9726, July +1.0434. All three five-network seed groups
also worsen in both months when compared with their corresponding original
groups (mean changes +3.5540, +1.3419 and +1.2265). These group diagnostics use
five-member pools, so their control levels differ from the 15-member headline.
They do not provide independent validation.

This is a result for the fixed v58 network recipe with additional inputs.
Feature redundancy, changed optimization and changed early-stopping choices
are possible explanations, not experimentally separated causes. The result
does not show that every possible use of quantitative weather would fail.

## Data and checks

Source: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
fixed `models=era5`, rather than a changing mixture of weather products. Attribution:
Open-Meteo and Copernicus/ECMWF ERA5; weather data are provided under CC BY 4.0.
The cache contains 30 source files covering ten airports, all of 2025 and the
January/July 2026 prediction months, with preceding days for complete histories.
Source URLs, units, coordinates and SHA-256 hashes are retained in the manifest.

Inputs: hourly temperature, wind speed, precipitation, rain and snowfall, plus
6-, 24- and 48-hour precipitation/snowfall accumulations. Precipitation is in mm,
snowfall in cm and wind in m/s. Each departure uses the latest UTC hourly value
at or before its movement time, with maximum age one hour. Accumulations require
complete preceding windows. ERA5 is retrospective reanalysis, not a forecast
that would have been available at that instant.

The independent audit verifies all 30 source hashes, prepared-input hashes,
training membership and labels, unchanged shared original columns, disjoint
training/held-out IDs, and 14,080 weather values recomputed directly from raw
source arrays. Maximum discrepancy is 0.0000031, from float32 rounding. The
audit's initial calendar check used local DuckDB time; changing that check to
UTC resolved it. Training and weather joins already used UTC and were unaffected.
Four new weather tests and three existing weather tests passed.

## Limits and retained artifacts

These are reused development months, not independent confirmation. Paired
day-bootstrap intervals describe variability on these months; they do not
account for repeated model selection or all training variability. The existing
CatBoost controls were fitted on the laptop GPU; new fits use the desktop GPU,
so device-related numerical variability is not isolated from the feature effect.
No new HPO, blend tuning, laptop jobs or Airport Corner event features are used.

Run: [`v3_20261008_open_meteo_jan_jul`](../runs/v3_20261008_open_meteo_jan_jul/).
Runners: [`open_meteo_jan_jul.py`](../open_meteo_jan_jul.py),
[`open_meteo_all.py`](../open_meteo_all.py); independent audit:
[`audit_open_meteo.py`](../tools/audit_open_meteo.py).
All fitted models, predictions, prepared matrices, source snapshots and checks
are retained locally. Do not repeat completed fits or download the cache again.
The definitive score is `all_result.json`; `all_scored_rows.parquet` retains
the combined candidate's errors. Execution finished successfully. A stale exit
code left by the first audit failure was cleared after completion, and the
driver now clears it when resuming. That metadata fix changed no predictions.

## Authorized full-year submissions

The user explicitly requested both variants on 8 October. They retain v58's
settings and network share, with no extra tuning or special held-flight values.
After v60 was uploaded, the user required asking before any further submission.
v61 fitting and preparation finished at 18:25 Amsterdam. After reviewing its
audit, the user explicitly said "submit v61" and requested a submissions-table
entry. It was uploaded at 20:09 Amsterdam on 8 October, using the original
January/July template with 344,841 rows. Remote bytes match the audited SHA-256.
The approval applies only to that file and has been consumed; further uploads
remain blocked pending fresh approval. SUBMISSIONS.md records it as row 65.
The upload guard was checked locally and blocks before contacting MinIO.
Runner: [`open_meteo_submit.py`](../open_meteo_submit.py); sequential driver:
[`run_open_meteo_submit.py`](../run_open_meteo_submit.py). Run records are in
[`v3_20261008_open_meteo_submit`](../runs/v3_20261008_open_meteo_submit/).

| Version | Weather added to | Jan/Jul CV | Official score | Change vs v58 |
|---|---|---:|---:|---:|
| v60 | Rome classifier | 295.2707 | 237.3022 | +1.1127 |
| v61 | All five parts | 296.8264 | 236.6319 | +0.4424 |

v61 scored **236.6319** on all 344,841 flights, **+0.4424 vs v58** and
−0.6703 vs Rome-only v60. The live check at 20:11 Amsterdam on 8 October reports
rank 26/239; v58 remains best at 236.1895. Both weather variants failed to improve
the official score. No further submission is authorized.

v60 was uploaded at 16:59 Amsterdam time and scored on all 344,841 flights.
Live snapshot: 8 October 2026, 17:00:59 Amsterdam time. Its ordinary regressors
are unchanged. The weather classifier uses 160,577 full-year LIRF training rows;
the all-part ordinary models use 2,051,327 full-year rows. An additional 6,688
weather values were independently checked across full-year and ranking inputs.
v58's formatted submission was reproduced exactly, and uploaded v60 bytes were
checked against the local SHA-256 hash. The initial control-file check compared
floating predictions to integer submission seconds; it was corrected to use the
same template rounding before exact comparison, without repeating any fit.

**Threshold sensitivity:** January flight 202681908 changes from 1,585.29 to
17,816 seconds. Its Rome probability rises from 0.030219 to 0.050346, crossing
the existing 5% rule-confidence threshold. Rule eligibility is unchanged, but
the probability-dependent correction changes. This is the only v60 movement
changing by more than one hour. Its contribution to the observed leaderboard
loss cannot be determined without its hidden label; the aggregate score alone
does not isolate that contribution.

## Full-year data audit after preparation

The user's follow-up audit is reproducible with
[`audit_open_meteo_submit.py`](../tools/audit_open_meteo_submit.py). Its complete
readout is [`full_year_audit.json`](../runs/v3_20261008_open_meteo_submit/full_year_audit.json).
The audit fits no models and contacts no submission endpoint.

- All 156 existing ordinary-regressor columns are unchanged in both training
  and ranking matrices; labels, row indices and ranking IDs also match the
  original full-year inputs. All ten airports and all twelve training months
  are present. No sampling or nearest-flight selection is used.
- The source has 2,085,047 departures and 2,082,750 arrivals. Of the departures,
  388 have nonpositive labels and 127 are excluded by v58's existing rule.
  The remaining population is exactly the union of ordinary and Rome-classifier
  training rows. Ordinary models use 2,051,327 rows. Rome's classifier uses
  160,577 rows, including the 33,205 substituted-label positives omitted from
  ordinary training; its other rows overlap ordinary training. Arrival movements
  remain context for the inherited arrival/traffic features, not taxi-out targets.
- Every weather entry in ordinary training, Rome training and ranking matrices
  was recomputed directly from source arrays using airport and UTC hour, including
  independently accumulated preceding-hour totals: **28,124,195 entries**, zero
  missing values, maximum difference **0.0000031** from float32 rounding. All
  overlapping raw weather observations agree and all 30 source hashes match.
- Saved LightGBM feature names and order match the actual matrices: 167 for
  main/per-airport models, 79 for Rome. CatBoost has the intended 145 columns.
  All include the 11 weather inputs; the main LightGBM splits on all 11 and
  CatBoost assigns all 11 nonzero feature importance. Some airport models and
  Rome do not split on every supplied weather input. Fifteen network checkpoints
  and the intended 167-column configuration are present. The runner refits every
  network on the full year after December chooses 16/18/18 epochs for its three
  groups; December is included in those final fits.
- All five prediction-part hashes and movement IDs pass. Recombining their
  predictions reproduces the saved v61 blend, including the Rome probabilities.
  The file's 344,841 IDs match the original template and its values are finite and
  nonnegative. SHA-256: `a2189a71e70548a88732b933f493f55b748abe85a9cfc8741986c99cd39e3fc8`.

The audit's first pass reached CatBoost importance reporting after passing all
weather/model-column checks, then stopped because the loaded model's cached
importance property was empty. Reading importance through `get_feature_importance`
resolved that reporting issue; the complete audit then passed. No training or
submission predictions changed.

These checks support correct use of the selected weather representation, not
exhaustion of the source's predictive potential. This experiment uses five ERA5
variables and six accumulated totals, not every Open-Meteo variable or IFS. The
[provider documentation](https://open-meteo.com/en/docs/historical-weather-api)
describes ERA5 at approximately 25 km and IFS at 9 km; neither alternative source
nor weather-specific HPO was tested here. Existing METAR inputs remain in the
ordinary models. The reused network encoding, feature choices and fixed v58
settings can still be investigated; the failed CV result does not establish that
every use of this source would fail. No further fits or uploads are authorized
by this audit.
