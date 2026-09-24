# Test-like snow and congestion

Historical rows must meet every fixed similarity limit. Unmatched extreme test rows remain unassessed.

**Amsterdam January:** 2,095 of 2,937 prolonged-snow test rows exceed the entire 2025 snow-duration maximum. 329 have a historical match with held-out predictions. Those matches use 269 historical flights from only 1 day(s).

| Conditions | Test flights | Match in 2025 | Match with predictions | Historical flights / days |
|---|---:|---:|---:|---:|
| Prolonged snow | 10,301 | 2,320 | 2,320 | 2,247 / 20 |
| Long recent arrival taxi times | 347 | 0 | 0 | 0 / 0 |
| Both | 256 | 0 | 0 | 0 / 0 |

| Model | Conditions | Before RMSE | After RMSE | Change | Evidence |
|---|---|---:|---:|---:|---|
| v39 | Prolonged snow | 393.71 | 395.69 | +1.98 | Uncertain |
| v39 | Long recent arrival taxi times | — | — | — | No comparable scored rows |
| v39 | Both | — | — | — | No comparable scored rows |
| v40 | Prolonged snow | 395.24 | 395.17 | -0.07 | Uncertain |
| v40 | Long recent arrival taxi times | — | — | — | No comparable scored rows |
| v40 | Both | — | — | — | No comparable scored rows |

Subset scores are unweighted RMSE in seconds, averaging squared errors over three seeds. Example predictions average the three predictions. January and any extra February–May months are retrospective checks, using models that did not train on those departure rows. The selected conditions overlap and are not a new validation headline.

“Match with predictions” counts test flights with a similar historical flight scored by a model that did not train on it. Test labels are unavailable. Matches require the same airport, missing-ID state and clock-gap band, plus similar snow and arrival histories. No match is forced.

[Overall Weighted CV](OPDI_DISRUPTION_EXPERIMENT.md) · [Input-selected before/after examples](OPDI_DISRUPTION_STRESS_EXAMPLES.md) · [Airport results](opdi_disruption_stress/paired_errors.csv) · [Coverage, uncertainty and limits](opdi_disruption_stress/results.json) · [Fixed matching rules](../docs/OPDI_DISRUPTION_STRESS_CHECK.md)
