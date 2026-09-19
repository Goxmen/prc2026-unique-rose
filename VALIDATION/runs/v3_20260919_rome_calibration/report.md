# Rome binary and three-class calibration

Run `v3_20260919_rome_calibration`; matched control `v3_20260919_artifact_control`.

Two frozen local-validation candidates; all original evaluation rows, 16 folds and three seeds. Calibrators use three expanding inner chronological splits within each outer training window. Ordinary predictions are reused. Neither candidate has an official leaderboard score.

| Recipe | R_A | R_B_long | Combined | Combined change vs control, 95% day interval |
|---|---:|---:|---:|---|
| Matched row-34 replay | 253.0698 | 228.9793 | 239.9468 | — |
| Rome-binary-cal | 296.7058 | 296.4860 | 296.5834 | +56.6366 [+19.1478, +97.5609] |
| Rome-three-cal | 294.2137 | 282.8235 | 287.9234 | +47.9767 [+17.9969, +82.8752] |

The separate saved artifact-combined candidate (submission-table row 37) scored 249.9460 / 224.8911 / 236.3151 s. This is a descriptive additional reference; the paired control above is row 34.

## What changed, and whether it helped

Both candidates replace the Rome hard override with a soft mixture. The binary classifier is reused and calibrated; the three-class classifier changes the outcomes and fits new Rome classifiers. The day class is restricted to no-ID Rome rows at >=14h schedule gap crossing one local date, then shrunk by training support. It cannot detect day artifacts outside that scope. The flight-plan window and no-ID cap remain; predictions outside Rome are checked unchanged.

| Comparison | R_A change | R_B_long change | Combined change [95% interval] | Pointwise verdict |
|---|---:|---:|---|---|
| binary_vs_control_block1 | +43.6360 | +67.5068 | +56.6366 [+19.1478, +97.5609] | regression |
| three_vs_control_block1 | +41.1439 | +53.8442 | +47.9767 [+17.9969, +82.8752] | regression |
| binary_vs_binary_uncal_block1 | +7.1032 | +11.5178 | +9.5535 [+4.5618, +14.9432] | regression |
| three_vs_binary_block1 | -2.4921 | -13.6626 | -8.6599 [-16.8143, -0.5422] | improvement |
| three_vs_three_uncal_block1 | +8.8916 | +9.1653 | +9.0395 [+4.2008, +15.0783] | regression |
| binary_cal_hard_vs_control_block1 | -0.2926 | +0.8987 | +0.3419 [-0.0117, +0.6782] | inconclusive |
| binary_uncal_vs_control_block1 | +36.5328 | +55.9890 | +47.0831 [+13.8505, +85.3371] | regression |
| binary_vs_control_block3 | +43.6360 | +67.5068 | +56.6366 [+19.5575, +93.8933] | regression |
| three_vs_control_block3 | +41.1439 | +53.8442 | +47.9767 [+18.6778, +80.2414] | regression |

Negative changes favour the first recipe. `binary_uncal` is the original binary mixture without the hard override; `binary_cal_hard` keeps the original override. These isolate override removal from calibration. `three_uncal` keeps the same day restrictions but omits calibration. One-day and three-day intervals use 2,000 paired bootstrap draws. All comparisons reuse development folds; no result is independent confirmation or automatic promotion.

## Seed and label-slice checks

| Candidate | Readout | Seed 42 | Seed 142 | Seed 242 | Target <=2h RMSE | Target >2h RMSE |
|---|---|---:|---:|---:|---:|---:|
| Rome-binary-cal | R_A | 294.4949 | 296.9078 | 298.6997 | 235.5953 | 10048.0900 |
| Rome-binary-cal | R_B_long | 293.8691 | 298.9867 | 296.5803 | 214.5396 | 15007.5128 |
| Rome-three-cal | R_A | 293.5283 | 296.9061 | 292.1867 | 236.5695 | 9745.2226 |
| Rome-three-cal | R_B_long | 282.5718 | 284.0717 | 281.8223 | 215.0121 | 13474.9620 |

## Probability and failure diagnostics

Losses below average each seed/origin loss with the same flight/month weighting as its score group. Probabilities are not averaged into an ensemble. Brier is the sum across classes; compare it within a classifier, since the binary and three-class outcomes differ. The JSON includes all/present-ID/missing-ID reliability bins for every class.

| Group | Probability stage | ID stratum | Log loss | Multiclass Brier |
|---|---|---|---:|---:|
| R_A | binary_raw | all | 0.340517 | 0.216161 |
| R_A | binary_raw | id_missing | 0.428281 | 0.275492 |
| R_A | binary_cal | all | 0.340568 | 0.215704 |
| R_A | binary_cal | id_missing | 0.417693 | 0.267795 |
| R_A | three_raw | all | 0.399738 | 0.224826 |
| R_A | three_raw | id_missing | 0.746347 | 0.305504 |
| R_A | three_cal_before_day | all | 0.384753 | 0.225307 |
| R_A | three_cal_before_day | id_missing | 0.649349 | 0.317916 |
| R_A | three_cal | all | 0.369035 | 0.222014 |
| R_A | three_cal | id_missing | 0.550446 | 0.292286 |
| R_B_long | binary_raw | all | 0.386564 | 0.250781 |
| R_B_long | binary_raw | id_missing | 0.346789 | 0.219923 |
| R_B_long | binary_cal | all | 0.387086 | 0.250912 |
| R_B_long | binary_cal | id_missing | 0.408603 | 0.270424 |
| R_B_long | three_raw | all | 0.428853 | 0.257044 |
| R_B_long | three_raw | id_missing | 0.840433 | 0.272457 |
| R_B_long | three_cal_before_day | all | 0.420412 | 0.257455 |
| R_B_long | three_cal_before_day | id_missing | 0.795512 | 0.337811 |
| R_B_long | three_cal | all | 0.406836 | 0.254934 |
| R_B_long | three_cal | id_missing | 0.687608 | 0.322297 |

`three_cal_before_day` isolates calibration; `three_cal` includes the deployment gate and shrinkage.

| Group | Cohort | Unique flights | Candidate | RMSE | Mean error | Predicted >2h | Predicted >12h |
|---|---|---:|---|---:|---:|---:|---:|
| R_A | target_at_most_2h | 82577 | control | 462.4396 | +14.0206 | 0.0307% | 0.0000% |
| R_A | target_at_most_2h | 82577 | binary | 438.9148 | +16.3260 | 0.0202% | 0.0000% |
| R_A | target_at_most_2h | 82577 | three | 445.7107 | +23.0003 | 0.0198% | 0.0000% |
| R_A | day_class | 8 | control | 22720.3720 | -17084.8750 | 100.0000% | 100.0000% |
| R_A | day_class | 8 | binary | 53161.3218 | -49438.0502 | 100.0000% | 37.5000% |
| R_A | day_class | 8 | three | 51248.0086 | -48984.5857 | 100.0000% | 33.3333% |
| R_A | day_scope | 13 | control | 17823.3264 | -10513.3846 | 100.0000% | 100.0000% |
| R_A | day_scope | 13 | binary | 43064.1655 | -35070.8122 | 100.0000% | 46.1538% |
| R_A | day_scope | 13 | three | 41820.7364 | -36075.4405 | 100.0000% | 41.0256% |
| R_A | day_class_in_scope | 8 | control | 22720.3720 | -17084.8750 | 100.0000% | 100.0000% |
| R_A | day_class_in_scope | 8 | binary | 53161.3218 | -49438.0502 | 100.0000% | 37.5000% |
| R_A | day_class_in_scope | 8 | three | 51248.0086 | -48984.5857 | 100.0000% | 33.3333% |
| R_B_long | target_at_most_2h | 23346 | control | 334.9168 | +23.8509 | 0.0043% | 0.0000% |
| R_B_long | target_at_most_2h | 23346 | binary | 336.1144 | +26.6664 | 0.0014% | 0.0000% |
| R_B_long | target_at_most_2h | 23346 | three | 340.3289 | +29.9589 | 0.0014% | 0.0000% |
| R_B_long | day_class | 3 | control | 23732.7243 | -19820.6667 | 100.0000% | 100.0000% |
| R_B_long | day_class | 3 | binary | 64799.2907 | -63896.7389 | 100.0000% | 5.5556% |
| R_B_long | day_class | 3 | three | 57397.3565 | -56111.8049 | 100.0000% | 27.7778% |
| R_B_long | day_scope | 3 | control | 23732.7243 | -19820.6667 | 100.0000% | 100.0000% |
| R_B_long | day_scope | 3 | binary | 64799.2907 | -63896.7389 | 100.0000% | 5.5556% |
| R_B_long | day_scope | 3 | three | 57397.3565 | -56111.8049 | 100.0000% | 27.7778% |
| R_B_long | day_class_in_scope | 3 | control | 23732.7243 | -19820.6667 | 100.0000% | 100.0000% |
| R_B_long | day_class_in_scope | 3 | binary | 64799.2907 | -63896.7389 | 100.0000% | 5.5556% |
| R_B_long | day_class_in_scope | 3 | three | 57397.3565 | -56111.8049 | 100.0000% | 27.7778% |

Positive mean error is overprediction. `day_class` uses schedule precedence; `day_scope` uses observable fields only. Counts are unique flights within each group; groups overlap and must not be added. All rare day cases/scope rows are in [day_cases.csv](day_cases.csv); ordinary flights predicted above two hours by control/either candidate are in [ordinary_false_high.csv](ordinary_false_high.csv), retaining seed/origin identifiers. These are diagnostic slices; headline scores keep every row.

## Training support and workload

| Window | Ordinary/other | Schedule | Day | Day examples in deployment scope |
|---|---:|---:|---:|---:|
| v3_2025-06 | 61084 | 16741 | 4 | 3 |
| v3_2025-07 | 64974 | 16742 | 7 | 6 |
| v3_2025-08 | 69237 | 17235 | 7 | 6 |
| v3_2025-09 | 71656 | 16939 | 7 | 6 |
| v3_2025-10 | 71735 | 16974 | 7 | 6 |
| v3_2025-11 | 68934 | 16560 | 9 | 8 |

Completed 126 new classifiers, reused 18 outer binary classifiers and all ordinary predictions. Saved inner predictions and calibrators, 48 outer prediction files, nine paired comparisons and artifact/source hashes. No full-year fit or upload.

The design is in [ROME_CALIBRATION.md](../../docs/ROME_CALIBRATION.md). Class outcomes use training labels only; deployment accepts observable inputs. The day class is sparse and overlaps across training windows; its counts are not independent evidence.

```bash
.venv/bin/python VALIDATION/rome_calibration_run.py --output-dir VALIDATION/runs/v3_20260919_rome_calibration --resume
```
