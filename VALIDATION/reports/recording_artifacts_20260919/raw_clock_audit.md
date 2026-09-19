# Raw recording-pattern audit

Raw-label and observable-feature audit, not model validation. Ranking labels and block times are explicitly NULL.

CV prevalence counts each July–December 2025 departure once, rather than once per fold and seed. The full training-year audit includes labels later used for validation, so discovered rules need explicit exploratory status.

| population | n | labels | outside_training_filter | max_label_vs_block_gap_seconds |
| --- | --- | --- | --- | --- |
| cv_unique_jul_dec | 1079528 | 1079528 | 159 | 0.000 |
| ranking_jan | 152719 | 0 | 0 | — |
| ranking_jan_jul | 344841 | 0 | 0 | — |
| ranking_jul | 192122 | 0 | 0 | — |
| train_2025 | 2085047 | 2085047 | 388 | 0.000 |
| train_jan | 153706 | 153706 | 48 | 0.000 |
| train_jul | 190713 | 190713 | 35 | 0.000 |

| pattern | population | n | per_100k | median_label_minus_lobt | lobt_rmse | aobt3_rmse | schedule_matches | day_shift_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_departures | cv_unique_jul_dec | 1079528 | 100000.000 | -304.000 | 789.642 | 400.493 | 96732 | 9 |
| all_departures | ranking_jan_jul | 344841 | 100000.000 | — | — | — | 0 | 0 |
| all_departures | train_2025 | 2085047 | 100000.000 | -251.000 | 740.131 | 384.851 | 199634 | 13 |
| missing_flight_id | ranking_jan_jul | 5283 | 1532.010 | — | — | — | 0 | 0 |
| missing_flight_id | cv_unique_jul_dec | 13617 | 1261.385 | — | — | — | 771 | 8 |
| missing_flight_id | train_2025 | 22424 | 1075.467 | — | — | — | 1430 | 12 |
| id_lobt_ge_6h | train_2025 | 1 | 0.048 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_6h | ranking_jan_jul | 1 | 0.290 | — | — | — | 0 | 0 |
| id_lobt_ge_6h | cv_unique_jul_dec | 1 | 0.093 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_12h | train_2025 | 1 | 0.048 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_12h | ranking_jan_jul | 1 | 0.290 | — | — | — | 0 | 0 |
| id_lobt_ge_12h | cv_unique_jul_dec | 1 | 0.093 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_20h | ranking_jan_jul | 1 | 0.290 | — | — | — | 0 | 0 |
| id_lobt_ge_20h | cv_unique_jul_dec | 1 | 0.093 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_20h | train_2025 | 1 | 0.048 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_12h_agree_300s | train_2025 | 1 | 0.048 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_12h_agree_300s | cv_unique_jul_dec | 1 | 0.093 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_6h_agree_300s | cv_unique_jul_dec | 1 | 0.093 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_lobt_ge_6h_agree_300s | train_2025 | 1 | 0.048 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_aobt_ge_2h_agree_30min | cv_unique_jul_dec | 5 | 0.463 | 537.000 | 1676.521 | 2142.633 | 3 | 1 |
| id_aobt_ge_2h_agree_30min | ranking_jan_jul | 2 | 0.580 | — | — | — | 0 | 0 |
| id_aobt_ge_2h_agree_30min | train_2025 | 7 | 0.336 | 1.000 | 1773.803 | 2152.940 | 4 | 1 |
| id_aobt_ge_6h_agree_30min | ranking_jan_jul | 1 | 0.290 | — | — | — | 0 | 0 |
| id_aobt_ge_6h_agree_30min | train_2025 | 1 | 0.048 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_aobt_ge_6h_agree_30min | cv_unique_jul_dec | 1 | 0.093 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_aobt_ge_12h_agree_30min | ranking_jan_jul | 1 | 0.290 | — | — | — | 0 | 0 |
| id_aobt_ge_12h_agree_30min | cv_unique_jul_dec | 1 | 0.093 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| id_aobt_ge_12h_agree_30min | train_2025 | 1 | 0.048 | 1.000 | 1.000 | 179.000 | 1 | 1 |
| rome_noid_ge_4h | cv_unique_jul_dec | 86 | 7.966 | — | — | — | 67 | 8 |
| rome_noid_ge_4h | train_2025 | 127 | 6.091 | — | — | — | 101 | 11 |
| rome_noid_ge_4h | ranking_jan_jul | 43 | 12.470 | — | — | — | 0 | 0 |
| rome_noid_ge_14h | ranking_jan_jul | 6 | 1.740 | — | — | — | 0 | 0 |
| rome_noid_ge_14h | train_2025 | 20 | 0.959 | — | — | — | 9 | 11 |
| rome_noid_ge_14h | cv_unique_jul_dec | 14 | 1.297 | — | — | — | 6 | 8 |
| rome_noid_ge_24h | train_2025 | 2 | 0.096 | — | — | — | 1 | 1 |
| rome_noid_ge_24h | cv_unique_jul_dec | 1 | 0.093 | — | — | — | 0 | 1 |
| rome_noid_ge_24h | ranking_jan_jul | 2 | 0.580 | — | — | — | 0 | 0 |
| nonrome_noid_1to4h | cv_unique_jul_dec | 8935 | 827.677 | — | — | — | 3 | 0 |
| nonrome_noid_1to4h | ranking_jan_jul | 3437 | 996.691 | — | — | — | 0 | 0 |
| nonrome_noid_1to4h | train_2025 | 13916 | 667.419 | — | — | — | 17 | 0 |
| nonrome_noid_ge_4h | ranking_jan_jul | 279 | 80.907 | — | — | — | 0 | 0 |
| nonrome_noid_ge_4h | cv_unique_jul_dec | 241 | 22.325 | — | — | — | 0 | 0 |
| nonrome_noid_ge_4h | train_2025 | 413 | 19.808 | — | — | — | 0 | 0 |
| nonrome_noid_ge_6h | cv_unique_jul_dec | 87 | 8.059 | — | — | — | 0 | 0 |
| nonrome_noid_ge_6h | train_2025 | 144 | 6.906 | — | — | — | 0 | 0 |
| nonrome_noid_ge_6h | ranking_jan_jul | 70 | 20.299 | — | — | — | 0 | 0 |
| nonrome_noid_ge_12h | cv_unique_jul_dec | 27 | 2.501 | — | — | — | 0 | 0 |
| nonrome_noid_ge_12h | ranking_jan_jul | 13 | 3.770 | — | — | — | 0 | 0 |
| nonrome_noid_ge_12h | train_2025 | 46 | 2.206 | — | — | — | 0 | 0 |
| nonrome_noid_ge_24h | cv_unique_jul_dec | 13 | 1.204 | — | — | — | 0 | 0 |
| nonrome_noid_ge_24h | train_2025 | 16 | 0.767 | — | — | — | 0 | 0 |
| nonrome_noid_ge_24h | ranking_jan_jul | 1 | 0.290 | — | — | — | 0 | 0 |
| missing_id_missing_aircraft | ranking_jan_jul | 388 | 112.516 | — | — | — | 0 | 0 |
| missing_id_missing_aircraft | cv_unique_jul_dec | 969 | 89.761 | — | — | — | 426 | 8 |
| missing_id_missing_aircraft | train_2025 | 1507 | 72.277 | — | — | — | 722 | 12 |
| negative_aobt_gap | ranking_jan_jul | 169 | 49.008 | — | — | — | 0 | 0 |
| negative_aobt_gap | cv_unique_jul_dec | 645 | 59.748 | 597.000 | 1443.015 | 4055.180 | 34 | 0 |
| negative_aobt_gap | train_2025 | 1012 | 48.536 | 621.000 | 1473.090 | 3903.949 | 57 | 0 |
| negative_lobt_gap | ranking_jan_jul | 2458 | 712.792 | — | — | — | 0 | 0 |
| negative_lobt_gap | cv_unique_jul_dec | 7405 | 685.948 | 822.000 | 1125.249 | 943.160 | 111 | 0 |
| negative_lobt_gap | train_2025 | 17133 | 821.708 | 818.000 | 1095.652 | 792.330 | 252 | 0 |
| negative_sched_gap | ranking_jan_jul | 2322 | 673.354 | — | — | — | 0 | 0 |
| negative_sched_gap | cv_unique_jul_dec | 7124 | 659.918 | 660.000 | 839.013 | 444.900 | 16 | 0 |
| negative_sched_gap | train_2025 | 17011 | 815.857 | 666.000 | 841.711 | 437.359 | 48 | 0 |

All clock errors above are single-clock diagnostic RMSE, not experiment validation scores. Full airport, airline, hour/day-offset and duplicate summaries are in `raw_clock_audit.json`. Row examples stay in ignored `tmp/recording_artifacts_20260919/`.

Reproduce: `.venv/bin/python VALIDATION/tools/audit_recording_patterns.py`. Resource cap: four DuckDB threads, 2 GB memory; no fits, uploads or hidden ranking labels.
