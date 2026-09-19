# Where the recording-artifact model fails

Ranking targets are hidden. The errors below are measured on 2025 validation flights; ranking counts describe observable patterns only.

Each flight has equal weight within a score group. Squared errors are averaged across seeds and repeated training origins before scoring. R_A uses July–December; R_B_long uses November–December, averaging its two December origins. All sixteen folds and every target remain in the report.

![Error concentration](error_concentration.png)

| Candidate | R_A | R_B_long | R_combined | Combined Δ [95% CI] |
| --- | --- | --- | --- | --- |
| control | 253.07 | 228.98 | 239.95 | — |
| twins | 252.75 | 228.59 | 239.59 | -0.35 [-0.63, -0.11] |
| clock12 | 253.05 | 228.98 | 239.94 | -0.01 [-0.04, +0.00] |
| clock2 | 253.00 | 228.98 | 239.91 | -0.03 [-0.09, +0.01] |
| day_shrunk | 250.29 | 225.28 | 236.69 | -3.26 [-7.49, -0.06] |
| day_handler | 250.29 | 225.28 | 236.69 | -3.26 [-7.49, -0.06] |
| rule_confidence | 253.07 | 228.98 | 239.95 | +0.00 [+0.00, +0.00] |
| combined | 249.95 | 224.89 | 236.32 | -3.63 [-7.83, -0.44] |
| rule_shrunk | 252.68 | 229.09 | 239.82 | -0.12 [-0.36, +0.10] |
| combined_midgap | 249.55 | 225.00 | 236.19 | -3.76 [-7.92, -0.56] |

These are reused development folds, not untouched confirmation. Combined CV is a provisional summary, not a leaderboard forecast.

![Candidate changes](candidate_changes.png)

| Group | Rows | ≤2h RMSE | >2h rows | Tail error % | Worst 20 % | Worst 100 % | Worst 1000 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R_A | 1079528 | 239.01 | 348 | 10.8 | 11.3 | 15.7 | 27.5 |
| R_B_long | 328009 | 214.41 | 61 | 12.3 | 12.8 | 17.7 | 30.8 |

| Regime | A rows | A error % | B rows | B error % |
| --- | --- | --- | --- | --- |
| Rome next-day hard rule | 13 | 6.0 | 3 | 9.8 |
| Rome other hard rule | 73 | 4.3 | 7 | 0.0 |
| Rome no-ID 1–4h | 811 | 7.1 | 113 | 3.3 |
| Rome other no-ID | 60 | 1.1 | 17 | 0.2 |
| Other airport no-ID | 12660 | 4.8 | 2852 | 4.9 |
| ID twin | 53 | 0.4 | 18 | 0.5 |
| Other ID | 1065858 | 76.3 | 324999 | 81.2 |

Regimes are disjoint and use observable clocks and the raw flight-ID status. The next-day regime means exactly one Europe/Rome date crossed and schedule gap ≥14 h; it does not assume the hidden label is a day shift.

**Largest 20 unique control errors: R_A.** Mean prediction is descriptive; RMS error averages seed/origin squared errors.

| ID | Flight | Airport | Date | Target | Prediction | RMS error | Schedule gap | Regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 196129531 | ITYTY630 | LIRF | 2025-09-28 | 87543 | 50581 | 36962 | 50581 | Rome next-day hard rule |
| 200297323 | ITYTY680 | LIRF | 2025-11-04 | 87480 | 56520 | 30960 | 56520 | Rome next-day hard rule |
| 192628959 | ITYTY680 | LIRF | 2025-07-30 | 87186 | 58563 | 28623 | 58563 | Rome next-day hard rule |
| 200300302 | BAWW561D | LIRF | 2025-11-12 | 87598 | 60599 | 26999 | 60599 | Rome next-day hard rule |
| 195015201 | NOS430 | LIRF | 2025-08-28 | 1078 | 23039 | 21961 | 23039 | Rome other hard rule |
| 195000946 | WMT8PV | LIRF | 2025-08-02 | 25018 | 3573 | 21448 | 1679 | Rome other no-ID |
| 192621312 | PGT2WE | LIRF | 2025-07-28 | 1377 | 20878 | 19501 | 20878 | Rome other hard rule |
| 195011313 | CYF131 | LIRF | 2025-08-17 | 1143 | 19141 | 17998 | 19141 | Rome other hard rule |
| 195010959 | WMT8EJ | LIRF | 2025-08-21 | 1322 | 19261 | 17939 | 19261 | Rome other hard rule |
| 196126418 | BBC356 | LIRF | 2025-09-21 | 1681 | 17820 | 16139 | 17820 | Rome other hard rule |
| 192628379 | RYR68SF | LIRF | 2025-07-28 | 723 | 15301 | 14578 | 15301 | Rome other hard rule |
| 192626949 | NBT40H | LIRF | 2025-07-27 | 842 | 15361 | 14519 | 15361 | Rome other hard rule |
| 198944225 | WZZ3KD | LIRF | 2025-10-23 | 1190 | 15595 | 14405 | 15595 | Rome other hard rule |
| 196123310 | WZZ6KG | LIRF | 2025-09-10 | 1027 | 15183 | 14156 | 15183 | Rome other hard rule |
| 192615553 | VOE6BB | LIRF | 2025-07-03 | 87361 | 73260 | 14101 | 73260 | Rome next-day hard rule |
| 192624005 | BAW551G | LIRF | 2025-07-19 | 1077 | 15178 | 14101 | 15178 | Rome other hard rule |
| 196130204 | DAH2025 | LIRF | 2025-09-18 | 1145 | 14702 | 13557 | 14702 | Rome other hard rule |
| 196125965 | UALAL127 | LIRF | 2025-09-21 | 12774 | 1751 | 11023 | 12777 | Rome no-ID 1–4h |
| 191483190 | SHT14L | EGLL | 2025-07-30 | 11167 | 2076 | 9091 | 10563 | Other airport no-ID |
| 201194517 | DLHLH572 | LIRF | 2025-12-21 | 10264 | 1768 | 8497 | 10262 | Rome no-ID 1–4h |

**Largest 20 unique control errors: R_B_long.** Mean prediction is descriptive; RMS error averages seed/origin squared errors.

| ID | Flight | Airport | Date | Target | Prediction | RMS error | Schedule gap | Regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200297323 | ITYTY680 | LIRF | 2025-11-04 | 87480 | 56520 | 30960 | 56520 | Rome next-day hard rule |
| 200300302 | BAWW561D | LIRF | 2025-11-12 | 87598 | 60599 | 26999 | 60599 | Rome next-day hard rule |
| 201099330 | CES570 | LFPG | 2025-12-26 | 9182 | 1045 | 8140 | 10741 | Other airport no-ID |
| 201194517 | DLHLH572 | LIRF | 2025-12-21 | 10264 | 2149 | 8121 | 10262 | Rome no-ID 1–4h |
| 201193674 | WZZZ8068 | LIRF | 2025-12-17 | 782 | 8454 | 7718 | 13561 | Rome no-ID 1–4h |
| 199350134 | AF1019 | EDDF | 2025-11-03 | 6907 | 1377 | 5530 | 6878 | Other airport no-ID |
| 200298723 | AEE5RX | LIRF | 2025-11-07 | 6786 | 1504 | 5301 | 3543 | Rome other no-ID |
| 201103619 | AFR85XG | LFPG | 2025-12-05 | 6600 | 1538 | 5063 | 11100 | Other ID |
| 201113278 | DAL153 | LFPG | 2025-12-05 | 7392 | 2433 | 4965 | 7566 | Other airport no-ID |
| 199519773 | UPS22 | LFPG | 2025-11-22 | 7863 | 3143 | 4724 | 11281 | Other airport no-ID |
| 201104714 | AFR22UR | LFPG | 2025-12-24 | 7805 | 3201 | 4608 | 7865 | ID twin |
| 201194087 | RYR6522 | LIRF | 2025-12-18 | 10497 | 6051 | 4605 | 10498 | Rome no-ID 1–4h |
| 201196521 | RYR2496 | LIRF | 2025-12-22 | 7629 | 3151 | 4492 | 7624 | Rome no-ID 1–4h |
| 200298668 | RYR9025 | LIRF | 2025-11-07 | 1077 | 5478 | 4412 | 8278 | Rome no-ID 1–4h |
| 201194106 | WMT3251 | LIRF | 2025-12-18 | 10382 | 6069 | 4405 | 10381 | Rome no-ID 1–4h |
| 201194712 | AEA19SN | LIRF | 2025-12-28 | 5351 | 1051 | 4300 | 5345 | Other ID |
| 199513736 | AFR758 | LFPG | 2025-11-13 | 6512 | 2238 | 4274 | 6818 | ID twin |
| 200297147 | MSC2828 | LIRF | 2025-11-03 | 2218 | 6479 | 4261 | 9419 | Other ID |
| 199522384 | AFR406 | LFPG | 2025-11-22 | 6355 | 2139 | 4217 | 6539 | Other ID |
| 201195803 | MSC2818 | LIRF | 2025-12-29 | 1073 | 5249 | 4183 | 5816 | Other ID |

**Largest fixes and harms.** Counts refer to unique flights whose seed/origin-averaged loss improves or worsens. Examples show control RMS error → candidate RMS error in seconds.

| Candidate | Group | Helped | Harmed | Largest fix | Largest harm |
| --- | --- | --- | --- | --- | --- |
| twins | R_A | 44 | 9 | AFR758 (LFPG): 4856 → 2364 | AFR67FZ (LFPG): 175 → 3155 |
| twins | R_B_long | 13 | 5 | AFR22UR (LFPG): 4608 → 2370 | AFR67FZ (LFPG): 175 → 3155 |
| clock12 | R_A | 1 | 0 | AEZ146R (LIRF): 3661 → 29 | — |
| clock12 | R_B_long | 0 | 0 | — | — |
| clock2 | R_A | 3 | 2 | OR1507 (EHAM): 5756 → 3282 | RYR8HC (LIRF): 659 → 3581 |
| clock2 | R_B_long | 0 | 0 | — | — |
| day_shrunk | R_A | 8 | 5 | ITYTY630 (LIRF): 36962 → 26608 | QFAQFA6 (LIRF): 5 → 9558 |
| day_shrunk | R_B_long | 3 | 0 | ITYTY680 (LIRF): 30960 → 25407 | — |
| day_handler | R_A | 8 | 5 | ITYTY630 (LIRF): 36962 → 26608 | QFAQFA6 (LIRF): 5 → 9558 |
| day_handler | R_B_long | 3 | 0 | ITYTY680 (LIRF): 30960 → 25407 | — |
| rule_confidence | R_A | 0 | 0 | — | — |
| rule_confidence | R_B_long | 0 | 0 | — | — |
| combined | R_A | 53 | 14 | ITYTY630 (LIRF): 36962 → 26608 | QFAQFA6 (LIRF): 5 → 9558 |
| combined | R_B_long | 16 | 5 | ITYTY680 (LIRF): 30960 → 25407 | AFR67FZ (LFPG): 175 → 3155 |
| rule_shrunk | R_A | 11 | 53 | NOS430 (LIRF): 21961 → 20434 | WZZZ2062 (LIRF): 0 → 3779 |
| rule_shrunk | R_B_long | 0 | 6 | — | AAL239 (LIRF): 0 → 2235 |
| combined_midgap | R_A | 64 | 67 | ITYTY630 (LIRF): 36962 → 26608 | QFAQFA6 (LIRF): 5 → 9558 |
| combined_midgap | R_B_long | 16 | 11 | ITYTY680 (LIRF): 30960 → 25407 | AFR67FZ (LFPG): 175 → 3155 |

**Influence of the most affected UTC day.** This removes that date only for an influence calculation. It does not change validation scores or select easier folds.

| Candidate | Group | Day | Full Δ | Without-day Δ |
| --- | --- | --- | --- | --- |
| twins | R_A | 2025-11-13 | -0.32 | -0.29 |
| twins | R_B_long | 2025-12-24 | -0.38 | -0.29 |
| twins | R_combined | 2025-12-24 | -0.35 | -0.30 |
| clock12 | R_A | 2025-07-15 | -0.02 | 0.00 |
| clock12 | R_B_long | — | 0.00 | 0.00 |
| clock12 | R_combined | 2025-07-15 | -0.01 | 0.00 |
| clock2 | R_A | 2025-07-20 | -0.07 | -0.03 |
| clock2 | R_B_long | — | 0.00 | 0.00 |
| clock2 | R_combined | 2025-07-20 | -0.03 | -0.02 |
| day_shrunk | R_A | 2025-09-28 | -2.78 | -1.59 |
| day_shrunk | R_B_long | 2025-11-04 | -3.70 | -1.66 |
| day_shrunk | R_combined | 2025-11-04 | -3.26 | -1.79 |
| day_handler | R_A | 2025-09-28 | -2.78 | -1.59 |
| day_handler | R_B_long | 2025-11-04 | -3.70 | -1.66 |
| day_handler | R_combined | 2025-11-04 | -3.26 | -1.79 |
| rule_confidence | R_A | — | 0.00 | 0.00 |
| rule_confidence | R_B_long | — | 0.00 | 0.00 |
| rule_confidence | R_combined | — | 0.00 | 0.00 |
| combined | R_A | 2025-09-28 | -3.12 | -1.94 |
| combined | R_B_long | 2025-11-04 | -4.09 | -2.06 |
| combined | R_combined | 2025-11-04 | -3.63 | -2.17 |
| rule_shrunk | R_A | 2025-08-28 | -0.39 | -0.27 |
| rule_shrunk | R_B_long | 2025-12-18 | 0.11 | 0.07 |
| rule_shrunk | R_combined | 2025-08-28 | -0.12 | -0.07 |
| combined_midgap | R_A | 2025-09-28 | -3.52 | -2.33 |
| combined_midgap | R_B_long | 2025-11-04 | -3.98 | -1.95 |
| combined_midgap | R_combined | 2025-11-04 | -3.76 | -2.29 |

Largest fixes and harms for each candidate, all seed scores, all sixteen folds and regime/target slices are retained in `error_diagnostics.json`.

| Fold | Rows | Control | Combined candidate |
| --- | --- | --- | --- |
| A1 | 190713 | 306.85 | 303.51 |
| A2 | 191182 | 251.60 | 252.11 |
| A3 | 183950 | 251.66 | 246.63 |
| A4 | 185674 | 228.34 | 228.00 |
| A5 | 162332 | 245.21 | 234.33 |
| A6 | 165677 | 219.57 | 219.29 |
| B1 | 183950 | 261.04 | 257.04 |
| B2 | 185674 | 232.98 | 232.61 |
| B3 | 162332 | 235.00 | 227.37 |
| B4 | 165677 | 221.58 | 221.01 |
| B5 | 185674 | 230.55 | 230.21 |
| B6 | 162332 | 235.73 | 225.17 |
| B7 | 165677 | 224.25 | 223.85 |
| B8 | 162332 | 239.31 | 228.22 |
| B9 | 165677 | 229.11 | 228.72 |
| B10 | 165677 | 237.19 | 236.87 |

The raw training/ranking prevalence audit is linked in [RAW_PATTERN_FINDINGS.md](RAW_PATTERN_FINDINGS.md). No submission was created or uploaded by this report.
