# Overnight CV scores before new leaderboard outcomes

Full-data 800-round validation; three seeds.

All sixteen chronological folds are retained. Seeds and repeated training origins average squared errors, not predictions. Seasonal holdouts are retrospective. The fixed blends combine month-balanced and eleven-month seasonal MSE; blend number is the percentage assigned to month-balanced MSE.

| Recipe | v3 combined | A | B long | Month balanced | Unique flight | 10m | 11m |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 274.3622 | 296.2717 | 255.6100 | 297.0224 | 299.0417 | 379.0118 | 377.6050 |
| no_month | 272.5082 | 296.4686 | 251.8408 | 296.6403 | 298.7169 | 380.4677 | 378.0763 |
| no_history | 274.6319 | 296.5108 | 255.9093 | 297.2956 | 299.3355 | 378.4996 | 376.1919 |
| clock_features | 273.1284 | 293.6614 | 255.6326 | 294.7321 | 296.6656 | 374.3450 | 370.7952 |
| small_trees | 274.5645 | 296.6377 | 255.6624 | 297.1049 | 299.0829 | 380.2407 | 379.0468 |
| rule_routing | 267.5759 | 286.6291 | 251.4024 | 287.4159 | 289.1555 | 368.1369 | 368.3539 |

The JSON retains every fold, monthly and per-seed score, artifact hash, and 2,000-draw paired interval (or the explicitly requested repetition count). No new official score is read by this command.
