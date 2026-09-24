# New features on v39/v40

Same 12 snow-duration and completed-arrival inputs as the v38 test. Lower is better.

| Local comparison | Control | New features | Change |
|---|---:|---:|---:|
| v39 Weighted CV | 264.4169 | 263.6462 | -0.7708 |
| v39 Original CV | 233.7456 | 233.1510 | -0.5946 |
| v39 January check | 346.5944 | 346.7118 | +0.1174 |
| v40 Weighted CV | 264.3388 | 263.5929 | -0.7459 |
| v40 Original CV | 233.2322 | 232.6647 | -0.5675 |
| v40 January check | 344.8053 | 344.8786 | +0.0733 |

Controls were rebuilt with the same local OPDI cache as their candidates. Historical cache bytes differ, so gains are measured against these matched controls.

- **v39:** historical Weighted CV 264.4218; rebuilt 264.4169. Change intervals: 1-day [-1.379, -0.294]; 3-day [-1.283, -0.325] s. Seed changes: 42: -1.142, 142: -0.525, 242: -0.647 s.
- **v40:** historical Weighted CV 264.3397; rebuilt 264.3388. Change intervals: 1-day [-1.332, -0.274]; 3-day [-1.256, -0.320] s. Seed changes: 42: -0.899, 142: -0.586, 242: -0.753 s.

All 16 folds, adopted weights, rules and recording corrections remain. Only ordinary regressors receive the new features; each pair shares its classifier. v40 keeps its training exclusions. January uses June–November models and is outside Weighted CV.

These reused historical conditions do not establish performance during January 2026’s longer snow. The package was tested together, so gains cannot be assigned to snow versus arrivals.

[Design](../docs/OPDI_DISRUPTION_EXPERIMENT.md) · [Detailed results](opdi_disruption_experiment/)

The comparison used no full-year fits or uploads. The user later selected the v40
candidate for submission: **v42 scored 274.9076 s** ([submission report](V42_SUBMISSION.md)).
