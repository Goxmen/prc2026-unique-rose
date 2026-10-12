# Public reference totals do not resolve taxi-source composition error

The fixed source-only pilot fails its gate. No taxi CV, candidate, production
fit, upload or parameter expansion follows.

| Outer calibration window | Cells | Original source RMSE | Conditional source RMSE | Change; paired-month 95% interval |
|---|---:|---:|---:|---|
| January fold's permitted months | 5,374 | 24.0441 | 24.0319 | −0.0122; [−0.0332, +0.0231] |
| July fold's permitted months | 4,098 | 47.0470 | 47.0736 | +0.0266; [+0.0073, +0.0405] |

These seconds measure error in a public runway/day **mean**, not flight-level
taxi RMSE. Each comparison leaves one permitted calibration month out. Neither
window meets the frozen requirement of a 5% source-error reduction with an
interval below zero. Correlations between the baseline's source error and the
new reference discrepancy are −0.0036 and +0.0449.

The new hypothesis was that public reference-time totals reveal which flights
are absent from the public taxi cohort. The retained public-only inverse model
estimates stand/runway reference means; exact leave-one-source-day-out deletion
provides a reference discrepancy for every supported cell. One new coefficient
per airport conditions the original airport/count-deficit Huber calibration on
that discrepancy. Both arms retain Huber scale 20 and ridge 0.001. The reference
and taxi statistics share a source cohort: the proposal models conditional bias,
and does not count them as independent measurements.

This differs from the closed inverse-reference LightGBM screen, which added
stand estimates as two flight features, and the runway covariance screen, which
pooled source noise by runway. No departure labels outside each outer permitted
training window were used here. The ranking view only supplies public reference
reconstruction checks; it supplies no departure outcomes. Transductive public
reference reconstruction remains distinct from taxi-out validation.

The independent audit passes 9,472 source evaluation rows, all input hashes,
SQL source/label arithmetic, inner/outer membership, coefficient application,
scores and paired-month bootstrap. Direct deletion checks on 196 independently
reconstructed source groups agree within 4.55e−12 seconds. A SQL reserved-alias
parse error occurred after public reference preparation but before departure
outcomes were read. Its original source, frozen design and log are preserved;
the completed public preparation was reused after the operational fix.

Artifacts: `../runs/v3_20261012_reference_population_coherence/` contains the
frozen identities, reference cells, calibration coefficients, held-source
errors, `result.json`, `audit.json` and airport breakdown. Runner:
`../tools/pilot_reference_population_coherence_20261012.py`; separate auditor:
`../tools/audit_reference_population_coherence_20261012.py`.
