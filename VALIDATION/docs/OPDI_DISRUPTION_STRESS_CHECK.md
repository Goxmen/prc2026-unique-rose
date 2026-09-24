# Test-like snow and congestion: follow-up check

Added at the user's request on 24 September, while the frozen v39/v40 fits
were running. No candidate scores had been inspected. This is a descriptive
follow-up on reused data, not independent confirmation or another tuned model.

Identify conditions using observable inputs only. Never select examples by
departure labels, prediction errors or whether the new model helps.

- Prolonged snow: the existing SN/SG fraction over 24 hours is above 0.25.
- Heavy arrival congestion: mean completed-arrival taxi-in is at least 1,800
  seconds in the previous 60 minutes (at least five arrivals), or the previous
  180 minutes (at least ten arrivals). This is an arrival-delay proxy, not a
  measured departure queue. The existing 7,200-second per-arrival cap remains.
- Also report their intersection. Show counts and severity ranges for these
  groups in January/July 2026 and throughout 2025, by airport.

A historical analogue must be at the same airport, have the same missing-ID
state and schedule-gap band (missing, negative, 0–15m, 15–30m, 30–60m,
1–2h, 2–4h, 4–12h, 12h+), and belong to the same stress group. Require known
snow and arrival summaries. No match is allowed above the historical maximum
snow fractions, arrival means or long-arrival shares for that airport/group.
Among remaining cases, require **all** of these limits:

- Six-hour snow fraction differs by at most 1/6; 24-hour by at most 1/24.
- Each 60-/180-minute arrival count plus one is within a factor of 1.5.
- Each arrival mean, floored at 60 seconds only for distance scaling, is
  within a factor of 1.25.
- Each over-30-minute share differs by at most 0.10; each over-60-minute
  share by at most 0.05.
- Also compare the **uncapped** arrival means (same factor of 1.25) and
  over-two-hour shares (difference at most 0.025), with the same no-extrapolation
  maximum check. This prevents the model's two-hour input cap from disguising
  a more extreme target. These are matching diagnostics, not new model inputs.
  For held-out rows, require the full-history and permitted-history arrival
  counts to agree in both windows before using these diagnostics. The latter
  donor set is a subset, so equal counts establish identical donors; otherwise
  reject the row as lacking comparable measurements.

Keep up to ten nearest neighbours within these limits, without widening them
to obtain examples. Allow 1e-6 in normalized distance for float32 rounding at
exact boundaries; this tolerance is fixed before matching. Report unmatched
targets, missing measurements, targets
beyond observed severity and unique historical flights/dates. These limits
describe similarity on the stated inputs, not equivalence on every model input.

Search all 2025 first, then separately measure coverage by rows with paired
held-out predictions. Use A1–A6 for July–December (one origin per flight) and
the separate January retrospective check. If February–May contain analogues,
the existing June–November models can predict those months without refitting;
rebuild their original categories/context and score the full month. June was
in every existing training window: do not substitute in-sample predictions.
Distinguish 'no analogue in 2025' from 'analogue exists but is not honestly
scored'. Ranking departure labels remain unknown.

Compare both models on exactly the same selected historical flights. Report
RMSE before/after, each seed, 1-/3-day paired intervals, unique flights/dates,
and improved/worsened counts. Scores average squared errors over seeds; any
example prediction averages predictions and is labelled accordingly. Show
at most six input-selected examples per group, at most one per airport/day;
include regressions when selected. No cherry-picking by error change.

Call an improvement supported on this historical subset only with at least
100 flights and ten UTC dates, improvement in every seed, and both paired
intervals below zero. Otherwise state uncertainty/sparse evidence. Even a
supported subset gain says nothing about the unmatched extreme test cases.
Do not fold these selected subsets into the adopted Weighted CV or edit the
submission ledger. Run full-data follow-up checks after fitting, to avoid
memory contention on the laptop.
