# Recording mechanisms in raw training and ranking features

The largest observable anomalies are different recording mechanisms, so one
global clock correction is unsafe. These counts come from
[`audit_recording_patterns.py`](../../tools/audit_recording_patterns.py);
[`raw_clock_audit.json`](raw_clock_audit.json) contains the complete aggregates.
No ranking departure label or block time was read. A ranking feature anomaly
is a hypothesis about its hidden label, not a known prediction error.

The audit covers 2,085,047 training departures and 344,841 ranking departures.
The CV prevalence denominator is 1,079,528 unique July–December departures;
these prevalence counts do not repeat rows across folds or seeds. Every 2025
label exactly equals takeoff minus recorded off-block. The problem is which
clock was recorded as off-block, rather than inconsistent target arithmetic.

| Observable mechanism | Training 2025 | Unique CV | Ranking Jan/Jul | Implication |
|---|---:|---:|---:|---|
| Flight ID; takeoff−LOBT ≥12 h | 1 | 1 | 1 | One 24 h example supports extrapolation; too sparse to establish a general residual distribution. |
| Flight ID; takeoff−AOBT3 ≥2 h; AOBT3 and LOBT within 30 min | 7 | 5 | 2 | Both clocks can agree and still differ substantially from the recorded target. |
| Rome, no flight ID; schedule gap ≥14 h | 20 | 14 | 6 | Training labels split into 11 day shifts and 9 schedule substitutions. |
| EHAM, no flight ID; schedule gap ≥4 h | 91 | 53 | 148 | About 8.7 times the CV prevalence in ranking; inspect ordinary taxi calibration and weather, not a schedule override. |

**The 24 h flight-plan anomaly is already mostly handled by row 33.** The
training example has label 87,002 s, takeoff−LOBT 87,001 s and takeoff−AOBT3
87,181 s. The ranking example has v34 prediction 83,160 s, takeoff−LOBT
86,820 s and takeoff−AOBT3 86,340 s. Moving to a clock centre changes only
3,180–3,660 s now. Per-row prevalence is 3.13 times higher in ranking than
unique CV, but there is only one example in each. For the broader seven-row
training cohort, raw AOBT3 / LOBT / schedule diagnostic RMSE is
2,152.94 / 1,773.80 / 724.67 s. These are clock diagnostics, not model CV scores.

**Rome mixes a previous-date off-block with schedule substitution.** Among
no-ID departures with schedule gap ≥14 h, 11 labels are one local calendar
day plus 770–1,992 s of ordinary taxi, while 9 equal the schedule gap within
60 s. This includes a takeoff after local midnight on the same UTC day.
Eight of fourteen CV cases have the day-shift shape. Five of the six ranking
cases cross a Rome-local date. A correction should model uncertainty between
the two mechanisms; a deterministic next-day correction would damage genuine
schedule substitutions. A previous global day-shift mixture was already
inconclusive in [ROME_SIGNALS.md](../ROME_SIGNALS.md); these counts do not make
that experiment independent new evidence.

Airline is useful context but still sparse: ITY has three day shifts and one
schedule case; QFA and WMT together have four schedule cases and no day shifts.
Flight-specific exceptions would overfit. Any airline-conditioned estimate
must learn only from its fold's training window and shrink small groups.

**The four-hour Rome rule also overpredicts ordinary flights.** Fifteen training
rows meet that rule yet have labels under two hours (all are 723–1,681 s).
They cover different airlines and months, with schedule gaps 4.07–6.40 h.
Eleven are in July–December. The late ordinary component should remain possible
beyond the present four-hour hard override. This is distinct from the next-day
underpredictions above.

**Outside Rome, very late no-ID rows are not schedule substitutions.** None of
413 training rows with gap ≥4 h matches the schedule gap within 60 s. Their
labels are at most 4,667 s outside Istanbul; Istanbul has five labels over two
hours. All matching ranking v34 predictions are already below two hours, so
a broad tail cap offers little obvious remaining headroom. EHAM has the largest
population shift, with six ranking predictions above one hour (maximum 4,689 s).

**Some large labels have no large visible clock gap.** Six training no-ID rows
have labels over four hours despite schedule gaps under two hours: LFPG has two
in January, Rome two in May/August, and Zurich two in February/May. Only the
August Rome row enters July–December CV. Its label is 25,018 s versus a schedule
gap of 1,679 s. Raw schedule delay alone cannot identify these errors; a rule
targeting short gaps globally would harm many normal rows.

**The existing twin rule captures almost all useful duplicates.** The raw
audit finds 110 same-flight clusters within three minutes in training, including
102 two-row clusters with exactly one flight ID. Beyond that rule are only seven
two-ID pairs and one four-row no-ID cluster. Ranking has sixteen eligible LFPG
mixed-ID pairs, one two-ID LFPG pair and two three-row no-ID Zurich clusters.
Repeated FLIGHT_ID values do not uncover additional changed-flight-name pairs.
There is no well-supported new duplicate rule in this audit.

Hour-offset coincidences occur, but no clean universal DST correction emerged.
LOBT stays within the existing 61-minute target window at every airport; AOBT3
has airport-specific outliers beyond that window. Preserve the LOBT guard when
testing actual-clock residual models.

All rules suggested by this full-year label exploration are exploratory when
scored on the same 2025 folds. Retain all validation rows and report sparse-case
influence alongside aggregate scores. No model was fitted and nothing uploaded
by this audit. Row-level examples and the projected audit database stay under
ignored `tmp/recording_artifacts_20260919/`.
