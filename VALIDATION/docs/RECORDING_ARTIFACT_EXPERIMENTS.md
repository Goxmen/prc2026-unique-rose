# Recording-artifact experiments — 2026-09-19

The user requested an audit of the largest training/validation errors and suspicious
ranking records, followed by local experiments. **No new submission or full-year
refit is authorized or performed.** Ranking departure labels remain hidden.

## Control and evidence

Start at row 34 (row 30 stand model + plan window + no-ID cap). The Mac runs' boosters
and OOF files are absent on this laptop. `artifact_control_run.py` reconstructs only
the selected row-30 components: six chronological windows, three seeds, 800 ordinary
rounds, 300 classifier rounds, all 16 v3 folds. It then applies rows 33/34 unchanged.
The METAR download matches the historical SHA-256 exactly. Historical results stay
unchanged; all new comparisons use this matched local control. Hardware: i7-13650HX,
one fit at a time, six training threads and 20 prediction threads.

These are development-set experiments. Prior error reports and observable 2025/2026
patterns inform their design. Fitted offsets/weights use only the actual training
window, never OOF labels. Every evaluation row stays in the score. None of v3's
validation months is January; a rare test-set pattern is not a known test error.

## Recipes fixed before examining the new OOF results

All changes are independently applied to row 34. Units below are seconds.

| Candidate | Exact change | Why |
|---|---|---|
| `twins` | Add the existing row-35 twin fix to row 34, offset 330 s. | The two fixes were previously tested separately. The inherited offset was selected on all 2025; retain that optimism caveat. |
| `clock12` | ID present; LOBT and AOBT3 both at least 12 h before takeoff and within 30 min of each other. Predict takeoff − AOBT3 + fitted offset, inside the existing LOBT ±61 min window. | Trees cannot extrapolate an almost day-long gap; clipping only reaches a window edge. |
| `clock2` | Same correction, with the fixed 2 h gate. | Test whether the clock anchor also helps less extreme clock-consistent rows. |
| `day_shrunk` | Rome, no ID, gap ≥14 h, exactly one Rome-local date crossed: `gap + w × (86400 + late_mean − gap)`. | Learn the squared-error tradeoff between schedule substitution and a date-shifted record. |
| `day_handler` | Same, with separate low/high arrival schedule-share weights at 0.5; require 20 arrival observations and five distinct qualifying training dates **in both groups**. Otherwise use pooled weight. | Arrival logging may distinguish schedule substitution from a date shift. |
| `rule_confidence` | Rome no-ID gap in [4 h,12 h), classifier p <0.05: replace the hard rule with `p×gap+(1−p)×late_mean` (late term capped as in row 29). | Test a high-confidence exception for normal taxis incorrectly given hours of delay. |
| `combined` | Twin fix + handler day blend + confidence exception + 12 h clock correction, in that order. | Test interactions without choosing the best parts after seeing results. |

Clock calibration uses unique retained training rows with both plan-clock gaps ≥2 h
and agreement ≤30 min. Fit `y − (takeoff − AOBT3)`, clipped to ±3660 s; shrink the
pooled mean toward zero by 25 pseudo-rows and airport means toward it by 25. The
final prediction is projected into the established plan window. Schedule is never
a fallback clock.

Day calibration solves a one-parameter constrained least-squares blend, `0≤w≤1`.
With step `d=86400+late_mean−gap`, use
`w=clip(sum(d*(y−gap))/(sum(d²)+5*median(d²)),0,1)`.
Handler-specific weights use the same penalty scale and shrink toward pooled w.
Training-support amendment, before any candidate OOF scoring: the first window
had six eligible rows on five dates, all in the low-share group. Require five
dates in **both** groups, so a single populated group does not merely fit the same
data twice and weaken the shrinkage. Identical pooled/handler results mean there
was insufficient support to test handler stratification, not evidence against it.
No eligible training examples means w=0, so the current rule is retained. The late
taxi mean is the same training-window estimate already used by row 30.

The high-confidence gate is a fixed exploratory hypothesis, not a recalibrated
probability guarantee. Earlier broad softening of the Rome rule failed. Record
unchanged, unsuccessful and sparse candidates alongside any improvements.

### Mid-gap calibration amendment (before candidate scoring)

The raw audit found 15 training flights with ordinary taxi labels that satisfy
the four-hour rule. A real-data execution check on A1 found **zero** flights
passing the proposed p<0.05 exception. No candidate scores were inspected.
Add two separately identified candidates; preserve all seven original recipes:

- `rule_shrunk`: Rome no-ID flights with gap in [4 h,12 h). Fit
  `gap + w*(late_mean-gap)` directly by training-window squared error, using
  the same five-pseudo-observation penalty and w in [0,1]. This shrinks toward
  the existing hard rule. The late term uses row 29's existing cap. It does
  not use p and does not change the day-shift population.
- `combined_midgap`: the original combination, with `rule_shrunk` replacing
  the high-confidence exception in that band.

This is a narrow re-test of rule softening, not independent confirmation of
an entirely new hypothesis. All fitting remains within the training window.
The raw target inspection and A1 feature-coverage check are part of model
development and must remain visible in the results.

## Evaluation and test-set hypotheses

Report R_A, R_B_long and provisional combined RMSE; candidate-minus-control changes;
2,000 paired UTC-day bootstrap intervals; seed checks; airports; ordinary/tail error;
changed-row counts; and rare-flight/day influence. Do not treat repeated appearances
of one flight across folds/seeds as independent examples. Use a three-day block
sensitivity check when a close decision depends on disruption clusters.

Training and ranking prevalence are measured on unique movements. Standard CV does
not observe January. Mark any proposed larger leaderboard benefit explicitly as a
hypothesis, give the affected population/count and explain its training scarcity.
Do not turn a hidden target into an assumed measurement or call combined CV a
leaderboard forecast. Optional ranking corrections operate on the already-scored
v34 file only; they do not create or upload a new submission.

Source: `recording_artifacts.py`; raw audit: `tools/audit_recording_patterns.py`.
