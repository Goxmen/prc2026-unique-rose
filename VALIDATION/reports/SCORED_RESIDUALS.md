# Residuals of officially scored recipes

Generated 2026-09-19T13:51:58+00:00. Residual = prediction − actual; positive means overprediction.

The ranking labels are hidden. These plots diagnose held-out 2025 predictions of scored recipes; they do not reveal which 2026 flights caused a leaderboard error.

Main charts use **1,079,528 distinct July–December flights**, A1–A6, three seeds. Squared errors are averaged across seeds, not predictions. Signed distributions and scatter use seed 42. All 16 folds are shown separately, and R_B_long preserves the original averaging over origins. No rows, folds or extreme labels are removed. Target-conditioned residual patterns include regression-to-the-mean effects; they are not by themselves evidence for a deployable correction.

**Coverage:** original OOF for v12–v16; v19 reconstructed from saved T1 model predictions with the registered four-hour rule; v30/v33/v34 use the September 19 replay. These last four are marked †. Their official scores belong to the original submitted files. The v34 replay differs by +0.00697 s combined from the original. All 27 scored submissions appear in the inventory; unavailable predictions are not invented.

Official-score source: live, 2026-09-19T13:36:31+00:00.

[Open the interactive report](scored_residuals/index.html) · [Download the eight-page PDF](scored_residuals/residual_atlas.pdf)

## What remains in v34

- Next-month RMSE is **253.07 s**; longer-gap is **228.98 s**. The seed-42 median absolute error is 103.1 s, 99th percentile 782.0 s, and maximum 36,962 s.
- The worst **100 flights contribute 15.7%** of squared error; the worst 1,000 contribute **27.5%**. These are distinct flights, not repeated folds.
- Missing-ID flights are **1.261%** of flights but **23.3%** of squared error (RMSE 1088 s). ID-present flights still carry **76.7%** of squared error (RMSE 223.0 s). Both the extreme records and the broad operational population matter.

| Observable regime | Flights | Days | RMSE (s) | Error share | Bias (s) |
|---|---:|---:|---:|---:|---:|
| Rome next-day hard rule | 13 | 12 | 17823.3 | 5.97% | -10513.4 |
| Rome other hard rule | 73 | 53 | 6392.0 | 4.31% | +2450.0 |
| Rome no-ID 1–4h | 811 | 163 | 2454.1 | 7.06% | +182.9 |
| Rome other no-ID | 60 | 50 | 3617.2 | 1.14% | -974.8 |
| Other airport no-ID | 12,660 | 184 | 513.2 | 4.82% | -3.2 |
| ID twin | 53 | 45 | 2181.2 | 0.36% | -1756.7 |
| Other ID | 1,065,858 | 184 | 222.5 | 76.32% | +2.5 |

## A concrete next experiment: the Rome 1–4h mixture

Among **811 distinct no-ID Rome flights** at 1–4-hour schedule gaps, the 278 flights assigned probabilities between 0.3 and 0.7 average **47.6% predicted schedule substitution versus 34.9% observed**. The excess is positive in all three seeds. A descriptive day-bootstrap interval for the seed-42 excess is [4.4, 19.6] percentage points. The broad middle band was chosen during inspection, so this is exploratory evidence.

The classifier distinguishes the extremes reasonably, but its middle probabilities often put too much weight on the schedule gap. Test a **shrunk probability calibration layer**, trained using inner chronological out-of-fold predictions within each training window. Judge it by taxi-time squared error as well as probability calibration: the cost of a wrong probability increases with the distance between the schedule-gap and ordinary predictions. Keep a separate ordinary/late estimate; do not convert uncertain probabilities into a new hard rule. No such calibrator was fitted here. This targets the 1–4h mixture, separately from the already completed day-shift corrections and 4–12h rule experiments.

[Probability-calibration figure (PDF)](scored_residuals/09_rome_calibration.pdf) · [Per-seed bin counts and errors](scored_residuals/rome_probability_bins.csv)

## Other modelling implications

1. **Treat Rome as a competing-clock problem.** The gap/label scatter separates ordinary taxi, schedule-gap labels, and roughly one-day labels. The hard rule can badly overpredict an ordinary flight and underpredict a day-shift record. A route must use observable clock agreement, record provenance and arrival-handler evidence; the true label cannot define prediction routes. The September 19 artifact study has already tested day blending and twins. Reuse those results; these plots are not a new validation of the same corrections.
2. **Inspect broad within-airport operational errors too.** The hour, queue and weather plots condition on an existing flight ID and compare within airports. Persistent underprediction at high queue/clock gaps suggests testing airport-specific queue interactions or a separate congestion component. A feature trend is association, not proof that adding it improves the model; weather and queue features are already in v34.
3. **Use recurring bias as a small, separately validated correction candidate.** The table below clusters uncertainty by day. Any airport/hour or stand correction needs shrinkage and must be fitted inside each training window; fitting offsets to these residuals and reporting the same flights as validation would leak outcomes.
4. **Check complementarity before an ensemble.** The shared-error table compares the same flights and a fixed 50:50 blend on seed 42. It is descriptive, with no tuned weight or new candidate selection. Repeated catastrophic errors will not disappear by averaging highly correlated models.

The plots are exploratory on reused development folds. There is no January residual coverage here; airport, weather and season patterns may transfer differently to the January/July 2026 ranking set. Tiny groups are hypotheses, not established general improvements.

### Airport bias with flight ID present

Three-seed mean residual, 2,000 calendar-day bootstrap draws; descriptive, unadjusted intervals. Intervals condition on the fitted models and do not establish future transfer.

| Airport | Flights | Bias (s) | 95% day-bootstrap interval |
|---|---:|---:|---|
| EDDF | 119,375 | +1.6 | [-1.8, +4.7] |
| EDDM | 86,726 | -3.8 | [-8.2, +0.3] |
| EGLL | 120,910 | +4.9 | [+1.4, +8.3] |
| EHAM | 124,708 | -4.9 | [-8.4, -1.7] |
| LEBL | 91,711 | +0.3 | [-4.5, +5.0] |
| LEMD | 108,643 | +1.7 | [-1.3, +4.9] |
| LFPG | 121,256 | +2.5 | [-1.8, +6.9] |
| LIRF | 81,918 | +4.9 | [-2.1, +11.2] |
| LSZH | 69,166 | +0.2 | [-4.0, +4.1] |
| LTFM | 141,498 | +12.5 | [+7.0, +17.4] |

## Inventory and reproduction

| Version | Official LB | Residual coverage | R_A | R_B_long |
|---|---:|---|---:|---:|
| v1 | 661.5737 | unavailable | — | — |
| v2 | 533.3759 | unavailable | — | — |
| v3 | 405.6987 | unavailable | — | — |
| v4 | 405.7838 | unavailable | — | — |
| v5 | 518.3068 | unavailable | — | — |
| v6 | 498.3336 | unavailable | — | — |
| v7 | 486.0517 | unavailable | — | — |
| v8 | 472.7133 | unavailable | — | — |
| v9 | 433.6797 | unavailable | — | — |
| v12 | 406.4281 | original | 366.746 | 334.178 |
| v13 | 354.4303 | original | 300.590 | 256.197 |
| v14 | 354.0004 | original | 296.904 | 258.551 |
| v15 | 344.3114 | original | 296.278 | 255.620 |
| v16 | 344.4821 | original | 292.879 | 257.479 |
| v17 | 345.8984 | unavailable | — | — |
| v18 | 342.5049 | unavailable | — | — |
| v19 | 338.1113 | reconstructed | 293.648 | 255.661 |
| v20 | 345.3113 | unavailable | — | — |
| v21 | 342.4009 | unavailable | — | — |
| v26 | 333.8712 | unavailable | — | — |
| v27 | 329.1609 | unavailable | — | — |
| v28 | 322.6253 | unavailable | — | — |
| v29 | 314.4667 | unavailable | — | — |
| v30 | 309.9872 | replay | 256.194 | 229.692 |
| v33 | 282.4525 | replay | 253.950 | 229.286 |
| v34 | 280.4658 | replay | 253.070 | 228.979 |
| v37 | 280.1743 | unavailable; recipe unverified | — | — |

```bash
.venv/bin/python VALIDATION/tools/plot_scored_residuals.py --leaderboard tmp/residual_audit/leaderboard.json
```

Outputs include file hashes, exact score-reproduction checks, all fold/slice metrics, the 250 worst distinct flights and self-contained HTML. No training or upload was performed.
