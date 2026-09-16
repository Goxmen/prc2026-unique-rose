# Rome rule thresholds — consolidated results

Both saved models were evaluated at **1–8 hours**. Four hours has the lowest
next-month and combined RMSE for both parents, and the lowest longer-gap RMSE
for B0-800. R3's longer-gap minimum is two hours, only 0.0744 s better than four;
that difference is inconclusive.

Local RMSE in seconds, lower is better. R_A means next-month validation;
R_B_long means longer-gap validation. Combined is the existing weighted
squared-error summary, not an official leaderboard score.

| Threshold | B0-800 R_A | B0-800 R_B_long | B0-800 combined | R3 R_A | R3 R_B_long | R3 combined |
|---|---:|---:|---:|---:|---:|---:|
| 1 h | 306.6383 | 256.1773 | 279.6505 | 307.0006 | 257.4945 | 280.4992 |
| 2 h | 300.6796 | 256.3447 | 276.8564 | 299.7589 | 257.4043 | 276.9620 |
| 3 h | 297.7620 | 256.2949 | 275.4306 | 295.7965 | 257.7390 | 275.2434 |
| 4 h | 296.2785 | 255.6200 | 274.3707 | 292.8785 | 257.4787 | 273.7215 |
| 5 h | 300.6098 | 255.6390 | 276.4591 | 297.0856 | 257.6968 | 275.8356 |
| 6 h | 300.5904 | 256.1971 | 276.7374 | 296.9037 | 258.5505 | 276.1937 |
| 7 h | 300.5749 | 256.1971 | 276.7300 | 297.6080 | 258.5505 | 276.5292 |
| 8 h | 301.0414 | 256.1971 | 276.9544 | 297.8710 | 258.5505 | 276.6546 |

Compared with four hours, one and two hours have supported next-month and
combined regressions for both parents. Three-hour differences are inconclusive.
Compared with six hours, four hours improves next-month RMSE by 4.3119 s for
B0-800 and 4.0252 s for R3; longer-gap gains are inconclusive. Four hours helps
82 and harms four next-month flights for B0-800, and helps 80 and harms six for
R3. These counts refer to squared-error changes from the unmodified models.

The grid was extended after viewing the first results. Confidence intervals
are pointwise on reused development folds; four hours is the leading candidate,
not an independently confirmed optimum. Q0/Q1 retain their six-hour definitions.

The two sweeps took 1.621 and 1.631 minutes, with no retraining. All 47 tests
passed. No full-year refit or submission has been made for these sweeps.

Detailed evidence: [4–8 hour report](STAGE2_THRESHOLD_SWEEP.md) and
[1–3 hour extension](STAGE2_THRESHOLD_SWEEP_LOW.md), including paired intervals,
monthly scores, figures and changed-flight CSVs. This table combines their
saved JSON results without rerunning experiments.

The user's proposed follow-up—check leaderboard transfer, then discuss routing
rule-handled rows out of model training and eventually learning corrections
to the rule—is recorded in [the improvement plan](../IMPROVEMENT_PLAN.md#future-follow-up--leaderboard-transfer-training-routing-and-rule-corrections).
