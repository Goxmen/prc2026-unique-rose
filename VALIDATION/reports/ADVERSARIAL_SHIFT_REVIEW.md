# Adversarial test review — 20 September 2026

I reproduced the reported numbers exactly. **The code uses the wrong weights
for part of validation.** Fixing this changes the weighted CV score from
**252.30 to 269.40 seconds**. The test also does not prove that the remaining
gap comes from changes in the labels.

**The weighting bug**

Weights change how much each validation flight counts, to make validation
better match the ranking set. They must account for which flights are in each
validation group.

The code learns weights for July–December, then uses them for both groups.
That fits group A, which covers those six months. But group B contains only
November–December and needs weights learned for those two months.

I reran the weighting models for B with the correct months. The taxi-out
predictions stayed the same:

| Weighted RMSE, seconds | Original | After fixing B |
|---|---:|---:|
| A: July–December | 270.85 | 270.85 |
| B: November–December | 236.52 | **268.23** |
| Combined | 252.30 | **269.40** |

There is a visible check on this: ranking has **1.53%** of flights without a
flight ID. The original weights bring B to only **0.99%**; the corrected
weights bring it to **1.43%**. The match improves, but remains imperfect.

**The remaining gap has no proven explanation**

Fixing B reduces the gap to the official 279.38 score from **27.08 to 9.98
seconds**. Neither number tells us why the gap remains. The weighting model
omits weather, traffic and history. Some ranking airport/stand combinations
never appear in validation. And the CV models trained on six months, while
the submitted model trained on twelve.

The new 269.40 score has not been validated as a leaderboard forecast. Nor
have we shown that better weights could not move it further.

EHAM's January arrival taxi-in tail really did grow. But the study did not
check whether weather, routes or congestion explain that change. Arrivals
also have a different target from departures. This finding warrants a closer
look; it does not prove a change in how departure labels were recorded.

**Four other claims need weaker wording**

- **D0 does not prove “no leakage.”** It shows chance performance on random
  labels. Adding a year column would still pass D0, because all its rows are
  from 2025, while making the 2025-versus-2026 test trivial.
- **D3 beating D2 does not make 2026 harmless.** AUC measures how easily the
  classifier distinguishes the datasets. It does not measure how much their
  differences hurt taxi-out predictions.
- **Stand and destination importance is not yet verified.** They receive
  **86% of training gain even in D0**, where predictions are no better than
  chance. Check whether removing or shuffling these features hurts predictions
  on held-out rows before treating them as the explanation.
- **Moderate weights can still amplify a few large errors.** The ten flights
  that add most to each group's increase account for **25% of A's** and
  **37% of B's** net increase in mean squared error. Weight size alone misses
  this concentration.

**What the test does support**

The original weights consistently raise CV error. Resampling days gives an
increase of roughly **10–22 seconds**, with those weights held fixed. That
supports the observed increase, but does not show that it “explains 37%” of
the leaderboard gap.

Before using these scores to guide model choices, check the corrected weights
against ranking flight proportions and test the method on a historical period
where departure labels are known.

[Detailed numbers](adversarial_shift_review.json) ·
[Review script](../tools/review_adversarial_shift.py) ·
[Original results](adversarial_shift.json)

No taxi-out models were refitted and no submissions were uploaded.
