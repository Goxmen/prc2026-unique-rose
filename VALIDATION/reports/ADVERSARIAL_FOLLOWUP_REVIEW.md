# Follow-up review — 20 September 2026

**The two earlier calculation bugs are fixed. The transport test gives a mixed
result. The near-perfect classifier score does not survive holding out whole days.**

The unseen-support figures match the independent audit: the hypothetical
stand/runway benefit is 0.416 seconds, with missing-key flights retained.
The weighting tool now uses the right months for each group and corrects for
class counts. Its latest experimental combined score is **267.58 seconds**;
269.40 belongs to the earlier version, which used all ranking rows in each fit.

**Does weighting predict errors better?**

The design hash matches. I checked the saved target scores, unweighted estimates,
summary arithmetic and pass/fail rules. Across the five tested months:

| Average error in the estimate | Unweighted | Weighted |
|---|---:|---:|
| Overall RMSE | 33.04 s | **8.98 s** |
| Candidate improvement over control | **1.28 s** | 1.54 s |

Weighting helps estimate overall scores here. It has not demonstrated better
model selection. January remains untested. The nine candidate names include a
duplicate and an inactive change; removing them as a descriptive check leaves
the same conclusion. They are not nine independent model comparisons.

**The calendar check needs changing.**

`check_rich_feature_leakage.py:101–105` writes a fixed conclusion that near-perfect
AUC means calendar recognition and no costly shift. Season matching does not
establish either claim.

I reproduced its July-versus-July AUCs, then kept all flights from a date together
in training or testing. Each whole-day test withheld ten dates from each July:

| Split | Raw features | Raw + rich features |
|---|---:|---:|
| Random flights | 0.784 | 1.000 |
| Whole days, seed 42 | 0.564 | 0.557 |
| Whole days, seed 142 | 0.603 | 0.642 |
| Whole days, seed 242 | 0.697 | 0.653 |

The original scores depend heavily on seeing flights from the same days during
training and testing. This supports the concern about recognizing daily patterns.
It does not prove the features are useless for taxi-out prediction, or that the
remaining differences have no cost. These three splits reuse the same two months;
they are sensitivity checks, not new independent datasets.

Replace the hard-coded verdict with these narrower conclusions and use whole-day
holdouts for further classifier checks. Update the artifact's “untested” wording
to report the mixed transport result. Do not promote weighted CV for model selection.

Audit scripts, feature cache, results and five preserved run logs are under
`tmp/adversarial_followup_review/`. Original analysis files were unchanged.
No taxi-out fits or submissions ran during this review.
