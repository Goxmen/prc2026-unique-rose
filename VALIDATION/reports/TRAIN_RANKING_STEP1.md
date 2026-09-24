# Training versus ranking: step 1

**January at EHAM is the clearest lead. Missing flight plans, long schedule gaps
and snow exposure all increased. Its observable inputs were not ordinary.**
This establishes differences to investigate, not their cause or prediction cost.

Compared January with January and July with July at all ten airports: **2,085,047
training departures and all 344,841 ranking departures**. References use all 11
adjacent-month changes and all 66 pairs of months in 2025.

| Departure measure | 2025 | 2026 | Change | Largest change between any two 2025 months at that airport |
|---|---:|---:|---:|---:|
| EHAM January: missing flight ID | 1.59% | 4.96% | +3.37 pp | 1.37 pp |
| EHAM January: takeoff >4 h after schedule | 0.40% | 2.58% | +2.18 pp | 0.19 pp |
| EHAM January: linked to a snow report | 1.96% | 6.68% | +4.73 pp | 2.36 pp |
| EDDF January: linked to a snow report | 0.67% | 7.02% | +6.36 pp | 0.85 pp |
| LSZH January: linked to a snow report | 1.49% | 9.02% | +7.53 pp | 2.73 pp |

“pp” means percentage points; changes use unrounded values. Missing AOBT3 and
operator accompany the missing flight IDs at EHAM. Three January 2026 dates
account for **40.4% of missing-ID flights** there; the three busiest dates for
the >4-hour cases account for **56.5%** of them. Whole-day holdouts matter.

**July has a different pattern.** Overall missing IDs fall from **2.06% to 1.48%**.
Stand-use changes at EDDM and EDDF exceed every 2025 month pair. At EDDM,
**1,182 / 15,013 flights (7.87%)** use stand codes absent from all of 2025;
at EDDF, **299 / 20,650 (1.45%)** do. Codes may reflect renaming or reassignment.

**New EHAM stands look less compelling:** only **11 / 16,232 January flights
(0.068%)** use codes absent from all of 2025. Turnover against January alone
is 1.30%, within the 2025 range. Aircraft mix also changed: EHAM's A21N share
rose from 4.27% to 9.90%, using a type already present in training.

Overall daily departure volume changes by **−0.64% in January and +0.74% in July**;
EHAM January falls **12.07%**. Route and operator distribution changes stay within
their respective 2025 pairwise ranges at every airport. That does not establish
that they are harmless.

Next: test January EHAM/weather and July stand/fleet differences on unseen days
using propensity scores and nearest neighbours. Prediction cost remains step 3;
the arrival-label hypothesis remains step 4.

The 2025 ranges are descriptive, not significance tests or normal year-to-year
baselines. Weather uses cached METAR reports; traffic uses identical within-month
windows. No departure labels, classifiers, prediction errors or weights are used.

Reproduce: `.venv/bin/python VALIDATION/tools/describe_train_ranking_shift.py`.
Five tests passed; independent SQL matched 420 rates/denominators and all 20
airport-month stand-novelty rates. All airport-months cover every calendar day.
Six movements in neighbouring monthly files are assigned by actual timestamp.

[Numeric changes](train_ranking_step1/scalar_changes.csv) ·
[Category changes](train_ranking_step1/category_changes.csv) ·
[Individual categories](train_ranking_step1/category_details.csv) ·
[Monthly profiles](train_ranking_step1/profiles.csv) ·
[Coverage](train_ranking_step1/coverage.csv) ·
[Method and hashes](train_ranking_step1/metadata.json)
