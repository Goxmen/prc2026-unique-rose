# Training versus ranking: step 2

**Real year-to-year differences survive the date holdouts. They are modest overall,
and similarity still does not measure prediction cost.**

| Comparison | Raw-feature AUC | With weather/traffic |
|---|---:|---:|
| January 2025 → January 2026 | 0.624 | 0.585 |
| July 2025 → July 2026 | 0.664 | 0.646 |
| January → February 2025 control | 0.539 | 0.543 |
| July → August 2025 control | 0.531 | 0.486 |

AUC averages ten airports and four held-out date blocks; 0.5 is chance. These two
controls do not cover every seasonal change. Weather/traffic does not reliably help.

**Where to look next:**

- **EHAM January:** 9.4% of ranking flights are unusually distant in raw-feature
  kNN, versus 4.9% of held-out 2025 flights. This increase remains with 1, 5 or 10
  neighbours. Among ranking flights missing an ID, 39.5% are distant; among those
  taking off over four hours after schedule, 63.9% are. Weak whole-airport AUC
  (0.539) does not rule out these groups. Adding weather weakens the contrast:
  10.3% versus 8.8%, with wide uncertainty across days.
- **July EDDF and EDDM:** raw AUCs are 0.761 and 0.713, respectively. Shuffling
  stands/runways hurts the richer classifier on every held-out block at both
  airports. Raw kNN flags 8.4% versus 5.0% at EDDF, but only 6.2% versus 5.8% at
  EDDM. Munich's changed stand mix is identifiable without a broad loss of nearby
  examples. Its new-stand subset is more distant: 15.2% of those flights are flagged.

**Propensities need caution.** Calibration on separate dates still varies by week.
January's richer model is worse than constant 0.5 probabilities on Brier loss:
0.261 versus 0.250. Adopted validation weights are unchanged.

**A weather-distance bug was corrected.** Nearly constant visibility/ceiling
fields fell back to a scale of one metre/foot and dominated kNN. Using their
observed donor range fixes that; constant fields use a bounded change indicator.
Barcelona January's flagged share falls from 26.4% to 9.2%. Use **`knn_current.csv`**
for corrected results. The initial run is preserved; this subsequent correction
is not independent confirmation.

Exact neighbours use the same airport/month's 2025 fitting dates. “Distant” means
above the calibration dates' 95th percentile of fifth-neighbour distance. Test
dates are separate, with a one-day buffer. This tests a chosen distance and donor
bank, not all possible training support. No departure errors were inspected.

Ten tests passed. Shuffled-label AUC is 0.502. Independent checks verified IDs,
dates and full-bank distances for 160 original and 160 corrected queries.

[Airport scores](train_ranking_step2/classifiers.csv) ·
[Current kNN results](train_ranking_step2/knn_current.csv) ·
[Affected groups](train_ranking_step2/slices_corrected.csv) ·
[Held-out feature groups](train_ranking_step2/permutation.csv) ·
[Design](train_ranking_step2/design.json) ·
[Scale correction](train_ranking_step2/weather_scale_design.json) ·
[Audit](train_ranking_step2/audit.json) ·
[Saved row/neighbor files for step 3](train_ranking_step2/delivery.json)
