# Training versus ranking: step 4

**The evidence points toward prolonged snow and airport disruption. The arrival
checks did not find a clear change in timestamp recording.**

| EHAM arrivals | January 2025 | January 2026 |
|---|---:|---:|
| Taxi-in over 30 minutes | 50 / 18,469 (0.27%) | 264 / 16,246 (1.63%) |
| Over 30 minutes, without snow in the previous day | 0.28% | 0.24% |
| Flights exposed to prolonged snow | 32 (0.17%) | 2,890 (17.79%) |

“Prolonged” means snow at more than a quarter of the previous day's 48 half-hour
weather checks. It measures reported snowfall duration, not snow depth.
**228 of the 264 long arrivals (86%) occurred in these conditions.** For 2,036
arrivals, this duration exceeded anything seen among EHAM arrivals in all of 2025;
those flights include 187 of the long arrivals.

This matches Schiphol's published account of heavy, prolonged snowfall and
operational disruption in early January, including de-icing and handling delays.
[Official review, 28 April 2026](https://news.schiphol.com/schiphol-and-klm-strengthen-joint-approach-to-winter-weather/).
That corroborates the context; it does not verify individual labels.

**Recording checks weaken the broad recording-change hypothesis.** Of the 264
long arrivals, 260 have flight IDs; only one in-block time is within a minute of
schedule. None belong to batches of three identical in-block timestamps. There
is no broad increase in clock rounding. Label/clock arithmetic agrees throughout
the audited data, although that is not independent proof that the timestamps are correct.

**The departure lead overlaps the disruption.** January 2–5 contain 192 long
arrivals, 409 of January's 805 missing-ID departures, and 278 of its 418 departures
over four hours behind schedule. This is an association across different flights;
2026 departure labels remain hidden.

**Matching cannot give a clean percentage explained.** Weather/traffic/runway
matching retains 84% of January flights and still leaves higher tail rates.
Adding stand, time, route and fleet retains only 16.5% and four of the 264 long
arrivals. A small remaining difference after those exclusions is misleading.
The snow-duration follow-up also loses many extreme cases; sparse day coverage
prevents reliable intervals for several adjusted comparisons. EDDF/LSZH January
and EHAM July do not show the same tail increase.

Next candidate: test prolonged-snow and recently completed-arrival congestion
features. Historical severe-weather examples are sparse, so improvement remains
unproven. No taxi-out fits, uploads or Weighted CV changes were made.

Eight tests passed; independent checks covered 710,038 labels, weather/traffic
joins, snow duration and aggregation. A traffic-bin type mismatch found in the
July control was corrected and regression-tested. All data stays cached locally.

[Conditional comparisons](train_ranking_step4/conditional_tail_rates.csv) ·
[Snow duration](train_ranking_step4/snow_duration_groups.csv) ·
[Recording checks](train_ranking_step4/recording_patterns.csv) ·
[Departure/arrival dates](train_ranking_step4/departure_arrival_days.csv) ·
[Audit](train_ranking_step4/independent_audit.json)

Reproduce: run `investigate_arrival_shift.py`, `check_arrival_snow_severity.py`,
then `audit_arrival_shift.py` under `VALIDATION/tools/` with `.venv/bin/python`.
