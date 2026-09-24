# Training versus ranking: step 3

**Amsterdam's missing-flight-ID group is the strongest lead. kNN helps find
expensive flights; the cost of the changed stands remains uncertain.**

These are actual held-out **2025** errors, within each airport/month:

| Group | Model | Flights | Share of flights | Share of squared error | Group / other flights RMSE |
|---|---|---:|---:|---:|---:|
| EHAM January: missing ID | v15 | 293 | 1.59% | **29.51%** | 1,019 / 200 s |
| EDDF July: distant in raw kNN | v38 | 1,075 | 5.04% | **16.46%** | 353 / 183 s |
| EDDM July: distant in raw kNN | v38 | 915 | 5.78% | **21.28%** | 349 / 166 s |

**Amsterdam deserves the next investigation.** Its missing-ID share rose to
4.96% in January 2026. The older January model finds this group expensive under
both saved holdout designs. v38 also finds it more expensive in every July–December
month: pooled, 1.84% of flights contribute 6.94% of squared error. January's size
estimate is fragile: ten flights supply 65% of the missing-ID group's error.
Long schedule gaps overlap this group; their error shares must not be added.

**kNN can prioritize flights, but its error estimates are too optimistic.**
Using five neighbours' saved errors, the highest-risk 20% of July flights capture
34–77% of actual squared error across the ten airports. One- and ten-neighbour
checks also concentrate errors. However, average predicted risk understates
actual RMSE at every airport. This does not establish accurate 2026 score prediction.

**Propensity is not a reliable cost ranking.** In July EDDF/EDDM, the highest
raw-propensity bins have *lower* RMSE than their airport averages: 150/129 s
versus 195/182 s. Separating years and finding costly flights are different tasks.

**New stands still need evidence.** July has only eight EDDF and two EDDM
validation flights on stands absent from the model's training window; across
July–December, there are 28 and 138. Ranking's new stands have moderately higher
historical-neighbour risk, but that cannot measure their actual 2026 errors.

January uses the **older v15 retrospective holdout**, because v38 has no January
OOF predictions. v40's OOF files were unavailable on both machines. These are
exploratory findings on reused development data; the departure-label hypothesis
and any attribution of the leaderboard gap remain open.

Five tests and independent checks of 50 cost slices, 200 neighbour estimates and
two stand-support counts passed. Original hashes and saved scores reproduce.
Only 13 MB was downloaded; inputs are now cached locally. No taxi-out fits,
uploads or changes to Weighted CV.

[Actual costs](train_ranking_step3/actual_cost.csv) ·
[Neighbour checks](train_ranking_step3/proxy_validation.csv) ·
[Ranking proxies](train_ranking_step3/neighbour_proxies.csv) ·
[Audit and coverage](train_ranking_step3/audit.json) ·
[Independent audit](train_ranking_step3/independent_audit.json)

Reproduce locally: `.venv/bin/python VALIDATION/tools/analyze_train_ranking_cost.py`,
then `.venv/bin/python VALIDATION/tools/audit_train_ranking_cost.py`.
