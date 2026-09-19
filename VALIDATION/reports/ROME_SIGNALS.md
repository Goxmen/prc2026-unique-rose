# Rome recording signals: one change at a time on v28

Run `v3_20260918T120331999764Z` (protocol v3, 16 folds, seeds 42/142/242, 45 minutes
on macOS). Reproduce with:

```sh
.venv/bin/python VALIDATION/rome_signals_run.py --label rome-signals
.venv/bin/python VALIDATION/tools/summarize_rome_signals.py --run v3_20260918T120331999764Z
```

## Why

[V28_FAILURES.md](V28_FAILURES.md) showed that v28 loses most at LIRF on rows without a
flight ID. Those rows are one of two things:

- **Schedule-recorded:** off-block was never reported and was logged at the scheduled
  time, so the label is takeoff minus schedule.
- **Genuinely late:** a LIRF departure whose real off-block was recorded loses its flight
  ID only once that off-block is more than an hour after schedule (0.02% at 5–60 minutes,
  about 8.5% beyond). Its label is an ordinary taxi of about 1,050 s.

v28 blends the two hypotheses with p × gap + (1 − p) × ordinary. It is weak on both parts:

- **p:** the classifier cannot see the airline on these rows, because `operator` comes
  from the flight plan they lack.
- **ordinary:** the regressor reads a one-to-four-hour gap as a long taxi. On the
  genuinely late rows it has RMSE 2,122 s and bias +725 s.

The airline matters because it stands for the ground handler. At the same gap, ITA,
KLM, easyJet, BA and El Al flights are almost never schedule-recorded, while Israir
(`ISR`), AJet (`TKJ`), T'way (`TWB`) and Ethiopian almost always are.

## Design

The control is v28 itself. Its boosters are reloaded from run
`v3_20260917T172934966809Z`, and on every fold and seed they reproduced v28's saved
predictions exactly. Each candidate changes one thing, and every candidate sits in the
same prediction files. So each paired day-bootstrap comparison against `pred` measures
one change on identical flights, folds and seeds.

| Candidate | The one change | Refit |
|---|---|---|
| `late` | For LIRF rows with no flight ID and a 1–4 h gap, the (1 − p) part becomes min(late mean, max(gap − 3,600, 60)) instead of the ordinary regressor. The late mean is the mean label of the window's late no-ID LIRF departures (1,009–1,120 s) | none |
| `cls_airline` | Classifier input: airline read from `FLIGHT_mvt`. The first three characters if they are letters (ICAO), otherwise the first two (IATA, as at EDDF and EHAM) | classifier |
| `cls_handler` | Classifier inputs: the airline's share of arrivals at the airport in the same UTC month whose in-block equals the schedule within 60 s (shrunk toward the airport-month share), and their count | classifier |
| `cls_format` | Classifier input: flight-number format — clean, mangled (ICAO prefix + last five callsign characters, e.g. `ITYTY630`, LIRF only), or missing | classifier |
| `dayshift` | When the four-hour rule fires, the takeoff is on a later Rome-local day than the schedule and the gap is at least 14 h: w × (days × 86,400 + ordinary) + (1 − w) × gap, where w is the window's share of such rows whose label was day-shifted (0.50–0.67) | none |
| `reg_airline` | The regressor trained without schedule-recorded rows gets the airline as a categorical input, at every airport | regressor |
| `cls_all` | Classifier with all three classifier inputs | classifier |
| `all` | `reg_airline` + `cls_all` + `late` + `dayshift` | – |

Arrival in-block is landing time plus taxi-in; both are present for every ranking
arrival. A validation month's handler share uses only that month's arrivals, the same
context a ranking month gets.

## Results

Change against the control (v28: 265.3809 / 234.4385 / 248.6174). Negative is better.
The 95% interval is the paired day bootstrap. "Seeds" gives the sign of the change for
each of the three seeds.

| Candidate | Next-month (R_A) | Longer-gap (R_B_long) | Combined | Seeds | Decision |
|---|---|---|---|---|---|
| `late` | −2.03 [−3.43, −0.84] | −0.18 [−1.30, +0.65] | **−1.05** [−2.01, −0.25] | − − − | improvement |
| `cls_airline` | −2.13 [−3.01, −1.28] | −0.43 [−1.06, +0.21] | **−1.23** [−1.79, −0.69] | − − − | improvement |
| `cls_handler` | −1.67 [−2.33, −0.91] | −0.69 [−1.15, −0.23] | **−1.15** [−1.58, −0.71] | − − − | improvement |
| `cls_format` | −0.31 [−0.63, −0.06] | +0.07 [−0.05, +0.19] | −0.11 [−0.27, +0.03] | − − − | inconclusive |
| `dayshift` | +0.01 [−8.52, +8.51] | −8.83 [−21.83, +0.11] | −4.59 [−13.88, +2.97] | − − − | inconclusive |
| `reg_airline` | −0.61 [−0.94, −0.29] | −0.34 [−0.68, −0.01] | **−0.47** [−0.72, −0.23] | − − − | improvement |
| `cls_all` | −2.17 [−3.04, −1.22] | −0.65 [−1.27, −0.00] | **−1.37** [−1.94, −0.79] | − − − | improvement |
| `all` | −5.64 [−14.50, +2.91] | −9.95 [−23.11, −0.43] | −7.88 [−17.28, −0.19] | − − − | improvement |
| **`all` without `dayshift`** | **−5.04** [−7.18, −3.06] | **−1.18** [−2.54, +0.00] | **−3.00** [−4.24, −1.83] | − − − | improvement |

The last row is derived exactly, not fitted. The day-shift rewrites only rule rows,
which otherwise keep the rule's gap prediction. The summary tool rebuilds that column
in a temporary copy and runs the same comparison. As a model it scores
**260.3360 / 233.2610 / 245.6201**.

Where each change acts, as a change in the control's pooled squared error (16 folds,
all seeds):

| Candidate | Rome rule rows | LIRF no ID, 1–4 h | LIRF, the rest | Other airports |
|---|---:|---:|---:|---:|
| control's share | 8.29% | 7.93% | 16.01% | 67.77% |
| `late` | 0 | −1.12% | 0 | 0 |
| `cls_airline` | 0 | −0.81% | −0.06% | 0 |
| `cls_handler` | 0 | −0.80% | −0.12% | 0 |
| `cls_format` | 0 | −0.08% | −0.02% | 0 |
| `dayshift` | −2.50% | 0 | 0 | 0 |
| `reg_airline` | 0 | −0.13% | −0.05% | −0.22% |
| `cls_all` | 0 | −0.95% | −0.10% | 0 |

## What this says

1. **Four changes hold up alone,** and every one of them has all three seeds on the same
   side: the late estimate, airline in the classifier, the arrival handler share, and
   airline in the regressor. Together, without the day-shift, they give −3.00 combined
   and −5.04 next-month.
2. **Airline and handler share measure the same thing.** Each is worth about −1.2 alone;
   together with the format flag they give −1.37. The handler share is the one that also
   helps longer-gap. It is computed from the month's own arrivals, so it should also
   follow a handler that changes airlines in 2026, which fixed 2025 airline rates cannot.
3. **The flight-number format adds almost nothing.** Most mangled rows are already
   explained by their airline. It stays in `cls_all` only because that is the recipe
   that was tested.
4. **The day-shift is a coin flip on about 11 flights.** It gains when a 14–18 h rule
   flight really was day-shifted: Rome–Buenos Aires on 4 November and Rome–London
   Heathrow on 12 November each appear in four November folds. It loses when the flight
   was schedule-recorded: two Qantas flights to Sydney in September. R_A is +0.01 and
   R_B_long is −8.83, depending only on which of those flights a score contains. It is
   not adopted.
5. **The late estimate is the most transferable part.** It follows from the
   flight-ID cliff rather than from tuning, and its one fitted number was stable across
   windows (1,009–1,120 s).

## Caveats

- **Optimistic CV.** Every candidate was defined after exploring v28's out-of-fold
  predictions on these same folds. The refits are new, but the choice of what to test
  was not blind.
- **No January.** None of the folds contains January, and January is half the ranking
  set.
- **More of these rows in ranking.** The ranking set has 305 LIRF rows without a flight
  ID at 1–4 h: 0.088% of rows, against 0.056% here. If the gain transfers, it should be
  larger there in proportion.
- **Not yet in the submission path.** Nothing has been refitted on all of 2025 or
  submitted. That needs a full-year variant of `rome_hybrid_submit.py`.

## Files

- `VALIDATION/rome_signals.py`: inputs, fitted corrections and prediction path.
  With every switch off it reproduces `rome_hybrid.predict`.
- `VALIDATION/rome_signals_run.py`: the A/B run.
- `VALIDATION/tests/test_rome_signals.py`: unit tests.
- `VALIDATION/tools/summarize_rome_signals.py`: comparisons, the derived row and the
  population split.
- In the run directory: `comparison_*_vs_pred_*.json` for each candidate,
  `rome_signals_summary.json`, and `progress.json` with the fitted corrections per
  window.
