# Stage 0 complete — recovery-off baseline

Completed 2026-09-16 local time. Recovery experiments have not started.

## Frozen baseline

- **Run:** [`v3_20260915T225301657023Z`](../runs/v3_20260915T225301657023Z/result.json).
- **Recipe:** 1,500 fixed rounds; seeds 42, 142, 242; original model features;
  recovery off; training labels 1–200,000 seconds; all validation labels scored.
- **Coverage:** six six-month training windows, 18 fitted models, 16 folds and
  48 fold/seed prediction files. No twelve-month fit or submission was performed.
- Source snapshots, data hashes, fitted models, training-loss records and
  validation predictions are saved with the run. The interruption/continuation
  and actual per-fit thread counts are recorded; completed work was reused.

| Readout | Unique validation flights | RMSE (seconds) | Seed SD |
|---|---:|---:|---:|
| Next-month `R_A` | 1,079,528 | **371.0708** | 0.6578 |
| Longer-gap `R_B_long` | 328,009 | **335.2416** | 0.5901 |
| Provisional `R_combined` | — | 351.5600 | — |

The combined number is a local diagnostic, not a leaderboard forecast.
Per-fold, per-airport, per-seed, flight-plan, tail, and context-coverage
statistics are in `result.json`.

## Did 1,500 rounds help?

The 800-round predictions were saved from the same fitted models and compared
on the same validation rows. No extra 800-round model fits were needed.
Differences below are **1,500 minus 800**; positive is worse. Intervals use
2,000 paired UTC-day bootstrap replicates and the existing v3 flight weighting.

| Readout | 800 rounds | 1,500 rounds | Difference | 95% interval | Finding |
|---|---:|---:|---:|---|---|
| `R_A` | 366.7673 | 371.0708 | +4.3035 | [+1.3487, +7.4475] | Regression |
| `R_B_long` | 334.2028 | 335.2416 | +1.0387 | [−1.7224, +3.7062] | Inconclusive |
| `R_combined` | 348.9996 | 351.5600 | +2.5603 | [+0.4185, +4.7543] | Regression |

Training loss continued falling while held-out performance deteriorated on the
main next-month readout. The requested 1,500-round B0 is frozen for controlled
recovery comparisons; any return to 800 rounds must be explicit rather than
bundled with a recovery change. These are reused development folds, not a new
independent test or an official score.

- [Paired checkpoint report](../runs/v3_20260915T225301657023Z/comparison_round800.json)
- [Training-loss chart](../runs/v3_20260915T225301657023Z/training_loss.png)
- [Training-loss PDF](../runs/v3_20260915T225301657023Z/training_loss.pdf)

## Recovery-off equivalence

The shared loader now retains observable fields and raw-missingness flags
needed by later recovery work. It does not load hidden block timestamps or
perform recovery. Model inputs remain selected by the existing feature list.

On the complete January–June matrix and July validation matrix, legacy and
expanded loading gave **exactly equal values, dtypes, row IDs/order, targets,
category order, and model feature matrices**. The check covered 1,005,290
training rows after filtering and 190,713 validation rows.

All **23 tests passed**, including checkpoint scoring, parallel/sequential
prediction equivalence, recovery-field isolation, and reuse of completed fits
on resume. Final artifact checks confirmed all 18 1,500-round models, all 16
folds, unchanged model-source hashes, and one completed experiment-log entry.

## CPU decision

Use **sequential fits with eight threads** on this 7800X3D/WSL setup.

The short 300-round probe favoured eight over sixteen threads (median 13.15 s
versus 22.86 s). A two-fit/four-thread shared-matrix probe initially looked
faster, but that advantage did not hold in sustained validation: M2's parallel
fits took approximately 272 s each, while its sequential eight-thread seed took
58 s. Later sequential fits remained near one minute. The precise cause of the
parallel slowdown was not established; no swapping or cgroup throttling was
observed. Do not extrapolate the short-probe gain to a complete experiment.

The run preserved six completed fits and seven folds when switching execution
mode. Training and scoring across both execution segments took **31.44 minutes**,
excluding the CPU probe and the separate paired-report pass.

[CPU timings and equivalence record](benchmarks/stage0_threads_20260915T224428Z.json).

**Stop point:** Stage 0 is complete. Stage 1's exact-twin recovery is next and
has not been implemented or run.
