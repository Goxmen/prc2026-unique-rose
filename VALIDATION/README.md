# Validation harness

The plan is in [`ch07-prc-validation.html`](ch07-prc-validation.html). This is the
part of it that runs, plus four corrections to the design that page describes.

**Read [`FINDINGS.md`](FINDINGS.md) first** — what the harness actually turned up,
including the reason the score is decided by about a hundred corrupt rows and
why the page's central staleness argument does not survive contact with the data.

## What the page got right, and what changed here

The page's core rule is kept exactly: **features built from past departure
taxi-outs are computed for validation rows at the fold's training cutoff, never
at the row's own date.** In the 2026 test months the freshest taxi-out label we
will ever own is 2025-12-31, so any window feature must be evaluated in that
condition or the fold score is measuring a model we cannot deploy.

Four things are done differently.

**1. Staleness is measured as a difference, not a level.** The page reads the
July rehearsal's absolute RMSE as an estimate of July 2026. It cannot be one:
that fold trains on 6 months while the real July model gets 12, so its level is
inflated by a training-size handicap that has nothing to do with staleness. The
harness therefore runs the July fold twice — `jul_stale` (freeze Jan 31) and
`jul_fresh` (freeze Jun 30) — and reports the **paired difference**. Training
size cancels in the difference, so that number is comparable across runs and is
the headline output.

**2. The iterate folds are spread across seasons.** The page tunes on Sep/Oct/Nov,
all autumn, then ships to a January and a July test. Here they are April, August
and October.

**3. Every fold runs under several seeds and carries a bootstrap band.** The page
reports single numbers and then argues elsewhere that taxi-out is heavy-tailed
enough to need a bootstrap. Both bands are printed; a difference smaller than
them is not a result.

**4. The lag-training experiment is redone properly.** The page's simulation
"killed" the idea of training on staleness-matched features, but its
implementation lagged *every* rolling feature by six months, including a
same-day congestion term that is pure noise once lagged. `--lag-train-days`
lags only the taxi-out windows and leaves the fresh features alone.

## The availability contract

Three feature classes, in [`features_time.py`](features_time.py). The class
decides the freeze; it is not a flag anyone has to remember.

| class | features | frozen? | why |
|---|---|---|---|
| `STALE_ABLE` | `op_taxi_30d`, `rwy_taxi_30d`, `aphour_taxi_30d`, `stand_taxi_90d` | **yes** | built from DEP taxi-outs, which stop at 2025-12-31 |
| `FRESH_ROLL` | `arr_taxiin_7d` | no | ARR rows keep `BLOCK_TIME` and `TAXITIME` in `ranking.parquet` (verified: all 152,871 Jan + 191,822 Jul arrivals populated) |
| `FRESH_NOW` | `dep_actual_30m_before`, `arr_actual_30m_before`, `dep_sched_60m_around` | no | movement counts from timestamps that survive in ranking; no target involved |

The "after" window is built from **scheduled** times, not actual ones, so it
stays knowable at off-block.

## Folds

| fold | role | train | validate | freeze |
|---|---|---|---|---|
| `apr` | iterate | Jan–Mar | Apr | Mar 31 |
| `aug` | iterate | Jan–Jul | Aug | Jul 31 |
| `oct` | iterate | Jan–Sep | Oct | Sep 30 |
| `dec_jan26` | rehearsal, Jan-2026 island | Jan–Nov | Dec | Nov 30 |
| `jul_stale` | rehearsal, Jul-2026 island | Jan–Jun | Jul | **Jan 31** |
| `jul_fresh` | control for the above | Jan–Jun | Jul | Jun 30 |
| `sep` | iterate, **off by default** | Jan–Aug | Sep | Aug 31 |
| `nov` | iterate, **off by default** | Jan–Oct | Nov | Oct 31 |

`sep` and `nov` exist so `--train-months` has enough feasible folds to build a
fixed-length iterate set; name them with `--folds` to switch them on.

Tune on the iterate folds. Leave the rehearsals alone for comparing finished
candidates. Training rows are filtered to the plausible taxi-out range;
**validation rows never are**, because the leaderboard does not filter either.

## Running it

```bash
# wiring check, ~2 min. Scores are meaningless (a 5% sample shrinks every
# congestion count by 20x) — this only proves the plumbing works.
.venv/bin/python VALIDATION/harness.py --smoke

# Phase 0 audit, ~1-2 min, no training. Read this before trusting any fold.
.venv/bin/python VALIDATION/audit.py

# the real run, roughly an hour on a laptop (6 folds x 3 seeds x 800 rounds)
.venv/bin/python VALIDATION/harness.py --label baseline

# does training on stale features help the July island? (2 folds only, ~15 min)
.venv/bin/python VALIDATION/harness.py --label lag180 \
    --folds jul_stale,jul_fresh --lag-train-days 180
```

Three more knobs, all documented in `--help`:

```bash
# keep the corrupt long-tail rows in training. This is the change that moved the
# official score 533 -> 406; see FINDINGS.md section 3.
--max-taxi-sec 200000

# fixed rolling training window, so fold scores are comparable to each other.
# Folds whose window will not fit inside 2025 are skipped with a message.
--train-months 6

# add has_flight_plan, an explicit flag for the rows where corrupt labels
# concentrate. Wired but never run — see FINDINGS.md section 7.
--fpl-flag
```

## Submitting

`submit.py` trains the harness-validated configuration on all of 2025 and fills
the template. It is deliberately separate from `pipeline/train.py`, which has
none of the time features and hardcodes the old filter.

```bash
.venv/bin/python VALIDATION/submit.py --version 3 --run-id <harness run_id>
mc cp submissions/unique-rose_v3.parquet opensky/prc-2026-unique-rose/
mc cat opensky/prc-2026-unique-rose/unique-rose_v3.parquet_result.json   # the score
```

The horizon split matters here and is easy to get wrong in either direction:
departure taxi-out windows freeze at 2025-12-31 forever, but arrival taxi-ins and
every movement timestamp exist inside the ranking months, so those tables are
built from 2025 **plus** the ranking rows and stay current. Feature tables span
577 days.

Each run writes `runs/<run_id>/result.json`, per-fold OOF predictions, and one
line in `runs/experiments.jsonl`.

## Reading the output

- **Per-fold RMSE** with `seed sd` (model noise) and `boot se` (sampling noise).
  Ignore differences inside those bands.
- **Staleness penalty** — the paired `jul_stale − jul_fresh`. The trustworthy number.
- **Pooled projection** — Dec and July squared errors pooled on the real test
  weights (Jan 152,719 / Jul 192,122 rows, so July is 55.7% of the score).
  It is **not** a forecast. Its known biases are printed with it, and they do not
  all point the same way. Its purpose is to be tracked against official scores.

## Closing the loop with the leaderboard

```bash
.venv/bin/python VALIDATION/log_official.py --run <run_id> --official 412.7 \
    --submission unique-rose_v3.parquet
.venv/bin/python VALIDATION/log_official.py --report
```

One pair proves nothing. After three or more, the report prints the offset, its
spread, and the local↔official correlation. A **stable offset** means the
leaderboard score can be derived as `official ≈ local + offset`. A **sign flip**
means the harness has stopped tracking what pays — stop feature work and find
out why.

## Known caveats

- Neither rehearsal ages the *model*, only its features. The real July model is
  seven months old at prediction time and no fold built from one year can copy
  that. Expect July to be the harder half.
- `--sample` and `--smoke` distort the congestion counts. Never compare a
  sampled score to a full one.
- `AOBT_3_flt` was checked and is **not** a leak (exact match on 0.7% of rows,
  median error 175 s), so the fold scores measure something real. The audit
  re-checks this on every run.
- Comparing two models? Use a **paired** bootstrap on their OOF predictions, not
  the `boot se` column. That column is the uncertainty of an absolute level; two
  models scored on the same rows share their tail draws, and the SE of the
  difference is roughly three times smaller.
