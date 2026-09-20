# Label-anchor search: what does a corrupted off-block copy?

Measured 2026-09-20 on the pooled seed-42 OOF of `v3_20260919T223006736822Z`
(2,784,530 rows; RMSE `pred37` 238.896, `pred37_opdi` 236.393). **No fits, no
uploads, no labels read from the ranking set.** Scripts are session scratch;
every number below is reproducible from the saved OOF joined to
`data/training_2025-*.parquet`.

## Why this framing

The label is `takeoff - off-block`. `MVT_TIME_UTC_mvt` (takeoff) is supplied for
all 344,841 ranking departures and `BLOCK_TIME_UTC_mvt` is **empty on every one**,
so the hidden quantity is exactly the off-block timestamp. A label is *recoverable*
when the recorded off-block is a copy of some timestamp we can see, because then
the label is observable arithmetic rather than a taxi time to be predicted.
That is what v33 (plan window), the Rome rule, the no-ID caps and the CDG twin
rule each exploit. This search asks whether any **further** such class exists.

## 1. Complete anchor taxonomy

Each row is assigned to the first anchor its recorded off-block matches within
60 s, in the order `aobt3, sched, lobt, iobt, eobt1`.

| class | flight ID | rows | % of SE | RMSE | median label |
|---|---|---:|---:|---:|---:|
| real pushback | yes | 1,850,057 | 60.66 | 225.9 | 947 |
| **real pushback** | **no** | **28,714** | **15.05** | **903.2** | 957 |
| copy of schedule | yes | 205,001 | 10.30 | 279.6 | 963 |
| copy of AOBT_3 | yes | 578,952 | 7.11 | 138.3 | 862 |
| copy of schedule | no | 1,800 | 3.52 | 1743.3 | 4312 |
| copy of LOBT | yes | 52,493 | 1.86 | 234.9 | 958 |
| copy of EOBT_1 | yes | 67,262 | 1.48 | 185.0 | 962 |
| copy of IOBT | yes | 251 | 0.01 | 247.6 | 1089 |

Restricted to rows the model gets badly wrong (`|error| > 1800 s`: 3,495 rows =
**25.75%** of all squared error), 1,268 are schedule copies and **2,085 match no
anchor at all**, carrying 17.67% of SE. The no-anchor bad rows have median label
2,817 s against median prediction 1,551 s: real aircraft that really did taxi
40-50 minutes. That is modelling error, not a recoverable label.

## 2. Schedule copies are recoverable in principle and undetectable in practice

Where the off-block is a schedule copy the label equals `takeoff - schedule`
exactly (100.00% within 60 s, by construction), and the model is badly calibrated
on the delayed ones. The SE is real: 3.62% in the 30-60 min gap band with a
flight ID, 3.44% in 1-2 h, 1.99% in 2-4 h without one.

The rule fails on base rates. Among **all** rows in each gap band:

| gap band | flight ID | rows | schedule copies | precision | ΔRMSE if `pred := gap` |
|---|---|---:|---:|---:|---:|
| 30-60 m | yes | 743,580 | 11,397 | 1.53% | **+540.54** |
| 1-2 h | yes | 204,334 | 1,268 | 0.62% | **+788.74** |
| 2-4 h | yes | 41,657 | 49 | 0.12% | **+836.72** |
| 1-2 h | no | 14,452 | 403 | 2.79% | **+162.86** |
| 2-4 h | no | 6,283 | 389 | 6.19% | **+202.71** |
| > 4 h | no | 684 | 112 | 16.37% | **+318.64** |

Best discriminator found: `LOBT == schedule` (the plan was never revised) lifts the
copy rate from 0.60% to 2.25% — a 3.7x lift onto a still-hopeless base. Adding
"`takeoff - AOBT_3` is not a plausible taxi" gives 780 rows at **2.4%** precision.
`AOBT_3`-derived delay and taxi separate the classes barely at all (median
`takeoff - AOBT_3` 1,198 s for copies vs 1,029 s for real pushbacks).

This extends the earlier Rome-only finding (`prc-ruled-out-experiments`: 49.1% of
LIRF no-plan late rows have `y ~ delay` vs 0.0-0.9% elsewhere) to **every airport
and every gap band**, with the cost of the mistake quantified. The soft version is
already dead too (calibrated blend +0.46, recalibrated `p_substituted` +21.5).

**Verdict: closed.** Do not generalise the Rome rule.

## 3. Two new structural hypotheses, both dead

- **Rotation copy** (the handler logs the inbound aircraft's in-block time as the
  outbound off-block). ASOF-joined the last same-airport-same-stand arrival to
  every departure; 99.99% match. Off-block equals that in-block within 60 s on
  **358 rows (0.013%) holding 0.157% of SE**; within 5 s, 57 rows. Off-block
  equals the inbound *landing* time on **zero** rows.
- **Hand-typed timestamps** (off-block at exactly :00 seconds). 6.43% of rows,
  8.07% of SE, RMSE 264.9 against 234.3 for the rest — mildly noisier, but the
  median label is 960 s and no arithmetic recovers them. Excluding rows that
  already match an anchor leaves 99,925 rows at 5.95% of SE, RMSE 304.5, median
  label 1,079 s. Elevated noise, not an artifact class.

## 4. The one live pocket, and why it is a bet rather than a model

The largest concentrated residual is **29 rows with a label above 50,000 s,
holding 4.22% of all squared error**; predicting them perfectly is worth
**-5.05 s pooled RMSE**. Every damaging one is LIRF, no flight ID, and the
four-hour Rome rule is **already triggering** on all of them.

They are a mixture. Within LIRF no-ID rows by schedule gap:

| gap band | pooled / distinct rows | % of SE | P(day-shift) | current RMSE | Δ if `pred := mean` | Δ if perfect |
|---|---|---:|---:|---:|---:|---:|
| 4-6 h | 64 / 42 | 2.29 | 0.0% | 7,458 | -0.26 | -2.72 |
| 6-10 h | 18 / 12 | 0.31 | 0.0% | 5,176 | +0.38 | -0.37 |
| 10-14 h | 37 / 18 | 0.00 | 0.0% | 3 | +0.39 | -0.00 |
| **14-18 h** | **18 / 9** | **4.07** | **61.1%** | **18,765** | **-1.32** | **-4.86** |
| > 18 h | 10 / 5 | 0.15 | 80.0% | 4,855 | +0.45 | -0.18 |

Below 14 h the label is the schedule gap and the rule is right. From 14 h up, most
rows are day-shifted instead: the label is `86,400 + a normal taxi` (median 87,598 s),
while the rule predicts the gap (median 60,599 s). The best constant on the 14-18 h
cell is worth **-1.32 s CV**; perfect separation is worth -4.86 s.

**Ranking exposure is 4 rows.** The whole cell is `LIRF, no flight ID, gap 14-18 h`,
and `ranking.parquet` contains four of them (0.0012%); the >18 h cell has two.
Because ranking rows carry ~3.6x the per-row weight of pooled CV rows, four rows
decided correctly or incorrectly are worth roughly ±5-10 s of leaderboard score.

That is the honest shape of it: a 61/39 coin flip on four rows, not an improvement
in the model. It is also already-trodden ground — the day-shift hedge shipped as
v37 and moved the leaderboard by -0.29 s.

## Conclusion

The recoverable-label well is dry. The taxonomy in §1 accounts for every row: the
copy classes that remain unexploited are either harmless (AOBT_3, LOBT, EOBT_1
copies sit at RMSE 138-235 because the copied timestamp is close to the real
pushback) or undetectable (schedule copies, base rate <= 6% outside Rome). The
25.75% of squared error in badly-missed rows is dominated by 2,085 rows that match
no anchor and represent genuine long taxis.

There is no second v33 in these fields. Remaining leverage is either the four-row
Rome day-shift bet, or information we do not currently have.
