# Next steps

**Active target: 205.00 seconds or lower.** Best/latest is **v79 at 219.6130**,
leaving **14.6130 seconds**. Live check: **12 October 02:40:30 Amsterdam**,
rank **9/242**, all **344,841 flights**. Autonomous submissions and both machines
remain authorized, with one heavy job per host. Four October 12 UTC submission
slots are used; **one remains for a new candidate after experimental feedback**.

| Candidate | Change | Jan/Jul CV | Official RMSE |
| --- | --- | ---: | ---: |
| v76 | v75 adds public daily inputs to the main tree and physical networks. | 292.4494 | 221.0948 |
| v77 | v76 adds the selected daily and tree refinements. | 291.5497 | 220.8328 |
| v78 | v77 adds a joint public taxi-count, taxi-sum and departure-delay correction. | 290.9301 | 220.2933 |
| v79 | v78 is recombined with earlier scored predictions using updated affine weights. | 290.6744 | 219.6130 |

V76, v77 and v78 uploaded once at 02:10, 02:20 and 02:30. After their feedback,
the new 51-vector diagnosis excluded unsubmitted v75 and used only actual
successful scores. V79 uploaded at 02:36 after independent geometry and final
file audits; its remote hash matches. These four submissions improve v74 by
**2.0751 seconds**. The v79 weights are selected using the leaderboard;
its Jan/Jul change of −0.2558 has interval [−0.6214, +0.0491] on reused development
data. Do not describe that CV result as independent confirmation.

**V75 remains cancelled and must not be submitted.** Its prepared changes are
included in v76. The old 50-/51-/52-vector evaluation queues and automatic
fifth-slot coordinator stay cancelled; they require v75's unavailable score.
Preserve their artifacts and do not impute that score or restart them.

Research resumed only after the user's 02:24 instruction. The 02:00 timer fired,
but its IDE message was rejected because the request omitted `threadId`;
no inference ran from the timer. Its failed receipt is retained. No replacement
timer is currently scheduled. The native goal object remains `usageLimited`
with its old text; no native limit or status was changed. The local project
objective records the user's current 205 target.

Current work:

- **Laptop GPU:** full-input physical-track TabM (168 tabular inputs, 32 heads),
  seed42, fixed 21/17 epochs, against same-laptop original-network controls.
  Four fits run sequentially; preflight and independent loss/update/stack review
  pass. See `v3_20261012_nn_public_tabm/`. No HPO or extra seeds are authorized
  by this design; preserve completed checkpoints and preprocessing.
- **Desktop:** fixed cohort-context January/July fits started at 03:40 Amsterdam,
  PID1337640, after independent source and CUDA audits passed. Linear-leaf
  full-row probes also pass: 7.159 GiB cumulative peak memory and 1.14× late-tree
  cost. The fixed four-fit matched linear comparison passed independent review
  and is frozen; it will follow after the desktop is released.
- **Timestamp features: closed.** Exact-v78 CV worsens0.012388seconds,
  interval[−0.004784,+0.027121]; frozen-v79 transport worsens0.017603seconds,
  interval[+0.000630,+0.032125]. Both original controls, saved candidate model
  reloads and all344,419 final score rows pass. Retain fits; no production.

The phase inputs are all missing for Rome's no-ID 1–4h group, so they cannot
directly resolve that costly group. Arrival landing-clock context passed its
arrival-source pilot but failed the necessary departure transfer check; the
same-flight endpoint pilot also failed. Both are closed without a taxi-model fit.
All-discrete Sobol and Gibbs public-source inference failed convergence; no CV
score or candidate follows. Reapplying existing bounds to v79 changes only nine
ranking rows and can improve official RMSE by at most 0.088502 seconds, so it is
also closed. A new cross-flight context encoder has passed an input-only
four-date coverage pilot and is authorized for one fixed comparison, pending
implementation audits and an allocated host. No fit or gain is claimed yet.

See [the current research report](VALIDATION/reports/TARGET205_RESEARCH.md).
The [11 October report](VALIDATION/reports/TARGET215_RESEARCH.md) retains the
historical queue decisions and completed screens. Recent closed branches follow.

The fixed learned attention-pooling screen is complete and closed. Replacing
one of three pool members changes exact-v78 CV by−0.0109 seconds, interval
[−0.0951,+0.0610]; the matched single-member comparison worsens0.0672.
Both fits, full saved-model reloads,1,094 independent model rows and all344,419
SQL CV rows pass their audits. This is inconclusive; no extra seeds, production
or submission follow. Preserve the completed models and receipts.

The fixed non-Rome extension is complete and closed: CV290.930119→290.905917,
change−0.024202, interval[−0.073983,+0.017770]. Both months improve slightly,
but the result is inconclusive. The laptop completed inference in219.5seconds;
all317,891 inputs,9,966 calibration cells,344,419 score rows and219 scalar
messages pass audits. Native priors, scope gates and unchanged Rome rows are
verified. No production or submission follows.

The uncertain-calendar joint-source experiment is also complete and closed.
Integrating Rome's possible UTC off-block days changes CV by+0.002117seconds,
interval[−0.030178,+0.040981]; January−0.000581 and July+0.004815.
Laptop inference takes52.2seconds and reproduces the old controls. Numerical
quadrature, all92,386 support-state calendar assignments, all scope gates,
58 independent scalar messages and all344,419 SQL score rows pass. Only10
targets have support on multiple days, although uncertain peers can affect
other targets. No expansion, production or submission follows.

The explicit-neighbour public-source screen is complete and closed. Retaining
one largest-variance peer's full discrete support changes v78 CV by−0.0500,
interval[−0.3854,+0.1814]; January worsens0.0466 and July improves0.1467.
The laptop completed both months with exact original controls and row-order/
label-poison invariance. All2,336 peer memberships and all344,419 CV rows pass
SQL checks;148 independent scalar peer/density checks agree within7.3e-11seconds.
The result is inconclusive; retain v78, with no production or submission.

A fixed recurrent track screen is complete and closed. One forward GRU of width 32
summarizes the existing local physical states through a zero-initialized
projection; all inputs, original weights, seed 42 and 21/17 epochs are retained.
CPU and both GPUs pass initialization, padding, gate-equation and gradient
checks. All sequence-index bounds and 2,096 raw examples pass the input audit.
Actual training used 4.754 GiB. Both fits, full saved-model reloads and independent
model/SQL checks pass. Through exact v77, matched CV worsens0.0628 and replacing
one of three pool members worsens0.0014, interval[−0.0778,+0.0746]. There is no
useful gain; retain the models without further seeds, production or upload.

Increasing the physical pool's internal neural share from 0.5 to 0.75 loses
0.1683 seconds on exact-v77 CV, interval [+0.0617, +0.3186]. The all-row audit
passes; retain the fixed 0.5 share without a new submission or weight search.

Time-weighted pooling is complete and closed: matched CV+0.1153 and
one-of-three replacement+0.0286 are inconclusive, with both months worse.
All original/new model reloads and independent model/SQL checks pass.
No production or upload follows.

The continuous Rome mixture-coefficient screen is complete and closed. Six
same-laptop fits retain the original inputs and population; all six models
reload exactly on the desktop. The continuous target worsens v77 CV by0.1588
and the matched Logloss control by0.1711, with both months worse. Full-stack
replays and independent SQL audits pass. No production or upload follows.

Removing coefficient limits while accounting for official-score uncertainty
adds only0.0061 seconds of projected gain over the bounded recipe on the same
48 scored vectors. Its serialized upper bound is221.8268, above current best.
No candidate or upload was selected. This historical screen used only then-scored
vectors; the queue-pruning instructions above supersede its old watcher plan.

The Gaussian observation-variance screen is complete and closed: matched CV
worsens4.6579 seconds and one-of-three replacement worsens0.6469 seconds,
interval[+0.3594,+1.1554]. Both months worsen. All original/new mean and variance
reloads and independent model/SQL audits pass. Retain both models; no production,
loss tuning or submission follows this result.

The early-public neural projection screen is complete and closed: matched
CV−0.1786 and one-of-three replacement−0.0739 are inconclusive; January worsens
0.0794 in the pool while July improves0.2273. Both original/new full held
reloads and the independent model/SQL audits pass. Retain both models, without
production expansion or a submission row. Its corrected pre-fit CUDA check
and untouched training implementation are recorded in the research report.

Closed directions: inverse public-reference inputs give only −0.0025 versus
v76 and −0.0004 in the refined recipe, both inconclusive; both months/models
and the final all-row model/SQL audit are complete. Daily neural features
(+0.0751 in the pool), daily Rome inputs (+0.1190), wider tracks, track
withholding, neural Rome, punctuality, fourth-tree XGBoost, parameter averaging
and the public-callsign three-member pool failed or were inconclusive.
Retain their results and models; do not repeat or add submission rows.

A label-informed diagnostic of four fixed broad constant-offset families finds
at most 0.081 seconds of CV reduction per family, even when held labels choose
the offset. This only bounds those unrounded constant corrections on CV, not
2026 gains or nonlinear corrections. No candidate, deployment offset or upload
was created. Broad offset probes are deprioritized on this evidence.

All three existing subagents remain unavailable due to the account usage limit.
The root goal also reports `usageLimited`; no replacement agents or account-limit
overrides are authorized. Research is on hold until the requested wake-up. Sources, failures, recovery
receipts and limitations are in the [215-second research report](VALIDATION/reports/TARGET215_RESEARCH.md).
The earlier 225.00 goal is complete; its result remains below.

**Previous goal achieved, 11 October 2026:** v71 is best/latest at **224.3602 seconds**,
**0.6398 below** the requested 225.00 target. It improves v69 by 1.0392 and
v58 by 11.8293 seconds on 344,841 flights; live rank is **15/242** at 02:04
Amsterdam. All training and audits are complete; preserve the retained models.

Physical-scale track pooling passed CV **292.8675 (−0.7891 versus v64)** and
scored **231.9066** as v70. Adding that separately scored direction to v69's
basis produced v71. Its Jan/Jul CV **296.3160** is 0.5037 better than v69 but
remains worse than v64/v70. Weights use leaderboard feedback; this is not
independent validation. Ground-only pooling, weighted Rome calibration and
airport bias corrections did not pass and receive no production fit/upload.
See the [research report](VALIDATION/reports/RESEARCH_NETWORK_20261010.md) and
[submission record](VALIDATION/runs/v3_20261011_physical_affine/submission_v71/manifest.json).
New future candidates still require Jan/Jul CV; older 16-fold validations stay unused.

The table below preserves the September idea inventory and its original estimates.
Its gains are combined-CV seconds against **row 43 (233.2338, uploaded as v40)**,
not estimates against today's model. Some historical statuses are superseded by
the current report and [submission ledger](SUBMISSIONS.md). Negative is better.
Measured historical numbers come from
`.venv/bin/python VALIDATION/tools/estimate_ceilings.py`; "est." is a judgement
from the evidence available when the table was written.

The **#** column numbers the rows in order, so an idea can be named as "idea 3" in
conversation. It is not a priority ranking, and removing a row renumbers the ones
below it, so a number is only reliable within the current version of this file.
Rows 1-13 were ordered by expected gain when the table was written.

🔴 DEAD: tried, worth nothing — nobody spends a second evening on it.
🟡 DONE: the useful part is already in the model, the rest measured and worth
nothing. The colour needs GitHub's maths rendering; the dot and the word carry
the meaning anywhere else.

| # | Idea | What it is | Expected gain | Status&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|---|---|---|---|---|
| 1 | **ADS-B off-block witness at Rome** | Recover each departure's real pushback time from ADSB.lol ground traces at the airports OPDI cannot see, starting with LIRF. See [the section below](#ads-b-off-block-witness-at-rome) | ~~ceiling -15 (LIRF) to -18 (LIRF+EHAM)~~ **unreachable** | 🔴 $\color{red}{\textsf{DEAD}}$ **2026-09-24** — LIRF apron is an ADS-B blind spot; [details below](#ads-b-off-block-witness-at-rome) |
| 2 | Seed averaging | Average the predictions of seeds 42/142/242 instead of shipping one seed; needs two more full-year fits | **-1.30** measured on row 33 (row 36) | CV measured as row 36, never submitted; v42 still ships seed 42 only |
| 3 | **Second model family** | Fit a CatBoost over the same inputs and blend it with the LightGBM stack. We have no second family, and our three seeds correlate 0.983 ([scan](#ideas-taken-from-public-competitor-repos)) | ceiling **-3** at &rho;=0.95, **-6** at &rho;=0.90 (from `estimate_ceilings.py`); zestful-fountain measured **-7.11 LB** for this exact step | Not started |
| 4 | **Per-airport expert models** | One model per departure airport, averaged with the global model, instead of splitting by row type as we do now | est. **-0.5 to -2**; improved all four of zestful-fountain's held-out months | Not started |
| 5 | Congestion component | A dedicated congestion/queue term, and queue inputs interacted with the airport instead of pooled | measured **~0** (was -1 to -3 est.) | 🟡 $\color{orange}{\textsf{DONE}}$ **2026-09-24** — congestion is already row 28's 11 queue inputs; a per-airport term gains in-month, +0.53 to +0.63 worse next month |
| 6 | Taxi occupancy count | Count other aircraft between their own `AOBT_3` and their takeoff, at this flight's off-block and at its takeoff, airport and runway scope | CV **-0.29**, leaderboard **+0.54 (worse)** | 🔴 $\color{red}{\textsf{DEAD}}$ **2026-09-24** — submitted as v41; see row 44. CV -0.29, leaderboard +0.54. Ageing readouts warned; do not refit; **Not worth pursuing**: dead |
| 7 | Neighbour-duration statistics at stand scope | Extend `runway_queue.py` to stand scope, and to standard deviation, own-minus-mean and the next neighbour's wait | est. **-0.5 to -1.5** | Not started — **Not worth pursuing**: the stand-scope ceiling is small |
| 8 | Arrival taxi-in at stand/runway scope | Arrival taxi-in *durations* per airport x stand and airport x runway. v42 already uses them per airport over the last 60/180 min | bounded by the stand&times;runway oracle at 1.90% of SE; zestful-fountain measured **-2.62 LB** at airport and runway scope | Airport scope done in v42 (row 45), tested together with snow inputs; stand/runway scope not started; stand/runway part **not worth pursuing**: small, and v42 already has the airport part |
| 9 | TimesFM-3 airport-hour forecast | A foundation-model hourly forecast of each airport's mean taxi, fed back in as a model input | attacks the **8.25%** airport&times;hour oracle, the largest family we have not reached; needs a GPU | Not started — elegant-alligator's own direct test gained under 0.1 s and was rejected; licence bars commercial use, prize clearance unresolved; **Not worth pursuing** |
| 10 | `EOBT_1` pair inputs | Feed `AOBT_3 - EOBT_1` and `takeoff - EOBT_1` beside the existing delay; the notebook's own ladder moves 265.5 → 237.5 s on them | ~0: already in the model | 🟡 $\color{orange}{\textsf{DONE}}$ **2026-09-25** — row 26's clock inputs carry `AOBT_3 - EOBT_1`, and `takeoff - EOBT_1` is its sum with `takeoff - AOBT_3`; **Not worth pursuing**: already done |
| 11 | "Nothing reaches this row" flag | A flag for no-plan rows that no layer (twins or high-confidence history) identifies; found four separate times in the notebook | **-1 to -2** est. | Not started |
| 12 | Learned rule corrections | Fit a model on `recorded taxi - rule prediction` for rule-handled rows and add the correction back | ceiling **-9.20**, realistic **-0.5 to -2** | Not started |
| 13 | Rule-row separator | Tell apart the 112 rows the four-hour Rome rule gets right from the ~30 it misses by more than an hour | same ceiling; **high variance**, four ranking rows decide it | Not started |
| 14 | Hour × no-ID aggregates | Corruption aggregates at airport × hour × missing-flight-ID level, as model inputs | oracle **-2.42**, honest **-0.62** | Screened 2026-09-24 — no bias signal, duration variant impossible; **Not worth pursuing**: measured small |
| 15 | Live arrival taxi-in witness | Mean taxi-in of arrivals that in-blocked in the previous 60 min; arrivals keep their labels in `ranking.parquet` | est. **-1 to -3** before testing | Screened 2026-09-24 — model already unbiased, not pursued; **Not worth pursuing**: measured small |
| 16 | Airport-day corruption aggregates | Day-level dirty-day shares as model inputs | oracle **-0.53**, honest **-0.44** | Tested — rows 31/32, ~0, not shipped; **Not worth pursuing**: measured small |
| 17 | Recurring bias corrections | Shrunk per-airport / per-hour / per-stand residual offsets fitted inside each training window | oracle **-1.44**, honest **-0.44** | Not started — **Not worth pursuing**: honest estimate only -0.44 |
| 18 | Rome 4–6 h cell hedge | Probability-weighted mean instead of the hard rule on the one genuinely mixed cell | measured | Tested — inconclusive, not shipped; **Not worth pursuing**: measured small |
| 19 | Ensembling scored recipes | Blend two of our own prediction stacks | **+0.69 to +1.55 (worse)** | Tested — rejected; **Not worth pursuing**: measured small |
| 20 | Aircraft type coalesce | `coalesce(plan, twins' plan, movement)` for the aircraft type in `pipeline/common.py` | ~0 on scored labels | Not started — cleanup and bookkeeping, about 0 s |
| 21 | Twin-join `ADEP` guard | Require the source arrival's `ADEP_flt` to equal the departure's `ADEP_mvt` in the twin join; moves every derived count | unknown, correctness | Not started — cleanup and bookkeeping, about 0 s |
| 22 | `rule_final` in the pipeline | Use the cleaned flight rule instead of raw `FLIGHT_RULE_mvt` | ≤0.3 s | Not started — cleanup and bookkeeping, about 0 s |
| 23 | `wake_lk` replacing `typ_wake` | Key the wake class on the plan's aircraft code, not the movement's | ~0 | Not started — cleanup and bookkeeping, about 0 s |
| 24 | `ROTORCRAFT` replacing `HELI_ALL` | Finish the helicopter-flag replacement in the two sections still using the old one | ~0 | Not started — cleanup and bookkeeping, about 0 s |
| 25 | Month-balanced aggregation | Give July–December 1/6 each in the validation overview instead of the current mix | 0 direct | Not started — proposed, never installed; cleanup and bookkeeping, about 0 s |
| 26 | January weighted-CV coverage | Extend the weighted-CV transport test to January; it currently uses five H2-2025 months only | 0 direct | Not started — cleanup and bookkeeping, about 0 s |
| 27 | Leakage-check verdict | Replace the hard-coded conclusion in `check_rich_feature_leakage.py:101-105`; use whole-day holdouts | 0 direct | Not started — cleanup and bookkeeping, about 0 s |
| 28 | Training-vs-ranking identification | Finish the investigation into what separates training rows from ranking rows | unknown | 🟡 $\color{orange}{\textsf{DONE}}$ **2026-09-21** — steps 1–4 complete; its January snow lead became v42's inputs, no departure-label drift proven; cleanup and bookkeeping, about 0 s |
| 29 | `FEATURES.md` corrections | Fix the `MARKET_SEGMENT_flt` levels and the claim that military movements were removed | 0 direct | Not started — cleanup and bookkeeping, about 0 s |
| 30 | Notebook cross-references | Fix the section numbers §9/§14/§15/§16 point at, and move §16a's cells into §26 | 0 direct | Not started — cleanup and bookkeeping, about 0 s |
| 31 | Reproducible tie-breakers | Add explicit `ORDER BY` tie-breakers to cells 94–96 and 102–103, whose row order changes between runs | 0 direct | Not started — cleanup and bookkeeping, about 0 s |
| 32 | Public GPLv3 repo | `gh repo edit Goxmen/PRC --visibility public`; `LICENSE` is already GPLv3 | 0 direct | Not started — about 0 s, but **required** for prize eligibility before 11 October |
| 33 | OPDI confirmation | Confirm directly with the organisers that open ground-track data may be used | 0 direct | Not started — only second-hand: the organiser OK'd OPDI and ADSB.lol on Discord, quoted in kind-mango's repo; cleanup and bookkeeping, about 0 s |
| 34 | **X-Plane Scenery Gateway stand coordinates** | Stands with lat/lon from one X-Plane Gateway `apt.dat` per airport, giving stand-to-runway distance and apron geometry. We have **no stand coordinates at all** today | unknown; the stand&times;runway oracle bounds this family at 1.90% of SE | Submitted **2026-09-25** as v43 (row 46): LB 274.7660, −0.14 on v42; Weighted CV −0.47 ([report](VALIDATION/reports/STAND_COORDINATES_EXPERIMENT.md)) |
| 35 | **ADSB.lol as surface traffic, not a pushback witness** | The idea-1 traces read as ground *traffic* instead of an off-block time: taxiway occupancy, threshold queue length, runway in use | unknown | Not started — **Not worth pursuing**: a large ADS-B pipeline, and the Rome test found no reception; rank 3 uses adsb.lol, 2.2 s off the lead |

## ADS-B off-block witness at Rome

The one idea in this file sized to the gap to the leader. Everything else here is
worth 0 to 3 seconds; this is worth 15 or nothing.

### Why

`VALIDATION/reports/` and the leaderboard history give a stable relation: official
score = combined CV + 42.8 ± 1.5 s, and that offset is the test set's month mix
(56% July), not a penalty on hidden labels. The leader's 234.14 therefore implies a
combined CV near 197 against row 43's 233.23 — about 37 s, or a quarter of all
squared error. No feature in the table above is that size; OPDI, the largest recent
addition, was 2.47 s.

OPDI's gain turned out to be a pure dose-response in one column: whether the ground
tracks showed a flight *leaving its own stand*. Zurich, at 39.3% own-stand coverage,
gained 36.95 s of airport RMSE. Matching a flight (90–97% nearly everywhere) and
seeing it somewhere on the ground (23–80%) were both worth nothing. The five airports
at **0.0% own-stand coverage carry 74.8% of all squared error**, so OPDI paid only at
Zurich and Frankfurt — 7.6% of the error. The witness is the edge; OPDI just fails to
deliver one where it matters.

### What was probed on 2026-09-24

ADSB.lol is a different receiver network, so it is not bound by OPDI's coverage.
Historical traces are at
`https://globe.adsb.lol/globe_history/YYYY/MM/DD/traces/<last2hex>/trace_full_<icao24>.json`
(302-redirects to `adsb.lol`; the body is gzipped JSON with no `Content-Encoding`
header, so decompress it manually). One file is one aircraft-day, 100–300 KB. The
`icao24` comes from the OPDI flight lists already on disk under
`data/external/opdi/flight_list/`, which cover 2025-01 to 2026-07 including both
ranking months.

Departures with ground points before takeoff, three 2025 days, about 90 flights per
airport:

| airport | % of squared error | OPDI own-stand | ADSB.lol ground points |
|---|---:|---:|---:|
| **LIRF** | **31.6** | **0.0%** | **96.6%** |
| LSZH | 2.2 | 39.3% | 91.1% |
| LFPG | 13.6 | 0.0% | 45.6% |
| EGLL | 10.7 | 0.0% | 15.4% |
| LTFM, EDDM | 18.5 | 0.0% | 0% — blind |

Rome is 96.6% visible on the ground where OPDI sees nothing, with a median of 78
ground points per departure. LTFM, EGLL and EDDM stay blind, so 36% of the squared
error is out of reach by this route.

### What is not solved

Turning those points into an off-block. Two detectors were tried and both are wrong:

- *First ground point in the three hours before takeoff* catches the *previous
  rotation's arrival taxi-in*. LSZH gives a lead/taxi ratio of 1.24, which is
  impossible for a departure. Correlation with the recorded taxi is -0.17 at LIRF.
- *Last stationary stretch before takeoff* catches the runway hold, because aircraft
  stop repeatedly while queuing — median "taxi" 92 s. Requiring a stop of at least
  300 s gives LIRF a correlation of 0.52 but a median error of -598 s and only 8.3%
  of flights within two minutes; LSZH -316 s and 30.7%. OPDI reaches about 56% within
  two minutes at Zurich, so a real extractor should do much better.

A usable extractor needs OSM stand polygons and dwell detection — what OPDI does,
applied where OPDI has no data.

### Steps

1. Build a proper pushback extractor and validate it against `LSZH` 2025, where the
   labels are clean and OPDI's own-stand figure gives a target to beat (~56% within
   two minutes). Do not fetch anything at scale until it clears that bar.
2. Fix the callsign join. `FLIGHT_mvt` is the IATA flight number and OPDI's `flt_id`
   is the ICAO callsign; a naive equality join matched **zero** EHAM and EDDF rows.
3. Find a second `icao24` source for 2–9 January 2026. OPDI's 2026-01 flight list is
   missing those days, which is exactly the EHAM anomaly window in the test set.
4. Fetch LIRF first: roughly 450 aircraft-days per day over about 244 days, so ~110k
   files and ~22 GB, several hours at four parallel requests. Be polite to the server.
5. Build the features in the shape that paid for OPDI — seconds before takeoff of
   leaving the stand — and screen on A folds only, per the stale-model warning in
   `VALIDATION/reports/` about residual stacks on pooled A+B folds.

### Outcome, 2026-09-24: the witness cannot be built at Rome

Both open questions were settled. The callsign join is fixed; the extractor is
built and validated; and Rome turns out to be unreachable.

**The join was a whitespace bug.** OPDI's `flt_id` is space-padded, which is the
whole reason a naive equality join matched zero rows. With `trim()`, plus a
one-day window because `dof` is not the takeoff date, match rates are 96.7% LSZH
and 98.4% LIRF. Validated against three fields the join never uses: aircraft type
agrees 98.3% / 97.5%, destination 96.8% / 96.0%. `VALIDATION/tools/adsb_join.py`.
The production `VALIDATION/opdi.py` join was independently checked and is correct
(type agrees 98.2-99.8% at all ten airports); its LTFM rate of 54.6% is OPDI
coverage, not a defect.

**Stationary aircraft emit nothing.** ADSB.lol records a point when something
changes, so a parked aircraft is a *gap* in the trace, not a run of stationary
fixes. This is why "last stationary stretch" found the runway queue: queueing is
the only stop that keeps producing points. Detecting the dwell gap instead, and
requiring it at the movement record's own stand, reaches **56.1% within two
minutes at LSZH** (median -25 s) -- parity with OPDI's ~56% there. Restricting to
the own stand is the entire signal: any-stand dwells score 32.0%. Back-extrapolating
from the resume fix by distance over speed adds nothing (34.5%).

**Rome's apron has no ADS-B reception.** Measured on 1,301 LIRF and 1,130 LSZH
departures over 2025-03-12, 2025-07-16 and 2025-10-08:

| | LSZH | LIRF |
|---|---:|---:|
| Ground fixes per departure, median | 125 | 38 |
| Share of fixes within 200 m of a stand | 85.9% | 22.1% |
| First fix after off-block, median lag | 176 s | 624 s |
| Its distance to the nearest stand, median | 88 m | 250 m |
| Dwell detected at the own stand | 28.6% | **4.3%** |

By the time Rome's receivers hear a departure it left the stand ten minutes
earlier and is 250 m away. The 2026-09-24 probe's "96.6% of LIRF departures have
ground points" holds, but those points are on taxiways: visible on the ground is
not visible at the stand. Two independent datasets agree -- OPDI's `first_seen`
lands a median 33 s after off-block at LSZH and 1,052 s at LIRF.

Where the detector does fire at Rome it is as accurate as at Zurich (53.6% within
two minutes, n=56), so this is a coverage wall, not a modelling failure. There is
no second receiver network to try: ADSB.lol was already the alternative to OPDI.

Costs, for anyone reconsidering: 910 aircraft-days took 12.0 minutes at three
parallel requests, which puts a full LIRF fetch near 27 hours rather than the
"several" estimated above, for about 13 GB gzipped (82 KB mean per file, measured,
against 472 KB decompressed). Do not decompress to disk.

Tools left behind, all working and reusable at any of the ten airports:
`adsb_join.py`, `adsb_fetch.py`, `adsb_geometry.py`, `adsb_pushback.py`,
`validate_pushback.py`. OSM geometry is cached for LSZH and LIRF.

### Rules

ADSB.lol is ODbL 1.0 and the organiser has approved open trajectory data if declared.
OpenSky state vectors and the Trino historical database are **not** allowed. The
declaration obligation is real: record the source in `SUBMISSIONS.md` alongside the
OPDI entry. This does not change the standing item above about confirming ground-track
use directly with the organisers.

## Ideas taken from public competitor repos

Scanned 2026-09-24. Of the 35 teams above us, **three** have published model code and
**none of the top ten** has. Team pseudonyms were mapped to real identities through
`euctrl-pru/prc_data_challenge_website_2026`, which carries a `teams/<name>.qmd` page
per registered team; every name above us was then searched by code search and by
GitHub user search on its named participants.

| Rank | Team | Score | Repo |
|---|---|---:|---|
| 12 | quick-boat | 254.52 | `arquicanedo/prc-challenge-2026` — leaderboard viewer only, no model |
| 25 | kind-earthquake | 268.52 | `col3name/air-data` — CatBoost baseline, rolling traffic counts, leave-one-out target encodings |
| 26 | elegant-alligator | 268.56 | `javidmardanov/PRC-Data-Challenge-2026` — TimesFM-3, seasonal CatBoost blends, arrival stand/runway context |
| 33 | zestful-fountain | 273.56 | `Phoenix-Ops-LTD/prc2026-taxiout` — CatBoost/LightGBM ensemble, per-airport experts, arrival and neighbour features |

### Where the expected gains come from

zestful-fountain documents every submission with its official score, so each idea can
be priced from their ladder. These are leaderboard seconds, not CV; the two are close
here because our own CV-to-LB offset is a near-constant +42.8 &plusmn; 1.5 s.

| Step | Change | LB gain |
|---|---|---:|
| v6 &rarr; v7 | 26 arrival taxi-in features, airport and runway, 15/30/60 min | **-2.62** |
| v7 &rarr; v8 | CatBoost depth 9, 4,999 trees, blended 50/50 with v7 | **-7.11** |
| v8 &rarr; v9 | Per-airport experts plus 30 neighbour features | **-0.88** |
| v9 &rarr; v10 | The same neighbour features inside the CatBoost | **-2.58** |
| v10 &rarr; v11 | A fixed clock-overlay rule on 31 rows | **-0.20** |
| v11 &rarr; v12 | Following-context features, 183 in total | **-1.42** |

Their local validation is far worse than ours — reused January/July RMSE around 313
against our combined CV of 233 — and they still reach a better official score. The
single largest step in the whole ladder is model capacity and blending, not a new
signal.

### What they do that we already have, or have already ruled out

- **The &plusmn;3,606 s bound on `takeoff - LOBT`.** elegant-alligator clips every
  prediction to it. We found the same bound independently as the plan window (v33,
  -27.5 s LB). No gap.
- **A Rome/LIRF specialist.** All three have one; ours is further developed.
- **Weather.** We carry 21 inputs; zestful-fountain has none at all.
- **Open ground-track data.** Only we use OPDI.
- **A missing-timestamp mixture of experts.** elegant-alligator's works where our
  Stage 4 M1 regressed. The difference in shape is worth noting: theirs is a delta
  blend with a fitted alpha per region (Rome 0.42, elsewhere 1.0), not a three-class
  classifier mixture.
- **Airport-hour and month-level arrival taxi-in witnesses.** Both screened and dead;
  only the stand/runway-scoped variant in the table above is untested.

### Datasets the teams above us use

Stated by the teams themselves, so this is what to try where we have no equivalent.

| Dataset | Who | Do we use it? |
|---|---|---|
| ADSB.lol ground traces | rank 3 (gentle-tractor, 236.31) | Probed only. Idea 1 fetched it and found no stand witness at Rome; **never tried as traffic** — idea 35 |
| X-Plane Scenery Gateway `apt.dat`, ten airport packs, stands + coordinates | rank 3 | Yes, with OpenStreetMap — idea 34, row 46, v43 |
| METAR, Iowa Environmental Mesonet | us (row 27, 21 inputs); elegant-alligator (`docs/weather_notes.md`) | Yes, same archive |
| OPDI ground tracks | us | Yes; no competitor uses it |

Rank 3 is the team page whose one listed member is HOANG ANH THONG LE, ACV Vietnam,
matching the "Im Le (currently as rank 3)" message on 2026-09-24; rank read from the
live standings the same day. zestful-fountain uses no external data at all.

### What this does not explain

The gap to the leader is 41.27 s. The best public repo above us is 1.9 s ahead, and
its methods are ones we have covered or bounded. Nothing here accounts for a 234, and
the two largest items above — second family and per-airport experts — plausibly buy
3 to 8 s together, which is rank ~20, not the podium. The ADS-B witness remains the
only idea in this file sized to the gap.
