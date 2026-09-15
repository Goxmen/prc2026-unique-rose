# `feature_sweep.ipynb` — the map

Read **this file instead of the notebook**. The notebook is 4.5 MB (~510k characters of text plus
3.6 MB of embedded figures); reading it in full costs roughly 300k tokens. This map is ~4k. Open
individual cells when you need them — see "Working with the notebook" in `CLAUDE.md`.

Shape: 180 cells. §N of the sweep = one column of the challenge files, in file order. Section form
is fixed by `SECTION_STYLE.md`; the names each section defines are listed in its §7.

## Reference numbers every section is measured against

Honest holdout: group means fitted on **Jan–Oct 2025**, scored on **Nov+Dec 2025**, cell support
≥ 30 (≥ 200 for high-cardinality levels) falling back to the coarser level, target rows
`PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt BETWEEN 1 AND 3600`.

| rung | RMSE | from |
|---|---|---|
| global mean | 392.5 s | §5 |
| **airport mean** | **357.3 s** | §5 — the baseline every feature must beat |
| + runway | 341.8 s | §12 |
| + `AOBT_3` anchor (20 bins) | **283.6 s** | §9 — the base most later ladders sit on |
| + stand / pier | 265.5 s | §13 |
| + the plan's unfiled delay | 261.0 s | §15 |
| airport mean, **raw labels** (no 1–3600 filter) | 496.5 s | §14 — what the score actually sees |

## The sections

| § | column | what it concluded |
|---|---|---|
| 1 | `MVT_ID_mvt` | join key only; issued in time order (rank corr 0.99), so never a feature |
| 2 | `FLIGHT_ID_mvt` | missing ⇒ the whole `_flt` block is missing (1.08% of training departures, 1.53% of ranking). The is-missing flag is a feature |
| 2a | | unmatched IFR: 22% fly to Ercan (`LCEN`), where no plan can exist; the rest are matching failures of delayed flights |
| 2b | | exact-twin recovery → `recovered_flt_blocks.parquet` (892 + 226 rows) |
| 2c | | relaxed twin waterfall A/B/C → `recovered_flt_relaxed.parquet` (1,007 + 253); flight-number history imputation → `imputed_flt_static.parquet` (25,329). 265 predictions stay bare |
| 3 | `FLIGHT_mvt` | heavy-tailed identity; 80.7% of 2026 departures reuse a 2025 number. Per-number means 357.3 → 337.6. Needs fallbacks, not raw one-hot |
| 4 | `FLIGHT_RULE_mvt` | 99.85% `I`; `V` is two hubs' special-mission fleets; keep `Y`/`Z` distinct |
| 4a | | the 689 nulls recovered → `rule_recovered.parquet` (773); `FINAL` is the precedence expression |
| 5 | `ADEP_mvt` | the anchor feature: ten levels, never missing. 392.5 → **357.3** |
| 6 | `ADES_mvt` | destination: 357.3 → 339.4 (country prefix) → 336.4 (full code) |
| 6a | | sentinels ∅/`ZZZZ` → `dest_recovered.parquet` (243); `OFF_AIRPORT` ≠ `UNKNOWN`; `FINALD` |
| 7 | `PHASE_mvt` | not a feature — but the leak map: **only** `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blanked, and only on ranking departures. The whole arrival stream is fair game |
| 7a | | EHAM's rolling taxi-in is a *runway-configuration* sensor, not congestion (corr −0.28); keep rolling stats per airport |
| 8 | `MVT_TIME_UTC_mvt` | half the target (takeoff − off-block = label). Local hour 357.3 → 351.5; a window **centred** on takeoff, 15–30 min, beats a trailing one (−5.5 s) |
| 9 | `BLOCK_TIME_UTC_mvt` | the hidden endpoint. `AOBT_3` is the nearest visible clock (median 52 s) but its offset is per hub and per season; departures' off-blocks are a **leak** |
| 10 | `SCHED_TIME_UTC_mvt` | the only pre-day clock; the hour read on both clocks 342.3 |
| 10a | | **the corruption gauge** — `LIRF` + no plan + takeoff−schedule ≥ 2 h ⇒ predict the gap: raw-label 496.5 → 391.4. The largest lever the sweep found |
| 11 | `AIRCRAFT_TYPE_mvt` | 283.6 → 274.5; `typ_wake` lookup fills the plan's wake class; `ROTORCRAFT` list |
| 11a | | `TYPE_FINAL` = the plan's code first (Schiphol changed its 777F spelling in Nov 2025) |
| 12 | `RUNWAY_mvt` | 357.3 → 343.5 alone; its value is the pair with the stand (306.0) |
| 13 | `STAND_mvt` | largest step after the anchor: 283.6 → 265.5. The prefix (pier) is the fallback where 2026 uses new stands |
| 14 | `TAXITIME_SEC_mvt` | the target. Train on every label **≥ 1 s, no ceiling**; squared loss on the raw label; a few dozen tail rows decide the score |
| 15 | `LOBT_flt` | the unfiled delay `AOBT_3` − `LOBT`: 293.5 → **261.0** |
| 16 | `CALLSIGN_flt` | the callsign's 3-letter prefix **de-anonymises `AIRCRAFT_OPERATOR_flt`'s hash** (V 0.9966). The operator is 357.3 → 336.0 and 283.6 → 275.1, but only 0.8 s once §13's pier and §11's type are in → `operator_token.parquet` (25,235) |
| 17 | `ADEP_flt` | a copy of §5's `ADEP_mvt` on departures — 0 disagreements in 2.4 M rows — and worse as a key (357.7 vs 357.3). Its worth is three guards: the matched-plan flag is `ADEP_flt IS NULL`, not `FLIGHT_ID_mvt IS NULL` (53 departures carry an id over an empty block); a recovered `_flt` block must start at the departure's own airport (it rejects 3 of §2b/§2c's 2,378); arrivals disagree on 1.19% so the origin key there is `ADEP_mvt` |
| 18 | `ADES_flt` | not a copy: 1.42% of departures name a different airport than `ADES_mvt`. Two causes — 7,399 rows use a code the plan's registry never uses (`ETNB`/`EDDB`, `LSSB`/`LFSB`), 21,622 name a different airport (the movement records a leg further down the rotation). **The plan is right 1,908 of 1,908** times its landing can be checked against the arrival row — but it buys nothing: 335.1 vs `dest_final`'s 335.2, and the disagreement flag is a cargo-apron proxy worth −0.1 s on §13's stand base. Read `ADES_flt` where a feature needs geography, not as the category. **None of the disagreement is ours**: all 2,378 recovered blocks carry the departure's own destination, and cannot do otherwise — the mirror of §17's three bad origins |
| 19 | `ADES_FILED_flt` | the destination **as first filed**. Present on exactly the rows `ADES_flt` is present on (0 rows carry either without the other), and differs on 0.100% of departures — and that difference is a diversion. The referee settles the pair: on the 551 diversions with a findable landing the aircraft landed where `ADES_flt` said **551 of 551** and where `ADES_FILED_flt` said **0**. So `ADES_flt` is a field the data rewrites *after* takeoff, and the filed code is the one that was knowable at off-block — read it wherever a geographic feature is meant as a *cause* of taxi-out. It also closes **4.79%** of §18's 29,160 disagreements (there the movement kept the filed code and the plan moved; the other 95.10% is §18's later leg). Worth nothing at the taxiway: the diverted flag is 335.2 → 335.1 s and 274.0 → 274.0 s on §13's base. Its arrival side — flights diverting *into* a hub, the only such signal in the data — was built and measured rather than argued away: **274.1 s**, an event of ~0.55 per hub per day whose per-hub rate varies fifteen-fold on filing practice (LFPG 0.010%, LEMD 0.155%) |
| 20 | `MARKET_SEGMENT_flt` | the operator's commercial class — eight levels, **not** the nine `data/FEATURES.md` lists (`Low-Cost`, `Business Aviation`, `All-Cargo`, `Charter`, `Not classified` never appear; the doc needs fixing). Missing **exactly** when the plan is missing — 0 of 4,811,570 plan-carrying rows lack one — so its nulls are §2's flag, not new information. Worth 357.3 → 350.7 s alone, but on raw labels its apparent 26.5 s splits **21.5 s to §2's missing-plan flag and 5.0 s to the segment** (the no-plan rows run over an hour 4.892% of the time, p99 10,254 s). On §13's base it is **0.2 s**: the hubs pre-sort segments onto piers (EHAM `H` 100% `Lowcost`, LFPG `J` 94.6% `Regional`, LTFM `K` 88.7% `Cargo`). Not the operator either — 91.42% of rows sit in their operator's modal segment, and the mainline/regional split inside a flag carrier is worth 239 s at DLH. **No derived file**: the 5,290 bare 2026 departures could be filled from operator history (4,110 of them) but the map is 91.42% pure and the column is worth 0.2 s |
| 21 | `IOBT_flt` | the plan's **initial** off-block — `LOBT_flt`'s value before any revision, present on **exactly** the same rows (no departure in either file carries one clock without the other) and equal to it on **99.45%** of departures. On the 11,267 that differ the *initial* clock is the one closer to the actual off-block, **73.9%** of the time (mean error 11.4 vs 14.3 min, within 5 min 40.0% vs 11.6%): **the refile moves the plan's clock away from the pushback**, and most so where refiling is common — LEMD 87.8%, LEBL 83.9%, LTFM 74.8% on 2.09% of its departures; LIRF (41.8%) is the only hub where it helps. Refiled *earlier* it is wrong 86.8% of the time. Across the whole file that edge is 1.2 s of clock accuracy (8.57 vs 8.59 min) and on the ladder 260.4 against §15's 261.0, so **no pipeline change**: 0.6 s is inside one holdout's resolution and §15 keeps its feature. No derived file; the matrix reads the twin layers' clock as `TWINSI` |
| — | closing matrix | every covered column against every other; `assoc_base` + `FEATURES`, then a reading table |

## Derived files (all gitignored, all joined on `MVT_ID_mvt`)

`data/derived/`: `recovered_flt_blocks` 1,118 · `recovered_flt_relaxed` 1,260 · `imputed_flt_static`
25,329 · `rule_recovered` 773 · `dest_recovered` 243 · `operator_token` 25,235.

They are **not in git**. Rebuild them by running the notebook's own cells 15, 18, 19, 35, 53, 54 and
§16a — about 30 s. If your counts differ from the numbers above, something upstream changed and the
findings tables are stale; stop and say so.

## Still open

- Roadmap in cell 0: 21 of 30 columns done. Next in file order is `FLIGHT_RULE_flt`.
- **§15 points at "§16" for `IOBT_flt`, twice** — the column is §21 (the numbering moved when `CALLSIGN_flt`
  took §16). Left as written on purpose: §21 stays in its own lane and does not edit §15. Fix both references
  when §15 is next touched.
- **§19 and §21 ran the same referee on a `_flt` pair and got opposite answers**: the destination's *later*
  value was right 551 of 551, the off-block clock's *earlier* value on 73.9%. Where a plan field has an initial
  and a current form, ask which one is true — do not assume the latest is.
- **`data/FEATURES.md` is wrong about `MARKET_SEGMENT_flt`'s levels** (§20). The fix was flagged, not
  applied — it is a tracked, shared doc and deserves its own commit.
- §17 and §18 together: the two systems agree perfectly about **the hub end** of a flight (a
  departure's `ADEP`, an arrival's `ADES`, both 100.000%) and disagree about the far end on ~1.2–1.4%
  of rows. Expect the same shape wherever a `_mvt`/`_flt` pair names an airport. §19 then split that far-end
  disagreement in two: 4.79% of it is a diversion the movement was never told about, the rest is the
  movement naming a later leg.
- **§17 recommended a guard §2 has not applied**: §2b/§2c's twin join should require the source
  arrival's `ADEP_flt` to equal the departure's `ADEP_mvt`. Applying it drops `recovered_flt_blocks`
  to 1,116 and `recovered_flt_relaxed` to 1,259 and moves every count downstream, so it was left for
  §2 to do deliberately rather than reached in from §17.
- `ROTORCRAFT` (§11) is meant to supersede `HELI_ALL`, but §12 and §13 still use `HELI_ALL`.
- **Some cells are not reproducible run to run.** Cells 94, 95, 96, 102 and 103 print different row
  *orders* on each execution — `mode()`, `arg_max()` and unordered `LIMIT`s break ties arbitrarily.
  The values are stable; only tied rows move. Anything a findings table quotes needs an explicit
  tie-breaker (rule 5 in `CLAUDE.md`); §16's cells have them. Worth fixing in §11/§11a when that
  section is next touched — until then, do not re-splice those cells' outputs just because they
  changed, and do not quote a tied row from them.
