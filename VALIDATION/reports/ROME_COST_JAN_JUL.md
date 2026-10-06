# Rome classifier v2: cost weights and a flight-number input (6 October 2026)

Runner `VALIDATION/rome_cost_jan_jul.py`, run `VALIDATION/runs/v3_20261006_rome_cost_jan_jul/`.
All arms use R20's settings (the v53/v54 Rome classifier) and are scored on Jan/Jul CV with C4's
parts at the same seed; only the Rome probability changes. Three seeds; each arm's checkpoint is
the best of its seed-averaged curve. Seed 42 of the control reproduces R20_s42 exactly.

| Arm | Change | Jan/Jul CV | vs control | Seeds better |
|---|---|---|---|---|
| C | R20 as in v54 | 296.5761 (c2000) | – | – |
| W | training flights weighted by the cost of a wrong copy call | 297.9490 (c200) | +1.37 [−0.23, +2.85] | 0/3 |
| **F** | **+ flight number's 2025 copy share and count** | **295.0409 (c1700)** | **−1.54 [−2.44, −0.75]** | **3/3** |
| WF | both | 296.4486 (c100) | −0.13 [−1.61, +1.46] | 1/3 |

F improves both months (January −0.44, July −2.63). The three control seeds alone range from
295.96 to 297.55, so Rome differences under about one second are seed noise.

**Unseen flight numbers.** In the practice months 2–3% of LIRF departures have a flight number not
seen in the training months. In July 2026 it is 48% (7,603 flights: Ryanair, Wizz Malta, easyJet,
ITA and others renumbered or added routes). On those flights F raised the copy chance by 0.04 on
average, which made 93% of the July change. The submitted version keeps v54's probability for
unseen flight numbers. On Jan/Jul, at 1,700 rounds: control 296.7497, F 295.0409, F with this
fallback 295.3014.

**Submission file:** `VALIDATION/rome_fn_submit.py --version 56` (v54's saved parts, classifier
refitted on all 2025 LIRF departures; checks that v54 and R20 are reproduced and that only LIRF
flights change). No flight moves by more than an hour. The build without the fallback is kept in
`VALIDATION/runs/v3_20261006_rome_fn_submit/nofallback_build/`; it moved one Vueling flight from
2,167 s to 6,138 s.

**Leaderboard (uploaded as v56 on top of v54):** 236.5988, **+0.27 worse** than v54 (236.3328).
On v54's own practice recipe the same classifier scored 295.3603 → 294.5503 (−0.81 [−1.46, −0.21],
classifier seed 42). This is the second Rome-classifier change whose Jan/Jul gain did not carry over
(v53 transferred at 0.22×). Not pursued further.
