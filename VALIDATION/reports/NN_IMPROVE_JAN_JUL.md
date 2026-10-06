# Improving v47's neural network: Jan/Jul CV

Three changes to the network in v47's 0.3 network / 0.7 LightGBM blend, each scored on
Jan/Jul CV against v47 (2026-10-03).

| Arm | What changed from v47's network |
|---|---|
| avg5 | Nothing; five seeds averaged (network 1 is v47's own) |
| ple | Each rank-gauss input becomes a piecewise-linear embedding (48 quantile bins, 16 dimensions; Gorishniy et al. 2022), and training stops at the epoch that is best on the last training month; five seeds averaged |
| tabm | TabM (32 networks sharing most weights; Gorishniy et al. 2024) on ple's inputs, same stopping; one network, since it is about eight times slower per epoch |

The recipe, decision rule and blend weight were fixed before any arm ran; see the docstrings of
[`nn_improve_jan_jul.py`](../nn_improve_jan_jul.py) and
[`score_nn_improve_jan_jul.py`](../tools/score_nn_improve_jan_jul.py). Rows, inputs, LightGBM
predictions, Rome classifier, corrections and rules are v47's. Stopping is chosen on November
(January fit) or December (July fit), training on the other months with the same schedule, then
every network is refitted on all training months and stopped there. Held-out months are never
read for a choice.

## Result

Blend weight 0.3, seeds 42/142/242, squared errors averaged over seeds; intervals resample
whole days within each month.

| Arm | January | July | **Jan/Jul CV** | Change vs v47 | 95% interval | Seeds better |
|---|---:|---:|---:|---:|---|---:|
| v46 (LightGBM alone) | 333.74 | 273.77 | 303.7536 | | | |
| v47 | 332.88 | 271.86 | 302.3701 | | | |
| avg5 | 332.71 | 271.55 | 302.1317 | −0.24 | [−0.34, −0.16] | 3/3 |
| **ple** | **330.98** | **270.74** | **300.8628** | **−1.51** | **[−2.27, −1.08]** | **3/3** |
| tabm | 332.27 | 271.07 | 301.6720 | −0.70 | [−1.13, −0.41] | 3/3 |

| Comparison | Change | 95% interval |
|---|---:|---|
| ple vs avg5 | −1.27 | [−1.97, −0.87] |
| tabm vs avg5 | −0.46 | [−0.86, −0.19] |
| tabm vs ple | +0.81 | [+0.54, +1.31] |

**Candidate: ple.** It passes against v47 and improves on avg5; tabm is worse than ple.
The rule preferring ple unless tabm beats it was added after avg5 and ple were scored and
before tabm finished. Three arms were tried on the same months as v44–v47's choices, so these
gains are not independent confirmation.

## Diagnostics

- **Most of ple's gain does not need averaging.** One ple network gives −0.96 [−1.61, −0.58];
  averaging five adds −0.55, against −0.24 for v47's network.
- **ple's network alone (301.87) beats LightGBM alone (303.75).** v47's network alone scores
  309.79; tabm's 302.95.
- **ple improves every airport in both months except Rome in July (+0.09 s).** The largest
  gains are EGLL, EHAM, LTFM and EDDF in January (−3.6 to −4.5 s).
- **The blend weight 0.3 is now too low for ple.** The best weight is 0.65 in January and 0.55
  in July. Blending each month at the other month's best weight scores 300.0652, −0.80
  [−1.35, −0.43] against ple at 0.3. For v47 the same check is +0.09: 0.3 suited it.
- **Stopping epochs (of 24):** ple 17/11/15 (January) and 19/17/16 (July); tabm 14/11/14 and
  23/20/21. The last-training-month curve is flat over the middle epochs, so the exact epoch
  matters little.

## Leaderboard

Both weights were uploaded on 2026-10-03 at the user's request (`nn_improve_submit.py`; five
ple networks fitted on all of 2025, epoch chosen on December 2025).

| Upload | Blend weight | Official LB | Change vs v47 (245.3758) |
|---|---:|---:|---:|
| v49 | 0.3 | 242.9764 | −2.40 |
| **v48** | **0.6** | **239.7731** | **−5.60** |

The network change transferred at about 1.6× its Jan/Jul CV gain; the weight change gave −3.20
on the leaderboard against −0.80 out-of-month.

## Run

| Arm | Networks | Minutes (choosing epochs + fits) |
|---|---:|---:|
| avg5 | 24 new + 6 reused | 13.5 |
| ple | 30 | 5.7 + 21.2 |
| tabm | 6 | 50.7 + 39.3 |

RTX 3080, torch 2.14.1, `tabm` 0.0.3 and `rtdl_num_embeddings` 0.0.12 (Apache-2.0). Network
predictions, per-epoch curves and scores are in `VALIDATION/runs/v3_20261003_nn_improve_jan_jul/`
(`result.json`, `score.json`; `score_avg5_ple.json` is the same scoring before tabm finished).
The full-year submission runner is [`nn_improve_submit.py`](../nn_improve_submit.py); it
accepts only the scorer's candidate, at weight 0.3 or at the cross-month-checked weight.
