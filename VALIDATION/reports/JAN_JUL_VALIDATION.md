# Jan/Jul CV: does it track the leaderboard?

Checked on the last five uploads, v39–v43, before using it for hyperparameter search.

## Design

Two fits; the score is the average of their RMSEs.

| Fit | Held out | Training months | History inputs freeze at |
|---|---|---|---|
| 1 | January 2025 | March–November 2025 | 30 November |
| 2 | July 2025 | January–March, September–December 2025 | 31 December |

Each upload's recipe is refitted unchanged: same inputs, 800 rounds, the same
rule-row handling, the same corrections and rules, seeds 42/142/242. Inputs are built
in the order each submission script used. Every held-out flight is scored: 153,706
in January and 190,713 in July. Within a fit, seeds are averaged in squared error, as
the 16-fold CV does. Refitting v43 at seed 42 reproduced the prepared control exactly.

Run with `.venv/bin/python VALIDATION/tuning_seasonal.py validate`. Results are in
[jan_jul_validation/submissions.json](jan_jul_validation/submissions.json). It took
7.3 minutes to prepare and 63.7 minutes for 30 fits on the Apple M3.

## Scores

| Upload | January | July | **Jan/Jul CV** | Seed 42 only | Official LB |
|---|---:|---:|---:|---:|---:|
| v39 | 344.984 | 297.563 | **321.2736** | 321.2888 | 277.3629 |
| v40 | 342.726 | 295.151 | **318.9388** | 319.2585 | 275.4146 |
| v41 | 342.327 | 295.137 | **318.7320** | 318.8743 | 275.9592 |
| v42 | 342.633 | 294.555 | **318.5942** | 318.6740 | 274.9076 |
| v43 | 342.193 | 293.888 | **318.0408** | 318.0950 | 274.7660 |

## Against the leaderboard

The lineage steps are v39→v40, v40→v41, v40→v42 and v42→v43. "Step error" is the
mean of |validation change − leaderboard change| over those four steps.

| Validation | Score − LB | Spread of that gap | Step error | Directions |
|---|---:|---:|---:|---:|
| **Jan/Jul CV** | +43.43 | **0.39** | **0.43** | 3/4 |
| CV next-month | −29.77 | 0.40 | 0.49 | 3/4 |
| CV combined | −42.73 | 0.53 | 0.67 | 3/4 |
| Weighted CV | −11.81 | 0.66 | 0.85 | 3/4 |
| CV longer-gap | −53.57 | 0.68 | 0.87 | 3/4 |

| Step | LB | Jan/Jul CV | Next-month | Combined | Weighted | Longer-gap |
|---|---:|---:|---:|---:|---:|---:|
| v39→v40 | −1.95 | **−2.34** | −1.07 | −0.51 | −0.08 | −0.03 |
| v40→v41 | +0.54 | −0.21 | −0.19 | −0.29 | −0.45 | −0.38 |
| v40→v42 | −0.51 | −0.35 | −0.42 | −0.57 | −0.75 | −0.71 |
| v42→v43 | −0.14 | −0.55 | −0.39 | −0.50 | −0.45 | −0.59 |

Jan/Jul CV intervals (95%, whole days resampled within each fit):

| Step | Change | Interval | Reading |
|---|---:|---|---|
| v39→v40 | −2.34 | [−3.97, −1.29] | improvement |
| v40→v41 | −0.21 | [−0.49, +0.08] | inconclusive |
| v40→v42 | −0.35 | [−0.58, −0.13] | improvement |
| v42→v43 | −0.55 | [−0.82, −0.38] | improvement |

## Reading

- **The level is not close.** Jan/Jul CV sits 43 s above the official score, which
  is about as far as CV combined sits below it. The gap is the most stable of the
  five validations, though (spread 0.39 s), so it ranks recipes in the same order as
  the leaderboard would. Only Weighted CV is close in level.
- **It tracks the steps best,** with a mean error of 0.43 s against 0.49–0.87 s for
  the others. It is the only validation that gets the size of the largest step right:
  v40's rule-row exclusion is −2.34 here against −1.95 on the leaderboard.
  Weighted CV saw −0.08.
- **Every validation misses v41.** Jan/Jul CV is the only one whose interval does
  not call v41 a gain: it is inconclusive. All five also overstate v43's step.
- **This is weak evidence.** It rests on four steps, and the lead over next-month CV
  is 0.06 s of step error. These months also overlap the 2025 data used to choose
  recipes before, so the check is not independent.
- **Seed noise is about the size of a typical step.** v40's three seeds span 0.59 s.
  Compare tuning candidates at three seeds, or demand a larger change than that.

Other differences to keep in mind:

- The submitted v42 used gunes's rebuilt OPDI cache; these refits use this Mac's copy.
- The ledger's v42 CV values come from the x86 desktop; its v43 values come from this Mac.
