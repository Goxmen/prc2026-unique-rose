# Arrival-supervised ground representation: provisional frontier

Source-only assessment, 13 October 2026. No prediction arrays, raw labels, model
checkpoints or CV data were decoded; no experiment or final artifact was changed.

**Retain one provisional mechanism:** use additional arrival examples to learn a
shared representation of the moving/parked transition near the stand. Its nearest
prior is departure-only point-boundary supervision, which was inconclusive. No
arrival multitask implementation was found in the novelty index or neural source
search; the cohort owner independently confirms that the recent consistency and
cohort models used departures only. This is evidence of a gap, not exhaustive proof
of novelty or evidence of a several-second gain.

## What is already covered

| Proposed information | Existing evidence | Assessment |
|---|---|---|
| Same-flight destination clocks identify departure clock error | Original paired endpoint pilot: 10,245 training-date pairs, correlation −0.00992; additive proxy RMSE 384.90→392.70; all four months worse. Original AOBT/ARVT copies agree across endpoints. | Closed; do not repeat or call pairing privileged information. |
| Arrival lag or stand drift transfers to departure errors | Prior arrival-clock transfer and four-month stand-drift transfer failed. Previous-aircraft rotation inputs were also tested. | Closed scalar/context mechanisms. |
| Supervise missingness, clock differences or recorder modes | All five off-block estimates are already baseline inputs. The 12-output ordinal/copy auxiliary had matched CV +0.2894, inconclusive; its half replacement regressed against v58. Missingness is already represented. | New names for existing information/objectives are not a new source. |
| Supervise departure motion around the hidden boundary | Existing fixed point-boundary auxiliary: matched CV +0.1697, interval [−0.0551,+0.4056]. | Closed departure-only screen; no weight or threshold retry. |
| Distill a stronger teacher | Whole-v80 distillation is now selected as v83; the earlier missing-affine proxy failed. | Already done; neither supplies new privileged source information. |

Departure BLOCK is exactly takeoff minus the target. Thus any target made solely
from BLOCK and existing input clocks is a transformation of the same label, not
extra information about the conditional mean. It can change optimization, but that
alone does not establish new headroom. Likewise, arrival taxi-in/BLOCK is supplied
in ranking: arrival supervision supplies additional examples, not a feature that
exists only during training. Artificially hiding reported clocks does not establish
transfer to naturally missing plans, whose missingness mechanism can differ.

## Exact remaining hypothesis

The existing raw-track model has motion, stand/runway distance, missingness and
clock features. It may nevertheless estimate the stand transition inefficiently
from departure labels: a departure label conflates true pushback, taxiing and
recording error. Additional arrival examples could improve shared motion features
without assuming that arrival and departure recording lags have equal distributions.
The proposed transfer concerns motion representation, **not** a common scalar lag,
stand-month correction, paired-flight timestamp alias or new target reconstruction.

Conditional on the full observed track, arrival supervision creates no extra
per-flight input information. Any benefit is finite-sample representation learning.
Recorded in-block is also noisy; positive arrival-only accuracy is insufficient.
The test must measure transfer to unseen departures against an identical control.

## Available local inputs

The retained `v3_20261010_arrival_clock_context_pilot/prepared.json` records 67,741
arrivals and 89,948,755 bytes across a query table and twelve cached point packets.
Dates are 5/15/25 March, September, October and November 2025, all inside both
outer training windows. Queries include landing, in-block, input-only aircraft
identity, stand geometry and airline; point packets contain timestamp, position,
speed and altitude. Only 1,796 strict physical-stop witnesses were found, so a
witness-only learner is a narrow, selected population. Use the existing packets;
do not rebuild that failed lag pilot or download more traces.

Existing departure sparse-track caches and original training files cover these
dates. An adapter for arrival sequences does not yet exist. Array-level identity,
point support, input-only eligibility and phase-normalization remain unverified.
This report pins metadata and source, not the contents of the cached parquet files.

## Proposed bounded falsifier, not an allocated run

1. **Coverage first, inputs only.** On the twelve dates, count uniquely identified,
   mapped movements with at least two ground observations in fixed event windows;
   preserve the existing departure window and arrival landing→landing+5400s window.
   Limit source reads to the retained packet and necessary departure cache rows.
   Report common airport/stand, speed and missingness support by phase and temporal
   half. Require at least 500 usable arrivals and 500 departures in each half and
   at least two common airports. Sparse support closes as inconclusive. Check
   input-known later rotations explicitly; do not silently treat all post-arrival
   movement as parked.
2. **One inexpensive transfer test.** After a separate review/freeze, use one fixed
   ridge-regularized point logistic learner and identical features for both arms.
   Control learns from departures; candidate uses a fixed 50/50 departure/arrival
   policy with the same total loss mass and penalty, explicitly halving departure
   weight. Each flight has equal weight within phase. Use existing motion/geometry
   transforms, excluding absolute timestamps, time-to-BLOCK, the target, and any
   feature reconstructed from it. Phase alignment and the exact feature list must
   be frozen before arrays are read. The supervised label is the existing 30-second
   smoothed taxi-side boundary indicator: after recorded off-block for DEP, before
   recorded in-block for ARR. The target enters only the label function.
3. **Whole-date transfer.** Train on March+September dates, evaluate departures on
   October+November dates, then reverse. Neither phase from an evaluation date may
   train its learner; quarantine shared aircraft/day groups and input-linked paired
   movements across the split. Freeze preprocessing on the fit side. No January or
   July departures, old/final ranking, selected teacher predictions or public-source
   posterior labels enter this test. Save both arms before label scoring.
4. **Allocation-changing evidence.** Require at least 5% lower flight-weighted
   boundary log loss in both directions and a whole-day paired interval strictly
   below zero. Report all common-support/missingness groups without selecting one.
   A separate no-fit diagnostic uses candidate motion coefficients with control
   intercept/airport coefficients. It must also improve both directions; otherwise
   the mechanism remains inconclusive. Correlated coefficients mean that failure
   is not a proof against all arrival transfer. This supports one representation
   experiment only; it is neither taxi-out CV nor
   a lower bound on attainable RMSE improvement. Failure closes this fixed low-cost
   transfer probe; it cannot prove that every arrival-pretraining architecture fails.

The proposal needs a lean reviewed implementation and synthetic phase/label tests
before numerical allocation. Bound the learner to one fixed configuration, one CPU
thread, existing 128-point cap, no hyperparameter or airport scan. A practical
initial budget is 2 GB and five minutes after preparation; this is a planning cap,
not a benchmark. If the source adapter cannot fit that budget, stop and report its
measured cost instead of expanding automatically.

The concrete JSON design fixes a maximum 5,000 input-ID-hash-selected training
flights per phase/half; held evaluation is streamed. Coverage gates apply after
finite-label supervision. Legacy arrival caches turn negative taxi-in into NULL:
preserve and report that population restriction rather than silently reconstructing
its labels. Input-known other-movement windows are excluded symmetrically. Motion
pairs retain pre-filter adjacency so they cannot jump over observed airborne slots.

An honest residual test against the selected estimator requires training-date
predictions made without those dates' labels. Such predictions were not established
by this source audit; in-sample full-year outputs must not stand in for them.
Consequently the cheap probe tests conditional transfer relative to its matched
motion-feature control, not residual headroom conditional on the complete v80 stack.

## CV successor, only after a positive falsifier

One fixed seed-42 Jan/Jul comparison would retain the selected tabular/public inputs,
departure population and prediction head. The candidate changes only shared motion
training by adding allowed-training-month ARR boundary supervision; a matched
departure-only control separates arrival transfer from extra optimization. The
original departure-only auxiliary result stays closed. Arrival label/clock fields
are excluded from model inputs, held-month arrivals are excluded from training, and
all cross-phase/aircraft identity groups obey the outer firewall. Save models before
inference and score the exact serialized parent recipe, both months and paired-day
interval. A successor protocol must be frozen separately; this report does not
authorize fits, choose its loss weight or propose a final change. One heavy job per
host remains mandatory.

**Recommendation:** keep this behind the currently leading retrieval investigation.
It is a defensible untested transfer mechanism with a local, cheap falsifier, but
current evidence does not justify full neural training or a claimed score gain.
