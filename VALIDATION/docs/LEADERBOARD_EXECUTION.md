# Executing the selected five-candidate comparison

The user authorized the frozen queue on 2026-09-16. Results and interpretation
are in [the leaderboard report](../reports/LEADERBOARD_RESULTS.md); the
[manifest](../reports/leaderboard_selected_20260916.json) binds filenames,
validation records, model and rule sources, data hashes and official results.

Two sequential full-year fits used the original 27 features, 800 rounds, seed
42 and 2,084,659 retained training rows each. Baseline preparation/fit/prediction
took 2.067 minutes; R3 took 4.461 minutes. Both produced 344,841 ranking
predictions, keeping January and July 2026 separate. The i7-13650HX settings
were six fit threads and twenty prediction threads, with one fit at a time.

`tools/refit_frozen.py` verifies every archived source file, copies it into
an isolated directory and invokes that implementation's unchanged
`submit.resolve_config` source guards. It also checks dataset hashes and
library versions. This preserves the exact B0-800 and R3 implementations
despite subsequent T1/T2/M1 changes in the active source tree. Baseline's old
prediction wrapper lacked a thread argument; explicit twenty-thread prediction
keeps its chunking/clipping and matched the original wrapper bit-for-bit on
512 ranking rows in each month. R3 uses its archived thread-aware wrapper.

The six-hour rule's implementation and existing Q0/Q1 records are unchanged.
The four-hour implementation is registered separately in
`pipeline/registered_rules.py`. Its boundary and original-ID eligibility agree
with the saved threshold sweep. Registration used saved predictions, zero
CV fits, and reproduced all three original four-hour scores within 1e-9 s:

| Candidate | Registered validation run | Registration minutes |
|---|---|---:|
| Q0-4h | `v3_20260916T164054429797Z` | 1.630 |
| Q1-4h | `v3_20260916T164247756836Z` | 1.627 |

`tools/leaderboard_batch.py prepare` verifies these links, then derives all
five outputs from the two saved, unrounded prediction arrays. It fills the
original template, preserving IDs/order/schema and rounding only at the end.
The six-hour rule affects 11 January and 15 July rows; four hours affects 15
and 28. Those eligibility sets are identical for baseline and R3.

Preparation never uploads. The explicit `upload` action checks current UTC
day usage before each upload, detects filename/content collisions, and verifies
remote SHA-256 after upload. It stops when five uploads already exist today;
there is no automatic reset retry. `refresh` attaches result payloads only to
verified uploads. `tools/report_leaderboard_batch.py` compares the five explicit
pairs using frozen CV scores and the original saved paired intervals.

All 56 validation tests passed, including the new four-hour boundary/sweep
equivalence test. Model/data source checks passed for both actual full-year
fits. Large model/prediction/source artifacts remain local and ignored; another
machine must copy the original run snapshots and prediction artifacts to reuse
them. Do not rerun completed fits merely to refresh official scores.
