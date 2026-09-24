# V42: v40 with snow and arrival summaries

**Official RMSE: 274.9076 s, 0.5070 s better than v40 (275.4146).**
Uploaded on 24 September 2026; all 344,841 ranking flights scored successfully.
Weighted CV is **263.5929 s**, versus **264.3388** for the matched rebuilt v40 control.

The full-year refit keeps v40's seed 42, 800-round ordinary model, 300-round Rome
classifier, 127 training exclusions and correction stack. Only the ordinary model
gets the 12 new inputs. January and July use separate ranking-month context.
The local OPDI cache is the one used in the completed validation; its bytes differ
from the original v40 cache, so this is not an exact cache-controlled leaderboard comparison.

Two boosters took 91.8 seconds; preparation and prediction took 3.70 minutes total.
Six wiring tests and ten upload tests pass. The final audit checks all 344,841
correction chains and template predictions, plus 2,560 independent arrival values
and 1,280 snow values on 320 sampled flights. The uploaded file hash matches locally.
An initial pre-fit check stopped on a 5.6e-17 rounding difference in a correction
weight; allowing at most 1e-12 fixed the check. No completed fit was repeated.

The overall gain does not establish an improvement on prolonged-snow flights;
the earlier [stress check](OPDI_DISRUPTION_STRESS_CHECK.md) remains inconclusive.

[Manifest](../runs/v3_20260924_opdi_disruption/submission_v42/submission_v42.json) ·
[Audit](../runs/v3_20260924_opdi_disruption/submission_v42/preupload_audit.json) ·
[Runner](../opdi_disruption_submit.py) (`prepare`, `upload`, `result`)
