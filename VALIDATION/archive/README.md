# Historical validation work

The [current plan](../IMPROVEMENT_PLAN.md) and
[Stage 0 results](../reports/STAGE0.md) take precedence over this archive.

## Earlier experiments

Each directory was moved intact, including locally retained predictions,
models and source snapshots. Run IDs remain resolvable by `records.read_run`;
`../runs/experiments.jsonl` remains the single experiment ledger.

| Run | Historical purpose |
|---|---|
| [20260906T230806Z](runs/20260906T230806Z/result.json) | Original baseline |
| [20260906T234524Z](runs/20260906T234524Z/result.json) | Baseline with full-tail evaluation |
| [20260907T000018Z](runs/20260907T000018Z/result.json) | Keep-tail recipe associated with submission v3 |
| [v3_20260907T202123182899Z](runs/v3_20260907T202123182899Z/result.json) | Original 800-round protocol-v3 baseline |
| [v3_20260909T161759260709Z](runs/v3_20260909T161759260709Z/result.json) | Explicit flight-plan-flag experiment |

## Earlier findings

[Findings from 7 September](FINDINGS_20260907.md) explain the early experiments.
Some conclusions were later revised: the notebook's schedule-gap mechanism
shows that a subset of extreme labels is predictable, and the flight-plan flag
has since been tested. Use the current plan and notebook map for active decisions.

## Removed obsolete files

These have no callers in the current workflow and are recoverable from Git:

- [Protocol v2](https://github.com/Goxmen/PRC/blob/ea3bce8/VALIDATION/VALIDATION_PROTOCOL_V2.md), superseded by v3.
- [Original simulation](https://github.com/Goxmen/PRC/blob/ea3bce8/VALIDATION/ch07-prc-validation.py) and
  [HTML](https://github.com/Goxmen/PRC/blob/ea3bce8/VALIDATION/ch07-prc-validation.html), superseded by the implemented harness and measured findings.
- [One-off CPU benchmark script](https://github.com/Goxmen/PRC/blob/ea3bce8/VALIDATION/benchmark_threads.py); its
  measurements and conclusions remain in [Stage 0](../reports/STAGE0.md).
