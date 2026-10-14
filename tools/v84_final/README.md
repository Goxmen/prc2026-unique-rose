# Unselected v84 reconstruction and weight recovery

The owner retained **v83 as the competition final** on14October2026. These tools
prepare and audit a separate v84 reconstruction. They do not authorize replacing
the accepted final, changing its release selection or publishing before the
deadline. See [the selected final](../../release/SELECTED_FINAL.md).

v84 keeps the fixed historical v80 ensemble arithmetic, with recovered original
tree models, a reconstruction using the recovered v3 source and saved neural
refits where early original weights were missing. It is not an exact recovery of
historical v80. Old-input agreement is distinct from prediction accuracy or a
leaderboard score.

The working artifacts are under
`VALIDATION/runs/v3_20261014_v84_final/`. Source and audit receipts identify the
exact model and feature inputs. No training or output vector is silently
substituted after a gate failure.

## Actual model weights

`weight_inventory.py` traces provenance to real model/preprocessing files;
prediction Parquets are never counted as model states. Its broad inventory
includes historical controls and therefore needs the selected-family audit.
The selected neural inventory separately records actual production checkpoints.

The original October8 HPO156 producer discarded its15 returned models. Retained
prediction vectors do not resolve that gap. `recover_hpo_weights.py` runs the
unchanged original data constructor and trainer, saves the returned model and
actual encoder/bin/scaling states, then requires exact float32 agreement with
every original old and final prediction and an exact saved-state reload. Any
mismatch stops the remaining fits; the failed checkpoint and comparisons remain.
Only an explicitly allocated attempt may run it. Its `plan` command is read-only.

This recovery was stopped at the owner's request after the first member passed
exact old/final prediction and saved-state checks. Those artifacts are retained;
the remaining weights are incomplete. The owner subsequently authorized an
assembly replay package without requiring full model-weight recovery.

## Offline replay and private packaging

`package.py` exports the51 integer component vectors, fixed coefficients, three
correction inputs and blank final template. `replay.py` recomputes the output;
the final prediction file is excluded from the package. Every exported file and
the regenerated Parquet bytes are checked against SHA-256 manifests.

An eventual reviewed source commit is required:

```bash
.venv/bin/python tools/v84_final/package.py export --source-commit REVIEWED_COMMIT
```

Packaging and clean offline replay are authorized; GitHub draft/upload actions
are not part of the current task. The separate `draft` stage can create a **private draft** only. It never changes
the active publication schedule, publishes, commits, pushes or submits to the
competition. Do not invoke it without renewed distribution authorization.
The v83 release and replay assets remain the selected rollback.

After obtaining a verified standalone bundle, the offline command is:

```bash
python replay.py reproduce --bundle . --output unique-rose_final.parquet
```

This is exact retained-component assembly replay. Full raw-data retraining and
arbitrary-new-month inference have not been demonstrated from a clean clone.
