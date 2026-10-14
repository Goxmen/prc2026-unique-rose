# Manual v80 full-year refit and final submission

**Superseded on 14 October 2026.** The owner restored the original v83 final;
all 670,790 rows were accepted at 08:36 Amsterdam. Use
[selected final status](../../release/SELECTED_FINAL.md) and the v83 replay guide.
The command below documents the earlier refit and is not the selected final route.

The owner's renewed recovery instruction permits replacing v83 only with fully
recovered, verified v80. This historical approximate-refit uploader is now blocked
before changing its selection receipt or contacting the competition. Its `check`
command remains available for read-only inspection; an assembly replay alone does
not authorize delivery.

The superseded execution command was:

```bash
.venv/bin/python -u tools/v80_refit/run.py run --threads 4
```

**Historical behavior before the delivery block:** this command uploaded the
resulting final prediction, replacing v83 after saving its bytes. The following
execution notes describe that superseded run. It used the desktop for missing tree
fits and `glaptop-wsl` for the missing neural pool, one heavy worker per host.
The optional `--desktop-only` mode ran both workers sequentially on the desktop;
hosts and thread settings were fixed when the run was initialized.

The original execution estimate was 2–4 hours, including transfer, correction
inference and packaging; no timed end-to-end measurement supported that estimate. The
submission deadline is 14 October 2026, 10:00 UTC / 12:00 Amsterdam. The runner
refuses a new final PUT after that deadline.

The metadata-only plan starts no model work and performs no uploads:

```bash
.venv/bin/python tools/v80_refit/run.py plan --threads 4
```

## Recipe and scope

The complete v80 assembly is retained: the historical 48-component base, nested
v71/v73 corrections, daily/monthly and public-model updates, the ordered 51-vector
v79 blend with its saved signed coefficients, and v80's three-TabM/runway deltas.
All prediction-dependent downstream corrections are recalculated. Joint/runway
increments can be reused because their original v70 prior is unchanged.

Thirty archived version components already have audited final predictions.
Twelve missing tree versions use 13 missing fits plus retained matching models;
six missing neural versions share 31 fixed training trajectories. Original
full-year recipes, selected stopping epochs, feature definitions, calibration,
blend weights and rounding are fixed. No hyperparameter search, held-out model
selection or new validation score is produced by this command.

This is a **fresh refit**, not recovery of every historical weight. Exact original
source for early v3/v5–v9 was not retained; the archived original27 implementation
uses their recorded features, filters, rounds and seeds. That source matches the
recovered v4/v8 results, but exact early historical predictions are not claimed.
Fresh GPU fits can also differ. Old-input differences are recorded as diagnostics,
never optimized away. The historical v80 leaderboard RMSE **218.8525** does not
establish a score for the new final predictions.

## Resume and evidence

Work stays in `VALIDATION/runs/v3_20261013_v80_refit_final/`. Full logs are in
`execution/`; compact worker states, model/checkpoint hashes, audits, delivery
receipts and release metadata remain separate. Source/input/configuration hashes
are frozen before execution. Required laptop files are transferred only when
missing or different, with backups and hash verification.

The superseded runner resumed ordinary interruptions from saved neural epochs,
including optimizer/scheduler/RNG state, and retained completed tree models and
correction groups. Changed source, input, model or configuration stopped the run.
A tree model saved without its completion receipt was preserved for inspection.

The first run stopped when the rebuilt v5 frame used newly discovered category
codes instead of the retained model's codes. The repair checks every training
value, index and target exactly, then keeps the saved frame's category mapping,
including unused levels, and checks it against the model. Completed v2/v3 weights
are unchanged. Original source identities and completion receipts are archived
under `recovery/`; the recorded source-only migration preserves their pre-repair
provenance. These records are included in the eventual replay bundle.

The overnight run completed both training workers, then encountered an
unbracketed support-tilt root during final assembly. Resume now verifies the
collected exports and skips training and laptop access. The numerical repair
widens only a failing root bracket, keeps the same 55 bisections and original
mean tolerance, and independently checks repaired weights with Brent's method.
Successful weights and completed airport-day predictions are retained. The
source/receipt migration and original records are preserved under `recovery/`.

Before its historical delivery the runner checked all 670,790 IDs, schema, integer
predictions, component arithmetic and exact standalone replay. The final PUT was
conditional on the reviewed v83 object's ETag. It recorded the attempt before PUT,
reconciled uncertain delivery by remote hash, checked the daily quota, and required
fresh official acceptance. These were delivery and assembly checks; they did not
establish recovery of the original trained weights.

## Release and reproducibility

Before the historical final PUT, the runner prepared an offline component replay
and private GitHub draft. After acceptance it updated five generated release files,
committed/pushed them on `main`, and switched the deadline publication policy.
The subsequent v83 restoration superseded those updates. The v83 draft is again
selected for publication at the previously authorized deadline.

The replay bundle regenerates the final bytes from 51 prediction vectors, fixed
weights, correction inputs and the blank template; it excludes the final output.
This is verified component-assembly replay. The refit runner depends on retained
local research inputs/models, so a clean-clone rebuild from raw data alone has
not been demonstrated. Source, provenance, model checksums and data attribution
are included; trained models/checkpoints stay retained on the machines.

Synthetic CPU tests cover original neural update/resume parity, tree contracts,
correction arithmetic, ID/hash failures, offline replay and upload/restart guards:

```bash
.venv/bin/python -m unittest discover -s tools/v80_refit -p 'test_*.py'
```

No full production preparation, refit, inference or submission was run while
preparing this command. The four-month validation experiment is separate.
