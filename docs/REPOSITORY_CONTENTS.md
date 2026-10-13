# Repository contents for review

Git contains the model and feature source, validation and replay code, tests,
pinned dependencies, licensing/attribution, submission history, method documents,
and selected configuration, calibration and audit receipts. Start with
[selected final status](../release/SELECTED_FINAL.md) and the
[reproduction guide](REPRODUCING.md).

The local research workspace is roughly 90–100 GB. Before this cleanup, the
tracked checkout was about 96 MB; the datasets, Python environment, model weights,
prepared matrices and prediction files were already excluded from Git. Removing
generated report outputs and operational records reduces the tracked checkout
to approximately 32 MB. Git history has not been rewritten, so historical commits
still contain previously tracked reports; this is not a claim about full-clone size.

## Kept in Git

- All existing Python source and tests, including the exact source set frozen by
  the running v80 refit. Historical source is needed to interpret its recipes and
  provenance; removing it mid-run would invalidate the source guard.
- Final v80/v83 dependencies and selected upstream recipes, coefficient files,
  calibration parameters, provenance and audit receipts. The metadata allowlist
  is [reviewer_files.json](../release/reviewer_files.json).
- Concise method/validation reports and the exploratory notebook with its figures.
- Release status, exact replay tooling, dependency versions, license and third-party
  attribution. The selected final's replay archive is a separate release asset.

## Kept locally

Competition/external datasets, generated feature tables, checkpoints, fitted
models, neural encodings, predictions, full logs, temporary files, caches,
duplicate generated plots/HTML/row dumps, failed-experiment operational records,
agent handoffs and internal research bookkeeping are ignored. No local data or
checkpoint was deleted by the cleanup.

The `VALIDATION/runs/.gitignore` and `VALIDATION/reports/.gitignore` files exclude
generated outputs by default and explicitly retain selected review metadata.
Future useful metadata should be added deliberately to those allowlists; source
files remain visible. Existing historical reports may mention or link to detailed
generated outputs that are available only in the local research workspace.

The cleanup does not turn the solution into a raw-data-only rebuild. Follow the
documented replay/artifact requirements. The active refit keeps all its local
inputs and frozen Python sources unchanged; its generated final-release receipts
and bundle are produced by the user-launched runner.
