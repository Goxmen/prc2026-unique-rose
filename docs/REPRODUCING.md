# Reproducing the final solution

This document distinguishes an **exact replay of the accepted final assembly**
from **training the entire solution from raw data**. The former has a standalone
offline export/replay tool. The latter has extensive retained source and run
records, but has not been demonstrated in a fresh clone and must not be described
as a verified one-command rebuild.

The accepted final at the time of this release preparation is v83, uploaded as
`unique-rose_final.parquet`, with 670,790 January, February, June and July 2026
departures. Its SHA-256 is
`f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f`.
The organiser acknowledged success for every row; no final numeric score was
published in that acknowledgement. The old two-month leaderboard score and
local validation RMSE are different measurements.

## Environment

Use the pinned [requirements.txt](../requirements.txt). The verified replay
environment is Linux/WSL with Python 3.12.3, NumPy 2.5.2, pandas 3.0.5 and
PyArrow 25.0.1. The original research environment requires an NVIDIA CUDA-capable GPU for
CatBoost/neural training. Hardware used is documented in [AGENTS.md](../AGENTS.md).
GPU/backend versions and exact execution details vary between historical runs;
their individual identity and training receipts are authoritative. No universal
bitwise training guarantee is claimed across GPUs, package versions or hosts.

The component replay needs only Python, NumPy, pandas, PyArrow and threadpoolctl
at the pinned versions. It uses one CPU thread, no GPU, network, credentials,
labels, training or uploads. It preserves the original matrix multiplication
order, coefficient precision, plan constraints, template schema and ID ordering.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r release/requirements-replay.txt
.venv/bin/python -m unittest discover -s tools/release -p 'test_reproduce_final.py' -q
```

The [environment snapshot](../release/environment.json) records the exact desktop
packages and hardware. Training and four-month validation need the full root
requirements; the smaller replay requirements avoid downloading GPU frameworks.

## Exact final assembly from retained artifacts

The accepted recipe uses a retained portable-v82 anchor `z`, 28 retained component
prediction vectors `P`, and one saved production coefficient vector `beta`:

```text
raw = z + (P - z[:, None]) @ beta
final = round_to_integer_nonnegative(apply_plan_constraints(raw))
```

The ordered components are versions 1, 4, 8, 12–21, 30, 33, 34, 37, 38, 45, 46,
54, 56, 57, 58, 60, 61, 64 and 70. The first constraint bounds predictions using
the matched flight-plan gap. The second projects predictions outside an excluded
interval for qualifying recovered plans. Exact formulas and tie handling are
in [reproduce_final.py](../tools/release/reproduce_final.py), with the original
formula sources included in the export for review.

On the original research checkout, export the retained inputs once:

```bash
.venv/bin/python tools/release/reproduce_final.py inventory
.venv/bin/python tools/release/reproduce_final.py export --bundle release/bundles/v83-component-replay
```

The exporter checks the frozen identity, production coefficient hash, every
selected original input hash and the accepted output hash. It projects only
the needed prediction/input columns into the bundle, and records original and
exported hashes. It includes the official template, saved coefficients, original
identity, formula sources and acceptance/audit evidence. **It does not copy the
accepted final prediction into the bundle.** It reads that file only to establish
the exact expected IDs/predictions digest. Export directories are never silently
overwritten; an interrupted export retains an `EXPORT_INCOMPLETE.json` marker.

Provide the complete bundle separately to the reviewer, together with its
`manifest.json` checksum. Data/prediction artifacts remain gitignored; code and
documentation alone do not contain this bundle. No distribution upload is
performed by these commands. Its contents include derived competition inputs;
retain applicable data attribution and distribution terms.

The prepared release asset is `unique-rose-v83-replay.tar`, with its archive
checksum in [artifact_manifest.json](../release/artifact_manifest.json).
It includes a standalone replay script, its CPU requirements, license,
attribution documents and the `v83-component-replay/` directory. The associated
GitHub draft tag is `prc-final-v83-20261013`; the owner authorized publication at
14 October 2026, 10:00 UTC / 12:00 Amsterdam. It remains private until then.
A private draft is not a public download.

After obtaining the archive and `SHA256SUMS` from that release:

```bash
sha256sum -c SHA256SUMS
mkdir -p replay-package
tar -xf unique-rose-v83-replay.tar -C replay-package
```

Follow its standalone `README.md`, or use the repository commands below with
`replay-package/v83-component-replay` as the bundle path.

In a fresh clone with that bundle copied to any local folder:

```bash
.venv/bin/python tools/release/reproduce_final.py verify \
  --bundle /path/to/v83-component-replay \
  --manifest-sha256 019021aa429548f028a563a8cdf24174d4e412932c7776468799a65c19639d12
.venv/bin/python tools/release/reproduce_final.py reproduce \
  --bundle /path/to/v83-component-replay \
  --manifest-sha256 019021aa429548f028a563a8cdf24174d4e412932c7776468799a65c19639d12 \
  --output release/reproduced/unique-rose_final.parquet
```

Replay recomputes all 670,790 predictions from the 28 component vectors, anchor,
coefficients and constraints. It checks the regenerated values against the
accepted semantic digest, then writes the original two-column schema. The JSON
receipt alongside the output records `values_exact`, `byte_exact` and both output
hashes. Identical pinned PyArrow versions are needed for byte-level Parquet
identity; semantic equality is checked separately. It never overwrites the
accepted submission or uploads the regenerated file.

The prepared bundle contains 43 files totalling 124,256,721 bytes, excluding its
manifest. Its replay was checked on all 670,790 rows and reproduced the accepted
Parquet bytes exactly. The small tracked
[verification receipt](../tools/release/accepted_final_replay.json) records this
result. The manifest hash above identifies this prepared bundle; an export with
different Parquet writer metadata may have a different manifest while retaining
the same original input and prediction digests.

This is reproducible **ensemble assembly**. It does not retrain the upstream
base models, recompute the public-source anchor or supply a general inference
pipeline for an arbitrary new month.

## From raw data to the retained solution

Competition files are obtained through the official OpenSky MinIO endpoint
using the participant's own credentials, as described in the [README](../README.md#setup).
Never distribute the team's credential files. External inputs and their
attributions must accompany the release; obtaining the competition files alone
does not recreate the external caches.

The retained dependency path is:

| Stage | Source and authoritative record |
| --- | --- |
| Movement inputs, history and features | `pipeline/`, `VALIDATION/modeling.py`, `VALIDATION/seasonal.py`, `VALIDATION/tuning_seasonal.py`; individual run configuration/identity receipts |
| Callsign/rotation and observed tracks | `VALIDATION/callsign_clocks.py`, `VALIDATION/adsblol.py`, `VALIDATION/sparse_track_sequence.py`; their input manifests |
| Historical base-model training | Individual historical run recipes in `VALIDATION/runs/`; indexed by `VALIDATION/research_registry.json` |
| Saved-model final inference and reconstruction | `VALIDATION/final_legacy_*.py`, `VALIDATION/final_missing_*.py`, `VALIDATION/final_weather_inference.py`; each component's result/audit receipt |
| Portable affine mixture | `VALIDATION/portable_affine_20261013.py`; run `v3_20261013_portable_affine` |
| Public-source correction anchor | `VALIDATION/portable_public_source_20261013.py`; run `v3_20261013_portable_public_source`, including the retained final tilt-recovery supplement |
| Fixed teacher distillation | `VALIDATION/portable_v80_distill_20261013.py`; run `v3_20261013_v80_teacher_distillation`, saved `production/model.json` |
| Accepted final packaging and audit | `VALIDATION/package_final_v83_20261013.py`; that run's `final_submission/` receipts |

These historical scripts have fixed paths, source/input identity guards and
single-attempt execution protections. They are source references, not a command
sequence to launch blindly from a fresh clone. Their complete retained model,
cache and prediction dependencies are not in Git. Some historical missing models
were reconstructed and audited against retained original predictions; machine
and reconstruction qualifications remain in those records. Exact full v80 final
inference was unavailable, and v83 is a distinct portable student rather than an
exact v80 replay.

A full training handoff still needs an ordered acquisition/preparation/training
runner and a completed clean-environment rebuild. Preserve the archive and
document these limitations rather than deleting dependencies or substituting
new models under historical version names.

## Four-month candidate selection currently in progress

[FINAL_MONTH_SELECTION.md](../VALIDATION/docs/FINAL_MONTH_SELECTION.md) documents
the newer ten-candidate suite. Every fold excludes the held-out month and its
immediate preceding/following calendar months. The four held-out months are
January, February, June and July 2025; each trains on nine months. The primary
summary is the equally weighted mean of those four monthly RMSEs; pooled RMSE
and the January/July and February/June summaries are reported separately.

Those are new direct recipes, **not historical v80/v83 replays**. They use fresh
training-only feature/calibration fits and fixed seeds/configurations, preserve
models and hashes, and share the same postprocessing across all ten candidates.
The runner does not automatically select a winner, produce a full-year model,
replace the accepted final, or upload anything.

After score review, a chosen new recipe requires its matching full-2025 production
fit, four-month final-input preparation, prediction/ID/schema audits and an updated
release manifest. Until that happens, the v83 replay above remains explicitly
tied to the currently accepted artifact. Never represent a local candidate's
validation result as its official final score.

Keep source and cached inputs unchanged while the current validation runs.
Release packaging lives under `tools/release/` and `docs/`; it does not modify
the runner's frozen `VALIDATION/*.py` or `pipeline/*.py` sources.
