# Reproducing the final solution

The selected final is **v84**, a reconstruction of the fixed historical v80
ensemble recipe. It contains 670,790 January, February, June and July 2026
predictions and is submitted as `unique-rose_final.parquet`. Its SHA-256 is:

```text
76938ead67f333e10a57290ad3cea71cb2aba85f94158de929724789e074cb41
```

The organiser accepted all 670,790 rows at 10:39:25 Amsterdam on 14 October
2026; the uploaded bytes match the hash above. No numeric final score was
published. [Selected final status](../release/SELECTED_FINAL.md) records the
authoritative submission and acceptance evidence. Historical v80's leaderboard score does not
measure v84's accuracy; agreement with old predictions is also not validation
accuracy or an official final score.

The verified deliverable is an **exact standalone replay of retained-component
assembly**. Recovered original tree models, the recovered v3 source and saved
neural refits produced these components. Some original neural weights were not
saved, so v84 is not an exact recovery of historical v80. A complete raw-data
training rebuild and inference on arbitrary new months have not been demonstrated
from a clean clone. The replay does not require those missing weights.

## Download and reproduce

The release tag is `prc-final-v84-76938ead67f3`. Its source is frozen at
`16ce6238236051555a6296d1a6d733de90a2b253`; later documentation and delivery
receipts do not change the frozen computation. The repository and draft remain
private until the authorized publication at 14 October 2026, 09:30 UTC /
11:30 Amsterdam. After publication, obtain these two assets from the
[v84 release](https://github.com/Goxmen/PRC/releases/tag/prc-final-v84-76938ead67f3):

- `v84-final-76938ead67f333e1.tar` — 78,264,320 bytes.
- `v84-final-76938ead67f333e1-SHA256SUMS` — archive checksum.

The archive SHA-256 is
`e49157508392f312bb0ef10491eac12048e1230908de2c865a28a5afb101b6a7`.
The bundled `manifest.json` SHA-256 is
`7aae3b9cadbaf6c0faf6725e64da502c807d396c63423e46fa3406ffd95f9099`.
These identify the independently tested archive, rather than an unspecified
future export. [artifact_manifest.json](../release/artifact_manifest.json)
records the selected release assets.

Run from a directory containing both downloaded assets:

```bash
sha256sum -c v84-final-76938ead67f333e1-SHA256SUMS
mkdir replay-v84
tar -xf v84-final-76938ead67f333e1.tar -C replay-v84
cd replay-v84
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-replay.txt
.venv/bin/python replay.py verify --bundle . \
  --manifest-sha256 7aae3b9cadbaf6c0faf6725e64da502c807d396c63423e46fa3406ffd95f9099
.venv/bin/python replay.py reproduce --bundle . \
  --manifest-sha256 7aae3b9cadbaf6c0faf6725e64da502c807d396c63423e46fa3406ffd95f9099 \
  --output unique-rose_final.parquet
sha256sum unique-rose_final.parquet
```

Use Python 3.12 and the included pinned CPU requirements: NumPy 2.5.2, pandas
3.0.5, PyArrow 25.0.1 and threadpoolctl 3.7.0. The verified environment used Python
3.12.3 on Linux/WSL. Dependency installation needs a package source; the replay
itself needs no network, GPU, credentials, labels, training or uploads. It refuses
to overwrite an existing output and fails if the regenerated Parquet bytes do
not match the selected hash.

## What the archive computes

The archive contains 51 integer component prediction vectors `P`, their fixed
historical coefficient vector `w`, three correction vectors, and the official
blank final template. [replay.py](../tools/v84_final/replay.py) computes:

```text
v79 = rint(max(P @ w, 0))
v84 = rint(max(v79 + (tabm_v78 - v78) + (robust_v78 - v78), 0))
```

`rint` uses NumPy's nearest-even integer rounding. The component order and full
coefficient precision are retained in `recipe.json`; the matrix multiplication
uses one BLAS thread. The three correction vectors preserve the selected
three-TabM and robust-runway adjustments. All inputs are aligned to the template's
unchanged movement IDs, and the output uses the original two-column schema.

The archive has 1,077 files, including the standalone script, pinned requirements,
source, computational identity/audit receipts, attribution documents and license.
The final prediction file is deliberately absent: replay derives it from the
retained inputs. The manifest binds every exported file, its size and SHA-256,
and the expected regenerated output hash. Source and dependency receipts retain
upstream model/input identities; a hash entry does not imply its model or raw
data is included in the archive.

Both the producer replay and an independent fresh extraction reproduced all
670,790 rows and the exact final Parquet bytes. The independent run used Python
isolated mode with networking disabled in a separate network namespace. It
verified the archive's safe paths and every member hash, and ran only the bundled
replay with its installed dependencies. Full local receipts are retained under
`VALIDATION/runs/v3_20261014_v84_final/release/`, including
`portable_replay_complete.json` and `independent_local_archive_audit.json`.

## Source, upstream training and data

The main entry points are [assemble.py](../tools/v84_final/assemble.py),
[v3.py](../tools/v84_final/v3.py), [package.py](../tools/v84_final/package.py) and
[replay.py](../tools/v84_final/replay.py). The archive includes the frozen tracked
Python source and selected computational provenance. Source-only GitHub contents
cannot replace the separately distributed replay archive.

| Stage | Source and authoritative record |
| --- | --- |
| Movement history and features | `pipeline/`, `VALIDATION/modeling.py`, `VALIDATION/seasonal.py`; individual configuration and input identities |
| Callsign, rotation and observed tracks | `VALIDATION/callsign_clocks.py`, `VALIDATION/adsblol.py`, `VALIDATION/sparse_track_sequence.py`; input manifests |
| Historical model training | Historical producers in `VALIDATION/`; frozen source and run provenance |
| Original trees and recovered v3 source | `tools/v84_final/v3.py`; recovered-tree receipts referenced by the dependency inventory |
| Retained neural refits | `tools/v80_refit/`; selected epoch checkpoints and preprocessing identities |
| Saved-model final components | `VALIDATION/final_legacy_*.py`, `VALIDATION/final_missing_*.py`, `VALIDATION/final_weather_inference.py`; component audits |
| Fixed v84 assembly | `tools/v84_final/assemble.py`; `assembly_identity.json`, `leaf_catalog.json`, `dependency_inventory.json` and final audit |
| Portable replay export | `tools/v84_final/package.py` and `tools/v84_final/replay.py`; frozen archive manifest and replay receipts |

Competition files require the participant's own official OpenSky MinIO access,
as described in the [README](../README.md#setup). External inputs and their terms
are documented in [DATA_SOURCES.md](DATA_SOURCES.md) and
[THIRD_PARTY.md](THIRD_PARTY.md), also included in the archive. Do not distribute
credentials. The replay archive contains derived prediction inputs; raw datasets
and large training caches remain outside Git.

Historical training scripts have fixed paths and source/input identity guards;
they are not a clean-clone command sequence. Training requires the full root
[requirements.txt](../requirements.txt), GPU/framework-specific environments and
upstream caches. Individual run receipts record host/backend differences. No
universal bitwise training guarantee is claimed across devices or package
versions. Completing that larger handoff would require a tested ordered data
acquisition, feature preparation and training pipeline.

## Historical validation and rollback

In the four-month blocked validation design, each
held-out month excludes itself and its immediate calendar neighbours from
training. Those candidate runs are separate research artifacts and do not prove
an official v84 score or a full reproducible training rebuild.

The previously accepted v83 final and its verified replay archive are retained
as rollback evidence. Its SHA-256 is
`f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f`;
the old release tag is `prc-final-v83-20261013`. Its replay uses
`tools/release/reproduce_final.py` and a different 28-component student recipe.
Do not combine its manifest, coefficients or commands with the selected v84
bundle. [Selected final status](../release/SELECTED_FINAL.md) identifies the
current submission.
