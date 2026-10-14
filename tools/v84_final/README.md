# v84 final reconstruction and exact assembly replay

The owner selected **v84 as the competition final** on 14 October 2026. See
[the selected final](../../release/SELECTED_FINAL.md) for submission/acceptance
receipts and [reproduction instructions](../../docs/REPRODUCING.md) for the
standalone archive. The repository and release became public at 09:25:18 UTC /
11:25:18 Amsterdam on 14 October, following the owner's immediate-publication
request. The earlier 11:30 timer is disabled.

v84 keeps the fixed historical v80 ensemble arithmetic, with recovered original
tree models, a reconstruction using the recovered v3 source and saved neural
refits where early original weights were missing. It is not an exact recovery of
historical v80. Old-input agreement is distinct from prediction accuracy or a
leaderboard score.

The organiser accepted all 670,790 rows at 10:39:25 Amsterdam on 14 October
2026, with no numeric final score published. The remote file matches SHA-256
`76938ead67f333e10a57290ad3cea71cb2aba85f94158de929724789e074cb41`.
Working artifacts and audits are retained under
`VALIDATION/runs/v3_20261014_v84_final/`. Source/input identities bind the selected
components; failed checks never silently substitute another vector or model.

## Offline replay

`package.py` exports the 51 integer component vectors, fixed coefficients, three
correction inputs and blank final template. `replay.py` recomputes the output;
the final prediction file is excluded from the archive. Every exported file and
the regenerated Parquet bytes are checked against SHA-256 manifests.

The verified archive is `v84-final-76938ead67f333e1.tar`, SHA-256
`e49157508392f312bb0ef10491eac12048e1230908de2c865a28a5afb101b6a7`.
Its manifest SHA-256 is
`7aae3b9cadbaf6c0faf6725e64da502c807d396c63423e46fa3406ffd95f9099`.
Both a producer replay and an independent fresh extracted replay reproduced the
selected Parquet bytes exactly; the independent replay ran with networking
disabled. After verifying and extracting the archive:

```bash
python -m pip install -r requirements-replay.txt
python replay.py verify --bundle . \
  --manifest-sha256 7aae3b9cadbaf6c0faf6725e64da502c807d396c63423e46fa3406ffd95f9099
python replay.py reproduce --bundle . \
  --manifest-sha256 7aae3b9cadbaf6c0faf6725e64da502c807d396c63423e46fa3406ffd95f9099 \
  --output unique-rose_final.parquet
```

The frozen source commit is
`16ce6238236051555a6296d1a6d733de90a2b253`. The public
[v84 release](https://github.com/Goxmen/PRC/releases/tag/prc-final-v84-76938ead67f3)
contains the exact archive and checksum assets, whose digests were verified. The `draft` stage never activates publication, commits, pushes or
submits to the competition. Those are separate authorized operations. The old
v83 release and replay assets remain available for rollback.

This is exact retained-component assembly replay. Full raw-data retraining and
arbitrary-new-month inference have not been demonstrated from a clean clone.

## Retained model-state work

`weight_inventory.py` traces provenance to real model/preprocessing files;
prediction Parquets are never counted as model states. Its broad inventory
includes historical controls, while the selected neural inventory separately
records production checkpoints.

The original October 8 HPO156 producer discarded its 15 returned models.
`recover_hpo_weights.py` preserves the original trainer and saves returned models
and actual preprocessing states, then checks original old/final float32
predictions and a saved-state reload. That attempt stopped at the owner's
request after the first member passed all checks; the remaining weights are
incomplete. The completed state and stop receipt are retained. No additional
training is queued or required for the verified assembly replay above.
