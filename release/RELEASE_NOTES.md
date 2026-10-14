# Accepted final v83: reproducible ensemble assembly

This release preserves the accepted 670,790-row final submission as of 13 October 2026. The replay recomputes predictions from 28 retained component vectors, the portable anchor, saved coefficients and recorded constraints. It does not copy the final output.

All predicted values and Parquet bytes reproduce SHA-256 `f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f`. No hidden target labels, credentials, training or network access are required for replay.

Download `unique-rose-v83-replay.tar` and `SHA256SUMS`, verify the archive checksum, and extract it. Follow `docs/REPRODUCING.md` in the source repository. Full acquisition/method/attribution/environment records accompany the source.

This is verified retained-component ensemble replay, not a demonstrated clean raw-data training rebuild or arbitrary-new-month inference pipeline. The owner restored these exact v83 bytes as the selected final on 14 October after an intervening v80 refit. Provider data terms remain applicable to derived inputs; repository source is GPLv3.

This release was prepared privately. The owner authorized public repository and release publication at the challenge deadline, 14 October 2026 at 10:00 UTC (12:00 Amsterdam).

Restoration was accepted at 2026-10-14T08:36:43+02:00 on all 670,790 rows; the final numeric score remains unpublished. The replay archive and original source tag are unchanged.
