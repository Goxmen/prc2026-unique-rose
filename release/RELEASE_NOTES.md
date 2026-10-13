# Accepted final v83: reproducible ensemble assembly

This release preserves the accepted 670,790-row final submission as of 13 October 2026. The replay recomputes predictions from 28 retained component vectors, the portable anchor, saved coefficients and recorded constraints. It does not copy the final output.

All predicted values and Parquet bytes reproduce SHA-256 `f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f`. No hidden target labels, credentials, training or network access are required for replay.

Download `unique-rose-v83-replay.tar` and `SHA256SUMS`, verify the archive checksum, and extract it. Follow `docs/REPRODUCING.md` in the source repository. Full acquisition/method/attribution/environment records accompany the source.

This is verified retained-component ensemble replay, not a demonstrated clean raw-data training rebuild or arbitrary-new-month inference pipeline. The ongoing four-month candidate selection has not replaced this accepted submission. Provider data terms remain applicable to derived inputs; repository source is GPLv3.

This draft is prepared in the private repository. Public release awaits the owner's visibility decision.
