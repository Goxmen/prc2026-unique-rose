#!/bin/bash
# After chain.sh (pid 50283): nearest neighbours on the GPU, fast fits for the neighbour arms, then all scan scores.
cd /home/gunes/PRC
D=VALIDATION/runs/v3_20261007_similar_rows_jan_jul
while kill -0 50283 2>/dev/null; do sleep 20; done
TORCH_DISABLE_NATIVE_JIT=1 .venv/bin/python VALIDATION/similar_rows_jan_jul.py knn > $D/logs/knn.log 2>&1 || exit 1
.venv/bin/python VALIDATION/similar_rows_jan_jul.py scan > $D/logs/scan_knn.log 2>&1 || exit 2
.venv/bin/python VALIDATION/similar_rows_jan_jul.py scan-score > $D/logs/scan_score_all.log 2>&1 || exit 3
echo chain_knn done
