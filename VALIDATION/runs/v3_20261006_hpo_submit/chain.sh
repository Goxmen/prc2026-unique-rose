#!/bin/bash
# v54 refit, one part at a time on the desktop, then the submission file (no upload).
cd /home/gunes/PRC
L=VALIDATION/runs/v3_20261006_hpo_submit/logs
for part in nn catboost lgb airport; do
  .venv/bin/python VALIDATION/hpo_submit.py fit --part $part > $L/$part.log 2>&1 || { echo "$(date -Is) $part failed" >> $L/chain.log; exit 1; }
  echo "$(date -Is) $part done" >> $L/chain.log
done
.venv/bin/python VALIDATION/hpo_submit.py prepare --version 54 > $L/prepare.log 2>&1
echo "$(date -Is) prepare exit $?" >> $L/chain.log
