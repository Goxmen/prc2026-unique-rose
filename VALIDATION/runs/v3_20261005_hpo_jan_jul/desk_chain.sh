#!/bin/bash
# Desktop side of the HPO split: wait for the running LightGBM search, then per-airport,
# seed checks (networks, LightGBMs), pull CatBoost from the laptop, and the final report.
cd /home/gunes/PRC
while pgrep -f "hpo_jan_jul.py lgb" > /dev/null; do sleep 60; done
.venv/bin/python VALIDATION/hpo_jan_jul.py all --stages airport confirm pull report >> VALIDATION/runs/v3_20261005_hpo_jan_jul/logs/driver.log 2>&1
