# Reports

- [Stage 0: baseline and round-count verdict](STAGE0.md).
- [Stage 1: recovery experiments](STAGE1.md).
- [Stage 2: fixed Rome six-hour rule, gains and damaging overrides](STAGE2.md).
- [Stage 2 follow-up: 4–8 hour rule sweep on baseline and R3](STAGE2_THRESHOLD_SWEEP.md).
- [Stage 1 laptop CPU and memory settings](STAGE1_CPU.md).
- [Notebook-derived data: section mapping and reuse limits](STAGE1_DERIVED_DATA.md).
- `benchmarks/`: measurements supporting the CPU execution settings.
- `sweeps/`: threshold-to-submission-ID mappings, recorded scores and loss charts.

These are saved snapshots, not live leaderboard data. For a fresh score check,
run `.venv/bin/python pipeline/leaderboard_status.py --json` from the repository
root. Detailed current baseline artifacts remain in `../runs/`.
