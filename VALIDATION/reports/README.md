# Reports

- [Stage 0: baseline and round-count verdict](STAGE0.md).
- `benchmarks/`: measurements supporting the CPU execution settings.
- `sweeps/`: threshold-to-submission-ID mappings, recorded scores and loss charts.

These are saved snapshots, not live leaderboard data. For a fresh score check,
run `.venv/bin/python pipeline/leaderboard_status.py --json` from the repository
root. Detailed current baseline artifacts remain in `../runs/`.
