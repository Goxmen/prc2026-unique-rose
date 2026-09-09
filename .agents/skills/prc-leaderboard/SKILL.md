---
name: prc-leaderboard
description: Check the live PRC Data Challenge 2026 score, unique-rose submission history, rank, or public leaderboard standings. Use for questions such as "what score did I get?", "how are we doing?", "show our score history", or "what is the leaderboard state?" within the PRC project. Do not use for model validation metrics or unrelated competitions.
---

# PRC Leaderboard

Use the repository's read-only status command so every answer reflects a fresh
snapshot of both authoritative sources: the public competition API and the
`prc-2026-unique-rose` MinIO bucket.

## Workflow

1. From the PRC repository root, run:

   ```sh
   .venv/bin/python pipeline/leaderboard_status.py --json --top 10
   ```

2. Answer the user's requested scope from that output. Always include the
   snapshot timestamp. Scores are RMSE seconds and lower is better.
3. For "my score" or "what did I get", report the latest upload's status and
   score, then separately state the best score/file when they differ. Never
   describe the best score as the latest score.
4. For history, preserve upload order and describe positive
   `improvementVsPrevious` as improvement; a negative value is a regression.
5. For public standings, explain that rank is based on each team's best score.
   Include unique-rose's rank among scoring teams, registered-team count,
   leader, podium threshold or gap, and any top/nearby rows relevant to the
   question. Add `--all` only if the user asks for the complete standings.

If the live command fails, run the same command with `--cached`. Clearly label
the result as stale and give its `updatedAt`; do not silently present cached
data as current. Do not open the competition webpage, print credentials, alter
the MinIO configuration, upload submissions, or mutate external state during a
score check.
