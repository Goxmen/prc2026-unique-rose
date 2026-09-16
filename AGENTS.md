# Agent notes — PRC Data Challenge 2026 (team unique-rose)

Predicting taxi-out times for the PRC Data Challenge 2026. Read `README.md` for
the competition rules, datasets, and submission format. This file covers the
local environment so a fresh agent session can work without rediscovery.

## Credentials & tooling (nothing secret lives in this repo — never commit any)

- **GitHub**: remote is `https://github.com/Goxmen/PRC`. Auth is handled by the
  `gh` CLI configured as git's credential helper — plain `git push` just works.
  The OAuth token lives in `~/.config/gh/hosts.yml`; verify with
  `gh auth status`. On gunes's WSL machine `gh` is at `~/.local/bin/gh`
  (add `~/.local/bin` to PATH); on macOS it's a Homebrew install on PATH.
  There is also a GitHub PAT for this repo in `.env` at the repo root
  (`GHP=...`) for direct API access when `gh` isn't available. `.env` is
  gitignored — it must NEVER be committed (repo goes public later).
- **Data (OpenSky MinIO)**: keys live in `~/.mc/config.json` under the alias
  `opensky` (endpoint `https://s3.opensky-network.org`). On WSL, `mc` is at
  `~/.local/bin/mc`. Datasets are NOT in git; sync with
  `mc mirror opensky/prc-2026-datasets/ data/` (~315 MB of parquet).
- **Python**: virtualenv at `.venv/` (not tracked); deps pinned in
  `requirements.txt`. Use `.venv/bin/python`, not system python (PEP-668 locked).

## Submissions

Fill `TAXITIME_SEC_mvt` in a copy of `data/submitting.parquet` (don't touch
`MVT_ID_mvt`), name it `unique-rose_v<N>.parquet`, upload with
`mc cp unique-rose_vN.parquet opensky/prc-2026-unique-rose/`.

## Leaderboard checks

When asked about our latest score, best score, score history, rank, or the
public leaderboard, run `.venv/bin/python pipeline/leaderboard_status.py
--json`. It reads the public competition API and our MinIO result files live;
the broken competition webpage is not needed. Distinguish the latest upload
from the best score. If live access fails, `--cached` is an allowed fallback,
but identify the result as stale and report its timestamp.

## Constraints

- Repo is private during the challenge but MUST be public under GPLv3 before it
  ends (11 Oct 2026) for prize eligibility: `gh repo edit Goxmen/PRC --visibility public`.
  Write everything as if it were already public.
- `data/` and `*.parquet` are gitignored; keep it that way. `data/FEATURES.md`
  is the one tracked exception (column docs).
- Team: gunes (GitHub `Goxmen`, WSL) and Ioana (GitHub `ioana2303`, macOS).
  `git pull` before working; branches + PRs for larger changes.
- `eda/feature_sweep.ipynb` follows the section rules in `eda/SECTION_STYLE.md` (skeleton,
  honest-holdout numbers, tables-over-prose, process: explain and get approval before
  restructuring a section). Read it before touching the notebook.

## Current improvement work — handoff

When asked to **continue the improvement plan**, read
[`VALIDATION/IMPROVEMENT_PLAN.md`](VALIDATION/IMPROVEMENT_PLAN.md), starting with
its "Resume here" section, and
[`VALIDATION/reports/STAGE0.md`](VALIDATION/reports/STAGE0.md).

- **Stage 0 is complete. Stage 1 / R1 (exact-twin flight-plan recovery) is next.**
  Do not repeat Stage 0 or jump to the Rome rule before the recovery experiment.
- Baseline run: `v3_20260915T225301657023Z`. Its results, checkpoint comparison,
  loss charts, source snapshots and predictions are saved under `VALIDATION/runs/`.
- Latest verdict: **800 rounds performed better than 1,500**. The saved B0
  record is the completed 1,500-round experiment; the plan explains how its
  saved 800-round predictions can supply a reference. Keep the chosen round
  count and baseline explicit; never attribute a simultaneous round-count
  change to recovery.
- The shared loader/provenance fields are implemented and feature equivalence
  was verified. Recovery matching itself is **not implemented**. The Rome
  six-hour rule has exploratory results only; it has not been submitted.
- The user authorizes local training/validation. Defer twelve-month refits and
  uploads until experiments finish and submission configurations are selected.
- The implementation, plans, concise results, and charts are maintained on
  `main`. Models, prediction parquets, and source snapshots remain gitignored
  and available on this machine; a fresh clone needs those copied or regenerated.

## Working cheaply — read this before touching the feature sweep

A full read of `eda/feature_sweep.ipynb` costs ~300k tokens and a carelessly built section another
~160k. A section should cost **60–80k**. Every tool call re-sends the whole conversation, so cost
scales with calls × context, not with the size of the thing you are looking at.

1. **Never read the notebook in full.** Read `eda/NOTEBOOK_MAP.md` (~4k) for orientation, then open
   only the cells you need. Never `Read` the `.ipynb` directly — it embeds 3.6 MB of figures.

   ```sh
   # the section index, with cell numbers
   .venv/bin/python -c "import json;[print(i,l) for i,c in enumerate(json.load(open('eda/feature_sweep.ipynb'))['cells']) for l in ''.join(c['source']).split(chr(10)) if l.startswith('#'+chr(35))]"
   # one cell's source, no outputs
   .venv/bin/python -c "import json,sys;print(''.join(json.load(open('eda/feature_sweep.ipynb'))['cells'][int(sys.argv[1])]['source']))" 104
   ```

2. **Measure in one script, not six.** Put every exploratory query for a section in a single
   labelled script and run it once. Cap every `LIMIT`, print aggregates, never dump raw rows.
   Redirect long output to a file in the scratchpad and `grep`/`sed` the part you need.

3. **Decide the whole cell list before the first clean-kernel run.** Write all the code cells *and*
   the list of numbers the findings will quote, then execute once. Each extra run–splice–reverify
   cycle costs ~30k. If you find yourself adding "one more piece of evidence", you planned too late.

4. **Copy the ladder helper, do not rewrite it.** Group-mean ladders must cascade rung by rung
   (`std_rungs` in §13 is the model). A helper that falls back straight to the airport mean produces
   nonsense on 4+ key groups and you will not notice until the numbers look absurd.

5. **Quote only reproducible numbers.** `mode()` and an unordered `LIMIT` break ties arbitrarily and
   change between runs. Add explicit tie-breakers (`ORDER BY n DESC, <key>`, or
   `QUALIFY row_number() OVER (PARTITION BY k ORDER BY n DESC, v) = 1`) anywhere a findings table
   quotes the result.

6. **Rebuild the derived files once, at the start** (~30 s) if `data/derived/` is absent, and check
   the row counts against `NOTEBOOK_MAP.md` before trusting anything downstream.

7. **Do not re-verify what a tool already told you.** No re-reading a file you just wrote, no second
   pass over output you have already seen. Check numbers against the executed outputs *once*, and
   remember that printed tables have no thousands separators.

8. **Say what a step will cost before an expensive detour.** Rebuilding derived data, a multi-minute
   run, or a full-file read: mention it in one line and continue, so the cost is visible.
