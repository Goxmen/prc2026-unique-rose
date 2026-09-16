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

- **Stage 3 is authorized and in progress.** Work is on
  `stage3-clock-features-residual`, based on Stage 2. The frozen design is in
  `VALIDATION/docs/CLOCK_MODELS.md`: T1 adds 18 features, T2 uses the same
  features with residual regression. Both retain recovery off, 800 rounds,
  three seeds and the fixed six-hour rule. All 51 tests pass before training.
  Read the plan's ledger and
  `VALIDATION/reports/STAGE2.md`. Select Q0 as the conservative parent:
  `v3_20260916T123230629423Z`, R_A 300.5904, R_B_long 256.1971. Q1
  (`v3_20260916T123433565511Z`) scores 296.9037/258.5505; versus Q0, it has
  a supported next-month gain and a supported longer-gap regression. Retain
  Q1 as an alternative, not a universal improvement.
- The user-authorized 4/5/6/7/8-hour sensitivity sweep is complete on both
  parents; see `VALIDATION/reports/STAGE2_THRESHOLD_SWEEP.md`. Four hours is
  best in this grid: B0-800 296.2785/255.6200, R3 292.8785/257.4787. Next-month
  gains versus six hours have pointwise support; longer-gap intervals touch
  zero. Keep it as a candidate; Q0/Q1 remain six-hour controls. Saved-prediction
  scoring took 1.621 minutes, zero fits. Do not rerun it without a reason.
- The requested 1/2/3-hour extension is also complete; see
  `VALIDATION/reports/STAGE2_THRESHOLD_SWEEP_LOW.md`. Four hours remains the
  best next-month/combined point estimate for both parents. One and two hours
  regress on those scores; three versus four is inconclusive. R3's two-hour
  longer-gap advantage is only 0.0744 s and inconclusive. Scoring took 1.631
  minutes, zero fits. Q0/Q1 remain six-hour controls; no threshold is promoted.
- Consolidated results: `VALIDATION/reports/STAGE2_THRESHOLD_RESULTS.md`.
  The plan records the user's future sequence: submit selected candidates to
  test validation-to-leaderboard transfer; if gains transfer, discuss excluding
  rule-handled rows from model training while retaining all evaluation rows;
  later consider learning corrections to the rule. These are future proposals,
  not started submissions or changes to the current training population.
- Stage 1's provisionally selected recovery model was R3:
  `v3_20260916T113256202842Z`, R_A 360.5041, R_B_long 333.3043. Its next-month
  gain versus control is supported; its longer-gap change is inconclusive and
  longer-gap ordinary flights regress. Preserve that tradeoff.
- Matched local control: `v3_20260916T095948645262Z`, **800 rounds**. The old
  1,500-round B0 (`v3_20260915T225301657023Z`) remains a separate historical run;
  its model/prediction artifacts were absent on this laptop, so the control was
  regenerated with the same data hashes. Keep round counts and identities clear.
- R1 exact recovery (`v3_20260916T102029635587Z`) and R2 relaxed recovery
  (`v3_20260916T104452470012Z`) have supported longer-gap regressions and are
  **not promoted**. R2/R3 were compared with both their immediate parents and
  B0-800. Coverage is not the promotion criterion.
- Q0 = B0-800 + fixed Rome six-hour rule; Q1 = R3 + the same rule. Both are
  complete and reuse saved predictions. Next: T1 then T2 from Q0, keeping
  recovery off and the rule fixed. No Stage 3 models have run. Do not repeat
  recovery audits or CPU tuning unless their relevant code/data changes.
- All 47 tests pass. Stage 1 recovery-off matrices exactly match frozen control
  source across six training windows and 16 validation folds. R3's model-source
  hashes match current code. B0-800's older wrapper hashes differ; preserve its
  frozen source and submission source guard for any later full-year refit.
- The shared pipeline implements `--recovery off|exact|relaxed|history`, with
  raw precedence, fold-scoped donors, ownership guards and causal/frozen static
  history. `pipeline/prediction_rules.py` supplies the fixed rule to both
  `VALIDATION/postprocess.py` and linked submission recipes. Rule and parent
  model hashes are separate. Every overridden ID/error is in candidate CSVs.
- The user authorizes local training/validation. Defer twelve-month refits and
  uploads until experiments finish and submission configurations are selected.
- **Hardware is machine-specific.** The current i7-13650HX/WSL laptop exposes
  20 logical CPUs and about 11.5 GiB RAM. Local measurements select one worker,
  six fit threads and 20 prediction threads. See `VALIDATION/reports/STAGE1_CPU.md`.
  Run full-data audits sequentially with experiments; overlapping them caused
  swapping. Do not copy the Stage 0 desktop's 7800X3D settings to another host.
- Existing `data/derived/` notebook caches are useful audit references, but are
  not fold-valid inputs. Their sections, counts, guards and reuse limits are in
  `VALIDATION/reports/STAGE1_DERIVED_DATA.md`. Do not rebuild them unnecessarily.
- Work is on `stage3-clock-features-residual`, based on the open Stage 2 PR
  branch `stage2-rome-six-hour-rule`. Track source, plans, concise results
  and charts; models, predictions, source snapshots and row-level audit parquets
  remain gitignored and local. A fresh clone must copy or regenerate them.

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
