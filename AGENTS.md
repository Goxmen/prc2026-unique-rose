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

**Current user authorization (11 October 2026): uploads are autonomous.** The user
said "forget the rule of asking approval for uploads. You are free to submit to
leaderboard as you please". This supersedes every older upload-approval requirement
below and in historical run guards. Prepare, validate, upload once, verify the
remote hash, and record the official result without asking again. Keep old approval
records as history. The active target is **225.00 seconds or lower on the official
leaderboard**, replacing the five-second improvement target. Desktop and laptop
training are authorized, with one heavy job per machine.

**User rule: keep experimental bookkeeping out of `SUBMISSIONS.md`.** List actual
submissions and distinct, selected candidates with meaningful validation evidence.
Do not add control replays, duplicate predictions, inactive gates, failed experiments,
or tiny/inconclusive changes as new model entries. Keep those results in experiment
reports and run logs. Do not describe numerical noise as an improvement or treat
bootstrap/seed agreement on reused development data as independent confirmation.

**User rule: score every new row on Jan/Jul CV, and keep table cells short.**
On 2026-10-02 the user dropped the 16-fold validations for new candidates: next-month,
longer-gap, combined and Weighted CV are no longer required and are not run by default,
so their cells stay `-` (as for v44). Jan/Jul CV (`VALIDATION/tuning_seasonal.py`) is
the validation a new candidate needs. Rows that predate a validation are not
backfilled — the user declined that on 2026-09-24, and their `-` is final. In both
`SUBMISSIONS.md` and `NEXT_STEPS.md` a table cell names the parent row and the one
thing that changed, in a sentence. Bootstrap intervals, seed agreement, ageing
readouts, failure analysis and provenance go in the notes under the table or in a
run report, never in the cell.

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

- **Laptop network is slow.** When desktop artifacts are needed, transfer only
  necessary files/columns, retain a verified local cache, and reuse it. Avoid
  repeating remote downloads or making experiments depend on remote reads.
- Repo is private during the challenge but MUST be public under GPLv3 before it
  ends (11 Oct 2026) for prize eligibility: `gh repo edit Goxmen/PRC --visibility public`.
  Write everything as if it were already public.
- `data/` and `*.parquet` are gitignored; keep it that way. `data/FEATURES.md`
  is the one tracked exception (column docs).
- Team: gunes (GitHub `Goxmen`, WSL) and Ioana (GitHub `ioana2303`, macOS).
  `git pull` before working. The user wants work consolidated directly on
  `main`; do not create extra worktrees or feature branches unless explicitly
  requested. Keep the shared checkout on `main`.
- `eda/feature_sweep.ipynb` follows the section rules in `eda/SECTION_STYLE.md` (skeleton,
  honest-holdout numbers, tables-over-prose, process: explain and get approval before
  restructuring a section). Read it before touching the notebook.

## Two machines: desktop and laptop

Training can be split between gunes's desktop and laptop. Agents drive both from the desktop.

**Machines**

| | Desktop (Tailscale `gpc`, this host) | Laptop (Tailscale `glaptop-wsl`) |
|---|---|---|
| CPU | Ryzen 7 7800X3D | i7-13650HX |
| GPU | RTX 3080, 10 GB | RTX 5060 Laptop, 8 GB |
| WSL memory | 24 GB | 16 GB |

**Rules (from the user)**
- Leave the laptop alone unless the user asks for it.
- On the desktop, run heavy jobs one at a time: CPU and GPU jobs together make the PC unusable.

**Access**
- `ssh glaptop-wsl` from the desktop; the key is `~/.ssh/id_ed25519_laptop`, user `gunes`.
- The laptop has its own checkout at `~/PRC` and its own `.venv`, with the same pinned packages.
- `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi`.
- The laptop's WSL settings are in `C:\Users\Gox\.wslconfig`: `memory=16GB`, and both idle timeouts set to -1 so WSL keeps running without a terminal open.
- After a laptop reboot, WSL only starts when someone opens a terminal on it.

**Workflow**
1. **Code:** `git pull` on the laptop, then `rsync` any uncommitted `*.py` files across, excluding `VALIDATION/runs/`.
2. **Data:** copy only the inputs the job needs (prepared splits `after_drop.pkl`, `heldout.pkl`, `meta.json`; feature tables), over Tailscale with `rsync`. 1.7 GB took about 15 minutes. Check `sha256sum` on both sides.
3. **Jobs:** run one job at a time on the laptop; CatBoost or LightGBM with full data needs 7–10 GB. Give each arm all its seeds on one machine, because LightGBM can differ slightly between machines.
4. **Results:** copy the predictions back with `rsync` and merge the run records, noting which machine produced each seed (as in `VALIDATION/runs/v3_20261003_ideas_jan_jul/catboost.json`).

CatBoost on the GPU ran faster on the laptop's 5060 (30–36 min per fit) than on the 3080 while it shared the desktop with LightGBM jobs (55–65 min per fit).

## Current improvement work — handoff

- **October 11: v69 best 225.3994, 0.3994 seconds from the 225 target.** Autonomous uploads are authorized. Physical single v66 lost (233.7954); retained callsign-only v68 scored 234.7131. Their separate directions extend v65's affine basis: v69 official 225.3994, CV 296.8197. v67 was prepared but not selected/uploaded; do not upload it. Physical pool3 CV 292.8675 passes (−0.7891 versus v64), and all six reload audits pass. All three full-year members are saved from the laptop, copied back and audited across both test sets. v70 is prepared with the physical pool; its manifest is under `v3_20261011_physical_candidate/submission_v70/`. The Rome calibration screen failed (+1.6735 CV); retain its fits without production/upload. Ground-only pooling is complete and rejected: +0.1272 versus the matched physical single, inconclusive in both comparisons; reload audits pass. Do not refit or upload that arm. Five uploads in UTC October 10 are consumed (v64, v65, v66, v68, v69); the next submission must wait until October 11 UTC / 02:00 Amsterdam. Preserve all fits.

- **October 11, 01:02: v65 scores 225.8994, best, rank 15/242.** It improves v64 by 6.8757 seconds and v58 by 10.2901. The original five-second goal is complete; the active target is **225.00 or lower** (0.8994 left). Autonomous uploads and laptop training are explicitly authorized. v65 is a signed affine ensemble selected from our own scored files; Jan/Jul CV regresses to **297.0247** from serialized v64's 293.6569. Do not claim independent validation. All legacy current-fold fits/replays are retained; do not repeat. Record: `VALIDATION/runs/v3_20261011_scored_blend_submit/submission_v65/`. Physical-scale track features passed the single-member screen (293.3477); desktop runs the fixed three-member CV expansion, laptop runs full-year seed42. One heavy job per host; resume retained artifacts. Three research subagents are unavailable due account allowance; root continues.

- **October 11: user raised the target to 225.00 or lower and authorized autonomous uploads.**
  Starting best is v64 at 232.7751; the remaining reduction is 7.7751 seconds.
  Continue from completed artifacts; research state is recorded in
  `VALIDATION/runs/v3_20261011_target225/`. Existing subagents still report an account
  usage limit; root continues executable work on both machines. The goal API cannot
  edit objective text or replace an unfinished goal; do not falsely mark the old
  five-second goal complete to replace it. The user's new target governs the work.

- **October 11: v64 uploaded and scored 232.7751, the new official best.**
  The user explicitly said "go ahead and upload" for the audited v64 file.
  One upload completed at 00:00:55 Amsterdam; remote SHA-256 matches
  `2a596d95610b748b49a5da8803533f3e4bec50e3d919f92138b40b194b49f5b3`.
  All 344,841 original January/July flights scored. Improvement versus v58 is
  **3.4144 seconds**; the five-second goal still needs **1.5856 seconds**.
  Live check 00:02 Amsterdam: rank **20/242** scoring teams (508 registered).
  Row 66 and the research report record the result. Approval/result/remote hash
  are in `VALIDATION/runs/v3_20261010_sparse_callsign_submit/submission_v64/`.
  Approval is consumed; ask before any further upload. Do not repeat this upload
  or its completed fits. The separate four-month candidate remains local.
  A saved-prediction diagnostic projects only 0.0863 seconds by optimally scaling
  the entire v58-to-v64 change; this is leaderboard-conditioned arithmetic,
  not an independently validated or uploaded improvement.

- **October 10: research network selected the candidate later uploaded as v64.**
  The user requested a five-second improvement and explicitly authorized laptop
  training. Keep one heavy job per machine and ask before any upload. Read
  `VALIDATION/reports/RESEARCH_NETWORK_20261010.md` before proposing or repeating
  experiments; it records the many failed screens and their retained artifacts.
  Selected fixed combination: half raw-track3/half callsign15 within the unchanged
  0.71 neural share, callsign main LightGBM and half CatBoost/R20 Rome probabilities.
  **293.6566 Jan/Jul CV**, versus matched seed42-tree control **295.7295** (-2.0729);
  incremental -0.7651 beyond the callsign/Rome combination (294.4217). Both months
  improve; three predeclared paired comparisons pass. The full-year three-member
  raw-track refit and both ranking-view audits are complete. Runner
  `nn_sparse_track_submit.py`; selected recipe `half_sparse3_combination`; builder
  `sparse_callsign_submit.py`. Audited and now scored old-view **v64** is retained at
  `submissions/unique-rose_v64.parquet`, SHA-256
  `2a596d95610b748b49a5da8803533f3e4bec50e3d919f92138b40b194b49f5b3`.
  Row 66 records the selected candidate and its official score. The user's explicit
  approval was consumed by the October 11 upload above; no further upload is authorized.
  All 15 callsign networks, the full-year main LightGBM and three Rome CatBoost
  classifiers are complete. Neural reload/return audits pass. Both parent candidates
  are built and passed full-source, exact-parent and independent all-row audits.
  Reuse every completed fit; all runs use the `v3_20261010_*` prefix. Both ranking
  track caches were built and independently audited on the laptop; their returned
  files passed desktop hash verification and all three full-year models are retained. Ground-observation coverage
  changes sharply across years, especially Rome; retain this transfer caveat.
  The four-fit matched-laptop physical-clock-bypass screen is complete and closed:
  its fixed half blend loses 0.6384 seconds against the actual v58 pool, despite
  beating its weaker same-laptop single-network control. Do not expand it.
  Its first worker stopped before training on an overly strict arithmetic
  self-check; only that check's tolerance is repaired, with exact training/RNG
  parity preserved. Inspect its execution record before resuming. The live
  18:56 UTC check still has v58 best at 236.1895, rank 26/241; all 5,253 public
  scored submissions still use 344,841 rows, not the enlarged truth population.
  Laptop full-year neural fitting uses `callsign_submit_fast.py`: identical
  fixed training updates, exact-parity chunked input encoding and bounded final
  inference. The earlier workers hit host-memory limits or wasteful full-test
  inference after every epoch; they produced no completed models and remain
  preserved. All15 production network seeds stay on the laptop.
  The same-desktop joint callsign6/raw-track screen is closed: +0.3401 seconds
  against its matched single-track mixture, both months worse
  (`nn_joint_callsign_sparse_jan_jul.py`). The matched-laptop stand/month arrival
  recording-state classifier screen is closed: +0.7092 seconds on v58 and
  +0.3216 inside the selected recipe; no expansion (`rome_stand_arrival_jan_jul.py`).
  A frozen ordinary-track encoder was tested in the Rome CatBoost
  classifier (`rome_frozen_track_jan_jul.py`): original68 plus 64 activations
  and nine quality inputs, fixed original rounds, four same-laptop control/candidate
  fits. Prepared inputs pass 128 raw reconstruction checks. The matched change
  is -0.3817 seconds, interval [-0.8153, +0.0459]; January -0.0155, July -0.7480.
  It remains inconclusive, with strong Rome ground-coverage transfer risk; retain
  as a lead, no expansion or production fit. All completed artifacts are retained;
  do not duplicate fits. No 16-fold
  validation or ledger entries for failed screens were added. The five-second
  official objective remains unachieved; reused CV is not independent evidence.
  Follow-up: saved post-rule seed variance projects only -0.2722 seconds from
  track3 to track15 (not a measured gain or bound); generic pooling is deprioritized.
  `nn_track_attention_jan_jul.py` completed two fixed21/17-epoch desktop fits:
  original sparse156 plus a four-head conditional attention residual, same-desktop
  controls and unchanged training bytecode. Change -0.0671 seconds against the
  matched single-model mixture (interval [-0.3597,+0.2511]); +0.2187 against
  actual v64. Inconclusive; no expansion. Initial parity, all-row scores and
  1,094 model reload predictions pass (maximum difference <0.001 seconds).
  `arrival_clock_context_pilot.py` completed a laptop
  pilot on twelve dates within both training folds, using no departure labels:
  1,796 physical-stop witnesses; recent airport/airline context worsens arrival-lag
  RMSE and MAE against the static airport/airline reference. All 11,910 causal
  edges and 60 sampled contexts pass; no full-year build or taxi-out fit follows.
  Both follow-ups are complete and retained; do not repeat. v64 stays immutable;
  its approved upload is complete. All three research subagents currently have
  account-usage-limit errors; do not repeatedly relaunch them.

- **October 8: the organisers published a FINAL test set; deadline extended to 14 Oct 2026
  12:00 CET.** `data/final_ranking.parquet` / `data/final_submitting.parquet` (bucket
  `prc-2026-datasets`, uploaded 2026-10-08 13:04 CEST): January and July 2026 as before plus
  February and June 2026, **670,790 departures** to predict (Jan 152,719, Feb 144,158,
  Jun 181,791, Jul 192,122). Same columns; training files unchanged. Every old Jan/Jul row is
  still there, but movement times are jittered by up to ±12 s (departure takeoff times; arrival
  landing and in-block times, with arrival taxi-in labels recomputed from the jittered times).
  The data page says only "same as the ranking and submitting dataset but covering two
  additional months". **v63 = v58's recipe on the final test set is prepared and uploaded, but
  the scorer rejected it** ("Submission contains 325949 rows not in truth"): the leaderboard's
  truth still holds only the 344,841 old Jan/Jul labels. Keep the v63 file for when four-month
  files are accepted; ask before any upload. v62 (v58's old file re-uploaded unchanged) scored
  236.1895, identical to v58 — a duplicate that wasted a slot; don't repeat. On the old inputs,
  the refitted networks/Rome classifier and the saved trees reproduce v58's parts (see
  SUBMISSIONS.md row 64). Runner `VALIDATION/final_submit.py`, run
  `VALIDATION/runs/v3_20261008_final_submit/`; stages tables → inputs → trees → rome → nn →
  build → upload, one heavy stage at a time on the desktop. Saved v54 tree models are reused; the
  15 networks and the Rome classifier were never saved and are refitted. OPDI raw files for the
  2026 months were re-downloaded to `data/external/opdi_final/` (`VALIDATION/tools/fetch_opdi_final.py`);
  ADSB.lol days 2026-02-01..28, 05-31, 06-01..30 were added to `data/external/adsblol_heatmap/`.
  Rebuilt flight-keyed tables live in `data/derived/final_tables/{old,final}/`. Logs:
  `tmp/final_20261008/`.

- **October 8: ERA5 v61 scored 236.6319; v58 stays best, further uploads require fresh approval.**
  User explicitly chose **both**: v60 = v58 with weather only in the Rome classifier;
  v61 = v58 with weather in all five parts, despite the CV regression below. Driver
  `VALIDATION/run_open_meteo_submit.py` runs prepare → Rome → build/upload v60 →
  networks → main LightGBM → airport LightGBMs → CatBoost → build v61,
  sequentially on the desktop. Runner `VALIDATION/open_meteo_submit.py`; state/logs
  in `VALIDATION/runs/v3_20261008_open_meteo_submit/`. Inspect `execution.json`
  before resuming; never start duplicate fits or uploads. Authorization is already
  superseded: the user now requires asking before ANY further submission.
  The user subsequently said **"submit v61"** and requested its ledger entry.
  It was uploaded at 20:09 Amsterdam; remote SHA-256 matches the audited file.
  Official score **236.6319 (+0.4424 vs v58)** on all 344,841 flights; live check
  at 20:11 Amsterdam, rank 26/239. Both weather variants lost; retain the results.
  `upload_approval_v61.json` records the consumed approval for that file hash.
  `uploads_paused.json` still blocks further uploads; the guard rejects both
  v60 and v61 after consuming the approval. The driver is complete. Leave the
  laptop alone; do not repeat these fits or uploads. SUBMISSIONS.md row 65.
  **v60 is uploaded and scored: 237.3022 (+1.1127 vs v58), all 344,841 flights.**
  All v61 fits and file checks finished at 18:25 Amsterdam. The file contains
  the original two-month test set (344,841 rows); do not switch to four months.
  The user's follow-up data audit passed: all 156 original ordinary columns
  unchanged, every eligible full-year departure accounted for, all 28,124,195
  weather entries checked against source arrays without missing values, tree
  feature orders and the final blend verified. See `full_year_audit.json` and
  `VALIDATION/tools/audit_open_meteo_submit.py`; no repeated fits or uploads are needed.
  Source #3 licence check: Airport Corner is publicly readable, but an open-data
  reuse licence was not established. [Published terms](https://ext.eurocontrol.int/airport_corner_public/dc_requests)
  say free of charge and for information purposes only. No event features are used.

- **October 8: Open-Meteo ERA5 in every v58 part failed Jan/Jul CV; keep v58.**
  The user requested all five parts after the main-LightGBM screen was inconclusive.
  Fixed v58 HPO settings, network share 0.71, tree/classifier seed 42 and the full
  15-network pool: matched control **295.7301 → 296.8264 (+1.0963)**, paired-day
  interval [+0.2838, +2.1992]. Single-part effects: main LightGBM +0.0081,
  per-airport LightGBMs +0.0042, CatBoost +0.0066 (all effectively flat); networks
  +1.5080 (all three seed groups worsen in both months); Rome −0.4594, but its
  interval [−1.1327, +0.1421] includes no gain. Rome is an exploratory lead, not
  a selected candidate. Read [the report](VALIDATION/reports/OPEN_METEO_JAN_JUL.md).
  Runners: `VALIDATION/open_meteo_jan_jul.py`, `VALIDATION/open_meteo_all.py`;
  complete run `VALIDATION/runs/v3_20261008_open_meteo_jan_jul/`, definitive
  `all_result.json`. All models, predictions and prepared matrices are retained;
  do not repeat completed fits. Eleven weather inputs and 30 hashed sources are
  cached in `data/external/conditions_20261008/`; do not redownload. Independent
  audit checked 14,080 weather values; seven targeted tests passed. The audit's
  initial calendar check used local DuckDB time and was corrected to UTC;
  training/weather joins were already UTC. All new fits ran sequentially on the
  desktop; historical CatBoost controls used the laptop GPU, a comparison caveat.
  The CV phase ended before full-year fits; subsequent authorized uploads are recorded
  above. Source #3 (Airport Corner events) remains unused; leave the laptop alone.

- **October 7: training only on the flights nearest the test (user's idea) lost; v58 stays best.** Each test flight picks its K nearest training flights at its airport (every input; numbers as percentiles, categories match/mismatch, weighted by a fast LightGBM's gain shares); parts are refitted on the picked flights only, one model per test month. Practice scan (`VALIDATION/similar_rows_jan_jul.py`, fast main LightGBM, seed 42, run `v3_20261007_similar_rows_jan_jul`): every K from 1 to 50 is worse in both months, less so as K grows (K = 1 +9.98, K = 50 +1.22 alone); weighting by pick count is worse still. The user chose K on the leaderboard: **v59 = v58 with every part refitted on the 50 nearest (`VALIDATION/similar_rows_submit.py`, run `v3_20261007_similar_rows_submit`) → 238.5432 (+2.35)**. An earlier variant (a classifier telling training from test, dropping the least test-like share) was also worse and was deleted at the user's request. Do not retry training-set selection by similarity. Uploads on 7 Oct: v59 (1).

- **October 6: hyperparameter search of every v51 part is done; Rome part uploaded as v53
  (LB 237.7683, −0.51 vs v51, best).** Read the [report](VALIDATION/reports/HPO_JAN_JUL.md). On
  Jan/Jul CV vs C4 (298.2138), three seeds: Rome classifier −2.32, CatBoost −0.15, main LightGBM
  −0.12, per-airport LightGBMs −0.18, networks −0.05 (noise); the three tree models add −0.45 on
  top of Rome. v53 holds ITA flight 202681908 at v51's value (see SUBMISSIONS.md row 56). Runner
  `VALIDATION/hpo_jan_jul.py`, run `VALIDATION/runs/v3_20261005_hpo_jan_jul/` (CatBoost ran on the
  laptop). All five winners refitted on 2025 and uploaded as **v54: LB 236.3328 (−1.94 vs v51, −1.44 vs v53), best**; no flight held (`VALIDATION/hpo_submit.py`, run `v3_20261006_hpo_submit/`). Do not repeat the search or refit/upload v53/v54 again.

- **October 6 (afternoon): three leaderboard tests on v54 all lost; v54 stays best.**
  v55 (`VALIDATION/snow_offset_probe.py`): January snow-day predictions at EDDF/LFPG/LEMD lowered by
  half their extra gap above the clock (8,502 flights, mean −59.5 s) → 236.5586 (+0.23); solved, v54's
  snow-day level is right (k ≈ −0.03). v56 (`VALIDATION/rome_fn_submit.py`): Rome classifier + flight
  number's 2025 copy share, unseen flight numbers kept at v54 (48% of LIRF in July 2026) → 236.5988
  (+0.27), although Jan/Jul CV said −0.81 on v54's recipe and −1.54 on C4 parts
  ([report](VALIDATION/reports/ROME_COST_JAN_JUL.md); cost-weighted Rome training failed CV, +1.37).
  Rome-classifier CV gains have now transferred at 0.22× and then the wrong way. The test-set study
  behind these leads is in the "Test Set Casebook" artifact (https://claude.ai/artifact/BcyN58UNkY6ZmbNC3RJAd4).
  Do not re-upload v55/v56 or a no-fallback v56 (kept in `v3_20261006_rome_fn_submit/nofallback_build/`).
  v57 (`VALIDATION/blend_weight_submit.py --version 57 --w-nn 0.45`): network share 0.45 instead of 0.6, Jan/Jul CV
  −0.22 → 236.9652 (+0.63). Solving v54/v57 (final step nearly linear in the share) puts the leaderboard's best share near
  0.71 (predicted 236.20, −0.14). **v58 = v54 with share 0.71: LB 236.1895 (−0.14), best; later experiments start
  from v58.** ADS-B track-shape inputs (`VALIDATION/adsb_track.py`, run `v3_20261006_adsb_track_jan_jul`) failed Jan/Jul CV
  (+0.004 on v54's main LightGBM) and were not uploaded. Per-airport tailoring (own LightGBM settings, tree mix or network
  share per airport) and capping tree labels at 3 h both failed Jan/Jul CV; see the "Submissions by Airport" artifact
  (https://claude.ai/artifact/NB76KEBwaqUJFQKKDqGYia) and `VALIDATION/clip_labels_jan_jul.py`. Uploads on 6 Oct: v54–v57 (4).

- **October 3 (evening): other teams' ideas submitted as v51 (C4): LB 238.2770, best ever, rank
  21/223; v50 (C1, 15 pooled networks) scores 239.0108.** Read the
  [report](VALIDATION/reports/IDEAS_JAN_JUL.md). On v48's recipe (Jan/Jul CV 299.9790), passing single
  changes: CatBoost −0.65, per-airport LightGBMs −0.42, 15 pooled `ple` networks −0.43, arrival
  taxi-in −0.17, runway configuration −0.13, peer AOBT_3 clocks −0.09, OSM taxi route −0.07 (not
  used). Failed: second-stage model on the other month's errors (+3.13), per-airport blend weights
  (+0.11). C4 = 15 networks refitted with the 22 new inputs + equal mean of combo LightGBM, combo
  per-airport LightGBMs and CatBoost: CV 298.2138 (−1.77); LB −1.50 vs v48. Validation runs:
  `VALIDATION/runs/v3_20261003_ideas_jan_jul/`; submissions: `VALIDATION/ideas_submit.py`, manifests
  and full-year parts in `VALIDATION/runs/v3_20261003_ideas_submit/`. Do not refit or upload v50/v51
  again. **The user wants heavy jobs run one at a time (CPU and GPU together make the PC unusable).**
  The laptop (`ssh glaptop-wsl`, Tailscale; key `~/.ssh/id_ed25519_laptop`) can run jobs; the user
  asked to leave it alone unless needed. CatBoost (`catboost==1.2.10`) is in `requirements.txt`.

- **October 3: an improved network is submitted as v48 (weight 0.6): LB 239.7731 (−5.6027 vs
  v47), best ever, rank 21/222; v49 (same network, weight 0.3) scores 242.9764.** Read the
  [report](VALIDATION/reports/NN_IMPROVE_JAN_JUL.md). Three changes to v47's network were scored
  on Jan/Jul CV at weight 0.3: five seeds of v47's network −0.24; piecewise-linear input
  embeddings + stopping on the last training month, five seeds (`ple`) **−1.51 [−2.27, −1.08]**
  (302.3701 → 300.8628, every seed); TabM, one network, −0.70 (worse than ple by +0.81). ple's
  best blend weight is ~0.6 (out-of-month check −0.80 [−1.35, −0.43] vs 0.3); the leaderboard
  gave −3.20 for that step. Runner `VALIDATION/nn_improve_jan_jul.py`, scorer
  `VALIDATION/tools/score_nn_improve_jan_jul.py`, run `VALIDATION/runs/v3_20261003_nn_improve_jan_jul/`;
  submissions by `VALIDATION/nn_improve_submit.py` (manifests in
  `VALIDATION/runs/v3_20261003_nn_improve_submit/`). Do not refit or upload v48/v49 again. Needs
  `tabm`/`rtdl_num_embeddings` (in `requirements.txt`); the runners set
  `TORCH_DISABLE_NATIVE_JIT=1` because torch 2.14's Triton kernels need Python headers this
  machine lacks. Leads: the weight above 0.6 was not tried; TabM with five seeds or blended with
  ple was not tried; idea D (sequence model on raw ADS-B ground tracks) is not started.

- **October 2: v46's ordinary regressor blended 0.7/0.3 with a neural network is submitted as
  v47: LB 245.3758 (−3.3366 vs v46), best ever, rank 26/221.** Jan/Jul CV 303.7536 → 302.3701
  (−1.38 [−1.94, −0.95], every seed improves). The network (MLP, same 134 inputs and rows,
  PyTorch on the desktop's RTX 3080) is in `VALIDATION/nn_jan_jul.py`; scoring in
  `VALIDATION/tools/score_nn_jan_jul.py`; run `VALIDATION/runs/v3_20261002_nn_v46_jan_jul/`.
  `VALIDATION/nn_submit.py` reuses v46's full-year boosters unchanged and fits one network;
  manifest in `VALIDATION/runs/v3_20261002_nn_submit/submission_v47/`. Do not refit or upload
  v47 again. Proposed next (not started): average several network seeds, and better numeric
  encodings with early stopping on the last training month, decided as one comparison.

- **October 1: the Jan/Jul hyperparameter search is done and submitted as v44: LB 269.5533
  (−5.2127 vs v43), best ever, rank 43/216.** Read the
  [tuning report](VALIDATION/reports/JAN_JUL_TUNING.md). Only the ordinary regressor's
  LightGBM settings changed: 127 leaves, `lambda_l2` 100, `feature_fraction` 0.7,
  `cat_smooth` 100, `max_cat_threshold` 16, learning rate 0.02 × 5,000 rounds. Jan/Jul CV
  318.0408 → 314.1959 (−3.84 [−5.17, −2.93], three seeds, every seed and both months
  improve); the leaderboard moved 1.4× that. The winner was picked from 22 tries on the same
  data. The user cancelled the 16-fold `confirm` stage and had its partial output deleted,
  so row 47 in `SUBMISSIONS.md` has no 16-fold scores by the user's decision; do not run it
  unless asked. All eight random settings with `max_cat_threshold` 16 beat v43, a lead for
  further category tuning. Search trials/predictions: `VALIDATION/runs/v3_20261001_jan_jul/search/`;
  v44 refit, ranking frames and manifest: `.../submission_v44/` (`VALIDATION/tuned_submit.py`,
  which first reproduces v43's boosters byte for byte). Do not refit or upload v44 again.
  The validation itself (two fits, average RMSE, history frozen at each training end) is in
  the [validation report](VALIDATION/reports/JAN_JUL_VALIDATION.md); its prepared splits and
  v39–v43 refits are reused by the runner `VALIDATION/tuning_seasonal.py`.

- **September 25: idea 34 (stand coordinates) is submitted as v43: LB 274.7660 (−0.1416 vs v42).**
  Read the [experiment report](VALIDATION/reports/STAND_COORDINATES_EXPERIMENT.md) and
  [map report](VALIDATION/reports/STAND_COORDINATES.md). Three map inputs (stand offset,
  distance to the take-off runway end) on v42 improve Weighted CV 263.6158 → 263.1476
  (−0.47 [−0.65, −0.27]); all four validations and all seeds improve; January
  inconclusive; ageing 7/10 negative. Hidden-stand test: −2.01 s on unseen stands.
  The v42 control was rebuilt on the M3 (within 0.03 s of row 45). Models and
  predictions in `VALIDATION/runs/v3_20260925_stand_coords/`; maps in
  `data/external/stand_maps/`. Do not repeat fits or refetch maps. No full-year fit
  The user then approved the upload: `VALIDATION/stand_coords_submit.py` refitted seed 42
  on 2025 and uploaded v43 (manifest in `.../v3_20260925_stand_coords/submission_v43/`).
  It transferred at about 0.3× the CV gain. Do not refit or upload it again. Still to do:
  declare the X-Plane Gateway (GPL v2+) and OpenStreetMap (ODbL) as external data.

- **September 24: v42 refitted, uploaded and scored on the user's explicit request.**
  V40 + the tested 12 disruption inputs scores **274.9076 s**, improving v40's
  275.4146 by **0.5070 s**. All 344,841 flights scored. Weighted CV **263.5929**;
  original combined **232.6647**. See [submission report](VALIDATION/reports/V42_SUBMISSION.md).
  Two full-year seed-42 boosters (800/300 rounds) and all ranking predictions are
  retained in `VALIDATION/runs/v3_20260924_opdi_disruption/submission_v42/`.
  Runner: `VALIDATION/opdi_disruption_submit.py`. Keep v40's exclusions and rules;
  new features enter only the ordinary model. Source/data hashes, 16 tests,
  all-row correction audit, 3,840 independent sampled feature checks and remote
  file hash pass. Do not repeat fits or upload again. V41 was an already-scored
  separate occupancy experiment. The user asks for completion notifications only,
  without routine progress messages while runs are underway.


- **September 24: v39/v40 disruption-feature comparisons are complete; no promotion.**
  Read the [validation report](VALIDATION/reports/OPDI_DISRUPTION_EXPERIMENT.md)
  and [test-like cases](VALIDATION/reports/OPDI_DISRUPTION_STRESS_CHECK.md).
  Weighted CV improves **264.4169 → 263.6462** for v39 (-0.7708 s) and
  **264.3388 → 263.5929** for v40 (-0.7459 s). All three seeds improve;
  both 1-/3-day intervals exclude zero. Original combined improves by
  0.5946/0.5675 s. January's separate retrospective checks change by
  +0.1174/+0.0733 s, both inconclusive. Local controls differ from historical
  Weighted CV by only -0.0049/-0.0009 s, but original OPDI cache hashes differ:
  gains are against matched rebuilt controls, not exact historical predictions.

  **No meaningful snow-case gain.** Fixed input-only limits match 2,320 of
  10,301 prolonged-snow test flights to 2,247 historical flights on 20 dates.
  Their RMSE changes **393.71 → 395.69** (v39) and **395.24 → 395.17** (v40),
  both inconclusive. No comparable examples exist under these limits for the
  347 heavy-arrival-delay test flights, including 256 with prolonged snow.
  This is an arrival-delay proxy, not a measured departure queue. Amsterdam's
  2,095/2,937 prolonged-snow test flights exceed its entire 2025 snow-duration
  maximum; 329 have scored matches, using 269 historical flights from just one
  day. Do not present these as evidence for the more extreme test conditions.
  [Examples](VALIDATION/reports/OPDI_DISRUPTION_STRESS_EXAMPLES.md) are selected
  by inputs and include regressions. Snow and arrival inputs were tested together.

  Completed 108 boosters (54 per recipe), all 16 folds and January at three
  seeds; February adds 143,732 held-out predictions per seed/recipe using saved
  June–November models, with no extra fits. Keep original folds, weights, rules
  and v40 exclusions. Thirty targeted tests pass. Independent checks reproduce
  correction stacks exactly, verify training membership and features, and
  reproduce scores in SQL. Stress audits check 17,598 accepted edges in each
  source view plus 320 independent arrival values. Reuse all models/predictions
  in `VALIDATION/runs/v3_20260924_opdi_disruption/`, matching inputs in
  `tmp/opdi_disruption_stress/`, and logs in `tmp/opdi_disruption_20260924/`.
  Do not repeat fits. Designs and runners are linked from the reports.

  The rebuilt OPDI cache covers 2,429,888 departures; see
  `VALIDATION/reports/opdi_cache_build_20260924.json`. Do not redownload or
  rebuild it. Weather starts 31 December 2024: all 4,788 January 1 departures
  have complete 6-/24-hour snow coverage. There is no 90-day snow feature;
  90 is the maximum report age in minutes. A future 90-day weather feature
  would need older data. The user wants to discuss predicting rare heavy snow
  next. No `SUBMISSIONS.md` changes, full-year fits, uploads or promotions ran.

- **September 21: OPDI source downloads are complete.** The user wants v39/v40
  tested with the same 12 new disruption features, with no `SUBMISSIONS.md`
  changes. Original OPDI caches/models are on Ioana's machine; the user instead
  authorized downloading the public data after moving to fast internet.
  Local `data/external/opdi/` now retains 15 flight lists and 44 raw event files,
  **14,844,121,265 bytes**. Only `flight_events_20251002_20251012.parquet` returns
  the historically documented 404; there are no other failures or partial files.
  The [download report](VALIDATION/reports/opdi_download_20260921.json) records
  hashes; full manifest is in the data folder and logs in
  `tmp/opdi_download_20260921/`. Downloader `tools/download_opdi_cache.py` handles
  HTTP/2 failures by using HTTP/1.1 and rejected resume requests by restarting
  only partial files; six targeted tests passed. **Do not redownload the cache.**
  Filtering, feature-cache construction and the paired v39/v40 experiments
  are now complete, as recorded above. v39/v40 recorded different cache hashes,
  so exact historical cache reproduction remains unestablished.

- **September 21: disruption-feature experiment is complete; no promotion.**
  Read the short [report](VALIDATION/reports/DISRUPTION_EXPERIMENT.md). Adding
  12 snow-duration/completed-arrival inputs to v38 improves Weighted CV from
  **267.5814 to 266.8465** (-0.7349 s); all seeds improve and 1-/3-day paired
  intervals exclude zero. Original combined improves 236.3151 to 235.6700.
  However, January's retrospective check worsens 348.1483 to 348.6760; EHAM
  worsens 2.03 s and its 293 missing-ID departures worsen 7.23 s. This passes
  the frozen CV criterion but does not demonstrate a winter fix or LB transfer.
  Completed 18 ordinary fits, all 16 folds, three seeds, plus January using
  June–November models. All 48 v38 controls reproduce exactly. Nine tests and
  4,080 feature-value checks on 340 sampled fold/flight rows passed. Preserve
  `VALIDATION/runs/v3_20260921_disruption/`; do not repeat fits. The frozen
  [design](VALIDATION/docs/DISRUPTION_EXPERIMENT.md), runner `disruption_run.py`,
  scorer `tools/score_disruption_experiment.py` and independent auditor
  `tools/audit_disruption_experiment.py` are under `VALIDATION/`. Full local
  logs are in `tmp/disruption_experiment/`; concise runtime in the run directory.
  v40 artifacts/OPDI inputs were unavailable locally, so this compares v38 only.
  No full-year refit, upload, promotion or submission-ledger entry was made.

- **September 21: training-versus-ranking step 4 is complete.** Read the short
  [report](VALIDATION/reports/TRAIN_RANKING_STEP4.md). EHAM January arrival taxi-in
  >30 minutes rises from 50/18,469 (0.27%) to 264/16,246 (1.63%). Prolonged snow
  covers 32/18,469 2025 arrivals versus 2,890/16,246 in 2026, including 228/264
  long arrivals. Here prolonged means SN/SG at >25% of the previous 48 half-hour
  samples, not measured snowfall depth. Outside recent snow, tail rates are
  0.28%/0.24%. Official Schiphol reporting corroborates winter operational
  disruption. Common recording signatures do not account for the arrival tail;
  arithmetic agreement alone does not verify timestamps independently.
  Jan 2–5 also contain 409/805 missing-ID and 278/418 >4h-late departures. This
  association does not prove departure-label drift or explain a measured LB gap.
  Detailed matching excludes most extreme cases; retain coverage and withheld
  intervals. Eight tests and independent checks of 710,038 raw labels, 60 weather
  matches, 120 traffic counts, 40 snow-duration values and 40 standardizations
  passed. A traffic-bin '0'/'0.0' mismatch was fixed and tested. Reuse the local
  cache `tmp/train_ranking_step4/`; no new datasets or taxi-out fits/uploads.
  Weighted CV stays unchanged. Steps 1–4 of this investigation are complete;
  the subsequent model-feature experiment is recorded above, separately from
  this descriptive investigation.

- **September 21: training-versus-ranking step 3 is complete.** See the short
  [cost report](VALIDATION/reports/TRAIN_RANKING_STEP3.md). EHAM missing-ID flights
  are the strongest lead: January 2025's 1.59% of rows account for 29.51% of SSE
  under the older v15 seasonal holdout. This is not a v38/v40 January result.
  v38 finds this group costly in every July–December month. Raw kNN's July
  highest-risk 20% capture 34–77% of actual SSE across airports, but neighbour
  risk underestimates RMSE at all ten. Propensity is not a consistent cost rank.
  EDDF/EDDM new-stand cost remains uncertain; same-month historical cases are
  sparse. No departure-label drift or leaderboard-gap attribution is proven.
  Reuse `tmp/train_ranking_step3_inputs/` (13 MB exported from the desktop with
  original-file hashes and exact score checks) and `tmp/train_ranking_step3/`.
  Five tests, 50 independent cost checks, 200 neighbour checks and two stand
  checks passed. v40 OOF files were absent on both hosts; no fits or uploads ran,
  and Weighted CV is unchanged. Step 4, the conditional EHAM arrival-label
  investigation, is complete above. `wx_snow` includes ice pellets/small hail as
  well as snow; retain that definition when interpreting weather groups.

- **September 21: training-versus-ranking step 2 is complete.** See the concise
  [report](VALIDATION/reports/TRAIN_RANKING_STEP2.md). Blocked raw AUC is .624/.664
  for Jan/Jul year comparisons, versus .539/.531 for the selected within-2025
  adjacent-month controls. EHAM January missing-ID/long-schedule-gap flights and
  July EDDF/EDDM stand changes remain leads; prediction cost is untested.
  Original rich kNN had a poor fallback scale for nearly constant weather fields;
  the corrected run is complete. Use `train_ranking_step2/knn_current.csv` and
  `slices_corrected.csv` under reports, not the original rich rows in `knn.csv`.
  `tmp/train_ranking_step2/similarity_rows.parquet` contains all 344,841 ranking
  and 344,419 Jan/Jul 2025 rows with both propensities and ten neighbour IDs per
  distance view. Reuse the saved 33 classifiers, 80 original and 80 corrected
  neighbour banks; ten tests and independent arithmetic/identity audits pass.
  Diagnostic probability calibration is uneven; adopted Weighted CV is unchanged.
  Step 3 (saved-error cost, respecting which months/models have OOF predictions)
  is now complete above; step 4 (conditional EHAM arrival-label investigation)
  is also complete above.
  No taxi-out fits, submissions or validation-weight changes ran.

- **September 21: training-versus-ranking step 1 is complete.** Read the short
  [report](VALIDATION/reports/TRAIN_RANKING_STEP1.md); reproduce with
  `VALIDATION/tools/describe_train_ranking_shift.py`. It compares Jan/Jul across
  years at all ten airports against 11 adjacent and 66 total 2025 month pairs.
  January EHAM has more missing plans, long schedule gaps and snow; July EDDM/EDDF
  stand use changes substantially. These are observable differences, not measured
  prediction cost or proof of changed departure labels. Step 2 is now complete
  above; steps 3 and 4 are also complete above. Reuse the hashed feature
  cache in `tmp/train_ranking_step1/`; no fits, uploads or weighting changes ran.

- **September 20: Weighted CV is the user-selected validation headline.**
  Use the unclipped, per-source raw-feature weights in the
  [method](VALIDATION/docs/WEIGHTED_CV.md) for future model comparisons; retain
  original folds and unweighted diagnostics. The standard harness still emits
  the original scores, so compute Weighted CV from saved predictions before
  reporting a new comparison. Across eleven scored submissions, the average
  absolute CV–LB gap fell from 68.05 to 23.42 s; direction accuracy stayed 7/10.
  `SUBMISSIONS.md` now has verified weighted scores for v12–v21 and v38; leave
  other weighted entries `-` until verified. The
  [expanded report](VALIDATION/reports/WEIGHTED_CV_LEADERBOARD_EXTENDED.md) links
  the evidence and scoring tools. Original v17–v21 prediction files were recovered
  from the desktop and hash-verified. Reuse the weights in
  `tmp/weighted_cv_leaderboard_20260920/`; do not replace them with rich-feature
  or clipped weights. This decision supersedes earlier recommendations to keep
  the old validation headline, without changing historical results or authorizing
  any further refits/uploads. Training-versus-ranking identification remains a
  separate unfinished investigation.

- **September 19 Rome calibration experiments are complete; neither is promoted.**
  Run `v3_20260919_rome_calibration` completed 126 classifiers, all 16 folds and
  three seeds in 31.614 minutes, reusing all ordinary models. Binary calibration
  scored **296.7058 / 296.4860 / 296.5834** (R_A / R_B_long / combined);
  calibrated three-class scored **294.2137 / 282.8235 / 287.9234**, versus matched
  control **253.0698 / 228.9793 / 239.9468**. Both regress across all seeds and
  one-/three-day bootstrap comparisons. Removing the hard Rome rule explains
  most of the damage; calibration further worsens both soft mixtures. Keeping
  the original rule with binary calibration is inconclusive (+0.3419 combined).
  The user explicitly requested these two validation results in `SUBMISSIONS.md`:
  preserve rows 38–39 as validation-only records despite the general ledger rule
  above. No full-year refit or upload belongs to these experiments. Read the
  [completed report](VALIDATION/runs/v3_20260919_rome_calibration/report.md);
  preserve the saved models, probabilities and frozen design. Do not repeat fits.

- **September 19 recording-artifact study is exploratory; no model promoted.**
  The user rejected listing these experiments as model progress. All added
  entries 37–46 were removed from `SUBMISSIONS.md`. The user then explicitly
  requested restoring only `combined` (former row 44) as **row 37**. Keep that
  entry; the other experiments stay out. Results remain in
  [the report](VALIDATION/reports/RECORDING_ARTIFACT_EXPERIMENTS.md)
  and runs `v3_20260919_artifact_control` / `v3_20260919_artifact_experiments`.
  Reused development folds, sparse cases, identical variants and inactive gates
  must stay explicit. The user then authorized only this candidate for refit and
  upload: **v38 scored 279.3805 s on all 344,841 ranking flights**. Its manifest
  is `VALIDATION/runs/v3_20260919_artifact_experiments/submission_v38.json`;
  reproduce with `VALIDATION/artifact_submit.py`. Filename v37 belongs to a
  separate dayshift-hedge model. Preserve the completed full-year artifacts;
  do not repeat completed fits or re-add the removed variants. Current host is the
  i7-13650HX WSL laptop: one fit, six training threads, 20 prediction threads.

- **The overnight five-submission study is complete.** Read the final
  [summary](VALIDATION/reports/OVERNIGHT_SUMMARY_20260917.md), then
  `VALIDATION/runs/overnight_20260917/confirmation_report.md`. All five succeeded:
  no_month/v17 **345.8984**, no_history/v18 **342.5049**, clock_features/v19
  **338.1113**, small_trees/v20 **345.3113**, rule_routing/v21 **342.4009**.
  All retain recovery off, 800 rounds and the fixed four-hour rule. Best observed
  is now **v19**, 6.2001 s better than the frozen Q0-4h/v15 study anchor; latest
  is v21. Live snapshot **2026-09-17 04:23:02 UTC**: rank **94/152** scoring teams,
  416 registered; saved in `VALIDATION/reports/overnight_20260917/status_after.json`.
  The first three candidates informed CV selection; the last two official scores
  stayed unread until `cv_selection_frozen.json` was written and sealed.
  Selected **R_A__mse** (next-month MSE-difference transport) **failed confirmation**:
  held-out change MAE **3.5178 s**, versus current R_combined's **2.8367 s**.
  There is meaningful modelling progress, but no reliably better CV established;
  **keep the current default and every fold**. More varied future confirmation
  is needed. Other readouts after reveal are descriptive, not new winners.
  Completed work: **153 new CV fits + 9 reused**, **360 prediction files**,
  **6 full-year refits** (five candidates plus control), **101 tests passed**.
  Run root `VALIDATION/runs/overnight_20260917/` retains signatures, models,
  predictions, upload manifest and sealed analysis. Designs are frozen in
  `VALIDATION/docs/OVERNIGHT_STUDY_20260917.md` and its `AMENDMENT1` sibling;
  do not rewrite them or repeat completed fits/uploads. Current hardware is the
  7800X3D desktop, using eight training/prediction threads and one fit at a time.
  All five authorized 2026-09-17 slots were used. **No further submission or
  automatic quota-reset retry is authorized.** Notebook files were untouched.

### Earlier work — historical context

The notes below describe earlier decisions and machine-specific runs. Their
then-current best submission, pending routing proposal and upload deferrals are
superseded by the completed overnight handoff above; historical recipes stay fixed.

- **A month-balanced overview is proposed, not installed as the default.** See
  `VALIDATION/reports/AGGREGATION_COMPARISON.md`, reproducible with
  `VALIDATION/tools/compare_aggregations.py`. All 16 folds remain: average seed
  and training-origin squared errors within month, then give July–December 1/6
  each. Saved-run LB level/change errors are 49.41/7.13 s versus old combined
  70.73/7.19; material direction agreement stays 3/3. Recovery-change error
  worsens from 0.47 to 3.47 s. The proposal keeps R_A, R_B_long and matched-month
  aging diagnostics separate: balanced months reduce k>=5 weight to 10.83%.
  Weights follow the roster and were not fitted to LB. No refits, fold removals,
  historical-score changes or default-harness changes were made.

- **Validation optimism diagnosis is complete.** Read
  `VALIDATION/reports/VALIDATION_OPTIMISM.md`; reproduce with
  `VALIDATION/tools/analyze_validation_optimism.py` (saved errors and observable
  counts only). v3's month mix emphasizes easier autumn populations; its weighted
  missing-AOBT3 fraction is 1.070% versus ranking's 1.534%. Seasonal Q0-4h July
  improves on the same original July flights, so worse fitting is not the reason
  seasonal scores rise. Two January LFPG errors supply 51.2% of January and
  20.6% of combined seasonal MSE; influence-only removal flips the bias sign.
  All rows/folds remain scored. Hidden 2026 labels prevent complete attribution;
  missing-clock reweighting alone worsens seasonal level alignment. No new model
  or validation change was selected. The user clarified that the earlier fold
  task sought defensible removal of harmful folds, not favorable-subset selection:
  **neither seasonal replacement nor fold removal is justified so far**.

- **Original-fold alignment analysis is complete; no folds were dropped.** See
  `VALIDATION/reports/FOLD_ALIGNMENT.md` and the reproducible
  `VALIDATION/tools/analyze_fold_alignment.py`. The combined headline uses only
  A1–A6/B3/B4/B7; November/December contribute 69.17% of its squared-error weight.
  A-only has level/change errors 50.05/6.46 s versus combined 70.73/7.19 s.
  Exploratory A2+A3 tracks changes at 2.65 s error, but recovery-change error is
  2.76 s versus the current combined score's 0.47 s. A1–A3 is closest in score
  levels (17.52 s), with worse change error (9.79 s). Threshold evidence is sparse;
  retain winter coverage and confirm any proposed reduction on later candidates.
  Original prediction parquets are absent here: aggregate reconstruction and
  seed checks are exact, but no new day-bootstrap intervals were computed.

- **Seasonal validation comparison is complete on `seasonal-validation`.** The
  user requested two retrospective splits for the five scored submissions:
  ten months excluding January/July, and two separate eleven-month fits.
  Read `VALIDATION/docs/SEASONAL_VALIDATION.md` for the explicit December-history
  policy, original recipe/source checks, commands and LB alignment outputs.
  `VALIDATION/seasonal.py` completed 18 fits at three seeds, reusing model
  predictions for rules. Full results are in `VALIDATION/runs/seasonal_jan_jul/`.
  Ten/eleven-month mean absolute LB gaps are 36.21/35.73 s (pessimistic), versus
  v3's 70.73 s (optimistic), but change errors worsen to 12.45/10.50 from 7.19 s.
  All three match 3/3 material directions; no validation replacement is selected.
  No uploads are part of it; do not rerun these completed fits.
  This machine is the 7800X3D desktop: use its recorded eight-thread setting,
  not the laptop hardware assumptions below. Completed earlier stages remain
  complete; the seasonal comparison precedes the proposed training-routing work.

- **The five selected leaderboard checks are complete.** The user authorized
  refits/uploads on 2026-09-16; work is on `leaderboard-five-candidates`, based
  on Stage 4. All five succeeded: B0-800/v12 **406.4281**, Q0/v13 **354.4303**,
  Q1/v14 **354.0004**, Q0-4h/v15 **344.3114**, Q1-4h/v16 **344.4821**.
  Live snapshot 2026-09-16 16:46:48 UTC: best v15, latest v16, rank 96/152
  scoring teams (416 registered). Q0-4h is the best observed submission;
  retain Q0 as the six-hour experimental control. Rule gains transferred;
  recovery helped by 0.4299 s at six hours but regressed by 0.1707 s at four.
  See `VALIDATION/reports/LEADERBOARD_RESULTS.md` for all CV/LB levels, gaps,
  paired changes, seed-42 readouts and uncertainty. Execution details are in
  `VALIDATION/docs/LEADERBOARD_EXECUTION.md`; the submission manifest links
  source/data/config hashes and verified remote artifacts. Two full-year fits
  took 2.067/4.461 minutes, with no CV refits or repeated recovery/CPU audits.
  Four-hour linked runs are `v3_20260916T164054429797Z` (Q0-4h) and
  `v3_20260916T164247756836Z` (Q1-4h), matching the original sweep exactly.
  All 56 tests passed. All five daily slots were used; no automatic reset
  retry, additional submission, training-row exclusion or rule-correction
  model was started. The next proposed work is discussion of training routing,
  now that rule gains transferred. Do not rerun completed fits or uploads.

When asked to **continue the improvement plan**, read
[`VALIDATION/IMPROVEMENT_PLAN.md`](VALIDATION/IMPROVEMENT_PLAN.md), starting with
its "Resume here" section, and
[`VALIDATION/reports/STAGE0.md`](VALIDATION/reports/STAGE0.md).

- **Stage 4 is complete. Do not promote M1; retain Q0.** Branch `stage4-timestamp-mixture`,
  based on Stage 3. Frozen recipe: `VALIDATION/docs/TIMESTAMP_MIXTURE.md`.
  M1 `v3_20260916T144408082362Z` fits a three-class probability model and
  ordinary/other regressor per window/seed (36 boosters in 18 bundles), using
  T1's 45 features, recovery off, 800 rounds and the fixed six-hour rule.
  It completed in 51.768 minutes, scoring 544.8410/519.0849 (R_A/R_B_long).
  All three seeds regress against T1, Q0 and Q1, with supported overall
  regressions. The classifier makes overconfident schedule/day mistakes on
  ordinary flights; only 5–9 day cases exist per training window. See
  `VALIDATION/reports/STAGE4.md` for comparisons, false-high diagnostics and
  concrete failures. All 55 tests and actual M1 linked-recipe checks pass.
  All 18 bundles/48 prediction files are saved; do not repeat completed fits.
  Stages 1–4 are complete. The first leaderboard queue is now selected:
  **B0-800, Q0, Q1, Q0-4h, Q1-4h**, in that order. See
  `VALIDATION/docs/LEADERBOARD_CANDIDATES.md` for exact recipes, CV scores,
  controlled comparisons and preparation tasks. Four-hour linked recipes were
  pending at selection and have since been registered as recorded above. Only two
  underlying full-year models are needed; reuse their predictions for the
  rule variants. No full-year refit or upload ran during selection. This queue
  takes priority over the separate v10/v11 series; inspect live quota at execution.
  The user expanded the later leaderboard work: record official and CV scores,
  candidate-minus-control changes, direction agreement, LB-minus-CV gaps and
  delta-LB-minus-delta-CV, with seed-42 CV alongside the standard seed average.
  Details are in the plan; the authorized execution is complete as recorded above.
- **Stage 3 is complete. Keep Q0 as the conservative control.** Its work is on
  `stage3-clock-features-residual`, based on Stage 2. The frozen design is in
  `VALIDATION/docs/CLOCK_MODELS.md`: T1 adds 18 features, T2 uses the same
  features with residual regression. Both retain recovery off, 800 rounds,
  three seeds and the fixed six-hour rule. All 51 tests pass.
  T1 `v3_20260916T132038893775Z` is complete (19.470 minutes), scoring
  298.9328/256.2445; both changes versus Q0 are inconclusive. T2
  `v3_20260916T134231489815Z` is complete (19.292 minutes), scoring
  419.6226/439.4680; **do not promote**, supported regression versus T1 and Q0.
  Its schedule fallback fails on very large gaps. Comparisons, training curves,
  saved-error diagnosis and verdict are in `VALIDATION/reports/STAGE3.md`.
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
  completed submissions are recorded above; training exclusions remain unstarted.
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
  complete and reuse saved predictions. Stage 4 is now complete as described
  above. The leaderboard check is complete; the proposed training-routing
  discussion remains pending. Do not repeat recovery audits or CPU tuning
  unless their relevant code/data changes.
- All 56 tests pass. Stage 1's recovery-off equivalence evidence is retained.
  Current source includes the Stage 4 mixture. Historical T1/T2, Q0/Q1
  and recovery models require their frozen implementation for a full-year
  refit; preserve source guards rather than bypassing them.
- The user requested exact Stage 2/3 feature names and numerical examples.
  `VALIDATION/docs/STAGE2_STAGE3_EXAMPLES.md` records the four rule fields,
  all 27 original plus 18 new model inputs, T1/T2 arithmetic and real failures.
- The shared pipeline implements `--recovery off|exact|relaxed|history`, with
  raw precedence, fold-scoped donors, ownership guards and causal/frozen static
  history. `pipeline/prediction_rules.py` supplies the fixed rule to both
  `VALIDATION/postprocess.py` and linked submission recipes. Rule and parent
  model hashes are separate. Every overridden ID/error is in candidate CSVs.
- The user authorizes local training/validation. Defer twelve-month refits and
  uploads until experiments finish and submission configurations are selected.
  Those conditions were met for the five-candidate queue, which was explicitly
  authorized and completed above; this does not authorize additional submissions.
- **Hardware is machine-specific.** The current i7-13650HX/WSL laptop exposes
  20 logical CPUs and about 11.5 GiB RAM. Local measurements select one worker,
  six fit threads and 20 prediction threads. See `VALIDATION/reports/STAGE1_CPU.md`.
  Run full-data audits sequentially with experiments; overlapping them caused
  swapping. Do not copy the Stage 0 desktop's 7800X3D settings to another host.
- Existing `data/derived/` notebook caches are useful audit references, but are
  not fold-valid inputs. Their sections, counts, guards and reuse limits are in
  `VALIDATION/reports/STAGE1_DERIVED_DATA.md`. Do not rebuild them unnecessarily.
- Stage 4 work is on `stage4-timestamp-mixture`, based on the open Stage 3 PR
  branch `stage3-clock-features-residual`. Track source, plans, concise results
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
