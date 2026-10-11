# PRC Data Challenge 2026 — Team unique-rose

Predicting taxi-out times at major European airports for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/).

Current experiments, retained models and submission approvals are tracked in
[AGENTS.md](AGENTS.md), [SUBMISSIONS.md](SUBMISSIONS.md) and the
[October 10 research report](VALIDATION/reports/RESEARCH_NETWORK_20261010.md).

## Competition rules & key information

### The task

Predict the **taxi-out time** (`TAXITIME_SEC_mvt`, in seconds) of departing flights at 10 major European airports:

| ICAO | Airport | ICAO | Airport |
|------|---------|------|---------|
| LTFM | Istanbul | LEMD | Madrid–Barajas |
| EHAM | Amsterdam Schiphol | LEBL | Barcelona–El Prat |
| LFPG | Paris Charles de Gaulle | EDDM | Munich |
| EGLL | London Heathrow | LIRF | Rome Fiumicino |
| EDDF | Frankfurt | LSZH | Zurich |

- **Training data:** all airport movements for the full year **2025** (~4.17 M rows).
- **Evaluation data:** originally January and July 2026; the final files add
  **February and June 2026** ([official data description](https://prc-data-challenge-2026.netlify.app/data.html)).

### Metric & ranking

- Submissions are scored by **Root Mean Square Error (RMSE)** on predicted
  taxi-out times for the accepted evaluation population. Historical scores
  through v61 use the 344,841 original January/July departures.
- A team's rank is its **best RMSE across its scored submissions**. The user
  authorized autonomous uploads on 11 October; the **225.00** target was achieved
  with **v71 at 224.3602** on the accepted original January/July population.
- Leaderboard REST API: `https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard` (paginate via `nextCursor`; Swagger docs at `https://datacomp.opensky-network.org/api/swagger-ui/`).

### Timeline & prizes

- Runs from **1 September 2026** until **14 October 2026, 10:00 UTC**.
  The [official extension](https://prc-data-challenge-2026.netlify.app/) explicitly
  gives `2026-10-14T10:00Z` alongside its displayed `12:00 CET` deadline.
- Combined prize pool of **€5,000** for the top 3 teams.
- Organizers reserve the right to change the rules or stop the challenge.

### Datasets

Hosted on OpenSky Network's MinIO object store (S3-compatible), endpoint `https://s3.opensky-network.org`, bucket `prc-2026-datasets`:

| File | Rows | Contents |
|------|------|----------|
| `training_2025-MM-01_*.parquet` (×12) | 4,167,797 total | One month each of 2025 movements (departures **and** arrivals) with full features and targets |
| `ranking.parquet` | 689,534 | Jan + Jul 2026 movements, same columns; `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are **blanked for departures** (`PHASE_mvt = "DEP"`, 344,841 rows) — these are what we predict |
| `submitting.parquet` | 344,841 | Submission template: `MVT_ID_mvt` (double) + `TAXITIME_SEC_mvt` (int32, all null) |
| `final_ranking.parquet` | 1,341,197 | Final January, February, June and July 2026 movements; departure targets are blanked |
| `final_submitting.parquet` | 670,790 | Final four-month departure template, with the same two columns |

The final files were released on 8 October. Shared old/final movements have
jittered timestamps, so retain separate flight-keyed features and predictions.
The first four-month upload, v63, was rejected because the scorer still had
the old truth population. Check the accepted population before choosing a
template; see [AGENTS.md](AGENTS.md) for current submission state.

Each row carries movement fields (`*_mvt`: airport, phase, times, aircraft type, runway, stand, taxi time) joined with flight-plan fields (`*_flt`: callsign, operator, market segment, wake category, OBT/ARVT timestamps). Note from the organizers: *"Real world data is messy. You will find some inconsistencies between movement and flight information."*

### Submission format

1. Start from the template matching the accepted ranking population and fill `TAXITIME_SEC_mvt` with predictions (do **not** modify `MVT_ID_mvt`).
2. Name the file `<team-name>_v<incremental integer>.parquet` — for us: `unique-rose_v1.parquet`, `unique-rose_v2.parquet`, …
3. Upload it to the team's own MinIO bucket (ours: `prc-2026-unique-rose`).

The ranking script **rejects** submissions with mismatched `MVT_ID_mvt` values, missing rows, or extra rows — always start from the template.

### Eligibility & open-science requirements (prize conditions)

To be eligible for prizes, a team must ensure:

- The team is **not** related to sanctioned non-EUROCONTROL states (see the [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html)).
- All external datasets used are **openly accessible and documented**, under an open-source license.
- All produced source code is published **openly on GitHub under GNU GPLv3**.
- Documentation is sufficient to **understand and reproduce** the results.
- The solution is **original**: reused code requires rights from the original authors *plus* significant modifications to the algorithm/model (input/output-only changes don't count).

Publication of results in the [Journal of Open Aviation Science](https://journals.open.tudelft.nl/joas) is encouraged.

## Team

- **Team name:** `unique-rose`
- **Submission bucket:** `prc-2026-unique-rose` (write access via team MinIO keys)
- **Data access:** via the `mc` CLI (AIStor/MinIO client), alias `opensky` → `https://s3.opensky-network.org`. Credentials live in `~/.mc/config.json` — **never commit keys to this repo** (it will be public).

```sh
# refresh datasets (only pulls new/changed files)
mc mirror opensky/prc-2026-datasets/ data/

# submit predictions
mc cp unique-rose_vN.parquet opensky/prc-2026-unique-rose/
```

## Setup

Works on Linux/WSL and macOS. You need Python 3.11+, git, and the MinIO client `mc`.

```sh
git clone https://github.com/Goxmen/PRC.git
cd PRC

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MinIO client — macOS:  brew install minio-mc
#                Linux:  https://min.io/docs/minio/linux/reference/minio-mc.html
mc alias set opensky https://s3.opensky-network.org ACCESS_KEY SECRET_KEY

# pull the datasets (~300 MB, not tracked in git)
mc mirror opensky/prc-2026-datasets/ data/
```

The team MinIO keys are shared privately (ask a teammate) — never commit them.

### Where credentials live (never commit them)

- **GitHub** — authenticated via the `gh` CLI (`gh auth login`); the token sits in
  `~/.config/gh/hosts.yml` and `gh` acts as git's credential helper, so `git push`
  needs no further setup. Check with `gh auth status`. A GitHub PAT for this repo
  is also stored as `GHP` in the gitignored root `.env` for direct API access when
  `gh` is unavailable. Never commit `.env` or token values.
- **OpenSky MinIO** — keys in `~/.mc/config.json` (the `opensky` alias set above).

`AGENTS.md` documents this environment for AI coding agents.

### Repo layout

- `data/` — parquet datasets, gitignored; refresh with `mc mirror` (see above). `data/FEATURES.md` documents the columns.
- `eda/` — exploratory analysis scripts, notebooks, and generated figures.
- `pipeline/` — training & submission pipeline. `train.py` fits the model and reports holdout RMSE (Nov+Dec 2025); `predict.py` fills the submission template into `submissions/` (gitignored, like `models/`).
- `VALIDATION/` — the validation protocol ([v3](VALIDATION/docs/PROTOCOL.md)) and the harness that runs it; see [VALIDATION/README.md](VALIDATION/README.md) and [VALIDATION/archive/FINDINGS_20260907.md](VALIDATION/archive/FINDINGS_20260907.md).
- `dashboard/` — source of the team's live leaderboard page (a Claude artifact). `pipeline/dashboard_data.py` rebuilds the data payloads from the leaderboard API + team bucket into `tmp/dashboard/`; the page reads them from the artifact's database, so refreshing the data needs no republish. `pipeline/leaderboard_status.py` provides the same live score history and public standings directly in text or JSON (`--json`), without using the webpage.

## Approach

The best scored recipe combines piecewise-linear-embedding neural networks,
main and per-airport LightGBMs, and CatBoost. It uses supplied flight plans,
traffic/history features and cached external observations. A separate Rome
classifier handles schedule-copy recording behavior, followed by retained
corrections. The target is the recorded taxi duration, which sometimes differs
from physical surface movement alone.

New candidates use the two-fold January/July validation in
[`VALIDATION/tuning_seasonal.py`](VALIDATION/tuning_seasonal.py). Its monthly
RMSE average is distinct from the official test score. These development months
have been reused extensively; bootstrap intervals do not make them independent
confirmation. Full recipes, failed ideas and reproducible run locations are in
the [research report](VALIDATION/reports/RESEARCH_NETWORK_20261010.md).

## Results

See [SUBMISSIONS.md](SUBMISSIONS.md) for actual uploads and selected candidates.
For live scores, run `.venv/bin/python pipeline/leaderboard_status.py --json`;
distinguish the latest upload from the best score.

## License

[GNU GPLv3](LICENSE), as required by the competition's open-source rules.
