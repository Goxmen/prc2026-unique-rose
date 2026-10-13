# PRC Data Challenge 2026 — Team unique-rose

Predicting taxi-out times at major European airports for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/).

This repository contains the team's feature engineering, model training,
validation, and final prediction code. Models combine LightGBM, CatBoost,
piecewise-linear-embedding networks and TabM, with a separate treatment of
Rome's recorded taxi-time modes and public operational aggregates.

## Reproducing the submission

Start with [the reproduction guide](docs/REPRODUCING.md). It distinguishes
exact replay of the accepted final prediction ensemble from retraining models
and from the current four-month model-selection experiment. Reproduction
commands never submit predictions.

See [selected final status](release/SELECTED_FINAL.md) for the authoritative
accepted artifact. The separate [manual v80 refit command](tools/v80_refit/README.md)
does upload a replacement when the owner runs it.

| Document | Contents |
| --- | --- |
| [Reproduction](docs/REPRODUCING.md) | Environment, required artifacts, executable commands and output verification |
| [Method](docs/METHOD.md) | Features, models, calibration, validation and limitations |
| [Data sources](docs/DATA_SOURCES.md) | Acquisition, snapshots, provider terms and attribution |
| [Third-party work](docs/THIRD_PARTY.md) | Dependencies, upstream implementations and our contributions |
| [Release checklist](docs/RELEASE_CHECKLIST.md) | Competition obligations and remaining publication steps |
| [Four-month validation](VALIDATION/docs/FINAL_MONTH_SELECTION.md) | Ten candidate definitions and embargoed folds |
| [Submission record](SUBMISSIONS.md) | Actual uploads and selected candidates |

The historical v83 final artifact was uploaded on
13 October 2026: 670,790 predictions, SHA-256
`f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f`.
Acceptance is not a published final RMSE. The current four-month experiment
does not replace that artifact automatically. The owner has separately selected
a full v80 recipe refit; its manual runner updates the selected-final record only
after the replacement is accepted.

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
- A team's rank is its **best RMSE across its scored submissions**. As recorded
  on 13 October, best is **v80 at 218.8525** on 344,841 original departures; latest scored
  is **v83 at 220.9278**. The separate final submission was accepted on all
  670,790 rows on 13 October; its score is unpublished.
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
The first four-month upload, v63, was rejected by the versioned leaderboard
route, which scores the old population. The separately named final route is
documented below; our final submission was accepted on 13 October with all
670,790 rows, without a published numeric score.
See [SUBMISSIONS.md](SUBMISSIONS.md) for the recorded submission history.

Each row carries movement fields (`*_mvt`: airport, phase, times, aircraft type, runway, stand, taxi time) joined with flight-plan fields (`*_flt`: callsign, operator, market segment, wake category, OBT/ARVT timestamps). Note from the organizers: *"Real world data is messy. You will find some inconsistencies between movement and flight information."*

### Submission format

1. Start from the template matching the accepted ranking population and fill `TAXITIME_SEC_mvt` with predictions (do **not** modify `MVT_ID_mvt`).
2. Name the file `<team-name>_v<incremental integer>.parquet` — for us: `unique-rose_v1.parquet`, `unique-rose_v2.parquet`, …
3. Upload it to the team's own MinIO bucket (ours: `prc-2026-unique-rose`).

The ranking script **rejects** submissions with mismatched `MVT_ID_mvt` values, missing rows, or extra rows — always start from the template.

The [official final-submission instructions](https://prc-data-challenge-2026.netlify.app/ranking.html#final-submission),
checked 12 October, specify **`unique-rose_final.parquet`** for the unique final
submission. Prepare this separately using the four-month final template; a
versioned January/July leaderboard upload does not satisfy this requirement.

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

The recorded training environment is Linux/WSL, Python 3.12.3, and NVIDIA CUDA.
The current neural validation requires an NVIDIA GPU. CPU-only artifact replay
is documented separately. See [the environment snapshot](release/environment.json)
and [reproduction guide](docs/REPRODUCING.md) for hardware and runtime details.
You also need git and the MinIO client `mc` for competition data access.

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

Use credentials issued to your own OpenSky account; team submission credentials
are not needed to reproduce predictions. Never commit credentials.

Credentials belong in local client configuration or ignored environment files.
The repository contains no example requiring real keys. Local operational
instructions for contributors are in `AGENTS.md`.

### Repo layout

- `data/` — parquet datasets, gitignored; refresh with `mc mirror` (see above). `data/FEATURES.md` documents the columns.
- `eda/` — exploratory analysis scripts, notebooks, and generated figures.
- `pipeline/` — feature utilities and the historical baseline pipeline. Its `train.py`/`predict.py` commands do not reproduce the final ensemble; use the reproduction guide above.
- `VALIDATION/` — the validation protocol ([v3](VALIDATION/docs/PROTOCOL.md)) and the harness that runs it; see [VALIDATION/README.md](VALIDATION/README.md) and [VALIDATION/archive/FINDINGS_20260907.md](VALIDATION/archive/FINDINGS_20260907.md).
- `tools/release/` — offline final-ensemble replay and repository publication checks.
- `release/` — portable environment, final-model provenance and release records; large bundles stay outside git.
- `dashboard/` — source of the team's live leaderboard page (a Claude artifact). `pipeline/dashboard_data.py` rebuilds the data payloads from the leaderboard API + team bucket into `tmp/dashboard/`; the page reads them from the artifact's database, so refreshing the data needs no republish. `pipeline/leaderboard_status.py` provides the same live score history and public standings directly in text or JSON (`--json`), without using the webpage.

## Approach

The best scored recipe combines piecewise-linear-embedding neural networks,
main and per-airport LightGBMs, and CatBoost. It uses supplied flight plans,
traffic/history features and cached external observations. A separate Rome
classifier handles schedule-copy recording behavior, followed by retained
corrections. The target is the recorded taxi duration, which sometimes differs
from physical surface movement alone.

Historical candidates used January/July validation in
[`VALIDATION/tuning_seasonal.py`](VALIDATION/tuning_seasonal.py). Final model
selection now uses January, February, June and July 2025, with exactly one
adjacent month excluded on each side of each held month; the July fold now
includes April and May in training. See the [frozen four-fold protocol](VALIDATION/docs/FINAL_MONTH_SELECTION.md).
Its mean monthly RMSE is distinct from the official test score. These development
data have been reused extensively; bootstrap intervals do not make them independent
confirmation. The [method guide](docs/METHOD.md) explains the final ensemble and
the distinction between historical versions and the three shortlisted direct candidate recipes (reduced from ten).

## Results

See [SUBMISSIONS.md](SUBMISSIONS.md) for actual uploads and selected candidates.
For live scores, run `.venv/bin/python pipeline/leaderboard_status.py --json`;
distinguish the latest upload from the best score.

## License

[GNU GPLv3](LICENSE), as required by the competition's open-source rules.
