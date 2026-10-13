# Data sources and attribution

This declaration covers the accepted v83-derived final submission and the newer
four-month model-selection experiment. These are different recipes; a validation
candidate is not the accepted final model. Snapshot references and recorded
SHA-256 values are in [the data manifest](../release/data_sources.json).

On 13 October 2026, the team owner confirmed that the previously raised question
about use of EUROCONTROL performance sources was resolved for this competition
("it's fine, we are good"). We record this as **team confirmation**, not as an
organizer statement or a change to the publishers' licenses. The source terms
below remain applicable. Data licenses are separate from this repository's GPLv3
source-code license; datasets and credentials are not committed to Git.

## Inputs used

| Source and attribution | Role and local location | Availability and terms |
|---|---|---|
| [PRC Data Challenge 2026 / EUROCONTROL / OpenSky Network](https://prc-data-challenge-2026.netlify.app/data.html) | Twelve 2025 movement files, original Jan/Jul 2026 ranking/template, final Jan/Feb/Jun/Jul ranking/template, under `data/`. Supplies targets for training only, flight plans, movements, known arrival information and identifiers. Used by both recipes. | Competition MinIO bucket `prc-2026-datasets` at `https://s3.opensky-network.org`. Obtain challenge access credentials separately; this repository does not grant redistribution rights to the supplied data. |
| [Iowa Environmental Mesonet, Iowa State University](https://mesonet.agron.iastate.edu/request/download.phtml) | ASOS/METAR observations at the ten airports; `data/external/metar/metar.parquet`. Backward weather joins, snow and recent weather history. Both recipes. | [IEM states its materials are public domain](https://mesonet.agron.iastate.edu/disclaimer.php). Attribution: Iowa Environmental Mesonet of Iowa State University; underlying reports are supplied through its meteorological feeds. |
| [EUROCONTROL Open Performance Data Initiative, with OpenSky Network](https://www.opdi.aero/data.html) | OPDI v0.0.2 flight lists and ground events; `data/external/opdi/`, with Feb/Jun extension in `opdi_final/`. Callsign/time/airport matching provides ground-event clocks and traffic summaries. Both recipes. | Public downloads from the official OPDI portal. The reviewed OPDI pages do not name a standalone dataset license; retain OPDI/OpenSky attribution and the linked [EUROCONTROL terms](https://ansperformance.eu/about/disclaimer/), rather than inventing an SPDX data license. |
| [ADSB.lol contributors and data partners](https://github.com/adsblol/globe_history_2025), [2026 archive](https://github.com/adsblol/globe_history_2026) | Daily heatmap positions and callsigns filtered around the ten airports; `data/external/adsblol_heatmap/`. Provides aircraft identity, ground clocks, sparse tracks and callsign/squawk histories. Both recipes. | The archive database declares [ODbL 1.0](https://github.com/adsblol/globe_history_2025/blob/main/LICENSE-ODbL.txt). Preserve attribution and the applicable database terms when redistributing a derived database. The existence of a separate CC0 file is not a reason to label the whole database CC0. |
| [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) | Airport stands, runway/taxiway/apron geometry, historically fetched through Overpass. Derived coordinates in `data/external/stand_maps/`; additional geometry caches under `data/external/osm_taxi/`. Both recipes use retained stand/runway tables. | © OpenStreetMap contributors, ODbL 1.0. Geometry is transformed by stand-name normalization, coordinate joins and distance calculations; derived tables retain this provenance. |
| [X-Plane Scenery Gateway contributors / Laminar Research](https://gateway.x-plane.com/) | Recommended airport `apt.dat` stand and runway coordinates, merged with OSM into `stand_coords.parquet` and `runway_ends.parquet`. Both recipes. | [Gateway contributor terms specify GNU GPL version 2](https://gateway.x-plane.com/docs/TermsAndConditionsForUploadOfAirportScenery.pdf). Original code comments say “v2 or later”; the reviewed public terms establish v2 only, so this declaration makes no broader claim. Original scenery IDs/authors belong in the acquisition manifest when available. |
| [EUROCONTROL performance data](https://ansperformance.eu/data/) | `all_pre_departure_delays_{2025,2026}.csv`, downloaded 11 October. Airport-day departure counts and total positive pre-departure delay, used as features and noisy aggregate evidence. Both recipes use training-year data; accepted final uses 2026 data at inference. | Public CSV downloads; [EUROCONTROL copyright and reuse terms](https://ansperformance.eu/about/disclaimer/), with team confirmation above. Credit EUROCONTROL. Original CSV files are retained unmodified and are not redistributed by this source repository. |
| [EUROCONTROL taxi-out performance dashboard](https://ansperformance.eu/efficiency/taxiout/) | Daily local-date/runway counts and taxi/reference/additional-time sums, public Power BI model 3679962; cached 11 October as `data/external/eurocontrol_daily_taxi_20261011/daily_runway.parquet`. Both recipes use features; the accepted final also uses aggregate corrections. | Same EUROCONTROL terms and team confirmation. Raw anonymous dashboard responses, requests, schema, refresh time and extraction hashes are retained; no private account data is used. |
| [EUROCONTROL performance data](https://ansperformance.eu/data/) | `taxi_out_additional_time_{2025,2026}.csv`: airport-month taxi/reference/additional time, counts and quality information. Used in the accepted final's correction chain; not an input to the ten new direct validation recipes. | Same EUROCONTROL terms and team confirmation. The [published indicator methodology](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf) explains the population and reference calculation. |
| [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api), ERA5 / Copernicus Climate Change Service, ECMWF | Temperature, precipitation, rain, snowfall and wind, with 6/24/48-hour histories. `conditions_20261008/` and `conditions_final_20261012/`. Used through accepted-final historical weather components. Disabled in the ten new validation recipes (`include_era=False`). | [Open-Meteo publishes API data under CC BY 4.0](https://open-meteo.com/). Credit Open-Meteo and the underlying ERA5 providers. Features are transformations of hourly data, not unmodified observations. |

Aircraft rotation and callsign clocks are **derived features**, not additional
external datasets. `aircraft_rotation.py` combines supplied movement identity
with ADSB.lol callsign/address observations; `callsign_clocks.py` constructs the
six clock/ambiguity/censoring inputs. Sparse-track tensors are another derived
view of the same ADSB.lol observations. Public aggregates describe a different
population from the challenge data and are modeled as noisy evidence, not as
exact individual taxi labels.

Airport Corner event feeds and punctuality probes occur in exploratory scripts;
they are not declared inputs to either recipe documented here. A script's
presence in the research archive does not imply use in the selected model.

## Acquisition and derivation commands

Run these only in a new preparation environment. They download or build large
files and are not a request to overwrite the retained experiment snapshots.
Commands below were checked against the source entry points; they were not
executed during release preparation. They are individual preparation stages,
not a claim that rerunning current public downloads reproduces old bytes.

```bash
# Organizer-provided files; configure the opensky mc alias privately first.
mc mirror opensky/prc-2026-datasets/ data/

# METAR: defaults cover 2024-12-31 through 2026-07-31, UTC.
.venv/bin/python VALIDATION/tools/fetch_metar.py

# OPDI acquisition, then old-population feature derivation.
.venv/bin/python VALIDATION/tools/fetch_opdi.py --workers 4
.venv/bin/python VALIDATION/tools/fetch_opdi_final.py --workers 4
.venv/bin/python VALIDATION/opdi.py

# OSM and X-Plane source maps, then the derived coordinate tables.
.venv/bin/python VALIDATION/tools/fetch_stand_maps.py
.venv/bin/python VALIDATION/tools/build_stand_coords.py

# ADSB.lol: all 2025 and original January/July 2026 days by default.
.venv/bin/python VALIDATION/tools/download_adsblol_heatmap.py --workers 3
.venv/bin/python VALIDATION/adsblol.py
.venv/bin/python VALIDATION/adsblol.py stand

# Input-only aircraft identity, callsign features, and full-2025 sparse tracks.
.venv/bin/python VALIDATION/aircraft_rotation.py identity
.venv/bin/python VALIDATION/aircraft_rotation.py folds
.venv/bin/python VALIDATION/callsign_clocks.py
.venv/bin/python VALIDATION/build_sparse_track_full.py

# Public daily runway taxi source and extracted table.
.venv/bin/python VALIDATION/tools/bootstrap_public_daily_taxi_source.py
.venv/bin/python VALIDATION/tools/fetch_public_daily_taxi.py fetch

# ERA5 cache for historical final components; not needed by the ten candidates.
.venv/bin/python VALIDATION/external_conditions.py weather
.venv/bin/python VALIDATION/final_weather_features.py download
```

The ADSB downloader's default does **not** include final February/June 2026.
Request those days with `--days YYYY-MM-DD ...`; the retained release catalogs
and per-day manifests identify the actual `prod`/`staging` selection. Do not
replace a saved staging fallback with today's default prod feed. Final feature
views are separate from the original ranking view because the organizer's
timestamps differ; `final_submit.py tables --set final` is the historical
final-view builder, not an upload command. Further required prepared-artifact
dependencies are described in [REPRODUCING.md](REPRODUCING.md).

The four EUROCONTROL CSVs have direct, recorded URLs in
`data/external/eurocontrol_performance_20261011/manifest.json` and the release
manifest. There is no dedicated retained CSV-fetch CLI. For a fresh acquisition,
download the listed URLs to their listed paths and compare SHA-256; a changed
upstream CSV is a new snapshot, not an exact reproduction of the retained run.

## Snapshot guarantees and gaps

- The release manifest copies expected hashes from retained receipts and records
  where each value came from. It does not claim that every large input was
  rehashed during this documentation task. Twelve training-file hashes and the
  exact four-fold external feature tables are bound by the completed January
  input-preparation receipt.
- EUROCONTROL CSVs, daily dashboard responses, Open-Meteo requests and ADSB daily
  extraction manifests retain their acquisition provenance. The archive uses
  SHA-256 for content, with IDs checked separately before prediction joins.
- On this desktop, the original METAR acquisition manifest and raw stand-map
  acquisition manifest/source files are absent, although the derived METAR and
  coordinate tables used by the running validation are present and hash-bound.
  Their exact historical raw-source reconstruction is not claimed. Keep those
  derived snapshots in the reproducibility bundle and recover original manifests
  from a verified backup if available; a new download must not be silently
  substituted into an old run.
- OPDI's original downloader explicitly records a known missing upstream event
  interval. Missing observations remain missing; no target-derived replacement
  is downloaded. A new upstream release may change coverage.

No external service is needed when replaying an already complete, verified
artifact bundle. Fresh acquisition does require access to the providers above.
