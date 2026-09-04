# Feature & target reference

Column-level documentation for the challenge parquet files, based on the [official data description](https://prc-data-challenge-2026.netlify.app/data.html) and verified against the actual parquet schemas. All files share the same 30 columns; `submitting.parquet` carries only `MVT_ID_mvt` and `TAXITIME_SEC_mvt`.

Every row is one **movement** (a departure *or* an arrival at a challenge airport). Columns come in two groups: `*_mvt` fields from the airport movement record, and `*_flt` fields from the matched Network Manager (NM) flight plan. Several `_mvt` fields are **phase-dependent**: their meaning flips between departure and arrival rows.

## Target

| Column | Type | Description |
|--------|------|-------------|
| `TAXITIME_SEC_mvt` | int32 | Taxi time in seconds — **taxi-out if `PHASE_mvt = "DEP"`, taxi-in if `"ARR"`**. This is the prediction target for departure rows; it is blanked (together with `BLOCK_TIME_UTC_mvt`) for `DEP` rows in `ranking.parquet`. |

Taxi-out = time between off-block (pushback from the stand) and takeoff, i.e. `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` for departures.

## Movement fields (`*_mvt`)

| Column | Type | Description |
|--------|------|-------------|
| `MVT_ID_mvt` | double | Unique identifier for the movement record. Join key with the submission template. |
| `FLIGHT_ID_mvt` | double | Unique identifier for the NM flight, **if matched** (may be missing). |
| `FLIGHT_mvt` | string | Flight number ("what you see on your boarding pass"). |
| `FLIGHT_RULE_mvt` | string | `I` = IFR, `V` = VFR, `NA` = unknown. |
| `ADEP_mvt` | string | ICAO code of the aerodrome of departure. |
| `ADES_mvt` | string | ICAO code of the aerodrome of destination. |
| `PHASE_mvt` | string | `DEP` = departure, `ARR` = arrival. Determines the meaning of every phase-dependent field below. |
| `MVT_TIME_UTC_mvt` | timestamp (UTC) | Best available movement time: **takeoff** (DEP) / **landing** (ARR). |
| `BLOCK_TIME_UTC_mvt` | timestamp (UTC) | Block time: **off-block** (DEP) / **in-block** (ARR). Blanked for DEP rows in `ranking.parquet` (it would give away the target). |
| `SCHED_TIME_UTC_mvt` | timestamp (UTC) | Scheduled time of departure (DEP) / arrival (ARR). |
| `AIRCRAFT_TYPE_mvt` | string | ICAO aircraft type code, e.g. `A21N` = Airbus A321neo. |
| `RUNWAY_mvt` | string | Runway ID used for departure (DEP) / arrival (ARR). |
| `STAND_mvt` | string | Stand ID of departure (DEP) / arrival (ARR). |

## Flight-plan fields (`*_flt`)

| Column | Type | Description |
|--------|------|-------------|
| `LOBT_flt` | timestamp (UTC) | Last known off-block time. |
| `CALLSIGN_flt` | string | Callsign of the flight, e.g. `BAW6VB`. |
| `ADEP_flt` | string | ICAO code of the ADEP per the flight plan. |
| `ADES_flt` | string | ICAO code of the ADES per the flight plan. |
| `ADES_FILED_flt` | string | ADES initially filed in the flight plan; if different from `ADES_flt`, the flight was **diverted**. |
| `MARKET_SEGMENT_flt` | string | One of: Mainline, Regional, Low-Cost, Business Aviation, All-Cargo, Charter, Military, Other, Not classified. |
| `IOBT_flt` | timestamp (UTC) | Initial Off-Block Time. |
| `FLIGHT_RULE_flt` | string | `I` = IFR, `V` = VFR, `Y` = IFR then VFR, `Z` = VFR then IFR. |
| `FLIGHT_TYPE_flt` | string | `S` = scheduled, `N` = non-scheduled, `G` = general aviation, `M` = military, `X` = other. |
| `AIRCRAFT_TYPE_flt` | string | ICAO aircraft type code per the flight plan (may disagree with `AIRCRAFT_TYPE_mvt`). |
| `WK_TBL_CAT_flt` | string | Wake turbulence category: `L` = light, `M` = medium, `H` = heavy, `J` = super. |
| `AIRCRAFT_OPERATOR_flt` | string | **Anonymized** ICAO airline designator. |
| `EOBT_1_flt` | timestamp (UTC) | Estimated Off-Block Time of the FPL-based (M1) trajectory. |
| `ARVT_1_flt` | timestamp (UTC) | Arrival time of the FPL-based (M1) trajectory. |
| `AOBT_3_flt` | timestamp (UTC) | Actual Off-Block Time of the flown (M3) trajectory. |
| `ARVT_3_flt` | timestamp (UTC) | Arrival time of the flown (M3) trajectory. |

## Data quirks to keep in mind

- Organizers: *"Real world data is messy. You will find some inconsistencies between movement and flight information"* — e.g. `_mvt` vs `_flt` fields (aircraft type, ADEP/ADES) may disagree, and `FLIGHT_ID_mvt` / all `_flt` fields can be missing when no NM flight was matched.
- Military, Head of State, and sensitive movements have been removed from the dataset.
- In `ranking.parquet`, only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blanked, and only on `DEP` rows — arrival rows are fully populated, and off-block estimates survive in `LOBT_flt` / `IOBT_flt` / `EOBT_1_flt` / `AOBT_3_flt` where matched.
