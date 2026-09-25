# Stand coordinates: idea 34, step 1

**99.07% of ranking departures with a named stand now have a stand position
(training: 99.23%). Every airport is above 96%.** The positions come from two
public maps, and both checks we can make without a model support them. Where the
two maps have the same stand, they agree to within 22–69 m (median per airport).
And stands close together on the map have similar taxi times.

| Airport | X-Plane map uploaded | Ranking located | Training located | Map agreement (median) | Departure taxi, 3 nearest / airport guess (MAE) | 2026 arrival taxi-in, 3 nearest / airport (MAE) |
|---|---|---:|---:|---:|---:|---:|
| EDDF | 2024-08-16 | 97.57% | 98.38% | 44 m | 70 / 172 s | 26 / 59 s |
| EDDM | 2026-06-05 | 99.95% | 99.97% | 29 m | 37 / 84 s | 22 / 39 s |
| EGLL | 2026-07-01 | 98.95% | 99.07% | 53 m | 68 / 97 s | 37 / 65 s |
| EHAM | 2026-08-31 | 99.36% | 99.32% | 42 m | 30 / 122 s | 36 / 111 s |
| LEBL | 2024-05-13 | 99.65% | 99.83% | 30 m | 55 / 199 s | 27 / 62 s |
| LEMD | 2023-12-22 | 99.35% | 99.33% | 47 m | 43 / 148 s | 32 / 139 s |
| LFPG | 2026-05-26 | 100.00% | 100.00% | 31 m | 52 / 112 s | 32 / 87 s |
| LIRF | 2026-04-09 | 98.70% | 99.05% | 22 m | 30 / 171 s | 39 / 77 s |
| LSZH | 2025-03-24 | 96.54% | 96.38% | 47 m | 40 / 163 s | 22 / 68 s |
| LTFM | 2025-01-30 | 99.66% | 99.90% | 69 m | 27 / 112 s | 42 / 138 s |

"Located" excludes departures with no stand name (null or `UNKNOWN`). These are
0.82% of LEBL ranking departures and at most 0.01% elsewhere. The neighbour
columns take each stand's median taxi time and guess it two ways: from its three
nearest stands, or from all other stands at the airport. The departure column
uses stands with at least 200 departures in 2025. The arrival column uses stands
with at least 50 labelled arrivals in `ranking.parquet` (January/July 2026).
Those are supplied labels, not leaderboard feedback. When positions are shuffled
within an airport, the nearest-stand guess gets worse than the airport guess
(e.g. EHAM departures: 144 s). So the advantage comes from where the stands are.

This shows the positions are right and that taxi time varies smoothly across an
airport. It is not a model gain. The model already knows every well-used stand
by name. Coordinates can only add something where the name does not help.

## Where coordinates can matter

Ranking departures at stands with few or no 2025 departures:

| Airport | Unseen in 2025 | Located | Under 50 in 2025 | Located | Still unlocated |
|---|---:|---:|---:|---:|---|
| EDDM | 1,182 | 99.58% | 1,212 | 98.93% | a few single flights |
| EDDF | 300 | 67.33% | 2,335 | 85.40% | J1 153, J13 87, V333/V335/V337 97 |
| LSZH | 58 | 27.59% | 290 | 75.86% | T50 38, GA8 8, F3 6, X17 6 |
| LTFM | 6 | 16.67% | 378 | 87.57% | K27 41, 701 5 |
| LEBL | 1 | 100% | 174 | 77.01% | 9xx remote stands, 40 flights |
| other five | 42 | — | 915 | 86.2–100% | under 25 flights each |

Munich 103–108 (1,177 ranking departures, no 2025 departures) all have positions.
Frankfurt's new Terminal 3 area is partly covered. J5 and J10 come from
OpenStreetMap. J1 and V333–V337 appear in neither map. J13 appears three times in
OpenStreetMap, and one entry is 500 m from the other two, so it is treated as
ambiguous and dropped. Frankfurt's X-Plane map predates Terminal 3.

## How stands are matched

Each stand name is reduced to one spelling on both sides: upper case, words like
"Gate"/"Stand"/"Ramp" removed, no spaces, no leading zeros, and no LTFM size code.
So "H01" = "H1", "Gate E 58" = "E58" and "K08-(E)" = "K8". A stand then matches
by the first rule that works:

1. **exact** in either map (X-Plane first). This covers 97.7% of located ranking departures.
2. **substands**: the map splits the stand into lettered halves ("210" → "210A",
   "210B"), and their positions are averaged. This covers EDDM's 2xx stands: 29.2% of EDDM ranking departures.
3. **parent**: the map lacks the stand's last letter ("151A" → "151").

A map name that appears twice more than 150 m apart is dropped. Two stand groups
have the same name in both maps but for places 150 m to 1.4 km apart. For these,
the map that matches our data is used, set in `SOURCE_ORDER`:

- **LSZH 101–105 → OpenStreetMap.** 98% of their 2025 departures are business
  aviation (mostly Citations, from the same operator as 106–112). X-Plane's
  "Stand 101–105" is an airline apron 1.3 km away. Its "GAC 101–105" agrees
  with OpenStreetMap.
- **EDDM 103–105 → OpenStreetMap.** In 2026, arrivals at these stands take a median
  415–428 s to taxi in. OpenStreetMap's neighbours 110–112 take 364–485 s, while
  X-Plane's neighbours 175/181/182 take 306–357 s. The choice matters: the
  departure guess from nearby stands for 104/105 is 923 s (OSM) versus 797 s (X-Plane).
  **EDDM 106/107** (304 ranking departures) exist only in X-Plane, whose layout
  for this group disagrees with OpenStreetMap's. Treat them as low confidence.

LEBL 124A (4 ranking departures) is 154 m apart between the maps. It stays with X-Plane.

**Runways.** Every departure runway has an X-Plane end, except `NA`/`HEL`/`H`
and six LIRF `01R` rows (at most 0.3% per airport). Each end matches an
OpenStreetMap runway end. It matches either within 25 m or within 25 m of the
X-Plane displaced-threshold distance, because OpenStreetMap draws some runways
from the threshold. The one exception is EDDF 36, which has no departures.

## Files and reproduction

```sh
.venv/bin/python VALIDATION/tools/fetch_stand_maps.py      # ~8 min, mostly Overpass waits
.venv/bin/python VALIDATION/tools/build_stand_coords.py    # ~2 s, local only
.venv/bin/python -m unittest VALIDATION/tests/test_stand_geometry.py
```

- Matching: `VALIDATION/stand_geometry.py`. 18 tests pass.
- Outputs (gitignored): `data/external/stand_maps/stand_coords.parquet` (one
  row per airport × stand, with source, rule and the two maps' gap) and
  `runway_ends.parquet`. The manifest records scenery IDs, dates and SHA-256 hashes.
- Full figures: `stand_coords_20260925.json` in this folder.
- The ADS-B probe's partial Rome OpenStreetMap file (parking and aprons only) was
  kept as `data/external/adsb/osm_LIRF.adsb_probe.json`. It was replaced by a full fetch.

**External data to declare:** the X-Plane Scenery Gateway airport data (GPL v2 or later),
fetched 2026-09-25, and OpenStreetMap (ODbL). The organisers approved
OpenStreetMap on 2026-09-09. The Gateway is not on their list, although rank 3
declares using it.

## Next

Step 2: add stand position and straight-line distance to the departure runway's
end as inputs to v42's recipe. Then score them on all four validations, plus a test
that hides some stands from training, because ordinary validation almost never
contains unseen stands. No fits, uploads or ledger changes were made in step 1.
