"""Give every departure stand a position, and check the positions (idea 34, step 1).

Matches each (airport, STAND_mvt) seen on a training or ranking departure to the
X-Plane Gateway or OpenStreetMap stand maps (`stand_geometry.locate`), then runs
four checks that need no model fit:

* coverage: share of training and ranking departures whose stand has a position;
* agreement: distance between the two maps where both have the same stand;
* runway ends: every runway in `RUNWAY_mvt` has an X-Plane end, and each end
  lies at an end of OpenStreetMap's drawing of the same runway;
* neighbours: a well-used stand's median taxi is guessed from its three nearest
  well-used stands and from the airport as a whole. If positions are right and
  carry signal, the neighbour guess wins, and loses that edge when positions are
  shuffled within the airport. The same check runs on the 2026 arrivals' taxi-in
  in `ranking.parquet` (supplied labels, not leaderboard feedback), which also
  covers stands first used in 2026.

Reads only local files (`fetch_stand_maps.py` downloads the maps).
Run: .venv/bin/python VALIDATION/tools/build_stand_coords.py
"""

import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stand_geometry as sg

REPORT = sg.REPO / 'VALIDATION/reports/stand_coords_20260925.json'
MIN_DEPS = 200          # "well-used" stand for the neighbour check
K_NEAREST = 3
SHUFFLES = 50
RUNWAY_TOL_M = 25.0
RARE_BELOW = 50        # training departures below which a stand counts as rare
MIN_ARRS = 50           # 2026 arrivals for a stand to enter the arrival check
# stands that were never used in 2025 but carry ranking departures, named in the report
NEW_STANDS = {'EDDM': ['103', '104', '105', '106', '107', '108'],
              'EDDF': ['J5', 'J10', 'V333', 'V335', 'V337']}


def departure_stands():
    con = duckdb.connect()
    return con.sql(f"""
      with tr as (select ADEP_mvt ap, STAND_mvt stand, count(*) n_train,
                         median(TAXITIME_SEC_mvt) med_taxi
                  from read_parquet('{sg.REPO}/data/training_*.parquet')
                  where PHASE_mvt = 'DEP' group by all),
           rk as (select ADEP_mvt ap, STAND_mvt stand, count(*) n_rank
                  from read_parquet('{sg.REPO}/data/ranking.parquet')
                  join read_parquet('{sg.REPO}/data/submitting.parquet') using (MVT_ID_mvt)
                  group by all)
      select coalesce(tr.ap, rk.ap) airport, coalesce(tr.stand, rk.stand) stand,
             coalesce(n_train, 0) n_train, coalesce(n_rank, 0) n_rank, med_taxi
      from tr full join rk on tr.ap = rk.ap and tr.stand is not distinct from rk.stand
      order by all""").df()


def arrival_taxi_in():
    """Median 2026 taxi-in per arrival stand, from the labelled ranking arrivals."""
    return duckdb.connect().sql(f"""
      select ADES_mvt airport, STAND_mvt stand, count(*) n_arr, median(TAXITIME_SEC_mvt) med_in
      from read_parquet('{sg.REPO}/data/ranking.parquet')
      where PHASE_mvt = 'ARR' and TAXITIME_SEC_mvt is not null group by all""").df()


def departure_runways():
    con = duckdb.connect()
    return con.sql(f"""
      select ADEP_mvt airport, RUNWAY_mvt runway, count(*) n
      from read_parquet(['{sg.REPO}/data/training_*.parquet', '{sg.REPO}/data/ranking.parquet'])
      where PHASE_mvt = 'DEP' group by all order by all""").df()


def osm_runway_ends(data):
    """Runway designator -> list of (lat, lon) way endpoints from OSM."""
    out = {}
    for el in data.get('elements', []):
        tags = el.get('tags', {})
        if tags.get('aeroway') != 'runway' or el['type'] != 'way' or not el.get('geometry'):
            continue
        g = el['geometry']
        for des in str(tags.get('ref', '')).replace(' ', '').split('/'):
            if des:
                out.setdefault(des.upper(), []).extend(
                    [(g[0]['lat'], g[0]['lon']), (g[-1]['lat'], g[-1]['lon'])])
    return out


def neighbour_check(df, rng, n_col='n_train', y_col='med_taxi', min_n=MIN_DEPS):
    """Mean absolute error (s) of three guesses for each well-used stand's median."""
    d = df[(df[n_col] >= min_n) & df.lat.notna()].reset_index(drop=True)
    if len(d) < K_NEAREST + 2:
        return None
    y = d[y_col].to_numpy()

    def knn_mae(lat, lon):
        dist = sg.metres(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        np.fill_diagonal(dist, np.inf)
        near = np.argsort(dist, axis=1, kind='stable')[:, :K_NEAREST]
        return float(np.mean(np.abs(y[near].mean(axis=1) - y)))

    lat, lon = d.lat.to_numpy(), d.lon.to_numpy()
    airport = float(np.mean(np.abs((y.sum() - y) / (len(y) - 1) - y)))
    shuffled = [knn_mae(lat[p], lon[p]) for p in (rng.permutation(len(d)) for _ in range(SHUFFLES))]
    return {'stands': len(d), 'airport_mae': round(airport, 1),
            'neighbour_mae': round(knn_mae(lat, lon), 1),
            'shuffled_mae_mean': round(float(np.mean(shuffled)), 1),
            'shuffled_mae_min': round(float(np.min(shuffled)), 1)}


def pct(num, den):
    return round(100.0 * num / den, 2) if den else None


def main():
    deps = departure_stands()
    arrs = arrival_taxi_in()
    runways = departure_runways()
    rng = np.random.default_rng(0)
    rows, report, ends_out = [], {'airports': {}}, []
    for icao in sg.AIRPORTS:
        maps, xp_raw, osm_raw, ends = sg.load_maps(icao)
        d = deps[deps.airport == icao].copy()
        loc = pd.DataFrame([sg.locate(s, maps, icao) for s in d.stand], index=d.index)
        d = pd.concat([d, loc], axis=1)
        both = d.key.isin(maps['xplane'].keys() & maps['osm'].keys())
        xy = np.array([maps['xplane'][k] + maps['osm'][k] for k in d.key[both]]).reshape(-1, 4)
        d['maps_gap_m'] = np.nan
        d.loc[both, 'maps_gap_m'] = sg.metres(xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3])
        rows.append(d)
        r = {'xplane_stands': len(maps['xplane']), 'osm_stands': len(maps['osm'])}

        # coverage, by departures; unnamed stands (null / UNKNOWN) reported apart
        for col in ('n_rank', 'n_train'):
            tot = d[col].sum()
            named = d.loc[d.key.notna() & ~d.key.isin(sg.NOT_A_STAND), col].sum()
            got = d.loc[d.lat.notna(), col].sum()
            by = d[d.lat.notna()].groupby(['source', 'rule'])[col].sum()
            r[col] = {'departures': int(tot), 'no_stand_name_pct': pct(tot - named, tot),
                      'located_pct_of_named': pct(got, named),
                      'by_source_rule_pct_of_named': {f'{s}/{u}': pct(v, named) for (s, u), v in by.items()},
                      'missing_top': [[s, int(n)] for s, n in d[d.lat.isna() & d.key.notna()
                                      & ~d.key.isin(sg.NOT_A_STAND)].sort_values(
                                          [col, 'stand'], ascending=[False, True])[['stand', col]]
                                      .head(8).itertuples(index=False) if n > 0]}

        # where positions can matter: ranking departures at stands 2025 barely saw
        for label, lo, hi in (('unseen_2025', 0, 0), ('under_50_in_2025', 0, RARE_BELOW - 1)):
            rare = d[d.key.notna() & ~d.key.isin(sg.NOT_A_STAND) & d.n_train.between(lo, hi)]
            r[f'rank_{label}'] = {'departures': int(rare.n_rank.sum()),
                                  'located_pct': pct(rare.loc[rare.lat.notna(), 'n_rank'].sum(),
                                                     rare.n_rank.sum()),
                                  'unlocated': [[t.stand, int(t.n_rank)] for t in rare[rare.lat.isna()
                                                & (rare.n_rank > 0)].sort_values(
                                                    ['n_rank', 'stand'], ascending=[False, True]).itertuples()]}

        # two raw names that land on one canonical key would share a position
        dup = d[d.key.notna()].groupby('key').stand.nunique()
        r['canonical_collisions'] = sorted(dup[dup > 1].index.tolist())

        # agreement where both maps have the exact stand; a gap over SAME_STAND_M
        # means one name for two places, resolved by SOURCE_ORDER or left to X-Plane
        b = d.dropna(subset=['maps_gap_m']).rename(columns={'maps_gap_m': 'm'})
        if len(b):
            r['map_agreement_m'] = {'stands': len(b), 'median': round(float(b.m.median()), 1),
                                    'p95': round(float(b.m.quantile(.95)), 1),
                                    'over_100m': int((b.m > 100).sum()),
                                    'worst': [[k, round(float(m))] for k, m in
                                              b.sort_values(['m', 'key'], ascending=[False, True])
                                              [['key', 'm']].head(5).itertuples(index=False)]}
            c = b[b.m > sg.SAME_STAND_M].sort_values('key')
            r['map_conflicts'] = [{'stand': t.stand, 'gap_m': round(float(t.m)), 'used': t.source,
                                   'ranking_departures': int(t.n_rank), 'training_departures': int(t.n_train)}
                                  for t in c.itertuples()]

        # positions must sit on the airport: within 6 km of the runway ends' centre
        clat, clon = ends.lat.mean(), ends.lon.mean()
        far = d[d.lat.notna() & (sg.metres(d.lat, d.lon, clat, clon) > 6000)]
        r['positions_over_6km'] = far.stand.tolist()

        # runway ends: every departure runway present; ends agree with OSM
        rw = runways[runways.airport == icao]
        have = set(ends.runway)
        r['runway_departures_with_end_pct'] = pct(rw[rw.runway.isin(have)].n.sum(), rw.n.sum())
        r['runways_without_end'] = rw[~rw.runway.isin(have)].runway.tolist()
        osm_ends = osm_runway_ends(json.loads((sg.OSM_DIR / f'osm_{icao}.json').read_text()))
        # OSM draws some runways from the landing threshold, X-Plane from the
        # paved end: a gap equal to X-Plane's displaced threshold is agreement
        gaps = {}
        for e in ends.itertuples():
            pts = osm_ends.get(e.runway)
            if pts:
                gap = float(min(sg.metres(e.lat, e.lon, p[0], p[1]) for p in pts))
                gaps[e.runway] = {'gap_m': round(gap), 'displaced_m': round(e.displaced_m),
                                  'agrees': bool(min(gap, abs(gap - e.displaced_m)) <= RUNWAY_TOL_M)}
        r['runway_end_vs_osm'] = gaps
        r['runway_ends_unverified'] = sorted(set(ends.runway) - {k for k, v in gaps.items() if v['agrees']})
        ends_out.append(ends.assign(airport=icao))

        r['neighbour_check'] = neighbour_check(d, rng)
        a = d.merge(arrs[arrs.airport == icao].drop(columns='airport'), on='stand', how='left')
        r['arrival_neighbour_check_2026'] = neighbour_check(a, rng, 'n_arr', 'med_in', MIN_ARRS)
        if icao in NEW_STANDS:
            r['new_stands'] = {s: (None if d[d.stand == s].lat.isna().all() else
                                   {'source': d[d.stand == s].source.iloc[0],
                                    'rule': d[d.stand == s].rule.iloc[0],
                                    'ranking_departures': int(d[d.stand == s].n_rank.iloc[0])})
                               for s in NEW_STANDS[icao]}
        report['airports'][icao] = r

    table = pd.concat(rows, ignore_index=True)
    for col in ('n_rank', 'n_train'):
        named = table.key.notna() & ~table.key.isin(sg.NOT_A_STAND)
        report[f'all_{col}_located_pct_of_named'] = pct(table.loc[table.lat.notna(), col].sum(),
                                                        table.loc[named, col].sum())
    sg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    table[['airport', 'stand', 'key', 'lat', 'lon', 'source', 'rule', 'maps_gap_m', 'n_train', 'n_rank']] \
        .to_parquet(sg.OUT_DIR / 'stand_coords.parquet', index=False)
    pd.concat(ends_out, ignore_index=True)[['airport', 'runway', 'lat', 'lon', 'displaced_m']] \
        .to_parquet(sg.OUT_DIR / 'runway_ends.parquet', index=False)
    report['manifest'] = json.loads((sg.OUT_DIR / 'manifest.json').read_text())
    REPORT.write_text(json.dumps(report, indent=1) + '\n')

    print('airport  rank%  train%  noname%  agree_med/p95/>100m   dep nbr/airport/shuffled MAE'
          '   arr2026 nbr/airport/shuffled   runways%')
    for icao, r in report['airports'].items():
        a, n = r.get('map_agreement_m', {}), r['neighbour_check'] or {}
        v = r['arrival_neighbour_check_2026'] or {}
        print(f"{icao}  {r['n_rank']['located_pct_of_named']:6.2f} {r['n_train']['located_pct_of_named']:6.2f}"
              f"  {r['n_rank']['no_stand_name_pct']:6.2f}   {a.get('median')}/{a.get('p95')}/{a.get('over_100m')}"
              f"   {n.get('neighbour_mae')}/{n.get('airport_mae')}/{n.get('shuffled_mae_mean')}"
              f"   {v.get('neighbour_mae')}/{v.get('airport_mae')}/{v.get('shuffled_mae_mean')}"
              f"   {r['runway_departures_with_end_pct']}")
    for icao, r in report['airports'].items():
        u, w = r['rank_unseen_2025'], r['rank_under_50_in_2025']
        if w['departures']:
            print(f"{icao} rare stands: unseen {u['departures']} dep ({u['located_pct']}% located), "
                  f"<{RARE_BELOW} in 2025 {w['departures']} dep ({w['located_pct']}%), unlocated {w['unlocated'][:6]}")
    print('all ranking located:', report['all_n_rank_located_pct_of_named'],
          '| all training located:', report['all_n_train_located_pct_of_named'])
    print('report:', REPORT.relative_to(sg.REPO))


if __name__ == '__main__':
    main()
