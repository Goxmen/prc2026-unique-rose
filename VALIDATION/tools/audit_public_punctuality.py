"""Independent SQL arithmetic and raw-public-source checks for nine inputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import duckdb
import numpy as np
import pandas as pd
import public_punctuality_features as build
from records import write_json


def main():
    build.prepare()
    raw = json.loads(build.SOURCE.read_text())
    names = {'EHAM': 'Amsterdam', 'LEBL': 'Barcelona', 'EDDF': 'Frankfurt',
        'LTFM': 'Istanbul', 'EGLL': 'London Heathrow', 'LEMD': 'Madrid Barajas',
        'EDDM': 'Munich', 'LFPG': 'Paris Charles de Gaulle', 'LIRF': 'Rome Fiumicino', 'LSZH': 'Zurich'}
    reverse = {v: k for k, v in names.items()}
    records = []
    for values in raw[2:]:
        if values[1] in reverse:
            records.append(dict(day=values[0], airport=reverse[values[1]],
                dep_otp=float(values[2]), arr_otp=float(values[3]), dep_mean=60*float(values[4]),
                arr_mean=60*float(values[5]), operated=float(values[7])))
    source = pd.DataFrame(records)
    for airport in names:
        observed = set(source.loc[source.airport.eq(airport), 'day'])
        needed = set(pd.date_range('2025-01-01', '2025-12-31').strftime('%Y-%m-%d'))
        for start, end in [('2026-01-01', '2026-02-28'), ('2026-06-01', '2026-07-31')]:
            needed.update(pd.date_range(start, end).strftime('%Y-%m-%d'))
        assert needed <= observed
    with duckdb.connect(config={'threads': 1}) as con:
        con.register('public', source)
        q = con.execute('''with agg as (
          select airport, day, avg(cast(gap<=1800 as double)) otp900,
            avg(cast(gap<=2700 as double)) otp1800, avg(gap-900) mean900,
            avg(gap-least(greatest(coalesce(clock,1200),30),7200)) clockmean
          from read_parquet(?) where fltrule is distinct from 'V' group by airport,day)
          select r.mvt_id, p.dep_otp pun_dep_otp15, p.arr_otp pun_arr_otp15,
            p.dep_mean pun_dep_mean, p.arr_mean pun_arr_mean, p.operated pun_operated,
            p.dep_otp-a.otp900 pun_otp900_resid, p.dep_otp-a.otp1800 pun_otp1800_resid,
            p.dep_mean-a.mean900 pun_mean900_resid, p.dep_mean-a.clockmean pun_clock_mean_resid
          from read_parquet(?) r left join agg a using(airport,day)
            left join public p using(airport,day) order by r.mvt_id''',
          [str(build.daily.OUT / 'inputs2025.parquet')]*2).fetchdf()
        # Descriptive source comparison only. These labels never enter features.
        truth = con.execute('''select r.airport,r.day,count(*) n,avg(r.gap-r.y)/60 own_mean,
            avg(cast(r.gap-r.y<=900 as double)) own_otp
          from read_parquet(?) r join read_parquet(?) f using(mvt_id)
          where f.fltrule is distinct from 'V' group by r.airport,r.day''',
          [str(build.daily.base.OUT / 'rows2025.parquet'), str(build.daily.OUT / 'inputs2025.parquet')]).fetchdf()
    actual = pd.read_parquet(build.OUT / 'features2025.parquet').sort_values('mvt_id').reset_index(drop=True)
    np.testing.assert_array_equal(actual.mvt_id, q.mvt_id)
    np.testing.assert_allclose(actual[build.FEATURES], q[build.FEATURES], atol=1e-9, rtol=0)
    comparison = truth.merge(source, on=['airport', 'day'], validate='one_to_one')
    assert len(comparison) == 3650
    stats = []
    for airport, group in comparison.groupby('airport'):
        public_mean = group.dep_mean/60
        stats.append(dict(airport=airport, days=len(group),
            signed_mean_bias_minutes=float((public_mean-group.own_mean).median()),
            mean_absolute_difference_minutes=float((public_mean-group.own_mean).abs().mean()),
            mean_correlation=float(public_mean.corr(group.own_mean)),
            otp_absolute_difference_percentage_points=float(100*(group.dep_otp-group.own_otp).abs().mean()),
            otp_correlation=float(group.dep_otp.corr(group.own_otp))))
    result = dict(status='pass', source=build.daily.base.artifact(build.SOURCE),
        auditor=build.daily.base.artifact(__file__), feature_manifest=build.daily.base.artifact(build.OUT / 'manifest.json'),
        rows=len(q), feature_values=int(len(q)*len(build.FEATURES)),
        maximum_absolute_difference=float(np.max(np.abs(actual[build.FEATURES].to_numpy()-q[build.FEATURES].to_numpy()))),
        full_2025_and_all_four_ranking_months_covered=True,
        independent_raw_json_parse=True, independent_sql_arithmetic=True,
        descriptive_source_comparison=stats,
        interpretation='Closely correlated but different airline schedules/population; features only, no exact aggregate constraint')
    write_json(build.OUT / 'audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
