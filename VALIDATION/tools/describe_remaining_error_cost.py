"""Describe retained v77 validation errors without fitting or selecting a model.

Fixed input groups and label-tail summaries are exploratory descriptions. Error
rankings and perfect-prediction ceilings use held labels, not deployable rules.
"""
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261011_remaining_error_cost'
SCORED = ROOT / 'VALIDATION/runs/v3_20261011_daily_main_refinements_cv/scored.parquet'
RAW = ROOT / 'VALIDATION/runs/v3_20261011_daily_ifr/rows2025.parquet'
TRACKS = ROOT / 'data/derived/sparse_track_sequence_20261010/full_index.parquet'


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def describe():
    assert not (OUT / 'result.json').exists(), 'Preserve completed diagnostic'
    design = dict(sources=[art(p) for p in [Path(__file__), SCORED, RAW, TRACKS]],
        groups=['airport', 'id_group', 'track_group', 'label_group', 'airport_id', 'airport_track'],
        label_use='descriptive errors and optimistic perfect-prediction ceilings only',
        no_fit=True, no_candidate=True, no_upload=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'design.json', design)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('p', SCORED), ('r', RAW), ('t', TRACKS)]:
            con.read_parquet(str(path)).create_view(name)
        n, unique, errors = con.execute('''select count(*), count(distinct p.mvt_id),
            sum(case when p.y is distinct from r.y or p.day is distinct from r.day
                or p.month is distinct from r.month or r.airport is distinct from t.airport
                then 1 else 0 end)
            from p join r using(mvt_id) join t using(mvt_id)''').fetchone()
        assert n == unique == 344419 and errors == 0
        f = con.execute('''select p.mvt_id,p.month,p.day,p.y,p.combined pred,
            r.airport,r.has_id,r.gap,r.nm_gap,t.count,t.raw_ground_count,t.span_ms,
            t.stand,t.runway,t.clock_ms,t.to_ms,
            case when r.has_id then 'has_id' when r.airport='LIRF' and r.gap>=14400
              then 'noid_rome_4h_plus' when r.airport='LIRF' and r.gap>3600
              then 'noid_rome_1_4h' else 'other_noid' end id_group,
            case when t.count<2 then 'unusable' when t.raw_ground_count=0
              then 'airborne_only' else 'ground_available' end track_group,
            case when p.y>43200 then 'over12h' when p.y>10800 then '3_12h'
              when p.y>3600 then '1_3h' else 'under1h' end label_group
            from p join r using(mvt_id) join t using(mvt_id) order by p.mvt_id''').fetchdf()
        f['error'] = f.pred - f.y
        f['se'] = f.error**2
        f['airport_id'] = f.airport + ':' + f.id_group
        f['airport_track'] = f.airport + ':' + f.track_group
        result = {}
        for month, rows in f.groupby('month'):
            total, size = float(rows.se.sum()), len(rows)
            rmse = float(np.sqrt(total / size))
            groups = {}
            for key in design['groups']:
                groups[key] = sorted([dict(group=str(name), n=len(g),
                    sse_share=float(g.se.sum()/total), rmse=float(np.sqrt(g.se.mean())),
                    bias=float(g.error.mean()), mean_abs_error=float(g.error.abs().mean()),
                    perfect_group_ceiling=rmse-float(np.sqrt(max(total-g.se.sum(), 0)/size)))
                    for name, g in rows.groupby(key)], key=lambda x: x['sse_share'], reverse=True)
            ranked = rows.sort_values('se', ascending=False)
            top = {str(k): dict(sse_share=float(ranked.head(k).se.sum()/total),
                perfect_rows_ceiling=rmse-float(np.sqrt((total-ranked.head(k).se.sum())/size)))
                for k in [1, 10, 100, 1000]}
            result[month] = dict(n=size, rmse=rmse, groups=groups, top_k=top)
            # A separately reduced SQL score checks all descriptive Python SSE.
            con.register('month_rows', rows)
            score = con.execute('select sqrt(avg(pow(pred-y,2))) from month_rows').fetchone()[0]
            np.testing.assert_allclose(score, rmse, atol=1e-9, rtol=0)
            for values in groups.values():
                assert sum(g['n'] for g in values) == size
                np.testing.assert_allclose(sum(g['sse_share'] for g in values), 1, atol=1e-12, rtol=0)
        expected = json.loads(SCORED.with_name('result.json').read_text())['cv']['combined']
        actual = float(np.mean([v['rmse'] for v in result.values()]))
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        f.nlargest(100, 'se').to_csv(OUT / 'top100.csv', index=False)
    write_json(OUT / 'result.json', dict(design=design, cv=actual, months=result,
        audits=dict(truth_calendar_airport_ids_exact=True, sql_scores_exact=True,
                    groups_exhaustive=True),
        interpretation='Held-label diagnostic only; ceilings are not achievable gains or a 2026 forecast.'))
    print(json.dumps(dict(cv=actual, months={m: dict(top_k=v['top_k'],
        airports=v['groups']['airport'][:5], id_groups=v['groups']['id_group'],
        tracks=v['groups']['track_group'], tails=v['groups']['label_group']) for m,v in result.items()}), indent=2))


if __name__ == '__main__':
    with threadpool_limits(1): describe()
