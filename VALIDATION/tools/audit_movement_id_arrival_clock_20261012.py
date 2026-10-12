"""Independent scalar interpolation and SQL outcome audit of the frozen pilot."""
import bisect
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_movement_id_arrival_clock'
OUT = ROOT / 'VALIDATION/reports/source_clocks_20261012/movement_id_arrival_clock_audit.json'


def read(p):
    return json.loads(p.read_text())


def art(p):
    return dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    start = time.monotonic()
    assert not OUT.exists()
    identity, result = read(RUN / 'identity.json'), read(RUN / 'result.json')
    assert result['identity'] == art(RUN / 'identity.json')
    assert result['no_held_labels'] and result['no_fit'] and result['no_cv']
    for item in identity['sources'] + result['sources']:
        assert art(ROOT / item['path']) == item
    months = identity['design']['months']
    assert months == ['2025-03', '2025-09', '2025-10', '2025-11']
    cache = ROOT / 'data/derived/aircraft_rotation_20261010/identity.parquet'
    labels = ROOT / 'VALIDATION/runs/v3_20261011_daily_offblock_alignment/rows2025.parquet'
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        raw = con.execute('''SELECT mvt_id,phase,strftime(t,'%Y-%m') AS "month",
            epoch(t) AS landing,epoch(inblock) AS inblock FROM read_parquet(?)
            WHERE airport='LIRF' AND strftime(t,'%Y-%m') IN (SELECT unnest(?))''', [str(cache), months]).df()
        clocks = pd.read_parquet(RUN / 'input_clocks.parquet').set_index('mvt_id')
        assert clocks.index.is_unique and len(clocks) == result['rows']
        assert set(clocks.index) == set(raw.loc[raw.phase.eq('DEP'), 'mvt_id'])
        checked = 0
        for month in months:
            group = raw.loc[raw.month.eq(month)]
            arrivals = group.loc[group.phase.eq('ARR')].dropna(subset=['landing', 'inblock']).sort_values('mvt_id')
            ids = arrivals.mvt_id.tolist()
            landing, inblock = arrivals.landing.tolist(), arrivals.inblock.tolist()
            for query in group.loc[group.phase.eq('DEP'), 'mvt_id']:
                p = bisect.bisect_left(ids, query)
                row = clocks.loc[query]
                assert query not in ids[max(0, p-1):p+1] and row['month'] == month
                valid = p >= 5 and p + 5 <= len(ids)
                assert bool(row.available) == valid
                if not valid:
                    assert row[['landing', 'inblock', 'anchor_time_spread', 'id_span']].isna().all()
                    continue
                left, right = ids[p-5:p], ids[p:p+5]
                a, b = statistics.median(left), statistics.median(right)
                assert max(left) < query < min(right)
                weight = (query-a)/(b-a)
                for key, values in [('landing', landing), ('inblock', inblock)]:
                    low, high = statistics.median(values[p-5:p]), statistics.median(values[p:p+5])
                    expected = low + weight*(high-low)
                    assert math.isclose(row[key], expected, rel_tol=0, abs_tol=1e-6)
                assert row.id_span == b-a
                assert row.anchor_time_spread == max(inblock[p-5:p+5])-min(inblock[p-5:p+5])
                checked += 1
        for name, path in [('f', RUN/'diagnostic_rows.parquet'), ('inputs', RUN/'input_clocks.parquet'), ('truth', labels)]:
            con.read_parquet(str(path)).create_view(name)
        n, bad = con.execute('''SELECT count(*),sum(CASE WHEN
            f.month IS DISTINCT FROM t.month OR f.gap IS DISTINCT FROM t.gap OR
            f.has_id IS DISTINCT FROM t.has_id OR f.y IS DISTINCT FROM t.y OR
            f.takeoff_epoch IS DISTINCT FROM t.takeoff_epoch OR f.available IS DISTINCT FROM i.available OR
            f.landing IS DISTINCT FROM i.landing OR f.inblock IS DISTINCT FROM i.inblock OR
            t.airport <> 'LIRF' THEN 1 ELSE 0 END)
            FROM f JOIN truth t USING(mvt_id) JOIN inputs i USING(mvt_id)''').fetchone()
        assert n == result['rows'] and bad == 0
        for record in result['records']:
            where = 'TRUE' if record['month'] == 'ALL' else f"month='{record['month']}'"
            if record['scope'] == 'primary':
                where += ' AND NOT has_id AND gap>3600 AND gap<=14400'
            con.execute('CREATE OR REPLACE TEMP VIEW subset AS SELECT * FROM f WHERE ' + where)
            rows, covered, positives = con.execute('''SELECT count(*), count(*) FILTER(WHERE available),
                count(*) FILTER(WHERE available AND abs(gap-y)<=60) FROM subset''').fetchone()
            assert (rows, covered, positives) == (record['rows'], record['covered'], record['positives'])
            assert record['negatives'] == covered-positives and record['coverage'] == covered/max(rows, 1)
            for key in ['landing', 'inblock']:
                con.execute(f'''CREATE OR REPLACE TEMP VIEW ranked_source AS SELECT abs(gap-y)<=60 AS label,
                    (takeoff_epoch-{key}-1050)/(gap-1050) AS score FROM subset WHERE available''')
                rank_sum, n1, n0 = con.execute('''WITH finite AS (SELECT * FROM ranked_source WHERE isfinite(score)),
                    ranks AS (SELECT *,rank() OVER(ORDER BY score)+(count(*) OVER(PARTITION BY score)-1)/2.0 AS r FROM finite)
                    SELECT sum(r) FILTER(WHERE label),count(*) FILTER(WHERE label),count(*) FILTER(WHERE NOT label) FROM ranks''').fetchone()
                auc = None if not n1 or not n0 else (rank_sum-n1*(n1+1)/2)/(n1*n0)
                saved = record['clocks'][key]['copy_auc']
                assert auc is None and saved is None or math.isclose(auc, saved, abs_tol=1e-12, rel_tol=0)
                quantiles = con.execute(f'SELECT quantile_cont(abs({key}-(takeoff_epoch-y)),[.5,.9,.99]) FROM subset WHERE available').fetchone()[0]
                if quantiles is not None:
                    np.testing.assert_allclose(quantiles, record['clocks'][key]['absolute_error_quantiles'], rtol=0, atol=1e-6)
    primary = [r for r in result['records'] if r['scope'] == 'primary']
    pooled = next(r for r in primary if r['month'] == 'ALL')
    individual = [r for r in primary if r['month'] != 'ALL']
    enough = len(individual) == 4 and all(min(r['positives'], r['negatives']) >= 10 for r in individual)
    retain = enough and pooled['coverage'] >= .8 and pooled['clocks']['inblock']['copy_auc'] >= .65
    retain = retain and all(r['clocks']['inblock']['copy_auc'] >= .55 for r in individual)
    assert bool(retain) == result['decision']['retain_for_future_model_design']
    assert enough == result['decision']['enough_class_support']
    value = dict(status='pass', source=art(Path(__file__)), identity=art(RUN/'identity.json'),
        result=art(RUN/'result.json'), rows=n, scalar_clocks_checked=checked, seconds=time.monotonic()-start,
        decision=result['decision'], checks=['Every available clock independently reconstructed with bisect and stdlib medians',
        'All IDs, source labels/calendar, counts, coverage, error quantiles and SQL tie-aware AUC',
        'Original frozen success gate reproduced; no secondary clock selection'],
        scope='Root independent implementation audit; no unavailable subagent review claimed.')
    OUT.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass', rows=n, decision=value['decision'], seconds=value['seconds']), indent=2))


if __name__ == '__main__':
    main()
