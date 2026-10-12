"""Read-only independent SQL audit of the fixed post-v79 bounds diagnostic."""
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261012_reapply_selected_bounds'


def run():
    import sys
    sys.path.insert(0, str(ROOT / 'VALIDATION'))
    from records import sha256, write_json
    import daily_delay_reconcile as base
    result = json.loads((OUT / 'result.json').read_text())
    design = json.loads((OUT / 'design.json').read_text())
    for record in design['sources']:
        assert sha256(ROOT / record['path']) == record['sha256']
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as con:
        for name, path in [('t', base.OUT / 'rows2025.parquet'),
                           ('r', base.OUT / 'ranking_inputs.parquet'),
                           ('h', OUT / 'cv.parquet'), ('p', OUT / 'ranking.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        counts = {}
        for table, rows in [('h', 't'), ('p', 'r')]:
            expression = '''round_even(case when r.has_id and isfinite(r.nm_gap)
                then greatest(0,least(greatest(p.parent,r.nm_gap-3606),r.nm_gap+3606))
                else p.parent end,0)'''
            n, unique, bad, maximum, rms = con.execute(f'''
                select count(*),count(distinct p.mvt_id),sum(case when
                    p.candidate is distinct from ({expression})
                    or p.matched is distinct from ({expression}) then 1 else 0 end),
                    max(abs(p.candidate-p.parent)),sqrt(avg(pow(p.candidate-p.parent,2)))
                from {table} p join {rows} r using(mvt_id)''').fetchone()
            assert bad == 0 and n == unique == (344419 if table == 'h' else 344841)
            counts[table] = dict(rows=n, bad=bad, max_change=maximum, rms_change=rms)
        monthly = dict(con.execute(
            'select month,sqrt(avg(pow(candidate-y,2))) from h group by month').fetchall())
        for month, value in monthly.items():
            assert abs(value - result['months'][month]['candidate']) < 1e-8
    report = dict(status='pass', source_sha256=sha256(__file__),
        method='Separate SQL reconstruction of every bound and score; no new fit or upload',
        result_sha256=sha256(OUT / 'result.json'), counts=counts,
        ranking_absolute_rmse_change_ceiling=counts['p']['rms_change'],
        decision='Closed: absolute possible leaderboard gain below0.089s; no selected candidate or slot')
    assert not (OUT / 'audit.json').exists()
    write_json(OUT / 'audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
