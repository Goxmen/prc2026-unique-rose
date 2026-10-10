"""Bounded independent production clock arithmetic, without model fitting."""
import hashlib
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'data/derived/callsign_submit_20261010'
FEATURES = ['cc_call_start_sec', 'cc_squawk_start_sec', 'cc_previous_call_end_sec',
            'cc_before_own_arrival', 'cc_ambiguous', 'cc_censored']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    con = duckdb.connect(config={'threads': 2})
    con.execute("SET TimeZone='UTC'")
    report = dict(scope='Independent derived-clock arithmetic and ranking view isolation; '
                       'raw interval extraction audited separately. No raw-source rescan or fit.', views={})
    for view in ['train', 'old', 'final']:
        record = json.loads((FOLDER / f'{view}.json').read_text())
        path = FOLDER / f'{view}.parquet'
        assert sha(path) == record['artifact']['sha256']
        con.read_parquet(str(path)).create_view('f', replace=True)
        n, unique = con.execute('SELECT count(*),count(DISTINCT mvt_id) FROM f').fetchone()
        assert n == unique == record['rows']
        counts = {c: con.execute(f'SELECT count({c}) FROM f').fetchone()[0] for c in FEATURES}
        assert counts == record['nonmissing']
        for c in FEATURES:
            lo, hi = (-180 if c in FEATURES[:2] else 0), (14400 if c in FEATURES[:3] else 1)
            bad = con.execute(f'''SELECT count(*) FROM f WHERE {c} IS NOT NULL
                AND (NOT isfinite({c}) OR {c}<? OR {c}>?)''', [lo, hi]).fetchone()[0]
            assert bad == 0, (view, c, bad)
        if view != 'train':
            refs = record['per_query_month_reference_months']
            assert refs == {m: [f'2025-{i:02d}' for i in range(1, 13)] + [m] for m in refs}
            con.read_parquet(str(FOLDER / view / 'shared.parquet')).create_view('s', replace=True)
            con.read_parquet(str(FOLDER / view / 'identity.parquet')).create_view('i', replace=True)
            assert con.execute("SELECT count(*) FROM f LEFT JOIN i USING(mvt_id) "
                               "WHERE i.phase IS DISTINCT FROM 'DEP'").fetchone()[0] == 0
            query = 'SELECT count(*) FROM f JOIN i USING(mvt_id) WHERE i.hex IS NULL AND ('
            query += ' OR '.join(f'f.{c} IS NOT NULL' for c in FEATURES) + ')'
            assert con.execute(query).fetchone()[0] == 0
            for c, raw in zip(FEATURES[:3], ['call_start_ms', 'squawk_start_ms', 'previous_call_end_ms']):
                lo = 0 if c == FEATURES[2] else -180
                query = f'''SELECT count(*) FROM f JOIN s USING(mvt_id) WHERE f.{c} IS DISTINCT FROM
                    CAST(CASE WHEN (s.to_ms-s.{raw})/1000 BETWEEN {lo} AND 14400
                    THEN (s.to_ms-s.{raw})/1000 END AS FLOAT)'''
                assert con.execute(query).fetchone()[0] == 0, (view, c)
            for c in FEATURES[4:]:
                query = f'SELECT count(*) FROM f JOIN s USING(mvt_id) WHERE f.{c} IS DISTINCT FROM CAST(s.{c} AS FLOAT)'
                assert con.execute(query).fetchone()[0] == 0
        report['views'][view] = dict(rows=n, sha256=record['artifact']['sha256'],
                                     nonmissing=counts, all_checks_pass=True)
    for view in ['old', 'final']:
        con.read_parquet(str(FOLDER / view / 'shared.parquet')).create_view(view + '_s')
    row = con.execute('''SELECT count(*),count(*) FILTER(WHERE a.to_ms<>b.to_ms),
        max(abs(a.to_ms-b.to_ms))/1000,
        count(*) FILTER(WHERE a.cc_call_start_sec IS NOT NULL AND b.cc_call_start_sec IS NOT NULL
                         AND a.cc_call_start_sec<>b.cc_call_start_sec),
        count(*) FILTER(WHERE a.call_start_ms=b.call_start_ms AND a.cc_call_start_sec IS NOT NULL
          AND b.cc_call_start_sec IS NOT NULL
          AND abs((b.cc_call_start_sec-a.cc_call_start_sec)-(b.to_ms-a.to_ms)/1000)>1e-8)
        FROM old_s a JOIN final_s b USING(mvt_id)''').fetchone()
    assert row[0] == 344841 and row[2] <= 12 and row[4] == 0
    report['isolated_jittered_views'] = dict(zip(['common_ids', 'changed_takeoff_clocks',
        'max_jitter_sec', 'changed_call_age', 'same_start_jitter_arithmetic_mismatches'], row))
    output = ROOT / 'tmp/research_20261010/error_mechanisms/callsign_production_features_audit.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(all_checks_pass=True, rows={v: r['rows'] for v, r in report['views'].items()},
                         jitter=report['isolated_jittered_views'], report=str(output.relative_to(ROOT)))))


if __name__ == '__main__':
    main()
