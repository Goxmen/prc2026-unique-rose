"""Independent file/formula/CV audit, without importing any candidate runner."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_tabm_other_runway_combo_submit'
CV = RUNS / 'v3_20261012_tabm_other_runway_combo'
TABM = RUNS / 'v3_20261012_nn_public_tabm_submit_stack'
RUNWAY = RUNS / 'v3_20261012_joint_public_other_runway_submit'


def read(path):
    return json.loads(path.read_text())


def art(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(record):
    path = ROOT / record['path']
    assert art(path)['sha256'] == record['sha256'], record['path']
    return path


def main():
    output = OUT / 'independent_audit.json'
    assert not output.exists(), 'Preserve completed independent audit'
    identity = read(OUT / 'identity.json')
    result = read(OUT / 'result.json')
    assert result['identity'] == art(OUT / 'identity.json')
    assert result['no_version'] and result['no_upload']
    assert identity['design']['coefficients'] == [1, 1]
    assert identity['cv'] == art(CV / 'result.json')
    verify(identity['source'])
    for record in identity['sources'] + result['artifacts']:
        verify(record)
    for name, folder in [('tabm', TABM), ('runway', RUNWAY)]:
        component = read(folder / 'result.json')
        audit = read(folder / 'audit.json')
        assert identity['parts'][name]['result'] == art(folder / 'result.json')
        assert audit['status'] == 'pass' and audit['result'] == art(folder / 'result.json')
        assert component['identity'] == art(folder / 'identity.json')
        for record in component['artifacts']:
            verify(record)
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    table = pq.read_table(OUT / 'candidate.parquet')
    assert table.schema == template.schema and table.num_rows == 344841
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt':
            assert table[name].equals(template[name])
    values = table['TAXITIME_SEC_mvt'].to_numpy()
    assert table['TAXITIME_SEC_mvt'].null_count == 0
    assert np.isfinite(values).all() and (values >= 0).all()
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        paths = dict(candidate=OUT / 'candidate.parquet', tabm=TABM / 'predictions.parquet',
                     runway=RUNWAY / 'predictions.parquet', detail=OUT / 'predictions.parquet',
                     cv=CV / 'scored.parquet')
        for version in ['78', '79']:
            paths['v' + version] = verify(identity['controls'][version])
            parent = pq.read_table(paths['v' + version])
            assert parent.schema == template.schema
            for name in template.column_names:
                if name != 'TAXITIME_SEC_mvt':
                    assert parent[name].equals(template[name])
        for name, path in paths.items():
            con.read_parquet(str(path)).create_view(name)
        n, unique, bad, different = con.execute('''SELECT count(*),count(DISTINCT s.MVT_ID_mvt),
            sum(CASE WHEN s.TAXITIME_SEC_mvt IS DISTINCT FROM
                greatest(0,v79.TAXITIME_SEC_mvt::BIGINT
                           +t.tabm_v78::BIGINT+r.robust_v78::BIGINT
                           -2*v78.TAXITIME_SEC_mvt::BIGINT)
                OR t.v78 IS DISTINCT FROM v78.TAXITIME_SEC_mvt
                OR r.v78 IS DISTINCT FROM v78.TAXITIME_SEC_mvt
                OR t.v79 IS DISTINCT FROM v79.TAXITIME_SEC_mvt
                OR r.v79 IS DISTINCT FROM v79.TAXITIME_SEC_mvt
                OR d.pred IS DISTINCT FROM s.TAXITIME_SEC_mvt
                OR d.tabm_delta IS DISTINCT FROM t.tabm_v78-v78.TAXITIME_SEC_mvt
                OR d.runway_delta IS DISTINCT FROM r.robust_v78-v78.TAXITIME_SEC_mvt
                OR t.tabm_v78 IS NULL OR r.robust_v78 IS NULL
                OR NOT isfinite(t.tabm_v78) OR NOT isfinite(r.robust_v78)
                OR t.tabm_v78 != trunc(t.tabm_v78)
                OR r.robust_v78 != trunc(r.robust_v78)
                THEN 1 ELSE 0 END),
            sum(CASE WHEN s.TAXITIME_SEC_mvt != v79.TAXITIME_SEC_mvt THEN 1 ELSE 0 END)
            FROM candidate s JOIN v78 USING(MVT_ID_mvt) JOIN v79 USING(MVT_ID_mvt)
            JOIN tabm t ON t.mvt_id=s.MVT_ID_mvt
            JOIN runway r ON r.mvt_id=s.MVT_ID_mvt
            JOIN detail d ON d.mvt_id=s.MVT_ID_mvt''').fetchone()
        assert n == unique == 344841 and bad == 0 and different == result['changed']
        cv_count, cv_unique = con.execute('SELECT count(*),count(DISTINCT mvt_id) FROM cv').fetchone()
        assert cv_count == cv_unique == 344419
        scores = con.execute('''SELECT month,sqrt(avg(pow(control-y,2))),
            sqrt(avg(pow(combined-y,2))) FROM cv GROUP BY month ORDER BY month''').fetchall()
        assert [r[0] for r in scores] == ['2025-01', '2025-07']
        means = {'control': sum(r[1] for r in scores)/2,
                 'combined': sum(r[2] for r in scores)/2}
        for key, score in means.items():
            assert abs(score-result['cv'][key]) < 1e-8
        assert means['combined'] < means['control']
    report = dict(status='pass', source=art(Path(__file__)), identity=art(OUT / 'identity.json'),
        result=art(OUT / 'result.json'), file=art(OUT / 'candidate.parquet'),
        component_audits=[art(folder / 'audit.json') for folder in [TABM, RUNWAY]],
        rows=n, cv_rows=cv_count, changed_rows=different, cv=means,
        checks=['Integer-only independent SQL combination from exact scored v78/v79 files',
                'All source/component bindings and complete ID joins',
                'No separate-component clipping substituted into the frozen formula',
                'Original template schema and every non-target column preserved',
                'Independent monthly and equally averaged Jan/Jul CV arithmetic'],
        scope='Final composition/file audit; component model/source and raw CV truth checks remain in their prior audits.',
        no_fit=True, no_upload=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='pass', rows=n, changed=different, cv=means)))


if __name__ == '__main__':
    main()
