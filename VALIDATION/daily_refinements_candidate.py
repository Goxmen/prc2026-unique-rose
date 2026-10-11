"""Package the selected combined component as one reviewable submission.

No model fitting, new calibration or upload. The retained component already
contains the exact original serialization boundaries and Jan/Jul validation.
"""
import argparse
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import daily_main_refinements_submit as component
from records import sha256, write_json

ROOT = component.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_daily_refinements_candidate'
art, read, verify = component.art, component.read, component.verify


def build(version):
    destination = OUT / f'submission_v{version}'
    output = ROOT / f'submissions/unique-rose_v{version}.parquet'
    assert version > 76 and not destination.exists() and not output.exists()
    component.guard()
    audit = read(component.OUT / 'audit.json')
    result = read(component.OUT / 'result.json')
    assert audit['status'] == 'pass' and audit['result'] == art(component.OUT / 'result.json')
    assert audit['identity'] == art(component.MANIFEST)
    assert result['parent_control_all_stages_exact'] and result['paired']['decision'] == 'improvement'
    assert result['rows'] == 344841
    for record in result['artifacts']: verify(record)
    parent_directory = component.base.aggregate.V76
    parent = read(parent_directory / 'manifest.json')
    assert parent['version'] == 76
    assert result['jan_jul_cv'] < parent['jan_jul_cv']
    for name, digest in parent['sources'].items(): assert sha256(ROOT / name) == digest, name
    assert sha256(ROOT / parent['file']) == parent['sha256']
    table = pq.read_table(component.OUT / 'candidate.parquet')
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    assert table.schema == template.schema and table.num_rows == 344841
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt': assert table[name].equals(template[name])
    values = table['TAXITIME_SEC_mvt'].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    old = pq.read_table(ROOT / parent['file'])
    assert old['MVT_ID_mvt'].equals(table['MVT_ID_mvt'])
    assert np.any(values != old['TAXITIME_SEC_mvt'].to_numpy()), 'Duplicate candidate'
    sources = dict(parent['sources'])
    # Bind immutable artifacts and each ancestor's identity/source records.
    # Live upload manifests stay mutable; v76 is bound through its snapshot.
    paths = {Path(__file__), component.MANIFEST, component.OUT / 'audit.json',
        component.OUT / 'result.json', component.OUT / 'predictions.parquet',
        component.OUT / 'candidate.parquet', ROOT / 'data/submitting.parquet'}
    for module in [component, component.parent, component.base, component.base.aggregate,
                   component.main, component.cv]:
        paths.add(Path(module.__file__))
        for name in ['identity.json', 'audit.json', 'result.json']:
            path = module.OUT / name
            if path.exists(): paths.add(path)
        identity = read(module.OUT / 'identity.json')
        for record in identity.get('sources', []): paths.add(verify(record))
        for record in identity.get('files', []): paths.add(verify(record))
    sources.update({str(path.relative_to(ROOT)): sha256(path) for path in paths})
    destination.mkdir(parents=True)
    write_json(destination / 'parent_v76_snapshot.json', parent)
    shutil.copyfile(component.OUT / 'candidate.parquet', output)
    shutil.copyfile(component.OUT / 'cv.parquet', destination / 'cv.parquet')
    assert sha256(output) == sha256(component.OUT / 'candidate.parquet')
    sources.update({str(path.relative_to(ROOT)): sha256(path) for path in
        [destination / 'parent_v76_snapshot.json', destination / 'cv.parquet']})
    write_json(destination / 'manifest.json', dict(version=version, state='prepared', parent='v76',
        change='Selected airport/main/CatBoost inputs and daily/monthly aggregate refinements',
        file=str(output.relative_to(ROOT)), sha256=sha256(output), rows=344841,
        jan_jul_cv=result['jan_jul_cv'], cv_months=result['cv_months'],
        cv_record=art(component.cv.OUT / 'result.json'), component=art(component.OUT / 'result.json'),
        sources=sources, authorization='User authorized autonomous leaderboard uploads',
        changed_rows=int(np.sum(values != old['TAXITIME_SEC_mvt'].to_numpy())),
        prediction_rms_change=float(np.sqrt(np.mean((values.astype(float)-old['TAXITIME_SEC_mvt'].to_numpy())**2))),
        limitations=['Extensively reused Jan/Jul development months, no independent confirmation',
                    'Production CatBoost and daily main trained on laptop; preserved model receipts',
                    'No leaderboard score or projected gain yet'],
        audit=dict(component_all_row_checks_pass=True, byte_identical_component=True,
            schema_ids_exact=True, independent_package_audit_pending=True)))
    print(dict(version=version, state='prepared; package audit pending', cv=result['jan_jul_cv']), flush=True)


def audit(version):
    destination = OUT / f'submission_v{version}'
    path = destination / 'manifest.json'
    value = read(path)
    assert value['state'] == 'prepared' and not (destination / 'audit.json').exists()
    for name, digest in value['sources'].items(): assert sha256(ROOT / name) == digest, name
    assert sha256(ROOT / value['file']) == value['sha256'] == sha256(component.OUT / 'candidate.parquet')
    with duckdb.connect(config={'threads': 1}) as con:
        for name, source in [('s', ROOT / value['file']), ('p', component.OUT / 'predictions.parquet'),
            ('v', destination / 'cv.parquet'), ('r', component.cv.OUT / 'scored.parquet')]:
            con.read_parquet(str(source)).create_view(name)
        n, u, bad = con.execute('''select count(*),count(distinct s.MVT_ID_mvt),
            sum(cast(s.TAXITIME_SEC_mvt is distinct from p.candidate_pred as int))
            from s join p on s.MVT_ID_mvt=p.mvt_id''').fetchone()
        assert n == u == 344841 and bad == 0
        ncv, ucv, bad = con.execute('''select count(*),count(distinct v.mvt_id),sum(case when
            v.y is distinct from r.y or v.month is distinct from r.month or v.day is distinct from r.day
            or v.pred is distinct from r.combined or v.control is distinct from r.full_cat then 1 else 0 end)
            from v join r using(mvt_id)''').fetchone()
        assert ncv == ucv == 344419 and bad == 0
        scores = dict(con.execute('select month,sqrt(avg(pow(pred-y,2))) from v group by month').fetchall())
        for month, score in scores.items(): np.testing.assert_allclose(score, value['cv_months'][month], rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.mean(list(scores.values())), value['jan_jul_cv'], rtol=0, atol=1e-8)
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    table = pq.read_table(ROOT / value['file'])
    assert table.schema == template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    report = dict(status='pass', source=art(__file__), file=art(ROOT / value['file']),
        component_audit=art(component.OUT / 'audit.json'), rows=n, cv_rows=ncv,
        all_row_sql_predictions_truth_calendars_and_cv_pass=True,
        template_schema_and_ids_exact=True, byte_identical_component=True)
    write_json(destination / 'audit.json', report)
    value['audit']['independent_package_audit_pending'] = False
    value['audit']['independent_package_audit'] = art(destination / 'audit.json')
    value['sources'][str((destination / 'audit.json').relative_to(ROOT))] = sha256(destination / 'audit.json')
    write_json(path, value)
    print(dict(version=version, audit='pass', rows=n, jan_jul_cv=value['jan_jul_cv']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'audit'])
    parser.add_argument('--version', required=True, type=int)
    args = parser.parse_args()
    with threadpool_limits(1): globals()[args.stage](args.version)
