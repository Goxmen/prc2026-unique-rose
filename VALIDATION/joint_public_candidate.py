"""Package the audited fixed joint-source increment without fitting or uploading."""
import argparse
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import joint_public_moments_submit as component

ROOT=component.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_public_candidate'
art,read,write,sha=component.art,component.read,component.write,component.sha


def build(version):
    destination=OUT/f'submission_v{version}'
    output=ROOT/f'submissions/unique-rose_v{version}.parquet'
    assert version>77 and not destination.exists() and not output.exists()
    identity=component.guard()
    result=read(component.OUT/'result.json');audit=read(component.OUT/'audit.json')
    assert audit['status']=='pass' and audit['identity']==art(component.MANIFEST)
    assert audit['result']==art(component.OUT/'result.json')
    assert result['paired']['decision']=='improvement' and result['parent_and_unchanged_rows_exact']
    for record in result['artifacts']:component.verify(record)
    parent=read(component.OUT/'parent_v77_snapshot.json')
    assert parent['version']==77 and sha(ROOT/parent['file'])==parent['sha256']
    assert result['jan_jul_cv']<parent['jan_jul_cv']
    sources=dict(parent['sources'])
    for record in identity['sources']:sources[record['path']]=record['sha256']
    for path in [Path(__file__),component.MANIFEST,component.OUT/'result.json',
                 component.OUT/'audit.json',ROOT/'VALIDATION/tools/audit_joint_public_submit.py',
                 component.OUT/'inference.json',component.OUT/'inference.parquet',
                 component.OUT/'predictions.parquet',component.OUT/'candidate.parquet']:
        sources[str(path.relative_to(ROOT))]=sha(path)
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    table=pq.read_table(component.OUT/'candidate.parquet');template=pq.read_table(ROOT/'data/submitting.parquet')
    assert table.schema==template.schema and table.num_rows==344841
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt':assert table[name].equals(template[name])
    values=table['TAXITIME_SEC_mvt'].to_numpy()
    old=pq.read_table(ROOT/parent['file'])['TAXITIME_SEC_mvt'].to_numpy()
    assert np.isfinite(values).all() and (values>=0).all() and np.any(values!=old)
    destination.mkdir(parents=True)
    shutil.copyfile(component.OUT/'candidate.parquet',output)
    shutil.copyfile(component.OUT/'cv.parquet',destination/'cv.parquet')
    sources[str((destination/'cv.parquet').relative_to(ROOT))]=sha(destination/'cv.parquet')
    write(destination/'manifest.json',dict(version=version,state='prepared',parent='v77',
        change='v77 adds one joint public taxi-count, taxi-sum and departure-delay increment',
        file=str(output.relative_to(ROOT)),sha256=sha(output),rows=344841,
        jan_jul_cv=result['jan_jul_cv'],cv_months=result['cv_months'],paired=result['paired'],
        cv_record=art(component.cv.OUT/'result.json'),component=art(component.OUT/'result.json'),
        sources=sources,authorization='User authorized autonomous leaderboard uploads',
        changed_rows=int(np.sum(values!=old)),prediction_rms_change=result['rms_change'],
        limitations=['Extensively reused Jan/Jul development data, no independent confirmation',
                    'January worsens0.0340 seconds; July improves1.2732 seconds',
                    'Approximate peer moments and fixed half increment on an already public-informed parent',
                    'No official score or projected leaderboard gain yet'],
        audit=dict(component_all_row_checks_pass=True,byte_identical_component=True,
                   independent_package_audit_pending=True)))
    print(dict(version=version,state='prepared; package audit pending'),flush=True)


def audit(version):
    destination=OUT/f'submission_v{version}';path=destination/'manifest.json';value=read(path)
    assert value['state']=='prepared' and not (destination/'audit.json').exists()
    for source,digest in value['sources'].items():assert sha(ROOT/source)==digest,source
    assert sha(ROOT/value['file'])==value['sha256']==sha(component.OUT/'candidate.parquet')
    with duckdb.connect(config={'threads':1}) as con:
        for name,source in [('s',ROOT/value['file']),('p',component.OUT/'predictions.parquet'),
                           ('cv',destination/'cv.parquet'),('r',component.cv.OUT/'scored.parquet')]:
            con.read_parquet(str(source)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct s.MVT_ID_mvt),
            sum(cast(s.TAXITIME_SEC_mvt is distinct from p.pred as int))
            from s join p on s.MVT_ID_mvt=p.mvt_id''').fetchone()
        assert n==u==344841 and bad==0
        nc,uc,bad=con.execute('''select count(*),count(distinct cv.mvt_id),sum(case when
            cv.y is distinct from r.y or cv.month is distinct from r.month or cv.day is distinct from r.day
            or cv.pred is distinct from r.candidate or cv.control is distinct from r.control then 1 else 0 end)
            from cv join r using(mvt_id)''').fetchone()
        assert nc==uc==344419 and bad==0
        scores=dict(con.execute('select month,sqrt(avg(pow(pred-y,2))) from cv group by month').fetchall())
        for month,score in scores.items():np.testing.assert_allclose(score,value['cv_months'][month],rtol=0,atol=1e-8)
        np.testing.assert_allclose(np.mean(list(scores.values())),value['jan_jul_cv'],rtol=0,atol=1e-8)
    template=pq.read_table(ROOT/'data/submitting.parquet');table=pq.read_table(ROOT/value['file'])
    assert template.schema==table.schema
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt':assert table[name].equals(template[name])
    report=dict(status='pass',source=art(__file__),file=art(ROOT/value['file']),
        component_audit=art(component.OUT/'audit.json'),rows=n,cv_rows=nc,
        all_row_sql_predictions_truth_calendars_and_cv_pass=True,
        template_schema_and_non_target_columns_exact=True,byte_identical_component=True,
        reviewer='root-authored separate arithmetic and SQL audits; no independent subagent available')
    write(destination/'audit.json',report)
    value['sources'][str((destination/'audit.json').relative_to(ROOT))]=sha(destination/'audit.json')
    value['audit'].update(independent_package_audit_pending=False,
        independent_package_audit=art(destination/'audit.json'),
        independent_review=dict(status='pass',method='Separate root-authored component and package checks',
                                record=art(destination/'audit.json')))
    write(path,value)
    print(dict(version=version,audit='pass',rows=n,jan_jul_cv=value['jan_jul_cv']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['build','audit']);parser.add_argument('--version',type=int,required=True)
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage](args.version)
