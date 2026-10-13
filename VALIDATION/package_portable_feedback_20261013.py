"""Prepare exactly v81/v82 for root review; read-only remote checks, no upload.

Authenticate local copies of actual bucket submissions by remote metadata SHA
or single-part ETag, then compare canonical prediction vectors. Both candidates
regress v80 CV; their purpose is choosing a portable final fallback by feedback.
"""
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import dashboard_data as live
import submission_slot_policy as slots

RUNS=ROOT/'VALIDATION/runs'
OUT=RUNS/'v3_20261013_portable_feedback_candidates'
DIRECT=RUNS/'v3_20261013_portable_direct_reserve'
CHAIN=RUNS/'v3_20261013_portable_public_source'
PLAN=ROOT/'VALIDATION/reports/portable_final_feedback_plan_20261013.json'
SNAPSHOT=OUT/'live_before_packaging.json'
AUTHORIZATION='User permits autonomous leaderboard submissions; root authorizes preparation only here. At most two exploratory portable parents, with final daily slot reserved; uploads remain a separate root action.'


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)==record,path;return path
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def canonical(path,ids):
    table=pq.read_table(path,columns=['MVT_ID_mvt','TAXITIME_SEC_mvt']).to_pandas()
    # The actual failed v63 upload contains the four-month population. Project
    # every historical file onto all old template IDs, preserving this scope.
    assert len(table)>=len(ids)==344841 and table.MVT_ID_mvt.is_unique
    values=table.set_index('MVT_ID_mvt').loc[ids,'TAXITIME_SEC_mvt'].to_numpy(dtype='<f8')
    assert np.isfinite(values).all()
    return hashlib.sha256(values.tobytes()).hexdigest()


def history_check(snapshot,ids):
    cache=OUT/'actual_submission_fingerprints.json';assert not cache.exists()
    paths=subprocess.check_output(['rg','--files','--no-ignore','-g','unique-rose_v*.parquet',
        '-g','!.venv/**','-g','!node_modules/**','.'],cwd=ROOT,text=True).splitlines()
    by={}
    for value in paths:
        path=(ROOT/value).resolve();by.setdefault(path.name,[]).append(path)
    records=[];stats=OUT/'remote_stats';stats.mkdir(exist_ok=True)
    for entry in snapshot['history']:
        name=entry['file'];assert name.endswith('.parquet') and '/' not in name
        assert name in by,'Actual remote prediction absent locally: '+name
        stat_path=stats/(name+'.json')
        if stat_path.exists():
            remote=read(stat_path)
        else:
            proc=subprocess.run([str(Path.home()/'.local/bin/mc'),'stat','--json',live.BUCKET+name],
                                env=live.mc_environment(),capture_output=True,check=True)
            exact=[json.loads(line) for line in proc.stdout.decode().splitlines()
                   if json.loads(line).get('name')==name]
            assert len(exact)==1;remote=exact[0];write(stat_path,remote)
        assert remote['name']==name and remote['status']=='success' and remote['size']==entry['sizeBytes']
        metadata={str(k).lower():v for k,v in remote.get('metadata',{}).items()}
        remote_sha=metadata.get('x-amz-meta-sha256',metadata.get('sha256'))
        etag=remote['etag'].strip('"');found=[]
        for path in sorted(by[name]):
            if path.stat().st_size!=remote['size']:continue
            raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();md5=hashlib.md5(raw).hexdigest()
            if (remote_sha and sha==remote_sha) or ('-' not in etag and md5==etag):
                found.append((path,sha,'metadata_sha256' if remote_sha and sha==remote_sha else 'single_part_etag_md5'))
        assert found,'No locally authenticated actual submission: '+name
        path,sha,method=found[0]
        records.append(dict(file=name,local=dict(path=str(path.relative_to(ROOT)),sha256=sha),
            remote_stat=art(stat_path),authentication=method,canonical_prediction_sha256=canonical(path,ids),
            source_rows=pq.ParquetFile(path).metadata.num_rows,canonical_scope='All344841 old template IDs in template order'))
    write(cache,dict(snapshot=art(SNAPSHOT),records=records,actual_remote_files_only=True,no_upload=True))
    return records,art(cache)


def main():
    assert not (OUT/'prepared.json').exists()
    plan=read(PLAN);assert plan['maximum_exploratory_uploads']==2 and plan['last_daily_slot_reserved']
    assert plan['candidates']==['latest_direct_public_model_reserve','portable_public_source_v1']
    snapshot=read(SNAPSHOT);assert snapshot['snapshotSource']=='live'
    now=datetime.now(timezone.utc);then=datetime.fromisoformat(snapshot['updatedAt'])
    assert 0<=(now-then).total_seconds()<3600,'Refresh preflight snapshot; never imply stale quota is fresh'
    policy=read(slots.POLICY);assert policy['utc_date']==now.date().isoformat()
    assert policy['daily_limit']==5 and policy['reserved_slots']==1
    used=[r['file'] for r in snapshot['history'] if r.get('uploadedAt') and
          datetime.fromisoformat(r['uploadedAt'].replace('Z','+00:00')).astimezone(timezone.utc).date()==now.date()]
    assert len(used)+2<=policy['daily_limit']-policy['reserved_slots']
    occupied={r['file'] for r in snapshot['history']}
    assert all(f'unique-rose_v{v}.parquet' not in occupied for v in [81,82])
    template_path=ROOT/'data/submitting.parquet';template=pq.read_table(template_path)
    assert template.num_rows==344841;ids=template['MVT_ID_mvt'].to_numpy()
    records,duplicate_receipt=history_check(snapshot,ids)
    seen={r['canonical_prediction_sha256'] for r in records}
    dr,da=read(DIRECT/'result.json'),read(DIRECT/'audit.json')
    cr,ca=read(CHAIN/'old/result.json'),read(CHAIN/'old/audit.json')
    cv,cv_audit=read(CHAIN/'score.json'),read(CHAIN/'cv/audit.json')
    assert da['status']==ca['status']==cv_audit['status']=='pass'
    assert da['result']==art(DIRECT/'result.json') and ca['result']==art(CHAIN/'old/result.json')
    assert da['identity']==dr['identity']==art(DIRECT/'identity.json')
    assert ca['identity']==cr['identity']==cv['identity']==art(CHAIN/'identity.json')
    assert cv_audit['metrics']['score']==art(CHAIN/'score.json')
    assert ca['candidate']==cr['candidate'] and da['views']['old']['candidate']==dr['views']['old']['candidate']
    assert da['views']['final']['candidate']==dr['views']['final']['candidate']
    paths={Path(__file__),PLAN,SNAPSHOT,verify(duplicate_receipt),template_path,slots.POLICY,Path(slots.__file__),
        ROOT/'VALIDATION/audited_candidate_upload.py',ROOT/'VALIDATION/scored_blend_submit.py',
        ROOT/'VALIDATION/records.py',Path(live.__file__),DIRECT/'ready_for_root_review.json',
        DIRECT/'result.json',DIRECT/'audit.json',DIRECT/'identity.json',DIRECT/'chain_comparison.json',
        CHAIN/'old/result.json',CHAIN/'old/audit.json',CHAIN/'identity.json',CHAIN/'cv/audit.json',CHAIN/'score.json',
        CHAIN/'old/audit_sql_join_recovery/identity.json',CHAIN/'old/audit_sql_join_recovery/numerical_audit.json',
        ROOT/'VALIDATION/tools/recover_portable_public_audit_joins_20261013.py'}
    for folder in [DIRECT,CHAIN]:
        for record in read(folder/'identity.json')['sources']:paths.add(verify(record))
    for record in records:paths.update([verify(record['local']),verify(record['remote_stat'])])
    for record in plan['sources']:paths.add(verify(record))
    for result in [dr['views']['old'],dr['views']['final'],cr]:
        paths.add(verify(result['candidate']));paths.add(verify(result['prediction']))
    sources={}
    for path in sorted(paths):
        record=art(path);sources[record['path']]=record['sha256']
    manifests=[]
    for version,name,record,metric,month_metrics,audit_path in [
        (81,'latest_direct_public_model_reserve',dr['views']['old'],dr['cv'],dr['months'],DIRECT/'audit.json'),
        (82,'portable_public_source_v1',cr,cv['cv']['submitted'],{m:x['submitted'] for m,x in cv['months'].items()},CHAIN/'old/audit.json')]:
        folder=OUT/f'submission_v{version}';output=ROOT/'submissions'/f'unique-rose_v{version}.parquet'
        assert not folder.exists() and not output.exists();folder.mkdir()
        source=verify(record['candidate']);shutil.copyfile(source,output)
        assert art(output)['sha256']==record['candidate']['sha256']
        table=pq.read_table(output);assert table.schema==template.schema and table.num_rows==344841
        for column in template.column_names:
            if column!='TAXITIME_SEC_mvt':assert table[column].equals(template[column])
        target=table['TAXITIME_SEC_mvt'].to_numpy()
        assert table['TAXITIME_SEC_mvt'].null_count==0 and np.isfinite(target).all() and (target>=0).all()
        np.testing.assert_array_equal(target,np.rint(target))
        digest=canonical(output,ids);assert digest not in seen,'Duplicate actual submitted/candidate prediction';seen.add(digest)
        delta=metric-cv['cv']['v80'];assert delta>0
        manifest=dict(version=version,state='prepared',name=name,file=str(output.relative_to(ROOT)),
            sha256=art(output)['sha256'],rows=344841,jan_jul_cv=metric,cv_months=month_metrics,
            cv_change_vs_v80=delta,cv_interpretation='Regresses exact v80 on reused Jan/Jul; no improvement claimed.',
            purpose=plan['purpose'],feedback_plan=art(PLAN),maximum_exploratory_uploads=2,
            reserved_daily_slot=True,source_candidate=record['candidate'],source_prediction=record['prediction'],
            audit=art(audit_path),cv_audit=art(CHAIN/'cv/audit.json'),cv_record=art(CHAIN/'score.json'),
            cv_metric_key='direct_reserve' if version==81 else 'submitted',
            sources=sources,authorization=AUTHORIZATION,basis_scores={'unique-rose_v80.parquet':218.8525},
            prepared_at=now.isoformat(),template=art(template_path),
            duplicate_check=dict(canonical_prediction_sha256=digest,actual_submission_receipt=duplicate_receipt,
                                 compared_actual_files=[r['file'] for r in records]),
            no_official_gain_claimed=True,upload_not_performed=True,
            final_capability=(dict(state='audited',candidate=dr['views']['final']['candidate'],audit=art(DIRECT/'audit.json'))
                              if version==81 else dict(state='fixed_route_prepared_but_final_posterior_not_run')))
        receipt=slots.check_slot(manifest,used+[f'unique-rose_v{v}.parquet' for v in [81] if v<version],now.date())
        assert receipt and receipt['reserved_slots']==1
        write(folder/'packaging_audit.json',dict(status='pass',source=art(__file__),file=art(output),
            source_candidate=record['candidate'],all_template_columns_exact=True,finite_integer_nonnegative=True,
            duplicate_check=manifest['duplicate_check'],slot_policy=receipt,uploads_today=used,
            parent_audit=art(audit_path),no_upload=True))
        manifest['packaging_audit']=art(folder/'packaging_audit.json')
        manifest['sources'][manifest['packaging_audit']['path']]=manifest['packaging_audit']['sha256']
        write(folder/'manifest.json',manifest);manifests.append(art(folder/'manifest.json'))
    write(OUT/'prepared.json',dict(status='prepared_for_root_review',manifests=manifests,plan=art(PLAN),
        snapshot=art(SNAPSHOT),actual_submissions=len(records),used_today=used,maximum_exploratory=2,
        reserved_slots=1,no_upload=True,no_SUBMISSIONS_edit=True))
    print(json.dumps(read(OUT/'prepared.json')))


if __name__=='__main__':main()
