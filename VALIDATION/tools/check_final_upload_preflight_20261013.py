"""Synthetic final-upload validator checks; no network, upload or real arrays."""
import importlib.util
import json
from datetime import date
from pathlib import Path
import tempfile
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'VALIDATION/final_candidate_upload_preflight_20261013.py'
OUT=ROOT/'VALIDATION/reports/final_upload_preflight_synthetic_20261013.json'


def main():
    assert not OUT.exists()
    spec=importlib.util.spec_from_file_location('final_upload_preflight',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    template=pa.table({'MVT_ID_mvt':pa.array([9,2,7],type=pa.int64()),
                       'TAXITIME_SEC_mvt':pa.array([None,None,None],type=pa.int32())})
    candidate=template.set_column(1,template.schema.field(1),pa.array([0,12,999],type=pa.int32()))
    assert module.check_tables(template,candidate,candidate,3)['maximum']==999
    tests=['Exact template IDs/schema, zero and positive integer values']
    def reject(name,fn):
        try:fn()
        except (AssertionError,KeyError):tests.append('Reject '+name)
        else:raise AssertionError(name)
    for name,values in [('negative target',[-1,12,999]),('null target',[None,12,999]),('different audited source',[0,13,999])]:
        changed=candidate.set_column(1,candidate.schema.field(1),pa.array(values,type=pa.int32()))
        source=changed if name!='different audited source' else candidate
        reject(name,lambda changed=changed,source=source:module.check_tables(template,changed,source,3))
    for name,indices in [('wrong order',[1,0,2]),('duplicate IDs',[0,0,2])]:
        changed=candidate.take(pa.array(indices))
        reject(name,lambda changed=changed:module.check_tables(template,changed,changed,3))
    m=dict(file=module.EXPECTED_FILE);today=date(2026,10,13)
    def uploads(n):return [dict(file=f'unique-rose_test{i}.parquet',uploadedAt='2026-10-13T06:00:00Z') for i in range(n)]
    assert len(module.check_live_slots(m,uploads(3),today)['uploads_today'])==3
    tests.append('Three current-day uploads permitted before reserved slot')
    reject('reserved fourth-used slot without selection',lambda:module.check_live_slots(m,uploads(4),today))
    reject('five-upload quota',lambda:module.check_live_slots(m,uploads(5),today))
    reject('occupied final filename',lambda:module.check_live_slots(m,[dict(file='unique-rose_final.parquet',uploadedAt='2026-10-12T06:00:00Z')],today))
    reject('duplicate live records',lambda:module.check_live_slots(m,uploads(1)*2,today))
    reject('missing upload time',lambda:module.check_live_slots(m,[dict(file='a.parquet')],today))
    reject('timezone-free time',lambda:module.check_live_slots(m,[dict(file='a.parquet',uploadedAt='2026-10-13T06:00:00')],today))
    assert module.check_live_slots(m,[dict(file='yesterday.parquet',uploadedAt='2026-10-13T01:00:00+02:00')],today)['uploads_today']==[]
    tests.append('Quota dates normalized to UTC')
    with tempfile.TemporaryDirectory(prefix='_final_upload_synthetic_',dir=ROOT/'VALIDATION/runs') as folder:
        folder=Path(folder);sources={}
        def save(name,data):
            path=folder/name;path.write_text(json.dumps(data)+'\n');record=module.artifact(path)
            sources[record['path']]=record['sha256'];return record
        parent_identity=save('parent_identity.json',{'test':'synthetic'})
        parent_result=save('parent_result.json',dict(identity=parent_identity))
        cv_audit=save('cv_audit.json',dict(status='pass',identity=parent_identity,result=parent_result,
            views={'cv':dict(rows=344419,months={'2025-01':300.,'2025-07':200.},cv=250.)}))
        portable=dict(sources=sources,jan_jul_cv=250.,jan_jul_cv_scope='complete_portable_recipe',jan_jul_cv_source=cv_audit)
        assert module.cv_provenance(portable,dict(identity=parent_identity))['jan_jul_cv']==250.
        tests.append('Complete portable CV bound to matching source identity and independent audited metrics')
        reject('misstated complete CV',lambda:module.cv_provenance(dict(portable,jan_jul_cv=249.),dict(identity=parent_identity)))
        transfer_identity=save('transfer_identity.json',dict(portable_result=parent_result))
        transfer_result=dict(identity=transfer_identity,qualifications={'not_exact_v80_final_replay':True},no_new_cv_score=True,coefficient=2)
        transfer=dict(sources=sources,jan_jul_cv=None,jan_jul_cv_scope='unavailable_for_conditional_shared_clock_transport',
            fallback_jan_jul_cv=250.,fallback_jan_jul_cv_source=cv_audit)
        assert module.cv_provenance(transfer,transfer_result)['score_is_fallback_only'] is True
        tests.append('Conditional transfer has null CV and separately authenticated fallback-only score')
        reject('fallback score misrepresented as complete transfer CV',lambda:module.cv_provenance(dict(transfer,jan_jul_cv=250.),transfer_result))
        reject('wrong transfer CV scope',lambda:module.cv_provenance(dict(transfer,jan_jul_cv_scope='complete_portable_recipe'),transfer_result))
        decision=dict(status='selected_best_available',utc_date=today.isoformat(),file=module.EXPECTED_FILE,sha256='fixture',
            compared_candidates=['fixture1','fixture2'],selection_reason='synthetic selection check',remaining_promising_experiments_resolved=True,
            evidence=[cv_audit])
        selection=save('selection.json',decision)
        final=dict(file=module.EXPECTED_FILE,sha256='fixture',sources=sources,reserved_slot_selection=selection)
        assert module.check_final_selection(final,today)==selection
        tests.append('Final selection independently required with zero used slots')
        reject('missing final selection despite unused slots',lambda:module.check_final_selection(dict(file=module.EXPECTED_FILE),today))
        reject('wrong final selection date',lambda:module.selection_metadata(final,decision,date(2026,10,14)))
        reject('wrong final selection hash',lambda:module.selection_metadata(dict(final,sha256='different'),decision,today))
        reject('unresolved promising experiments',lambda:module.selection_metadata(final,dict(decision,remaining_promising_experiments_resolved=False),today))
    result=dict(status='pass',source=module.artifact(Path(__file__)),validator=module.artifact(SOURCE),
        policy=module.artifact(module.submission_slot_policy.POLICY),checks=tests,no_network=True,no_real_arrays=True,no_upload=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='pass',checks=len(tests),receipt=module.artifact(OUT))))


if __name__=='__main__':main()
