"""Synthetic final-upload validator checks; no network, upload or real arrays."""
import importlib.util
import json
from datetime import date
from pathlib import Path
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
        except AssertionError:tests.append('Reject '+name)
        else:raise AssertionError(name)
    for name,values in [('negative target',[-1,12,999]),('null target',[None,12,999]),('different audited source',[0,13,999])]:
        changed=candidate.set_column(1,candidate.schema.field(1),pa.array(values,type=pa.int32()))
        reject(name,lambda changed=changed:module.check_tables(template,changed,candidate,3))
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
    result=dict(status='pass',source=module.artifact(Path(__file__)),validator=module.artifact(SOURCE),
        policy=module.artifact(module.submission_slot_policy.POLICY),checks=tests,no_network=True,no_real_arrays=True,no_upload=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='pass',checks=len(tests),receipt=module.artifact(OUT))))


if __name__=='__main__':main()
