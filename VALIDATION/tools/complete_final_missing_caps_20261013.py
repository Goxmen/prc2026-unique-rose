"""Fixed four-model recovery batch, only after explicit root desktop allocation.

    .venv/bin/python VALIDATION/tools/complete_final_missing_caps_20261013.py

Frozen versions5/6/7/9, one missing fit per version and exact old-vector gates.
Stop the whole batch on mismatch or failure. Never restart a completed fit.
The 25-minute bound is checked before each next version, preserving any active
model/inference stage. No upload, neural expansion or coefficient solve.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
from records import runtime_identity

RUN = ROOT/'VALIDATION/runs/v3_20261012_final_missing_early_reconstruct'
OUT = RUN/'completion_caps_20261013'
VERSIONS = [5, 6, 7, 9]


def read(path):
    return json.loads(path.read_text())


def art(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def verify(plan):
    for source in plan['sources']:
        assert source==art(ROOT/source['path']),source['path']
    assert runtime_identity()==read(RUN/'identity.json')['runtime']


def main():
    plan_path=OUT/'plan.json';plan=read(plan_path);verify(plan)
    assert plan['versions']==VERSIONS and plan['maximum_minutes_before_next_version']==25
    allocation=read(OUT/'allocation.json')
    assert allocation['authority']=='/root' and allocation['state']=='released_to_fixed_caps'
    assert allocation['desktop_exclusive'] is True and allocation['versions']==VERSIONS
    assert allocation['plan_sha256']==art(plan_path)['sha256']
    v8_audit=read(RUN/'v8/independent_audit.json')
    v8_result=read(RUN/'v8/result.json')
    v8_resume=read(RUN/'resume_v8_20261013/execution.json')
    assert v8_audit['status']=='pass' and v8_audit['result']==art(RUN/'v8/result.json')
    assert v8_result['views']['old']['exact_original_integer_replay']
    assert v8_resume['state']=='complete'
    assert [(x['stage'],x['evidence']) for x in v8_resume['steps']]==[
        ('final_inference',art(RUN/'v8/result.json')),
        ('independent_audit',art(RUN/'v8/independent_audit.json'))]
    target=OUT/'execution.json'
    assert not target.exists(),'No automatic restart of this batch'
    assert all(not (RUN/f'v{v}').exists() for v in VERSIONS),'Inspect any partial prior version first'
    state=dict(state='running',pid=os.getpid(),started_at=now(),plan=art(plan_path),
        allocation=art(OUT/'allocation.json'),runtime=runtime_identity(),steps=[],completed_versions=[],
        no_upload=True,no_neural_fit=True,no_solve=True)
    save(target,state);started=time.monotonic()
    try:
        for version in VERSIONS:
            if time.monotonic()-started>25*60:
                state.update(state='bounded_stop_before_next_version',next_version=version,
                    finished_at=now(),active_stage=None,child_pid=None)
                save(target,state);return
            directory=RUN/f'v{version}'
            expected=read(RUN/'identity.json')['expected'][str(version)]
            for stage in ['prepare','fit','infer','audit']:
                verify(plan)
                assert art(plan_path)==state['plan'] and art(OUT/'allocation.json')==state['allocation']
                state.update(active_version=version,active_stage=stage);save(target,state)
                command=plan['commands'][str(version)][stage]
                with (OUT/f'v{version}_{stage}.log').open('x') as log:
                    child=subprocess.Popen([str(ROOT/'.venv/bin/python'),*command],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT)
                    state['child_pid']=child.pid;save(target,state);code=child.wait()
                assert code==0,f'v{version} {stage} exited {code}; preserve completed stages'
                if stage=='prepare':
                    item=read(directory/'preflight.json')
                    assert item['identity']==art(RUN/'identity.json') and item['rows']==expected['rows']
                    assert item['features']==expected['features'] and len(item['features'])==27
                    assert item['source_exact'] is False and item['history_helper_hash_exact']
                    assert item['original_metadata_feature_order_config_and_count_exact']
                    assert item['independent_SQL_raw_membership_labels_basic18_and_fpl_exact']
                    assert item['raw_order_matches_retained_200000_cap_control']
                    assert item['training']==art(directory/'training.pkl')
                    evidence=art(directory/'preflight.json')
                elif stage=='fit':
                    item=read(directory/'trained.json')
                    assert item['runtime']==state['runtime'] and item['identity']==art(RUN/'identity.json')
                    assert item['model']==art(directory/'model.txt') and item['rounds']==800
                    assert item['saved_before_inference']
                    evidence=art(directory/'trained.json')
                elif stage=='infer':
                    item=read(directory/'result.json');evidence=art(directory/'result.json')
                    if item['state']=='closed_nonexact_reconstruction':
                        state.update(state='closed_nonexact_stop_batch',failed_version=version,result=evidence,
                            finished_at=now(),active_stage=None,child_pid=None,no_sweep=True)
                        save(target,state);print(json.dumps(dict(state=state['state'],version=version)),flush=True);return
                    assert item['state']=='exact_old_replay_final_inference_complete'
                    assert item['views']['old']['exact_original_integer_replay']
                    assert item['views']['old']['rows']==344841 and item['views']['final']['rows']==670790
                else:
                    item=read(directory/'independent_audit.json')
                    assert item['status']=='pass' and item['prediction_rows_checked']==1015631
                    evidence=art(directory/'independent_audit.json')
                state['steps'].append(dict(version=version,stage=stage,evidence=evidence,finished_at=now()))
                save(target,state)
            state['completed_versions'].append(version);save(target,state)
            print(json.dumps(dict(completed_version=version,no_next_after=VERSIONS[-1])),flush=True)
        state.update(state='complete',finished_at=now(),active_version=None,active_stage=None,child_pid=None)
        save(target,state);print(json.dumps(dict(state='complete',receipt=art(target))),flush=True)
    except BaseException as exc:
        state.update(state='failed',failure=repr(exc),traceback=traceback.format_exc(),finished_at=now())
        save(target,state);raise


if __name__=='__main__':
    main()
