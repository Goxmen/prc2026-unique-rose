"""Run only the root-allocated fixed v8 recovery attempt with frozen runtime guards.

    .venv/bin/python VALIDATION/tools/complete_final_missing_v8_20261012.py

Exactly prepare -> fit -> original-vector gate/final inference -> SQL audit.
Original v8 modeling.py is absent; exact old-vector replay is required.
No automatic restart, second version, tuning, upload or failed-replay fallback.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from records import runtime_identity

RUN = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_early_reconstruct'
OUT = RUN / 'completion_v8'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def verify(plan):
    for record in plan['sources']:
        assert record == art(ROOT / record['path']), record['path']
    assert runtime_identity() == read(RUN / 'identity.json')['runtime'], 'Frozen runtime changed'


def main():
    plan_path = OUT / 'plan.json'
    plan = read(plan_path)
    verify(plan)
    allocation = read(OUT / 'allocation.json')
    assert allocation['state'] == 'released_to_v8' and allocation['authority'] == '/root'
    assert allocation['desktop_exclusive'] and allocation['plan_sha256'] == sha(plan_path)
    assert allocation['version'] == plan['version'] == 8
    target = OUT / 'execution.json'
    assert not target.exists(), 'Preserve prior execution; no automatic restart'
    directory = RUN / 'v8'
    assert not directory.exists(), 'Preserve interrupted preparation or any existing fit'
    state = dict(state='running', pid=os.getpid(), started_at=now(), steps=[],
                 version=8, plan_sha256=sha(plan_path), allocation=art(OUT / 'allocation.json'),
                 runtime=runtime_identity(), no_expansion=True, no_upload=True)
    save(target, state)
    try:
        for stage in ['prepare', 'fit', 'infer', 'audit']:
            verify(plan)
            assert sha(plan_path) == state['plan_sha256']
            assert art(OUT / 'allocation.json') == state['allocation']
            state.update(active_stage=stage)
            save(target, state)
            started = now()
            with (OUT / f'{stage}.log').open('x') as log:
                child = subprocess.Popen([str(ROOT / '.venv/bin/python'), *plan['commands'][stage]],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save(target, state)
                code = child.wait()
            assert code == 0, f'{stage} exited {code}; preserve all completed states'
            if stage == 'prepare':
                record = read(directory / 'preflight.json')
                assert record['identity'] == art(RUN / 'identity.json')
                assert record['rows'] == 2084607 and len(record['features']) == 27
                assert record['source_exact'] is False and record['history_helper_hash_exact']
                assert record['original_metadata_feature_order_config_and_count_exact']
                assert record['independent_SQL_raw_membership_labels_basic18_and_fpl_exact']
                assert record['raw_order_matches_retained_200000_cap_control']
                assert record['training'] == art(directory / 'training.pkl')
                evidence = art(directory / 'preflight.json')
            elif stage == 'fit':
                record = read(directory / 'trained.json')
                assert record['runtime'] == state['runtime'] == read(RUN / 'identity.json')['runtime']
                assert record['identity'] == art(RUN / 'identity.json')
                assert record['rounds'] == 800 and record['saved_before_inference']
                assert record['model'] == art(directory / 'model.txt')
                evidence = art(directory / 'trained.json')
            elif stage == 'infer':
                record = read(directory / 'result.json')
                evidence = art(directory / 'result.json')
                if record['state'] == 'closed_nonexact_reconstruction':
                    state.update(state='closed_nonexact', active_stage=None, child_pid=None,
                                 finished_at=now(), result=evidence, no_sweep=True)
                    save(target, state)
                    print(json.dumps(dict(state=state['state'], result=evidence)), flush=True)
                    return
                assert record['state'] == 'exact_old_replay_final_inference_complete'
                assert record['views']['old']['exact_original_integer_replay']
                assert record['views']['old']['rows'] == 344841 and record['views']['final']['rows'] == 670790
            else:
                record = read(directory / 'independent_audit.json')
                assert record['status'] == 'pass' and record['prediction_rows_checked'] == 1015631
                evidence = art(directory / 'independent_audit.json')
            state['steps'].append(dict(stage=stage, started_at=started, finished_at=now(), evidence=evidence))
            save(target, state)
        state.update(state='complete', active_stage=None, child_pid=None, finished_at=now())
        save(target, state)
        print(json.dumps(dict(state='complete', version=8, receipt=str(target.relative_to(ROOT)))), flush=True)
    except BaseException as exc:
        state.update(state='failed', failure=repr(exc), traceback=traceback.format_exc(), finished_at=now())
        save(target, state)
        raise


if __name__ == '__main__':
    main()
