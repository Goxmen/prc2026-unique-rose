"""Run the one absent v56 classifier after explicit desktop allocation release.

    .venv/bin/python VALIDATION/tools/complete_final_missing_flightnumber_20261012.py

The frozen plan and parent allocation receipt are mandatory. This worker never
restarts an existing execution or refits a saved checkpoint automatically.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_flightnumber_reconstruct'
OUT = RUN / 'completion'


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
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def verify(plan):
    for item in plan['sources']:
        assert art(ROOT / item['path']) == item, item['path']


def main():
    plan_path = OUT / 'plan.json'
    plan = read(plan_path)
    verify(plan)
    allocation = read(OUT / 'allocation.json')
    assert allocation['state'] == 'released_to_v56'
    assert allocation['plan_sha256'] == sha(plan_path)
    assert allocation['authority'] == '/root' and allocation['desktop_exclusive']
    target = OUT / 'execution.json'
    assert not target.exists(), 'Keep prior worker receipt; no automatic restart'
    assert not (RUN / 'model.txt').exists() and not (RUN / 'trained.json').exists()
    preflight = read(RUN / 'preflight.json')
    assert preflight['identity'] == art(RUN / 'identity.json')
    assert preflight['rows'] == 160577 and preflight['positives'] == 33205
    assert preflight['independent_SQL_leave_month_out_rate_and_counts_exact']
    assert preflight['original_ordered_LIRF_membership_and_raw_gaps_targets_exact']
    for key in ['training', 'history']:
        assert art(ROOT / preflight[key]['path']) == preflight[key]
    state = dict(state='running', pid=os.getpid(), started_at=now(), steps=[],
                 plan_sha256=sha(plan_path), allocation=art(OUT / 'allocation.json'), no_upload=True)
    save(target, state)
    try:
        for stage in ['fit', 'infer', 'audit']:
            verify(plan)
            assert sha(plan_path) == state['plan_sha256']
            state.update(active_stage=stage)
            save(target, state)
            started = now()
            with (OUT / f'{stage}.log').open('x') as log:
                child = subprocess.Popen([str(ROOT / '.venv/bin/python'), *plan['commands'][stage]],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save(target, state)
                code = child.wait()
            assert code == 0, f'{stage} exited {code}; preserve every checkpoint and log'
            if stage == 'fit':
                trained = read(RUN / 'trained.json')
                assert trained['identity'] == art(RUN / 'identity.json')
                assert trained['model'] == art(RUN / 'model.txt')
                assert trained['saved_before_inference'] and trained['trees'] == 1700
                evidence = art(RUN / 'trained.json')
            elif stage == 'infer':
                result = read(RUN / 'result.json')
                evidence = art(RUN / 'result.json')
                if result['state'] == 'closed_approximation_old_replay_failed':
                    state.update(state='closed_approximation', active_stage=None, child_pid=None,
                                 finished_at=now(), result=evidence, no_sweep=True)
                    save(target, state)
                    print(json.dumps(dict(state=state['state'], result=evidence)), flush=True)
                    return
                assert result['state'] == 'exact_old_replay_final_inference_complete'
                assert result['views']['old']['exact_original_integer_replay']
                assert result['views']['old']['rows'] == 344841
                assert result['views']['final']['rows'] == 670790
            else:
                assert read(RUN / 'independent_audit.json')['status'] == 'pass'
                evidence = art(RUN / 'independent_audit.json')
            state['steps'].append(dict(stage=stage, started_at=started, completed_at=now(), evidence=evidence))
            save(target, state)
        state.update(state='complete', active_stage=None, child_pid=None, finished_at=now())
        save(target, state)
        print(json.dumps(dict(state='complete', receipt=str(target.relative_to(ROOT)))), flush=True)
    except BaseException as exc:
        state.update(state='failed', finished_at=now(), failure=repr(exc), traceback=traceback.format_exc())
        save(target, state)
        raise


if __name__ == '__main__':
    main()
