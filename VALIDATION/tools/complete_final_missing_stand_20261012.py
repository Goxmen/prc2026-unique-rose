"""Frozen serial missing-model recovery, gated on completed desktop inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_stand_reconstruct'
OUT = RUN / 'completion'
PLAN = OUT / 'plan.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def verify(plan):
    for item in plan['sources']:
        assert artifact(ROOT / item['path']) == item, item['path']


def process_alive(record):
    try:
        return (Path('/proc') / str(record['pid']) / 'stat').read_text().split()[21] == record['start_ticks']
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    plan = read(PLAN)
    verify(plan)
    receipt = OUT / 'execution.json'
    assert not receipt.exists(), 'Preserve any prior worker execution; no automatic restart'
    state = dict(state='waiting_upstream', pid=os.getpid(), started_at=now(),
                 plan_sha256=sha(PLAN), steps=[], no_upload=True)
    save(receipt, state)
    try:
        upstream = plan['upstream']
        while True:
            observed = read(ROOT / upstream['execution'])
            assert observed['plan_sha256'] == upstream['plan_sha256']
            phase = observed['state']
            assert phase in ['waiting_daily_main', 'running', 'complete'], observed
            if phase == 'complete':
                audited = read(ROOT / upstream['audit'])
                assert audited['status'] == 'pass'
                assert audited['old_controls_and_final_reloads_exact']
                assert audited['all_SQL_template_membership_and_domains_exact']
                if not process_alive(upstream['process']):
                    break
            else:
                assert process_alive(upstream['process']), 'Upstream exited without successful completion'
            time.sleep(5)
        state['upstream_evidence'] = [artifact(ROOT / upstream[key]) for key in ['execution', 'audit']]
        verify(plan)
        for stage in ['prepare', 'fit', 'infer', 'audit']:
            verify(plan)
            assert sha(PLAN) == state['plan_sha256']
            state.update(state='running', active_stage=stage)
            save(receipt, state)
            command = plan['commands'][stage]
            started = now()
            with (OUT / f'{stage}.log').open('x') as log:
                child = subprocess.Popen([str(ROOT / '.venv/bin/python'), *command], cwd=ROOT,
                                         stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save(receipt, state)
                code = child.wait()
            assert code == 0, f'{stage} exited {code}; preserve log and all checkpoints'
            if stage == 'prepare':
                preflight = read(RUN / 'preflight.json')
                identity = read(RUN / 'identity.json')
                assert preflight['identity'] == artifact(RUN / 'identity.json')
                assert preflight['rows'] == 2084659 and preflight['fits_started'] == 0
                for name in ['raw_SQL_targets_membership_and_original_ID_order_exact',
                             'basic18_inputs_and_classifier_membership_independent_SQL_exact',
                             'all_five_stand_features_independent_SQL_exact']:
                    assert preflight[name] is True, name
                assert preflight['model_config'] == identity['model_config']
                assert preflight['parameters'] == identity['parameters']
                assert preflight['dayshift_history'] == identity['dayshift_history']
                for key in ['training', 'categories']:
                    assert preflight[key] == artifact(ROOT / preflight[key]['path'])
                evidence = [artifact(RUN / 'preflight.json')]
            elif stage == 'fit':
                evidence = []
                for part in ['ordinary', 'classifier']:
                    trained = read(RUN / f'{part}_trained.json')
                    assert trained['saved_before_inference']
                    assert trained['model'] == artifact(RUN / f'{part}.txt')
                    evidence.append(artifact(RUN / f'{part}_trained.json'))
            elif stage == 'infer':
                result = read(RUN / 'result.json')
                evidence = [artifact(RUN / 'result.json')]
                if result['state'] == 'closed_approximation_old_replay_failed':
                    state.update(state='closed_approximation', active_stage=None,
                                 finished_at=now(), inference_result=evidence[0], no_sweep=True)
                    save(receipt, state)
                    print(json.dumps(dict(state=state['state'], receipt=str(receipt.relative_to(ROOT)))), flush=True)
                    return
                assert result['state'] == 'exact_old_replay_final_inference_complete'
                assert set(result['views']['old']) == {'30', '33', '34', '37'}
                for result_view in result['views']['old'].values():
                    assert result_view['exact_original_integer_replay'] and result_view['rows'] == 344841
            else:
                assert read(RUN / 'independent_audit.json')['status'] == 'pass'
                evidence = [artifact(RUN / 'independent_audit.json')]
            state['steps'].append(dict(stage=stage, started_at=started, completed_at=now(), evidence=evidence))
            save(receipt, state)
        state.update(state='complete', active_stage=None, child_pid=None, finished_at=now())
        save(receipt, state)
        print(json.dumps(dict(state='complete', receipt=str(receipt.relative_to(ROOT)))), flush=True)
    except BaseException as exc:
        state.update(state='failed', finished_at=now(), failure=repr(exc), traceback=traceback.format_exc())
        save(receipt, state)
        raise


if __name__ == '__main__':
    main()
