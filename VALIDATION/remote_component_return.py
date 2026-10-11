"""Collect a hash-bound remote component, without running training or inference."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def checked_plan(path):
    plan = read(path)
    assert plan['source_sha256'] == sha256(__file__)
    assert plan['host'] == 'glaptop-wsl'
    assert sha256(ROOT / plan['identity']['path']) == plan['identity']['sha256']
    for name in plan['files'] + [plan['remote_queue'], plan['receipt'], plan['record']]:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
    return plan


def collect(path):
    plan = checked_plan(path)
    receipt = ROOT / plan['receipt']
    assert not receipt.exists(), 'Preserve existing collector; inspect before restarting'
    state = dict(state='waiting', pid=os.getpid(), process_start=process_start(os.getpid()),
        plan_sha256=sha256(path), started_at=stamp())
    write_json(receipt, state)
    code = ('import json; from pathlib import Path; '
        f'q=json.loads(Path({plan["remote_queue"]!r}).read_text()); '
        "p=Path('/proc/'+str(q['pid'])+'/stat'); "
        "s=p.read_text().split(') ',1)[1].split()[19] if p.exists() else None; "
        "print(json.dumps(dict(queue=q,process_start=s)))")
    try:
        while True:
            try:
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', plan['host'],
                    'cd ~/PRC && .venv/bin/python -c ' + shlex.quote(code)],
                    capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                time.sleep(30)
                continue
            if result.returncode != 0:
                time.sleep(30)
                continue
            remote = json.loads(result.stdout); q = remote['queue']
            assert q['pid'] == plan['remote_pid']
            state.update(remote=remote, checked_at=stamp())
            write_json(receipt, state)
            if q['state'] == 'complete':
                break
            assert q['state'] not in ['error', 'cancelled'], q
            assert remote['process_start'] == plan['remote_start'], 'Remote queue ended without completion'
            time.sleep(30)
        listing = receipt.with_suffix('.files.txt')
        assert all(not (ROOT / p).exists() for p in plan['files']), 'Inspect partial returned artifacts before retry'
        listing.write_text(''.join(p + '\n' for p in plan['files']))
        subprocess.run(['rsync', '-a', '--files-from=' + str(listing),
            plan['host'] + ':PRC/', str(ROOT) + '/'], check=True)
        record = read(ROOT / plan['record'])
        assert record['identity'] == plan['identity']
        for key in ['model', 'prediction']:
            assert sha256(ROOT / record[key]['path']) == record[key]['sha256']
        assert record['all_ranking_reload_exact'] is True and record['rows'] == 344841
        assert q['jobs'][-1]['completed_result_sha256'] == sha256(ROOT / plan['record'])
        model_record = read(ROOT / plan['model_record'])
        assert model_record['identity'] == plan['identity'] and model_record['model'] == record['model']
        assert model_record['host'] == 'GLAPTOP' and model_record['rounds'] == 10000
        assert sha256(ROOT / model_record['input_audit']['path']) == model_record['input_audit']['sha256']
        assert read(ROOT / model_record['input_audit']['path'])['status'] == 'pass'
        state.update(state='complete', finished_at=stamp(),
            files={p: sha256(ROOT / p) for p in plan['files']}, no_fit_or_inference=True)
        write_json(receipt, state)
    except Exception as exc:
        state.update(state='error', error=str(exc), updated_at=stamp())
        write_json(receipt, state)
        raise


def wait(path):
    plan = checked_plan(path)
    receipt = ROOT / plan['receipt']
    while True:
        state = read(receipt)
        assert state['plan_sha256'] == sha256(path)
        if state['state'] == 'complete':
            for filename, digest in state['files'].items():
                assert sha256(ROOT / filename) == digest
            return
        assert state['state'] != 'error', state
        assert process_start(state['pid']) == state['process_start'], 'Collector stopped without completion'
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['collect', 'wait'])
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    globals()[args.stage](args.plan.resolve())
