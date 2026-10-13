"""Durable single-attempt laptop stage launcher; no training logic or retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'
RUNNER = ROOT / 'VALIDATION/nn_public_tabm_track_consistency.py'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def artifact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(stage):
    assert platform.node() == 'GLAPTOP'
    plan = read(OUT / 'deployment_plan.json')
    assert plan['authority'] == '/root' and plan['launcher'] == artifact(__file__)
    assert plan['runner'] == artifact(RUNNER)
    assert plan['identity'] == artifact(OUT / 'identity.json')
    assert plan['comparison_identity'] == artifact(OUT / 'comparison/identity.json')
    identity = read(OUT / 'identity.json')
    assert identity['source'] == artifact(RUNNER)
    if stage == 'fit':
        allocation = read(OUT / 'allocation.json')
        assert allocation['authority'] == '/root' and allocation['action'] == 'fit_fixed_track_consistency'
        assert allocation['identity'] == plan['identity']
        assert allocation['prefit_audit'] == artifact(OUT / 'prefit_audit.json')
    folder = OUT / 'runtime' / stage
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'execution.json'
    state = dict(state='starting', stage=stage, host=platform.node(), pid=os.getpid(),
                 process_start=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19],
                 started_at=stamp(), plan=artifact(OUT / 'deployment_plan.json'))
    with path.open('x') as stream:
        json.dump(state, stream, indent=2)
    command = [str(ROOT / '.venv/bin/python'), '-u', str(RUNNER), stage]
    process = None
    try:
        with (folder / 'worker.log').open('xb') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            state.update(state='running', child_pid=process.pid,
                         child_process_start=process_start(process.pid), command=command)
            save(path, state)
            result = process.wait()
        state['returncode'] = result
        assert result == 0, f'{stage} failed with exit code {result}; preserve logs and artifacts'
        receipt = OUT / ('preflight.json' if stage == 'preflight' else 'audit.json')
        completed = read(receipt)
        assert completed['status'] == 'pass' and completed['identity'] == plan['identity']
        state.update(state='complete', receipt=artifact(receipt), finished_at=stamp())
        save(path, state)
    except BaseException as error:
        state.update(state='error', error=str(error), traceback=traceback.format_exc(), finished_at=stamp())
        if process is not None:
            state.update(child_still_running=process.poll() is None,
                         child_current_process_start=process_start(process.pid))
        save(path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'fit'])
    run(parser.parse_args().stage)
