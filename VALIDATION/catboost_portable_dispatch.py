"""Supervise the laptop fit and return its checkpoint before resuming the idle queue."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'VALIDATION/runs/v3_20261011_catboost_full_inputs_submit/portable_worker'
OUT = WORK.parent
REMOTE_STATE = WORK / 'remote_state.json'
MIGRATION = WORK / 'migration.json'
COLLECT = WORK / 'collection.json'
LOG = 'tmp/research_20261011/catboost_full_inputs_laptop.log'


def read(path):
    return json.loads(Path(path).read_text())


def start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def worker():
    assert platform.node() == 'GLAPTOP'
    assert not REMOTE_STATE.exists(), 'Inspect existing remote job; never launch twice'
    status = dict(state='starting', pid=os.getpid(), process_start=start(os.getpid()),
        host=platform.node(), started_at=stamp(), manifest_sha256=sha256(WORK / 'manifest.json'))
    write_json(REMOTE_STATE, status)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='2')
    with (ROOT / LOG).open('ab', buffering=0) as log:
        child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u',
            'VALIDATION/catboost_full_inputs_portable.py', 'fit'], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        status.update(state='running', child_pid=child.pid, child_start=start(child.pid))
        write_json(REMOTE_STATE, status)
        code = child.wait()
    status.update(state='complete' if code == 0 else 'error', exit_code=code, ended_at=stamp())
    if code == 0:
        status['model_record_sha256'] = sha256(OUT / 'model.json')
    write_json(REMOTE_STATE, status)


def collect():
    assert platform.node() != 'GLAPTOP'
    migration = read(MIGRATION)
    pid = migration['desktop_queue_pid']
    assert start(pid) == migration['desktop_queue_start']
    assert Path(f'/proc/{pid}/status').read_text().split('State:', 1)[1].splitlines()[0].strip().startswith('T')
    status = dict(state='waiting_for_laptop', pid=os.getpid(), started_at=stamp(),
        migration_sha256=sha256(MIGRATION))
    write_json(COLLECT, status)
    relative = str(REMOTE_STATE.relative_to(ROOT))
    while True:
        try:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', 'glaptop-wsl',
                f'cd ~/PRC && cat {relative}'], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            status.update(last_timeout=stamp())
            write_json(COLLECT, status)
            time.sleep(30)
            continue
        if result.returncode == 0:
            remote = json.loads(result.stdout)
            assert remote['manifest_sha256'] == sha256(WORK / 'manifest.json')
            status.update(remote=remote, checked_at=stamp())
            write_json(COLLECT, status)
            if remote['state'] == 'error':
                status.update(state='error', error='Laptop fit failed; desktop duplicate remains stopped')
                write_json(COLLECT, status)
                return
            if remote['state'] == 'complete':
                break
        time.sleep(30)
    paths = [OUT / 'model.json', OUT / 'models/catboost.cbm', WORK / 'input_audit.json', REMOTE_STATE, ROOT / LOG]
    assert not (OUT / 'model.json').exists() and not (OUT / 'models/catboost.cbm').exists()
    listing = WORK / 'return_files.txt'
    listing.write_text(''.join(str(p.relative_to(ROOT)) + '\n' for p in paths))
    subprocess.run(['rsync', '-a', '--files-from=' + str(listing), 'glaptop-wsl:PRC/', str(ROOT) + '/'], check=True)
    record = read(OUT / 'model.json')
    assert sha256(OUT / 'model.json') == remote['model_record_sha256']
    assert record['identity']['sha256'] == sha256(OUT / 'identity.json')
    assert record['portable_worker']['sha256'] == sha256(WORK / 'manifest.json')
    assert record['host'] == 'GLAPTOP' and record['rounds'] == 10000 and record['train_rows'] == 2051327
    assert sha256(ROOT / record['model']['path']) == record['model']['sha256']
    assert sha256(ROOT / record['portable_input_audit']['path']) == record['portable_input_audit']['sha256']
    assert read(WORK / 'input_audit.json')['status'] == 'pass'
    assert start(pid) == migration['desktop_queue_start']
    status.update(state='import_verified', model_sha256=record['model']['sha256'], finished_at=stamp())
    write_json(COLLECT, status)
    os.kill(pid, signal.SIGCONT)
    status.update(state='complete', desktop_queue_resumed=True)
    write_json(COLLECT, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['worker', 'collect'])
    globals()[parser.parse_args().stage]()
