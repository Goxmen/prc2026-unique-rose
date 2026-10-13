"""Run the one allocated portable public-source CV, score and independent audit.

No old/final expansion, fit, coefficient solve, restart or upload. Every command
and exit is persisted; failed posterior stages retain their own checkpoints.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261013_portable_public_source'
FOLDER = OUT / 'cv/completion'
PYTHON = ROOT / '.venv/bin/python'


def read(path):
    return json.loads(path.read_text())


def artifact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    receipt_path = FOLDER / 'execution.json'
    assert not receipt_path.exists(), 'Preserve prior execution; no automatic restart'
    allocation = read(OUT / 'cv/allocation.json')
    identity = artifact(OUT / 'identity.json')
    assert allocation['identity'] == identity
    assert allocation['authority'] == '/root' and allocation['view'] == 'cv'
    assert allocation['state'] == 'released_for_portable_public_source'
    assert allocation['completion_source'] == artifact(__file__)
    for path in [OUT / 'cv/result.json', OUT / 'score.json', OUT / 'cv/audit.json']:
        assert not path.exists(), path
    commands = [
        ['VALIDATION/portable_public_source_20261013.py', 'infer', '--view', 'cv'],
        ['VALIDATION/portable_public_source_20261013.py', 'score'],
        ['VALIDATION/tools/audit_portable_public_source_20261013.py', '--view', 'cv'],
    ]
    receipt = dict(status='running', pid=os.getpid(), identity=identity,
                   source=artifact(__file__), allocation=artifact(OUT / 'cv/allocation.json'),
                   started_unix=time.time(), steps=[], no_old_final_expansion=True, no_upload=True)
    write(receipt_path, receipt)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for index, arguments in enumerate(commands):
        log = FOLDER / f'{index + 1:02d}.log'
        command = [str(PYTHON), *arguments]
        step = dict(command=command, started_unix=time.time(), log=str(log.relative_to(ROOT)))
        receipt['steps'].append(step)
        write(receipt_path, receipt)
        with log.open('x') as output:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        step.update(returncode=completed.returncode, finished_unix=time.time(), log_artifact=artifact(log))
        if completed.returncode:
            receipt.update(status='failed_preserved_stop', finished_unix=time.time())
            write(receipt_path, receipt)
            print(json.dumps(dict(status=receipt['status'], failed_step=index + 1, log=step['log'])), flush=True)
            return completed.returncode
        write(receipt_path, receipt)
    checked = read(OUT / 'cv/audit.json')
    assert checked['status'] == 'pass' and checked['identity'] == identity
    assert checked['result'] == artifact(OUT / 'cv/result.json')
    receipt.update(status='complete', finished_unix=time.time(),
                   outputs=[artifact(OUT / name) for name in ['cv/result.json', 'score.json', 'cv/audit.json']])
    write(receipt_path, receipt)
    print(json.dumps(dict(status='complete', receipt=artifact(receipt_path))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
