"""Sequential, resumable execution of `final_submit.py` up to the prepared v63 file (no upload)."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VALIDATION/runs/v3_20261008_final_submit'
JOBS = [('tables_old', ['tables', '--set', 'old']), ('tables_final', ['tables', '--set', 'final']),
        ('inputs_old', ['inputs', '--set', 'old']), ('inputs_final', ['inputs', '--set', 'final']),
        ('trees', ['trees']), ('rome', ['rome']), ('nn', ['nn']), ('build63', ['build', '--version', '63'])]

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / 'execution.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(completed={})
    for name, args in JOBS:
        if name in status['completed']:
            continue
        status.update(current=name, state='running', exit_code=None)
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        started = time.monotonic()
        with (OUT / f'{name}.log').open('a') as log:
            result = subprocess.run([str(ROOT / '.venv/bin/python'), '-u', 'VALIDATION/final_submit.py', *args],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed', exit_code=result.returncode)
            status_path.write_text(json.dumps(status, indent=2) + '\n')
            raise SystemExit(result.returncode)
        status['completed'][name] = dict(minutes=(time.monotonic() - started) / 60)
        status_path.write_text(json.dumps(status, indent=2) + '\n')
    status.update(current=None, state='complete', exit_code=0)
    status_path.write_text(json.dumps(status, indent=2) + '\n')
