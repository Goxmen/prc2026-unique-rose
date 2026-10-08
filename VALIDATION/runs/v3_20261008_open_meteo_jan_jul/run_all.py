"""Sequential, resumable desktop execution of the user-requested all-part comparison."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LOGS = ROOT/'tmp/external_sources_20261008'
JOBS = [('audit', ['VALIDATION/tools/audit_open_meteo.py'])] + [
    (p, ['VALIDATION/open_meteo_all.py',p]) for p in ['nn','rome','airport','catboost','score']]


def main():
    status_path = OUT/'execution.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(completed={})
    for name,args in JOBS:
        if name in status['completed']: continue
        started = time.monotonic()
        status.update(current=name,state='running',exit_code=None)
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        print(f'Starting {name}',flush=True)
        with (LOGS/f'all_{name}.log').open('a') as log:
            result = subprocess.run([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed',exit_code=result.returncode)
            status_path.write_text(json.dumps(status,indent=2)+'\n')
            raise SystemExit(result.returncode)
        status['completed'][name] = dict(minutes=(time.monotonic()-started)/60)
        status_path.write_text(json.dumps(status,indent=2)+'\n')
    status.update(current=None,state='complete',exit_code=0)
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    print('All-part comparison complete',flush=True)


if __name__=='__main__': main()
