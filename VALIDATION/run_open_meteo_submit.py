"""Sequential weather fits and preparation; v61 upload requires fresh approval."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'VALIDATION/runs/v3_20261008_open_meteo_submit'
JOBS=[('prepare',['prepare']),('rome',['rome']),('build60',['build','--version','60']),
      ('upload60',['upload','--version','60'])]+[(p,[p]) for p in ['nn','lgb','airport','catboost']]+[
      ('build61',['build','--version','61'])]

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    status_path=OUT/'execution.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else dict(completed={})
    for name,args in JOBS:
        if name in status['completed']: continue
        status.update(current=name,state='running',exit_code=None)
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        started=time.monotonic()
        with (OUT/f'{name}.log').open('a') as log:
            result=subprocess.run([str(ROOT/'.venv/bin/python'),'-u','VALIDATION/open_meteo_submit.py',*args],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed',exit_code=result.returncode)
            status_path.write_text(json.dumps(status,indent=2)+'\n')
            raise SystemExit(result.returncode)
        status['completed'][name]=dict(minutes=(time.monotonic()-started)/60)
        status_path.write_text(json.dumps(status,indent=2)+'\n')
    manifest = json.loads((OUT/'submission_v61/submission_v61.json').read_text())
    pending = (OUT/'uploads_paused.json').exists() and manifest['state'] == 'prepared'
    status.update(current=None,state='awaiting_upload_approval' if pending else 'complete',
                  pending_upload=61 if pending else None,exit_code=0)
    status_path.write_text(json.dumps(status,indent=2)+'\n')
