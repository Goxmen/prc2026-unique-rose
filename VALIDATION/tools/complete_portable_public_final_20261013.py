"""Run the allocated FINAL public posterior and additive audit recovery once.

Requires prepared versioned portable parents, audited CV/OLD and frozen source
bindings. No fit, calibration, repeated view, version assignment or upload.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
FOLDER=OUT/'final/completion'


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):
    temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(v,indent=2)+'\n');temporary.replace(p)


def main():
    allocation=read(OUT/'final/allocation.json')
    assert allocation['authority']=='/root' and allocation['identity']==art(OUT/'identity.json')
    assert allocation['source']==art(__file__) and allocation['view']=='final'
    for record in allocation['frozen_sources']:assert art(ROOT/record['path'])==record
    for view in ['cv','old']:
        audit=read(OUT/view/'audit.json');assert audit['status']=='pass' and audit['result']==art(OUT/view/'result.json')
    assert not (OUT/'final/result.json').exists()
    FOLDER.mkdir(exist_ok=True);execution=FOLDER/'execution.json';assert not execution.exists()
    receipt=dict(status='running',pid=os.getpid(),source=art(__file__),allocation=art(OUT/'final/allocation.json'),
                 steps=[],started_unix=time.time(),no_upload=True,no_fit=True)
    write(execution,receipt)
    commands=[['VALIDATION/portable_public_source_20261013.py','infer','--view','final'],
              ['VALIDATION/tools/recover_portable_public_audit_joins_20261013.py','freeze','--view','final'],
              ['VALIDATION/tools/recover_portable_public_audit_joins_20261013.py','audit','--view','final']]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for index,args in enumerate(commands):
        command=[str(ROOT/'.venv/bin/python'),*args];log=FOLDER/f'{index+1:02d}.log'
        step=dict(command=command,started_unix=time.time());receipt['steps'].append(step);write(execution,receipt)
        with log.open('x') as stream:r=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        step.update(returncode=r.returncode,finished_unix=time.time(),log=art(log));write(execution,receipt)
        if r.returncode:
            receipt.update(status='failed_preserved_stop',finished_unix=time.time());write(execution,receipt)
            print(json.dumps(dict(status=receipt['status'],failed_step=index+1,log=art(log))),flush=True)
            return r.returncode
    audit=read(OUT/'final/audit.json');assert audit['status']=='pass' and audit['result']==art(OUT/'final/result.json')
    receipt.update(status='complete',finished_unix=time.time(),result=art(OUT/'final/result.json'),audit=art(OUT/'final/audit.json'))
    write(execution,receipt);print(json.dumps(dict(status='complete',execution=art(execution))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
