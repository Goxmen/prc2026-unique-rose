"""Allocated direct serialization then OLD public-chain inference/audit only.

Sequential, fail closed, preserve all completed outputs. No final public-chain
posterior, assignment, upload, repeated fit or repeated numerical stage.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DIRECT=ROOT/'VALIDATION/runs/v3_20261013_portable_direct_reserve'
CHAIN=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
OUT=DIRECT/'completion'


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    allocation=read(DIRECT/'allocation.json')
    assert allocation['authority']=='/root' and allocation['identity']==art(DIRECT/'identity.json')
    assert allocation['completion_source']==art(__file__) and allocation['public_identity']==art(CHAIN/'identity.json')
    for record in allocation['frozen_sources']:assert art(ROOT/record['path'])==record
    OUT.mkdir(exist_ok=True);execution=OUT/'execution.json';assert not execution.exists()
    assert not (DIRECT/'result.json').exists() and not (CHAIN/'old/result.json').exists()
    receipt=dict(status='running',pid=os.getpid(),allocation=art(DIRECT/'allocation.json'),source=art(__file__),
                 started_unix=time.time(),steps=[],no_upload=True,no_final_public_posterior=True)
    write(execution,receipt)
    commands=[
        ['VALIDATION/portable_direct_reserve_20261013.py','build'],
        ['VALIDATION/tools/audit_portable_direct_reserve_20261013.py'],
        ['VALIDATION/portable_public_source_20261013.py','infer','--view','old'],
        ['VALIDATION/tools/recover_portable_public_audit_joins_20261013.py','freeze','--view','old'],
        ['VALIDATION/tools/recover_portable_public_audit_joins_20261013.py','audit','--view','old'],
        ['VALIDATION/portable_direct_reserve_20261013.py','compare'],
    ]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for index,arguments in enumerate(commands):
        if index==2:
            audit=read(DIRECT/'audit.json')
            assert audit['status']=='pass' and audit['result']==art(DIRECT/'result.json')
            folder=CHAIN/'old';folder.mkdir(exist_ok=True)
            old_allocation=folder/'allocation.json';assert not old_allocation.exists()
            write(old_allocation,dict(authority='/root',identity=art(CHAIN/'identity.json'),view='old',
                state='released_for_portable_public_source',source=art(__file__),root_allocation=art(DIRECT/'allocation.json'),
                direct_audit=art(DIRECT/'audit.json'),scope='OLD posterior and additive SQL audit recovery only; no final view or upload'))
        if index==5:
            audit=read(CHAIN/'old/audit.json')
            assert audit['status']=='pass' and audit['result']==art(CHAIN/'old/result.json')
        command=[str(ROOT/'.venv/bin/python'),*arguments];log=OUT/f'{index+1:02d}.log'
        step=dict(command=command,started_unix=time.time());receipt['steps'].append(step);write(execution,receipt)
        with log.open('x') as stream:
            outcome=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        step.update(returncode=outcome.returncode,finished_unix=time.time(),log=art(log));write(execution,receipt)
        if outcome.returncode:
            receipt.update(status='failed_preserved_stop',finished_unix=time.time());write(execution,receipt)
            print(json.dumps(dict(status=receipt['status'],step=index+1,log=art(log))),flush=True)
            return outcome.returncode
    receipt.update(status='complete',finished_unix=time.time(),
                   outputs=[art(p) for p in [DIRECT/'result.json',DIRECT/'audit.json',DIRECT/'chain_comparison.json',
                                             CHAIN/'old/result.json',CHAIN/'old/audit.json']])
    write(execution,receipt)
    print(json.dumps(dict(status='complete',execution=art(execution))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
