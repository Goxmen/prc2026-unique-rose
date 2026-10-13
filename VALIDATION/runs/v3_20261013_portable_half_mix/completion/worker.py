"""Allocated fixed CV mixture screen followed by independent audit, no restarts."""
from pathlib import Path
import hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_half_mix'
FOLDER=OUT/'completion'
def art(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(v):(FOLDER/'execution.json').write_text(json.dumps(v,indent=2)+'\n')
allocation=json.loads((OUT/'allocation.json').read_text())
assert allocation['completion_source']==art(Path(__file__))
assert allocation['identity']==art(OUT/'identity.json') and allocation['authority']=='/root'
assert not (FOLDER/'execution.json').exists()
receipt=dict(status='running',pid=os.getpid(),allocation=art(OUT/'allocation.json'),source=art(Path(__file__)),steps=[],started_unix=time.time())
write(receipt)
commands=[['VALIDATION/portable_half_mix_20261013.py','screen'],['VALIDATION/tools/audit_portable_half_mix_20261013.py']]
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for k,args in enumerate(commands):
    log=FOLDER/f'{k+1:02d}.log';command=[str(ROOT/'.venv/bin/python'),*args]
    step=dict(command=command,started_unix=time.time());receipt['steps'].append(step);write(receipt)
    with log.open('x') as stream:r=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    step.update(returncode=r.returncode,finished_unix=time.time(),log=art(log));write(receipt)
    if r.returncode:
        receipt.update(status='failed_preserved_stop',finished_unix=time.time());write(receipt);raise SystemExit(r.returncode)
a=json.loads((OUT/'audit.json').read_text());assert a['status']=='pass' and a['result']==art(OUT/'result.json')
receipt.update(status='complete',finished_unix=time.time(),result=art(OUT/'result.json'),audit=art(OUT/'audit.json'));write(receipt)
print(json.dumps(receipt))
