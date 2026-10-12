"""Finite completion worker: wait, return, build and audit; never fit or upload."""
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'VALIDATION/runs'
OUT=RUNS/'v3_20261012_tabm_pool3_completion'
PRODUCTION=RUNS/'v3_20261012_nn_public_tabm_pool3_submit'
STACK=RUNS/'v3_20261012_nn_public_tabm_pool3_submit_stack'
COMBO=RUNS/'v3_20261012_tabm_pool3_other_runway_combo_submit'
PY=ROOT/'.venv/bin/python'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(value):
    p=OUT/'execution.json';tmp=p.with_suffix('.tmp')
    value['updated_at']=datetime.now(timezone.utc).isoformat()
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def desktop_busy():
    needles=[b'VALIDATION/nn_public_tabm_duration.py',b'VALIDATION/nn_public_tabm_duration_score.py']
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=path.read_bytes().split(b'\0')
        except (OSError,ProcessLookupError):continue
        if len(parts)>1 and any(parts[1]==v for v in needles):return True
    return False


def remote_complete():
    code="from pathlib import Path; p=Path('/home/gunes/PRC/VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit/export.json'); print(int(p.exists()))"
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','glaptop-wsl',
        '/home/gunes/PRC/.venv/bin/python -'],input=code,text=True,capture_output=True,timeout=20,check=True)
    return r.stdout.strip()=='1'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'execution.json').exists(),'Inspect prior completion worker; do not duplicate'
    steps=[
      ('return','VALIDATION/tools/return_tabm_pool3_submit_20261012.py',[],PRODUCTION/'return_audit.json'),
      ('stack_freeze','VALIDATION/nn_public_tabm_pool3_submit_stack.py',['freeze'],STACK/'identity.json'),
      ('stack_build','VALIDATION/nn_public_tabm_pool3_submit_stack.py',['build'],STACK/'result.json'),
      ('stack_audit','VALIDATION/nn_public_tabm_pool3_submit_stack.py',['audit'],STACK/'audit.json'),
      ('combo_prepare','VALIDATION/tabm_pool3_other_runway_combo_submit.py',['prepare'],COMBO/'identity.json'),
      ('combo_build','VALIDATION/tabm_pool3_other_runway_combo_submit.py',['build'],COMBO/'result.json'),
      ('combo_audit','VALIDATION/tools/audit_tabm_pool3_runway_combination_20261012.py',[],COMBO/'independent_audit.json')]
    paths={ROOT/script for _,script,_,_ in steps}|{Path(__file__)}
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    state=dict(state='waiting',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),sources=hashes,
        deadline_hours=4,no_fit=True,no_upload=True,steps=[])
    write(state);deadline=time.monotonic()+4*3600
    try:
        while True:
            assert time.monotonic()<deadline,'Completion deadline reached; inspect jobs'
            if not desktop_busy() and remote_complete():break
            time.sleep(30)
        for name,script,args,output in steps:
            for path,digest in hashes.items():assert sha(ROOT/path)==digest,'Pinned worker source changed'
            assert not desktop_busy(),'Desktop training/scoring restarted; preserve one heavy job'
            assert not output.exists(),'Unexpected completed stage; inspect before resuming'
            state.update(state='running',active_stage=name);write(state)
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (OUT/f'{name}.log').open('w') as log:
                subprocess.run([str(PY),script,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert output.exists(),f'Missing receipt for{name}'
            state['steps'].append(dict(stage=name,receipt=str(output.relative_to(ROOT)),sha256=sha(output)))
            write(state)
        state.update(state='complete',active_stage=None,candidate=str((COMBO/'candidate.parquet').relative_to(ROOT)))
        write(state);print(json.dumps(dict(state='complete',candidate=state['candidate'],no_upload=True)),flush=True)
    except BaseException as error:
        state.update(state='failed',failure=repr(error));write(state);raise


if __name__=='__main__':main()
