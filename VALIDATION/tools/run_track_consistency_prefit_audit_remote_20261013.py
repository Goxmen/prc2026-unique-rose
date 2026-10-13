"""Single-attempt operational wrapper for the unchanged independent CPU audit.

Root authorizes this stage after the completed laptop GPU preflight. This wrapper
does not import the learner, run inference, train, modify frozen sources or retry.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,platform,subprocess,sys,traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'
AUDITOR=ROOT/'VALIDATION/tools/audit_track_consistency_prefit_20261013.py'
IDENTITY_SHA='9dd40491e2d5d12f505a7311cc7852a803bf3cc2d3b22ce63cfd0bae395a8a56'
COMPARISON_SHA='c6b5c366725c62ce656f52a7f556c6e7e9b5f8b352415c8bd8aabcb3810618fd'


def art(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):return json.loads(Path(p).read_text())


def start(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.read_text().split(') ',1)[1].split()[19] if p.exists() else None


def stamp():return datetime.now(timezone.utc).isoformat()


def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def main():
    assert platform.node()=='GLAPTOP'
    identity=art(OUT/'identity.json');assert identity['sha256']==IDENTITY_SHA
    assert art(OUT/'comparison/identity.json')['sha256']==COMPARISON_SHA
    assert read(OUT/'identity.json')['auditor']==art(AUDITOR)
    prior=read(OUT/'runtime/preflight/execution.json')
    assert prior['state']=='complete' and prior['returncode']==0
    assert prior['receipt']==art(OUT/'preflight.json')
    pref=read(OUT/'preflight.json');assert pref['status']=='pass' and pref['identity']==identity
    assert not (OUT/'prefit_audit.json').exists()
    folder=OUT/'runtime/independent_prefit';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'execution.json'
    state=dict(state='starting',stage='independent_prefit_audit',authority='/root',host=platform.node(),
        pid=os.getpid(),process_start=start(os.getpid()),started_at=stamp(),source=art(__file__),
        auditor=art(AUDITOR),identity=identity,preflight=art(OUT/'preflight.json'),
        preflight_execution=art(OUT/'runtime/preflight/execution.json'),no_fit_or_inference=True)
    with path.open('x') as f:json.dump(state,f,indent=2)
    child=None
    try:
        with (folder/'worker.log').open('xb') as log:
            command=[str(ROOT/'.venv/bin/python'),'-u',str(AUDITOR),'audit']
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(state='running',child_pid=child.pid,child_process_start=start(child.pid),command=command);save(path,state)
            code=child.wait();state['returncode']=code
        assert code==0,f'Independent audit failed ({code}); preserve outputs, never retry automatically'
        result=read(OUT/'prefit_audit.json')
        assert result['status']=='pass' and result['identity']==identity and result['source']==art(AUDITOR)
        assert result['preflight']==state['preflight'] and result['teachers']==pref['teachers']
        state.update(state='complete',finished_at=stamp(),receipt=art(OUT/'prefit_audit.json'));save(path,state)
        print(json.dumps(dict(state='complete',receipt=state['receipt'])),flush=True)
    except BaseException as error:
        state.update(state='error',error=str(error),traceback=traceback.format_exc(),finished_at=stamp())
        if child is not None:state.update(child_still_running=child.poll() is None,child_current_process_start=start(child.pid))
        save(path,state);raise


if __name__=='__main__':main()
