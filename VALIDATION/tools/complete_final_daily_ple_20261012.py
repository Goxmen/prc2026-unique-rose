"""Serial saved-model inference completion; no training or uploads."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_additional_public_components/completion'
DAILY=ROOT/'VALIDATION/runs/v3_20261012_final_daily_main_inference'
PLE=ROOT/'VALIDATION/runs/v3_20261012_final_public_ple_inference'
PLAN=OUT/'plan.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)

def main():
    plan=read(PLAN)
    for a in plan['sources']:assert sha(ROOT/a['path'])==a['sha256']
    state=dict(state='waiting_daily_main',steps=[],started_at=now(),plan_sha256=sha(PLAN))
    target=OUT/'execution.json';assert not target.exists();save(target,state)
    proc=Path('/proc')/str(plan['wait_pid'])
    if proc.exists():
        assert 'final_daily_main_inference.py' in (proc/'cmdline').read_bytes().decode().replace('\0',' ')
        initial=(proc/'stat').read_text().split()[21]
        while proc.exists():
            try:
                if (proc/'stat').read_text().split()[21]!=initial:break
            except FileNotFoundError:break
            time.sleep(5)
    try:
        assert read(DAILY/'final.json')['all_model_reload_predictions_exact']
        state['steps'].append(dict(stage='daily_main',completed_at=now(),receipt=str((DAILY/'final.json').relative_to(ROOT))))
        for label,script,args,receipt in [
            ('public_PLE','VALIDATION/final_public_ple_inference.py',['infer'],PLE/'final.json'),
            ('independent_SQL','VALIDATION/tools/audit_final_additional_public_20261012.py',[],OUT.parent/'audit.json')]:
            for a in plan['sources']:assert sha(ROOT/a['path'])==a['sha256']
            state.update(state='running',active_stage=label);save(target,state)
            if not receipt.exists():
                with (OUT/f'{label}.log').open('w') as log:
                    subprocess.run([str(ROOT/'.venv/bin/python'),script,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert receipt.exists()
            state['steps'].append(dict(stage=label,completed_at=now(),receipt=str(receipt.relative_to(ROOT))))
        state.update(state='complete',active_stage=None,finished_at=now());save(target,state)
    except BaseException as exc:
        state.update(state='failed',failure=repr(exc),finished_at=now());save(target,state);raise

if __name__=='__main__':main()
