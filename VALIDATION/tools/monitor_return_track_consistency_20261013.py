"""Durable read-only monitor/return of the six already allocated laptop fits.

Polls only the stage receipt/process identity, never epochs or predictions. No
remote source writes, model inference, fit, score, held-label access or restart.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'
FOLDER=OUT/'fit_return'
IDENTITY_SHA='9dd40491e2d5d12f505a7311cc7852a803bf3cc2d3b22ce63cfd0bae395a8a56'
ARMS=['supervised_control','training_consistency','target_consistency']
MONTHS=['2025-01','2025-07']


def stamp():return datetime.now(timezone.utc).isoformat()


def read(p):return json.loads(Path(p).read_text())


def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest(),bytes=p.stat().st_size)


def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def remote(source,payload):
    cp=subprocess.run(['ssh','-o','ConnectTimeout=20','glaptop-wsl','cd ~/PRC && .venv/bin/python -'],
        input='PAYLOAD='+repr(payload)+'\n'+source,text=True,capture_output=True,timeout=120)
    if cp.returncode:raise RuntimeError(dict(code=cp.returncode,stdout=cp.stdout,stderr=cp.stderr))
    return json.loads(cp.stdout)


STATUS='''from pathlib import Path
import hashlib,json
r=Path.cwd();out=r/PAYLOAD['out']
def art(p):
 p=Path(p);return dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def proc(pid):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 a=p.read_text().split(') ',1)[1].split();return dict(start=a[19],state=a[0])
for a in PAYLOAD['pins']:assert art(r/a['path'])==a,a['path']
s=json.loads((out/'runtime/fit/execution.json').read_text())
assert s['pid']==3410 and s['process_start']=='780770'
assert s['plan']=={k:v for k,v in PAYLOAD['plan'].items() if k!='bytes'}
print(json.dumps(dict(execution=s,parent_process=proc(s['pid']),child_process=proc(s.get('child_pid',-1)))))
'''

COMPLETE='''from pathlib import Path
import hashlib,json
r=Path.cwd();out=r/PAYLOAD['out'];seen={}
def read(p):return json.loads(Path(p).read_text())
def art(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return dict(path=str(p.relative_to(r)),sha256=h.hexdigest(),bytes=p.stat().st_size)
def verify(a):
 p=r/a['path']
 if a['path'] not in seen:seen[a['path']]=art(p)
 actual=seen[a['path']]
 assert all(actual[k]==v for k,v in a.items() if k in ['path','sha256','bytes']),a['path']
 return p
for a in PAYLOAD['pins']:verify(a)
identity=read(out/'identity.json');identity_art={k:v for k,v in art(out/'identity.json').items() if k!='bytes'}
fit=read(out/'audit.json');run=read(out/'runtime/fit/execution.json')
assert run['state']=='complete' and run['returncode']==0 and run['receipt']=={k:v for k,v in art(out/'audit.json').items() if k!='bytes'}
assert fit['status']=='pass' and fit['identity']==identity_art
assert fit['all_held_labels_nan'] and fit['all_held_reload_exact'] and len(fit['members'])==6
for key in ['preflight','prefit_audit','allocation']:
 assert fit[key]=={k:v for k,v in art(out/(key+'.json')).items() if k!='bytes'};verify(fit[key])
allocation=read(out/'allocation.json');assert allocation['authority']=='/root' and allocation['action']=='fit_fixed_track_consistency' and allocation['identity']==identity_art
files={str((out/'audit.json').relative_to(r))};members=[]
for month in PAYLOAD['months']:
 for arm in PAYLOAD['arms']:
  rp=out/arm/'networks'/(month+'_s42.json');a={k:v for k,v in art(rp).items() if k!='bytes'}
  assert a in fit['members'];record=read(rp)
  assert record['identity']==identity_art and record['design']==identity['design']
  assert record['month']==month and record['arm']==arm and record['seed']==42 and record['epochs']==2
  assert record['inputs']==identity['folds'][month] and record['all_held_reloaded_exact']
  assert record['scope']['adapter_targets_all_nan'] and record['scope']['held_labels_absent_from_cache']
  assert {k:record[k] for k in ['host','gpu','torch']}==identity['execution']
  for key in ['model','prediction','preprocessing','checkpoint','teacher']:verify(record[key])
  trained=read(verify(record['checkpoint']))
  assert trained['identity']==identity_art and trained['arm']==arm and trained['month']==month
  assert trained['model']==record['model'] and trained['teacher']==record['teacher'] and trained['updates_completed_before_inference']
  for key in ['preflight','prefit_audit','allocation']:assert trained[key]==fit[key];verify(trained[key])
  assert record['teacher']=={k:v for k,v in art(out/'teacher'/month/'receipt.json').items() if k!='bytes'}
  files.add(str(rp.relative_to(r)))
  files.update(record[k]['path'] for k in ['model','prediction','checkpoint'])
  started=out/arm/'started'/(month+'.json');s=read(started)
  assert s['identity']==identity_art and s['month']==month and s['arm']==arm and s['allocation']==fit['allocation']
  files.add(str(started.relative_to(r)))
  members.append(a)
files.update(str(p.relative_to(r)) for p in (out/'runtime/fit').iterdir() if p.is_file())
sources=[art(r/a['path']) for a in identity['files'] if a['path'].endswith('.py')]
for a in sources:
 original=next(z for z in identity['files'] if z['path']==a['path']);verify(original)
print(json.dumps(dict(files=[art(r/p) for p in sorted(files)],sources=sources,members=members,execution=identity['execution'],pins=PAYLOAD['pins'])))
'''


def main():
    identity=art(OUT/'identity.json');assert identity['sha256']==IDENTITY_SHA
    rel=str(OUT.relative_to(ROOT))
    pins=[art(OUT/p) for p in ['identity.json','comparison/identity.json','allocation.json','deployment_plan.json',
        'preflight.json','prefit_audit.json']]
    assert read(OUT/'allocation.json')['authority']=='/root'
    plan=art(OUT/'deployment_plan.json')
    FOLDER.mkdir(parents=True,exist_ok=True);execution=FOLDER/'execution.json'
    state=dict(state='monitoring',source=art(__file__),started_at=stamp(),pid=os.getpid(),
        process_start=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        identity=identity,pins=pins,poll_seconds=60,no_fit_no_inference_no_score=True)
    with execution.open('x') as f:json.dump(state,f,indent=2)
    payload=dict(out=rel,pins=pins,plan=plan,arms=ARMS,months=MONTHS)
    try:
        while True:
            status=remote(STATUS,payload);state.update(last_checked=stamp(),remote=status);save(execution,state)
            s=status['execution']
            if s['state']=='complete':break
            assert s['state'] in ['starting','running'],status
            parent=status['parent_process'];child=status['child_process']
            assert parent and parent['start']==s['process_start'] and parent['state']!='Z',status
            if s['state']=='running':assert child and child['start']==s['child_process_start'] and child['state']!='Z',status
            time.sleep(60)
        manifest=remote(COMPLETE,payload)
        for a in manifest['sources']:assert art(ROOT/a['path'])==a,a['path']
        files=[]
        for a in manifest['files']:
            p=ROOT/a['path']
            if p.exists():assert art(p)==a,'Refuse differing local artifact: '+a['path']
            else:files.append(a['path'])
        mp=FOLDER/'remote_manifest.json';assert not mp.exists();mp.write_text(json.dumps(manifest,indent=2)+'\n')
        fp=FOLDER/'files.txt';fp.write_text('\n'.join(files)+'\n')
        state.update(state='returning',return_files=len(files),remote_manifest=art(mp));save(execution,state)
        with (FOLDER/'rsync.log').open('xb') as log:
            subprocess.run(['rsync','-a','--files-from='+str(fp),'glaptop-wsl:PRC/',str(ROOT)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
        for a in manifest['files']:assert art(ROOT/a['path'])==a,a['path']
        for a in pins:assert art(ROOT/a['path'])==a,a['path']
        fit=read(OUT/'audit.json');assert fit['members']==manifest['members'] and fit['status']=='pass'
        result=dict(status='pass',source=art(__file__),identity={k:v for k,v in identity.items() if k!='bytes'},
            execution=manifest['execution'],fit_audit=art(OUT/'audit.json'),allocation=art(OUT/'allocation.json'),
            files=manifest['files'],members=manifest['members'],sources=manifest['sources'],
            remote_manifest=art(mp),returned_files=len(files),returned_bytes=sum(a['bytes'] for a in manifest['files'] if a['path'] in files),
            all_six_fits_complete_before_return=True,teacher_caches_reused_without_transfer=True,
            remote_original_sources_and_pins_exact=True,no_fit_inference_score_or_held_label_access=True)
        rp=OUT/'return_audit.json'
        with rp.open('x') as f:json.dump(result,f,indent=2)
        state.update(state='complete',finished_at=stamp(),receipt=art(rp));save(execution,state)
        print(json.dumps(dict(state='complete',receipt=art(rp),files=len(files),bytes=result['returned_bytes'])),flush=True)
    except BaseException as error:
        state.update(state='error',error=str(error),traceback=traceback.format_exc(),finished_at=stamp());save(execution,state)
        raise


if __name__=='__main__':main()
