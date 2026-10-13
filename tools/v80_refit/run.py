"""User-run full v80 refit, audited final delivery and private replay packaging.

Default: desktop tree work, laptop neural work; one heavy worker on each host.
No training or network mutation happens for `plan`. `run` explicitly submits.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from refit_io import ROOT, DEFAULT, PYTHON, DEADLINE, announce, artifact, call_json, environment, now, read, relative, resolve, save, sha, verify

SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30']

def source_paths():
    names = subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines()
    names += [relative(p) for p in (ROOT/'tools/v80_refit').glob('*.py')]
    return sorted(set(names))

def worker_plan(stage, out, threads):
    return call_json([PYTHON, f'tools/v80_refit/{stage}.py','plan','--out',str(out),'--threads',str(threads)],
                     cwd=ROOT, env=environment(threads))

def plan(out, threads, desktop_only):
    stages = {s: worker_plan(s,out,threads) for s in ('trees','neural','assemble')}
    dependencies = sorted(set(p for stage in stages.values() for p in stage['required_files']))
    missing = [p for p in dependencies if not resolve(p).is_file()]
    return dict(schema='v80_manual_refit_v1', stages=stages, required_files=dependencies, missing=missing,
                threads=threads, out=relative(out), neural_host='desktop' if desktop_only else 'glaptop-wsl',
                tree_host='desktop', submission='replace unique-rose_final.parquet after complete audit',
                source_caveat=stages['trees'].get('source_caveat'),
                original_v80_score_reference_only=218.8525, new_cv_not_run=True)

def remote_python(code, payload=None, timeout=600):
    command='cd '+shlex.quote(str(ROOT))+' && .venv/bin/python -c '+shlex.quote(code)
    result=subprocess.run([*SSH,'glaptop-wsl',command],input=None if payload is None else json.dumps(payload),
                          text=True,capture_output=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('Laptop command failed; check SSH and laptop availability (no training started by this check)')
    return json.loads(result.stdout)

def bootstrap_health():
    # Independent of source deployment so we never overwrite a running worker's code.
    return remote_python("import json,pathlib,os,shutil; "
        "print(json.dumps({'host':os.uname().nodename,'disk_free':shutil.disk_usage('.').free,'available_kb':int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))}))")

def remote_worker_state(out):
    path=relative(out/'execution/neural.json')
    return remote_python("""import pathlib,json
p=pathlib.Path("""+repr(path)+""")
r=json.loads(p.read_text()) if p.is_file() else {'status':'absent'}
try: actual=pathlib.Path('/proc/'+str(r['pid'])+'/stat').read_text().rsplit(')',1)[1].split()[19]
except (KeyError,OSError,IndexError):actual=None
r['alive']=bool(actual and actual==r.get('process_start'))
print(json.dumps(r))
""")

def sync_laptop(paths, out):
    hashes={p:sha(resolve(p)) for p in paths}
    inspect_code="""import json,sys,pathlib,hashlib
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):x.update(b)
 return x.hexdigest()
print(json.dumps({s:h(pathlib.Path(s)) if pathlib.Path(s).is_file() else None for s in json.load(sys.stdin)}))
"""
    remote=remote_python(inspect_code,paths)
    changed=[p for p in paths if remote[p]!=hashes[p]]
    if changed:
        # Block a source replacement during an unrelated Python experiment.
        busy=remote_python("""import pathlib,json,os
r=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=p.read_bytes().split(b'\\0'); text=[os.fsdecode(v) for v in a]
  if 'python' in pathlib.Path(text[0]).name and any(v.endswith('.py') and ('VALIDATION/' in v or 'tools/final_selection/' in v or 'tools/v80_refit/neural.py' in v) for v in text[1:]):r.append(int(p.parent.name))
 except (OSError,IndexError):pass
print(json.dumps(r))
""")
        if busy:
            raise RuntimeError('Laptop has active training; refusing to replace its sources/inputs')
        listing=out/'execution/transfer.txt'; listing.parent.mkdir(parents=True,exist_ok=True)
        listing.write_text('\n'.join(changed)+'\n')
        announce(f'Laptop cache: transferring {len(changed)} missing/changed files ({sum(resolve(p).stat().st_size for p in changed)/1e9:.2f} GB).')
        backup=str(out/'sync_backup'/now().strftime('%Y%m%dT%H%M%S'))
        subprocess.run(['rsync','-a','--relative','--backup','--backup-dir',backup,'--files-from',str(listing),
                        '-e','ssh -o BatchMode=yes -o ConnectTimeout=15',str(ROOT)+'/',
                        'glaptop-wsl:'+str(ROOT)+'/'],check=True)
        after=remote_python(inspect_code,changed)
        if any(after[p]!=hashes[p] for p in changed):
            raise ValueError('Laptop transfer hash mismatch')
    save(out/'execution/laptop_inputs.json',dict(hashes=hashes,transferred=changed,verified_at=now().isoformat()))

def host_call(command, out, stage, threads, remote=False):
    args=['tools/v80_refit/host.py',command,'--out',relative(out),'--stage',stage,'--threads',str(threads)]
    if remote:
        shell='cd '+shlex.quote(str(ROOT))+' && .venv/bin/python '+shlex.join(args)
        return call_json([*SSH,'glaptop-wsl',shell],timeout=60)
    return call_json([PYTHON,*args],cwd=ROOT,timeout=60)

def wait_workers(workers, out, threads):
    finished=set()
    try:
        while len(finished)!=len(workers):
            for stage,remote in workers:
                if stage in finished:
                    continue
                state=host_call('status',out,stage,threads,remote)
                if state['status']=='finished':
                    if state.get('returncode')!=0:
                        raise RuntimeError(f'{stage} worker returned a failure')
                    announce(stage+' worker complete; artifacts retained.')
                    finished.add(stage)
                elif state['status']!='running' or not state.get('alive'):
                    raise RuntimeError(f'{stage} stopped ({state["status"]}); inspect {relative(out)}/execution/{stage}.log on its host. Completed checkpoints are retained.')
            if len(finished)!=len(workers):
                time.sleep(15)
    except BaseException:
        for stage,remote in workers:
            if stage not in finished:
                try:
                    host_call('stop',out,stage,threads,remote)
                except Exception:
                    announce(f'Could not stop {stage} remotely; its host lock prevents a duplicate.')
        raise

def collect_neural(out):
    receipt_rel=relative(out/'neural/result.json')
    result=remote_python('import json,pathlib;print(pathlib.Path('+repr(receipt_rel)+').read_text())')
    if result['status']!='pass':
        raise ValueError('Laptop result did not pass')
    records={r['path']:r for r in result['artifacts']}
    for record in result['predictions'].values():
        records[record['path']]=record
    missing=[p for p,r in records.items() if not resolve(p).is_file() or sha(resolve(p))!=r['sha256']]
    listing=out/'execution/return.txt'
    listing.write_text('\n'.join(missing+[receipt_rel])+'\n')
    subprocess.run(['rsync','-a','--relative','--backup','--backup-dir',str(out/'return_backup'),
                    '--files-from',str(listing),'-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
                    'glaptop-wsl:'+str(ROOT)+'/',str(ROOT)+'/'],check=True)
    for record in records.values():
        verify(record)
    if read(resolve(receipt_rel))!=result:
        raise ValueError('Laptop result changed during collection')
    save(out/'execution/collection.json',dict(status='pass',artifacts=len(records),result=artifact(resolve(receipt_rel))))

def freeze(out, schedule):
    sources=source_paths()
    inputs=schedule['required_files']
    frozen_schedule=copy.deepcopy(schedule)
    frozen_schedule['stages']['assemble'].pop('inputs_ready',None)
    identity=dict(schedule=frozen_schedule, sources=[artifact(resolve(p)) for p in sources],
                  inputs=[artifact(resolve(p)) for p in inputs])
    path=out/'run_identity.json'
    if path.exists() and read(path)!=identity:
        raise ValueError('Frozen run source/config/input identity changed; do not silently mix resumed models')
    if not path.exists():
        save(path,identity)
    return identity

def stage_run(module, action, out, threads):
    command=[PYTHON,'-u',f'tools/v80_refit/{module}.py',action,'--out',str(out)]
    if module=='assemble':
        command+=['--threads',str(threads)]
    folder=out/'execution'; folder.mkdir(parents=True,exist_ok=True)
    log=folder/f'{module}_{action}.log'
    with log.open('ab') as stream:
        result=subprocess.run(command,cwd=ROOT,env=environment(threads),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{module} {action} failed; see {relative(log)}. Completed stages remain saved.')

def run(out, threads, desktop_only):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            completed=read(out/'complete.json')
            verify(completed['acceptance']); verify(completed['final'])
            announce('This run is already complete; saved final and acceptance hashes verified. No refit or upload repeated.')
            return
        schedule=plan(out,threads,desktop_only)
        if schedule['missing']:
            raise FileNotFoundError('Missing retained dependencies: '+str(schedule['missing']))
        # A delivered run may need only post-deadline acceptance/publication repair.
        if now()>=DEADLINE and not (out/'delivery/attempt.json').exists():
            raise ValueError('Submission deadline has passed; no fits or upload started')
        freeze(out,schedule)
        if not (out/'audit.json').exists():
            import package
            package.verify_repository()  # Read-only auth/private-source check before long fits.
            from host import busy_processes
            other=busy_processes()
            from host import alive
            owned=[read(p) for p in [out/'execution/trees.json',out/'execution/neural.json'] if p.exists()]
            own_pids={s.get('child_pid') for s in owned if alive(s)}
            if any(p['pid'] not in own_pids for p in other):
                raise RuntimeError('Desktop already has a heavy model job; no second job started')
            if shutil.disk_usage(ROOT).free<15*(1<<30):
                raise RuntimeError('Need at least 15 GiB free desktop disk for retained refits and replay')
            if not desktop_only:
                health=bootstrap_health()
                remote_state=remote_worker_state(out)
                if not remote_state['alive'] and health['available_kb']<8*1024*1024:
                    raise RuntimeError('Laptop has less than 8 GiB available RAM; stop other workloads before starting')
                if health['disk_free']<15*(1<<30):
                    raise RuntimeError('Laptop needs at least 15 GiB free disk for encodings and saved models')
                paths=sorted(set(source_paths()+schedule['stages']['neural']['required_files']))
                sync_laptop(paths,out)
                checked=host_call('health',out,'neural',threads,True)
                if any(p['pid']!=remote_state.get('child_pid') for p in checked['busy']):
                    raise RuntimeError('Laptop already has another heavy model job; no fits started')
                # CUDA is required; this small check creates no training job.
                cuda=remote_python("import torch,json;print(json.dumps({'available':torch.cuda.is_available(),'devices':torch.cuda.device_count()}))")
                if not cuda['available']:
                    raise RuntimeError('Laptop CUDA unavailable; no fits started')
                announce('Starting fixed v80 recipe: desktop trees; laptop neural pool. Logs are saved under execution/.')
                host_call('launch',out,'neural',threads,True)
                try:
                    host_call('launch',out,'trees',threads)
                    wait_workers([('trees',False),('neural',True)],out,threads)
                except BaseException:
                    host_call('stop',out,'neural',threads,True)
                    raise
                collect_neural(out)
            else:
                announce('Starting fixed v80 recipe sequentially on desktop.')
                for stage in ['trees','neural']:
                    host_call('launch',out,stage,threads)
                    wait_workers([(stage,False)],out,threads)
            announce('Rebuilding and auditing the complete v80 ensemble.')
            stage_run('assemble','run',out,threads)
        # Immutable input/source checks also apply to all resumed delivery paths.
        freeze(out,schedule)
        announce('Preparing verified offline replay and private draft release.')
        stage_run('package','prepare',out,threads)
        announce('Replacing final submission once; verifying remote hash and fresh official acceptance.')
        stage_run('upload','run',out,threads)
        announce('Final accepted. Updating the deadline publication bundle.')
        stage_run('package','activate',out,threads)
        save(out/'complete.json',dict(status='complete',finished_at=now().isoformat(),
                                     acceptance=artifact(out/'delivery/acceptance.json'),
                                     final=artifact(out/'final/candidate.parquet')))
        announce('Complete: v80 refit final accepted; replay release and deadline publication are updated.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--desktop-only',action='store_true',help='Use desktop sequentially; select before the first fit')
    args=parser.parse_args()
    if not 1<=args.threads<=16:
        parser.error('--threads must be between 1 and 16')
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'VALIDATION/runs'):
        parser.error('--out must be inside VALIDATION/runs')
    try:
        if args.command=='plan':
            print(json.dumps(plan(out,args.threads,args.desktop_only),indent=2))
        else:
            run(out,args.threads,args.desktop_only)
    except (Exception,KeyboardInterrupt) as exc:
        announce(f'Stopped: {exc or type(exc).__name__}')
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
