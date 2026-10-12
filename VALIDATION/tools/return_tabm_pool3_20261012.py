"""Return only completed TabM CV artifacts; verify both remote and local hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


REMOTE=r'''
import hashlib,json
from pathlib import Path
root=Path.cwd();run=root/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def art(path):return dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size)
def verify(record):
 p=root/record['path'];assert sha(p)==record['sha256'];return p
identity=json.loads((run/'identity.json').read_text())
audit=json.loads((run/'audit.json').read_text())
assert audit['status']=='pass' and audit['identity']['sha256']==sha(run/'identity.json')
assert audit['all_held_reload_exact'] and audit['original_seed42_reused_exactly']
files={run/'audit.json'}
for month in ['2025-01','2025-07']:
 for k,seed in [(1,1042),(2,2042)]:
  p=run/f'k{k}/networks/{month}_tabm.json';r=json.loads(p.read_text())
  assert r['identity']==audit['identity'] and r['month']==month and r['member']==k and r['seed']==seed
  assert r['all_held_reloaded_exact'] and r['inputs']==identity['folds'][month]
  assert r['preprocessing']==identity['folds'][month]['preprocessing']
  assert {key:r[key] for key in ['host','gpu','torch']}==identity['execution']
  files.add(p)
  for key in ['model','prediction','checkpoint','preprocessing']:files.add(verify(r[key]))
print(json.dumps(dict(identity=art(run/'identity.json'),audit=art(run/'audit.json'),
 files=[art(p) for p in sorted(files)],execution=identity['execution'])))
'''


def main():
    receipt=RUN/'return_audit.json';assert not receipt.exists(),'Preserve completed return'
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','glaptop-wsl',
        'cd /home/gunes/PRC && .venv/bin/python -'],input=REMOTE,text=True,capture_output=True,timeout=60,check=True)
    remote=json.loads(result.stdout)
    assert remote['identity']['sha256']==digest(RUN/'identity.json')
    identity=json.loads((RUN/'identity.json').read_text());assert remote['execution']==identity['execution']
    shared={value['preprocessing']['path'] for value in identity['folds'].values()}
    prefix=str(RUN.relative_to(ROOT))+'/'
    transfer=[]
    for record in remote['files']:
        relative=Path(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert record['path'].startswith(prefix) or record['path'] in shared
        path=ROOT/relative
        if path.exists():assert digest(path)==record['sha256'],'Existing artifact differs; inspect instead of overwriting'
        else:transfer.append(record['path'])
    (RUN/'remote_return_manifest.json').write_text(json.dumps(remote,indent=2)+'\n')
    listing=RUN/'return_files.txt';listing.write_text(''.join(p+'\n' for p in transfer))
    if transfer:
        subprocess.run(['rsync','-a','--files-from='+str(listing),
                        'glaptop-wsl:/home/gunes/PRC/',str(ROOT)+'/'],check=True)
    for record in remote['files']:
        path=ROOT/record['path']
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256']
    value=dict(status='pass',returned_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(Path(__file__)),remote_manifest_sha256=digest(RUN/'remote_return_manifest.json'),
        identity=remote['identity'],execution=remote['execution'],files=remote['files'],
        transferred_files=len(transfer),transferred_bytes=sum(r['bytes'] for r in remote['files'] if r['path'] in transfer),
        no_remote_training_or_inference=True,original_seed42_and_large_inputs_reused=True)
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['status','transferred_files','transferred_bytes']}))


if __name__=='__main__':main()
