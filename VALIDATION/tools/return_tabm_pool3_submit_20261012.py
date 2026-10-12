"""Return completed new full-year members without copying cached training data."""
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path

from return_tabm_pool3_20261012 import ROOT,digest

RUN=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit'
REMOTE=r'''
import hashlib,json
from pathlib import Path
root=Path.cwd();run=root/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def art(path):return dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size)
def verify(record):
 p=root/record['path'];assert sha(p)==record['sha256'];return p
def read(path):return json.loads(path.read_text())
identity=read(run/'identity.json');export=read(run/'export.json')
assert export['identity']['sha256']==sha(run/'identity.json')
assert export['rows']==344841 and export['seeds']==[42,1042,2042] and export['view']=='old'
assert export['members'][0]==identity['reused_seed42']['prediction']
execution=read(root/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3/identity.json')['execution']
files={run/'export.json',verify(export['pool'])}
for k,seed in [(1,1042),(2,2042)]:
 member=run/f'k{k}';mid=identity['members'][str(k)];verify(mid)
 frozen=read(member/'identity.json');trained=read(member/'trained/k0.json')
 prediction=read(member/'networks/k0.json');preflight=read(member/'preflight.json')
 assert trained['identity']==prediction['identity']==preflight['identity']==mid
 assert trained['design']==frozen['design'] and trained['resources']==frozen['resources']
 assert trained['seed']==seed and trained['epochs']==15
 assert trained['training_rows']==2051327 and trained['updates_completed_before_inference']
 assert [x['epoch'] for x in trained['curve']]==list(range(1,16))
 assert {key:trained[key] for key in ['host','gpu','torch']}==execution
 assert prediction['trained']['sha256']==sha(member/'trained/k0.json')
 assert prediction['rows']==344841 and prediction['ranking_view']=='old' and prediction['all_ranking_reload_exact']
 assert export['members'][k]==prediction['prediction']
 files.update([member/'trained/k0.json',member/'networks/k0.json',member/'preflight.json'])
 for record in [trained['model'],trained['preprocessing'],prediction['prediction']]:files.add(verify(record))
print(json.dumps(dict(identity=art(run/'identity.json'),execution=execution,files=[art(p) for p in sorted(files)])))
'''


def main():
    receipt=RUN/'return_audit.json';assert not receipt.exists(),'Preserve completed return'
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','glaptop-wsl',
        'cd /home/gunes/PRC && .venv/bin/python -'],input=REMOTE,text=True,capture_output=True,timeout=60,check=True)
    remote=json.loads(result.stdout)
    assert remote['identity']['sha256']==digest(RUN/'identity.json')
    expected=json.loads((ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3/identity.json').read_text())['execution']
    assert remote['execution']==expected
    prefix=str(RUN.relative_to(ROOT))+'/'
    transfer=[]
    for record in remote['files']:
        relative=Path(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts and record['path'].startswith(prefix)
        path=ROOT/relative
        if path.exists():assert digest(path)==record['sha256'],'Existing artifact differs; inspect instead of overwriting'
        else:transfer.append(record['path'])
    (RUN/'remote_return_manifest.json').write_text(json.dumps(remote,indent=2)+'\n')
    listing=RUN/'return_files.txt';listing.write_text(''.join(p+'\n' for p in transfer))
    if transfer:
        subprocess.run(['rsync','-a','--files-from='+str(listing),'glaptop-wsl:/home/gunes/PRC/',str(ROOT)+'/'],check=True)
    for record in remote['files']:
        path=ROOT/record['path'];assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256']
    value=dict(status='pass',returned_at=datetime.now(timezone.utc).isoformat(),source_sha256=digest(Path(__file__)),
        remote_manifest_sha256=digest(RUN/'remote_return_manifest.json'),identity=remote['identity'],
        execution=remote['execution'],files=remote['files'],transferred_files=len(transfer),
        transferred_bytes=sum(r['bytes'] for r in remote['files'] if r['path'] in transfer),
        no_training_or_inference=True,original_seed42_and_large_inputs_reused=True)
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['status','transferred_files','transferred_bytes']}))


if __name__=='__main__':main()
