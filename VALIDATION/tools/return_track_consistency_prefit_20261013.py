"""Return only completed preflight/independent-audit artifacts; no fit command."""
import hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'


def art(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def main():
    inventory=json.loads((OUT/'transfer_inventory.json').read_text())
    identity=json.loads((OUT/'identity.json').read_text())
    rel=str(OUT.relative_to(ROOT))
    sources=sorted(set([a['path'] for a in identity['files'] if a['path'].endswith('.py')]+[
        rel+'/identity.json',rel+'/comparison/identity.json',rel+'/source_review.json',rel+'/cpu_checks.json',
        rel+'/deployment_plan.json','VALIDATION/tools/run_track_consistency_stage_20261013.py',
        'VALIDATION/tools/run_track_consistency_prefit_audit_remote_20261013.py']))
    expected=inventory['expected_outputs']+[rel+'/prefit_audit.json']
    remote='''import hashlib,json\nfrom pathlib import Path\nr=Path.cwd()\nout=r/PAYLOAD['out']\ndef art(p):\n p=Path(p);return dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)\nfiles=set(PAYLOAD['expected'])\nfor stage in ['preflight','independent_prefit']:\n folder=out/'runtime'/stage\n state=json.loads((folder/'execution.json').read_text());assert state['state']=='complete' and state['returncode']==0\n files.update(str(p.relative_to(r)) for p in folder.iterdir() if p.is_file())\nresult=dict(files=[art(r/p) for p in sorted(files)],sources=[art(r/p) for p in PAYLOAD['sources']])\nprint(json.dumps(result))\n'''
    payload=dict(out=rel,expected=expected,sources=sources)
    source='PAYLOAD='+repr(payload)+'\n'+remote
    process=subprocess.run(['ssh','glaptop-wsl','cd ~/PRC && .venv/bin/python -'],input=source,text=True,capture_output=True,check=True)
    remote_result=json.loads(process.stdout)
    for a in remote_result['sources']:assert art(ROOT/a['path'])==a,a['path']
    for a in remote_result['files']:
        p=ROOT/a['path']
        if p.exists():assert art(p)==a,'Refuse to overwrite a different returned file: '+a['path']
    folder=OUT/'prefit_return';folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'remote_manifest.json';assert not manifest.exists()
    manifest.write_text(json.dumps(remote_result,indent=2)+'\n')
    files=folder/'files.txt';files.write_text('\n'.join(a['path'] for a in remote_result['files'])+'\n')
    with (folder/'rsync.log').open('xb') as log:
        subprocess.run(['rsync','-a','--files-from='+str(files),'glaptop-wsl:PRC/',str(ROOT)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
    for a in remote_result['files']:assert art(ROOT/a['path'])==a,a['path']
    audit=json.loads((OUT/'prefit_audit.json').read_text());pref=json.loads((OUT/'preflight.json').read_text())
    identity_art={k:v for k,v in art(OUT/'identity.json').items() if k!='bytes'}
    assert audit['status']==pref['status']=='pass' and audit['identity']==pref['identity']==identity_art
    assert audit['teachers']==pref['teachers']
    assert audit['preflight']=={k:v for k,v in art(OUT/'preflight.json').items() if k!='bytes'}
    value=dict(status='pass',source=art(__file__),identity=identity_art,preflight=art(OUT/'preflight.json'),
        prefit_audit=art(OUT/'prefit_audit.json'),remote_manifest=art(manifest),
        files=remote_result['files'],source_files_verified=len(remote_result['sources']),
        returned_files=len(remote_result['files']),returned_bytes=sum(a['bytes'] for a in remote_result['files']),
        audit_executed_on='GLAPTOP using separate unchanged independent source',
        reason='The exact frozen fold pickle containers are laptop-only; no2.55GB redundant transfer.',
        no_fit_or_inference=True,no_frozen_source_mutation=True)
    path=OUT/'prefit_return_audit.json'
    with path.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(status='pass',receipt=art(path),files=value['returned_files'],bytes=value['returned_bytes'])))


if __name__=='__main__':main()
