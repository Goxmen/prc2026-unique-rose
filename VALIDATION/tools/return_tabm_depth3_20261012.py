"""Return only the new deeper-member artifacts; reuse verified cached inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REL = 'VALIDATION/runs/v3_20261012_nn_public_tabm_depth3'
OUT = ROOT / REL


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


REMOTE = '''import hashlib,json
from pathlib import Path
root=Path('/home/gunes/PRC');out=root/REL
identity=json.loads((out/'identity.json').read_text())
audit=json.loads((out/'audit.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert audit['status']=='pass' and audit['all_held_reload_exact'] and audit['retained_controls_not_refitted']
assert audit['identity']==dict(path=REL+'/identity.json',sha256=sha(out/'identity.json'))
files={out/'audit.json'}
for month,epochs in [('2025-01',21),('2025-07',17)]:
 path=out/f'k0/networks/{month}_tabm.json';r=json.loads(path.read_text());files.add(path)
 assert r['identity']==audit['identity'] and r['member']==0 and r['seed']==42 and r['epochs']==epochs
 assert r['all_held_reloaded_exact'] and r['inputs']==identity['folds'][month]
 assert r['preprocessing']==identity['folds'][month]['preprocessing']
 assert {k:r[k] for k in ['host','gpu','torch']}==identity['execution']
 for key in ['model','prediction','checkpoint','preprocessing']:
  p=root/r[key]['path'];assert sha(p)==r[key]['sha256'];files.add(p)
print(json.dumps(dict(identity=audit['identity'],execution=identity['execution'],
 files=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)])))
'''


def main():
    assert not (OUT / 'return_audit.json').exists()
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'glaptop-wsl',
        '/home/gunes/PRC/.venv/bin/python -'], input='REL=' + repr(REL) + '\n' + REMOTE,
        text=True, capture_output=True, timeout=60, check=True)
    remote = json.loads(result.stdout)
    assert remote['identity'] == dict(path=REL + '/identity.json', sha256=sha(OUT / 'identity.json'))
    identity = json.loads((OUT / 'identity.json').read_text())
    assert remote['execution'] == identity['execution']
    shared = {r['preprocessing']['path'] for r in identity['folds'].values()}
    missing = []
    for a in remote['files']:
        relative = Path(a['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert a['path'].startswith(REL + '/') or a['path'] in shared
        p = ROOT / relative
        if p.exists():
            assert sha(p) == a['sha256'], p
        else:
            missing.append(a['path'])
    (OUT / 'remote_return_manifest.json').write_text(json.dumps(remote, indent=2) + '\n')
    listing = OUT / 'return_files.txt'
    listing.write_text(''.join(p + '\n' for p in missing))
    if missing:
        subprocess.run(['rsync', '-a', '--files-from=' + str(listing),
                        'glaptop-wsl:/home/gunes/PRC/', str(ROOT) + '/'], check=True)
    for a in remote['files']:
        p = ROOT / a['path']
        assert p.stat().st_size == a['bytes'] and sha(p) == a['sha256']
    value = dict(status='pass', identity=remote['identity'], files=remote['files'],
                 execution=remote['execution'], transferred_files=len(missing),
                 transferred_bytes=sum(a['bytes'] for a in remote['files'] if a['path'] in missing),
                 source_sha256=sha(Path(__file__)), cached_inputs_reused=True, no_training=True)
    (OUT / 'return_audit.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: value[k] for k in ['status', 'transferred_files', 'transferred_bytes']}))


if __name__ == '__main__':
    main()
