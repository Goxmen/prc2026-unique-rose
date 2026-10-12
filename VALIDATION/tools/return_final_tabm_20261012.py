"""Return only the new final-view TabM predictions and verified receipts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REL='VALIDATION/runs/v3_20261012_final_public_tabm_inference'
OUT=ROOT/REL


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run():
    code='''import hashlib,json
from pathlib import Path
r=Path('/home/gunes/PRC');out=r/REL
old=json.loads((out/'old.json').read_text());final=json.loads((out/'final.json').read_text())
assert old['all_old_predictions_exact'] and final['all_final_reloads_exact']
assert old['identity']==final['identity']
paths=[out/n for n in ['identity.json','old.json','final.json']]
paths += [r/a['path'] for a in final['predictions']]
print(json.dumps([dict(path=str(p.relative_to(r)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
'''
    response=subprocess.run(['ssh','glaptop-wsl','/home/gunes/PRC/.venv/bin/python -'],
        input='REL='+repr(REL)+'\n'+code,text=True,capture_output=True,check=True)
    records=json.loads(response.stdout);assert len(records)==6
    for a in records:
        p=ROOT/a['path'];assert p.is_relative_to(OUT)
        if p.exists():assert sha(p)==a['sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    paths=OUT/'return_files.txt';paths.write_text('\n'.join(a['path'] for a in records if not (ROOT/a['path']).exists())+'\n')
    subprocess.run(['rsync','-a','--files-from='+str(paths),'glaptop-wsl:/home/gunes/PRC/',str(ROOT)+'/'],check=True)
    for a in records:assert (ROOT/a['path']).stat().st_size==a['size'] and sha(ROOT/a['path'])==a['sha256']
    manifest=json.loads((OUT/'identity.json').read_text())
    for a in manifest['sources']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
    import numpy as np
    import pandas as pd
    expected=pd.read_parquet(ROOT/'data/final_submitting.parquet',columns=['MVT_ID_mvt']).MVT_ID_mvt
    for a in json.loads((OUT/'final.json').read_text())['predictions']:
        f=pd.read_parquet(ROOT/a['path']);assert len(f)==f.mvt_id.nunique()==670790
        np.testing.assert_array_equal(np.sort(f.mvt_id),np.sort(expected))
        assert np.isfinite(f.base).all() and (f.base>=0).all()
    value=dict(status='pass',host='glaptop-wsl',files=records,source_hashes_exact=True,
        old_replay_exact=True,final_reloads_exact=True,final_template_id_set_exact=True,
        prediction_order='retained packet order; assembly must join by ID',no_training=True)
    (OUT/'return_audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='pass',files=len(records),bytes=sum(a['size'] for a in records),rows=670790)))


if __name__=='__main__':run()
