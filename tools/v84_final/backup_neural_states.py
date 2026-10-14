#!/usr/bin/env python3
"""Export actual retained neural inference states and verify tensor-exact backup.

Full optimizer/restart checkpoints are preserved in place. The backup projects
only training-resume state away; it never derives or fits weights from outputs.
This is an incomplete research backup while any selected family is missing.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,tarfile
from datetime import datetime,timezone

os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline'),str(ROOT/'VALIDATION/tools')]
import torch

SKIP={'optimizer','scheduler','generator','torch_rng','numpy_rng','python_rng','cuda_rng','curve'}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def tensors(obj,prefix=''):
    if isinstance(obj,torch.Tensor):yield prefix,obj.detach().cpu()
    elif isinstance(obj,dict):
        for k,v in obj.items():yield from tensors(v,prefix+'/'+str(k))
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj):yield from tensors(v,prefix+'/'+str(i))


def tensor_manifest(obj):
    result={}
    for k,t in tensors(obj):
        raw=t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result[k]={'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(raw).hexdigest()}
    return result


def art(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}


def backup(inventory,directory):
    torch.set_num_threads(1)
    source=json.loads(inventory.read_text())
    if directory.exists():raise ValueError('Preserve existing backup; no silent overwrite')
    directory.mkdir(parents=True)
    rows=[]
    for item in source['artifacts']:
        path=ROOT/item['path']
        if sha(path)!=item['sha256']:raise ValueError('Actual original state changed')
        relative=Path(item['path']);target=directory/'states'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if item['kind']=='model':
            state=torch.load(path,map_location='cpu',weights_only=False)
            if not isinstance(state,dict) or not ('model' in state or 'state_dict' in state):
                raise ValueError('Unrecognized model state; do not omit tensors')
            compact={k:v for k,v in state.items() if k not in SKIP}
            original_tensors=tensor_manifest(compact)
            if not original_tensors:raise ValueError('No actual weight tensors in checkpoint')
            torch.save(compact,target)
            restored=torch.load(target,map_location='cpu',weights_only=False)
            if tensor_manifest(restored)!=original_tensors:raise ValueError('Saved tensor identity changed')
            if set(restored)!=set(compact):raise ValueError('Inference metadata keys changed')
            row=dict(source=item,export=art(target),tensor_identity=original_tensors,
                     all_inference_tensors_exact=True,removed_resume_fields=sorted(set(state)&SKIP))
            del state,compact,restored
        else:
            import shutil
            shutil.copyfile(path,target)
            if sha(target)!=item['sha256']:raise ValueError('Preprocessing backup changed')
            row=dict(source=item,export=art(target),preprocessing_byte_exact=True)
        rows.append(row)
    manifest={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'source_inventory':art(inventory),
        'status':'verified_actual_neural_state_backup','all_selected_states_present':False,
        'missing':source['missing'],'artifacts':rows,'full_restart_sources_unchanged':True,'no_fit':True,'no_inference':True}
    manifest_path=directory/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=directory.with_suffix('.tar');files=sorted(p for p in directory.rglob('*') if p.is_file())
    with tarfile.open(archive,'w') as tar:
        for p in files:
            info=tar.gettarinfo(str(p),arcname=str(p.relative_to(directory)));info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            with p.open('rb') as f:tar.addfile(info,f)
    expected={str(p.relative_to(directory)):art(p) for p in files};seen=set()
    with tarfile.open(archive,'r') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:raise ValueError('Unexpected backup member')
            stream=tar.extractfile(member);h=hashlib.sha256()
            for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
            if h.hexdigest()!=expected[member.name]['sha256']:raise ValueError('Archive member changed')
            seen.add(member.name)
    if seen!=set(expected):raise ValueError('Missing archive member')
    receipt=dict(status='pass',archive=art(archive),manifest=art(manifest_path),files=len(rows),
                 every_archive_member_hash_verified=True,all_inference_tensors_exact=True,
                 complete_selected_weight_set=False,missing=source['missing'])
    (directory.parent/(directory.name+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();backup(a.inventory,a.directory)
