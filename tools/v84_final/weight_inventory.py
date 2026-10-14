#!/usr/bin/env python3
"""Trace retained v84 provenance to actual model-state artifacts; no inference.

Prediction vectors are deliberately never classified as trained model weights.
Known missing HPO state is explicit even though its output vectors are retained.
"""
import argparse
from collections import deque
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'VALIDATION/runs'
DEFAULT=RUN/'v3_20261014_v84_final'
SEEDS=[
 RUN/'v3_20261013_v80_refit_final/assembly_identity.json',
 RUN/'v3_20261013_v80_refit_final/neural/result.json',
 RUN/'v3_20261013_v80_refit_final/trees/result.json',
 RUN/'v3_20261014_exact_v80_recovery/newly_recovered_components.json',
 RUN/'v3_20261014_exact_v80_recovery/tree_audit/v3_original_replay/trained.json',
 RUN/'v3_20261012_final_public_direct_components/identity.json',
 RUN/'v3_20261012_final_public_tree_inference/identity.json',
 RUN/'v3_20261012_final_public_ple_inference/identity.json',
 RUN/'v3_20261012_final_public_tabm_inference/identity.json',
 RUN/'v3_20261012_final_weather_inference/identity.json',
 RUN/'v3_20261012_final_legacy_modern_replay/identity.json',
 RUN/'v3_20261008_final_submit/work/parts/nn.json']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def resolve(name,parent):
    p=Path(name)
    if p.is_absolute():return p if p.is_relative_to(ROOT) else None
    if name.startswith(('VALIDATION/','pipeline/','tools/','data/','submissions/')):return ROOT/p
    for directory in [parent.parent,*list(parent.parents)[:5]]:
        candidate=directory/p
        if candidate.is_file() and candidate.is_relative_to(ROOT):return candidate
    return None


def refs(x,parent):
    if isinstance(x,dict):
        if isinstance(x.get('path'),str):
            p=resolve(x['path'],parent)
            if p is not None:yield p,x.get('sha256')
        for k,v in x.items():
            if isinstance(k,str) and '/' in k and k.endswith(('.pt','.pth','.cbm','.json','.txt','.pkl')):
                p=resolve(k,parent)
                if p is not None:yield p,v if isinstance(v,str) and len(v)==64 else None
            yield from refs(v,parent)
    elif isinstance(x,list):
        for v in x:yield from refs(v,parent)
    elif isinstance(x,str) and ('/' in x or x.startswith('model.')) and x.endswith(('.pt','.pth','.cbm','.json','.txt','.pkl')):
        p=resolve(x,parent)
        if p is not None:yield p,None


def model_kind(p):
    if p.suffix in {'.pt','.pth','.cbm'}:return 'model_state'
    if p.suffix=='.txt' and ('models' in p.parts or p.name in {'model.txt','ordinary.txt','classifier.txt'}):return 'tree_model'
    if p.suffix=='.pkl' and (p.name in {'preprocessing.pkl','encoder.pkl','scaler.pkl'} or 'preprocessing' in p.parts):return 'preprocessing_state'
    return None


def inventory(out,hash_models):
    todo=deque(SEEDS+[out/'assembly_identity.json',out/'dependency_inventory.json']);seen=set();models={};missing_json=[]
    while todo:
        p=todo.popleft()
        if p in seen:continue
        seen.add(p)
        if not p.is_file():missing_json.append(str(p.relative_to(ROOT)));continue
        if p.stat().st_size>40_000_000:continue
        try:value=json.loads(p.read_text())
        except (ValueError,UnicodeError):continue
        for q,digest in refs(value,p):
            kind=model_kind(q)
            if kind:
                item=models.setdefault(q,{'path':str(q.relative_to(ROOT)),'kind':kind,'exists':q.is_file(),'expected_hashes':set(),'referenced_by':set()})
                if digest:item['expected_hashes'].add(digest)
                item['referenced_by'].add(str(p.relative_to(ROOT)))
            elif q.suffix=='.json' and q not in seen:todo.append(q)
    # This corrected-source selected model is recorded with a hash-only key.
    v3=RUN/'v3_20261014_exact_v80_recovery/tree_audit/v3_original_replay/model.txt'
    models[v3]={'path':str(v3.relative_to(ROOT)),'kind':'tree_model','exists':v3.is_file(),
        'expected_hashes':{json.loads(v3.with_name('trained.json').read_text())['model_sha256']},'referenced_by':{str(v3.with_name('trained.json').relative_to(ROOT))}}
    for p,d in models.items():
        d['expected_hashes']=sorted(d['expected_hashes']);d['referenced_by']=sorted(d['referenced_by'])
        if p.is_file():
            d['bytes']=p.stat().st_size
            if hash_models:
                d['sha256']=sha(p);d['matches_recorded_hash']=not d['expected_hashes'] or d['sha256'] in d['expected_hashes']
    missing={'family':'HPO156 N09, batch2048, stops15/21/16, groups42/142/242×5',
        'actual_selected_production':'October8 final_submit refit; producer discarded returned model states',
        'affected_versions':[54,56,57,58,60],'missing_model_count':15,
        'original_oct6_hpo_states_also_absent':True,'saved_predictions_not_weights':str(RUN/'v3_20261008_final_submit/work/nn'),
        'original_weather167_checkpoints_inapplicable':True}
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'all_selected_weights_available':False,
        'known_missing_selected_weights':[missing],'scope':'Provenance-graph inventory includes some historical/control states; listed availability is not a claim every upstream execution was serialized.',
        'json_receipts_traversed':len(seen),'missing_referenced_json':missing_json,
        'actual_state_files':sorted(models.values(),key=lambda d:d['path']),
        'hashes_verified_now':hash_models,'predictions_are_not_weights':True,'no_fit':True,'no_inference':True,'no_upload':True}
    dest=out/'release/weight_inventory.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':str(dest.relative_to(ROOT)),'json_receipts':len(seen),'model_state_files':len(models),'available':sum(d['exists'] for d in models.values()),'bytes':sum(d.get('bytes',0) for d in models.values()),'missing_selected_original_models':15,'hash_mismatches':sum(d.get('matches_recorded_hash') is False for d in models.values())}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--hash-models',action='store_true');a=p.parse_args();inventory(a.out,a.hash_models)
