"""Bounded selected tree-leaf weight inventory from existing provenance receipts."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'VALIDATION/runs';OUT=RUNS/'v3_20261014_v84_final'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def art(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p),'bytes':Path(p).stat().st_size}

destination=OUT/'selected_tree_weights.json';assert not destination.exists()
recovery=RUNS/'v3_20261014_exact_v80_recovery';recovered=read(recovery/'newly_recovered_components.json')['components'];mapping={};evidence=[]
for v,rec in recovered.items():mapping[v]=[rec[k] for k in ['model','classifier'] if k in rec]
mapping['3']=[read(OUT/'v3/identity.json')['model']]
overnight=RUNS/'v3_20261012_final_legacy_overnight_replay/identity.json';j=read(overnight);evidence.append(art(overnight))
for v,key in [(17,'no_month'),(18,'no_history'),(19,'clock_features'),(20,'small_trees'),(21,'rule_routing')]:mapping[str(v)]=[x for x in j['sources'] if x['path'].endswith(f'refit_2025__{key}__42.txt')]
for v,parts in [(26,['v26_ordinary','v26_classifier']),(27,['v27_ordinary','v26_classifier']),(28,['v28_ordinary','v26_classifier']),(29,['v29_ordinary','v29_classifier'])]:
 mapping[str(v)]=[]
 for name in parts:
  p=RUNS/'v3_20261013_v80_refit_final/trees/models'/f'{name}.json';evidence.append(art(p));mapping[str(v)].append(read(p)['model'])
stand=[]
for name in ['ordinary','classifier']:
 p=RUNS/'v3_20261012_final_missing_stand_reconstruct'/f'{name}_trained.json';stand.append(read(p)['model']);evidence.append(art(p))
for v in [30,33,34,37]:mapping[str(v)]=stand
for folder,versions in [('v3_20261012_final_legacy_v38_replay',[38]),('v3_20261012_final_legacy_adsb_replay',[45,46])]:
 p=RUNS/folder/'identity.json';j=read(p);evidence.append(art(p));models=[x for x in j['sources'] if x['path'].endswith('.txt')]
 for v in versions:mapping[str(v)]=[x for x in models if len(versions)==1 or f'/submission_v{v}/' in x['path']]
selected=read(recovery/'prediction_fidelity_recheck/assembly_identity.json')['geometry71']['versions'];expected=[v for v in selected if 1<v<47];assert set(map(int,mapping))==set(expected)
unique={}
for v,refs in mapping.items():
 assert refs,v
 for ref in refs:
  p=ROOT/ref['path'];assert p.exists() and sha(p)==ref['sha256'],str(p)
  with p.open('rb') as f:assert f.read(5)==b'tree\n',str(p)
  unique[ref['path']]=dict(**art(p),model_format='LightGBM text',selected_versions=sorted(int(k) for k,rs in mapping.items() if any(r['path']==ref['path'] for r in rs)))
aliases=[]
for family in ['baseline','recovery']:
 local=recovery/'basic_original/original_models'/family/'model.txt';expected_hash=mapping['12' if family=='baseline' else '14'][0]['sha256']
 assert sha(local)==expected_hash
 aliases.append({'historical_laptop_path':f'submissions/selected-20260916/{family}/model.txt','desktop_copy':art(local)})
report={'status':'pass','source':art(__file__),'scope':'31 selected non-neural tree leaves only; v1 is analytic. Neural families and their shared tree parents/public corrections require separate complete inventory.','selected_tree_versions':expected,'mapping':mapping,'unique_weight_files':list(unique.values()),'historical_receipts':evidence,'resolved_laptop_path_aliases':aliases,'all_selected_tree_leaf_weights_present_and_hash_verified':True,'all_complete_v84_weights_available':False,'remaining_known_missing_family':'15 October8 HPO156 neural states affecting54/56/57/58/60; independent recovery attempt pending','no_fits':True,'no_inference':True,'no_upload':True}
destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'result':art(destination),'tree_leaves':len(mapping),'weight_files':len(unique)}))
