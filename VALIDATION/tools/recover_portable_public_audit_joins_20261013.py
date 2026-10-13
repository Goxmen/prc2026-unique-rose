"""Additive audit-only qualified JOIN repair; original producer/auditor immutable.

Only two SQL JOIN clauses change. No inference, prediction or score is repeated.
The executed snapshot and its exact two-string AST diff are retained and bound.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import traceback
import types
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
BASE=ROOT/'VALIDATION/tools/audit_portable_public_source_20261013.py'
BASE_SHA='2f8bc3e8dab2da4c0bbc52e5444b4e1b3fcce7769c779ee0a293a51a07a376c2'
IDENTITY_SHA='b186449ec102d81434c3b0e0ef9483c44e18146a855214e42c6a9c5313c7d660'
PATCHES=[('LEFT JOIN mc USING(airport,local_month)','LEFT JOIN mc ON m.airport=mc.airport AND m.local_month=mc.local_month'),
         ('LEFT JOIN dc USING(airport,local_day,runway)','LEFT JOIN dc ON i.airport=dc.airport AND i.local_day=dc.local_day AND i.runway=dc.runway')]


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def verify(a):
    path=ROOT/a['path'];assert art(path)=={k:a[k] for k in ['path','sha256']};return path

def patched():
    assert art(BASE)['sha256']==BASE_SHA
    original=BASE.read_text();changed=original
    for before,after in PATCHES:
        assert changed.count(before)==1 and after not in changed
        changed=changed.replace(before,after)
    a,b=ast.parse(original),ast.parse(changed)
    constants_a=[n for n in ast.walk(a) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    constants_b=[n for n in ast.walk(b) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert len(constants_a)==len(constants_b)
    differences=0
    for x,y in zip(constants_a,constants_b):
        if x.value!=y.value:
            expected=x.value
            for before,after in PATCHES:expected=expected.replace(before,after)
            assert expected==y.value;differences+=1;y.value=x.value
    assert differences==2 and ast.dump(a)==ast.dump(b)
    return original,changed


def freeze(view):
    folder=OUT/view/'audit_sql_join_recovery';assert not folder.exists()
    assert art(OUT/'identity.json')['sha256']==IDENTITY_SHA
    assert not (OUT/view/'audit.json').exists()
    result=read(OUT/view/'result.json');assert result['identity']==art(OUT/'identity.json')
    original,snapshot=patched();folder.mkdir()
    (folder/'original_auditor.py').write_text(original);(folder/'executed_auditor.py').write_text(snapshot)
    sources=[Path(__file__),BASE,folder/'original_auditor.py',folder/'executed_auditor.py',OUT/'identity.json',OUT/view/'result.json']
    for key in ['prediction','monthly_cells','daily_cells','allocation']:sources.append(verify(result[key]))
    if view=='cv':sources.extend([OUT/'score.json',OUT/'cv/completion/03.log',OUT/'cv/completion/execution.json'])
    else:sources.extend([OUT/'cv/audit.json',OUT/'cv/audit_sql_join_recovery/identity.json'])
    if view=='final':sources.append(OUT/'old/audit.json')
    for a in result['posterior_receipts']:
        path=verify(a);sources.append(path);receipt=read(path)
        sources.extend([verify(receipt['prediction']),verify(receipt['days'])])
    identity=dict(view=view,source=art(Path(__file__)),original_identity=art(OUT/'identity.json'),result=art(OUT/view/'result.json'),
        original_auditor=art(BASE),executed_auditor=art(folder/'executed_auditor.py'),sources=[art(p) for p in sources],
        patches=[dict(before=a,after=b) for a,b in PATCHES],only_two_SQL_string_constants_changed=True,
        no_inference_repeat=True,no_predictions_changed=True,no_scores_changed=True)
    (folder/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(dict(stage='frozen',identity=art(folder/'identity.json'))))


def audit(view):
    folder=OUT/view/'audit_sql_join_recovery';identity=read(folder/'identity.json')
    assert identity['source']==art(Path(__file__)) and identity['view']==view
    assert identity['original_identity']==art(OUT/'identity.json') and art(OUT/'identity.json')['sha256']==IDENTITY_SHA
    assert identity['result']==art(OUT/view/'result.json')
    for a in identity['sources']:verify(a)
    original,expected=patched();snapshot=verify(identity['executed_auditor'])
    assert snapshot.read_text()==expected and identity['patches']==[dict(before=a,after=b) for a,b in PATCHES]
    assert not (OUT/view/'audit.json').exists() and not (folder/'execution.json').exists()
    execution=dict(state='running',identity=art(folder/'identity.json'));started=time.monotonic()
    (folder/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    try:
        module=types.ModuleType('qualified_join_auditor')
        # Keep original path for ROOT/provenance lookups. The executed snapshot
        # is separately pinned and explicitly identified in the final receipt.
        module.__file__=str(BASE)
        exec(compile(expected,str(snapshot),'exec'),module.__dict__)
        with module.threadpool_limits(1):module.audit(view)
        original_receipt=OUT/view/'audit.json';saved=folder/'numerical_audit.json';original_receipt.replace(saved)
        report=read(saved);assert report['status']=='pass' and report['source']==art(BASE)
        report.update(source=art(Path(__file__)),original_auditor=art(BASE),executed_auditor=art(snapshot),
            recovery_identity=art(folder/'identity.json'),numerical_audit=art(saved),
            audit_only_join_repair=True,producer_identity_unchanged=True,no_inference_repeat=True)
        original_receipt.write_text(json.dumps(report,indent=2)+'\n')
        execution.update(state='complete',audit=art(original_receipt),seconds=time.monotonic()-started)
        (folder/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');print(json.dumps(execution))
    except BaseException:
        execution.update(state='failed',failure=traceback.format_exc(),seconds=time.monotonic()-started)
        (folder/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','audit']);parser.add_argument('--view',choices=['cv','old','final'],required=True)
    args=parser.parse_args();globals()[args.stage](args.view)
