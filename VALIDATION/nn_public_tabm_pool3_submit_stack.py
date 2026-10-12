"""Three-TabM pool through the retained audited production stack.

Only member loading, averaging and CV field references specialize the old
builder. Every correction, control replay and final-file audit is retained.
"""
import argparse
import ast
import inspect
from pathlib import Path
import textwrap
import types

from threadpoolctl import threadpool_limits
import nn_public_tabm_submit_stack as first
import nn_public_tabm_pool3_submit as production

ROOT = first.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit_stack'
MANIFEST = OUT / 'identity.json'
art, read, verify, write = first.art, first.read, first.verify, first.write_json
DESIGN = {**first.DESIGN,
          'change': 'replace equal public/physical pool with TabM seeds42/1042/2042',
          'retained_members': [], 'reused_tabm_member': 42,
          'cv': 'audited three-seed stack and fixed coefficient-one v79 transport'}


def member_frames():
    production.guard()
    export=read(production.OUT/'export.json')
    assert export['identity']==art(production.OUT/'identity.json')
    assert export['pool']==art(production.OUT/'replacement_pool.parquet')
    return [first.pd.read_parquet(verify(record)) for record in export['members']]


def build_function():
    tree=ast.parse(textwrap.dedent(inspect.getsource(first.build)))
    edits={
        'new_member=pd.read_parquet(verify(member.completed(identity)[\'prediction\']))':
            'new_members=member_frames()',
        '[*old_members,new_member]': '[*old_members,*new_members]',
        "new_pool=np.mean([new_member.base.to_numpy(float),*(f.base.to_numpy(float) for f in old_members[1:])],axis=0)":
            'new_pool=np.mean([f.base.to_numpy(float) for f in new_members],axis=0)',
    }
    def node(text):
        value=ast.parse(text).body[0]
        return value.value if isinstance(value,ast.Expr) else value
    replacements={ast.dump(node(a),include_attributes=False): node(b) for a,b in edits.items()}
    strings={'replace_tabm':'three_tabm','replace_tabm_vs_selected':'three_tabm_vs_selected',
             'VALIDATION/runs/v3_20261012_delta_transport/tabm/result.json':
             'VALIDATION/runs/v3_20261012_tabm_pool3_delta_transport/tabm_pool3/result.json'}
    counts={key:0 for key in replacements};string_counts={key:0 for key in strings}
    class Replace(ast.NodeTransformer):
        def visit(self,value):
            if not isinstance(value,ast.AST):
                return value
            key=ast.dump(value,include_attributes=False)
            if key in replacements:
                counts[key]+=1
                return replacements[key]
            return super().visit(value)
        def visit_Constant(self,value):
            if isinstance(value.value,str) and value.value in strings:
                string_counts[value.value]+=1
                return ast.copy_location(ast.Constant(value=strings[value.value]),value)
            return value
    tree=Replace().visit(tree)
    assert list(counts.values())==[1,1,1]
    assert string_counts=={'replace_tabm':3,'replace_tabm_vs_selected':1,
        'VALIDATION/runs/v3_20261012_delta_transport/tabm/result.json':1}
    adapter=types.SimpleNamespace(OUT=production.OUT,MANIFEST=production.first.MANIFEST)
    ns=dict(first.build.__globals__,OUT=OUT,MANIFEST=MANIFEST,DESIGN=DESIGN,
            guard=guard,member_frames=member_frames,member=adapter,
            validation=types.SimpleNamespace(OUT=production.STACK),__file__=__file__)
    exec(compile(ast.fix_missing_locations(tree),'<fixed-three-tabm-production-stack>','exec'),ns)
    return ns['build']


def freeze():
    assert not MANIFEST.exists()
    first.guard();production.guard()
    assert read(first.OUT/'audit.json')['status']=='pass'
    exported=read(production.OUT/'export.json')
    assert exported['identity']==art(production.OUT/'identity.json') and exported['seeds']==[42,1042,2042]
    assert read(production.AUDIT)['status']=='pass'
    build_function()
    paths=[Path(__file__),Path(first.__file__),Path(production.__file__),first.MANIFEST,
        first.OUT/'audit.json',production.OUT/'identity.json',production.OUT/'export.json',
        verify(exported['pool']),production.STACK/'result.json',production.STACK/'audit.json',
        production.TRANSPORT/'result.json',production.AUDIT]
    for rec in exported['members']:
        paths.append(verify(rec))
    for k in [1,2]:
        n=production.member(k)
        identity=read(n['MANIFEST']);record=n['completed'](identity)
        assert record is not None
        trained=n['trained'](identity)
        for field in ['model','preprocessing']:
            paths.append(verify(trained[field]))
        paths.extend([n['MANIFEST'],n['OUT']/'trained/k0.json',n['OUT']/'networks/k0.json'])
    write(MANIFEST,dict(design=DESIGN,source=art(__file__),
        sources=[art(p) for p in dict.fromkeys(paths)],
        specialization='Three exact AST node changes plus five CV string references; retained correction code and audit bytecode',
        no_version=True,no_upload=True))


def guard():
    identity=read(MANIFEST)
    assert identity['design']==DESIGN and identity['source']==art(__file__)
    for record in identity['sources']:
        verify(record)
    first.guard();production.guard()
    return identity


def build():
    build_function()()


def audit():
    source=first.audit
    ns=dict(source.__globals__,OUT=OUT,MANIFEST=MANIFEST,DESIGN=DESIGN,guard=guard,__file__=__file__)
    fn=types.FunctionType(source.__code__,ns,'three_tabm_stack_audit',source.__defaults__,source.__closure__)
    assert fn.__code__ is source.__code__
    fn()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','build','audit'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
