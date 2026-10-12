"""Conditional three-TabM/runway file; retain the audited combination bytecode."""
import argparse
import ast
import inspect
from pathlib import Path
import textwrap
import types

from threadpoolctl import threadpool_limits
import tabm_other_runway_combo_submit as first
import tabm_pool3_other_runway_combo_cv as scoring
import nn_public_tabm_pool3_submit as production

ROOT = first.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo_submit'
TABM = ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit_stack'
DESIGN = {**first.DESIGN,
    'inputs':'audited equal TabM seeds42/1042/2042 and robust other-runway integer deltas versus v78',
    'selection':'three-versus-one passes both standalone comparisons and the frozen runway combination',
    'seeds':[42,1042,2042]}


def selected_guard():
    scoring.guard()
    assert scoring.read(production.AUDIT)['status']=='pass'
    assert scoring.read(production.STACK/'result.json')['paired']['three_tabm_vs_one_tabm']['decision']=='improvement'
    assert scoring.read(production.TRANSPORT/'result.json')['paired_three_vs_one']['decision']=='improvement'
    result=scoring.read(scoring.OUT/'result.json')
    assert result['paired']['one_tabm_runway']['decision']=='improvement'
    assert result['all_row_sql_arithmetic_truth_calendar_scores']=='pass'


def namespace():
    cv=types.SimpleNamespace(OUT=scoring.OUT,guard=selected_guard,combine=scoring.first.combine,
                            __file__=scoring.__file__)
    scope={**first.__dict__,'OUT':OUT,'TABM':TABM,'DESIGN':DESIGN,'cv':cv,'__file__':__file__}
    for name in ['prepare','guard']:
        original=getattr(first,name)
        scope[name]=types.FunctionType(original.__code__,scope,name,original.__defaults__,original.__closure__)
        assert scope[name].__code__ is original.__code__
    tree=ast.parse(textwrap.dedent(inspect.getsource(first.build)))
    class Rename(ast.NodeTransformer):
        count=0
        def visit_Constant(self,node):
            if node.value=='combined':
                self.count+=1
                return ast.copy_location(ast.Constant(value='three_tabm_runway'),node)
            return node
    rename=Rename();tree=rename.visit(tree);assert rename.count==1
    exec(compile(ast.fix_missing_locations(tree),'<three-tabm-combination-cv-name>','exec'),scope)
    assert first.OUT!=OUT and first.TABM!=TABM
    return scope


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','build','check'])
    with threadpool_limits(1):
        scope=namespace();stage=parser.parse_args().stage
        if stage=='check':print('Exact prepare/guard bytecode; one output metric name changed in builder')
        elif stage=='prepare':
            scope['prepare']()
            path=OUT/'identity.json';value=first.read(path)
            value['sources'] += [first.art(p) for p in [first.__file__,scoring.first.__file__,
                production.AUDIT,production.STACK/'result.json',production.TRANSPORT/'result.json']]
            value['implementation']='Retained audited builder; isolated globals and one CV metric name substitution'
            first.write(path,value)
        else:scope[stage]()
