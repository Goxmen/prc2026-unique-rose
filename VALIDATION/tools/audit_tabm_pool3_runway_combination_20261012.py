"""Same independent combination audit, with only pool paths and CV name changed."""
import ast
import inspect
from pathlib import Path
import textwrap

import audit_tabm_runway_combination_20261012 as first


def implementation():
    root=first.ROOT
    tree=ast.parse(textwrap.dedent(inspect.getsource(first.main)))
    class Rename(ast.NodeTransformer):
        count=0
        def visit_Constant(self,node):
            if isinstance(node.value,str) and 'combined' in node.value:
                self.count+=1
                return ast.copy_location(ast.Constant(value=node.value.replace('combined','three_tabm_runway')),node)
            return node
    rename=Rename();tree=rename.visit(tree);assert rename.count==3,rename.count
    scope={**first.main.__globals__,
        'OUT':root/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo_submit',
        'CV':root/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo',
        'TABM':root/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit_stack',
        '__file__':str(Path(__file__).resolve())}
    exec(compile(ast.fix_missing_locations(tree),'<three-tabm-independent-combination-audit>','exec'),scope)
    return scope['main']


if __name__=='__main__':
    implementation()()
    path=first.RUNS/'v3_20261012_tabm_pool3_other_runway_combo_submit/independent_audit.json'
    value=first.read(path)
    value['original_auditor']=first.art(Path(first.__file__).resolve())
    value['specialization']='Only three CV column-name literals and three source folders changed'
    path.write_text(first.json.dumps(value,indent=2)+'\n')
