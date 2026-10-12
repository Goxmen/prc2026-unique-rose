"""Four static stand-context inputs in the retained184-input main tree.

Fit, saved-model recovery, exact control replay, scoring and final-stage audit
reuse the reviewed feature-screen implementation in an isolated namespace.
The original clock experiment and its sources/results remain unchanged.
"""
import argparse
import types
from threadpoolctl import threadpool_limits

import flight_id_clock_lgb as implementation
import stand_month_context as features

ROOT=features.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_stand_month_context_lgb'
MANIFEST=OUT/'identity.json'
DESIGN=dict(parent='exactv78 selected184-input main LightGBM; separate coefficient-one v79 transport',
    change='four label-free stand-month clock/arrival context inputs',features=features.FEATURES,
    original_columns=184,columns=188,seed=42,trial='LB1',rounds=10000,threads=4,
    rows_labels='all original ordinary rows and targets, same desktop; held Jan/Jul unchanged',
    hypothesis='typical stand geometry/context can transfer to rare and previously unseen stands',
    source='full-month supplied arrival labels and other departures actual AOBT_3 clocks; no departure target or block',
    no_drift_correction=True,no_airport_gate=True,no_feature_search=True,no_weight_search=True,
    implementation=features.art(implementation.__file__),
    limitation='reused development; does not inherit a gain from another team or from training-only diagnostics',
    no_production_fit=True,no_upload=True)


def namespace():
    ns={**implementation.__dict__,'features':features,'OUT':OUT,'MANIFEST':MANIFEST,'DESIGN':DESIGN,'__file__':__file__}
    names=['guard','append_clocks','inputs','preflight','completed','fit','score','audit']
    # Only dimensional assertions/text change; every numerical operation is retained.
    import ast,inspect,textwrap
    for name in names:
        fn=getattr(implementation,name)
        if name in ['inputs','preflight','fit']:
            tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
            class Dimensions(ast.NodeTransformer):
                def visit_Constant(self,node):
                    if isinstance(node.value,int) and node.value==186:
                        return ast.copy_location(ast.Constant(value=188),node)
                    if isinstance(node.value,str) and '186' in node.value:
                        return ast.copy_location(ast.Constant(value=node.value.replace('186','188')),node)
                    return node
            tree=Dimensions().visit(tree)
            exec(compile(ast.fix_missing_locations(tree),'<four-stand-context-input-screen>','exec'),ns)
        else:
            ns[name]=types.FunctionType(fn.__code__,ns,name,fn.__defaults__,fn.__closure__)
    assert implementation.OUT!=OUT and implementation.features is not features
    return ns


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['guard','preflight','fit','score','audit'])
    with threadpool_limits(1):namespace()[p.parse_args().stage]()
