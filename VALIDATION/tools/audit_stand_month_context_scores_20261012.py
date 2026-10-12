"""Reuse the independent tree-screen SQL audit for four stand-context inputs."""
import ast
import inspect
import json
from pathlib import Path
import textwrap

import audit_flight_id_clock_scores_20261012 as original

RUN=original.RUNS/'v3_20261012_stand_month_context_lgb'
FEATURES=original.RUNS/'v3_20261012_stand_month_context'
TRANSPORT=original.RUNS/'v3_20261012_stand_month_context_delta_transport'
OUT=original.ROOT/'VALIDATION/reports/stand_month_context_20261012/audit.json'
NAMES=['stand_month_peer_clock_median','stand_month_peer_clock_log_n',
       'stand_month_arrival_median','stand_month_arrival_log_n']


def implementation():
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.main)))
    count=dict(features=0,dimension=0,key=0,experiment=0)
    class Specialize(ast.NodeTransformer):
        def visit_List(self,node):
            if all(isinstance(v,ast.Constant) for v in node.elts) and [v.value for v in node.elts]==[
                    'idclock_iobt_residual','idclock_neighbor_span']:
                count['features']+=1
                return ast.copy_location(ast.List(elts=[ast.Constant(v) for v in NAMES],ctx=ast.Load()),node)
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value==186:
                count['dimension']+=1
                return ast.copy_location(ast.Constant(188),node)
            if isinstance(node.value,str) and node.value in ['flight_id_clock','flight_id_clock/result.json','flight_id_clock/scored.parquet']:
                count['key']+=1
                return ast.copy_location(ast.Constant(node.value.replace('flight_id_clock','stand_context')),node)
            return node
        def visit_keyword(self,node):
            if node.arg=='flight_id_clock':node.arg='stand_context';count['experiment']+=1
            return self.generic_visit(node)
    tree=Specialize().visit(tree)
    assert count==dict(features=1,dimension=1,key=4,experiment=1),count
    ns={**original.main.__globals__,'RUN':RUN,'TRANSPORT':TRANSPORT,'OUT':OUT,'__file__':__file__}
    exec(compile(ast.fix_missing_locations(tree),'<independent-four-input-score-audit>','exec'),ns)
    return ns['main'],count


def main():
    a=original.read(FEATURES/'audit.json');r=original.read(FEATURES/'result.json')
    assert a['status']=='pass' and a['result']==original.art(FEATURES/'result.json')
    assert a['manifest']==r['manifest']==original.art(FEATURES/'manifest.json')
    assert a['independent_scalar_queries']==1200 and a['exact_raw_source_query_exclusion_medians_counts']
    assert original.read(FEATURES/'manifest.json')['design']['features']==NAMES
    for item in r['outputs'].values():original.verify(item)
    fn,count=implementation();fn()
    value=original.read(OUT)
    value.update(feature_audit=original.art(FEATURES/'audit.json'),
        reused_auditor=original.art(original.__file__),specialization=count,
        scope=value['scope']+' Four feature names/dimension and experiment directory are specialized; '
            'raw-source medians/counts and query exclusion have a separate1200-row scalar audit.')
    OUT.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
