"""Tiny source parser and adversarial fixed-exclusion fixture; no real data."""
import ast
import importlib.util
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'VALIDATION/tools/audit_portable_public_source_20261013.py'
spec=importlib.util.spec_from_file_location('auditor',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
sub={'zones':"'Europe/Rome'",'truth':'FALSE','month':'2025-01','name':'M','keys':'airport','col':'B','expr':'0.','sql':'0.','WEIGHT':'.4032872515123992'}
queries=[]
for node in ast.walk(ast.parse(p.read_text())):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='execute' and node.args:
        expr=node.args[0]
        if isinstance(expr,ast.Constant):q=expr.value
        elif isinstance(expr,ast.JoinedStr):q=''.join(part.value if isinstance(part,ast.Constant) else sub.get(ast.unparse(part.value),'0.') for part in expr.values)
        else:continue
        duckdb.extract_statements(q);queries.append(q)
f=pd.DataFrame(dict(mvt_id=range(9),x=[5000.,0.,100.,5000.,1500.,1394.,8606.,999.,.5]))
d=pd.DataFrame(dict(mvt_id=range(9),has_id=[True]*3+[False]*6,nm_gap=[0,5000,-4000]+[np.nan]*6))
w=pd.DataFrame(dict(mvt_id=[3,4,5,6],recovered_gap=[5000,2000,5000,5000]))
with duckdb.connect() as con:
    for name,g in [('f',f),('d',d),('w',w)]:con.register(name,g)
    out=con.execute('select f.mvt_id,'+a.constraint('f.x')+' as bounded from f join d using(mvt_id) left join w using(mvt_id) order by f.mvt_id').df()
np.testing.assert_array_equal(out.bounded,[3606.,1394.,0.,1394.,5606.,1394.,8606.,999.,.5])
report=dict(status='pass',source=a.art(p),fixture_source=a.art(Path(__file__)),SQL_statements_parsed=len(queries),constraint_fixture_rows=9,negative_lower_edge_moves_upper=True,strict_edges_and_tie_lower=True,no_real_inputs_read=True)
out=ROOT/'VALIDATION/reports/portable_public_source_auditor_preflight_20261013.json'
if out.exists():out.rename(out.with_name(out.stem+'_v1.json'))
out.write_text(json.dumps(report,indent=2)+'\n');print(report)
