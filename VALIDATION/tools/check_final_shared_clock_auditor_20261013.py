"""Tiny SQL binding/arithmetic check only; no competition arrays or builder IO."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import ast
import hashlib
import json
from pathlib import Path
import duckdb
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
AUDITOR=ROOT/'VALIDATION/tools/audit_final_shared_clock_transport_20261013.py'
OUT=ROOT/'VALIDATION/reports/cohort_learning_20261013/final_shared_clock_auditor_preflight.json'


def art(path):return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert not OUT.exists()
    tree=ast.parse(AUDITOR.read_text());sql=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='execute':
            value=node.args[0]
            if isinstance(value,ast.Constant):sql.append(value.value)
            elif isinstance(value,ast.JoinedStr):
                sql.append(eval(compile(ast.Expression(value),str(AUDITOR),'eval'),{},dict(name='candidate',idcol='MVT_ID_mvt',valuecol='TAXITIME_SEC_mvt')))
            else:raise AssertionError(ast.dump(value))
    for query in sql:duckdb.extract_statements(query)
    checked=next(s for s in sql if s.startswith('CREATE TEMP VIEW checked'))
    bad_query=next(s for s in sql if s.startswith('SELECT count(*),count(distinct id)'))
    group_query=next(s for s in sql if s.startswith('SELECT utc_month,airport'))
    final_ids=[50,10,30,20,40]
    values=[75,0,12,64,90]
    candidate=pd.DataFrame(dict(MVT_ID_mvt=final_ids,TAXITIME_SEC_mvt=values))
    trace=pd.DataFrame(dict(MVT_ID_mvt=final_ids,shared=[False,True,True,True,False],
                            delta_takeoff_seconds=[0.,-12.,0.,12.,0.],submitted=values)).iloc[::-1].copy()
    old=pd.DataFrame(dict(MVT_ID_mvt=[20,30,10],TAXITIME_SEC_mvt=[40,12,8]))
    fallback=pd.DataFrame(dict(id=[40,20,10,50,30],pred=[90.,103.,101.,75.,102.]))
    final_raw=pd.DataFrame(dict(MVT_ID_mvt=final_ids,PHASE_mvt=['DEP']*5,ADEP_mvt=['LIRF']*5,
        MVT_TIME_UTC_mvt=pd.to_datetime(['2026-06-01','2026-01-01','2026-01-01','2026-07-01','2026-02-01'],utc=True),
        BLOCK_TIME_UTC_mvt=pd.Series([pd.NaT]*5,dtype='datetime64[ns, UTC]'),TAXITIME_SEC_mvt=[None]*5))
    old_raw=final_raw.set_index('MVT_ID_mvt').loc[[30,10,20]].reset_index()
    old_raw.MVT_TIME_UTC_mvt-=pd.to_timedelta([0,-12,12],unit='s')
    checks=[]
    with duckdb.connect(config={'threads':1,'memory_limit':'128MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,frame in [('candidate',candidate),('trace',trace),('old',old),('b',fallback),('final_raw',final_raw),('old_raw',old_raw)]:
            con.register('fixture',frame);con.execute(f'CREATE TABLE {name} AS SELECT * FROM fixture');con.unregister('fixture')
        con.execute(checked)
        result=con.execute(bad_query).fetchone();assert result==(5,5,3,2,0),result
        groups=con.execute(group_query).fetchdf();assert groups.rows.sum()==5 and groups.zero_clipped.sum()==1
        checks.append('Shuffled unique IDs, positive and negative shared shifts, zero shift, negative clamp and unchanged two-month fallback')
        for label,mutation in [
            ('wrong shared prediction','UPDATE candidate SET TAXITIME_SEC_mvt=63 WHERE MVT_ID_mvt=20'),
            ('wrong fallback','UPDATE candidate SET TAXITIME_SEC_mvt=76 WHERE MVT_ID_mvt=50'),
            ('wrong trace','UPDATE trace SET submitted=1 WHERE MVT_ID_mvt=10'),
            ('wrong membership','UPDATE trace SET shared=false WHERE MVT_ID_mvt=10'),
            ('wrong delta sign','UPDATE trace SET delta_takeoff_seconds=12 WHERE MVT_ID_mvt=10'),
            ('visible hidden truth','UPDATE final_raw SET TAXITIME_SEC_mvt=123 WHERE MVT_ID_mvt=10'),
            ('wrong phase',"UPDATE final_raw SET PHASE_mvt='ARR' WHERE MVT_ID_mvt=40"),
            ('oversized clock shift',"UPDATE old_raw SET MVT_TIME_UTC_mvt=MVT_TIME_UTC_mvt-INTERVAL '30 seconds' WHERE MVT_ID_mvt=20"),
        ]:
            con.execute('BEGIN');con.execute(mutation);bad=con.execute(bad_query).fetchone()[-1];con.execute('ROLLBACK')
            assert bad>0,label;checks.append('Reject '+label)
    value=dict(status='pass',source=art(Path(__file__)),auditor=art(AUDITOR),SQL_statements_parsed=len(sql),fixture_rows=5,
               checks=checks,no_competition_arrays_read=True,no_candidate=True,no_model_or_posterior_inference=True)
    OUT.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(status='pass',receipt=art(OUT),SQL_statements_parsed=len(sql),checks=len(checks))))


if __name__=='__main__':main()
