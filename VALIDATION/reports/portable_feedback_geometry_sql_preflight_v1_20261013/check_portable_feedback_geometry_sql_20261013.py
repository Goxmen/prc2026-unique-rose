"""Bind geometry audit SQL against tiny synthetic tables; never run optimizer."""
import ast
import hashlib
import json
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'VALIDATION/portable_feedback_geometry_20261013.py'
OUT=ROOT/'VALIDATION/reports/portable_feedback_geometry_sql_preflight_20261013.json'


def art(path):return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert not OUT.exists()
    # Only retained version identifiers are read, never their vectors/scores.
    versions=json.loads((ROOT/'VALIDATION/runs/v3_20261012_tabm_feedback_geometry/result.json').read_text())['versions']+[81,82]
    old_difference=' OR '.join(f'a.v{v} IS DISTINCT FROM b.v{v}' for v in versions[:-2])
    expr='h.v64'+''.join(f'+(0.)*(h.v{v}-h.v64)' for v in versions if v!=64)
    sql=[]
    for node in ast.walk(ast.parse(SOURCE.read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='execute':
            text=node.args[0]
            if isinstance(text,ast.Constant):sql.append(text.value)
            elif isinstance(text,ast.JoinedStr):
                sql.append(eval(compile(ast.Expression(text),str(SOURCE),'eval'),{},dict(name='r',old_difference=old_difference,version=81,expr=expr)))
            else:raise AssertionError(ast.dump(text))
    with duckdb.connect(config={'threads':1,'memory_limit':'128MB'}) as con:
        cols=','.join(f'10.0::DOUBLE v{v}' for v in versions)
        con.execute(f'CREATE TABLE r AS SELECT 1::BIGINT mvt_id,{cols}')
        con.execute("CREATE TABLE h AS SELECT *,12.0::DOUBLE y,'2025-01' month,'2025-01-01' day,'LIRF' airport FROM r")
        con.execute('CREATE VIEW old_r AS SELECT * FROM r')
        con.execute('CREATE VIEW old_h AS SELECT * FROM h')
        con.execute('CREATE TABLE c AS SELECT mvt_id,month,day,airport,y,10.0::DOUBLE control,10.0::DOUBLE pred FROM h')
        con.execute('CREATE TABLE source AS SELECT mvt_id AS MVT_ID_mvt,v81 AS TAXITIME_SEC_mvt FROM r')
        con.execute('CREATE TABLE cv_source AS SELECT *,v81 AS submitted FROM h')
        con.execute("CREATE TABLE truth AS SELECT 1::BIGINT MVT_ID_mvt,'DEP' PHASE_mvt,12.0 TAXITIME_SEC_mvt,TIMESTAMP '2025-01-01' MVT_TIME_UTC_mvt,'LIRF' ADEP_mvt")
        for query in sql:
            duckdb.extract_statements(query);con.execute(query).fetchall()
    receipt=dict(status='pass_source_SQL_preflight',source=art(Path(__file__)),runner=art(SOURCE),
        queries_parsed_and_bound=len(sql),synthetic_rows=1,versions=54,no_actual_predictions_or_labels=True,
        no_model_fit=True,no_weight_solve=True,no_official_score_read=True)
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status='pass',receipt=art(OUT),queries=len(sql))))


if __name__=='__main__':main()
