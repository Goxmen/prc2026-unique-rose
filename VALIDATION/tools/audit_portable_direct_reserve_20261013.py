"""Independent raw-ID/calendar/truth, SQL rounding, schema and CV score audit.

No producer builder or rounding function is used, no models are evaluated.
"""
import hashlib
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import duckdb
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_direct_reserve'
COUNTS=dict(cv=344419,old=344841,final=670790)


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)==record,path;return path


def audit():
    assert not (OUT/'audit.json').exists()
    identity,result=read(OUT/'identity.json'),read(OUT/'result.json')
    assert identity['auditor']==art(__file__) and result['identity']==art(OUT/'identity.json')
    for record in identity['sources']:verify(record)
    allocation=read(verify(result['allocation']))
    assert allocation['authority']=='/root' and allocation['identity']==result['identity']
    assert allocation['state']=='released_for_direct_reserve_serialization' and allocation['views']==list(COUNTS)
    cv_audit=read(ROOT/'VALIDATION/runs/v3_20261013_portable_direct_cv/audit.json')
    assert cv_audit['status']=='pass' and cv_audit['result']==art(ROOT/'VALIDATION/runs/v3_20261013_portable_direct_cv/result.json')
    prod_audit=read(ROOT/'VALIDATION/runs/v3_20261012_final_public_direct_components/independent_audit.json')
    assert prod_audit['status']=='pass'
    report={}
    for view,spec in identity['specs'].items():
        item=result['views'][view];path=verify(item['prediction'])
        with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
            con.execute("SET TimeZone='UTC'")
            con.read_parquet(str(path)).create_view('f')
            assert con.execute('SELECT count(*),count(distinct mvt_id) FROM f').fetchone()==(COUNTS[view],COUNTS[view])
            con.read_parquet(str(verify(spec['prediction']))).create_view('d')
            con.read_parquet([str(verify(x)) for x in spec['raw']]).create_view('r')
            n,u,bad=con.execute('''SELECT count(*),count(distinct f.mvt_id),count(*) FILTER(WHERE
                NOT isfinite(f.raw_direct) OR f.raw_direct IS DISTINCT FROM d.tabm_direct
                OR f.submitted IS DISTINCT FROM round_even(greatest(CAST(d.tabm_direct AS DOUBLE),0.),0)
                OR f.month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m')
                OR f.day IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m-%d')
                OR f.airport IS DISTINCT FROM r.ADEP_mvt OR r.PHASE_mvt IS DISTINCT FROM 'DEP')
                FROM f JOIN d USING(mvt_id) JOIN r ON f.mvt_id=r.MVT_ID_mvt''').fetchone()
            assert n==u==item['rows']==COUNTS[view] and bad==0,(view,n,u,bad)
            report[view]=dict(rows=n,prediction=item['prediction'],all_SQL_values_exact=True)
            if view=='cv':
                assert con.execute('''SELECT count(*) FROM f JOIN d USING(mvt_id) JOIN r ON f.mvt_id=r.MVT_ID_mvt
                    WHERE f.y IS DISTINCT FROM d.y OR f.y IS DISTINCT FROM r.TAXITIME_SEC_mvt
                    OR f.month IS DISTINCT FROM d.month OR f.day IS DISTINCT FROM d.day''').fetchone()[0]==0
                months=dict(con.execute('SELECT month,sqrt(avg(pow(CAST(submitted AS DOUBLE)-y,2))) FROM f GROUP BY month').fetchall())
                for month,value in months.items():assert abs(value-result['months'][month])<1e-8
                assert abs(np.mean(list(months.values()))-result['cv'])<1e-8
                report[view].update(months=months,cv=float(np.mean(list(months.values()))))
            else:
                assert prod_audit['views'][view]['prediction']==spec['prediction']
                assert con.execute('SELECT count(*) FROM f JOIN r ON f.mvt_id=r.MVT_ID_mvt WHERE r.TAXITIME_SEC_mvt IS NOT NULL').fetchone()[0]==0
                template=pq.read_table(verify(spec['template']));candidate=pq.read_table(verify(item['candidate']))
                frame=pq.read_table(path)
                assert template.schema==candidate.schema and frame['mvt_id'].to_pylist()==template['MVT_ID_mvt'].to_pylist()
                for name in template.column_names:
                    if name!='TAXITIME_SEC_mvt':assert candidate[name].equals(template[name])
                np.testing.assert_array_equal(candidate['TAXITIME_SEC_mvt'].to_numpy(),frame['submitted'].to_numpy())
                report[view]['candidate']=item['candidate']
    value=dict(status='pass',identity=art(OUT/'identity.json'),source=art(__file__),result=art(OUT/'result.json'),
               views=report,serialized_nearest_even_exact=True,no_fit_or_model_inference=True,no_upload=True,selected=False)
    (OUT/'audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='pass',rows=sum(COUNTS.values()),cv=result['cv'],audit=art(OUT/'audit.json'))))


if __name__=='__main__':audit()
