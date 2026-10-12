"""Scalar count distributions and independent SQL membership census."""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import joint_public_moments as model
from records import sha256,write_json
OUT=ROOT/'VALIDATION/runs/v3_20261012_cross_date_taxi_coverage'


def art(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha256(p))


def main():
    read=lambda p:json.loads(p.read_text())
    result=read(OUT/'result.json');identity=read(OUT/'identity.json')
    assert result['identity']==art(OUT/'identity.json')
    for record in [identity['source'],*identity['helpers'],*[r for x in identity['sources'].values() for r in x.values()]]:
        assert sha256(ROOT/record['path'])==record['sha256']
    comparisons=0
    for view,files in identity['sources'].items():
        columns=['mvt_id','month','gap','prior','eligible','quality','runway','local_day','assigned_day','fltrule','delay_scope']
        rows=pd.read_parquet(ROOT/files['inputs']['path'],columns=columns)
        parameters=read(ROOT/files['parameters']['path'])
        mass=[]
        sigma=model.SETTINGS['ordinary_sigma']
        for r in rows.itertuples(index=False):
            if r.gap<=3600:
                n=float(ndtr((7200-r.prior)/sigma)-ndtr(-r.prior/sigma))
            else:
                params=parameters[r.month] if view=='cv' else parameters
                values,weights,_=model.prior.support(r.gap,r.prior,False,np.nan,'LIRF',params['support'])
                n=sum(float(w) for y,w in zip(values,weights) if 0<=y<=7200)
            mass.append(n)
        rows['mass']=mass
        with duckdb.connect(config={'threads':1}) as con:
            con.register('rows',rows)
            con.read_parquet(str(ROOT/files['taxi_source']['path'])).create_view('source')
            con.read_parquet(str(ROOT/files['predictions']['path'])).create_view('pred')
            con.execute('''CREATE TEMP TABLE eligible AS
                SELECT r.month,r.local_day,r.runway,count(*) AS rows,sum(mass) AS mass,s.source_n
                FROM rows r JOIN source s USING(local_day,runway)
                WHERE r.eligible AND r.quality
                GROUP BY r.month,r.local_day,r.runway,s.source_n
                HAVING count(*)>=20 AND s.source_n>=20 AND sum(mass)>=20
                  AND s.source_n/sum(mass) BETWEEN .5 AND 1.2''')
            actual=con.execute('''SELECT r.month,r.local_day,r.runway,r.assigned_day,list(r.mvt_id ORDER BY r.mvt_id)
                FROM rows r JOIN pred p USING(mvt_id) WHERE p.touched
                GROUP BY r.month,r.local_day,r.runway,r.assigned_day ORDER BY 1,2,3,4''').fetchall()
            records=[r for r in result['records'] if r['view']==view]
            saved=sorted((r['month'],r['local_day'],r['runway'],r['assigned_day'],sorted(r['target_ids'])) for r in records)
            assert actual==saved
            for rec in records:
                found=con.execute('''SELECT e.local_day,e.runway,count(*) AS shared,
                    sum(CASE WHEN r.gap>3600 THEN 1 ELSE 0 END),e.rows,e.mass,e.source_n
                    FROM rows r JOIN eligible e USING(month,local_day,runway)
                    WHERE r.month=? AND r.local_day<>? AND r.assigned_day=?
                        AND r.eligible AND r.quality AND r.fltrule IS DISTINCT FROM 'V' AND r.delay_scope
                    GROUP BY e.local_day,e.runway,e.rows,e.mass,e.source_n ORDER BY 1,2''',
                    [rec['month'],rec['local_day'],rec['assigned_day']]).fetchall()
                expected=sorted((r['local_day'],r['runway'],r['shared_rows'],r['shared_long_rows'],
                    r['taxi_rows'],r['expected_count'],r['source_count']) for r in rec['extra_cells'])
                assert len(found)==len(expected)
                for a,b in zip(found,expected):
                    assert a[:5]==b[:5] and a[6]==b[6]
                    assert abs(a[5]-b[5])<1e-8
                comparisons+=1
    write_json(OUT/'audit.json',dict(status='pass',source=art(Path(__file__).resolve()),
        identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),cohort_comparisons=comparisons,
        independent_scalar_probability_and_SQL_membership=True,no_labels_or_posterior=True))
    print(dict(status='pass',cohort_comparisons=comparisons))


if __name__=='__main__':
    with threadpool_limits(1):main()
