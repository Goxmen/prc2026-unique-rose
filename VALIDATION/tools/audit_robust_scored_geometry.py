"""Independent SQL reconstruction of the read-only robust score diagnosis."""
import sys
from pathlib import Path
import duckdb
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import robust_scored_geometry as experiment
from records import write_json


def audit():
    out=experiment.OUT;read,verify,art=experiment.read,experiment.verify,experiment.art
    assert not (out/'audit.json').exists(),'Preserve completed audit'
    result=read(out/'result.json');design=read(verify(result['design']))
    assert design['source']==art(experiment.__file__)
    prepared=read(verify(design['prepared']))
    assert read(experiment.PREPARED.with_name('audit.json'))['prepared']==design['prepared']
    live=read(verify(design['snapshot']));by_version={r['version']:r for r in live['history'] if r['status']=='Succeeded'}
    for key,score in design['official_scores'].items():
        assert by_version[int(key)]['score']==score and by_version[int(key)]['usedPairs']==344841
    checks=[]
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('b',verify(prepared['ranking'])),('o',verify(result['ranking'])),
            ('v',verify(result['validation'])),('c',verify(prepared['cv'])),
            ('t',ROOT/'VALIDATION/runs/v3_20261011_daily_ifr/rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct v.mvt_id),sum(case when
            v.y is distinct from t.y or v.month is distinct from t.month or v.day is distinct from t.day
            or v.y is distinct from c.y or v.v74 is distinct from c.v74 then 1 else 0 end)
            from v join c using(mvt_id) join t using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for arm in ['bounded','robust']:
            weights=result[arm]['weights'];assert set(weights)==set(design['official_scores'])
            assert abs(sum(weights.values())-1)<1e-8
            expression='b.v64+'+'+'.join(f'({float(w):.17g})*(b.v{v}-b.v64)' for v,w in weights.items())
            sql=f'''with a as (select o.*,({expression}) independent from b join o using(mvt_id))
                select count(*),count(distinct mvt_id),max(abs(independent-{arm}_raw)),
                sqrt(avg(pow({arm}-{arm}_raw,2))),avg(pow({arm}_raw,2)),
                sum(case when {arm}<0 or {arm}!=round_even({arm},0)
                    or abs({arm}-greatest({arm}_raw,0))>0.50000001 then 1 else 0 end) from a'''
            count,unique,error,distance,square,invalid=con.execute(sql).fetchone()
            assert count==unique==344841 and error<1e-7 and invalid==0
            norm=con.execute('select '+','.join(f'avg(pow(v{v},2))' for v in weights)+' from b').fetchone()
            w=np.array(list(weights.values()));scores=np.array([design['official_scores'][v] for v in weights])
            mse=float(w@(scores**2-np.asarray(norm))+square)
            np.testing.assert_allclose(np.sqrt(mse),result[arm]['projected_rmse'],atol=1e-6,rtol=0)
            if arm=='robust':
                np.testing.assert_allclose(distance,result[arm]['serialization_rms'],atol=1e-10,rtol=0)
                radius=float(np.abs(w)@(np.abs(scores)*1e-4+2.5e-9))
                np.testing.assert_allclose(radius,result[arm]['score_mse_radius'],atol=1e-12,rtol=0)
            values=con.execute(f'select month,sqrt(avg(pow({arm}-y,2))) from v group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['months'][month][arm],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in values]),result['cv'][arm],atol=1e-8,rtol=0)
            checks.append(dict(arm=arm,ranking_rows=count,raw_max_difference=error,sql_mse=mse,cv_rows=n))
    write_json(out/'audit.json',dict(status='pass',result=art(out/'result.json'),source=art(__file__),
        live_scores_prepared_hashes_raw_truth_and_serialization_checked=True,checks=checks))
    print(checks)


if __name__=='__main__':audit()
