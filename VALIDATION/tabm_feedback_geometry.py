"""One fixed geometry diagnosis after the genuinely new v80 direction scored.

Retain the 51-vector scored basis, append v80, and use the unchanged bounds and
ridge. Do not add v79 as a rounding direction or impute cancelled v75. This is
leaderboard selection; reused CV and independent SQL checks remain explicit.
No candidate number, fitting, queue or upload is created.
"""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import target205_geometry as parent
from tools import audit_target205_geometry as independent
from records import write_json

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_tabm_feedback_geometry'
art,read,verify=parent.art,parent.read,parent.verify
MANIFEST=ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_candidate/submission_v80/manifest.json'
DESIGN=dict(parent='51 retained actually scored vectors plus v80',new_version=80,
    excluded=[75,79],bounds=[-3.,4.],ridge_mse=1.,sum_weights=1.,
    interpretation='Selection on own official aggregate scores; CV is reused development',
    no_fits=True,no_new_version=True,no_upload=True,no_score_imputation=True)


def run():
    assert not (OUT/'result.json').exists()
    previous=read(parent.OUT/'prepared.json'); prior=read(parent.OUT/'result.json')
    assert read(parent.OUT/'independent_solve_audit.json')['status']=='pass'
    m=read(MANIFEST)
    assert m['state']=='scored' and m['score']==m['official_result']['score']
    assert m['official_result']['status']=='Succeeded' and m['official_result']['used_pairs']==344841
    assert m['sha256']==m['remote_sha256']==art(ROOT/m['file'])['sha256']
    cvrec=read(verify(m['cv_record'])); cvpath=verify(cvrec['scored'])
    ranking=pd.read_parquet(verify(previous['ranking']))
    held=pd.read_parquet(verify(previous['cv']))
    new=pd.read_parquet(ROOT/m['file'])
    validation=pd.read_parquet(cvpath).set_index('mvt_id').loc[held.mvt_id]
    assert validation.index.is_unique
    np.testing.assert_array_equal(ranking.mvt_id,new.MVT_ID_mvt)
    for field in ['month','day','y']:
        np.testing.assert_array_equal(held[field],validation[field])
    ranking['v80']=new.TAXITIME_SEC_mvt.to_numpy(float)
    held['v80']=validation.three_tabm_runway.to_numpy(float)
    assert np.equal(held.v80,np.rint(np.maximum(0,held.v80))).all()
    versions=previous['versions']+[80]
    assert len(set(versions))==len(versions)==52 and not set(DESIGN['excluded'])&set(versions)
    scores=dict(previous['known_scores'])
    scores.update({v:r['score'] for v,r in prior['official'].items()})
    scores['80']=m['score']
    live=parent.transport.live.fetch_leaderboard()
    ours={r['filename']:r for r in live if r['teamName']==parent.transport.live.TEAM}
    for v in versions:
        r=ours[f'unique-rose_v{v}.parquet']
        assert r['usedPairs']==344841 and r['score']==scores[str(v)]
    cols=[f'v{v}' for v in versions]; s=np.array([scores[str(v)] for v in versions])
    solution,raw=parent.base.geometry.solve(ranking[cols].to_numpy(float),s,versions,-3.,4.,1.)
    w=np.array([solution['weights'][str(v)] for v in versions])
    submitted=np.rint(np.maximum(raw,0))
    values=np.rint(np.maximum(held[cols].to_numpy(float)@w,0))
    f=held[['mvt_id','month','day','airport','y']].copy()
    f['control']=held.v80; f['pred']=values
    months={month:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['control','pred']}
            for month,g in f.groupby('month')}
    pair=parent.base.legacy.ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.pred-f.y)**2)
    OUT.mkdir(parents=True,exist_ok=True)
    ranking.to_parquet(OUT/'ranking_basis.parquet',index=False)
    held.to_parquet(OUT/'cv_basis.parquet',index=False)
    f.to_parquet(OUT/'diagnostic_cv.parquet',index=False)
    pd.DataFrame(dict(mvt_id=ranking.mvt_id,raw=raw,submitted=submitted)).to_parquet(OUT/'diagnostic_ranking.parquet',index=False)
    with duckdb.connect(config={'threads':1,'memory_limit':'2GB'}) as con:
        for name,path in [('r',OUT/'ranking_basis.parquet'),('o',OUT/'diagnostic_ranking.parquet'),
                ('h',OUT/'cv_basis.parquet'),('c',OUT/'diagnostic_cv.parquet'),('source',ROOT/m['file']),('cv_source',cvpath)]:
            con.read_parquet(str(path)).create_view(name)
        assert con.execute('select count(*) from r full join source on r.mvt_id=source.MVT_ID_mvt where r.v80 is distinct from source.TAXITIME_SEC_mvt').fetchone()[0]==0
        assert con.execute('select count(*) from h full join cv_source s using(mvt_id) where h.v80 is distinct from s.three_tabm_runway or h.y is distinct from s.y or h.month is distinct from s.month or h.day is distinct from s.day').fetchone()[0]==0
        audit=independent.sql_geometry(con,versions,w,s)
        expr=independent.expression(versions,w,'h')
        # Summation order can differ only at an exact half-integer boundary.
        sql=con.execute(f'''select max(abs(c.pred-round_even(greatest(0,({expr})),0))),
            count(*),count(distinct h.mvt_id) from h full join c using(mvt_id)''').fetchone()
        assert sql[0]==0 and sql[1:]==(344419,344419)
        sql_months=con.execute('select month,sqrt(avg(pow(pred-y,2))) from c group by month').fetchall()
        for month,score in sql_months:np.testing.assert_allclose(score,months[month]['pred'],atol=1e-8,rtol=0)
    upper=audit['bounds']['conservative_last_decimal']['serialized_rmse_interval'][1]
    out=dict(design=DESIGN,versions=versions,official_scores=scores,checked_at=parent.transport.now(),
        solution=solution,independent_sql=audit,months=months,cv=float(np.mean([r['pred'] for r in months.values()])),
        paired=pair,current_best=218.8525,conservative_gain=218.8525-upper,
        clears_fixed_quarter_second_gate=upper<=218.8525-.25,
        source_artifacts=[art(p) for p in [Path(__file__),Path(independent.__file__),parent.OUT/'prepared.json',parent.OUT/'result.json',
            parent.OUT/'independent_solve_audit.json',MANIFEST,ROOT/m['file'],cvpath]],
        outputs=[art(p) for p in sorted(OUT.glob('*.parquet'))],historical_cv_caveat=previous['historical_cv_caveat'])
    write_json(OUT/'result.json',out)
    print(json.dumps({k:out[k] for k in ['cv','paired','conservative_gain','clears_fixed_quarter_second_gate']}),flush=True)
    print(json.dumps(dict(projected_rmse=solution['projected_rmse'],conservative_upper=upper)),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
