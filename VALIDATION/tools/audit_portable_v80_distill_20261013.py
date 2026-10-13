"""Independent fixed v80 student arithmetic and saved-result audit.

CV runs only after four immutable coefficient/prediction artifacts, independent
input-scope audit and scoring exist. No producer estimator, constraint or scorer
is imported. Cholesky independently checks the fixed ridge normal equations;
SQL checks raw metadata/truth, full-row serialization and daily loss sums.
No estimator selection, inference of base models, uploads or calibration fits.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_v80_teacher_distillation'
PARENT=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
DESIGN=ROOT/'VALIDATION/reports/cohort_learning_20261013/v80_distillation_fixed_design.json'
VERSIONS=[1,4,8,12,13,14,15,16,17,18,19,20,21,30,33,34,37,38,45,46,54,56,57,58,60,61,64,70]
MONTHS=['2025-01','2025-07']
FOLDS={'2025-01':[f'2025-{m:02d}' for m in range(3,12)],
       '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
COUNTS={'cv':344419,'old':344841,'final':670790}


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):h.update(part)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path']
    assert art(path)=={k:record[k] for k in ['path','sha256']},path
    if 'bytes' in record:assert path.stat().st_size==record['bytes']
    return path
def close(a,b,atol=1e-8,rtol=0):np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
def aligned(record,ids,columns,id_column='mvt_id'):
    f=pd.read_parquet(verify(record),columns=[id_column,*columns])
    assert len(f)==len(ids) and f[id_column].is_unique and f[id_column].notna().all()
    assert set(f[id_column])==set(ids)
    return f.set_index(id_column).loc[ids].reset_index()


def ridge_check(x,target,beta):
    """Different factorization; fixed tolerances certify, never alter, estimator."""
    x=np.asarray(x,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    beta=np.asarray(beta,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==len(beta)==28 and len(x)==len(target)>0
    assert np.isfinite(x).all() and np.isfinite(target).all() and np.isfinite(beta).all()
    gram=np.einsum('ni,nj->ij',x,x,optimize=False)/len(x)+np.eye(28)
    rhs=np.einsum('ni,n->i',x,target,optimize=False)/len(x)
    independently_solved=cho_solve(cho_factor(gram,lower=True),rhs)
    scale=np.linalg.norm(gram,np.inf)*np.linalg.norm(beta,np.inf)+np.linalg.norm(rhs,np.inf)
    residual=float(np.linalg.norm(gram@beta-rhs,np.inf)/max(scale,1.))
    assert residual<=5e-12,residual
    close(beta,independently_solved,atol=5e-8,rtol=5e-8)
    eigen=np.linalg.eigvalsh(gram)
    assert eigen[0]>=1-1e-7*max(1.,eigen[-1])
    condition=float(eigen[-1]/eigen[0]);assert condition*np.finfo(float).eps<1e-4
    return dict(relative_normal_residual=residual,condition_number=condition,
                cholesky_coefficient_max_error=float(abs(beta-independently_solved).max()),
                ridge=1.,intercept=False,rows=len(x),independent_gram_einsum=True)


def constraint_sql(expression):
    bounded=f'''greatest(0.,CASE WHEN d.has_id AND isfinite(d.nm_gap)
        THEN greatest(0.,least(greatest(({expression}),d.nm_gap-3606.),d.nm_gap+3606.))
        ELSE ({expression}) END)'''
    lo,hi='(w.recovered_gap-3606.)','(w.recovered_gap+3606.)'
    excluded=f'''CASE WHEN isfinite(w.recovered_gap) AND ({bounded})>{lo} AND ({bounded})<{hi}
        THEN CASE WHEN {lo}>=0 AND ({bounded})-{lo}<={hi}-({bounded}) THEN {lo} ELSE {hi} END
        ELSE ({bounded}) END'''
    return f'round_even(greatest(0.,({excluded})),0)'


def bootstrap(days,metrics):
    results={}
    for reference in ['v82','v80']:
        rng=np.random.Generator(np.random.PCG64(0));draws=np.zeros(2000)
        for month in MONTHS:
            g=days[days.month.eq(month)]
            n=g.n.to_numpy();a=g[reference].to_numpy();b=g.submitted.to_numpy()
            selection=rng.integers(0,len(g),size=(2000,len(g)))
            denom=n[selection].sum(1)
            draws+=(np.sqrt(b[selection].sum(1)/denom)-np.sqrt(a[selection].sum(1)/denom))/2
        interval=np.percentile(draws,[2.5,97.5])
        change=float(np.mean([metrics[m]['submitted']-metrics[m][reference] for m in MONTHS]))
        results[reference]=dict(change=change,ci95=interval.tolist(),
            decision='improvement' if interval[1]<0 else 'regression' if interval[0]>0 else 'inconclusive')
    return results


def spec_frame(spec):
    path=verify(spec['artifact']);schema=pq.ParquetFile(path).schema_arrow
    assert hashlib.sha256(str(schema).encode()).hexdigest()==spec['schema_sha256']
    f=pd.read_parquet(path,columns=spec['columns'])
    assert len(f)==spec['rows'] and list(f)==spec['columns']
    return f


def reorder(f,ids,key='mvt_id',full=True):
    assert f[key].notna().all() and f[key].is_unique and pd.Index(ids).isin(f[key]).all()
    if full:assert len(f)==len(ids) and set(f[key])==set(ids)
    return f.set_index(key).loc[ids].reset_index()


def provenance():
    m=read(OUT/'identity.json');design=read(DESIGN)
    assert m['auditor']==art(Path(__file__)) and m['design']==art(DESIGN)
    assert m['versions']==design['basis']['versions']==VERSIONS
    assert design['estimator']['ridge_MSE_seconds_squared']==1. and not design['estimator']['intercept']
    for record in m['sources']:verify(record)
    for key in ['source','review','fixture','input_auditor','teacher_upload','teacher_remote']:verify(m[key])
    assert read(verify(m['review']))['status']==read(verify(m['fixture']))['status']=='pass'
    for view in COUNTS:
        r,a=read(PARENT/view/'result.json'),read(PARENT/view/'audit.json')
        assert a['status']=='pass' and a['result']==art(PARENT/view/'result.json')
        assert a['identity']==r['identity']==art(PARENT/'identity.json')
        assert a['prediction']==r['prediction']==m['specs'][view]['anchor']['artifact']
        for path in [PARENT/view/'result.json',PARENT/view/'audit.json',PARENT/'identity.json']:
            assert art(path) in m['sources']
    return m


def inputs(m,view):
    s=m['specs'][view];anchor=spec_frame(s['anchor']);ids=anchor.mvt_id.to_numpy()
    assert len(ids)==anchor.mvt_id.nunique()==COUNTS[view] and anchor.mvt_id.notna().all()
    z=anchor.submitted.to_numpy(float)
    if view in ['cv','old']:
        b=reorder(spec_frame(s['basis']),ids)
        if view=='cv':
            for key in ['month','day','airport']:np.testing.assert_array_equal(b[key],anchor[key])
        matrix=b[[f'v{v}' for v in VERSIONS]].to_numpy(float)
    else:matrix=np.empty((len(ids),28))
    if view!='cv':
        for j,version in enumerate(VERSIONS):
            c=m['components'][str(version)];f=reorder(spec_frame(c[view]),ids,c['id_column'])
            value=f[c['value_column']].to_numpy(float)
            if view=='old':np.testing.assert_array_equal(matrix[:,j],value)
            else:matrix[:,j]=value
    for values in [matrix,z]:
        assert np.isfinite(values).all() and (values>=0).all()
        np.testing.assert_array_equal(values,np.rint(values))
    teacher=None
    if view=='cv':
        teacher=reorder(spec_frame(s['teacher']),ids).three_tabm_runway.to_numpy(float)
        np.testing.assert_array_equal(teacher,anchor.v80)
    elif view=='old':
        assert s['teacher']['artifact']==m['teacher_upload']
        teacher=reorder(spec_frame(s['teacher']),ids,'MVT_ID_mvt').TAXITIME_SEC_mvt.to_numpy(float)
    if teacher is not None:
        assert np.isfinite(teacher).all() and (teacher>=0).all()
        np.testing.assert_array_equal(teacher,np.rint(teacher))
    d=reorder(spec_frame(s['raw']),ids,full=False)
    np.testing.assert_array_equal(d.month,anchor.month)
    w=spec_frame(s['matches']);w=w.loc[w.accepted].copy()
    assert w.mvt_id.is_unique
    if view=='cv':
        w=w.merge(anchor[['mvt_id','month']],on='mvt_id',how='inner',validate='one_to_one')
        w=w.loc[[r.donor_month in FOLDS[r.month]+[r.month] for r in w.itertuples()]]
    return anchor,matrix-z[:,None],teacher,d,w


def allocation(record,stage):
    value=read(verify(record))
    assert value['authority']=='/root' and value['identity']==art(OUT/'identity.json')
    assert value['state']==f'released_for_distillation_{stage}'
    started=read(OUT/stage/'started.json')
    assert started['allocation']==record and started['identity']==value['identity']


def coefficients(record,x,target,ids,held_ids,outer,block):
    c=read(verify(record))
    assert c['identity']==art(OUT/'identity.json') and c['outer']==outer and c['block']==block
    assert c['columns']==VERSIONS and c['ridge']==1. and c['intercept'] is False
    assert c['train_rows']==len(ids) and c['held_rows']==len(held_ids)
    for label,values in [('train',ids),('held',held_ids)]:
        assert c[label+'_ids_sha256']==hashlib.sha256(np.asarray(values,dtype=np.int64).tobytes()).hexdigest()
    assert np.isfinite(c['normal_equation_max_abs']) and c['normal_equation_max_abs']>=0
    report=ridge_check(x,target,c['beta']);report.update(coefficient=record,outer=outer,block=block)
    return np.asarray(c['beta'],float),report


def raw_paths(m,view):
    names=[f'training_{month}-01_' for month in MONTHS] if view=='cv' else [
        'ranking.parquet' if view=='old' else 'final_ranking.parquet']
    paths=[]
    for name in names:
        records=[r for r in m['sources'] if Path(r['path']).name.startswith(name)]
        assert len(records)==1,(name,records)
        paths.append(str(verify(records[0])))
    return paths


def sql_check(con,m,view,f,anchor,d,w):
    assert len(f)==f.mvt_id.nunique()==COUNTS[view] and f.mvt_id.notna().all()
    np.testing.assert_array_equal(f.mvt_id,anchor.mvt_id)
    assert np.isfinite(f[['raw','submitted','v82']]).all().all()
    for name,frame in [('f',f),('a',anchor),('d',d),('w',w)]:con.register(name,frame)
    # Projection excludes every raw label column for old/final; CV truth opens
    # here only after input-scope, inference and score receipts were checked.
    con.read_parquet(raw_paths(m,view)).create_view('raw_source')
    truth=',TAXITIME_SEC_mvt AS y' if view=='cv' else ''
    con.execute(f'''CREATE TEMP VIEW r AS SELECT MVT_ID_mvt AS mvt_id,PHASE_mvt AS phase,ADEP_mvt AS airport,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day,
        (FLIGHT_ID_mvt IS NOT NULL) AS has_id,epoch(MVT_TIME_UTC_mvt-LOBT_flt) AS nm_gap {truth} FROM raw_source''')
    truth_condition='OR f.y IS DISTINCT FROM r.y' if view=='cv' else ''
    control='OR f.v80 IS DISTINCT FROM a.v80' if view=='cv' else ''
    n,u,bad=con.execute(f'''SELECT count(*),count(distinct f.mvt_id),count(*) FILTER(WHERE
        a.mvt_id IS NULL OR d.mvt_id IS NULL OR r.mvt_id IS NULL OR r.phase!='DEP'
        OR f.month IS DISTINCT FROM a.month OR f.month IS DISTINCT FROM r.month
        OR f.day IS DISTINCT FROM a.day OR f.day IS DISTINCT FROM r.day
        OR f.airport IS DISTINCT FROM a.airport OR f.airport IS DISTINCT FROM r.airport
        OR d.has_id IS DISTINCT FROM r.has_id OR d.nm_gap IS DISTINCT FROM r.nm_gap
        OR f.v82 IS DISTINCT FROM a.submitted
        OR f.submitted IS DISTINCT FROM {constraint_sql('f.raw')}
        OR f.v82 IS DISTINCT FROM {constraint_sql('CAST(f.v82 AS DOUBLE)')}
        {truth_condition} {control}) FROM f LEFT JOIN a ON f.mvt_id=a.mvt_id
        LEFT JOIN d ON f.mvt_id=d.mvt_id LEFT JOIN w ON f.mvt_id=w.mvt_id LEFT JOIN r ON f.mvt_id=r.mvt_id''').fetchone()
    assert n==u==COUNTS[view] and bad==0,(n,u,bad)
    return dict(rows=n,full_raw_ID_calendar_constraint_and_zero_beta_checks=True)


def audit_cv():
    start=time.monotonic();assert not (OUT/'cv/audit.json').exists();m=provenance()
    inf,pre,score=[read(OUT/'cv'/name) for name in ['inference.json','input_audit.json','score.json']]
    ident=art(OUT/'identity.json')
    assert inf['identity']==pre['identity']==score['identity']==ident and pre['status']=='pass'
    assert pre['inference']==score['inference']==art(OUT/'cv/inference.json')
    assert score['input_audit']==art(OUT/'cv/input_audit.json')
    assert inf['all_four_fits_before_truth'] and inf['no_taxi_labels_read'] and inf['rows']==COUNTS['cv']
    assert inf['coefficients']==inf['fit_receipts'] and len(inf['coefficients'])==4
    assert len({r['path'] for r in inf['coefficients']})==4
    allocation(inf['allocation'],'cv');allocation(score['allocation'],'score')
    anchor,x,teacher,d,w=inputs(m,'cv');ids=anchor.mvt_id.to_numpy()
    saved=pd.read_parquet(verify(inf['prediction']));f=pd.read_parquet(verify(score['scored']))
    pd.testing.assert_frame_equal(saved,f.drop(columns='y'))
    np.testing.assert_array_equal(saved.mvt_id,ids)
    reconstructed=np.empty(len(ids));seen=np.zeros(len(ids),int);fits=[]
    for index,(month,block) in enumerate((month,block) for month in MONTHS for block in ['A','B']):
        number=pd.to_datetime(anchor.day).dt.day.to_numpy();outer=anchor.month.eq(month).to_numpy()
        train=np.flatnonzero(outer & (number>=17 if block=='A' else number<=14))
        held=np.flatnonzero(outer & (number<=15 if block=='A' else number>=16))
        assert len(train)>0 and len(held)>0 and not np.intersect1d(train,held).size
        beta,report=coefficients(inf['coefficients'][index],x[train],teacher[train]-anchor.submitted.to_numpy()[train],ids[train],ids[held],month,block)
        fits.append(report);seen[held]+=1
        # Independent row-wise reduction, avoiding the producer's GEMV path.
        reconstructed[held]=anchor.submitted.to_numpy()[held]+np.sum(x[held]*beta,axis=1)
        assert (saved.iloc[held].block==block).all()
    np.testing.assert_array_equal(seen,np.ones(len(ids),int));close(saved.raw,reconstructed,atol=1e-7)
    np.testing.assert_array_equal(saved.v80,teacher)
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        checked=sql_check(con,m,'cv',f,anchor,d,w)
        columns=['v82','v80','submitted'];metrics={};fidelity={}
        for month in MONTHS:
            values=con.execute('SELECT '+','.join(f'sqrt(avg(pow(CAST({k} AS DOUBLE)-y,2)))' for k in columns)+
                ',sqrt(avg(pow(raw-v80,2))),sqrt(avg(pow(CAST(v82 AS DOUBLE)-v80,2))) FROM f WHERE month=?',[month]).fetchone()
            metrics[month]=dict(zip(columns,values[:3]))
            fidelity[month]=dict(student_RMSE=values[3],anchor_RMSE=values[4])
            for k,v in metrics[month].items():close(v,score['months'][month][k])
            for k,v in fidelity[month].items():
                close(v,inf['teacher_fidelity'][month][k]);close(v,score['teacher_fidelity'][month][k])
        cv={k:float(np.mean([v[k] for v in metrics.values()])) for k in columns}
        for k,v in cv.items():close(v,score['cv'][k])
        days=con.execute('''SELECT month,day,count(*) AS n,
            sum(pow(CAST(v82 AS DOUBLE)-y,2)) AS v82,sum(pow(CAST(v80 AS DOUBLE)-y,2)) AS v80,
            sum(pow(CAST(submitted AS DOUBLE)-y,2)) AS submitted,min(ordinal) AS first_position FROM
            (SELECT *,row_number() OVER() AS ordinal FROM f) GROUP BY month,day ORDER BY month,first_position''').fetchdf()
    paired=bootstrap(days,metrics)
    for key,value in paired.items():
        close(value['change'],score['paired'][key]['change']);close(value['ci95'],score['paired'][key]['ci95'])
        assert value['decision']==score['paired'][key]['decision']
    gate=bool(all(v['student_RMSE']<v['anchor_RMSE'] for v in fidelity.values()) and cv['submitted']-cv['v82']<=-.10
              and all(v['submitted']<v['v82'] for v in metrics.values()) and paired['v82']['ci95'][1]<0)
    assert gate==score['gate'] and score['reused_development'] and score['unseen_month_transfer_unvalidated']
    result=dict(status='pass',source=art(Path(__file__)),identity=ident,score=art(OUT/'cv/score.json'),inference=art(OUT/'cv/inference.json'),
        input_audit=art(OUT/'cv/input_audit.json'),prediction=inf['prediction'],scored=score['scored'],rows=len(f),fits=fits,
        months=metrics,cv=cv,paired=paired,teacher_fidelity=fidelity,gate=gate,raw_SQL=checked,
        reused_development=True,unseen_month_transfer_unvalidated=True,no_new_fit_or_base_inference=True,seconds=time.monotonic()-start)
    write(OUT/'cv/audit.json',result)


def audit_production():
    start=time.monotonic();assert not (OUT/'production/audit.json').exists();m=provenance()
    result=read(OUT/'production/result.json');ident=art(OUT/'identity.json')
    assert result['identity']==ident and result['solves']==1 and result['no_taxi_labels_read'] and not result['exact_v80_claim']
    score=read(verify(result['score']));audit=read(verify(result['cv_audit']))
    assert audit['status']=='pass' and audit['score']==result['score'] and score['gate'] and audit['gate']
    assert audit['identity']==score['identity']==ident
    allocation(result['allocation'],'production')
    a,x,teacher,d,w=inputs(m,'old');ids=a.mvt_id.to_numpy()
    beta,fit=coefficients(result['model'],x,teacher-a.submitted.to_numpy(float),ids,ids,'2026-all','production')
    views={'cv':dict(rows=COUNTS['cv'],months={month:score['months'][month]['submitted'] for month in MONTHS},
                     cv=score['cv']['submitted'],score=result['score'],audit=result['cv_audit'])}
    assert result['views']['cv']==dict(rows=COUNTS['cv'],prediction=score['scored'])
    for view in ['old','final']:
        if view=='final':a,x,teacher,d,w=inputs(m,view);ids=a.mvt_id.to_numpy()
        spec=result['views'][view];f=pd.read_parquet(verify(spec['prediction']))
        assert spec['rows']==COUNTS[view]
        close(f.raw,a.submitted.to_numpy(float)+np.sum(x*beta,axis=1),atol=1e-7)
        with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
            con.execute("SET TimeZone='UTC'");checked=sql_check(con,m,view,f,a,d,w)
        template=pq.read_table(verify(m['specs'][view]['template']['artifact']));candidate=pq.read_table(verify(spec['candidate']))
        assert candidate.schema==template.schema and len(candidate)==COUNTS[view]
        for key in template.column_names:
            if key!='TAXITIME_SEC_mvt':assert candidate[key].equals(template[key]),key
        np.testing.assert_array_equal(candidate['MVT_ID_mvt'].to_numpy(),ids)
        np.testing.assert_array_equal(candidate['TAXITIME_SEC_mvt'].to_numpy(),f.submitted)
        assert set(f.month)==({'2026-01','2026-07'} if view=='old' else {'2026-01','2026-02','2026-06','2026-07'})
        views[view]=dict(rows=len(f),prediction=spec['prediction'],candidate=spec['candidate'],raw_SQL=checked,template_all_other_columns_exact=True)
    value=dict(status='pass',source=art(Path(__file__)),identity=ident,result=art(OUT/'production/result.json'),model=result['model'],fit=fit,
        cv_audit=result['cv_audit'],views=views,no_raw_taxi_labels_decoded=True,exact_v80_claim=False,
        unseen_month_transfer_unvalidated=True,no_upload=True,seconds=time.monotonic()-start)
    write(OUT/'production/audit.json',value)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=value['status'],audit=art(path),gate=value.get('gate'),seconds=value['seconds'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cv','production'])
    with threadpool_limits(1):
        {'cv':audit_cv,'production':audit_production}[parser.parse_args().stage]()
