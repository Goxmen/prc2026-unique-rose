"""Tiny synthetic-only checks; never opens any challenge prediction or target."""
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import duckdb

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'VALIDATION/tools/audit_portable_v80_distill_20261013.py'
spec=importlib.util.spec_from_file_location('student_audit',SOURCE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def check():
    rng=np.random.default_rng(17);x=rng.normal(size=(320,28));x[:,3]=x[:,2]
    teacher=x@rng.normal(size=28)+rng.normal(size=320)
    beta=np.linalg.solve(x.T@x/len(x)+np.eye(28),x.T@teacher/len(x))
    arithmetic=a.ridge_check(x,teacher,beta)
    try:a.ridge_check(x,teacher,beta+np.r_[.01,np.zeros(27)])
    except AssertionError:pass
    else:raise AssertionError('Corrupt coefficient was not rejected')
    # Match bounds, negative lower exclusion, strict edges, tie to lower,
    # missing witnesses, negative raw and exact nearest-even half ties.
    f=pd.DataFrame(dict(mvt_id=np.arange(11),raw=[-.5,.5,1.5,2.5,200.,200.,100.,3710.,10000.,3610.,10.]))
    d=pd.DataFrame(dict(mvt_id=f.mvt_id,has_id=[False]*8+[True,True,False],nm_gap=[np.nan]*8+[4000.,4000.,np.nan]))
    w=pd.DataFrame(dict(mvt_id=f.mvt_id,recovered_gap=[np.nan]*4+[100.,3700.,3706.,3710.,np.nan,np.nan,3616.]))
    expected=[]
    for r,s,t in zip(f.itertuples(),d.itertuples(),w.itertuples()):
        v=max(0.,r.raw)
        if s.has_id and np.isfinite(s.nm_gap):v=max(0.,min(max(r.raw,s.nm_gap-3606),s.nm_gap+3606))
        if np.isfinite(t.recovered_gap):
            lo,hi=t.recovered_gap-3606,t.recovered_gap+3606
            if lo<v<hi:v=lo if lo>=0 and v-lo<=hi-v else hi
        expected.append(round(max(0.,v)))
    with duckdb.connect(config={'threads':1}) as c:
        for name,frame in [('f',f),('d',d),('w',w)]:c.register(name,frame)
        got=c.execute(f'SELECT {a.constraint_sql("f.raw")} AS v FROM f JOIN d ON f.mvt_id=d.mvt_id LEFT JOIN w ON f.mvt_id=w.mvt_id ORDER BY f.mvt_id').fetchnumpy()['v']
    np.testing.assert_array_equal(got,expected)
    # Exercise the complete raw/metadata/constraint query, including all aliases
    # and null semantics, with temporary synthetic parquet inputs only.
    stamps=pd.to_datetime(['2025-01-03']*6+['2025-07-04']*5)
    f['month']=stamps.strftime('%Y-%m');f['day']=stamps.strftime('%Y-%m-%d');f['airport']='LIRF'
    f['submitted']=got;f['v82']=got;f['v80']=got+3;f['y']=np.arange(11)+1.25
    anchor=f[['mvt_id','month','day','airport','submitted','v80']].copy()
    raw=pd.DataFrame(dict(MVT_ID_mvt=f.mvt_id,PHASE_mvt='DEP',ADEP_mvt='LIRF',MVT_TIME_UTC_mvt=stamps,
                         LOBT_flt=stamps-pd.to_timedelta(d.nm_gap,unit='s'),
                         FLIGHT_ID_mvt=[str(i) if h else None for i,h in enumerate(d.has_id)],TAXITIME_SEC_mvt=f.y))
    (ROOT/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='distill_audit_fixture_',dir=ROOT/'tmp') as folder:
        records=[]
        for month in a.MONTHS:
            path=Path(folder)/f'training_{month}-01_synthetic.parquet'
            raw.loc[f.month.eq(month)].to_parquet(path,index=False);records.append(a.art(path))
        count=a.COUNTS['cv'];a.COUNTS['cv']=len(f)
        try:
            with duckdb.connect(config={'threads':1}) as c:
                c.execute("SET TimeZone='UTC'")
                full_sql=a.sql_check(c,dict(sources=records),'cv',f,anchor,d,w)
        finally:a.COUNTS['cv']=count
    # SQL daily sums + independent row-index resampling match ordinary day
    # loss resampling; unequal day sizes and unsorted day order are deliberate.
    rows=pd.DataFrame(dict(month=np.repeat(a.MONTHS,8),day=['03','01','03','02','02','02','01','04']*2))
    for k in ['v82','v80','submitted']:rows[k]=rng.uniform(1,8,len(rows))
    with duckdb.connect(config={'threads':1}) as c:
        c.register('rows',rows)
        days=c.execute('''SELECT month,day,count(*) AS n,sum(v82) AS v82,sum(v80) AS v80,
            sum(submitted) AS submitted,min(ord) AS first_position FROM
            (SELECT *,row_number() OVER() AS ord FROM rows) GROUP BY month,day ORDER BY month,first_position''').fetchdf()
    metrics={m:{k:float(np.sqrt(rows.loc[rows.month.eq(m),k].mean())) for k in ['v82','v80','submitted']} for m in a.MONTHS}
    paired=a.bootstrap(days,metrics)
    for reference in ['v82','v80']:
        random=np.random.default_rng(0);samples=np.zeros(2000)
        for m in a.MONTHS:
            g=rows[rows.month.eq(m)];groups=[g[g.day.eq(day)] for day in g.day.unique()]
            picks=random.integers(0,len(groups),size=(2000,len(groups)))
            for i,draw in enumerate(picks):
                n=sum(len(groups[j]) for j in draw)
                samples[i]+=(np.sqrt(sum(groups[j].submitted.sum() for j in draw)/n)-np.sqrt(sum(groups[j][reference].sum() for j in draw)/n))/2
        a.close(np.percentile(samples,[2.5,97.5]),paired[reference]['ci95'])
    result=dict(status='pass',source=a.art(Path(__file__)),auditor=a.art(SOURCE),synthetic_only=True,
                ridge=arithmetic,corruption_rejected=True,constraint_rows=len(f),full_SQL=full_sql,independent_bootstrap_reps=2000)
    path=ROOT/'VALIDATION/reports/cohort_learning_20261013/v80_distillation_auditor_fixture.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':check()
