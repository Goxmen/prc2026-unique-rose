"""Independent all-row public-stack SQL audit, with bounded posterior checks.

No producer correction/posterior apply is called. Immutable support primitives
are reused only by a fixed, input-selected convolution sample, explicitly named.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
AIRPORTS=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
ZONES=['Europe/Berlin','Europe/Berlin','Europe/London','Europe/Amsterdam','Europe/Madrid','Europe/Madrid','Europe/Paris','Europe/Rome','Europe/Zurich','Europe/Istanbul']
FOLDS={'2025-01':[f'2025-{m:02d}' for m in range(3,12)],'2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
COUNTS={'cv':344419,'old':344841,'final':670790}
RAW_SHA={'old':'9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86','final':'4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f'}
WEIGHT=.4032872515123992


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def verify(a):
    p=ROOT/a['path'];assert art(p)=={k:a[k] for k in ['path','sha256']},p
    return p

def frame(a,ids,full=False):
    f=pd.read_parquet(verify(a));assert f.mvt_id.is_unique and f.mvt_id.notna().all()
    if full:assert len(f)==len(ids) and set(f.mvt_id)==set(ids)
    return f.set_index('mvt_id').loc[ids].reset_index()

def close(a,b,tol=1e-8):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)

def airport_case(values,alias='g'):
    return 'CASE '+alias+'.airport '+ ' '.join(f"WHEN '{a}' THEN {repr(float(v))}" for a,v in zip(AIRPORTS,values))+' END'

def constraint(expression):
    bounded=f'''greatest(0.,CASE WHEN d.has_id AND isfinite(d.nm_gap)
        THEN greatest(0.,least(greatest(({expression}),d.nm_gap-3606.),d.nm_gap+3606.)) ELSE ({expression}) END)'''
    lo,hi='(w.recovered_gap-3606.)','(w.recovered_gap+3606.)'
    return f'''CASE WHEN isfinite(w.recovered_gap) AND ({bounded})>{lo} AND ({bounded})<{hi}
        THEN CASE WHEN {lo}>=0 AND ({bounded})-{lo}<={hi}-({bounded}) THEN {lo} ELSE {hi} END ELSE ({bounded}) END'''


def provenance(view):
    identity=read(OUT/'identity.json');spec=identity['specs'][view];result=read(OUT/view/'result.json')
    assert identity['daily_weight']==WEIGHT and identity['no_fit'] and identity['no_solve']
    assert art(Path(__file__)) in identity['sources']
    for a in identity['sources']:verify(a)
    assert result['identity']==art(OUT/'identity.json') and result['view']==view and result['rows']==COUNTS[view]
    for key in ['prediction','monthly_cells','daily_cells','allocation']:verify(result[key])
    allocation=read(verify(result['allocation']))
    assert allocation['identity']==art(OUT/'identity.json') and allocation['authority']=='/root' and allocation['view']==view
    base=read(verify(identity['base_result']));audit=read(verify(identity['base_audit']))
    assert audit['status']=='pass' and audit['result']==identity['base_result'] and audit['identity']==base['identity']
    assert base['predictions'][view]==spec['base']==audit['predictions'][view]
    if view!='cv':
        for earlier in (['cv'] if view=='old' else ['cv','old']):
            previous=read(OUT/earlier/'audit.json')
            assert previous['status']=='pass' and previous['identity']==art(OUT/'identity.json')
            assert previous['result']==art(OUT/earlier/'result.json')
    # Original parameter bytes must appear in their historical selection records.
    if view=='cv':
        historical=ROOT/'VALIDATION/runs/v3_20261011_daily_heavy_tail_stack/identity.json'
        sources=read(historical)['sources']
        assert spec['daily_parameters'] in sources and spec['monthly_parameters'] in sources
        original=ROOT/'VALIDATION/runs/v3_20261011_public_daily_taxi/result.json'
        assert spec['quality_parameters'] in read(original)['artifacts']
    else:
        for key in ['daily_parameters','monthly_parameters']:
            path=verify(spec[key]);manifest=read(path.parent/'manifest.json')
            assert manifest['sources'][spec[key]['path']]==spec[key]['sha256']
        path=verify(spec['quality_parameters'])
        assert spec['quality_parameters'] in read(path.parent/'result.json')['artifacts']
        assert read(path)['allowed']==[f'2025-{m:02d}' for m in range(1,13)]
    return identity,spec,result


def compare_cells(actual,expected,keys,columns):
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    a=actual.set_index(keys).sort_index();b=expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index,b.index)
    for k in columns:
        if a[k].dtype.kind in 'biuf':close(a[k].astype(float),b[k].astype(float))
        else:pd.testing.assert_series_equal(a[k],b[k],check_dtype=False,check_names=False)
    return len(a)


def raw_check(con,identity,spec,view,f,d,m,i):
    paths=[]
    if view=='cv':
        for month in FOLDS:
            found=[a for a in identity['sources'] if Path(a['path']).name.startswith('training_'+month+'-01_')]
            assert len(found)==1;paths.append(str(verify(found[0])))
    else:
        name='ranking.parquet' if view=='old' else 'final_ranking.parquet'
        found=[a for a in identity['sources'] if Path(a['path']).name==name]
        assert len(found)==1 and found[0]['sha256']==RAW_SHA[view];paths=[str(verify(found[0]))]
    con.read_parquet(paths).create_view('raw')
    zones='CASE r.ADEP_mvt '+ ' '.join(f"WHEN '{a}' THEN '{z}'" for a,z in zip(AIRPORTS,ZONES))+' END'
    truth='f.y IS DISTINCT FROM r.TAXITIME_SEC_mvt' if view=='cv' else 'r.TAXITIME_SEC_mvt IS NOT NULL'
    n,u,bad=con.execute(f'''SELECT count(*),count(distinct f.mvt_id),sum(CASE WHEN
        r.PHASE_mvt!='DEP' OR f.airport IS DISTINCT FROM r.ADEP_mvt
        OR f.month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m')
        OR f.day IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m-%d')
        OR d.gap IS DISTINCT FROM epoch(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt)
        OR d.nm_gap IS DISTINCT FROM epoch(r.MVT_TIME_UTC_mvt-r.LOBT_flt)
        OR d.has_id IS DISTINCT FROM (r.FLIGHT_ID_mvt IS NOT NULL)
        OR m.eligible IS DISTINCT FROM (r.STAND_mvt IS NOT NULL AND r.RUNWAY_mvt IS NOT NULL)
        OR m.local_month IS DISTINCT FROM strftime(timezone(({zones}),r.MVT_TIME_UTC_mvt),'%Y-%m')
        OR i.local_day IS DISTINCT FROM strftime(timezone(({zones}),r.MVT_TIME_UTC_mvt),'%Y-%m-%d')
        OR i.runway IS DISTINCT FROM r.RUNWAY_mvt OR {truth} THEN 1 ELSE 0 END)
        FROM f JOIN raw r ON f.mvt_id=r.MVT_ID_mvt JOIN d USING(mvt_id) JOIN m USING(mvt_id) JOIN i USING(mvt_id)''').fetchone()
    assert n==u==COUNTS[view] and bad==0,(n,u,bad)
    return n


def cell_checks(con,spec,view,f,m,i):
    monthly_source=pd.read_csv(next(ROOT/a['path'] for a in read(OUT/'identity.json')['sources']
        if Path(a['path']).name==f'taxi_out_additional_time_{2025 if view=="cv" else 2026}.csv'))
    con.register('monthly_csv',monthly_source)
    con.execute('''CREATE TEMP VIEW ms AS SELECT APT_ICAO AS airport,
        CAST(YEAR AS VARCHAR)||'-'||lpad(CAST(MONTH_NUM AS VARCHAR),2,'0') AS local_month,
        60.*(TOTAL_REF_TIME_MIN+TOTAL_ADD_TIME_MIN)/TOTAL_REF_NB_FL AS mean_sec,
        (TF-VALID_FL)/TF AS invalid,NO_REF/TF AS noref FROM monthly_csv''')
    daily_source=next(ROOT/a['path'] for a in read(OUT/'identity.json')['sources'] if Path(a['path']).name=='daily_runway.parquet')
    con.read_parquet(str(daily_source)).create_view('daily_source')
    assert con.execute("SELECT bool_and(phase='DEP') FROM daily_source").fetchone()[0]
    con.execute('''CREATE TEMP VIEW ds AS SELECT airport,day AS local_day,runway,n AS source_n,
        60.*total_min/n AS source_mean FROM daily_source''')
    for name,keys in [('ms','airport,local_month'),('ds','airport,local_day,runway')]:
        assert con.execute(f'SELECT count(*)=count(DISTINCT ({keys})) FROM {name}').fetchone()[0]
    monthly_parameters=read(verify(spec['monthly_parameters']));quality_parameters=read(verify(spec['quality_parameters']))
    months=[];days=[];errors={}
    for month in sorted(f.month.unique()):
        coef=(monthly_parameters['folds'][month] if view=='cv' else monthly_parameters['full_year'])['coefficients']
        assert len(coef)==12
        con.execute(f'''CREATE OR REPLACE TEMP VIEW mc AS WITH g AS (
            SELECT m.airport,m.local_month,count(*) AS n,avg(CAST(f.D AS DOUBLE)) AS pred_mean
            FROM f JOIN m USING(mvt_id) WHERE f.month='{month}' AND m.eligible AND f.D<=7200 GROUP BY 1,2),
            c AS (SELECT g.*,ms.mean_sec,ms.invalid,ms.noref,
                {airport_case(coef[:10])}+({repr(float(coef[10]))})*ms.invalid+({repr(float(coef[11]))})*ms.noref AS composition
                FROM g LEFT JOIN ms USING(airport,local_month))
            SELECT *,mean_sec+composition-pred_mean AS raw_shift,
                CASE WHEN invalid<.05 AND n>=100 THEN .5*least(120.,greatest(-120.,mean_sec+composition-pred_mean)) ELSE 0. END AS shift
                FROM c''')
        months.append(con.execute(f"SELECT *, '{month}' AS fold FROM mc").fetchdf())
        half=con.execute(f'''SELECT f.mvt_id,greatest(0.,CAST(f.D AS DOUBLE)+CASE WHEN m.eligible AND f.D<=7200 THEN coalesce(mc.shift,0.) ELSE 0. END) AS half
            FROM f JOIN m USING(mvt_id) LEFT JOIN mc USING(airport,local_month) WHERE f.month='{month}' ''').fetchdf().set_index('mvt_id')
        sub=f[f.month.eq(month)];close(half.loc[sub.mvt_id,'half'],sub.half_monthly)
        params=quality_parameters[month] if view=='cv' else quality_parameters
        assert params['allowed']==(FOLDS[month] if view=='cv' else [f'2025-{k:02d}' for k in range(1,13)])
        coef=params['coefficients'];assert len(coef)==20 and set(params['noise'])==set(AIRPORTS)
        con.execute(f'''CREATE OR REPLACE TEMP VIEW dc AS WITH g AS (
            SELECT i.airport,i.local_day,i.runway,count(*) AS n,avg(CAST(f.N AS DOUBLE)) AS mean
            FROM f JOIN i USING(mvt_id) JOIN m USING(mvt_id)
            WHERE f.month='{month}' AND m.eligible AND f.N BETWEEN 0 AND 7200 GROUP BY 1,2,3),
            c AS (SELECT g.*,ds.source_n,ds.source_mean,
                {airport_case(coef[:10])}+{airport_case(coef[10:])}*(1.-CAST(ds.source_n AS DOUBLE)/g.n)/.1 AS composition,
                {airport_case([params['noise'][a] for a in AIRPORTS])} AS noise,40000./g.n+225. AS prior_var
                FROM g LEFT JOIN ds USING(airport,local_day,runway)),
            v AS (SELECT *,coalesce(n>=20 AND source_n>=20 AND CAST(source_n AS DOUBLE)/n BETWEEN .5 AND 1.2 AND isfinite(source_mean),false) AS supported,
                prior_var/(prior_var+noise*noise) AS gain,source_mean+composition-mean AS raw_gap FROM c)
            SELECT *,CASE WHEN supported THEN gain*least(120.,greatest(-120.,raw_gap)) ELSE 0. END AS shift FROM v''')
        days.append(con.execute(f"SELECT *, '{month}' AS fold FROM dc").fetchdf())
        actual=con.execute(f'''SELECT f.mvt_id,CASE WHEN m.eligible AND f.N BETWEEN 0 AND 7200 THEN coalesce(dc.shift,0.) ELSE 0. END AS shift
            FROM f JOIN i USING(mvt_id) JOIN m USING(mvt_id) LEFT JOIN dc USING(airport,local_day,runway)
            WHERE f.month='{month}' ''').fetchdf().set_index('mvt_id')
        error=float(np.max(abs(actual.loc[sub.mvt_id,'shift'].to_numpy()-sub.daily_shift.to_numpy())))
        assert error<=1e-8;errors[month]=error
    monthly_saved=pd.read_parquet(verify(read(OUT/view/'result.json')['monthly_cells']))
    daily_saved=pd.read_parquet(verify(read(OUT/view/'result.json')['daily_cells']))
    nm=compare_cells(monthly_saved,pd.concat(months),['fold','airport','local_month'],['n','pred_mean','mean_sec','invalid','noref','composition','raw_shift','shift'])
    nd=compare_cells(daily_saved,pd.concat(days),['fold','airport','local_day','runway'],['n','mean','source_n','source_mean','composition','noise','prior_var','supported','gain','raw_gap','shift'])
    bad=con.execute('''SELECT count(*) FROM f JOIN i USING(mvt_id)
        LEFT JOIN ms ON f.airport=ms.airport AND substr(i.local_day,1,7)=ms.local_month
        WHERE f.invalid_share IS DISTINCT FROM ms.invalid OR f.quality_eligible IS DISTINCT FROM coalesce(ms.invalid<.05,false)''').fetchone()[0]
    assert bad==0
    return dict(monthly_cells=nm,daily_cells=nd,max_shift_errors=errors)


def posterior_checks(spec,result,view,f,d):
    sys.path.insert(0,str(ROOT/'VALIDATION'))
    import daily_heavy_tail as heavy
    params=read(verify(spec['daily_parameters']))['parameters']
    source_path=next(ROOT/a['path'] for a in read(OUT/'identity.json')['sources']
        if Path(a['path']).name==f'all_pre_departure_delays_{2025 if view=="cv" else 2026}.csv')
    source=pd.read_csv(source_path).set_index(['APT_ICAO','FLT_DATE'])
    days=[];covered=[];samples=[];unsampled=[]
    z=np.linspace(-6,6,41);t5=(1+(z/heavy.SCALE)**2/5)**-3
    close(np.dot(t5,z*z)/sum(t5),np.dot(np.exp(-z*z/2),z*z)/sum(np.exp(-z*z/2)),1e-14)
    for record in result['posterior_receipts']:
        receipt=read(verify(record));month,airport=receipt['month'],receipt['airport']
        assert receipt['identity']==art(OUT/'identity.json') and receipt['view']==view and receipt['base']==spec['base']
        g=f[f.month.eq(month)&f.airport.eq(airport)];inputs=d.set_index('mvt_id').loc[g.mvt_id].reset_index()
        cached=pd.read_parquet(verify(receipt['prediction']))
        np.testing.assert_array_equal(cached.mvt_id,g.mvt_id);np.testing.assert_array_equal(cached.B,g.B)
        np.testing.assert_array_equal(cached.raw_posterior,g.raw_posterior)
        assert len(g)==receipt['rows'];covered.extend(g.mvt_id)
        logs=pd.read_parquet(verify(receipt['days']));assert not logs.duplicated(['airport','day']).any()
        chosen=params[month][airport] if view=='cv' else params[airport]
        eligible_samples=[]
        for day,ix in inputs.groupby('day',sort=True).indices.items():
            r=inputs.iloc[ix];prior=g.B.to_numpy(float)[ix];pred=g.raw_posterior.to_numpy()[ix]
            row=logs[logs.day.eq(day)];assert len(row)==1;info=row.iloc[0];pub=source.loc[(airport,day)]
            active=(r.gap<=3600)&(r.gap>prior);n=int(active.sum());ordinary=r.gap<=3600
            total=60*float(pub.DLY_ALL_PRE_2);count=float(pub.FLT_DEP_IFR_2)
            expected_ordinary=float(np.maximum(r.gap.to_numpy()[ordinary]-prior[ordinary],0).sum())
            sd=np.hypot(chosen['sigma'],chosen['kappa']*(count-len(r)))
            assert info.rows==len(r) and info.active==n and info.long_rows==sum(~ordinary)
            close([info.public_total,info.public_count,info.prior_ordinary,info.measurement_sd],[total,count,expected_ordinary,sd])
            variance=n*22500+n*n*225
            expected=prior.copy()
            if n:
                expected[active]-=(22500+n*225)/max(variance,1)*info.offset
                expected[ordinary&~active]-=.25*n*225/max(variance,1)*info.offset
            close(np.maximum(expected[ordinary],0),pred[ordinary])
            if info.long_rows<=8 and info.grid_size<=5000:eligible_samples.append((day,ix,info,total,count,chosen))
        # Fixed input/size rule, independent of labels/errors: one lexical first
        # nonempty-long day per airport/month, otherwise first ordinary-only day.
        if not eligible_samples:
            unsampled.append(dict(month=month,airport=airport,reason='No day under fixed8-long/5000-grid bound; all source/group/ordinary checks still run'))
            assert len(logs)==inputs.day.nunique();days.append(logs)
            continue
        long_candidates=[x for x in eligible_samples if x[2].long_rows>0]
        day,ix,info,total,count,chosen=(long_candidates or eligible_samples)[0]
        r=inputs.iloc[ix];prior=g.B.to_numpy(float)[ix];pred=g.raw_posterior.to_numpy()[ix]
        supports=[heavy.old.support(float(x.gap),prior[k],bool(x.has_id),float(x.nm_gap),airport,chosen)
            for k,x in enumerate(r.itertuples(index=False)) if x.gap>3600]
        distributions=[]
        for y,w,c in supports:
            assert (w>=0).all() and abs(w.sum()-1)<1e-12
            distributions.append(np.bincount(c,weights=w,minlength=int(max(c))+1))
        aggregate=np.array([1.])
        for v in distributions:aggregate=np.convolve(aggregate,v)
        assert len(aggregate)==info.grid_size
        variance=int(info.active)*22500+int(info.active)**2*225
        offsets=z*max(np.sqrt(variance),1.)
        residual=(total-chosen['bias']-info.prior_ordinary-np.arange(len(aggregate))[:,None]*60-offsets[None,:])/info.measurement_sd
        joint=t5[None,:]*(1+residual*residual/3)**-2
        posterior_offset=(aggregate[:,None]*joint).sum(0)
        offset=float(posterior_offset@offsets/posterior_offset.sum());close(offset,info.offset,2e-6)
        expected=[]
        for k,(y,w,c) in enumerate(supports):
            peer=np.array([1.])
            for j,v in enumerate(distributions):
                if j!=k:peer=np.convolve(peer,v)
            likelihood=joint.sum(1)
            weights=w*np.array([np.dot(peer,likelihood[int(v):int(v)+len(peer)]) for v in c])
            expected.append(float(weights@y/weights.sum()))
        close(expected,pred[r.gap.to_numpy()>3600],2e-6)
        samples.append(dict(month=month,airport=airport,day=day,long_rows=len(supports),grid_size=len(aggregate),offset_error=abs(offset-info.offset)))
        assert len(logs)==inputs.day.nunique();days.append(logs)
    np.testing.assert_array_equal(np.sort(covered),np.sort(f.mvt_id))
    logs=pd.concat(days);assert not logs.duplicated(['month','airport','day']).any()
    assert samples,'No bounded posterior convolution checks available in this view'
    return dict(days=len(logs),ordinary_all_rows_checked=True,sampled_convolutions=samples,unsampled_airport_months=unsampled,
        sampling='One first lexical bounded day per eligible airport/month, prefer nonempty long support; larger groups covered by all-source/ordinary checks and pinned original algorithms only',
        reused_primitive='Pinned daily_delay_posterior.support only; numerical likelihood/convolution/leave-one products independent',tolerance_sec=2e-6)


def score_check(con,f,result):
    score=read(OUT/'score.json');assert score['identity']==art(OUT/'identity.json') and score['result']==art(OUT/'cv/result.json')
    columns=['B','direct_reserve','v80','submitted'];metrics={m:{} for m in sorted(f.month.unique())}
    for col in columns:
        for month,value in con.execute(f'SELECT month,sqrt(avg(pow(CAST("{col}" AS DOUBLE)-y,2))) FROM f GROUP BY month').fetchall():
            close(value,score['months'][month][col]);metrics[month][col]=value
        close(np.mean([v[col] for v in metrics.values()]),score['cv'][col])
    paired={}
    for col in columns[:-1]:
        rng=np.random.default_rng(0);draws=np.zeros(2000)
        for month in sorted(f.month.unique()):
            part=f[f.month.eq(month)].copy();codes,_=pd.factorize(part.day,sort=False)
            counts=np.bincount(codes);old=np.bincount(codes,(part[col]-part.y)**2);new=np.bincount(codes,(part.submitted-part.y)**2)
            draws_ix=rng.integers(0,len(counts),size=(2000,len(counts)));den=counts[draws_ix].sum(1)
            draws+=(np.sqrt(new[draws_ix].sum(1)/den)-np.sqrt(old[draws_ix].sum(1)/den))/2
        delta=float(np.mean([x['submitted']-x[col] for x in metrics.values()]));ci=np.percentile(draws,[2.5,97.5])
        close(delta,score['paired'][col]['change']);close(ci,score['paired'][col]['ci95'])
        decision='improvement' if ci[1]<0 else 'regression' if ci[0]>0 else 'inconclusive'
        assert decision==score['paired'][col]['decision'];paired[col]=dict(change=delta,ci95=ci.tolist(),decision=decision)
    return dict(months=metrics,paired=paired,score=art(OUT/'score.json'),reused_development=True)


def audit(view):
    started=time.monotonic();assert not (OUT/view/'audit.json').exists()
    identity,spec,result=provenance(view)
    f=pd.read_parquet(verify(result['prediction']));ids=f.mvt_id.to_numpy()
    assert len(ids)==len(np.unique(ids))==COUNTS[view]
    base=frame(spec['base'],ids,True);d=frame(spec['raw'],ids);m=frame(spec['monthly'],ids);direct=frame(spec['direct'],ids,True)
    i=frame(spec['daily_inputs'],ids,view!='cv')
    if view=='cv':i=i.rename(columns={'local_takeoff':'local_day'})
    for g in [d,m,i]:np.testing.assert_array_equal(g.airport,f.airport)
    for k in ['month','day']:np.testing.assert_array_equal(d[k],f[k])
    for k,expected in [('B',base.submitted),('direct_control',direct.control),('tabm_direct',direct.tabm_direct)]:np.testing.assert_array_equal(f[k],expected)
    if view=='cv':
        np.testing.assert_array_equal(f.y,base.y);np.testing.assert_array_equal(f.v80,base.v80)
        joint=frame(spec['joint'],ids,True);robust=frame(spec['robust'],ids,True)
        for g in [joint,robust]:
            for k in ['y','month','day']:np.testing.assert_array_equal(g[k],f[k])
        np.testing.assert_array_equal(f.joint_increment,joint.increment);np.testing.assert_array_equal(f.robust_delta,robust.delta);np.testing.assert_array_equal(f.robust_active,robust.active)
    else:
        joint=frame(spec['increments'],ids,True)
        for k,source in [('joint_increment','increment'),('robust_delta','robust_delta'),('robust_active','robust_active')]:np.testing.assert_array_equal(f[k],joint[source])
        template=pq.read_table(verify(spec['template']));candidate=pq.read_table(verify(result['candidate']))
        assert candidate.schema==template.schema
        for k in template.column_names:
            if k!='TAXITIME_SEC_mvt':assert candidate[k].equals(template[k])
        np.testing.assert_array_equal(template['MVT_ID_mvt'].to_numpy(),ids)
        np.testing.assert_array_equal(candidate['TAXITIME_SEC_mvt'].to_numpy(),f.submitted)
        assert 'y' not in f
    assert np.isfinite(f.drop(columns='invalid_share').select_dtypes(include='number')).all().all()
    matches=pd.read_parquet(verify(spec['matches']));matches=matches[matches.accepted].copy();assert matches.mvt_id.is_unique
    if view=='cv':
        matches=matches.merge(f[['mvt_id','month']],on='mvt_id',validate='one_to_one')
        matches=matches[[r.donor_month in FOLDS[r.month]+[r.month] for r in matches.itertuples()]]
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,g in [('f',f),('d',d),('m',m),('i',i),('w',matches)]:con.register(name,g)
        raw_rows=raw_check(con,identity,spec,view,f,d,m,i)
        cells=cell_checks(con,spec,view,f,m,i)
        expected={'T':'round_even(greatest(raw_posterior,0.),0)',
            'D':f'round_even(greatest(B+{WEIGHT}*(CAST(T AS DOUBLE)-B),0.),0)',
            'full_monthly':'greatest(0.,2*half_monthly-CAST(D AS DOUBLE))',
            'N_raw':'CAST(M AS DOUBLE)+tabm_direct-direct_control',
            'applied_shift':'CASE WHEN quality_eligible THEN daily_shift ELSE 0. END',
            'direct_reserve':'round_even(greatest(tabm_direct,0.),0)'}
        for name,expr in expected.items():assert con.execute(f'SELECT count(*) FROM f WHERE "{name}" IS DISTINCT FROM ({expr})').fetchone()[0]==0,name
        for name,expr in [('M','f.full_monthly'),('N','f.N_raw'),('Q','CAST(f.N AS DOUBLE)+f.applied_shift'),('J','CAST(f.Q AS DOUBLE)+.5*f.joint_increment'),('submitted','CAST(f.J AS DOUBLE)+.5*f.robust_delta')]:
            sql=constraint(expr)
            bad=con.execute(f'SELECT count(*) FROM f JOIN d USING(mvt_id) LEFT JOIN w USING(mvt_id) WHERE f."{name}" IS DISTINCT FROM round_even(greatest(({sql}),0.),0)').fetchone()[0]
            assert bad==0,(name,bad)
        assert con.execute('SELECT count(*) FROM f WHERE NOT robust_active AND (robust_delta!=0 OR submitted!=J)').fetchone()[0]==0
        metrics=score_check(con,f,result) if view=='cv' else None
    post=posterior_checks(spec,result,view,f,d)
    value=dict(status='pass',view=view,source=art(Path(__file__)),identity=art(OUT/'identity.json'),result=art(OUT/view/'result.json'),
        prediction=result['prediction'],rows=len(f),raw_rows=raw_rows,cells=cells,posterior=post,metrics=metrics,
        all_stage_SQL_serialization_constraints_exact=True,parameter_refits=0,model_fits=0,
        calibration='Exact historical allowed-month/full2025 parameter bytes; no new calibration',
        joint='Exact original v70-prior increments transported; not posterior updates of the new base',seconds=time.monotonic()-started,no_upload=True)
    if view!='cv':value['candidate']=result['candidate']
    (OUT/view/'audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='pass',view=view,rows=len(f),cells=cells,posterior_days=post['days'],seconds=value['seconds'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--view',choices=list(COUNTS),required=True)
    with threadpool_limits(1):audit(parser.parse_args().view)
